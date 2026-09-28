use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, Produce, Produced, ProducingSource, Wait, element_pp_log},
    elements::{RtspOptions, rtsp::redact},
    error::Result,
    pad::SrcPad,
    produce::produce_source,
};

use super::file_demuxer::StreamInfo;

/// Errors specific to `RtspSource`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspSourceError {
    /// FFmpeg rejected connection setup, stream reading, or shutdown.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// The session has no stream of the kind asked for — see
    /// [`RtspSource::best`].
    #[error("the RTSP session has no {0:?} stream")]
    NoStream(ffmpeg::media::Type),
}

/// Demuxes a live RTSP stream — the client/receive counterpart to
/// [`crate::elements::RtspMuxer`] (which publishes). One src pad per
/// stream the server advertises, same shape as
/// [`crate::elements::FileDemuxer`].
///
/// Deliberately does **not** retry or reconnect internally: a read failure
/// (dropped connection, camera reboot, ...) ends this source's thread with
/// the error on the bus, the same way any other fatal source failure does,
/// instead of looping forever. Reconnecting means building a fresh
/// `RtspSource`/[`crate::pipeline::Pipeline`] — mirrors `Pipeline` itself
/// not being reusable once it ends: watch
/// [`crate::pipeline::Pipeline::bus`], and on error, call
/// [`RtspSource::open`] again.
///
/// Uses `Packet::read` directly instead of `Input::packets()` — the
/// latter silently retries forever inside its own `next()` on any non-EOF
/// error (network timeout, connection reset, ...), which would make a
/// stuck connection un-`Stop`-able (the pipeline's requests never get a
/// turn) and this element's "fail fast, don't retry" contract impossible to
/// keep.
pub struct RtspSource(ProducingSource<Reading>);

produce_source!(RtspSource);

/// What an [`RtspSource`] hands on, a packet at a time as the session
/// delivers it: all of its work, which the framework makes the source.
struct Reading {
    pp_log: PpLog,
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    /// What each stream's output declares, by stream index.
    contracts: Vec<OutputContract>,
}

impl RtspSource {
    /// Connects to `url` (e.g. `rtsp://host:port/path`) and returns the
    /// element alongside every stream the server advertised, so the
    /// caller can inspect them before deciding which of `src_pads()` to
    /// link — same pattern as `FileDemuxer::open`.
    pub fn open(
        name: impl Into<String>,
        url: impl AsRef<str>,
        options: RtspOptions,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        crate::ensure_ffmpeg();
        let input = ffmpeg::format::input_with_dictionary(url.as_ref(), options.to_dictionary())?;

        let streams: Vec<StreamInfo> = input.streams().map(|s| StreamInfo::of(&s)).collect();

        // Per stream, from the medium the session announced: every output
        // hands on `MediaBuffer::Packet`, so only this tells an audio
        // stream apart from a video one. A medium this crate does not model
        // declares nothing and is left to the runtime check.
        let contracts = streams
            .iter()
            .map(|s| match MediaKind::packet_for(s.kind) {
                Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
                None => OutputContract::Unknown,
            })
            .collect();

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::RtspSource, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: url={}, transport={:?}, {} stream(s)",
            // A camera's address often carries its password.
            redact(url.as_ref()),
            options.transport,
            streams.len()
        );
        Ok((
            Self(ProducingSource::new(Reading {
                name,
                pp_log,
                input,
                contracts,
            })),
            streams,
        ))
    }

    /// The stream of `kind` FFmpeg judges the one to play, as everything a
    /// branch for it is built from — or a [`RtspSourceError::NoStream`]
    /// naming the kind the session lacks. See
    /// [`FileDemuxer::best`](crate::elements::FileDemuxer::best).
    pub fn best(
        &self,
        kind: ffmpeg::media::Type,
    ) -> std::result::Result<StreamInfo, RtspSourceError> {
        self.0
            .inner()
            .input
            .streams()
            .best(kind)
            .map(|stream| StreamInfo::of(&stream))
            .ok_or(RtspSourceError::NoStream(kind))
    }
}

impl Reading {
    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }
}

impl Element for Reading {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Produce for Reading {
    fn is_live(&self) -> bool {
        true
    }

    /// One per stream the server advertised, at the stream's own index.
    fn outputs(&self) -> Vec<SrcPad> {
        self.contracts
            .iter()
            .enumerate()
            .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
            .collect()
    }

    /// The next packet the session delivers, on its stream's output.
    ///
    /// Waits inside the read, which the connection's own timeout bounds —
    /// see [`RtspOptions::timeout`] — rather than through `wait`: a `Stop`
    /// reaches this source between reads, at most that long after it is
    /// sent.
    fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut self.input) {
            Ok(()) => {
                let index = packet.stream();
                // FFmpeg does not guarantee a demuxer fills
                // `AVPacket::time_base`, and the packet contract is that
                // one carries it — what a decoder downstream hands on to
                // its frames, and a `Pacer` paces by. Same as
                // `FileDemuxer`'s.
                if let Some(time_base) = self.stream(index).map(|stream| stream.time_base()) {
                    packet.set_time_base(time_base);
                }
                Ok(Produced::On(index, MediaBuffer::Packet(Arc::new(packet))))
            }
            // A real on-demand RTSP stream can send a clean EOF; a
            // live camera essentially never will, but treat it the
            // same way `FileDemuxer` treats running out of packets.
            Err(ffmpeg::Error::Eof) => Ok(Produced::End),
            // Anything else (connection reset, socket timeout, ...) is
            // fatal — reported and this thread ends, rather than
            // retried. See this type's own docs on why: retrying
            // belongs to whoever's watching the bus, building a fresh
            // `RtspSource` to reconnect with.
            Err(error) => {
                pp_error!(self, "read failed: {error}");
                Err(RtspSourceError::Ffmpeg(error).into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// `RtspOptions::timeout`'s whole reason for existing: without it
    /// ffmpeg applies *no* timeout at all and `open` blocks forever against
    /// a server that never answers. `192.0.2.1` is RFC 5737 TEST-NET-1,
    /// reserved for documentation and guaranteed not to be routed, so this
    /// exercises the "no answer" path rather than a fast connection refusal.
    ///
    /// Only the upper bound is asserted: a network that replies with an ICMP
    /// unreachable makes this fail even sooner, which is equally correct. The
    /// bound sits well under the OS-level TCP connect timeout (~21s on
    /// Windows, far longer on Linux), so a regression that stops passing the
    /// option through is what actually trips it.
    #[test]
    fn open_gives_up_within_the_configured_timeout_instead_of_hanging() {
        let options = RtspOptions {
            timeout: Duration::from_millis(500),
            ..Default::default()
        };

        let started = Instant::now();
        let result = RtspSource::open("rtsp", "rtsp://192.0.2.1:554/none", options);
        let elapsed = started.elapsed();

        assert!(
            result.is_err(),
            "opening an unroutable address must not succeed"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "open took {elapsed:?} — the configured timeout is not reaching ffmpeg"
        );
    }

    /// Each stream's packets go out on that stream's own output, carrying
    /// its time base, and a clean end of the session ends every output.
    /// Read from a file, which FFmpeg opens as it would a session: no
    /// server is needed to see how the source hands on what it reads.
    #[test]
    fn each_stream_goes_out_on_its_own_output_and_the_end_ends_them_all() {
        use std::sync::{Mutex, atomic::AtomicUsize, atomic::Ordering};

        use crate::{elements::AppSink, pipeline::Pipeline, stream::StreamEvent};

        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let (source, streams) =
            RtspSource::open("rtsp", &path, RtspOptions::default()).expect("open the fixture");
        let kinds: Vec<_> = streams.iter().map(|stream| stream.kind).collect();
        let seen: Arc<Vec<Mutex<Vec<ffmpeg::Rational>>>> =
            Arc::new(streams.iter().map(|_| Mutex::default()).collect());
        let ends = Arc::new(AtomicUsize::new(0));
        let (pipeline, ()) = Pipeline::new("rtsp", source, |source, ctx| {
            for output in 0..kinds.len() {
                let sink = AppSink::with_events(
                    format!("stream-{output}"),
                    {
                        let seen = Arc::clone(&seen);
                        move |buffer| {
                            if let MediaBuffer::Packet(packet) = buffer {
                                seen[output].lock().unwrap().push(packet.time_base());
                            }
                            Ok(())
                        }
                    },
                    {
                        let ends = Arc::clone(&ends);
                        move |event: &StreamEvent| {
                            if let StreamEvent::Eos = event {
                                ends.fetch_add(1, Ordering::SeqCst);
                            }
                            Ok(())
                        }
                    },
                );
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, output, branch)?;
            }
            Ok(())
        })
        .expect("wire every stream");
        pipeline.run().expect("run");
        let deadline = Instant::now() + Duration::from_secs(20);
        while ends.load(Ordering::SeqCst) < kinds.len() {
            assert!(Instant::now() < deadline, "not every output ended");
            std::thread::sleep(Duration::from_millis(10));
        }
        pipeline.stop();

        for (output, stream) in streams.iter().enumerate() {
            let seen = seen[output].lock().unwrap();
            assert!(!seen.is_empty(), "stream {output} handed nothing on");
            assert!(
                seen.iter().all(|&time_base| time_base == stream.time_base),
                "stream {output}'s packets carry its time base"
            );
        }
    }

    /// A caller that never touches `RtspOptions` still has to get a bounded
    /// `open`, since the default this type supplies is the only thing
    /// standing between them and ffmpeg's unbounded one.
    #[test]
    fn the_default_options_still_bound_the_connection() {
        let options = RtspOptions::default();

        assert_eq!(options.transport, crate::elements::RtspTransport::Tcp);
        assert!(
            options.timeout > Duration::ZERO,
            "the default timeout must not be ffmpeg's unbounded one"
        );
    }
}
