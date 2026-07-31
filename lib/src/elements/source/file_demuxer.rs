use std::{path::Path, sync::Arc, time::Duration};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement},
    pad::SrcPad,
};

/// Errors specific to `FileDemuxer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileDemuxError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Metadata about one stream in an opened container, reported up front so
/// callers can decide what to build downstream before the pipeline runs.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    pub index: usize,
    pub kind: ffmpeg::media::Type,
}

/// Demuxes a file, exposing one src pad per container stream (indexed the
/// same way as `StreamInfo::index`). Linking a pad "selects" that stream;
/// leaving it unlinked just drops its packets. Real demuxer I/O is
/// blocking, so this is meant to be run as the pipeline's source thread.
///
/// Fan-out (e.g. routing video and audio to separate branches) needs no
/// separate "Tee" element here — it's just a matter of linking more than
/// one of these pads.
pub struct FileDemuxer {
    name: String,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
}

impl FileDemuxer {
    /// Opens the file and returns it alongside every stream it contains,
    /// so the caller can inspect them (count, media type, ...) before
    /// deciding which of `src_pads()` to link.
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<StreamInfo>), FileDemuxError> {
        let input = ffmpeg::format::input(&path)?;

        let streams: Vec<StreamInfo> = input
            .streams()
            .map(|s| StreamInfo {
                index: s.index(),
                kind: s.parameters().medium(),
            })
            .collect();

        let pads = streams
            .iter()
            .map(|s| SrcPad::new(format!("src_{}", s.index)))
            .collect();

        Ok((
            Self {
                name: name.into(),
                input,
                pads,
            },
            streams,
        ))
    }

    /// Codec parameters for one of this file's streams — what you need to
    /// construct a matching [`crate::elements::SwDecoder`] for it.
    pub fn stream_parameters(&self, index: usize) -> Option<ffmpeg::codec::Parameters> {
        self.stream(index).map(|s| s.parameters())
    }

    /// The unit decoded frame timestamps for this stream are expressed in —
    /// what you need to construct a matching [`crate::elements::Pacer`] for
    /// it.
    pub fn stream_time_base(&self, index: usize) -> Option<ffmpeg::Rational> {
        self.stream(index).map(|s| s.time_base())
    }

    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }
}

impl Element for FileDemuxer {
    fn name(&self) -> &str {
        &self.name
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileDemuxer
    }
}

impl Source for FileDemuxer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for FileDemuxer {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        // Deliberately re-creates `self.input.packets()` fresh every
        // iteration (cheap — it's just a short-lived wrapper, not a
        // stateful cursor of its own) instead of holding one `for` loop's
        // iterator across the whole function, the way this used to read.
        // That iterator borrows `input` for as long as it's alive; `Seek`
        // needs `drain_control` to be able to call `self.seek()` — a
        // *second* mutable borrow of `input` — in between reads, which a
        // single loop-spanning iterator would rule out.
        loop {
            if drain_control(control, self)? {
                // Stop: abandon in place, no final Eos.
                return Ok(());
            }
            let Some((stream, packet)) = self.input.packets().next() else {
                break;
            };
            let index = stream.index();
            if let Some(pad) = self.pads.get_mut(index) {
                // A downstream failure drops just this one packet — same
                // "report, don't die" contract `Queue`'s worker gives a
                // failing `Sink` — rather than ending this whole source
                // thread over it. `Pipeline::stop` is how a caller who
                // decides an error is fatal actually ends things.
                if let Err(error) = pad.push(MediaBuffer::Packet(Arc::new(packet))) {
                    bus.post(BusEvent::Error {
                        element_type: ElementType::FileDemuxer,
                        name: self.name.clone(),
                        error,
                    });
                }
            }
        }
        for pad in self.pads.iter_mut() {
            pad.push(MediaBuffer::Eos)?;
        }
        Ok(())
    }

    fn seek(&mut self, target: Duration) -> crate::error::Result<()> {
        // `Input::seek` takes microseconds (`AV_TIME_BASE` units) when
        // seeking the whole container (stream index -1, which is what it
        // uses internally) rather than one specific stream — an unbounded
        // range (`..`) just means "as close to `ts` as ffmpeg can manage",
        // no extra min/max constraint.
        let ts = target.as_micros().min(i64::MAX as u128) as i64;
        self.input.seek(ts, ..)?;
        Ok(())
    }
}
