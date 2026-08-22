use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;

use super::mp4_muxer::{Mp4Muxer, Mp4MuxerError};
use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// How a [`SegmentedMp4Muxer`] decides a segment is done and it's time to
/// cut to a new file.
#[derive(Debug, Clone, Copy)]
pub enum SegmentPolicy {
    /// Roughly this long per segment. "Roughly" because the actual cut
    /// only happens once this much time has elapsed *and* the video
    /// track's own next packet is a keyframe (see
    /// [`SegmentedMp4Muxer::open`]'s own docs) — a segment can run a bit
    /// longer than requested if keyframes are sparse.
    Duration(Duration),
}

/// One track's fixed description — everything [`SegmentedMp4Muxer::open`]
/// needs to re-add it to a fresh [`Mp4Muxer`] on every rotation.
/// `parameters` is cloned each time ([`Mp4Muxer::add_stream`] consumes its
/// own copy); the original stays here as the template.
struct StreamDef {
    name: Arc<str>,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    /// Whether this is the track [`SegmentGroup::maybe_rotate`] waits for
    /// a keyframe on before actually cutting — see
    /// [`SegmentedMp4Muxer::open`]'s own docs.
    is_video: bool,
}

/// Builds a [`SegmentedMp4Muxer`] the same two-phase way as a plain
/// [`Mp4Muxer`] (every track's shape must be known before the first byte
/// is written) — `create` picks the rotation policy and how segments get
/// named, `add_stream` registers each track exactly like
/// [`Mp4Muxer::add_stream`], `open` writes the first segment's header and
/// returns one [`Sink`] per track.
///
/// ```no_run
/// # use std::{path::PathBuf, time::Duration};
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{
/// #     AudioCodec, SegmentPolicy, SegmentedMp4Muxer, SwAudioEncoder,
/// #     SwAudioEncoderOptions, SwEncoder, SwEncoderOptions, VideoCodec,
/// # };
/// # fn main() -> media_pp::Result<()> {
/// # let video_time_base = ffmpeg::Rational(1, 30);
/// # let audio_time_base = ffmpeg::Rational(1, 48_000);
/// # let video_encoder = SwEncoder::new("video", SwEncoderOptions {
/// #     codec: VideoCodec::H264,
/// #     width: 640,
/// #     height: 360,
/// #     time_base: video_time_base,
/// #     frame_rate: ffmpeg::Rational(30, 1),
/// #     bit_rate: 2_000_000,
/// #     gop_size: 30,
/// # })?;
/// # let audio_encoder = SwAudioEncoder::new("audio", SwAudioEncoderOptions {
/// #     codec: AudioCodec::Aac,
/// #     sample_rate: 48_000,
/// #     channels: 2,
/// #     time_base: audio_time_base,
/// #     bit_rate: 128_000,
/// # })?;
/// let mut muxer = SegmentedMp4Muxer::create(
///     SegmentPolicy::Duration(Duration::from_secs(600)),
///     |index| PathBuf::from(format!("rec_{index:04}.mp4")),
/// );
/// muxer.add_stream("video", video_encoder.parameters(), video_time_base);
/// muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base);
/// let mut sinks = muxer.open()?;
/// # Ok(())
/// # }
/// ```
pub struct SegmentedMp4Muxer {
    policy: SegmentPolicy,
    naming: Box<dyn FnMut(u64) -> PathBuf + Send>,
    streams: Vec<StreamDef>,
}

impl SegmentedMp4Muxer {
    /// `naming(index)` names each segment file, `index` starting at `0` —
    /// called once up front for the first segment and again on every
    /// rotation. Typically a closure building a path from a fixed
    /// directory/prefix (e.g. `|i| dir.join(format!("rec_{i:04}.mp4"))`);
    /// a timestamp-based scheme works just as well since `index` is only
    /// ever used to *call* this, never to build the path itself.
    pub fn create(
        policy: SegmentPolicy,
        naming: impl FnMut(u64) -> PathBuf + Send + 'static,
    ) -> Self {
        Self {
            policy,
            naming: Box::new(naming),
            streams: Vec::new(),
        }
    }

    /// Registers one more track every segment file will hold — same
    /// contract as [`Mp4Muxer::add_stream`] (same order rules, same
    /// `name`/`parameters`/`time_base` meaning), except this can't fail:
    /// nothing here touches ffmpeg yet, it's only recorded for
    /// [`SegmentedMp4Muxer::open`] (and every later rotation) to replay.
    ///
    /// Whichever stream's `parameters.medium()` is
    /// [`ffmpeg::media::Type::Video`] (at most one is expected) becomes
    /// the keyframe-gating track described in [`SegmentedMp4Muxer::open`]'s
    /// own docs — no separate flag to pass.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) {
        let is_video = parameters.medium() == ffmpeg::media::Type::Video;
        self.streams.push(StreamDef {
            name: name.into().into(),
            parameters,
            time_base,
            is_video,
        });
    }

    /// Writes the first segment's header and returns one [`Sink`] per
    /// track, in the order [`SegmentedMp4Muxer::add_stream`] added them —
    /// same shape as [`Mp4Muxer::open`]. All returned `Sink`s share one
    /// rotation lock: a track's `consume` blocks while another track (on
    /// its own thread) is mid-rotation, same tradeoff [`Mp4Muxer`]'s own
    /// shared file lock already makes.
    ///
    /// **Rotation timing**: once a segment has run at least as long as the
    /// configured [`SegmentPolicy`], the cut happens on the *video*
    /// track's next keyframe (found via `add_stream`'s `parameters` — see
    /// its own docs) — not immediately, and not on an arbitrary packet —
    /// so every segment file is independently decodable from its own
    /// first frame, the same way a real segment/HLS muxer cuts. A segment
    /// can therefore run somewhat longer than requested if keyframes are
    /// sparse; there's no hard cap. If no track's `parameters.medium()`
    /// was `Video` (an audio-only recording), any packet on any track is
    /// an equally valid cut point, so rotation happens as soon as the
    /// policy is due.
    ///
    /// The final segment is finalized the same way a plain [`Mp4Muxer`]
    /// finalizes its one file: once every track has reported `Eos` *or*
    /// [`ControlMsg::Stop`] (see [`Mp4Muxer::open`]'s own docs) — a
    /// rotation mid-recording reuses that exact mechanism to close the
    /// outgoing segment before opening the next one.
    pub fn open(mut self) -> Result<Vec<Box<dyn Sink>>> {
        // Captured before `self.streams` moves into `GroupState` below —
        // just the names, in order, for building each `SegmentedTrackSink`
        // afterward.
        let names: Vec<Arc<str>> = self.streams.iter().map(|s| s.name.clone()).collect();
        let path = (self.naming)(0);
        let current_sinks = open_segment(&self.streams, path)?;
        let group = Arc::new(SegmentGroup {
            policy: self.policy,
            naming: Mutex::new(self.naming),
            state: Mutex::new(GroupState {
                streams: self.streams,
                current_sinks,
                segment_index: 0,
                segment_started: Instant::now(),
            }),
        });
        Ok(names
            .into_iter()
            .enumerate()
            .map(|(index, name)| -> Box<dyn Sink> {
                Box::new(SegmentedTrackSink {
                    pp_log: element_pp_log(ElementType::SegmentedMp4Muxer, &name, None),
                    name,
                    track_index: index,
                    group: group.clone(),
                })
            })
            .collect())
    }
}

fn open_segment(streams: &[StreamDef], path: PathBuf) -> Result<Vec<Box<dyn Sink>>> {
    let mut muxer = Mp4Muxer::create(&path)?;
    for stream in streams {
        muxer.add_stream(
            stream.name.to_string(),
            stream.parameters.clone(),
            stream.time_base,
        )?;
    }
    muxer.open()
}

struct GroupState {
    /// This group's fixed track descriptions — moved in here (rather than
    /// sitting on [`SegmentGroup`] directly) purely so `Mutex<GroupState>`
    /// covers it too: [`ffmpeg::codec::Parameters`] wraps a raw pointer and
    /// isn't `Sync`, so a field of this type living outside any `Mutex`
    /// would make `SegmentGroup` itself `!Sync` and unable to cross
    /// threads inside the `Arc` every [`SegmentedTrackSink`] holds.
    /// `Mutex<T>` only ever needs `T: Send` (already true here — see
    /// `ffmpeg-next`'s own `unsafe impl Send for Parameters`), never
    /// `T: Sync`, which is exactly what sidesteps that.
    streams: Vec<StreamDef>,
    /// The currently-open segment's own per-track sinks, in the same
    /// order as `streams` — index-aligned with
    /// [`SegmentedTrackSink::track_index`].
    current_sinks: Vec<Box<dyn Sink>>,
    segment_index: u64,
    segment_started: Instant,
}

/// Shared between every [`SegmentedTrackSink`] [`SegmentedMp4Muxer::open`]
/// hands out — one rotation lock around the whole group of tracks, so a
/// rotation triggered by one track's packet arrival is atomic with respect
/// to every other track (either all of them are still writing into the
/// outgoing segment, or all of them are already writing into the new one —
/// never a mix).
struct SegmentGroup {
    policy: SegmentPolicy,
    naming: Mutex<Box<dyn FnMut(u64) -> PathBuf + Send>>,
    state: Mutex<GroupState>,
}

impl SegmentGroup {
    /// Called for every `Packet` on every track, before it's written.
    /// Rotates first if this is the packet that should trigger it (see
    /// [`SegmentedMp4Muxer::open`]'s own docs), then writes into whichever
    /// segment is current by the time this returns.
    fn consume_packet(
        &self,
        track_index: usize,
        packet: Arc<ffmpeg::Packet>,
        pp_log: &PpLog,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let SegmentPolicy::Duration(due_after) = self.policy;
        let mut rotated_to = None;
        if state.segment_started.elapsed() >= due_after {
            let has_video = state.streams.iter().any(|s| s.is_video);
            let this_is_video = state.streams[track_index].is_video;
            let should_cut = if has_video {
                this_is_video && packet.is_key()
            } else {
                true
            };
            if should_cut {
                for sink in state.current_sinks.iter_mut() {
                    sink.control(ControlMsg::Stop)?;
                }
                let index = state.segment_index + 1;
                let path = (self.naming.lock().unwrap())(index);
                state.current_sinks = open_segment(&state.streams, path)?;
                state.segment_index = index;
                state.segment_started = Instant::now();
                rotated_to = Some(index);
            }
        }
        let result = state.current_sinks[track_index].consume(MediaBuffer::Packet(packet));
        // Formatting and emitting happen off the group lock: every track's
        // `consume_packet` contends for it, so nothing that isn't required
        // to be serialized with the rotation belongs inside it.
        drop(state);
        if let Some(index) = rotated_to {
            pp_info!(pp_log: pp_log, "rotated segment_index={index}");
        }
        result
    }

    /// One track's own natural `Eos` — forwarded into whatever segment is
    /// current, same as [`SegmentGroup::consume_packet`] but without a
    /// rotation check (ending is ending, not a cut point).
    fn finish_eos(&self, track_index: usize) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.current_sinks[track_index].consume(MediaBuffer::Eos)
    }

    /// One track's own [`ControlMsg::Stop`] — same as
    /// [`SegmentGroup::finish_eos`], just forwarded as `Stop` instead of
    /// `Eos` (matters for `Mp4Muxer`'s own docs on `Stop` finalizing a
    /// container even though it otherwise means "abandon, don't drain").
    fn finish_stop(&self, track_index: usize) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.current_sinks[track_index].control(ControlMsg::Stop)
    }
}

/// One track's own [`Sink`] — a lightweight handle sharing a
/// [`SegmentGroup`] with every other track [`SegmentedMp4Muxer::open`]
/// returned alongside it.
struct SegmentedTrackSink {
    pp_log: PpLog,
    name: Arc<str>,
    track_index: usize,
    group: Arc<SegmentGroup>,
}

impl Element for SegmentedTrackSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SegmentedMp4Muxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for SegmentedTrackSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.group
                    .consume_packet(self.track_index, packet, &self.pp_log)
            }
            MediaBuffer::Eos => self.group.finish_eos(self.track_index),
            // The `Mp4Muxer` each rotated segment wraps already rejects
            // this — matching its own `Mp4MuxerStreamSink::consume` here
            // instead of silently no-op'ing keeps that protection visible
            // through the rotation wrapper instead of swallowing it.
            other => Err(Mp4MuxerError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Stop {
            self.group.finish_stop(self.track_index)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elements::{SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource, VideoCodec},
        pipeline::Pipeline,
    };

    /// Opens an H.264 encoder for these tests, preferring the non-GPL
    /// `OpenH264` but falling back to the GPL `H264` (`libx264`) — a
    /// contributor's own ffmpeg build is more likely to carry `libx264`
    /// (most distro packages enable it) than the Cisco `libopenh264`
    /// library, and either one exercises the segment-rotation logic these
    /// tests actually check. `None` only when neither is available.
    fn open_h264_encoder(name: &str, mut options: SwEncoderOptions) -> Option<SwEncoder> {
        [VideoCodec::OpenH264, VideoCodec::H264]
            .into_iter()
            .find_map(|codec| {
                options.codec = codec;
                SwEncoder::new(name, options).ok()
            })
    }

    /// Drives a real `TestVideoSource -> SwEncoder -> SegmentedMp4Muxer`
    /// chain for a few real seconds with a short rotation policy, then
    /// checks every segment file it produced: each one has to be a real,
    /// independently-readable `.mp4` whose very first packet is a
    /// keyframe — proof the cut actually waited for one (see
    /// `SegmentedMp4Muxer::open`'s own docs on why that matters: cutting
    /// on an arbitrary packet would leave a segment starting mid-GOP,
    /// undecodable from its own frame 0).
    #[test]
    fn rotates_into_multiple_valid_keyframe_aligned_segments() {
        let video_options = TestVideoOptions {
            width: 160,
            height: 120,
            framerate: ffmpeg::Rational::new(15, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let time_base = video_source.time_base();
        let Some(encoder) = open_h264_encoder(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                time_base,
                frame_rate: video_options.framerate,
                bit_rate: 200_000,
                // Short on purpose (~0.5s @ 15fps) — this test needs
                // several real keyframes to show up quickly, not the
                // ~2s default every other caller uses.
                gop_size: 8,
            },
        ) else {
            eprintln!("skipping: no H.264 encoder available (openh264 or libx264)");
            return;
        };

        let dir = std::env::temp_dir();
        let prefix = format!("segmented_mp4_test_{}", std::process::id());
        let paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_paths = paths.clone();

        let mut muxer = SegmentedMp4Muxer::create(
            SegmentPolicy::Duration(Duration::from_millis(300)),
            move |index| {
                let path = dir.join(format!("{prefix}_{index:03}.mp4"));
                recorded_paths.lock().unwrap().push(path.clone());
                path
            },
        );
        muxer.add_stream("video", encoder.parameters(), time_base);
        let mut sinks = muxer.open().expect("open must succeed");
        let sink = sinks.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("segmented-test", video_source, |source, ctx| {
            let branch = ctx.branch().pipe(encoder).to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();
        std::thread::sleep(Duration::from_secs(3));
        pipeline.stop();
        pipeline.bus().log_events();

        let paths = paths.lock().unwrap().clone();
        assert!(
            paths.len() >= 2,
            "expected at least 2 segments, got {}: {paths:?}",
            paths.len()
        );

        for path in &paths {
            let mut input = ffmpeg::format::input(path)
                .unwrap_or_else(|error| panic!("segment {path:?} must be readable: {error}"));
            assert_eq!(
                input.streams().count(),
                1,
                "segment {path:?} should have exactly one stream"
            );
            let mut packet = ffmpeg::Packet::empty();
            if packet.read(&mut input).is_err() {
                panic!("segment {path:?} has no packets at all");
            }
            assert!(
                packet.is_key(),
                "segment {path:?}'s first packet must be a keyframe"
            );
        }

        for path in &paths {
            std::fs::remove_file(path).ok();
        }
    }

    /// A misrouted `Audio` buffer used to be silently dropped by
    /// `SegmentedTrackSink::consume`. The `Mp4Muxer` each segment wraps
    /// already rejects this via `Mp4MuxerError::UnsupportedBuffer` — the
    /// rotation wrapper must surface that instead of swallowing it.
    #[test]
    fn rejects_a_buffer_type_the_wrapped_mp4_muxer_does_not_accept() {
        let video_options = TestVideoOptions {
            width: 160,
            height: 120,
            framerate: ffmpeg::Rational::new(15, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let time_base = video_source.time_base();
        let Some(encoder) = open_h264_encoder(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                time_base,
                frame_rate: video_options.framerate,
                bit_rate: 200_000,
                gop_size: 8,
            },
        ) else {
            eprintln!("skipping: no H.264 encoder available (openh264 or libx264)");
            return;
        };

        let path = std::env::temp_dir().join(format!(
            "segmented_mp4_reject_test_{}.mp4",
            std::process::id()
        ));
        let recorded_path = path.clone();
        let mut muxer = SegmentedMp4Muxer::create(
            SegmentPolicy::Duration(Duration::from_secs(3600)),
            move |_index| recorded_path.clone(),
        );
        muxer.add_stream("video", encoder.parameters(), time_base);
        let mut sinks = muxer.open().expect("open must succeed");
        let mut sink = sinks.pop().expect("exactly one stream was added");

        let error = sink
            .consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .expect_err("an Audio buffer must be rejected, not silently dropped");
        assert!(
            matches!(
                error,
                crate::error::Error::Mp4MuxerError(Mp4MuxerError::UnsupportedBuffer("Audio"))
            ),
            "unexpected error: {error:?}"
        );

        drop(sink);
        std::fs::remove_file(&path).ok();
    }

    /// Regression test against a leaked file handle on the *previous*
    /// segment specifically: a rotation has to fully close (write the
    /// trailer, drop the underlying `Mp4Muxer` for that segment) the
    /// outgoing file the moment it cuts — not defer that until the whole
    /// recording later stops. Proven by reading the first segment back
    /// *while the pipeline is still running* (recording into the second
    /// one) — if `SegmentGroup::consume_packet` kept anything from the old
    /// segment alive past the cut, this would find it still unreadable
    /// (or, on Windows, fail to even open for read at all due to a
    /// lingering write lock).
    #[test]
    fn old_segment_is_released_immediately_not_deferred_until_the_whole_recording_stops() {
        let video_options = TestVideoOptions {
            width: 160,
            height: 120,
            framerate: ffmpeg::Rational::new(15, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let time_base = video_source.time_base();
        let Some(encoder) = open_h264_encoder(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                time_base,
                frame_rate: video_options.framerate,
                bit_rate: 200_000,
                gop_size: 8, // ~0.5s @ 15fps — see the other test's own note
            },
        ) else {
            eprintln!("skipping: no H.264 encoder available (openh264 or libx264)");
            return;
        };

        let dir = std::env::temp_dir();
        let prefix = format!("segmented_mp4_release_test_{}", std::process::id());
        let paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_paths = paths.clone();

        let mut muxer = SegmentedMp4Muxer::create(
            SegmentPolicy::Duration(Duration::from_millis(300)),
            move |index| {
                let path = dir.join(format!("{prefix}_{index:03}.mp4"));
                recorded_paths.lock().unwrap().push(path.clone());
                path
            },
        );
        muxer.add_stream("video", encoder.parameters(), time_base);
        let mut sinks = muxer.open().expect("open must succeed");
        let sink = sinks.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("segmented-release-test", video_source, |source, ctx| {
            let branch = ctx.branch().pipe(encoder).to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

        // Wait (bounded) for at least one rotation — the pipeline is
        // deliberately still running past this point.
        let waited = Instant::now();
        loop {
            if paths.lock().unwrap().len() >= 2 {
                break;
            }
            assert!(
                waited.elapsed() < Duration::from_secs(5),
                "no rotation happened within 5s"
            );
            std::thread::sleep(Duration::from_millis(50));
        }

        let first_segment = paths.lock().unwrap()[0].clone();
        let mut input = ffmpeg::format::input(&first_segment).unwrap_or_else(|error| {
            panic!("segment 0 must already be readable while still recording segment 1: {error}")
        });
        let mut packet = ffmpeg::Packet::empty();
        assert!(
            packet.read(&mut input).is_ok(),
            "segment 0 must have packets"
        );
        drop(input);

        pipeline.stop();
        pipeline.bus().log_events();

        for path in paths.lock().unwrap().iter() {
            std::fs::remove_file(path).ok();
        }
    }
}
