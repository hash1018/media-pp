# Changelog

Notable changes to `media-pp`. Versions before 0.2.0 have no entry here — this
file starts where the first breaking release did, and the git history is the
record for everything earlier.

The crate is pre-1.0, so a `0.x` bump is where breaking changes land. Each one
below says what to write instead, because a rename with no migration line is a
compile error with no explanation.

## Unreleased

### Added

- **`RtmpMuxer` publishes a live broadcast to an RTMP server** — Twitch,
  YouTube, or a local MediaMTX. It is the publishing half only: nothing here
  runs a server, and the address and stream key come from whoever receives.
  Shaped like `FileMuxer` rather than `RtspSink`, because a broadcast is video
  *and* audio in one FLV container and the header has to describe both up
  front: `create` connects, `add_stream` registers each track, and `open`
  writes the header and returns one `Sink` per track. It remuxes and does not
  encode, so H.264 and AAC come from the encoders upstream.

  A publish URL ends in a credential, so nothing logs the URL it was given —
  `redacted_url` is what reaches a log and what a caller should display. It
  does not reconnect: a connection lost mid-broadcast is a write error, and
  recovering means a new `RtmpMuxer` and so a fresh keyframe.

  New: `Error::RtmpMuxerError`, `ElementType::RtmpMuxer`, and the
  `rtmp_publish` example.

## 0.2.0

Two renames, a camera source on both platforms, and a good deal of runtime
control that used to be fixed at construction.

### Breaking

- **`Mp4Muxer` is now `FileMuxer`.** `format::output` has always guessed the
  container from the file name, so this type has always written whatever the
  path asked for; MP4 was the only thing still claiming otherwise. Rename
  `Mp4Muxer` → `FileMuxer`, `Mp4MuxerError` → `FileMuxerError`,
  `Mp4MuxerStreamSink` → `FileMuxerStreamSink`, `SegmentedMp4Muxer` →
  `SegmentedFileMuxer`. `Error::Mp4MuxerError` → `Error::FileMuxerError`. The
  `ElementType` variants carry the same names into logs and change with them.
  Deliberately without a compatibility alias.

- **`D3d11NvencEncoder` is now `D3d11VideoEncoder`**, and it reaches Intel and
  AMD hardware as well as NVIDIA's. The old element opened `h264_nvenc`, so a
  machine without an NVIDIA GPU fell back to encoding on the CPU — the
  expensive path the element existed to replace. It now asks Media Foundation
  for whichever hardware H.264/HEVC transform the installed driver registers.
  Rename `D3d11NvencEncoder` → `D3d11VideoEncoder`, `D3d11NvencEncoderOptions`
  → `D3d11VideoEncoderOptions`, `D3d11NvencEncoderError` →
  `D3d11VideoEncoderError`, `D3d11NvencCodec` → `D3d11VideoCodec`,
  `D3d11NvencInputFormat` → `D3d11VideoInputFormat`.
  `Error::D3d11NvencEncoderError` → `Error::D3d11VideoEncoderError`.

- **Elements are handed their clocks instead of being given them.**
  `Pacer::bind_playback_clock` and `bind_playback_clock_deferred` are gone.
  Nothing checked that the clock a caller passed was the one the pipeline
  actually runs on, and at least one example passed a `Clock::new()` the
  pipeline never paused, reset or interrupted. Construct a `Pacer` with only
  its name, time base and options; the pipeline supplies the rest through
  `Element::attach_context` when the element is wired. The same applies to an
  audio renderer claiming the playback clock as master.

- **`SegmentPolicy` has a second variant, `Size(u64)`** — an exhaustive match
  over it no longer compiles. See *Added*.

### Added

- **Camera capture, on both platforms.** `MfCaptureSource` (Windows, feature
  `mf-capture`) goes through Media Foundation's source reader;
  `V4l2CaptureSource` (Linux, feature `v4l2-capture`) goes through FFmpeg's
  own `video4linux2` demuxer, with four read-only ioctls behind the picker.
  Both push CPU-resident NV12 frames, so `D3d11Upload` and `CudaUpload` take
  them directly. A caller picks a device and a picture shape — `MfDevice` /
  `MfCaptureFormat`, `V4l2Device` / `V4l2CaptureFormat` — and never a subtype:
  which of MJPEG, YUY2 or NV12 a mode is natively is the element's business.

- **`PipelineBridge`**, which carries buffers from one pipeline into another.
  With `PipelineBridgeHandle`, `PipelineBridgeOptions` and
  `PipelineBridgeError`.

- **A settable frame rate.** The new `rate` module publishes `FrameRate` and
  `FrameRateHandle`, and the compositors and captures take one, so an output
  rate can change while the graph runs instead of being fixed when it was
  built.

- **`MixFormat`**, and an `AudioMixer` whose mix format can change while it
  runs.

- **`VideoSourceRect`** on a layer, so a compositor input can draw only part
  of its picture. Both compositors honour it; the CUDA one copies the region
  before scaling.

- **`FileDemuxerHandle`**, which plays a file again when it reaches the end
  and reports how far the loop has carried the timeline.

- **`Pipeline::is_running`**, so a caller can ask whether anything is still
  on a thread of its own without draining the bus for it.

- **`SegmentPolicy::Size`**, cutting a recording by bytes as well as by
  duration. Both wait for the video track's next keyframe, so a segment
  overruns by about a GOP either way.

- **B-frames.** `SwEncoder` and both hardware encoders take a count, and
  `SwEncoder` can now use FFmpeg's own H.264 encoder beside OpenH264.

- **A CUDA compositor layer that brings its own transparency**, blended under
  its per-pixel alpha.

- **`WebRtcTrackSink::set_source_parameters`**, one declaration that tells a
  track sink what feeds it, so a peer is sent SPS/PPS.

- **`Pacer::with_discontinuity_limit`**, for a source whose timeline may
  restart underneath it.

- New `ElementType` variants for everything above, plus `AudioMixerInput` so
  a mixer input says what it is in a topology diagram.

### Changed

- **One place holds where the media timeline sits.** `Clock` went back to
  being the monotonic control and pause clock; `PlaybackClock` owns the media
  origin and answers the mapping both ways. A pipeline used to carry two
  independent origins that could disagree; no graph mixed `Pacer` and
  `VideoSynchronizer` hard enough to expose it, which is why nothing broke.

- The test fixture is now synthesized from this crate's own sources and
  encoders rather than asked for through an environment variable, so the
  library's tests run everywhere against the same file.
  `MEDIA_PP_TEST_VIDEO` is read by `tests/soak.rs` alone.

### Fixed

- Encoders put their codec headers in extradata, so a non-MP4 container gets
  them; WebRTC puts them back in front of every keyframe.
- `AudioMixer` reads the mix in chunks rather than through a constant-sized
  window, and resamples an input through the engine that sizes its output.
- `ChangeGate` times its rate limit from a deadline rather than from the last
  frame it let through.
- A capture's tick loop follows a rate change instead of only reporting one.
- Each segment of a segmented recording starts its timeline at zero.
- WebRTC reads VVC's own NAL header, and accepts an avcC that carries no
  parameter sets.
