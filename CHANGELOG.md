# Changelog

Notable changes to `media-pp`. Versions before 0.2.0 have no entry here — this
file starts where the first breaking release did, and the git history is the
record for everything earlier.

The crate is pre-1.0, so a `0.x` bump is where breaking changes land. Each one
below says what to write instead, because a rename with no migration line is a
compile error with no explanation.

## Unreleased

### Breaking

- **A muxer's sinks are taken by track, not by position.** Every muxer's
  `add_stream` returns a `MuxerTrack`, and `open` returns `MuxerSinks`
  instead of a `Vec<Box<dyn Sink>>`: each track's sink is taken out with the
  track its `add_stream` returned. This applies to `FileMuxer`,
  `SegmentedFileMuxer`, `HlsMuxer`, `RtmpMuxer` and `RtspMuxer` alike.

  ```rust
  // before
  muxer.add_stream("video", video_params, video_time_base)?;
  muxer.add_stream("audio", audio_params, audio_time_base)?;
  let mut sinks = muxer.open()?;
  let audio_sink = sinks.pop().expect("audio was added second");
  let video_sink = sinks.pop().expect("video was added first");

  // after
  let video = muxer.add_stream("video", video_params, video_time_base)?;
  let audio = muxer.add_stream("audio", audio_params, audio_time_base)?;
  let mut sinks = muxer.open()?;
  let video_sink = sinks.take(video)?;
  let audio_sink = sinks.take(audio)?;
  ```

  The order of a `Vec` was the only thing saying which sink was which, so
  every caller repeated the order it had added tracks in, and a track added
  only sometimes — audio when the source has any — made that easy to get
  wrong in a way that still compiled and sent one medium's packets to the
  other's track. `SegmentedFileMuxer::add_stream` still cannot fail, and
  returns the `MuxerTrack` directly.

  `MuxerTrack` is neither `Clone` nor `Copy` and `take` consumes it, so a
  sink cannot be taken twice. A track from a different muxer is refused with
  `MuxerTrackError::ForeignTrack` (through `Error::MuxerTrackError`) rather
  than a panic. `MuxerTrack` is `#[must_use]`: a track whose sink is never
  taken never reports itself finished, and the output is never finalized.

- **`FileMuxerStreamSink`, `HlsMuxerStreamSink`, `RtmpMuxerStreamSink` and
  `RtspMuxerStreamSink` are gone.** Nothing could reach one: every muxer's
  sinks are handed out as `Box<dyn Sink>`, so the names were exports with
  no way to hold a value of them — `redacted_url` on the two publishing
  ones included, which `RtmpMuxer::redacted_url` and
  `RtspMuxer::redacted_url` still answer before `open`. The four were one
  implementation copied four times, and are now one type inside the crate.

- **`CudaConverter` converts either way round, so its constructor is told
  which.** `CudaConverter::new(name, device, width, height)` becomes
  `CudaConverter::new(name, device, output, width, height)`; pass
  `CudaFrameFormat::Nv12` for what it used to do.

  The new direction exists because a camera hands over NV12 and a chroma key
  takes BGRA, so a green screen had nowhere to be keyed on this backend.
  `CudaScaler` cannot fill the gap — it refuses a YUV/RGB pair either way
  and says so in its own docs. On Windows nothing was needed: `D3d11Scaler`
  already takes an output format, and `D3d11ScalerFormat::Bgra` names the
  chroma key among the things it is for.

  The kernel is the exact inverse of the one beside it, written in the order
  that undoes it, and the test asserts a round trip against that inverse
  computed in Rust rather than against a tolerance.

- **Both chroma keys hand out a `ChromaKeyHandle`, so their constructors
  return a tuple.** Keying is tuned by eye, and rebuilding the element for
  each nudge of a threshold means reopening whatever produces its frames —
  a visible stall for a camera, a portal dialog for a Wayland capture.

  ```rust
  // before
  let key = SwChromaKey::new("key", options);
  let key = D3d11ChromaKey::new("key", &device, context, options)?;

  // after
  let (key, handle) = SwChromaKey::new("key", options);
  let (key, handle) = D3d11ChromaKey::new("key", &device, context, options)?;
  ```

  Discard the handle with `let (key, _) = ...` where the settings never
  change. To retune, read, adjust and write back — one lock, and no window
  where half a change is live:

  ```rust
  let mut options = handle.options();
  options.threshold = 0.25;
  handle.set_options(options);
  ```

  `ChromaKeyHandle::set_enabled` turns keying off without disturbing the
  settings, and a disabled element hands each frame straight through — the
  same picture, not a copy. It is kept out of `ChromaKeyOptions` on purpose:
  that struct says how to key and this says whether to, the split
  `AudioVolume` has between its gain and its mute. It also means no
  construction site can produce a chroma key that silently does nothing by
  forgetting a field.

  `ChromaKeyOptions` now derives `PartialEq`, which is how each element
  notices a change and retires the frame its repeat cache was holding. That
  cache is why this is not purely additive: a picture keyed green is not an
  answer to the same picture once the key turned blue, and a still capture
  re-emitting one texture forever would otherwise stay frozen at the old
  settings with nothing to dislodge it.

- **`RtspSink` is now `RtspMuxer`, and carries more than one track.** It
  could only ever publish a single stream, so video and audio could not share
  one RTSP session — which is what the rename is about: every other muxer in
  this crate registers its tracks and then hands out one `Sink` each, because
  a header has to describe them all before the first packet, and this one now
  does the same. A type that returns sinks rather than being one is a
  `*Muxer` here.

  Rename `RtspSink` -> `RtspMuxer`, `RtspSinkError` -> `RtspMuxerError`,
  `Error::RtspSinkError` -> `Error::RtspMuxerError`, and
  `ElementType::RtspSink` -> `ElementType::RtspMuxer`. Deliberately without a
  compatibility alias.

  The call changes shape with the name:

  ```rust
  // before
  let sink = RtspSink::open("rtsp", url, RtspTransport::Tcp, params, time_base)?;

  // after
  let mut muxer = RtspMuxer::create(url, RtspTransport::Tcp)?;
  let video = muxer.add_stream("video", params, time_base)?; // and "audio", if there is one
  let mut sinks = muxer.open()?;                             // the RTSP handshake happens here
  let sink = sinks.take(video)?;
  ```

  The element name moves from `open` to each `add_stream`, so the tracks are
  told apart in logs and `BusEvent`s. `create` no longer touches the network:
  RTSP announces its streams in the header, so the
  `ANNOUNCE`/`SETUP`/`RECORD` handshake is `open`'s to perform and to fail
  at.

  `RtspSink::url` is gone; `RtspMuxer::redacted_url` replaces it. An RTSP URL
  can carry a password in its authority, and the old type logged the URL it
  was given.

  Unchanged: it still remuxes rather than encodes, still keeps a published
  timeline monotonic across an upstream seek (now per track), and still
  finalizes on `Eos` alone rather than on `Stop`.

### Added

- **Colour correction and a luma key: `VideoEffect`, on every backend.**
  `SwVideoEffect`, `D3d11VideoEffect` and `CudaVideoEffect` take a BGRA
  frame and apply one `VideoEffect`:

  ```rust
  let (effect, handle) = D3d11VideoEffect::new(
      "look", &device, context,
      VideoEffect::ColorCorrection(ColorCorrection {
          contrast: 1.2, saturation: 1.1, ..ColorCorrection::default()
      }),
  )?;
  // later, from a slider — or switch it to the other kind entirely
  handle.set_effect(VideoEffect::LumaKey(LumaKey { min: 0.1, ..LumaKey::default() }));
  ```

  `ColorCorrection` is brightness, contrast, saturation, hue, gamma and
  opacity; `LumaKey` cuts out what is darker than `min` or brighter than
  `max`, fading over a smoothing distance past each, and multiplies the
  alpha a pixel already has so it can follow a chroma key. Both defaults
  change nothing, and an element whose effect changes nothing — or that is
  turned off through `VideoEffectHandle::set_enabled` — hands each frame
  straight through.

  One element per backend rather than one per effect, because every effect
  resolves to the same small set of numbers: a colour matrix, an exponent,
  an opacity and a luma mask. One D3D11 shader, one CUDA kernel (PTX, so
  still no toolkit) and one software loop evaluate that, and an effect is
  added once. The GPU backends' tests compare their output with the
  software element's pixel for pixel: exact where the effect is linear,
  within one step where a gamma is set.

- **Speech becomes subtitles: `WhisperTranscriber`, and `subtitle` to carry
  what it says.** Two halves of one feature, kept apart on purpose.

  `WhisperTranscriber` is a terminal sink, like `OrtDetector` and for the
  same reason — what comes out is not media. It takes 16 kHz mono f32 (an
  `AudioResampler` in front is how audio reaches that shape) and hands
  `Segment { start_ms, end_ms, text }` to a callback.

  ```rust
  let transcriber = WhisperTranscriber::new(
      "transcribe", model_path, ChunkPolicy::default(),
      |segment| { println!("{}", segment.text); Ok(()) },
  )?
  .with_language("ko")?;
  ```

  The language is detected unless `with_language` names it, and naming it
  is both faster — detection is run again for every chunk — and steadier,
  since a chunk of music or two words can be heard as another language.
  Detection rather than whisper.cpp's own default, which is English: handed
  Korean under that, it does not fail but writes an English paraphrase.

  Lines arrive in order and never overlap. A line ends no later than the
  audio it was heard in, and one that overlaps the last starts where the
  last stopped — an MP4 text track cannot hold overlapping samples, and its
  muxer wrote negative durations for them.

  What a round reports is cut from what it heard by token, not by segment:
  the words starting inside its window are its own, the ones before were
  reported already, the ones at the edge are left for the next round. A
  segment the model opens in the context and runs into the new window is
  therefore neither lost nor said twice. `ChunkPolicy::token_timing` says
  how those times are found — `TokenTiming::Estimated`, whisper.cpp's own
  estimate, or `TokenTiming::Aligned`, dynamic time warping over the
  model's alignment heads, which are picked from the model file's own
  dimensions. On the minute of Korean below, aligned times left one piece
  said twice at the seams where estimated ones left several, for 6.1x real
  time against 6.4x.

  Whisper's encoder takes exactly 30 seconds, so transcribing a stream is a
  loop, and `ChunkPolicy` is what the loop is made of. `chunk_ms` (4000) is
  how much to gather before each inference; `live_edge_ms` (1000) is how
  much of the newest audio to distrust. The model's output nearest the edge
  is unstable — a word half-heard is guessed at, and the guess changes when
  the rest arrives — so that stretch is left for the next round instead of
  being reported and corrected. **A line, once reported, never changes**,
  which is what lets a caller write it straight to a file. Each inference
  is also given the previous chunk as context, so a word across the seam is
  heard whole; that costs a second pass over audio already transcribed.

  The delay is the sum of the two, so text runs about five seconds behind
  the speech. For a recording that is nothing, because a subtitle's place
  in a file is its timestamp and not its arrival.

  `crate::subtitle` is the other half: `subtitle::Codec` describes a text
  track (`parameters()`) and builds one line of it (`packet()`), in the
  codec its destination takes. `MovText` is for MP4, which takes 3GPP Timed
  Text and nothing else — not SRT, not ASS — and it writes the `tx3g`
  sample entry FFmpeg's own encoder writes, so a track says how its lines
  should look rather than leaving it to the player. `SubRip` and `WebVtt`
  are for an `.srt` or `.vtt` file, or a Matroska track: a sidecar file is
  a `FileMuxer` of its own, FFmpeg picking the writer from the extension,
  and it is written as the lines arrive — a recording that dies halfway
  leaves every line up to that moment. `MediaKind` gains `SubtitlePacket`
  so such a track states its contract like any other.

  Gaps need no packets: FFmpeg's muxer fills them with empty samples, so a
  caller pushes a line when there is something to say and nothing when
  there is not.

  Features: `whisper` builds whisper.cpp for the CPU, `whisper-vulkan` for
  any GPU Vulkan reaches. Vulkan rather than CUDA because CUDA needs a 3 GB
  toolkit to build and serves only NVIDIA, while Vulkan's runtime ships
  with every driver and only its shader compiler is a build requirement.
  Measured on an RTX 3050 against a minute of dense Korean speech, in the
  streaming loop above: `large-v3-turbo` at 6.5x real time and `small` at
  9.7x, where `large-v3-turbo` on the CPU had not finished after fourteen
  minutes. The loop costs more than transcribing a file in one pass — every
  inference pays for a whole 30-second encoder window to hear eight seconds.

  See `examples/core/transcribe`, which writes a copy of a file carrying
  video, audio and the transcription as three tracks.

- **`Rack` holds a stretch of chain whose contents can be replaced while
  frames are flowing.** Everything else in this crate settles its graph
  before the pipeline runs, so changing one element in the middle means
  building the branch again — which restarts whatever is at the top of it.
  For a camera that is a visible stall; on Wayland it is a portal dialog.

  ```rust
  let (rack, rack_handle) = Rack::new(
      "filters",
      InputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)),
      OutputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)),
  );
  // ...pipeline running...
  rack_handle.replace(vec![Box::new(key)])?;   // takes effect on the next buffer
  rack_handle.replace(Vec::new())?;            // and an empty rack is a wire
  ```

  What is in one is still a straight line, so an element with more than one
  output is refused at `RackHandle::replace` rather than a frame later. A
  replacement drops what the outgoing elements were holding, which makes a
  rack right for elements whose output depends on the buffer in front of them
  and nothing else — a scaler, a converter, a chroma key — and wrong for an
  encoder or a muxer. Draining the old line would put its last frames after
  the new line's first ones, and an element in the middle of a graph cannot
  put that timeline back in order.

  Its contracts are declared by the caller rather than derived from what it
  holds, because the caller putting one between two fixed elements knows
  both, and deriving would answer "unknown" across exactly the stretch most
  likely to be wired up wrong.

  Being inside one changes nothing else about an element. On its way in it
  is given what `ChainBuilder` gives a stage it builds: log records naming
  the pipeline it is in, the `Context` if it needs one, and the same tracer,
  so a failure it raises reaches a `Queue` naming *it* rather than naming
  the rack that was holding it. `Box<dyn Filter>` implements `Element`,
  `Source` and `Sink` for that last part — a rack's contents arrive already
  boxed, and the wrapper is written against a type that implements the
  three.

- **`CudaChromaKey` keys a green screen on the GPU under CUDA.** Chroma
  keying existed for the CPU and for D3D11, which left Linux without one at
  all: the compositor there is CUDA, and this crate refuses to wire a branch
  whose memory domains do not match, so the software element could only have
  gone in behind a `CudaDownload` and back out through a `CudaUpload` — two
  PCIe crossings per frame around a per-pixel transform.

  BGRA in, BGRA out, like both siblings, and it keeps PTS, duration and the
  colour tags: keying writes alpha and leaves the colour alone. Odd
  dimensions are fine, unlike `CudaConverter` — BGRA has no subsampled
  plane to halve. `CudaChromaKey::new` returns a `ChromaKeyHandle` beside
  the element.

  The kernel is hand-written PTX carried in the existing BGRA module, so a
  build needs no CUDA toolkit — the driver JIT-compiles it when the module
  loads, exactly as it already did for the conversion and blend kernels. It
  computes the same normalized BGR distance and the same feather ramp
  `SwChromaKey` does, from the same resolved band the D3D11 shader reads,
  which is now one shared `feather_band` rather than a copy per backend.

  New: `Error::CudaChromaKeyError` and `ElementType::CudaChromaKey`.

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

- **`AudioGate` and `NoiseSuppressor` clean up a microphone.** The two
  audio filters a streaming application reaches for first, working on the
  signal rather than its format: both take `f32` audio, packed or planar,
  and pass it on with its timestamps and layout unchanged.

  `AudioGate` silences what is below a level and passes what is above it,
  with a streaming application's five settings — open and close thresholds
  in dBFS, attack, hold and release — and their meaning. Two thresholds so a
  level hovering at the line does not open and close the gate on every
  syllable; a hold so a pause between words is not cut out. One gain for
  every channel, taken from the loudest, so a stereo image never leans.
  `AudioGateHandle::set_options` replaces all five at once from the next
  frame. No delay.

  ```rust
  let (gate, handle) = AudioGate::new("mic-gate");
  handle.set_options(AudioGateOptions { open_threshold_db: -30.0, ..handle.options() })?;
  ```

  `NoiseSuppressor`, behind the new `rnnoise` feature, takes steady
  background noise — a fan, a room — out of speech with RNNoise, through
  `nnnoiseless`: pure Rust, the weights compiled in, no model file to ship.
  48 kHz only, as the network was trained; about 0.4% of one core per
  channel. RNNoise gives each 10 ms block back one block late, and that
  block is taken off the front rather than left to push the sound late
  against its timestamps: frames come out one for one, a block after they
  arrive, each carrying its own samples denoised. `Eos` drains the block it
  holds.

  New: `Error::AudioGateError`, `Error::NoiseSuppressorError`,
  `ElementType::AudioGate`, `ElementType::NoiseSuppressor`, and the
  `rnnoise` feature.

- **`AudioCompressor` and `AudioLimiter` even a voice out.** The next two
  after a gate, with its format, its linked gain and its handle.

  `AudioCompressor` turns down what goes over a threshold by a ratio — at
  4:1 a level 12 dB over comes out 3 dB over — with a streaming
  application's five settings: threshold, ratio, attack, release and an
  output gain to make back up what it took off. The level it reacts to is
  followed per sample, rising over the attack and falling over the release,
  so it neither pumps on each waveform peak nor lets a word start at full
  volume.

  `AudioLimiter` is the last thing on a channel: nothing comes out louder
  than its threshold. It does not look ahead, so it reacts in the sample a
  peak arrives, turning down to exactly what brings that sample to the
  ceiling; that is what makes the threshold a guarantee rather than a
  target. It lets go over its release.

  ```rust
  let (compressor, _) = AudioCompressor::new("mic-compressor");
  let (limiter, handle) = AudioLimiter::new("mic-limiter");
  handle.set_options(AudioLimiterOptions { threshold_db: -3.0, ..handle.options() })?;
  ```

  New: `Error::AudioCompressorError`, `Error::AudioLimiterError`,
  `ElementType::AudioCompressor`, `ElementType::AudioLimiter`.

- **`Pipeline::stats` says what every element is doing while it runs.** The
  graph says what a pipeline looks like; this says whether anything moves
  through it — a capture that stopped delivering, a queue that is full and
  dropping, a branch still draining after it was finished. Each used to be
  found in a log afterwards.

  ```rust
  for element in pipeline.stats().elements {
      println!("#{} {} in={} busy={:?} idle={:?}",
          element.id, element.name, element.buffers_in,
          element.busy, element.idle_for);
  }
  ```

  Per element: buffers taken, time spent inside `consume`, how long since it
  last took or pushed a buffer, errors, and whether it has seen `Eos`; per
  output pad, what went through it, and the bytes of the packets among it —
  an encoder's bitrate; for a `Queue`, how full it is, what it dropped and
  how long its upstream waited on it; for a video compositor, the frames it
  drew, the ticks of its frame rate it missed because the one before ran
  late, and the time spent drawing — not handing the frame on, so a slow
  encoder downstream shows up as missed ticks with little drawing time,
  and is told apart from a compositor that is itself too slow. Running totals rather than
  rates — two readings and the time between them give the rate, matched by
  `ElementStats::id`. `busy` includes every stage after it on the same
  thread, up to the next `Queue`, because a chain runs as nested calls.

  Runtime `Tee` branches are covered as they come and go. An attached branch
  is reported from the moment it joins; a detached one disappears at once;
  one ended with `finish_branch` goes on appearing, as
  `ElementState::Finishing`, while its `Eos` drains, and disappears once it
  has been dropped. A re-attached branch is new elements with new ids.

  Always on, and counted in the wrappers every stage and pad already has,
  so an element written outside this crate is counted like one inside it.
  The cost is about a tenth of a microsecond per buffer per stage, nearly
  all of it reading the clock.

  New: the `stats` module (`PipelineStats`, `ElementStats`, `ElementState`,
  `PadStats`, `QueueStats`, `TickStats`).

### Changed

- **A chroma key multiplies the alpha it is given instead of replacing it.**
  `SwChromaKey`, `D3d11ChromaKey` and `CudaChromaKey` used to write the key's
  coverage straight into alpha, so a key placed after a luma key put back
  everything that one had taken out. An opaque input — every capture and
  decoder in this crate — keys exactly as before, byte for byte.

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
