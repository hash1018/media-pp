# Changelog

Notable changes to `media-pp`. Versions before 0.2.0 have no entry here — this
file starts where the first breaking release did, and the git history is the
record for everything earlier.

The crate is pre-1.0, so a `0.x` bump is where breaking changes land. Each one
below says what to write instead, because a rename with no migration line is a
compile error with no explanation.

## Unreleased

### Breaking

- **`Sink::control` is an element's own reaction; the graph passes the
  message on.** It took a `ControlMsg`, had no default, and every filter
  ended it by forwarding to its own pad — so a filter that forgot, or
  returned early on an error, stopped the message there, and a message
  added later had to be taught to every element. It now takes
  `&ControlMsg`, does nothing by default, and does not forward: whatever
  delivers a message to a filter hands it on through the filter's
  `src_pads()` afterwards — to every pad even where one fails, and even
  where the filter's own reaction failed. `SrcPad::control` is no longer
  public, since a filter still calling it would hand everything after it
  each message twice. To migrate, take `msg: &ControlMsg`, end with
  `Ok(())` where you called `self.pad.control(msg)`, and delete a
  `control` that did nothing else. Code that drives a filter by hand,
  outside a pipeline, calls `control::deliver(&mut filter, &msg)` for the
  old forwarding. `AppSink::with_control`'s callback is unchanged.

- **`RtspMuxer::open` gives up on a server that does not answer.** It
  waited without limit: publishing to an address with nothing listening,
  or to a server that took the connection and never replied, never
  returned. `RtspMuxer::create` takes the `RtspOptions` `RtspSource::open`
  already did — transport and a timeout, 5 seconds by default — and the
  timeout bounds the handshake and every write after it. Write
  `RtspMuxer::create(url, RtspOptions::default())`, or
  `RtspOptions { transport, ..Default::default() }` for a transport of your
  own. `RtspOptions` is still `media_pp::elements::RtspOptions`; the
  duplicate path through `elements::source` is gone.

- **`FileMuxer::create` names the file it could not create.** A name with
  no container in it — `out`, `out.unknownext` — was
  `FileMuxerError::Ffmpeg(Invalid argument)`, and a directory that is not
  there `No such file or directory`, neither saying which file. They are
  now `FileMuxerError::UnknownContainer { path }` and
  `FileMuxerError::Create { path, source }`. A match on `FileMuxerError`
  needs the two arms.

- **A presenter of your own is told what colour an NV12 frame is.**
  `D3d11FrameRenderer::submit_nv12_texture`,
  `D3d12FrameRenderer::submit_nv12_texture` and
  `CudaFrameRenderer::submit_nv12` take a last `color: ColorDescription`
  parameter: what the frame says of its matrix, range, primaries and
  transfer. Without it a presenter could only guess, and the usual guess —
  one fixed BT.601 matrix — draws a decoded HD stream, which is BT.709,
  with visibly wrong colours. Add the parameter to an implementation;
  `color.yuv_to_rgb_rows(height)`, also new, gives the three rows a shader
  converts with, filling in what the frame leaves unsaid the way this
  crate's own renderers do. `ColorDescription::of(frame)` reads the same
  description from any frame.

- **`FileDemuxer::open` names the file it could not open, and refuses one
  with nothing in it.** A missing or unreadable file was
  `FileDemuxerError::Ffmpeg("No such file or directory")`, with no word of
  which; it is now `FileDemuxerError::Open { path, source }`, which says. A
  file that opens with no streams at all — a recording finished before
  anything was written into it — was opened with an empty stream list for a
  caller to index into; it is now `FileDemuxerError::NoStreams { path }`.
  Code matching `FileDemuxerError::Ffmpeg` for an open failure matches
  `Open` instead.

- **`StreamInfo` has a `frame_rate` field**, so a literal of one needs it
  — see Added.

- **A file's source waits at its end instead of ending.** `FileDemuxer`
  sent `Eos` at the end of the file and ended its thread, and with it the
  queues it owns, a second or more of the file still in them. A `Pause` from
  then on reached nothing, so the last second played on through one; a
  paused seek near the end lost its picture and its `Eos` and never
  finished; and a file could not be sought back once it had ended —
  `PipelineError::NotRunning`. It now stays at the end, passing control on,
  until the pipeline is stopped, and a seek from there plays on. So a
  pipeline reading a file no longer ends by itself: `Pipeline::is_running`
  stays `true` and `BusReceiver::iter` does not end until it is stopped.
  Stop it on `BusEvent::Finished`, which says everything read has reached
  its terminals — what the examples and `Player` already did. A
  `FileDemuxer` driven by hand returns from `run` at the end once its
  control sender is gone.

- **D3D elements take their `D3d11Gpu` or `D3d12Gpu`.** Every D3D11
  element took an `&ID3D11Device` and, where it draws or copies, a separate
  `Arc<Mutex<ID3D11DeviceContext>>` — two values that had to belong to the
  same device, read out of a `D3d11Gpu` one at a time, next to a
  `D3d11WindowRenderer::open(&gpu)` that already took the GPU whole. Each
  now takes `gpu: &D3d11Gpu` where the device went, and the context
  argument is gone: the element reads both from the one GPU, so a context
  from another device can no longer be handed over. The D3D12 elements take
  `&D3d12Gpu` the same way. That is `D3d11ChromaKey::new`,
  `D3d11Decoder::new`, `D3d11Download::new`, `D3d11Scaler::new` and
  `to_format`, `D3d11ToneMap::new`, `D3d11Upload::new`,
  `D3d11VideoCompositor::new`, `D3d11VideoEffect::new`,
  `D3d11VideoEncoder::new` and `with_color`,
  `D3d11SharedTextureSource::new`, `DxgiCaptureSource::open_with_device`,
  `WgcCaptureSource::open_with_device`, `D3d12Decoder::new`,
  `D3d12Scaler::new` and `D3d12Upload::new`; and `DecodeTarget::D3d11`
  holds `gpu` in place of `device` and `context`, `DecodeTarget::D3d12`
  `gpu` in place of `device`.

  A capture that makes its own device hands it back ready to share:
  `DxgiCaptureSource::open` returns an `Option<D3d11Gpu>` where it returned
  an `Option<ID3D11Device>`, and `WgcCaptureSource::open` a `D3d11Gpu`
  where it returned an `ID3D11Device`, each with a new `Gpu` variant on its
  error for the one way that can fail. The `ContextDeviceMismatch` variants
  of `D3d11ChromaKeyError`, `D3d11ScalerError`, `D3d11ToneMapError`,
  `D3d11VideoEffectError` and `D3d11VideoEncoderError` are gone, there
  being no context left to mismatch; a single-threaded device is refused by
  `D3d11Gpu::from_device`, before any element has it.

  ```rust
  // before
  let upload = D3d11Upload::new("upload", gpu.device());
  let scaler = D3d11Scaler::new("scale", gpu.device(), gpu.context(), format, 1280, 720)?;
  let decoder = D3d11Decoder::new("decoder", params, gpu.device(), 8)?;
  let (capture, _format) = DxgiCaptureSource::open_with_device("screen", options, gpu.device())?;
  let target = DecodeTarget::D3d11 {
      device: gpu.device().clone(),
      context: gpu.context(),
      downstream_hw_frames: 8,
  };
  let (source, format, device) = DxgiCaptureSource::open("screen", options)?;
  let gpu = D3d11Gpu::from_device(device.expect("GPU mode returns a device"))?;
  let upload = D3d12Upload::new("upload", d3d12.device())?;

  // after
  let upload = D3d11Upload::new("upload", &gpu);
  let scaler = D3d11Scaler::new("scale", &gpu, format, 1280, 720)?;
  let decoder = D3d11Decoder::new("decoder", params, &gpu, 8)?;
  let (capture, _format) = DxgiCaptureSource::open_with_device("screen", options, &gpu)?;
  let target = DecodeTarget::D3d11 { gpu: gpu.clone(), downstream_hw_frames: 8 };
  let (source, format, gpu) = DxgiCaptureSource::open("screen", options)?;
  let gpu = gpu.expect("GPU mode returns a GPU");
  let upload = D3d12Upload::new("upload", &d3d12)?;
  ```

- **`PixelLayout::Yuv420p` and `Rgb24`: a link check tells them from other
  layouts.** YUV420P was `PixelLayout::Other`, so a consumer that draws it
  had to accept every other layout too, and a pipeline feeding it RGB24 or
  4:4:4 linked and failed on its first frame. `PixelLayout::of` maps
  `YUV420P` and `YUVJ420P` to the new `Yuv420p`, which `SwScaler` and
  `SwEncoder` state through it with no change of their own, and
  `TestVideoSource` states it too; `PixelLayoutSet::YUV420P` is the set of
  it alone. `PixelLayout::Rgb24` is added beside it for the same reason:
  `OrtDetector` takes RGB24 alone and said `Other`, which linked a YUV420P
  source to it; it says `Rgb24` now. `Other` is left for producers — a
  format with no layout of its own, which a consumer naming its layouts
  refuses — and no consumer here states it. A `match` on `PixelLayout`
  needs arms for both, and a contract written with `Other` to take YUV420P
  or RGB24 needs the new layout instead.

- **A preroll timeout names what it waited on.** `PrerollError::TimedOut`
  carried `pending: Vec<ElementId>`, so a failed seek said
  `pending terminals [ElementId(8)]` and left the caller to work out which
  branch that was. It carries `Vec<NodeInfo>` — each terminal's id, type
  and name, from the pipeline's graph — and reads `preroll timed out
  waiting on audio (Other #8)`.

- **`SwEncoder` is opened for one pixel format, and refuses any other.**
  It encoded whatever frame it was handed as if it were `YUV420P`: a BGRA
  screen capture wired straight in linked, ran without an error, and wrote
  a file that played as solid green. `SwEncoderOptions` has a new required
  `pixel_format` — `YUV420P` for what every example here does — which the
  encoder is opened for after checking that the codec takes it
  (`SwEncoderError::UnsupportedPixelFormat` names the ones it does), so a
  codec such as `libx264` can take NV12 or 10-bit frames with no conversion
  in front. A frame in another format or size is refused with
  `SwEncoderError::FrameMismatch`, and a wiring that can only deliver
  another layout no longer links. `YUVJ420P` fits a `YUV420P` encoder, being
  the same planes.

- **`FileDemuxError` is `FileDemuxerError`.** Every other element's error
  is named after the element — `FileMuxerError`, `SwDecoderError` — and a
  search for `FileDemuxerError` found nothing. The crate `Error` variant is
  renamed with it.

- **Screen captures take a `frame_rate`, not `fps`.** `DxgiCaptureOptions`,
  `WgcCaptureOptions` and `PipeWireScreenCaptureOptions` took `fps: u32`,
  while their own runtime `FrameRateHandle::set`, every encoder, the
  compositors, the test sources and the webcam all take an
  `ffmpeg::Rational`. A recording wrote the same rate twice in two shapes,
  and `30000/1001` could not be asked for at all. Each now has
  `frame_rate: ffmpeg::Rational` (`30/1` by default), so one value goes to
  the capture and the encoder alike. A rate that is not positive is refused
  by `open` with the new `InvalidFrameRate` on `DxgiCaptureSourceError` and
  `PipeWireScreenCaptureSourceError` — DXGI and PipeWire used to run a `0`
  at 1 fps — and `WgcCaptureSourceError::InvalidFps` becomes
  `InvalidFrameRate` carrying the rate. PipeWire refuses it before the
  portal shows a dialog.

- **A `Pipeline` says when `run` or `seek` did nothing.** Calling `run`
  on a pipeline that had already been run returned `Ok` and did nothing,
  so a second play-through showed neither an error nor any playback; it
  fails with the new `PipelineError::AlreadyStarted` and leaves the first
  run alone. `seek` before `run`, or once every source has stopped,
  returned `Ok` without moving anything; it fails with
  `PipelineError::NotRunning`. A source parked at the end of its stream
  still counts as running, so seeking back into a finished file works as
  it did. `PipelineError` is in `media_pp::pipeline` and converts into the
  crate's `Error`. `pause`, `resume` and `stop` stay quiet no-ops outside a
  run, since asking for a state a pipeline is already in is harmless.

- **`log::init` takes its directory as a path.** `log_path` was a `&str`,
  the one file location in the crate that was, so a caller holding a
  `PathBuf` had to convert it with `to_string_lossy` — which turns a name
  that is not UTF-8 into a different one, and logged somewhere other than
  where it was asked to. It is `impl AsRef<Path>` now: a `&str` or
  `&String` still works as it did, and a `Path` or `PathBuf` goes in
  directly. What no longer compiles is what relied on coercing to `&str`,
  such as `&path.to_string_lossy()`; pass `&path` itself.

- **`CudaUpload::new` no longer returns a `Result`.** Drop the `?`:
  `CudaUpload::new(name, &device, format)?` is
  `CudaUpload::new(name, &device, format)`. It never had a failure to
  report — it takes a reference to the device and builds its pads, and the
  surfaces it uploads into are allocated when the first frame says what
  size they are, which is where a CUDA error has always come from. The two
  other uploads are unchanged: `D3d11Upload::new` does not fail either and
  never said it did, and `D3d12Upload::new` does, when it allocates up
  front.

- **A muxer's `add_stream` takes the encoder or stream it describes.**
  `add_stream(name, parameters, time_base)` on `FileMuxer`,
  `SegmentedFileMuxer`, `HlsMuxer`, `RtmpMuxer`, `RtspMuxer` and
  `ReplayBuffer` is now `add_stream(name, format)`, where `format` is
  anything that converts into the new `TrackFormat`: `&encoder` for any of
  the four encoders, or a demuxed `&StreamInfo`. The two values described
  one stream and were passed separately, so one encoder's time base could
  go in beside another's parameters; now they come from the same place.
  `TrackFormat::new(parameters, time_base)` is left for a stream described
  by hand.

- **A Tee starts from the pipeline's context: `ctx.tee(name)`.**
  `TeeBuilder::new(name, ctx.clone())` is now `ctx.tee(name)`, beside
  `ctx.branch()`; the rest of the builder (`branch`, `build`,
  `build_dynamic`) is unchanged. A Tee's branches can only belong to the
  pipeline it is wired in, so there was never another context to hand it,
  and `TeeBuilder::new` is no longer public.

- **`FrameCounter` and `PacketCounter` hand back a `CounterHandle`.** Their
  `new` returned the `Arc<AtomicUsize>` they counted into, so reading it
  took an `Ordering` import and `.load(Ordering::Relaxed)`, and a caller
  could overwrite the count. `count.get()` reads it now; the handle is
  read-only, cheap to clone, and keeps reading the final number after its
  sink is dropped.

- **`media_pp::init` is gone; the library readies FFmpeg itself.** Delete
  the `media_pp::init()?;` line. It registered FFmpeg's error descriptions
  and its input devices, and forgetting it was silent: every FFmpeg error
  displayed as `ffmpeg error: ` with nothing after it, and a camera opened
  through FFmpeg was not found. Every element that reaches FFmpeg now does
  this once, on its way in, so there is nothing to remember.

- **A setter that refuses says why.** `set_frame_rate` on the three
  compositor handles, `MixerHandle::set_mix_format` and
  `rate::FrameRateHandle::set` (the captures' rate handles) returned
  `false` both for a value they could not take and for an element that had
  already stopped, so a caller could not tell which. Each returns a
  `Result`: `InvalidFrameRate` or `Stopped` from the compositor's own
  error, the new `AudioMixerError::InvalidMixFormat` or `Stopped` from the
  mixer, and the new `rate::FrameRateError` (`Invalid` or `Stopped`) from a
  capture. The running value is left alone either way, as before.

- **`PipelineBridgeHandle::connect` returns a `Result`.** It returned
  `None` once the bridge's pipeline had finished; it returns
  `PipelineBridgeError::Disconnected`, the bridge's own word for that, as
  the other handles now answer with an error of their own.

- **The demuxers' per-stream lookups are gone; `best` and `open`'s list
  carry the same.** `FileDemuxer` and `RtspSource` lose `best_stream`,
  `stream_parameters` and `stream_time_base`. `best(kind)` returns the
  `StreamInfo` of the stream to play — its `index`, `parameters` and
  `time_base` together — and `open` already returns one for every stream,
  each at its own index:

  ```rust
  // before
  let index = source.best_stream(media::Type::Video).ok_or(..)?;
  let parameters = source.stream_parameters(index).ok_or(..)?;
  let time_base = source.stream_time_base(index).ok_or(..)?;

  // after
  let video = source.best(media::Type::Video)?;
  // video.index, video.parameters, video.time_base
  ```

- **`TeeHandle::branch` returns a `Result`, like the rest of the handle.**
  It returned `None` once its `Tee` was gone, so every caller wrote its own
  `.ok_or("the Tee is gone")?`, while `attach` and `detach` on the same
  handle already returned `GraphError::ParentNotAttached`. It returns that
  error too now: `tee.branch()?` is the whole call.

- **`framerate` is `frame_rate` everywhere.** `TestVideoOptions::framerate`,
  `MfCaptureFormat::framerate`, `V4l2CaptureFormat::framerate` and the
  `framerate` field of `MfCaptureSourceError::FormatNotOffered` are renamed to
  `frame_rate`, the name the encoder and compositor options already used —
  so a test source's rate is handed to an encoder as `frame_rate:
  options.frame_rate` rather than across two spellings. The screen
  captures' `fps: u32` stays as it is: it is a cap in whole frames, not a
  rate.

- **`Pipeline::new` hands back what its wiring returns.** It returns
  `(Arc<Pipeline>, T)`, where `T` is whatever the `wire` closure returned —
  so something only the wiring can make, a `TeeHandle` from
  `build_dynamic` or a routing to keep, comes back out as a value rather
  than through a variable the closure fills in and the caller then has to
  `expect`. Several at once are a tuple. A closure with nothing to hand
  back returns `Ok(())` as before, and the call binds `()`:

  ```rust
  // before
  let mut tee = None;
  let pipeline = Pipeline::new("fan", source, |source, ctx| {
      let (branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
      ctx.attach(source, 0, branch)?;
      tee = Some(handle);
      Ok(())
  })?;
  let tee = tee.expect("wire ran");

  // after
  let (pipeline, tee) = Pipeline::new("fan", source, |source, ctx| {
      let (branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
      ctx.attach(source, 0, branch)?;
      Ok(handle)
  })?;

  // and where the closure returns nothing
  let (pipeline, ()) = Pipeline::new("play", source, |source, ctx| { /* ... */ Ok(()) })?;
  ```

  `PipelineBuilder::add_source` does the same: it returns `(builder, T)`,
  so a multi-source pipeline is built a statement at a time rather than as
  one chain:

  ```rust
  let builder = PipelineBuilder::new("record");
  let (builder, ()) = builder.add_source(video, |source, ctx| { /* ... */ Ok(()) })?;
  let (builder, routing) = builder.add_source(audio, |source, ctx| { /* ... */ Ok(routing) })?;
  let pipeline = builder.build();
  ```

- **Adding an input to a stopped compositor or mixer is an error, the same
  on every backend.** `D3d11VideoCompositorHandle::add_source`,
  `add_layer` and `add_text_layer` and `SwVideoCompositorHandle::add_source`
  returned `Ok(None)` once their compositor was gone, and
  `MixerHandle::add_source` returned `None`, so every caller wrote its own
  `.ok_or("the compositor is gone")?` — while the CUDA compositor already
  returned an error. Each now returns its input directly, or a new
  `Stopped` variant of its own error (`D3d11VideoCompositorError`,
  `SwVideoCompositorError`, `CudaVideoCompositorError`, `AudioMixerError`):
  drop the `Option` handling, `handle.add_source(name, layer)?` is the whole
  call. The CUDA compositor reports `Stopped` there too, where it used to
  borrow `SourceRemoved`, whose message is about a removed input.

- **`Error` is `#[non_exhaustive]`, and every public error converts into
  it.** A `match` on `media_pp::Error` needs a `_` arm; in return, an
  element added later — and each one adds a variant — no longer breaks
  it. `CudaDeviceError`, `CudaDriverError`, `CudaFrameError`, `RackError`,
  `PlaybackClockError`, `PipeWireDeviceError` and `DmaBufCudaError` gain
  variants of their own, so each passes through `?` where it used to have
  to be turned into a string first:

  ```rust
  // before
  let cuda = CudaDevice::new().map_err(|e| Error::Other(e.to_string()))?;

  // after
  let cuda = CudaDevice::new()?;
  ```

  A string keeps nothing to match on, and the examples taught it: 83 of
  them did it, most to errors that already converted. They no longer do.
  `SubmitError`, which a renderer implementation returns, is now a
  `std::error::Error` with messages of its own.

- **An upload, a download and a converter take the size from the frames.**
  `CudaUpload::new`, `D3d11Upload::new`, `D3d12Upload::new`,
  `CudaDownload::new`, `D3d11Download::new`, `D3d12Download::new` and
  `CudaConverter::new` no longer take `width` and `height`: drop the last
  two arguments. None of these elements changes a frame's size, so the
  size they were told had to agree with whatever the upstream element
  produced, and a caller that got it wrong learned so one frame later
  through a `DimensionMismatch` error. That error variant is gone from all
  seven.

  ```rust
  // before
  let converter = CudaConverter::new("to-nv12", &cuda, CudaFrameFormat::Nv12, width, height)?;

  // after
  let converter = CudaConverter::new("to-nv12", &cuda, CudaFrameFormat::Nv12)?;
  ```

  This is what the element a refused link names now costs to build: the
  remedy in the error message is the whole of it, with no stream
  parameters to open for a size — `hw_decode_render` opened a codec
  context solely for that and no longer does. The texture, staging buffer
  or `AVHWFramesContext` each element needs is made for the first frame's
  size and made again when that changes, so **a source that changes
  resolution mid-stream is now followed rather than refused** — an RTSP
  camera switching profile, a window capture being resized. A failed
  allocation for a new size leaves the element serving the size it already
  had. `CudaConverter`'s even-dimension requirement is now checked per
  frame, so `CudaConverterError::OddDimensions` comes back from `consume`
  rather than from `new`.

  `CudaChromaKey::new` and `CudaVideoEffect::new` lose the same two
  arguments, for the same reason and with the same consequences — keying and
  colour correction are per-pixel, and their D3D11 siblings never took a
  size. `CudaChromaKeyError::DimensionMismatch` and
  `CudaVideoEffectError::DimensionMismatch` are gone with them.

  A scaler, a compositor and an encoder still take a size: it is what they
  are for, not something they have to be told twice.

- **A CUDA element refuses a frame through one error, `CudaFrameError`.**
  `CudaConverter`, `CudaChromaKey`, `CudaVideoEffect`, `CudaDownload`,
  `CudaEncoder`, `CudaRenderer`, `CudaScaler` and `CudaVideoCompositor`
  each carried their own copy of the same four checks — a CUDA frame, with a
  frames context, from this element's device, in a layout it reads — as
  their own four error variants, and their own copy of the unsafe code
  asking them. Those variants (`UnsupportedFormat`, `MissingFramesContext`,
  `ForeignContext`, `UnsupportedSurfaceFormat`, and the renderer's
  `UnsupportedSoftwareFormat`) are replaced in every one of them by
  `Frame(CudaFrameError)`, and the checks by one function.

  ```rust
  // before
  Err(Error::CudaDownloadError(CudaDownloadError::ForeignContext)) => …

  // after
  Err(Error::CudaDownloadError(CudaDownloadError::Frame(
      CudaFrameError::ForeignContext { .. },
  ))) => …
  ```

  The messages still name the element — `CudaDownload reads NV12 surfaces,
  got BGRA` — and a refused layout now says what would have been taken,
  through `CudaSurfaces`, the same value that made the decision.

- **A frame's link contract states its pixel layout.**
  `PortContract::Frames` has a third field, a `PixelLayoutSet` — NV12, P010,
  BGRA or other — and `OutputContract` a new variant, `SameLayout`, for a
  port that passes on the layout it was given. Where a `Frames` value was
  built by hand, add `PixelLayoutSet::ALL`, which is what
  `PortContract::frame` and `any_frame` still start from; where
  `OutputContract` was matched, add an arm for `SameLayout`.

  This crate's elements state their layouts where construction settles
  them, so a branch that could only fail frame by frame is refused when it
  is built: a `VideoDecodeBin` that will put out BGRA wired straight into a
  `CudaRenderer`, which presents NV12, now fails at `Pipeline::new` naming
  both rather than erroring on every frame. A producer that states no
  layout is not checked against one, and a resize-only scaler
  (`D3d11Scaler` with `Preserve`, `CudaScaler::new`) or `D3d11Upload`
  passes on the layout of what reaches it. Every example and obs-rs's
  own pipelines link as before; one test did not — `TestVideoSource`'s
  YUV420P wired into a `D3d11Upload`, which only takes NV12 or BGRA and
  would have refused every frame.

- **`StreamInfo` carries each stream's parameters and time base, and is no
  longer `Copy`.** `FileDemuxer::open` and `RtspSource::open` report
  `parameters` and `time_base` beside `index` and `kind`, so a branch is
  built from what `open` returned without asking again by index. Holding
  FFmpeg's parameters makes it `Clone` only; where a `StreamInfo` was
  copied, clone it or take a reference. `stream_parameters` and
  `stream_time_base` still answer the same.

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
  `CudaConverter::new(name, device, output)` — the size going the way of the
  entry above; pass `CudaFrameFormat::Nv12` for what it used to do.

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

- **`CudaDecoderError::UnsupportedCodec`, and the same on `D3d11DecoderError`
  and `D3d12DecoderError`.** Each hardware decoder's `new` refuses a codec no
  decoder can decode on its device with it (see Fixed), so a `match` over
  any of the three needs an arm for it.

- **`Pacer` reads the time base off each buffer instead of being told it.**
  `Pacer::new(name, time_base)?` is `Pacer::new(name)`, and
  `Pacer::with_discontinuity_limit(name, time_base, limit)?` is
  `Pacer::with_discontinuity_limit(name, limit)`; neither can fail any more,
  so drop the `?`. Told separately, a pacer could be told another stream's
  unit — the video's handed to the audio branch — and play at the wrong
  speed with nothing to say so; a packet carries its own, and now every
  frame this crate makes does too (see Added). `PacerError::InvalidTimeBase`
  is gone, and `PacerError::NoTimeBase` refuses a timed frame that does not
  say what unit its `pts` is in — one pushed through an `AppSource` or made
  by an element of your own — rather than guessing. Stamp such a frame with
  `buffer::set_time_base`.

- **`AudioResampler`, `FrameRateLimiter` and `VideoSynchronizer` read it
  off the frame too.** `AudioResampler::new(name, target, input_time_base)?`
  is `AudioResampler::new(name, target)`,
  `VideoSynchronizer::new(name, time_base)?` is `VideoSynchronizer::new(name)`
  — neither can fail any more — and
  `FrameRateLimiter::new(name, input_time_base, rate)` is
  `FrameRateLimiter::new(name, rate)`. Each was the same value that had to
  agree with the stream. `AudioResamplerError::InvalidTimeBase` and
  `VideoSynchronizerError::InvalidTimeBase` give way to a `NoTimeBase`
  variant on each, and the new `FrameRateLimiterError` has the one variant
  `NoTimeBase` — all refuse a timed frame that does not say its unit, as
  `Pacer` does. No element takes a stream's time base any more except a
  muxer's `add_stream`, which writes it into a header before the first
  packet arrives.

- **The encoders count in a unit of their own and read the frames'.**
  `time_base` is gone from `SwEncoderOptions`, `SwAudioEncoderOptions`,
  `CudaEncoderOptions` and `D3d11VideoEncoderOptions`: drop the field. It
  had to match the unit of the frames' timestamps, and a wrong one
  encoded a file that played at the wrong speed. A video encoder now
  counts in 1/90000 (1/60000 for `VideoCodec::Mpeg4`, whose bitstream
  holds no more) and converts each frame's `pts` from the unit the frame
  carries; the audio encoder counts samples at its output rate, as it
  already did whatever its option said. Each says its unit through
  `time_base()` — new on the three video encoders — which is what to
  hand a muxer's `add_stream`:

  ```rust
  // before
  let encoder = SwEncoder::new("encode", SwEncoderOptions { time_base, .. })?;
  let track = muxer.add_stream("video", encoder.parameters(), time_base)?;

  // after
  let encoder = SwEncoder::new("encode", SwEncoderOptions { .. })?;
  let track = muxer.add_stream("video", encoder.parameters(), encoder.time_base())?;
  ```

  A timed frame with no unit is refused with a new `NoTimeBase` variant
  on `SwEncoderError`, `CudaEncoderError` and `D3d11VideoEncoderError`.
  And a muxer track now reads each packet in the unit the packet carries,
  falling back on the one `add_stream` was given only for a packet that
  carries none — so a track registered with the old frame unit still
  writes its packets at the right times.

### Added

- **`SourceElement::pausing` and `SourceElement::resuming`**, called as a
  pause begins — before the `Pause` goes downstream — and as it ends —
  after the `Resume` has, before it is acknowledged. What a source does to
  stop and restart its own input goes there, and `drain_control` does the
  rest; the WASAPI, Media Foundation and PipeWire audio captures each kept
  a copy of the pause wait to do it, and now use these. Both default to
  doing nothing, so no source has to change.

- **`DxgiCaptureSource::outputs`** lists what `CaptureArea::Output` can
  name, by the index that names it, with each monitor's name, place on the
  desktop, and which one is primary — output 0 need not be.
  **`MfCaptureSource::frame_rate`** says the rate a camera was opened at,
  which an encoder after it needs and `open` with no format chosen did not
  say.

- **Re-encoding a file needs nothing outside the crate.** `StreamInfo`
  says a video stream's `frame_rate` and, through `size` and `color`, its
  picture size and colour description — what an encoder re-encoding it is
  opened with, which took reaching into FFmpeg before.
  `EncodeInput::for_decoded` turns a decode bin's target and output format
  into the input an encode bin takes, and the new `transcode` example puts
  them together: `FileDemuxer -> VideoDecodeBin -> VideoEncodeBin ->
  FileMuxer`, decoded onto and encoded from the GPU, the sound copied as it
  is. The README shows a file pipeline watched to `Finished` and stopped.

- **`Player` plays sound on its own, and has a volume, a loop and a choice
  of track.** A file with no picture was refused as one with nothing to
  play; it now plays, with its waveform in the window — drawn by the new
  `AudioWaveform` from the same decoded sound and shown by a
  `VideoSynchronizer` as it is heard — and the keys arrive there as ever.
  `AudioWaveform` is an element of its own: `f32` sound in, a BGRA
  oscilloscope trace out at a steady rate, each picture stamped with the
  moment of the sound it ends at. `set_volume`, `set_muted` and the up and down arrows and M in
  `respond_to` turn the sound, through an `AudioVolume` in its branch;
  `set_looping` plays the file again from the start instead of ending, with
  `position` read within the lap; and `PlayerOptions::audio_stream` picks
  one of several sound tracks by its stream index. `decoding` is `None` for
  a file with no picture.

- **`VideoEncodeBin`: H.264 by whichever encoder opens.** Encoding took
  choosing between `SwEncoder`, `D3d11VideoEncoder` and `CudaEncoder` by
  hand, with the download, conversion and colour description each one
  needs in front — obs-rs had grown its own probe for it. The bin takes
  frames from system memory, D3D11 or CUDA (`EncodeInput`) and opens
  `h264_nvenc`, then `h264_mf` for D3D11, then software (`libx264`, or
  `libopenh264`), keeping the first that opens; `path` says which, and a
  muxer's `add_stream` takes the bin itself. Every path says in the stream
  what its YUV is — from RGB, what it converted with; from YUV,
  `VideoEncodeOptions::color`. Chosen once, when it opens: a muxer writes
  its headers from that encoder. `D3d11Download` reads NV12 textures as
  well as BGRA, into NV12 frames with their colour description, so the
  software path from a decoder's or an upload's textures needs no video
  processor — which a device with none, such as a CI runner's, lacks.

- **A picture is shown when its sound is heard, not a screen's delay
  later.** `VideoSynchronizer` handed each picture over when the audio
  position reached it, and the audio position is what the listener hears,
  device latency and all; the renderer then took its draw, the wait for the
  display's next refresh and a compositor's repaint to show it — 22 to 38 ms
  behind the sound, measured with a flash and a beep. A video renderer now
  tells the pipeline's playback clock what showing a picture takes, and the
  synchronizer hands each over that much early; the flash now reaches
  XWayland 10 ms before the beep is heard, the compositor's repaint landing
  it within a few milliseconds of it, with no picture dropped for it.
  `VulkanWindowRenderer` measures it on Wayland, waiting now and then for a
  present to be shown (`VK_KHR_present_wait`, turned on by `VulkanGpu`
  where the device has it), and on X11, where that wait returns once
  XWayland has the picture, estimates two refreshes of the monitor the
  window is on, read from RandR. `D3d11WindowRenderer` and
  `D3d12WindowRenderer` measure it from the swap chain's frame statistics,
  which say when each present was shown without waiting for it — 27 ms on
  a 60 Hz desktop here — and until they have, estimate two refreshes of the
  desktop's compositor. Each logs what it found.

- **`Player` decodes on the GPU.** It decoded every file in software, which
  is what a 4K file cannot afford. Its picture now goes through a
  `VideoDecodeBin` onto the window's own GPU — D3D11VA on Windows, NVDEC on
  Linux with `cuda` and an NVIDIA GPU — and stays there to be drawn, in
  software and uploaded where the GPU does not take the stream, and in
  software into system memory where there is no GPU to decode onto.
  `Player::decoding` says which. Nothing to change in a program using it.
  Where it decodes in software, it does so on every thread, several
  pictures at once: on the one thread FFmpeg opens a decoder with, 4K60
  HEVC fell behind its rate, and with the picture behind, the demuxer could
  read no further and the sound broke up — a hundred gaps in twelve seconds,
  none now.
- **The software compositor draws text.** A text overlay was on the D3D11
  and CUDA compositors only, so a program that followed the README to
  `VideoWindow` and the software path had none. `SwVideoCompositorHandle`
  has `add_text_layer`, taking the same `TextLayer`, and its
  `SwTextLayerHandle` sets, moves, fades, restacks and hides the text as
  the others' do; the text is rasterized on the calling thread into a
  straight-alpha BGRA picture the compositor blends like any layer.
  `ab_glyph`, which rasterizes, is now always a dependency rather than one
  the `cuda` and `d3d11` features bring.

- **`D3d11Upload` and `CudaUpload` take YUV420P.** A software decode
  needed a `SwScaler` to NV12 in front of either. Both now take YUV420P (and
  YUVJ420P, tagged full range) and put it up as NV12, interleaving its
  chroma planes on the way — the same samples, so no scaler is needed.
  `CudaUpload` does this when built for `CudaFrameFormat::Nv12`, and adds
  `CudaUploadError::PlaneTooSmall` for a plane too short for its rows. The
  link check follows: what takes NV12 on the GPU links after an upload of
  YUV420P, and the suggestion it makes for a missing upload says so. A
  compositor input still takes frames on the compositor's device only —
  the upload in front is where the copy shows, in the topology and in the
  stats.

- **`VulkanGpu` and `VulkanWindowRenderer`: video in a window on Linux,
  from system memory or CUDA.** A new `vulkan` feature, and the Linux
  counterpart of `D3d11WindowRenderer`. `open` opens a window of its own on
  a thread of its own and returns the same `WindowOptions` and
  `WindowEvents` as on Windows — keys, resizing, closing, which hides the
  window — as a plain X11 window through libxcb, not `winit`, so it does
  not collide with an application's event loop; on a Wayland desktop it is
  an XWayland window. `for_window` draws into a window the application
  gives it instead, X11 or Wayland, anything with a `raw-window-handle`,
  kept alive by the renderer. It takes NV12, YUV420P and BGRA frames in
  system memory, each drawn by a shader of its own with no conversion in
  front, and NV12 and BGRA CUDA frames too where its `VulkanGpu` was made
  with `for_cuda`, which pairs it with the GPU CUDA decodes on by UUID. A
  CUDA frame is copied device to device into memory the renderer's device
  allocated and CUDA imported — what `render_common`'s `CudaWindowRenderer`
  did for the examples, in the library. Each YUV frame is drawn by its own
  colour description — BT.709, BT.601 or BT.2020, limited or full range,
  and by height where it says nothing — and every frame letterboxed, and
  presented in step with the display. On X11 it follows the window's size
  by itself; on a Wayland window, where a client sets its own size, the
  application passes it on through `WindowSize`. Named for what draws
  rather than for the frames it takes, like a GStreamer sink, because it
  takes more than one memory domain. `cuda_decode_render` shows `open`,
  with NVDEC and no `winit`; `vulkan_window_render` shows `for_window`.

- **`D3d11Gpu` and `D3d11WindowRenderer`: D3D11 with a window of its own.**
  `D3d11Gpu::new()` makes the one device a pipeline's D3D11 elements share,
  with the flags they need, protected for use across threads, and its
  immediate context behind the one `Arc<Mutex<_>>` they all lock — what a
  program had to write for itself with the `windows` crate; `from_device`
  shares a device made elsewhere. `D3d11WindowRenderer` is a renderer that
  brings its window: `open` opens one on a thread of its own, the way a
  GStreamer video sink does, and returns `WindowEvents` — keys, resizing,
  closing — for the application to act on; `for_window` draws into a window
  the application owns, anything with a `raw-window-handle`, such as a
  `winit` window, kept alive by the renderer. Either way it follows the
  window's size on its own and keeps the picture's aspect ratio. It is a
  plain Win32 window, not a `winit` one, so it does not collide with an
  application's event loop. `d3d11_decode_render` uses both and no longer
  needs `render_common` or `winit`.

- **`Player`: a file played in a window with its sound.** The `playbin` of
  this crate: `Player::open(path, options)` builds `FileDemuxer`, a software
  decode of each stream, a `VideoSynchronizer` in front of a `VideoWindow`,
  and an `AudioResampler` in front of the default output
  (`WasapiRenderer`, `PipeWireAudioRenderer`), the picture following the
  samples played. `play`, `pause`, `seek`, `position`, `duration` and
  `window_control` drive it, `next_event` reports the window, the end and
  an element's failure as `PlayerEvent`s — `next_event_timeout` returns
  `None` only when its time is up, and `Stopped` once playback is over — and `respond_to` does what a
  player usually does with a window event — Space, the arrows, F or a double
  click, and `false` for Escape or a close. After a seek, `position` says
  where it went until playback moves on from there. `FileDemuxer::duration`
  is new with it. The `player` example is a whole player in one screen of
  code, for both platforms.

- **`VideoWindow`: a video window on either platform, no `#[cfg]`.** The
  `autovideosink` of this crate: `VideoWindow::open(name, options)` opens
  whichever window renderer the platform has — `D3d11WindowRenderer` on
  Windows (`D3d12WindowRenderer` with only `d3d12`), `VulkanWindowRenderer`
  on Linux — on a GPU of its own, and returns it with its `WindowEvents`;
  `window_control()` changes the window. It takes what they all take,
  system-memory NV12, YUV420P and BGRA, so a software decode goes straight
  in. GPU frames stay with the platform renderers, whose GPU the rest of a
  pipeline can share. `sw_decode_render` is one program for both platforms
  through it.

- **A window renderer's own window can be changed, and reports clicks.**
  `D3d11WindowRenderer`, `D3d12WindowRenderer` and `VulkanWindowRenderer`
  have `window_control()`, taken before the renderer goes into a pipeline:
  a `WindowControl` that sets the window's title and fills the screen with
  it or puts it back — the saved style and placement on Windows, EWMH's
  `_NET_WM_STATE_FULLSCREEN` on X11 — and says `WindowGone` once the window
  is. It is `None` for a window the application gave the renderer, which is
  the application's to change. `WindowEvent` gains `MouseDown` and
  `DoubleClick`, each with its `MouseButton` and where in the picture area
  it was; a double click is the desktop's own on Windows, and two presses
  within half a second on X11, which has none. `d3d11_decode_render` shows
  where playback is in its title, and F or a double click fills the screen.

- **The D3D window renderers take frames in system memory.**
  `D3d11WindowRenderer` and `D3d12WindowRenderer` draw NV12, YUV420P (and
  YUVJ420P) and BGRA from system memory as well as their backend's own
  textures, uploading each frame themselves into textures made for its
  layout and size — as `VulkanWindowRenderer` already did on Linux. A
  software decode, a CPU capture or an application's own frames go straight
  into one; on Windows they needed a `SwScaler` to NV12 and a
  `D3d11Upload`/`D3d12Upload` in front, which was also where the two
  platforms' graphs parted. The system-memory path needs no video support
  of the device, so a software adapter draws it too.

- **`D3d12Gpu` and `D3d12WindowRenderer`: the same for D3D12.** `D3d12Gpu`
  makes the one device a pipeline's D3D12 elements share, and the one queue
  its windows present through; `from_device` shares a device made
  elsewhere. `D3d12WindowRenderer` has `open` and `for_window` as the D3D11
  one does, with the same `WindowOptions` and `WindowEvents`, and draws the
  NV12 frames `D3d12Decoder` and `D3d12Upload` make zero-copy, waiting on
  the GPU for each frame's own fence. `d3d12_upload` uses both and no
  longer needs `render_common` or `winit`.

- **`Pipeline::position`.** Where playback is — the media time the
  playback master has reached, from the audio renderer's played samples or
  the wall clock a `Pacer` or `VideoSynchronizer` keeps — for a progress
  bar. It holds still while paused, moves on from where a seek put it, and
  is `None` when nothing paces the pipeline. A player used to rebuild it
  from the timestamps of frames going past.

- **`D3d11DecoderError::SurfacePoolExhausted`.** Frames held downstream past
  what the fixed D3D11VA surface pool can spare failed decoding with
  FFmpeg's `Invalid data found when processing input`, the pool's own
  complaint reaching only FFmpeg's log. That failure now says how many
  decoded frames were held and how big the pool is, and what to change.
  `D3d11Decoder::new` and `DecodeTarget::D3d11` document that FFmpeg caps
  the pool at 64 surfaces in all, references included.

- **`log::init` records FFmpeg's own messages.** An encoder's closing
  statistics and a codec library's warnings (`[aac @ 0x…] Qavg: …`,
  libopenh264's) went to stderr, with no time, no thread, and nothing
  tying them to the pipeline. While the logger runs they are recorded in
  its file instead, as element `FFmpeg` named after the codec or format
  that said them, at the matching level. FFmpeg's own threshold (`INFO`
  unless changed) still decides what it reports. This sets FFmpeg's
  process-wide log callback, and dropping the `LogGuard` restores FFmpeg's
  own; an application with its own callback should install it after
  `init`. A program that never calls `init` is unaffected.

- **A pipeline can start paused, and so take one frame from anywhere.**
  `pause` before `run` was ignored, and the run started playing. It now
  makes the run start paused: every source stops before producing
  anything, and `run` returns once they all have. A `seek` from there puts
  exactly one sample through every terminal — with `SeekMode::Accurate`,
  the one covering the requested position — and returns once it has
  arrived:

  ```rust
  pipeline.pause();
  pipeline.run()?;
  pipeline.seek(Duration::from_secs(600), SeekMode::Accurate)?; // the sink has the frame at 10:00
  ```

  which is a thumbnail from part way into a file without decoding
  everything before it. `resume` before `run` undoes it.

- **`BusReceiver::recv_timeout`.** Waits a bounded time for the next event,
  for a loop with something else to watch too, and says whether it came
  back empty because nothing was posted in time or because the bus has
  ended — which `try_recv` cannot tell apart.

- **`WebRtcStreamInfo::track_format`.** A received WebRTC track records
  with `muxer.add_stream(name, info.track_format()?)`, instead of
  assembling `TrackFormat::new(info.codec_parameters()?, info.time_base()?)`
  by hand.

- **`MediaBuffer::video` wraps a hand-made frame.** A `Video` buffer carries
  a pooled frame, so a frame made by hand — a still picture, a caption, a
  test pattern — had to go through a pool of zero first, four lines every
  caller copied along with a comment explaining the trick.
  `MediaBuffer::video(frame)` does it. Nothing recycles such a frame; what
  makes frames over and over keeps its own `UnboundObjectPool` as before.

- **A `BusEvent` prints as one line.** `BusEvent` implements `Display`
  as `[name] eos`, `[name] error: ...`, `[name] dropped a buffer (queue
  full)`, `[name] seeked: requested ..., landed ...` and `finished` — the
  lines `BusReceiver::log_events` always printed — so a loop that only
  reports its bus is `println!("{event}")` rather than a `match` with an
  arm per variant. The examples' 34 such `match`es are one line each now.

- **`SubmitError` converts into `Error`.** The one public error left that
  had no conversion, so a renderer that could not be created had to be
  unwrapped or turned into a string. With it, the examples' constructors
  pass their errors through `?` rather than `.expect()` — 86 of the 146
  `.expect()`s are gone; the rest are on an `Option`, a lock, a joined
  thread or another library's error, where `?` has nothing to convert.

- **`BusEvent::Finished`: the pipeline says when it has ended.** Every
  element that completes end-of-stream posts an `Eos` — a `Queue` as well
  as the muxer after it — so the first `Eos` on the bus is only the first
  thing to end, and a pipeline stopped on it cuts off whatever was still
  draining: the second track of a file, the other side of a `Tee`.
  `Finished` is posted once every terminal sink has accepted `Eos`, after
  the last one's own, and again only after a seek sends a new stream
  through. A branch a `Tee` detaches is no longer waited for; one it
  finishes is counted when its trailer is written. A terminal that fails
  its `Eos` does not count as ended — that failure is the `Error` to act
  on. Stop on it instead of counting `Eos` events:

  ```rust
  // before
  if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
      pipeline.stop();
  }

  // after
  if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
      pipeline.stop();
  }
  ```

  `BusEvent` is `#[non_exhaustive]`, so a `match` with a `_` arm keeps
  compiling. The examples that stopped on the first `Eos`, or counted them,
  now stop on `Finished`.

- **A decoded frame says what unit its timestamps are in.**
  `buffer::time_base` reads it and `buffer::set_time_base` sets it. FFmpeg's
  decoders leave `AVFrame::time_base` unset, so every element in this crate
  that makes a frame — decoders, captures, synthetic sources, mixers,
  compositors, the resampler and the rate limiter — sets it, and every one
  that transforms a frame carries it over with the `pts`.

- **`FileDemuxer::best` and `RtspSource::best` find the stream to play in
  one `?`.** Each returns the `StreamInfo` of FFmpeg's own choice for a
  kind — the same one `best_stream` names by index — or `NoStream(kind)`
  where there is none, so the three lines every example opened with are
  one:

  ```rust
  // before
  let video = source
      .best_stream(media::Type::Video)
      .and_then(|index| streams.get(index))
      .ok_or_else(|| Error::Other("no video stream in file".into()))?;

  // after
  let video = source.best(media::Type::Video)?;
  ```

  A stream that may be absent is `source.best(media::Type::Audio).ok()`.
  `best_stream` is unchanged.

- **Every scaler can be asked for a layout instead of a size:
  `SwScaler::to_format`, `D3d11Scaler::to_format`,
  `CudaScaler::to_format`.** The layout is what a refused link asks for —
  "a SwScaler to NV12 first" in front of an upload that takes NV12, "a
  D3d11Scaler with D3d11ScalerFormat::Bgra first" in front of a download —
  and the size the frames happen to be is no part of that answer, so it is
  no longer asked for. The remedies name these constructors.

  ```rust
  // before: a size fetched from somewhere just to say "the same"
  SwScaler::new("to-nv12", Pixel::NV12, width, height, Flags::BILINEAR)
  D3d11Scaler::new("to-bgra", &device, ctx, D3d11ScalerFormat::Bgra, width, height)?
  CudaScaler::with_format("to-nv12", &cuda, width, height, interp, CudaFrameFormat::Nv12)

  // after
  SwScaler::to_format("to-nv12", Pixel::NV12, Flags::BILINEAR)
  D3d11Scaler::to_format("to-bgra", &device, ctx, D3d11ScalerFormat::Bgra)?
  CudaScaler::to_format("to-nv12", &cuda, interp, CudaFrameFormat::Nv12)
  ```

  It also changes what a mid-stream resolution change does. A scaler given
  a size absorbs one by scaling the picture back to it, which is what keeps
  a fixed-geometry encoder downstream working and a broken aspect ratio
  anywhere else; `to_format` passes the new size on, so what is downstream
  has to be able to take one. The size-taking constructors are unchanged
  and still the right ones in front of an encoder, a muxer's stream or a
  model.

- **A refused link says what goes between, and elements can be asked
  directly.** A pipeline's refusal and `LinkCheck` both end with the
  element that makes the crossing where one does — `; convert it: a
  CudaConverter built for CudaFrameFormat::Nv12`, `; upload it: a
  D3d11Upload, which takes NV12 or BGRA — a SwScaler::to_format to one of
  those first…`
  — through `contract::remedy`. `contract::check_elements(&mut producer,
  &consumer)` asks about two elements with no pad index or trait import. The
  crate documentation's first page has a section on connecting elements:
  what is caught before a pipeline runs and what only a frame shows, asking
  before linking, and a table of what goes between which backends and
  layouts.

- **`contract::check_link` answers whether two elements fit before they are
  linked.** It takes a pad's `SrcPad::contract()` and a sink's
  `Sink::input_contract()` and returns a `LinkCheck` — `Fits`,
  `Refused { produced, accepted }`, or `Unknown` where one side says too
  little — by the same rules a pipeline refuses a link with, which now asks
  it too. Trying the link to find out cost the elements: a refused branch
  drops what it was given. `hw_decode_render` asks it to decide whether a
  `CudaConverter` goes between the decode bin and the renderer, where it
  used to compare `output_format()` against what it knew the renderer took.

- **HDR video is brought to SDR on the GPU.** A PQ or HLG stream came out
  washed out: nothing read its transfer. `VideoDecodeBin` now brings one to
  SDR BT.709 BGRA as soon as it is decoded, on `D3d11` and `Cuda` — by the
  new `D3d11ToneMap`, a pixel shader over P010 or NV12, and by
  `CudaConverter`, which now takes P010 for this. Both evaluate one
  definition: BT.2020's matrix, ST 2084's EOTF or HLG's with a 1000-nit
  OOTF, BT.2020 primaries into BT.709's, BT.2390's EETF on the largest
  channel from 1000 nits down to 203, and gamma 2.2. A D3D11 video processor
  would have been the place, and the RTX 3050's offers no conversion from
  PQ or HLG to SDR. A 4K PQ stream decoded this way ran at 463 frames a
  second on D3D11 and 289 on CUDA. The stream's own MaxCLL and mastering
  metadata are not read, and the software path does not tone map.

- **`FileDemuxer::best_stream` and `RtspSource::best_stream`**: the stream of a
  kind FFmpeg judges the one to play (`av_find_best_stream`). The first
  video stream is not always it — cover art or a thumbnail can come first as
  a still picture — and this passes over a stream marked as an attached
  picture and prefers one with more than a single frame.

- **`DecodeThreading`: how many threads a software video decoder has, and
  how they share the work.** `SwDecoder::with_threading` takes one, and
  `VideoDecodeBin::open` an `Option` of one for its software path;
  `SwDecoder::new`, and `None`, set nothing and decode on one thread, as
  before. It holds `threads` (`None` for as many as the machine has) and a
  `DecodeThreadKind`. `Frame` decodes several pictures at once — at 1080p on
  twelve threads, H.264 about five times one thread and HEVC about four —
  and hands each out a picture per thread later. `Slice` works within one
  picture, holds nothing back, and is for a live source: ProRes and VP9,
  which split a picture into slices or tiles, still gain about six and two
  times — and a codec whose every picture stands alone gains nothing from
  `Frame`: at 1080p ProRes decoded no faster and took 335 MB against 93.
  Which suits a stream depends on its codec and size, so the caller says;
  nothing chooses for it.

- **`VideoDecodeBin`: a video stream decoded onto a GPU by whichever path
  can take it.** `VideoDecodeBin::open(name, params, target, threading)` takes a
  stream and a `DecodeTarget` — `D3d11`, `D3d12` or `Cuda`, owning its device, or
  `System` — and is one element holding the line for it: the target's
  own hardware decoder where it has one for the codec and the pictures are
  4:2:0, and otherwise `SwDecoder` → `SwScaler` → the target's upload,
  so the stream reaches the same device either way. A 10-bit stream is
  decoded by the hardware to P010 and brought down to NV12 on the GPU after
  it — by a `D3d11Scaler` on `D3d11`, whose target carries the pipeline's
  shared immediate context for it, and a `CudaScaler` on `Cuda`; `D3d12`
  decodes it in software. At 4K, HEVC 10-bit decoded this way on CUDA ran
  at 277 frames a second with next to no CPU, against 104 on four
  software threads. A BT.2020 stream is put out as BT.709 BGRA instead —
  see the fix below. Nothing maps HDR to SDR. Pictures with alpha
  take the software path and arrive as BGRA with their alpha intact, since
  no hardware decoder keeps it. Should the hardware open for a stream
  and then refuse it at a frame — a profile the GPU lacks — the bin puts the
  software line in its place, re-arms any preroll in progress and feeds it
  again from the last keyframe; downstream sees the same device and layout
  throughout. `path()` and `output_format()` answer at open, and
  `VideoDecodeBinHandle::path` later, with the reason software was chosen
  (`SoftwareReason::Alpha`, `NoHardwareDecoder`, `PixelFormat`,
  `HardwareRefused`, `SystemMemory`). `System` is `SwDecoder` alone, so the
  same code builds a decoder in a build without any GPU backend; on `D3d12`,
  whose frames here are NV12 only, alpha is not kept.

- **10-bit surfaces come down to 8 bits on the GPU.** `D3d11Scaler` takes
  a P010 texture as input — what D3D11VA decodes 10-bit HEVC to — and
  `D3d11ScalerFormat::Nv12` brings it down to NV12, keeping its colour
  tags. `CudaScaler::with_format(name, device, width, height, interp,
  format)` puts out `format` whatever comes in, and takes NVDEC's P010 for
  `CudaFrameFormat::Nv12`; a conversion `scale_cuda` has no kernel for is
  refused as `CudaScalerError::UnsupportedConversion`. `CudaScaler::new`
  still keeps the input's layout, and now lets P010 through as well.

- **`CudaDevice` is `Clone`.** It was one reference-counted FFmpeg device
  context already; a clone is another reference to the same device.

- **`D3d11SharedTextureSource`: another device's textures, as this
  pipeline's own frames** (Windows, `d3d11`). Whoever produces the pictures
  pushes a shared-texture handle per picture, and each one is opened on the
  pipeline's device, copied into a texture of its own, and sent downstream
  as a `Pixel::D3D11` frame:

  ```rust
  let (source, handle) =
      D3d11SharedTextureSource::new("browser", &device, context.clone(), 1920, 1080, 2)?;
  // ...from the producer's own callback, while its handle is still valid:
  handle.push(shared_handle, None)?;
  ```

  It copies because the texture belongs to the producer and a producer
  reuses its textures, while a pipeline — a compositor holding an input's
  last frame, say — has to outlive the call. It only copies: a texture of
  another size, or anything but `DXGI_FORMAT_B8G8R8A8_UNORM`, is refused
  rather than adapted, and the handle is opened afresh each push rather than
  cached by its value, which a producer is free to reuse for a different
  texture. The producer owes a flush before it hands a picture over.

  `push` blocks while the source's queue is full and `try_push` drops the
  picture instead, which is what a producer whose own thread cannot stall
  wants: a pipeline this feeds can be paused, and a paused one consumes
  nothing at all — so a blocking push stops the producer for as long as the
  pause lasts, and for a browser engine that means every one of its pages.

  New: `D3d11SharedTextureHandle`, `D3d11SharedTextureSourceError`,
  `Error::D3d11SharedTextureSourceError` and
  `ElementType::D3d11SharedTextureSource`.

- **`VideoLayer::premultiplied_alpha`**: whether an input's colour already
  has its alpha multiplied into it. Default `false`, which is every source
  that was here before; a browser engine's output is the case this exists
  for, since it composites its page that way. `D3d11VideoCompositor` blends
  such a layer by what it already holds rather than multiplying the alpha in
  a second time — which is what turned a half-transparent picture dark —
  and scales its colour by the layer's opacity alongside its alpha, so
  opacity still works on it. The software and CUDA compositors take the
  field and ignore it for now.

- **`CudaDecoder::supports(codec)`**: whether this FFmpeg build has a decoder
  for the codec that decodes on CUDA — whether `CudaDecoder::new` gets past
  choosing one. It needs no device, so a stream can be sent to `SwDecoder`
  and `CudaUpload` before anything is built. It speaks for FFmpeg, not the
  GPU: a profile NVDEC lacks still fails at the first frame with
  `HwAccelUnavailable`. `D3d11Decoder::supports` and
  `D3d12Decoder::supports` answer the same for D3D11VA and D3D12VA.

- **`ReplayBuffer`: the last stretch of an encode, saved on request.** A
  muxer in shape — `add_stream` per track, `open` for one sink per track —
  that keeps what its tracks are handed instead of writing it, letting go of
  whatever has fallen out of its window:

  ```rust
  let mut replay = ReplayBuffer::create(Duration::from_secs(30));
  let video = replay.add_stream("video", video_params, video_time_base);
  let audio = replay.add_stream("audio", audio_params, audio_time_base);
  let (mut sinks, handle) = replay.open()?;
  // ...wire the sinks behind their encoders; later, from a hotkey:
  let length = handle.save("replay.mp4")?;
  ```

  A saved clip opens on a keyframe of the video track and starts at zero,
  every track moved by that one origin, so the window is let go of a GOP at
  a time: it holds at most its length and at least that less one keyframe
  interval. `ReplayBufferHandle` is cheap to clone and holds the window
  weakly — once the sinks are dropped a save answers
  `ReplayBufferError::Stopped` — and a save blocks for the write, holding
  the tracks up only while it gathers references to what is held. New:
  `Error::ReplayBufferError` and `ElementType::ReplayBuffer`.

- **A compositor layer hands back the frame it will draw.**
  `SwVideoLayerHandle::latest_frame`, `D3d11VideoLayerHandle::latest_frame`
  and `CudaVideoLayerHandle::latest_frame` answer the last frame the input
  was handed — the pooled reference itself — or `None` before the first
  and once the input is removed. It is how a still picture of one input is
  taken: the compositor keeps that frame however long ago it arrived, so it
  is there for a producer that has paused or pushed a single picture,
  where a branch on the producer's own pipeline would wait for a frame
  that may never come.

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

- **Encoders can say what colour their stream holds.** `CudaEncoder` and
  `SwEncoder` gain `with_color`, which is `new` plus a `ColorDescription` —
  matrix, range, primaries, transfer — told to the encoder before it opens,
  which is the only time it reads one. It ends up in the stream's headers
  and the container's, where every player finds it; `new` still says
  nothing. Nothing is converted: it names what the frames already are.

  ```rust
  let encoder = CudaEncoder::with_color(
      "encode", &device, options, ColorDescription::BT709_LIMITED,
  )?;
  ```

  Untagged, a player guesses, and FFmpeg guesses BT.601 at any size: a
  BT.709 recording's (230, 20, 20) decoded as (211, 0, 22).
  `ColorDescription::describe` says the same about a frame.

  `D3d11VideoEncoder` gains the same `with_color`, for NV12 input only:
  given BGRA the encoder converts for itself, by a matrix of its own —
  `h264_nvenc` BT.601, which it tags; `h264_mf` BT.601 at 320x240 and
  BT.709 at 1080p, which it does not — so describing BGRA is refused with
  `D3d11VideoEncoderError::DescribedBgraInput`. Convert with
  `D3d11Scaler` to `D3d11ScalerFormat::Nv12`, which states BT.709 limited,
  and describe that.

  New: `color::ColorDescription`.

### Changed

- **A seek made while playing pauses before it plays on.** Its preroll now
  ends in a `Pause`, and the `Resume` follows it, where it used to play on
  with the `Resume` alone. An element counting control messages sees one
  more `Pause` per seek. It is what makes sure no element is handed data
  before the `Resume` that lets it take any has reached it.

- **An idle or paused `Queue` sleeps until something happens.** Its worker
  woke every 20 ms to see whether it had been dropped, idle or paused
  alike, and a thread waiting to hand it a buffer looked every 5 ms for a
  request on its way — a graph with many queues, paused, kept every one of
  them waking fifty times a second. They are woken now by what they wait
  for: a request, the pipeline's state moving on, the queue being dropped,
  room being made. Only a worker whose downstream is not ready still looks
  again on a timer, for the downstream that becomes ready without saying so
  — a device playing out what it holds.

- **`ChainBuilder::to` takes any sink, boxed or not.** `.to(counter)` rather
  than `.to(Box::new(counter))`, and the same for `build`. A boxed element is
  now the element it holds — `Box<T>` is an `Element`, `Sink` and `Source`
  wherever `T` is — so a sink already boxed, as a muxer hands its sinks
  over, is still one, and `.to(Box::new(x))` compiles as it did.

- **A chroma key multiplies the alpha it is given instead of replacing it.**
  `SwChromaKey`, `D3d11ChromaKey` and `CudaChromaKey` used to write the key's
  coverage straight into alpha, so a key placed after a luma key put back
  everything that one had taken out. An opaque input — every capture and
  decoder in this crate — keys exactly as before, byte for byte.

### Fixed

- **The end of a stream is not lost when a file is paused, sought or
  finished at its last picture.** A `Pacer` or `VideoSynchronizer` waiting
  on that picture's time lets go of it when a request comes and keeps it,
  with whatever follows, for its next call — and with the `Eos` already
  behind it there is none: the thread feeding it has nothing left to hand
  over. What it kept was dropped with it, so the stream never ended and no
  `Finished` came. Once the end of the stream is in, they now hand
  everything on at once. And a queue being dropped waits for the
  pipeline to settle an interrupt before it drains, as it does while
  running, so it does not feed a pacer that is letting go.

- **A `Pacer` puts one sample through a seek, not two.** A seek's request
  can reach a `Pacer` before the interrupt the pipeline raises for it, and
  for that moment the pacer took itself for interrupted: it kept the
  preroll's buffer and handed it on together with the next, two samples
  into a terminal that takes one. A seek made while paused could then show
  a picture past the one it landed on. A preroll waits for nothing, so the
  pacer now lets its buffers straight through whatever the interrupt says.

- **A source tells every one of its pads, whatever one of them answers.**
  A control message went to a source's pads in turn and stopped at the
  first that failed: a file's picture branch refusing a `Pause` left the
  sound branch beside it playing. Every pad is told now, and the first
  failure is still what comes back.

- **A queue drops what a seek has left behind, however late it gets
  there.** A seek empties every queue with a `Flush` and then repositions
  the sources, and that relied on every source being paused and every
  thread's timing going its way. A source of your own that read on while
  paused — handed the `Flush` and the `Seek` as two requests, and pushing
  between them — put media from the old position into a queue the flush
  had just emptied, and it was delivered ahead of the new position's.
  Every buffer now carries the timeline it was read on, from the thread
  that read it across each `Queue`, and a queue drops what is from one a
  seek has left. Nothing in the public API changes, and a buffer made on
  a thread no pipeline numbered — an element's own worker, a caller
  driving elements by hand — is never dropped for it.

- **A seek no longer loses the stream that prerolls first.** After handing
  its branch the seek's one sample, a decoder suppressed what it decoded
  until the preroll ended — and went on taking packets meanwhile, since
  nothing said it was full. With a picture slow to preroll, the sound's
  decoder was fed the whole rest of the file in that time and threw it
  away: a player resumed after a seek with no sound at all. A decoder now
  says it is not ready while its preroll has its sample and runs on, so
  the queue in front of it holds the packets and the demuxer keeps them.
  `VideoDecodeBin` passes the answer on from the decoder inside it.

- **Every control request lets go of a thread blocked handing data on.**
  Only pause, stop, finish and a seek's opening check used to raise the
  interrupt that makes a `Queue` take such a buffer as held over and a
  `Pacer` let go of its wait; the rest were sent on the assumption that
  the graph was already paused and nothing could be blocked. A source that
  was not — as `FileDemuxer` at the end of its file was until 208af56 —
  then left a seek waiting on it for good. Every request raises one now,
  and a paced wait wakes the moment it is raised rather than at the end of
  a polling slice.

- **A `Tee` loses nothing while a seek's other branches catch up.** While
  a seek's preroll runs, a `Tee` stops feeding a branch that has taken its
  sample, so the branches stay level — and it dropped what came for that
  branch meanwhile rather than keeping it. In front of the decoders that
  was packets: a branch lost a third of a second of them, and the pictures
  that depended on them, whenever a sibling was slower to preroll. And a
  seek past the end of a file, whose last picture and `Eos` arrive
  together, left every branch without an end. What arrives for a held
  branch is now kept, in order, and handed on when playback resumes.

- **`Pipeline::finish` of a playing file hands its terminals their `Eos`.**
  Two things lost it. `finish` interrupts the clock, and a `Pacer` or
  `VideoSynchronizer` in a wait gave up and kept its picture — then took
  the interrupt as answered only by a control message of its own, which
  `finish` sends none of, so every wait after it gave up too and the
  `Eos` stayed behind the picture for good. And the queues a finished
  source owns are dropped as it ends: a worker that found its downstream
  busy at that moment — a queue in front of a `Pacer` usually is — ended
  there, dropping what it held and the `Eos` with it. A waiting element
  now lets go only while an interrupt is outstanding, and a dropped queue
  still holding an `Eos` hands everything up to it on first; one holding
  none, as a detached branch's does, is abandoned as before. A muxer at the
  end of a paced branch had never written its trailer on `finish`. Found by
  the new control conformance sequences — see `CONTRIBUTING.md`.

- **A seek the moment a file has ended no longer hangs or plays nothing.**
  A `FileDemuxer` waiting at the end of its file passed a `Pause` on
  without pausing itself, and a pipeline pauses around every seek — so a
  seek from the end read the file on at once, into the paused queue
  behind. Once that queue was full the source was stuck handing it a
  packet and never took the `Preroll` the seek waited on: `Pipeline::seek`
  did not return. Behind a queue that drops on a timeout, as `Player`'s
  do, it returned instead with the start of the file dropped, and nothing
  played — `Player::play` after `Ended` stayed at zero. It showed only
  where the queue filled before the seek's next step, as on a slow CI
  runner. Paused at the end, it now waits like anywhere else.

- **The streams `FileDemuxer::open` and `RtspSource::open` describe no
  longer hold the source open.** Each `StreamInfo`'s parameters shared the
  ownership of the whole input with the source, so a file stayed open —
  and on Windows could not be deleted or replaced — for as long as the
  caller kept the stream list, even after the pipeline had ended; an RTSP
  session stayed connected the same way. That sharing is counted by a
  non-atomic `Rc`, and with the source on its own thread and the list on
  the caller's, the two could drop it at once. They are copies of their
  own now.

- **`SegmentedFileMuxer` cuts the sound where it cuts the picture.** A
  picture reaches a muxer later than its sound — through a queue, and an
  encoder holding frames back — and the cut happened when the keyframe
  arrived, so the sound already there for the next segment ended the one
  before: 0.4 s of it at every cut behind NVENC, playing on past that
  file's last picture and missing from the start of its own. Another track's
  packets now wait for the picture to pass them, and at a cut, what comes
  before the keyframe's time ends the outgoing file and the rest opens the
  next. Sound that arrives before the first picture no longer makes a
  segment of its own either.

- **A `ReplayBuffer` clip ends with its picture.** For the same reason its
  sound ran on 0.46 s past the last frame; what the other tracks hold past
  the picture's end stays out of a clip.

- **`RtspSource` no longer logs the password in a camera's address.** It
  logged the URL it opened as given; it now logs it with the credentials
  removed, as `RtspMuxer` already did.

- **The docs match what the examples do.** `screen_record_software` and
  `screen_record_av` end with `finish()`, which keeps the last frames the
  encoder holds, and their READMEs said `stop()`, which drops them; the
  `rtsp_source` example took a developer's camera address when given
  none, and asks for one now; the README says RTSP output goes to a server
  such as MediaMTX, which this crate does not provide; and the muxers' doc
  examples use `VideoCodec::OpenH264`, which every build has, not
  `VideoCodec::H264`, which many do not.

- **`D3d11Upload` keeps a frame's primaries and transfer.** It passed on
  the matrix and range and dropped the other two, so an uploaded BT.2020
  PQ or HLG picture reached anything reading them after it looking like
  SDR. The other uploads already copied all four.

- **`Player` stays at the end once it gets there.** Its `position` went on
  counting past the file's length after `Ended`, a second a second; it now
  reads the file's length there. `play` once `Ended` has been reported
  starts the file again from the start, as a player's play button does.

- **`VideoEncodeBin` refuses a frame in another layout than
  `EncodeInput::System` said** — `VideoEncodeBinError::FormatMismatch` —
  rather than encoding it under a colour description made for the layout
  it was told.

- **The `fanout` and `tee` examples end at the end of their file.** They
  waited for the bus to close, which a file's pipeline no longer does by
  itself; they stop on `Finished` now.

- **The D3D11 and D3D12 window renderers draw a frame in its own colours.**
  Their NV12 shaders had BT.601 limited range written into them, carried
  over from the examples' presenters, so every HD picture — BT.709, what
  nearly every decoded stream is — was converted with the wrong matrix:
  the R'G'B' (230, 20, 20) a BT.709 frame holds came out as (212, 0, 23).
  `D3d11WindowRenderer` and `D3d12WindowRenderer` now draw their frames
  themselves rather than through a presenter, so they read each frame's
  colour description and convert with its own matrix and range — BT.709,
  BT.601 or BT.2020, limited or full, and for an untagged frame BT.709 over
  576 rows and BT.601 otherwise — as `D3d11VideoCompositor` already did.
  They check frames exactly as `D3d11Renderer` and `D3d12Renderer` do, and
  report the same errors. Those two, and their presenter traits, are
  unchanged: a presenter of one's own is still handed no colour
  description.

- **Pausing leaves a queue's backlog in the queue.** `Pipeline::pause`
  interrupts every paced wait before its request has reached the queues,
  and an interrupted `Pacer` or `VideoSynchronizer` takes whatever it is
  handed without waiting. A queue's worker went on handing it buffers in
  that time, and a source blocked on the full queue could only pass the
  pause on once the worker had made room — so the whole backlog moved
  into the pacer: a 12-frame queue lost 14 frames to it on nine pauses in
  ten. Held there until resume, they were out of their pool while the
  queue filled again, which a fixed D3D11VA or NVDEC pool had to be sized
  twice over for. A queue built by a pipeline now takes nothing but
  control from the interrupt until every source has acknowledged the
  request, and a buffer that finds its channel full meanwhile is held
  over rather than blocking the thread with the request to pass on; the
  pacer holds the one frame it was waiting on. `d3d11_decode_render`'s
  decoder budget is back to its queue and a few more.

- **A file whose picture is muxed ahead of its sound plays with any video
  queue.** `FileDemuxer` waited on a pad whose branch was full, which
  stopped its one read cursor. When the full branch was the picture —
  frames the audio had not reached yet — the audio packets that would let
  it reach them lay further on in the file, so the audio clock stopped and
  the picture waited on it for good: an 8-frame video queue froze a file
  whose picture leads its sound by a second at 0.667 s, with no error. A
  full pad's packets are now held back while another branch is still
  waiting on the cursor, for up to 5 s of file time, and go out in order
  once it has room; past that, or with nothing else waiting, the full pad
  is waited on as before.

- **A pipeline with several sources no longer loses one to a pause, seek
  or finish.** Control went to one source at a time, after the clock had
  been interrupted. A source waiting its turn behind another's cascade was
  woken with nothing to take: its interrupted `Pacer` handed each buffer
  straight back, so it read on unpaced — to the end of its file within
  milliseconds, where its thread ended. Two demuxers on one file, one for
  the picture and one for the sound, failed their seek's preroll on three
  runs in four. Every request is now queued on every source before
  anything is woken, and the sources handle it together rather than in
  turn.

- **`seek` no longer hangs a player that seeks while the video waits on
  the sound.** `seek` first asks every branch whether it can seek, and it
  interrupted the clock only after that. A `VideoSynchronizer` holding a
  frame the audio had not reached yet — the usual state of a file whose
  picture is muxed ahead of its sound — sat inside a `Queue` worker that
  takes control only between buffers; the audio it waited for came from
  the demuxer, which had stopped to ask that very question. None of them
  moved again, with no error and no timeout. A newcomer's player hung this
  way on four seeks in six right after resuming. The clock is interrupted
  before the question now.

- **The recording examples end with `finish`.** `audio_record`, `hls`,
  `rtmp_publish`, `screen_record_software`, `screen_record_av` and the
  compositor examples ended with `stop`, which finalizes a playable file
  but abandons what the encoders still hold — the last frame of video, two
  frames of AAC. The Linux halves of `screen_record_av`, `_software` and
  `_nvenc` also passed encoders a `time_base` they no longer take, and did
  not compile.

- **A pipeline played to its end can be sought through a `Queue`.** A
  queue's worker ended with the `Eos` it forwarded, so once a `FileDemuxer`
  had reached the end of its file, a seek repositioned it and its new
  stream stopped at the first `Queue`: the terminal behind it never saw a
  sample and `Pipeline::seek` failed with `PrerollError::TimedOut` five
  seconds later. The worker now goes back to waiting after `Eos`, and only
  `Stop` or dropping the `Queue` ends it, as before. What changes to see is
  that elements behind a queue now receive the `Stop` that ends a pipeline
  after their `Eos`, as those in a branch without one always did, and are
  released at that `Stop` or when their source is dropped, rather than at
  `Eos`.

- **`RtspSource`'s packets carry their stream's time base.** FFmpeg does not
  promise to fill a demuxed packet's, and `FileDemuxer` already stamped it;
  `RtspSource` did not, so a packet-level element downstream could read a
  `pts` in no unit at all.

- **`D3d11SharedTextureSource` says what a producer really owes it.** Its
  docs said a flush was enough before a push; it is not. A flush submits
  the producer's drawing without waiting for it, and this element copies on
  its own device's queue, which cannot wait on another device's — so under
  a loaded GPU a flushed picture was copied before it existed, as an empty
  frame, in about one push in forty. A producer either guards the texture
  with a keyed mutex released with key 0, which this element already takes
  around its copy, or waits for its drawing to complete before pushing.
  Nothing in the element changed; its own test had the same mistake and
  failed intermittently for it, and now waits, and the keyed-mutex path
  has a test of its own.

- **A 10-bit stream that did not say so no longer leaks P010.** A stream
  described only by its session has no layout in its parameters, so
  `VideoDecodeBin` sent it to the hardware as if it were 8-bit, and a 10-bit
  one reached whatever was downstream as P010, which nothing there reads.
  The first P010 frame now puts the target's converter to NV12 after the
  decoder, on `D3d11` and `Cuda`, and the frames come out NV12 as
  `output_format` said at open.

- **BT.2020 video comes out in the right colours.** `D3d11Scaler` described
  colour to the video processor with a bitfield whose one matrix bit says
  BT.601 or BT.709, and read BT.2020 as BT.601; `CudaConverter` and
  `CudaVideoCompositor` read every NV12 frame as BT.709, BT.601 included. A
  red (200, 40, 40) encoded with BT.2020's matrix came out (190, 32, 40)
  and (204, 48, 38). `D3d11Scaler` now hands the processor a
  `DXGI_COLOR_SPACE_TYPE`, and the CUDA kernel takes the matrix and range of
  the frame it reads, as `D3d11VideoCompositor` already did; a CUDA NV12
  layer not in the canvas's own BT.709 is converted on its way in rather
  than copied. BT.2020's primaries are brought into BT.709's too, which
  a D3D11 video processor does only from P010 to RGB and writes no BT.2020
  Y'CbCr at all — so `VideoDecodeBin` puts a BT.2020 stream out as BT.709
  BGRA on `D3d11` and `Cuda`, straight from the decoder, and every element
  after it reads it right. `output_format` says so at open.

- **A GPU refusing a stream is reported as that.** `D3d11Decoder`,
  `D3d12Decoder` and `CudaDecoder` returned FFmpeg's `EPERM` and then
  `AVERROR_INVALIDDATA` for a stream whose profile the GPU lacks — the same
  errors a damaged stream gives. They now return `HwAccelUnavailable` for
  every packet after the hardware turned the stream down.

- **The CUDA compositor says its canvas is BT.709, limited range** — which is
  what every fill and blend into it converts with. It said nothing, so
  whatever read it later picked its own answer; swscale's was BT.601. The
  D3D11 compositor's BGRA canvas already said `RGB`, full range.
- **`SwScaler` reads a YUV frame by what it says it is.** It used swscale's
  default whatever the frame said, which is BT.601 limited, so a BT.709
  picture scaled to RGB came out with its colours shifted — (230, 20, 20) as
  (211, 0, 22). Going YUV to YUV it keeps the matrix and range it was given
  rather than converting into the default, and every output says what it is.
  A frame that says nothing is read exactly as before.
- **`CudaDecoder` opens a decoder that decodes on CUDA, not FFmpeg's default
  one.** With `libdav1d` built in, FFmpeg picks it for AV1 ahead of its own
  `av1` decoder, and only the latter reaches NVDEC — so AV1 opened, as
  software, and failed at the first frame with `HwAccelUnavailable`. It now
  opens `av1` and decodes on the GPU. A codec with no such decoder at all,
  ProRes say, is refused by `new` instead of opening and failing the same
  way later. `D3d11Decoder` and `D3d12Decoder` opened FFmpeg's default too
  and failed on AV1 the same way; they now choose the same way, for D3D11VA
  and D3D12VA, and every frame of an AV1 stream comes out on the GPU.

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
