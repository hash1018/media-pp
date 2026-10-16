# Recipes

What to use for a common job, and which example does it. Every example is a
crate of its own: `cargo run -p <name> -- <arguments>`; its `README.md` says
what it needs. The full list is in [examples](../examples.md).

- [Play a file](#play-a-file)
- [Read a file's pictures in your own code](#read-a-files-pictures-in-your-own-code)
- [Convert a file](#convert-a-file)
- [Record the screen](#record-the-screen)
- [Capture a camera or sound](#capture-a-camera-or-sound)
- [Stream live](#stream-live)
- [Combine several sources into one picture](#combine-several-sources-into-one-picture)
- [Find, follow and hide objects](#find-follow-and-hide-objects)
- [Keep the last minute, save on demand](#keep-the-last-minute-save-on-demand)
- [Show the output as a webcam](#show-the-output-as-a-webcam)
- [Feed media in or take it out](#feed-media-in-or-take-it-out)
- [Write your own element](#write-your-own-element)

## Play a file

`Player` does all of it — decoding on the GPU where it can, a window, the
sound, seeking, speed and stepping:

```rust,ignore
let player = Player::open("video.mp4", PlayerOptions::default())?;
player.play()?;
```

| Example | Shows |
|---|---|
| [`player`](../../examples/render/player) | `Player` with keyboard control |
| [`hw_decode_render`](../../examples/render/hw_decode_render) | The same built by hand: `VideoDecodeBin -> Pacer -> renderer` |
| [`seek_render`](../../examples/render/seek_render) | Seeking a pipeline you built yourself |

On macOS a window needs the main thread: run your code inside
`media_pp::elements::run_with_windows`, as
[getting started](../getting-started.md#7-playing-a-file) shows.

## Read a file's pictures in your own code

`FileDemuxer -> SwDecoder -> AppSink`, where `AppSink` calls your closure with
each decoded picture.

| Example | Shows |
|---|---|
| [`app_sink`](../../examples/core/app_sink) | Pictures handed to a closure |
| [`probe`](../../examples/core/probe) | Looking at a file's streams before deciding how to wire it |
| [`fanout`](../../examples/core/fanout) | Video and audio each to their own branch |

## Convert a file

| To | Use | Example |
|---|---|---|
| Re-encode, GPU where possible | `VideoDecodeBin -> VideoEncodeBin -> FileMuxer` | [`transcode`](../../examples/core/transcode) |
| Change container only, no re-encoding | `FileDemuxer -> FileMuxer` | [`remux`](../../examples/core/remux) |
| Resize | `SwScaler`, or a GPU scaler | [`scale`](../../examples/vision/scale) |
| Add subtitles from speech | `WhisperTranscriber` | [`transcribe`](../../examples/core/transcribe) |

`FileMuxer` writes MP4, MOV, Matroska and more, by the file's extension; an
MP4 can be fast-start (plays while downloading) or fragmented (still opens
after a crash) — `Mp4Layout`.

To keep only the first part of a file — its first ten seconds, say — there is
no option to set: put a [`Filter`](#write-your-own-element) of your own on each
stream, after its decoder, that hands on each picture or sound until its time
(`buf.pts()` in `buf.time_base()`) reaches the end, drops every one after,
and tells your program the first time it drops one. Once every stream has
told it, `pipeline.finish()` ends the file there. For a part that begins
later, have the same filter drop what comes before its start too, and put a
`TimestampOrigin` between each encoder and the muxer so the file begins at
zero. The file is still read and decoded from its beginning: `Pipeline::seek`
is refused while a recording muxer is attached.

## Record the screen

| Platform | Capture |
|---|---|
| Windows | `DxgiCaptureSource` (a screen), `WgcCaptureSource` (a window) |
| Linux | `PipeWireScreenCaptureSource`, through the desktop's screen-sharing picker |
| macOS | `ScreenCaptureKitSource` (asks for the permission to record the screen) |

| Example | Records | Platform |
|---|---|---|
| [`screen_record_software`](../../examples/render/screen_record_software) | Encoded on the CPU — runs anywhere | All |
| [`screen_record_nvenc`](../../examples/render/screen_record_nvenc) | Encoded on the GPU, the picture never in system memory | All |
| [`screen_record_av`](../../examples/render/screen_record_av) | With the system's sound, in one file | All |
| [`screen_record_overlay`](../../examples/render/screen_record_overlay) | With text drawn over it on the GPU | Linux, macOS |

A capture never ends by itself: end the recording with `pipeline.finish()`.
Long recordings can be cut into files by time or size with
`SegmentedFileMuxer`.

## Capture a camera or sound

| | Windows | Linux | macOS |
|---|---|---|---|
| Camera | `MfCaptureSource` | `V4l2CaptureSource` | `AvFoundationCaptureSource` |
| Sound | `WasapiCaptureSource` | `PipeWireAudioCaptureSource` | `CoreAudioCaptureSource` |
| Playback | `WasapiRenderer` | `PipeWireAudioRenderer` | `CoreAudioRenderer` |

Examples: [`audio_capture`](../../examples/core/audio_capture),
[`audio_playback`](../../examples/core/audio_playback),
[`audio_record`](../../examples/core/audio_record). `AudioMixer` mixes several
sources; `NoiseSuppressor`, `AudioGate`, `AudioCompressor` and
`AudioLimiter` clean up a microphone.

## Stream live

| To | Use | Example |
|---|---|---|
| Twitch, YouTube, any RTMP server | `RtmpMuxer` | [`rtmp_publish`](../../examples/core/rtmp_publish) |
| HLS for browsers | `HlsMuxer` | [`hls`](../../examples/core/hls) |
| An RTSP server such as MediaMTX | `RtspMuxer` | [`rtsp_serve`](../../examples/rtsp/rtsp_serve) |
| Read an RTSP camera | `RtspSource` | [`rtsp_source`](../../examples/rtsp/rtsp_source) |
| WebRTC | `WebRtcPeer`, `WebRtcTrackSink`, `WebRtcTrackSource` | [`webrtc_video_call`](../../examples/webrtc/webrtc_video_call) |

To record and stream at once, split with a `Tee`: each output gets its own
branch, and either can be finished while the other runs —
[`dynamic_tee`](../../examples/core/dynamic_tee). A recording started while
the source is already running gets the source's timestamps, so its file
would begin seconds in; put a `TimestampOrigin` between each track's encoder
and the muxer and it begins at zero.

## Combine several sources into one picture

A compositor takes any number of inputs, each a layer you place, scale,
crop, fade and restack while it runs, plus text layers:

| Backend | Compositor |
|---|---|
| CPU | `SwVideoCompositor` |
| D3D11 / D3D12 | `D3d11VideoCompositor` / `D3d12VideoCompositor` |
| CUDA / Vulkan / Metal | `CudaVideoCompositor` / `VulkanVideoCompositor` / `MetalVideoCompositor` |

Examples: [`video_compositor`](../../examples/core/video_compositor) (CPU, with
a green screen keyed out),
[`gpu_video_compositor`](../../examples/render/gpu_video_compositor) (GPU,
shown and recorded at once), and the text overlays
[`d3d11_text_overlay`](../../examples/render/d3d11_text_overlay) and
[`cuda_text_overlay`](../../examples/cuda/cuda_text_overlay). For an export
where every frame must be exact, use `RenderMode::Offline`.

## Find, follow and hide objects

```text
decoder -> Detector -> ObjectTracker -> DetectionOverlay -> encoder -> FileMuxer
```

| Step | CPU | CUDA | D3D12 | Metal |
|---|---|---|---|---|
| Find objects (ONNX: YOLO, RetinaFace, DETR) | `SwOrtDetector` | `CudaOrtDetector` | `D3d12OrtDetector` | `MetalOrtDetector` |
| Number them across pictures | `ObjectTracker` (any backend) | | | |
| Draw boxes, or blur, pixelate or fill | `SwDetectionOverlay` | `CudaDetectionOverlay` | `D3d12DetectionOverlay` | `MetalDetectionOverlay` |

What was found travels with each picture as `Detections` metadata, so any
element after the detector can read it. `ObjectAnalytics` counts objects
in zones and across lines; a classifier names what was found; an embedder
turns it into a vector for matching the same person again. `StreamMux`
batches pictures of several streams through one detector.

Examples: [`detect`](../../examples/vision/detect) (CPU, boxes drawn),
[`cuda_detect`](../../examples/vision/cuda_detect),
[`cuda_track`](../../examples/vision/cuda_track),
[`cuda_multi_detect`](../../examples/vision/cuda_multi_detect),
[`d3d12_detect`](../../examples/vision/d3d12_detect) (any GPU on Windows,
tracked, boxed or hidden and recorded),
[`metal_detect`](../../examples/vision/metal_detect) and
[`cut_detect`](../../examples/vision/cut_detect) (where each shot of an
edit begins). The models are not included; each example's README says
which to download.

## Keep the last minute, save on demand

`ReplayBuffer` keeps the last stretch of what was encoded in memory and
writes it to a file when asked — the clip you only knew you wanted after it
happened. Feed it from a `Tee` beside your recording or stream.

## Show the output as a webcam

`MfVirtualCamera` on Windows 11 (its DLL registered once, as administrator)
and `V4l2VirtualCamera` on Linux (the v4l2loopback module) show a
pipeline's pictures as a camera that Teams, Zoom or a browser can pick —
[`virtual_camera`](../../examples/core/virtual_camera).

## Feed media in or take it out

- `AppSource` pushes buffers your code makes into a pipeline —
  [`app_source`](../../examples/core/app_source).
- `AppSink` hands every buffer to a closure —
  [`app_sink`](../../examples/core/app_sink).
- `PipelineBridge` carries buffers from one pipeline into another.
- `media_pp::ffmpeg` is the FFmpeg binding media-pp uses; use it rather than
  depending on `ffmpeg-next` yourself, so the versions match.

## Write your own element

Implement `Source`, `Filter` or `Sink` — see
[how media-pp works](concepts.md#writing-your-own-element) and the
[`custom_element`](../../examples/core/custom_element) example. For CUDA or
Metal pictures,
[`cuda_custom_element`](../../lib/tests/cuda_custom_element.rs) and
[`metal_custom_element`](../../lib/tests/metal_custom_element.rs) show how
to work on them without copying.
