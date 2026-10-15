# Elements

Every element of the library, by what it does and, for video, by the backend
its pictures live on. Backend-specific types carry their backend's prefix and
exist only where their feature is enabled — see [features](features.md).
Each type's own documentation says what it accepts, what it owns and how it
fails.

- [Video, by backend](#video-by-backend)
- [Sources and capture](#sources-and-capture)
- [Sound](#sound)
- [Outputs](#outputs)
- [Flow and timing](#flow-and-timing)
- [Detection and analysis](#detection-and-analysis)
- [Whole pipelines](#whole-pipelines)
- [Ten-bit and HDR video](#ten-bit-and-hdr-video)
- [Rotated video](#rotated-video)
- [Elements of your own](#elements-of-your-own)

## Video, by backend

| | Software | D3D11 | D3D12 | CUDA | Vulkan | VideoToolbox, Metal |
|---|---|---|---|---|---|---|
| Decode | `SwDecoder` | `D3d11Decoder` | `D3d12Decoder` | `CudaDecoder` | `VulkanDecoder` | `VideoToolboxDecoder` |
| Encode | `SwEncoder` | `D3d11VideoEncoder` | `D3d12VideoEncoder` | `CudaEncoder` | `VulkanEncoder` | `VideoToolboxEncoder` |
| Scale, convert | `SwScaler` | `D3d11Scaler` | `D3d12Scaler` | `CudaScaler`, `CudaConverter` | `VulkanScaler`, `VulkanConverter` | `MetalScaler`, `MetalConverter` |
| HDR to SDR | | `D3d11ToneMap` | `D3d12ToneMap` | `CudaConverter` | | `MetalConverter` |
| Composite | `SwVideoCompositor` | `D3d11VideoCompositor` | `D3d12VideoCompositor` | `CudaVideoCompositor` | `VulkanVideoCompositor` | `MetalVideoCompositor` |
| Key, colour | `SwChromaKey`, `SwVideoEffect` | `D3d11ChromaKey`, `D3d11VideoEffect` | | `CudaChromaKey`, `CudaVideoEffect` | `VulkanChromaKey`, `VulkanVideoEffect` | `MetalChromaKey`, `MetalVideoEffect` |
| Upload, download | | `D3d11Upload`, `D3d11Download` | `D3d12Upload`, `D3d12Download` | `CudaUpload`, `CudaDownload` | `VulkanUpload`, `VulkanDownload` | `VideoToolboxUpload`, `VideoToolboxDownload` |
| Detect | `SwOrtDetector` | | `D3d12OrtDetector` | `CudaOrtDetector` | | `MetalOrtDetector` |
| Classify what was found | `SwOrtClassifier` | | `D3d12OrtClassifier` | `CudaOrtClassifier` | | `MetalOrtClassifier` |
| Embed what was found | `SwOrtEmbedder` | | `D3d12OrtEmbedder` | `CudaOrtEmbedder` | | `MetalOrtEmbedder` |
| Draw or hide detections | `SwDetectionOverlay` | | `D3d12DetectionOverlay` | `CudaDetectionOverlay` | | `MetalDetectionOverlay` |
| Find scene cuts | `SwCutDetector` | | `D3d12CutDetector` | `CudaCutDetector` | | `MetalCutDetector` |
| Render | `VideoWindow` | `D3d11WindowRenderer`, `D3d11Renderer` | `D3d12WindowRenderer`, `D3d12Renderer` | `CudaRenderer` | `VulkanWindowRenderer` (Linux) | `MetalWindowRenderer`, `MetalRenderer` |

- `VideoDecodeBin` and `VideoEncodeBin` pick among these for a stream: the
  hardware where it opens, software where it does not, onto or off the
  device either way.
- Every compositor has text layers, runs live by default, and renders an
  export frame-exact with `RenderMode::Offline`; `VideoCompositorControl`
  lets one piece of code build a composition on any of them.
- `VideoWindow` is a window of its own on whatever renderer the platform
  has, taking pictures in system memory, so a software decode goes into it
  with no `#[cfg]`.
- A `…WindowRenderer` opens and draws into a window; a `…Renderer` hands
  each picture to your code to draw yourself.

## Sources and capture

| | |
|---|---|
| Files and the network | `FileDemuxer`, `RtspSource`, `WebRtcTrackSource` |
| From your code | `AppSource`; `PipelineBridge`, from another pipeline |
| Synthetic | `TestVideoSource`, `TestAudioSource` |

| | Windows | Linux | macOS |
|---|---|---|---|
| Screen | `DxgiCaptureSource` | `PipeWireScreenCaptureSource` | `ScreenCaptureKitSource` |
| Window | `WgcCaptureSource` | `PipeWireScreenCaptureSource` | `ScreenCaptureKitSource` |
| Camera | `MfCaptureSource` | `V4l2CaptureSource` | `AvFoundationCaptureSource` |
| Sound — microphone, system, one application | `WasapiCaptureSource` | `PipeWireAudioCaptureSource` | `CoreAudioCaptureSource` |
| Another process's GPU texture | `D3d11SharedTextureSource` | | `MetalSharedTextureSource` |

## Sound

| | |
|---|---|
| Mix | `AudioMixer`: any number of inputs, added and removed while it runs |
| Convert, adjust | `AudioResampler`, `AudioVolume`, `AudioTempo` (speed at the same pitch) |
| Clean up a microphone | `AudioGate`, `NoiseSuppressor`, `AudioCompressor`, `AudioLimiter` |
| Look at | `AudioWaveform`; `WhisperTranscriber`, speech to timed text |
| Encode | `SwAudioEncoder` |
| Play | `WasapiRenderer` (Windows), `PipeWireAudioRenderer` (Linux), `CoreAudioRenderer` (macOS) |

## Outputs

| | |
|---|---|
| Files | `FileMuxer` — MP4, MOV, Matroska and more by the extension; an MP4 or MOV laid out fast-start, for playing as it downloads, or in fragments, which a recording cut off still opens from (`Mp4Layout`) |
| Files cut by time or size | `SegmentedFileMuxer`, each cut on a keyframe |
| The last stretch, saved on demand | `ReplayBuffer` |
| Streaming | `HlsMuxer`, `RtmpMuxer`, `RtspMuxer`, `WebRtcTrackSink` with `WebRtcPeer` |
| A camera other applications open | `MfVirtualCamera` (Windows 11), `V4l2VirtualCamera` (Linux) |
| Your code | `AppSink` |
| Counting | `FrameCounter`, `PacketCounter` |

## Flow and timing

| | |
|---|---|
| `Queue` | A thread boundary, a bounded number of buffers deep |
| `Tee` | One stream to several branches, added and removed while it runs |
| `Rack` | A stretch of branch whose filters can be replaced while it runs |
| `Pacer` | Hands each picture on when it is due |
| `VideoSynchronizer` | The same, following the sound being played |
| `FrameRateLimiter` | Lets at most so many pictures a second through |
| `ChangeGate` | Lets a picture through only when it differs from the last, and at most so often — an idle scene costs nothing after it |
| `PauseGate` | Stops a branch taking pictures or sound, and takes the paused span out of its timeline — a recording paused |
| `TimestampOrigin` | Starts a packet stream's timeline at zero — a recording branch added to a running `Tee` |

## Detection and analysis

What is found travels with each picture as `Detections` metadata, so any
element after the detector can read it.

| | |
|---|---|
| **Detectors** — `SwOrtDetector`, `CudaOrtDetector`, `D3d12OrtDetector`, `MetalOrtDetector` | Find objects with an ONNX model: YOLO (v8, 11, v10, 26), RetinaFace, whose faces carry their eyes, nose and mouth as landmarks, a DETR-family model (D-FINE, RT-DETR, RF-DETR: `DetectorModel::Detr`), or a model of your own read by its `DetectorDecoder`. Small objects can be looked for in overlapping tiles (`Tiles`), those worth looking in picked picture by picture by a `TileChooser` |
| **Classifiers** — `SwOrtClassifier`, `CudaOrtClassifier`, `D3d12OrtClassifier`, `MetalOrtClassifier` | Name each object found with a second model |
| **Embedders** — `SwOrtEmbedder`, `CudaOrtEmbedder`, `D3d12OrtEmbedder`, `MetalOrtEmbedder` | Make each object found a vector that lies near another's where the two look alike — a face straightened by its five points, or a whole box — for finding the same person again; with `OrtEmbedderOptions::cutouts`, the picture it was made from beside it (`Embedding::cutout`) |
| `ObjectTracker` | Numbers each object across pictures and fills in the pictures a detector let by, following by look and saying on each box how sure the look was (`Detection::look`) |
| `ObjectAnalytics` | Counts the objects in zones of the picture and across lines |
| **Cut detectors** — `SwCutDetector`, `CudaCutDetector`, `D3d12CutDetector`, `MetalCutDetector` | Mark the first picture of each shot of an edited video with a `SceneCut`, where a tracker starts over |
| `StreamMux` | Gathers a picture of each of several streams into one batch for a model; the demux its handle makes splits them out again |
| **Overlays** — `SwDetectionOverlay`, `CudaDetectionOverlay`, `D3d12DetectionOverlay`, `MetalDetectionOverlay` | Draw boxes and labels, and an `ObjectAnalytics`'s zones and lines with their counts, as `OverlayParts` says; or hide what was found — faces, number plates — with a mosaic, a blur or a fill, of the box or the ellipse inside it, hard-edged or faded, each class as its `ClassRule` says |

## Whole pipelines

`Player` plays a file with its sound in a window: decoding on the GPU where
it can, seeking, speed, stepping and playing backwards.

## Ten-bit and HDR video

A 10-bit file — an iPhone's HDR — stays ten bits on the GPU:

- **CUDA**: `CudaDecoder` hands on P010, the detectors and classifiers look
  at an SDR copy of each picture, the overlays hide in its ten bits (drawing
  nothing), and `CudaEncoder` encodes HEVC Main 10 in the file's own colours
  (`CudaEncoder::for_stream`, as `SwEncoder` and `VideoToolboxEncoder`
  have), keeping an HDR10 file's mastering display and light levels, which
  `FileMuxer` writes as `hvc1`.
- **macOS**: `VideoToolboxDecoder` hands on P010
  (`VideoToolboxFrameFormat::P010`), the Metal detectors, classifier and
  embedder look at an SDR copy, `MetalDetectionOverlay` hides in the ten
  bits, and `VideoToolboxEncoder` encodes HEVC Main 10.
- **D3D12**: the DirectML detectors, classifier and embedder look at an SDR
  copy, and `D3d12DetectionOverlay` hides in the ten bits.

To show or encode it as SDR instead, the tone maps in the backend table turn
BT.2020, PQ and HLG into BT.709.

## Rotated video

A phone's portrait recording is stored on its side, with a display matrix
saying to turn it. Every decoder hands that on with each picture
(`Orientation::of`), and the demuxer says it of the stream
(`StreamInfo::orientation`).

- The pictures stay as they are stored, and so does everything said about
  them — a box, a track, a zone, a mosaic.
- The detectors and classifiers fit each picture into the model the right
  way up, the detection overlays lay labels out as the picture is shown, and
  `ObjectAnalytics` stands an object on the bottom of its box as shown.
- The window renderers — and so `VideoWindow` and `Player` — show each
  picture upright. The renderers that hand a picture to your code
  (`D3d11Renderer`, `CudaRenderer`, …) hand it over as stored.
- An element that makes a picture anew (a scaler, an upload, a tone map)
  carries the matrix on. A recording re-encoded from such a file keeps it
  only when told, with `TrackFormat::with_orientation(stream.orientation()?)`;
  a stream copied through keeps it as it is.

## Elements of your own

An element of your own is a `Source`, a `Filter` or a `Sink` — its work
alone: the framework keeps its name (its type's, or the one `named` gives
it) and hands it its log identity and the pipeline through `attached`. The
crate documentation's first page says what each is handed, and
[`custom_element`](../examples/core/custom_element) writes one of each.

**On CUDA pictures**, one of your own can work with CUDA calls of its own
and sit between the CUDA elements here, the pictures never leaving the GPU:
`CudaSurfaceView` says where a picture's planes are in device memory, a
`CudaFramePool` gives pictures to write into that the elements after it take
as from their own device, and `CudaDevice::ordinal` names the GPU whose
primary context they all live in. `CudaSurfaceView`'s documentation gives the
contract — a picture handed to you is ready for the default stream, finish
yours before handing it on, never write one you were handed — and
[`cuda_custom_element`](../lib/tests/cuda_custom_element.rs) is one, between
`CudaUpload` and `CudaDetectionOverlay`.

**On macOS** the same is done with Metal: `MetalSurfaceView` makes a
VideoToolbox picture's planes into textures over its pixel buffer, on any
Metal device since a pixel buffer belongs to none, and a
`VideoToolboxFramePool` gives pictures to write into, whose pixel buffers
its `describe` makes say their colour, where VideoToolbox reads it. Every
element here that writes a picture waits for the GPU before handing it on;
do the same before handing on yours, and never write one you were handed.
[`metal_custom_element`](../lib/tests/metal_custom_element.rs) is one,
between `VideoToolboxUpload` and `MetalDetectionOverlay`.
