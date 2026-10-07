# What is in it

Every element of the library, by what it does and, for video, by the backend
its frames live on. Backend-specific types carry their backend's prefix and
exist only where their feature is enabled — see [features.md](features.md).
Each type's own documentation says what it accepts, owns and how it fails.

## Video, by backend

| | Software | D3D11 | D3D12 | CUDA | Vulkan | VideoToolbox, Metal |
|---|---|---|---|---|---|---|
| Decode | `SwDecoder` | `D3d11Decoder` | `D3d12Decoder` | `CudaDecoder` | `VulkanDecoder` | `VideoToolboxDecoder` |
| Encode | `SwEncoder` | `D3d11VideoEncoder` | | `CudaEncoder` | `VulkanEncoder` | `VideoToolboxEncoder` |
| Scale, convert | `SwScaler` | `D3d11Scaler` | `D3d12Scaler` | `CudaScaler`, `CudaConverter` | `VulkanScaler`, `VulkanConverter` | `MetalScaler`, `MetalConverter` |
| HDR to SDR | | `D3d11ToneMap` | | `CudaConverter` | | |
| Composite | `SwVideoCompositor` | `D3d11VideoCompositor` | | `CudaVideoCompositor` | `VulkanVideoCompositor` | `MetalVideoCompositor` |
| Key, colour | `SwChromaKey`, `SwVideoEffect` | `D3d11ChromaKey`, `D3d11VideoEffect` | | `CudaChromaKey`, `CudaVideoEffect` | `VulkanChromaKey`, `VulkanVideoEffect` | `MetalChromaKey`, `MetalVideoEffect` |
| Upload, download | | `D3d11Upload`, `D3d11Download` | `D3d12Upload`, `D3d12Download` | `CudaUpload`, `CudaDownload` | `VulkanUpload`, `VulkanDownload` | `VideoToolboxUpload`, `VideoToolboxDownload` |
| Detect | `SwOrtDetector` | | | `CudaOrtDetector` | | `MetalOrtDetector` |
| Classify what was found | `SwOrtClassifier` | | | `CudaOrtClassifier` | | `MetalOrtClassifier` |
| Draw or hide detections | `SwDetectionOverlay` | | | `CudaDetectionOverlay` | | `MetalDetectionOverlay` |
| Render | `VideoWindow` | `D3d11WindowRenderer`, `D3d11Renderer` | `D3d12WindowRenderer`, `D3d12Renderer` | `CudaRenderer` | `VulkanWindowRenderer` (Linux) | `MetalWindowRenderer`, `MetalRenderer` |

- `VideoDecodeBin` and `VideoEncodeBin` pick among these for a stream: the
  hardware where it opens, software where it does not, onto or off the
  device either way.
- Every compositor has text layers, runs live by default, and renders an
  export frame-exact with `RenderMode::Offline`; `VideoCompositorControl`
  lets one piece of code build a composition on any of them.
- `VideoWindow` is a window of its own on whatever renderer the platform
  has, taking frames in system memory, so a software decode goes into it
  with no `#[cfg]`.

## Everything else

- **Inputs**: `FileDemuxer`, `RtspSource`, `WebRtcTrackSource`, `AppSource`,
  `PipelineBridge`, `TestVideoSource`, `TestAudioSource`.
- **Capture**:
  - Windows — `DxgiCaptureSource` (screen), `WgcCaptureSource` (window),
    `MfCaptureSource` (camera), `WasapiCaptureSource` (audio),
    `D3d11SharedTextureSource`.
  - Linux — `PipeWireScreenCaptureSource`, `PipeWireAudioCaptureSource`,
    `V4l2CaptureSource`.
  - macOS — `ScreenCaptureKitSource` (screen or window),
    `AvFoundationCaptureSource` (camera), `CoreAudioCaptureSource` (audio),
    `MetalSharedTextureSource`.
- **Audio**: `AudioMixer`, `AudioResampler`, `AudioVolume`, `AudioGate`,
  `AudioCompressor`, `AudioLimiter`, `NoiseSuppressor`, `AudioTempo`,
  `AudioWaveform`, `SwAudioEncoder`; playback through `WasapiRenderer`,
  `PipeWireAudioRenderer` and `CoreAudioRenderer`.
- **Outputs**: `FileMuxer`, `SegmentedFileMuxer`, `ReplayBuffer`, `HlsMuxer`,
  `RtmpMuxer`, `RtspMuxer`, `WebRtcTrackSink`, `AppSink`, and virtual
  cameras: `MfVirtualCamera` on Windows, `V4l2VirtualCamera` on Linux.
- **Flow and timing**: `Queue`, `Tee`, `Rack`, `Pacer`, `VideoSynchronizer`,
  `ChangeGate`, `FrameRateLimiter`, `PauseGate`, `TimestampOrigin`.
- **Analysis**: `SwOrtDetector`, `CudaOrtDetector` and `MetalOrtDetector`,
  filters that put what a YOLO model — or a model of the application's own,
  read by its `DetectorDecoder` — finds on each picture as `Detections`
  metadata; `SwOrtClassifier`, `CudaOrtClassifier` and
  `MetalOrtClassifier`, which name each object found with a second model; `ObjectTracker`, which numbers each
  object across pictures and fills in the pictures a detector let by;
  `ObjectAnalytics`, which counts the objects in zones of the picture and
  across lines; `SwCutDetector` and `CudaCutDetector`, which mark the first
  picture of each shot of an edited video with a `SceneCut`, where a
  tracker starts over; `StreamMux`, which gathers a picture of each of several
  streams into a batch for a model to take at once, and the demux its
  handle makes, which splits the streams out again; `SwDetectionOverlay`,
  `CudaDetectionOverlay` and `MetalDetectionOverlay`, which draw those boxes
  and labels onto the picture, and an `ObjectAnalytics`'s zones and lines
  with their counts, each as `OverlayParts` says; and hide what was found —
  faces, number plates — by a mosaic, a blur or a fill, of the box or the
  ellipse inside it, each class as its `ClassRule` says (an ellipse not yet
  on Metal); `WhisperTranscriber`, `FrameCounter`,
  `PacketCounter`.
- **Whole pipelines**: `Player`, a file played with its sound in a window.

A 10-bit file — an iPhone's HDR — stays ten bits on CUDA: `CudaDecoder`
hands on P010, the detectors and classifiers look at an SDR copy of each
picture, the overlays hide in its ten bits (drawing nothing), and
`CudaEncoder` encodes HEVC Main 10 in the file's own colours
(`with_color(stream.color())`), which `FileMuxer` writes as `hvc1`.

A phone's portrait recording is stored on its side, with a display matrix
saying to turn it. Every decoder hands that on with each picture
(`Orientation::of`), and the demuxer says it of the stream
(`StreamInfo::orientation`). The pictures stay as they are stored, and so
does everything said about them — a box, a track, a zone, a mosaic. The
detectors and classifiers fit each picture into the model the right way up,
the detection overlays lay labels out as the picture is shown, and
`ObjectAnalytics` stands an object on the bottom of its box as shown. A
recording re-encoded from such a file keeps the matrix only when told, with
`TrackFormat::with_orientation(stream.orientation()?)`; a stream copied
through keeps it as it is. The renderers and `Player` do not turn pictures
yet.

Elements of your own are a `Source`, a `Filter` or a `Sink`; the crate
documentation's first page says what each is handed, and
[`custom_element`](../examples/core/custom_element) writes one of each.

One of your own can work on CUDA pictures with CUDA calls of its own and
sit between the CUDA elements here, the pictures never leaving the GPU:
`CudaSurfaceView` says where a picture's planes are in device memory, a
`CudaFramePool` gives pictures to write into that the elements after it take
as from their own device, and `CudaDevice::ordinal` names the GPU whose
primary context they all live in. `CudaSurfaceView`'s documentation gives the
contract — a picture handed to you is ready for the default stream, finish
yours before handing it on, never write one you were handed — and
[`cuda_custom_element`](../lib/tests/cuda_custom_element.rs) is one, between
`CudaUpload` and `CudaDetectionOverlay`.

On macOS the same is done with Metal: `MetalSurfaceView` makes a VideoToolbox
picture's planes into textures over its pixel buffer, on any Metal device
since a pixel buffer belongs to none, and a `VideoToolboxFramePool` gives
pictures to write into. Every element here that writes a picture waits for
the GPU before handing it on; do the same before handing on yours, and never
write one you were handed.
[`metal_custom_element`](../lib/tests/metal_custom_element.rs) is one, between
`VideoToolboxUpload` and `MetalDetectionOverlay`.
