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
- **Analysis**: `SwOrtDetector` and `CudaOrtDetector`, filters that put what a
  YOLO model finds on each picture as `Detections` metadata;
  `SwDetectionOverlay` and `CudaDetectionOverlay`, which draw those boxes and
  labels onto the picture; `WhisperTranscriber`, `FrameCounter`,
  `PacketCounter`.
- **Whole pipelines**: `Player`, a file played with its sound in a window.

Elements of your own are a `Source`, a `Filter` or a `Sink`; the crate
documentation's first page says what each is handed, and
[`custom_element`](../examples/core/custom_element) writes one of each.
