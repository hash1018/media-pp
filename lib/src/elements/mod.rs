pub mod driver;
pub mod filter;
pub mod sink;
pub mod source;

#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-capture", feature = "wasapi-renderer")
))]
pub use crate::platform::windows::wasapi::{WasapiDevice, WasapiDeviceKind};

#[cfg(feature = "webrtc")]
pub use driver::{
    TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource,
};
pub use filter::{
    AudioCodec, AudioFormat, AudioResampler, AudioResamplerError, AudioVolume, AudioVolumeError,
    AudioVolumeHandle, AudioVolumeOptions, Pacer, PacerError, Scaler, ScalerError, SwAudioEncoder,
    SwAudioEncoderError, SwAudioEncoderOptions, SwDecoder, SwDecoderError, SwEncoder,
    SwEncoderError, SwEncoderOptions, Tee, TeeBuilder, TeeHandle, VideoCodec, VideoSynchronizer,
    VideoSynchronizerError,
};
#[cfg(all(target_os = "windows", feature = "d3d11-renderer"))]
pub use filter::{
    D3d11Decoder, D3d11Download, D3d11DownloadError, D3d11Upload, D3d11UploadError,
    D3d11vaDecoderError,
};
#[cfg(all(target_os = "windows", feature = "d3d12-renderer"))]
pub use filter::{D3d12Upload, D3d12UploadError, D3d12vaDecoder, D3d12vaDecoderError};
pub use sink::{
    AppSink, FrameCounter, HlsMode, HlsMuxer, HlsMuxerError, HlsMuxerStreamSink, HlsOptions,
    HlsSegmentFormat, Mp4Muxer, Mp4MuxerError, Mp4MuxerStreamSink, PacketCounter, SegmentPolicy,
    SegmentedMp4Muxer, SubmitError,
};
#[cfg(feature = "ort")]
pub use sink::{COCO_CLASS_LABELS, Detection, OrtDetector, OrtDetectorError};
#[cfg(all(target_os = "windows", feature = "d3d11-renderer"))]
pub use sink::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(all(target_os = "windows", feature = "d3d12-renderer"))]
pub use sink::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
#[cfg(all(target_os = "windows", feature = "rtsp-server"))]
pub use sink::{PortPolicy, PublishTransport, RtspServer, RtspServerError, ViewerTransport};
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
pub use sink::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
pub use source::{
    AppSource, AppSourceError, AppSourceHandle, AudioMixer, AudioMixerError, AudioMixerOptions,
    FileDemuxError, FileDemuxer, MixerHandle, MixerInputSink, RtspOptions, RtspSource,
    RtspSourceError, RtspTransport, StreamInfo, TestAudioOptions, TestAudioSource,
    TestAudioSourceError, TestVideoOptions, TestVideoSource, TestVideoSourceError, TextLayer,
    VideoCompositor, VideoCompositorError, VideoCompositorHandle, VideoCompositorInput,
    VideoCompositorInputSink, VideoCompositorOptions, VideoFit, VideoInputId, VideoLayer,
    VideoLayerHandle, VideoRect,
};
#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
pub use source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(all(target_os = "windows", feature = "d3d11-renderer"))]
pub use source::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
pub use source::{WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError};
