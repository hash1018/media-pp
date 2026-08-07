pub mod driver;
pub mod filter;
pub mod sink;
pub mod source;

#[cfg(feature = "webrtc")]
pub use driver::{
    TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource,
};
#[cfg(feature = "d3d11-renderer")]
pub use filter::{D3d11Decoder, D3d11Upload, D3d11UploadError, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
pub use filter::{D3d12Upload, D3d12UploadError, D3d12vaDecoder, D3d12vaDecoderError};
pub use filter::{
    Pacer, Scaler, ScalerError, SwDecoder, SwDecoderError, SwEncoder, SwEncoderError,
    SwEncoderOptions, Tee, TeeHandle, VideoCodec,
};
pub use sink::{AppSink, FrameCounter, PacketCounter, SubmitError};
#[cfg(feature = "ort")]
pub use sink::{COCO_CLASS_LABELS, Detection, OrtDetector, OrtDetectorError};
#[cfg(feature = "d3d11-renderer")]
pub use sink::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(feature = "d3d12-renderer")]
pub use sink::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
#[cfg(feature = "rtsp-server")]
pub use sink::{PortPolicy, PublishTransport, RtspServer, RtspServerError, ViewerTransport};
pub use source::{
    AppSource, AppSourceError, AppSourceHandle, FileDemuxError, FileDemuxer, RtspOptions,
    RtspSource, RtspSourceError, RtspTransport, StreamInfo, TestVideoOptions, TestVideoSource,
    TestVideoSourceError,
};
#[cfg(feature = "wasapi-capture")]
pub use source::{
    AudioCaptureOptions, AudioCaptureSource, AudioCaptureSourceError, AudioDevice, AudioDeviceKind,
};
#[cfg(feature = "dxgi-capture")]
pub use source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiScreenOptions, DxgiScreenSource,
    DxgiScreenSourceError,
};
