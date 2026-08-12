pub mod driver;
pub mod filter;
pub mod sink;
pub mod source;

#[cfg(feature = "webrtc")]
pub use driver::{
    TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource,
};
pub use filter::{
    AudioCodec, Pacer, Scaler, ScalerError, SwAudioEncoder, SwAudioEncoderError,
    SwAudioEncoderOptions, SwDecoder, SwDecoderError, SwEncoder, SwEncoderError, SwEncoderOptions,
    Tee, TeeBuilder, TeeHandle, VideoCodec,
};
#[cfg(feature = "d3d11-renderer")]
pub use filter::{D3d11Decoder, D3d11Upload, D3d11UploadError, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
pub use filter::{D3d12Upload, D3d12UploadError, D3d12vaDecoder, D3d12vaDecoderError};
pub use sink::{
    AppSink, FrameCounter, HlsMode, HlsMuxer, HlsMuxerError, HlsMuxerStreamSink, HlsOptions,
    HlsSegmentFormat, Mp4Muxer, Mp4MuxerError, Mp4MuxerStreamSink, PacketCounter, SegmentPolicy,
    SegmentedMp4Muxer, SubmitError,
};
#[cfg(feature = "ort")]
pub use sink::{COCO_CLASS_LABELS, Detection, OrtDetector, OrtDetectorError};
#[cfg(feature = "d3d11-renderer")]
pub use sink::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(feature = "d3d12-renderer")]
pub use sink::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
#[cfg(feature = "rtsp-server")]
pub use sink::{PortPolicy, PublishTransport, RtspServer, RtspServerError, ViewerTransport};
pub use source::{
    AppSource, AppSourceError, AppSourceHandle, AudioMixer, AudioMixerError, AudioMixerOptions,
    FileDemuxError, FileDemuxer, MixerHandle, MixerInputSink, RtspOptions, RtspSource,
    RtspSourceError, RtspTransport, StreamInfo, TestAudioOptions, TestAudioSource,
    TestAudioSourceError, TestVideoOptions, TestVideoSource, TestVideoSourceError,
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
