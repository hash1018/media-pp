mod audio_format;
pub mod driver;
pub mod filter;
mod rtsp;
pub mod sink;
pub mod source;
mod video_format;

pub use audio_format::AudioFormat;
pub use rtsp::RtspTransport;
pub use video_format::VideoFormat;

#[cfg(feature = "cuda")]
pub use crate::platform::cuda::{CudaDevice, CudaDeviceError};
#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-capture", feature = "wasapi-renderer")
))]
pub use crate::platform::windows::wasapi::{WasapiDevice, WasapiDeviceKind};

#[cfg(all(
    target_os = "linux",
    any(
        feature = "pipewire-audio-capture",
        feature = "pipewire-audio-renderer"
    )
))]
pub use crate::platform::linux::pipewire::{
    PipeWireAudioDevice, PipeWireAudioDeviceKind, PipeWireDeviceError,
};
#[cfg(feature = "webrtc")]
pub use driver::{
    TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource,
};
pub use filter::{
    AudioCodec, AudioResampler, AudioResamplerError, AudioVolume, AudioVolumeError,
    AudioVolumeHandle, AudioVolumeOptions, Pacer, PacerError, SwAudioEncoder, SwAudioEncoderError,
    SwAudioEncoderOptions, SwDecoder, SwDecoderError, SwEncoder, SwEncoderError, SwEncoderOptions,
    SwScaler, SwScalerError, Tee, TeeBuilder, TeeHandle, VideoCodec, VideoSynchronizer,
    VideoSynchronizerError,
};
#[cfg(feature = "cuda")]
pub use filter::{
    CudaCodec, CudaDecoder, CudaDecoderError, CudaDownload, CudaDownloadError, CudaEncoder,
    CudaEncoderError, CudaEncoderOptions, CudaScaler, CudaScalerError, CudaScalerInterp,
    CudaUpload, CudaUploadError,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use filter::{
    D3d11Decoder, D3d11Download, D3d11DownloadError, D3d11NvencCodec, D3d11NvencEncoder,
    D3d11NvencEncoderError, D3d11NvencEncoderOptions, D3d11NvencInputFormat, D3d11Upload,
    D3d11UploadError, D3d11vaDecoderError,
};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use filter::{D3d12Upload, D3d12UploadError, D3d12vaDecoder, D3d12vaDecoderError};
pub use sink::{
    AppSink, FrameCounter, HlsMode, HlsMuxer, HlsMuxerError, HlsMuxerStreamSink, HlsOptions,
    HlsSegmentFormat, Mp4Muxer, Mp4MuxerError, Mp4MuxerStreamSink, PacketCounter, RtspSink,
    RtspSinkError, SegmentPolicy, SegmentedMp4Muxer, SubmitError,
};
#[cfg(feature = "ort")]
pub use sink::{COCO_CLASS_LABELS, Detection, OrtDetector, OrtDetectorError};
#[cfg(feature = "cuda")]
pub use sink::{CudaFrameRenderer, CudaRenderer, CudaRendererError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use sink::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use sink::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
pub use sink::{PipeWireAudioRenderer, PipeWireAudioRendererError, PipeWireAudioRendererOptions};
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
pub use sink::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
pub use source::{
    AppSource, AppSourceError, AppSourceHandle, AudioMixer, AudioMixerError, AudioMixerOptions,
    FileDemuxError, FileDemuxer, MixerHandle, MixerInputSink, RtspOptions, RtspSource,
    RtspSourceError, StreamInfo, SwVideoCompositor, SwVideoCompositorError,
    SwVideoCompositorHandle, SwVideoCompositorInput, SwVideoCompositorInputSink,
    SwVideoLayerHandle, TestAudioOptions, TestAudioSource, TestAudioSourceError, TestVideoOptions,
    TestVideoSource, TestVideoSourceError, TextLayer, VideoCompositorOptions, VideoFit,
    VideoInputId, VideoLayer, VideoRect,
};
#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
pub use source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
pub use source::{
    CaptureSourceKind, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
    PipeWireScreenCaptureSourceError,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use source::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
#[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
pub use source::{
    PipeWireAudioCaptureOptions, PipeWireAudioCaptureSource, PipeWireAudioCaptureSourceError,
};
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
pub use source::{WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError};
