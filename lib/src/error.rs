use thiserror::Error;

#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
use crate::elements::DxgiCaptureSourceError;
#[cfg(feature = "ort")]
use crate::elements::OrtDetectorError;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
use crate::elements::PipeWireAudioCaptureSourceError;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
use crate::elements::PipeWireAudioRendererError;
#[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
use crate::elements::PipeWireScreenCaptureSourceError;
use crate::elements::RtspSinkError;
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
use crate::elements::WasapiCaptureSourceError;
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
use crate::elements::WasapiRendererError;
#[cfg(feature = "webrtc")]
use crate::elements::WebRtcError;
#[cfg(feature = "cuda")]
use crate::elements::{
    CudaConverterError, CudaDecoderError, CudaDownloadError, CudaEncoderError, CudaRendererError,
    CudaScalerError, CudaUploadError, CudaVideoCompositorError,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
use crate::elements::{
    D3d11ChromaKeyError, D3d11DecoderError, D3d11DownloadError, D3d11NvencEncoderError,
    D3d11RendererError, D3d11ScalerError, D3d11TextLayerError, D3d11UploadError,
    D3d11VideoCompositorError,
};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
use crate::elements::{
    D3d12DecoderError, D3d12DownloadError, D3d12RendererError, D3d12ScalerError, D3d12UploadError,
};
use crate::{
    elements::{
        AppSourceError, AudioMixerError, AudioResamplerError, AudioVolumeError, FileDemuxError,
        HlsMuxerError, Mp4MuxerError, PacerError, RtspSourceError, SwAudioEncoderError,
        SwChromaKeyError, SwDecoderError, SwEncoderError, SwScalerError, SwVideoCompositorError,
        TestAudioSourceError, TestVideoSourceError, VideoSynchronizerError,
    },
    graph::GraphError,
    log::LogInitError,
    queue::QueueError,
};

/// Crate-wide error. Each element defines its own `{Element}Error` (see
/// [`FileDemuxError`], [`SwDecoderError`], [`QueueError`]) for its own
/// domain-specific failures; this enum just aggregates them so trait
/// methods (`Sink::consume`, `SourceElement::run`, ...) — which have to
/// return one common error type to stay object-safe across arbitrary
/// `Box<dyn Sink>` — can report any of them. `?` chains through
/// automatically: an element's own function returns its own error type,
/// and the moment that gets used with `?` inside a function returning
/// this top-level `Result`, it's converted here via `#[from]`.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    FileDemuxError(#[from] FileDemuxError),

    #[error(transparent)]
    AppSourceError(#[from] AppSourceError),

    #[error(transparent)]
    RtspSourceError(#[from] RtspSourceError),

    #[error(transparent)]
    TestVideoSourceError(#[from] TestVideoSourceError),

    #[error(transparent)]
    TestAudioSourceError(#[from] TestAudioSourceError),

    #[error(transparent)]
    SwDecoderError(#[from] SwDecoderError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaDecoderError(#[from] CudaDecoderError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaRendererError(#[from] CudaRendererError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaUploadError(#[from] CudaUploadError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaDownloadError(#[from] CudaDownloadError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaScalerError(#[from] CudaScalerError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaConverterError(#[from] CudaConverterError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaVideoCompositorError(#[from] CudaVideoCompositorError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaEncoderError(#[from] CudaEncoderError),

    #[error(transparent)]
    SwEncoderError(#[from] SwEncoderError),

    #[error(transparent)]
    PacerError(#[from] PacerError),

    #[error(transparent)]
    VideoSynchronizerError(#[from] VideoSynchronizerError),

    #[error(transparent)]
    SwAudioEncoderError(#[from] SwAudioEncoderError),

    #[error(transparent)]
    AudioResamplerError(#[from] AudioResamplerError),

    #[error(transparent)]
    AudioVolumeError(#[from] AudioVolumeError),

    #[error(transparent)]
    SwScalerError(#[from] SwScalerError),

    #[error(transparent)]
    SwChromaKeyError(#[from] SwChromaKeyError),

    #[error(transparent)]
    QueueError(#[from] QueueError),

    #[error(transparent)]
    GraphError(#[from] GraphError),

    #[error(transparent)]
    LogInitError(#[from] LogInitError),

    #[error(transparent)]
    AudioMixerError(#[from] AudioMixerError),

    #[error(transparent)]
    SwVideoCompositorError(#[from] SwVideoCompositorError),

    #[error(transparent)]
    Mp4MuxerError(#[from] Mp4MuxerError),

    #[error(transparent)]
    HlsMuxerError(#[from] HlsMuxerError),

    #[error(transparent)]
    RtspSinkError(#[from] RtspSinkError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12RendererError(#[from] D3d12RendererError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12DecoderError(#[from] D3d12DecoderError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12UploadError(#[from] D3d12UploadError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12DownloadError(#[from] D3d12DownloadError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12ScalerError(#[from] D3d12ScalerError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11DecoderError(#[from] D3d11DecoderError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11UploadError(#[from] D3d11UploadError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11DownloadError(#[from] D3d11DownloadError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11ScalerError(#[from] D3d11ScalerError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11ChromaKeyError(#[from] D3d11ChromaKeyError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11NvencEncoderError(#[from] D3d11NvencEncoderError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11RendererError(#[from] D3d11RendererError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11VideoCompositorError(#[from] D3d11VideoCompositorError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11TextLayerError(#[from] D3d11TextLayerError),

    #[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
    #[error(transparent)]
    DxgiCaptureSourceError(#[from] DxgiCaptureSourceError),

    #[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
    #[error(transparent)]
    PipeWireAudioCaptureSourceError(#[from] PipeWireAudioCaptureSourceError),

    #[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
    #[error(transparent)]
    PipeWireAudioRendererError(#[from] PipeWireAudioRendererError),

    #[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
    #[error(transparent)]
    PipeWireScreenCaptureSourceError(#[from] PipeWireScreenCaptureSourceError),

    #[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
    #[error(transparent)]
    WasapiCaptureSourceError(#[from] WasapiCaptureSourceError),

    #[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
    #[error(transparent)]
    WasapiRendererError(#[from] WasapiRendererError),

    #[cfg(feature = "ort")]
    #[error(transparent)]
    OrtDetectorError(#[from] OrtDetectorError),

    #[cfg(feature = "webrtc")]
    #[error(transparent)]
    WebRtcError(#[from] WebRtcError),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
