use thiserror::Error;

#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
use crate::elements::DxgiCaptureSourceError;
#[cfg(feature = "ort")]
use crate::elements::OrtDetectorError;
use crate::elements::RtspSinkError;
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
use crate::elements::WasapiCaptureSourceError;
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
use crate::elements::WasapiRendererError;
#[cfg(feature = "webrtc")]
use crate::elements::WebRtcError;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
use crate::elements::{
    D3d11DownloadError, D3d11RendererError, D3d11TextLayerError, D3d11UploadError,
    D3d11VideoCompositorError, D3d11vaDecoderError,
};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
use crate::elements::{D3d12RendererError, D3d12UploadError, D3d12vaDecoderError};
use crate::{
    elements::{
        AppSourceError, AudioMixerError, AudioResamplerError, AudioVolumeError, FileDemuxError,
        HlsMuxerError, Mp4MuxerError, PacerError, RtspSourceError, ScalerError,
        SwAudioEncoderError, SwDecoderError, SwEncoderError, TestAudioSourceError,
        TestVideoSourceError, VideoCompositorError, VideoSynchronizerError,
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
    ScalerError(#[from] ScalerError),

    #[error(transparent)]
    QueueError(#[from] QueueError),

    #[error(transparent)]
    GraphError(#[from] GraphError),

    #[error(transparent)]
    LogInitError(#[from] LogInitError),

    #[error(transparent)]
    AudioMixerError(#[from] AudioMixerError),

    #[error(transparent)]
    VideoCompositorError(#[from] VideoCompositorError),

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
    D3d12vaDecoderError(#[from] D3d12vaDecoderError),

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12UploadError(#[from] D3d12UploadError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11vaDecoderError(#[from] D3d11vaDecoderError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11UploadError(#[from] D3d11UploadError),

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11DownloadError(#[from] D3d11DownloadError),

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
