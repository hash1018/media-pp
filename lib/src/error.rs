use thiserror::Error;

#[cfg(feature = "dxgi-capture")]
use crate::elements::DxgiCaptureSourceError;
#[cfg(feature = "ort")]
use crate::elements::OrtDetectorError;
#[cfg(feature = "rtsp-server")]
use crate::elements::RtspServerError;
#[cfg(feature = "wasapi-capture")]
use crate::elements::WasapiCaptureSourceError;
#[cfg(feature = "wasapi-renderer")]
use crate::elements::WasapiRendererError;
#[cfg(feature = "webrtc")]
use crate::elements::WebRtcError;
#[cfg(feature = "d3d11-renderer")]
use crate::elements::{D3d11RendererError, D3d11UploadError, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
use crate::elements::{D3d12RendererError, D3d12UploadError, D3d12vaDecoderError};
use crate::{
    elements::{
        AppSourceError, AudioMixerError, AudioResamplerError, FileDemuxError, HlsMuxerError,
        Mp4MuxerError, RtspSourceError, ScalerError, SwAudioEncoderError, SwDecoderError,
        SwEncoderError, TestAudioSourceError, TestVideoSourceError,
    },
    graph::GraphError,
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
    SwAudioEncoderError(#[from] SwAudioEncoderError),

    #[error(transparent)]
    AudioResamplerError(#[from] AudioResamplerError),

    #[error(transparent)]
    ScalerError(#[from] ScalerError),

    #[error(transparent)]
    QueueError(#[from] QueueError),

    #[error(transparent)]
    GraphError(#[from] GraphError),

    #[error(transparent)]
    AudioMixerError(#[from] AudioMixerError),

    #[error(transparent)]
    Mp4MuxerError(#[from] Mp4MuxerError),

    #[error(transparent)]
    HlsMuxerError(#[from] HlsMuxerError),

    #[cfg(feature = "rtsp-server")]
    #[error(transparent)]
    RtspServerError(#[from] RtspServerError),

    #[cfg(feature = "d3d12-renderer")]
    #[error(transparent)]
    D3d12RendererError(#[from] D3d12RendererError),

    #[cfg(feature = "d3d12-renderer")]
    #[error(transparent)]
    D3d12vaDecoderError(#[from] D3d12vaDecoderError),

    #[cfg(feature = "d3d12-renderer")]
    #[error(transparent)]
    D3d12UploadError(#[from] D3d12UploadError),

    #[cfg(feature = "d3d11-renderer")]
    #[error(transparent)]
    D3d11vaDecoderError(#[from] D3d11vaDecoderError),

    #[cfg(feature = "d3d11-renderer")]
    #[error(transparent)]
    D3d11UploadError(#[from] D3d11UploadError),

    #[cfg(feature = "d3d11-renderer")]
    #[error(transparent)]
    D3d11RendererError(#[from] D3d11RendererError),

    #[cfg(feature = "dxgi-capture")]
    #[error(transparent)]
    DxgiCaptureSourceError(#[from] DxgiCaptureSourceError),

    #[cfg(feature = "wasapi-capture")]
    #[error(transparent)]
    WasapiCaptureSourceError(#[from] WasapiCaptureSourceError),

    #[cfg(feature = "wasapi-renderer")]
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
