use thiserror::Error;

#[cfg(feature = "ort")]
use crate::elements::OrtDetectorError;
#[cfg(feature = "rtsp-server")]
use crate::elements::RtspServerError;
#[cfg(feature = "webrtc")]
use crate::elements::WebRtcError;
#[cfg(feature = "dx12-renderer")]
use crate::elements::{D3d12vaDecoderError, Dx12RendererError};
use crate::{
    elements::{AppSourceError, FileDemuxError, RtspSourceError, ScalerError, SwDecoderError},
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
    SwDecoderError(#[from] SwDecoderError),

    #[error(transparent)]
    ScalerError(#[from] ScalerError),

    #[error(transparent)]
    QueueError(#[from] QueueError),

    #[cfg(feature = "rtsp-server")]
    #[error(transparent)]
    RtspServerError(#[from] RtspServerError),

    #[cfg(feature = "dx12-renderer")]
    #[error(transparent)]
    Dx12RendererError(#[from] Dx12RendererError),

    #[cfg(feature = "dx12-renderer")]
    #[error(transparent)]
    D3d12vaDecoderError(#[from] D3d12vaDecoderError),

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
