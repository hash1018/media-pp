use thiserror::Error;

#[cfg(feature = "dx12-renderer")]
use crate::elements::Dx12RendererError;
use crate::{
    elements::{DecoderError, FileDemuxError},
    queue::QueueError,
};

/// Crate-wide error. Each element defines its own `{Element}Error` (see
/// [`FileDemuxError`], [`DecoderError`], [`QueueError`]) for its own
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
    DecoderError(#[from] DecoderError),

    #[error(transparent)]
    QueueError(#[from] QueueError),

    #[cfg(feature = "dx12-renderer")]
    #[error(transparent)]
    Dx12RendererError(#[from] Dx12RendererError),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
