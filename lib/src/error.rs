//! The crate-wide error type.
//!
//! [`enum@Error`] is the aggregate an element's own error converts into with `?`,
//! so that a pipeline built from unrelated elements still has one return type.
//! Each variant wraps a component error — `thiserror` enums that stay actionable
//! on their own, documented next to the element that produces them.
//!
//! Backend variants are behind the same Cargo features as the elements that
//! raise them, so this enum is exactly as wide as the build it belongs to.

use std::io;
use std::sync::Arc;

use thiserror::Error;

use crate::element::ElementType;

#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
use crate::elements::DxgiCaptureSourceError;
#[cfg(all(target_os = "windows", feature = "mf-capture"))]
use crate::elements::MfCaptureSourceError;
#[cfg(feature = "ort")]
use crate::elements::OrtDetectorError;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
use crate::elements::PipeWireAudioCaptureSourceError;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
use crate::elements::PipeWireAudioRendererError;
#[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
use crate::elements::PipeWireScreenCaptureSourceError;
use crate::elements::RtspMuxerError;
#[cfg(all(target_os = "linux", feature = "v4l2-capture"))]
use crate::elements::V4l2CaptureSourceError;
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
use crate::elements::WasapiCaptureSourceError;
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
use crate::elements::WasapiRendererError;
#[cfg(feature = "webrtc")]
use crate::elements::WebRtcError;
#[cfg(all(target_os = "windows", feature = "wgc-capture"))]
use crate::elements::WgcCaptureSourceError;
#[cfg(feature = "cuda")]
use crate::elements::{
    CudaChromaKeyError, CudaConverterError, CudaDecoderError, CudaDownloadError, CudaEncoderError,
    CudaRendererError, CudaScalerError, CudaUploadError, CudaVideoCompositorError,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
use crate::elements::{
    D3d11ChromaKeyError, D3d11DecoderError, D3d11DownloadError, D3d11RendererError,
    D3d11ScalerError, D3d11TextLayerError, D3d11UploadError, D3d11VideoCompositorError,
    D3d11VideoEncoderError,
};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
use crate::elements::{
    D3d12DecoderError, D3d12DownloadError, D3d12RendererError, D3d12ScalerError, D3d12UploadError,
};
use crate::{
    control::{PrerollError, SeekError},
    elements::{
        AppSourceError, AudioMixerError, AudioResamplerError, AudioVolumeError, FileDemuxError,
        FileMuxerError, HlsMuxerError, PacerError, RtmpMuxerError, RtspSourceError,
        SwAudioEncoderError, SwChromaKeyError, SwDecoderError, SwEncoderError, SwScalerError,
        SwVideoCompositorError, TestAudioSourceError, TestVideoSourceError, VideoSynchronizerError,
    },
    graph::GraphError,
    log::LogInitError,
    queue::QueueError,
};

/// Failure to create one of the background threads owned by this crate.
///
/// The operation that requested the thread returns this error before claiming
/// that it started successfully. The `thread` field identifies the worker so
/// callers can distinguish pipeline, queue, and standalone-driver failures.
#[derive(Debug, Error)]
#[error("failed to spawn {thread} thread: {source}")]
pub struct ThreadSpawnError {
    thread: String,
    #[source]
    source: io::Error,
}

impl ThreadSpawnError {
    pub(crate) fn new(thread: impl Into<String>, source: io::Error) -> Self {
        Self {
            thread: thread.into(),
            source,
        }
    }

    /// Name of the worker that could not be created.
    pub fn thread(&self) -> &str {
        &self.thread
    }
}

/// FFmpeg could not allocate the reference-counted buffer that owns a D3D11
/// texture attached to an `AVFrame`.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[derive(Debug, Error)]
#[error("FFmpeg could not allocate a D3D11 texture buffer wrapper")]
pub struct D3d11FrameWrapError;

/// A D3D11 device cannot be shared by the elements of one pipeline.
///
/// Every D3D11 element here funnels its GPU commands through the one immediate
/// context its device owns, and a `Queue` deliberately puts elements on
/// different threads. That context is not free-threaded, so each element
/// enables the runtime's `ID3D11Multithread` protection on the device it is
/// handed and refuses a device that cannot be protected — rather than leaving
/// the resulting data race to a caller who has no way to see it.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[derive(Debug, Clone, Error)]
pub enum D3d11SharedDeviceError {
    /// The device was created with `D3D11_CREATE_DEVICE_SINGLETHREADED`, which
    /// promises the runtime that it is used from one thread only. Nothing can
    /// make that device safe here; create it without the flag.
    #[error(
        "the D3D11 device was created with D3D11_CREATE_DEVICE_SINGLETHREADED and cannot be shared across a pipeline's threads"
    )]
    SingleThreaded,

    /// The runtime accepted the request but the protection did not take
    /// effect, so cross-thread use would still be undefined.
    #[error("the D3D11 runtime did not enable multithread protection on the shared context")]
    ProtectionRefused,

    /// The immediate context or its `ID3D11Multithread` interface could not be
    /// obtained from the device.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
}

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
    /// Waiting for a pipeline-wide preroll failed.
    #[error(transparent)]
    PrerollError(#[from] PrerollError),

    /// One or more elements rejected a pipeline-wide seek check.
    #[error(transparent)]
    SeekError(#[from] SeekError),

    /// A pipeline, queue, or driver worker thread could not be created.
    #[error(transparent)]
    ThreadSpawnError(#[from] ThreadSpawnError),

    /// FFmpeg could not allocate a D3D11 frame buffer wrapper.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11FrameWrapError(#[from] D3d11FrameWrapError),

    /// A D3D11 device cannot be shared across a pipeline's threads.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11SharedDeviceError(#[from] D3d11SharedDeviceError),

    /// A file demuxer operation failed.
    #[error(transparent)]
    FileDemuxError(#[from] FileDemuxError),

    /// An application source channel is closed.
    #[error(transparent)]
    AppSourceError(#[from] AppSourceError),

    /// An RTSP source operation failed.
    #[error(transparent)]
    RtspSourceError(#[from] RtspSourceError),

    /// A synthetic video source rejected an operation.
    #[error(transparent)]
    TestVideoSourceError(#[from] TestVideoSourceError),

    /// A synthetic audio source rejected an operation.
    #[error(transparent)]
    TestAudioSourceError(#[from] TestAudioSourceError),

    /// A software decoder operation failed.
    #[error(transparent)]
    SwDecoderError(#[from] SwDecoderError),

    /// A CUDA decoder operation failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaDecoderError(#[from] CudaDecoderError),

    /// A CUDA renderer operation failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaRendererError(#[from] CudaRendererError),

    /// Uploading a frame to CUDA failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaUploadError(#[from] CudaUploadError),

    /// Downloading a frame from CUDA failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaDownloadError(#[from] CudaDownloadError),

    /// A CUDA scaling operation failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaScalerError(#[from] CudaScalerError),

    /// A CUDA pixel-format conversion failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaConverterError(#[from] CudaConverterError),

    /// Keying a CUDA surface failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaChromaKeyError(#[from] CudaChromaKeyError),

    /// A CUDA compositor operation failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaVideoCompositorError(#[from] CudaVideoCompositorError),

    /// A CUDA encoder operation failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaEncoderError(#[from] CudaEncoderError),

    /// A software video encoder operation failed.
    #[error(transparent)]
    SwEncoderError(#[from] SwEncoderError),

    /// A pacer could not schedule an input timestamp.
    #[error(transparent)]
    PacerError(#[from] PacerError),

    /// A video synchronizer could not schedule an input frame.
    #[error(transparent)]
    VideoSynchronizerError(#[from] VideoSynchronizerError),

    /// A software audio encoder operation failed.
    #[error(transparent)]
    SwAudioEncoderError(#[from] SwAudioEncoderError),

    /// An audio resampling operation failed.
    #[error(transparent)]
    AudioResamplerError(#[from] AudioResamplerError),

    /// An audio gain operation failed.
    #[error(transparent)]
    AudioVolumeError(#[from] AudioVolumeError),

    /// A software scaling operation failed.
    #[error(transparent)]
    SwScalerError(#[from] SwScalerError),

    /// A software chroma-key operation failed.
    #[error(transparent)]
    SwChromaKeyError(#[from] SwChromaKeyError),

    /// A queue worker or capacity policy failed.
    #[error(transparent)]
    QueueError(#[from] QueueError),

    /// See [`crate::elements::PipelineBridgeError`].
    #[error(transparent)]
    PipelineBridgeError(#[from] crate::elements::PipelineBridgeError),

    /// A pipeline graph mutation violated a topology invariant.
    #[error(transparent)]
    GraphError(#[from] GraphError),

    /// Private file logging could not be initialized.
    #[error(transparent)]
    LogInitError(#[from] LogInitError),

    /// An audio mixer operation failed.
    #[error(transparent)]
    AudioMixerError(#[from] AudioMixerError),

    /// A software video compositor operation failed.
    #[error(transparent)]
    SwVideoCompositorError(#[from] SwVideoCompositorError),

    /// Writing a container file failed.
    #[error(transparent)]
    FileMuxerError(#[from] FileMuxerError),

    /// HLS muxing or option validation failed.
    #[error(transparent)]
    HlsMuxerError(#[from] HlsMuxerError),

    /// Sending a stream to an RTSP endpoint failed.
    #[error(transparent)]
    RtspMuxerError(#[from] RtspMuxerError),

    /// Publishing to an RTMP server failed.
    #[error(transparent)]
    RtmpMuxerError(#[from] RtmpMuxerError),

    /// A D3D12 renderer operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12RendererError(#[from] D3d12RendererError),

    /// A D3D12 decoder operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12DecoderError(#[from] D3d12DecoderError),

    /// Uploading a frame to D3D12 failed.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12UploadError(#[from] D3d12UploadError),

    /// Downloading a frame from D3D12 failed.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12DownloadError(#[from] D3d12DownloadError),

    /// A D3D12 scaling operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    #[error(transparent)]
    D3d12ScalerError(#[from] D3d12ScalerError),

    /// A D3D11 decoder operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11DecoderError(#[from] D3d11DecoderError),

    /// Uploading a frame to D3D11 failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11UploadError(#[from] D3d11UploadError),

    /// Downloading a frame from D3D11 failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11DownloadError(#[from] D3d11DownloadError),

    /// A D3D11 scaling operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11ScalerError(#[from] D3d11ScalerError),

    /// A D3D11 chroma-key operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11ChromaKeyError(#[from] D3d11ChromaKeyError),

    /// A D3D11-backed NVENC operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11VideoEncoderError(#[from] D3d11VideoEncoderError),

    /// A D3D11 renderer operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11RendererError(#[from] D3d11RendererError),

    /// A D3D11 compositor operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11VideoCompositorError(#[from] D3d11VideoCompositorError),

    /// A D3D11 text-layer operation failed.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    #[error(transparent)]
    D3d11TextLayerError(#[from] D3d11TextLayerError),

    /// Desktop duplication capture failed.
    #[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
    #[error(transparent)]
    DxgiCaptureSourceError(#[from] DxgiCaptureSourceError),

    /// Windows Graphics Capture failed.
    #[cfg(all(target_os = "windows", feature = "wgc-capture"))]
    #[error(transparent)]
    WgcCaptureSourceError(#[from] WgcCaptureSourceError),

    /// PipeWire audio capture failed.
    #[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
    #[error(transparent)]
    PipeWireAudioCaptureSourceError(#[from] PipeWireAudioCaptureSourceError),

    /// PipeWire audio rendering failed.
    #[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
    #[error(transparent)]
    PipeWireAudioRendererError(#[from] PipeWireAudioRendererError),

    /// PipeWire screen capture failed.
    #[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
    #[error(transparent)]
    PipeWireScreenCaptureSourceError(#[from] PipeWireScreenCaptureSourceError),

    /// Media Foundation camera capture failed.
    #[cfg(all(target_os = "windows", feature = "mf-capture"))]
    #[error(transparent)]
    MfCaptureSourceError(#[from] MfCaptureSourceError),

    /// V4L2 camera capture failed.
    #[cfg(all(target_os = "linux", feature = "v4l2-capture"))]
    #[error(transparent)]
    V4l2CaptureSourceError(#[from] V4l2CaptureSourceError),

    /// WASAPI audio capture failed.
    #[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
    #[error(transparent)]
    WasapiCaptureSourceError(#[from] WasapiCaptureSourceError),

    /// WASAPI audio rendering failed.
    #[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
    #[error(transparent)]
    WasapiRendererError(#[from] WasapiRendererError),

    /// ONNX Runtime inference or detector processing failed.
    #[cfg(feature = "ort")]
    #[error(transparent)]
    OrtDetectorError(#[from] OrtDetectorError),

    /// A WebRTC peer operation failed.
    #[cfg(feature = "webrtc")]
    #[error(transparent)]
    WebRtcError(#[from] WebRtcError),

    /// An FFmpeg error not assigned to a more specific element error.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    /// An application-defined error message without a more specific category.
    #[error("{0}")]
    Other(String),

    /// Any of the above, plus the element that raised it — see [`Traced`].
    ///
    /// Reads and behaves exactly like what it wraps: `Display` forwards, and
    /// `source()` reaches the inner error, so nothing that only prints or
    /// chains has to know this exists. What it adds is answerable through
    /// [`Error::origin`].
    #[error(transparent)]
    Traced(#[from] Traced),
}

/// Which element an error came from.
///
/// The name as well as the type, because a graph holds several of most
/// types: "a muxer failed" is not actionable where "the muxer named
/// `stream-video` failed" is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub element_type: ElementType,
    pub name: Arc<str>,
}

/// An error together with the element that raised it.
///
/// # Why an error has to carry this
///
/// A failure deep in a chain reaches whoever can act on it only after
/// travelling up through every stage in front of it, one `?` at a time. The
/// value survives that trip; the identity does not. So a `Queue` reporting a
/// downstream failure could say *what* went wrong and never *where* — and
/// "where" is what decides whether a broadcast has dropped or an encoder
/// hiccupped.
///
/// The identity is attached by whichever tracer sees the error first, which
/// is the one closest to the failure — see `FlowTracer` and `TerminalTracer`
/// in `pipeline::chain`. Once attached it is never replaced, so what an
/// observer reads is the element that raised it rather than the last one to
/// pass it on.
#[derive(Debug, Error)]
// Reads as the error it wraps and nothing more: the origin is for whoever
// asks [`Error::origin`], not for the message. A line that named the element
// twice — once here and once in the report that carries it — would be worse
// than one that names it where it is acted on.
#[error("{source}")]
pub struct Traced {
    pub origin: Origin,
    #[source]
    pub source: Box<Error>,
}

impl Error {
    /// Attaches `element_type`/`name` as this error's origin, unless it
    /// already has one.
    ///
    /// Idempotent by design: every stage between the failure and whoever
    /// reports it calls this, and the first one to — the innermost, nearest
    /// the failure — is the one whose answer is kept.
    #[must_use]
    pub fn traced_at(self, element_type: ElementType, name: Arc<str>) -> Self {
        if matches!(self, Self::Traced(_)) {
            return self;
        }
        Self::Traced(Traced {
            origin: Origin { element_type, name },
            source: Box::new(self),
        })
    }

    /// Which element raised this, where that is known.
    ///
    /// `None` for an error that never crossed a chain stage — one returned
    /// straight to its caller, which already knows who it asked.
    #[must_use]
    pub fn origin(&self) -> Option<&Origin> {
        match self {
            Self::Traced(traced) => Some(&traced.origin),
            _ => None,
        }
    }
}

/// The crate's `Result`, with [`enum@Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;
