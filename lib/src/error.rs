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
#[cfg(all(target_os = "windows", feature = "wgc-capture"))]
use crate::elements::WgcCaptureSourceError;
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
    control::{PrerollError, SeekError},
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

    /// MP4 muxing failed.
    #[error(transparent)]
    Mp4MuxerError(#[from] Mp4MuxerError),

    /// HLS muxing or option validation failed.
    #[error(transparent)]
    HlsMuxerError(#[from] HlsMuxerError),

    /// Sending a stream to an RTSP endpoint failed.
    #[error(transparent)]
    RtspSinkError(#[from] RtspSinkError),

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
    D3d11NvencEncoderError(#[from] D3d11NvencEncoderError),

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
}

/// The crate's `Result`, with [`enum@Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;
