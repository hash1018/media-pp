//! The built-in elements, grouped by the role they play in a graph.
//!
//! [`source`] produces buffers, [`filter`] transforms them, [`sink`] consumes
//! them, and [`driver`] holds the background tasks that have no pads at all.
//! These are built *on* the framework re-exported at the crate root, not part
//! of it — anything here could equally be written outside this crate against
//! the same traits.
//!
//! Every element is re-exported flat from this module, so a backend's type is
//! reached as `media_pp::elements::D3d11Scaler` regardless of which file it
//! lives in. Backend-specific types carry their backend's prefix, require the
//! matching Cargo feature, and exist only on that backend's platform; an
//! unprefixed type is a deliberately backend-independent contract.
//!
//! What a given element accepts, what it owns, and how it behaves under error
//! and runtime control is documented on the element itself rather than here.

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
pub use crate::platform::cuda::{CudaDevice, CudaDeviceError, CudaDriverError, CudaFrameFormat};
#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-capture", feature = "wasapi-renderer")
))]
pub use crate::platform::windows::wasapi::{WasapiDevice, WasapiDeviceKind};

/// The failure detail behind
/// [`PipeWireScreenCaptureSourceError::GpuImport`] — what went wrong setting
/// up or running the DMA-BUF import that
/// [`PipeWireScreenCaptureSource::open_gpu`] depends on.
#[cfg(all(
    target_os = "linux",
    feature = "pipewire-screen-capture",
    feature = "cuda"
))]
pub use crate::platform::linux::dmabuf_cuda::DmaBufCudaError;
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
    AttachedTrack, TrackEndpoints, TrackId, WebRtcError, WebRtcHandle, WebRtcPeer,
    WebRtcStreamInfo, WebRtcTrackSink, WebRtcTrackSource,
};
pub use filter::{
    AudioCodec, AudioResampler, AudioResamplerError, AudioVolume, AudioVolumeError,
    AudioVolumeHandle, AudioVolumeOptions, ChangeGate, ChromaKeyMethod, ChromaKeyOptions,
    FrameRateLimiter, Pacer, PacerError, PauseGate, PauseGateHandle, SwAudioEncoder,
    SwAudioEncoderError, SwAudioEncoderOptions, SwChromaKey, SwChromaKeyError, SwDecoder,
    SwDecoderError, SwEncoder, SwEncoderError, SwEncoderOptions, SwScaler, SwScalerError, Tee,
    TeeBuilder, TeeHandle, TimestampOrigin, VideoCodec, VideoSynchronizer, VideoSynchronizerError,
};
#[cfg(feature = "cuda")]
pub use filter::{
    CudaCodec, CudaConverter, CudaConverterError, CudaDecoder, CudaDecoderError, CudaDownload,
    CudaDownloadError, CudaEncoder, CudaEncoderError, CudaEncoderOptions, CudaScaler,
    CudaScalerError, CudaScalerInterp, CudaUpload, CudaUploadError,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use filter::{
    D3d11ChromaKey, D3d11ChromaKeyError, D3d11Decoder, D3d11DecoderError, D3d11Download,
    D3d11DownloadError, D3d11Scaler, D3d11ScalerError, D3d11ScalerFormat, D3d11Upload,
    D3d11UploadError, D3d11VideoCodec, D3d11VideoEncoder, D3d11VideoEncoderError,
    D3d11VideoEncoderOptions, D3d11VideoInputFormat,
};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use filter::{
    D3d12Decoder, D3d12DecoderError, D3d12Download, D3d12DownloadError, D3d12Scaler,
    D3d12ScalerError, D3d12Upload, D3d12UploadError,
};
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
pub use sink::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError};
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
pub use sink::{PipeWireAudioRenderer, PipeWireAudioRendererError, PipeWireAudioRendererOptions};
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
pub use sink::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
pub use source::{
    AppSource, AppSourceError, AppSourceHandle, AudioMixer, AudioMixerError, AudioMixerOptions,
    FileDemuxError, FileDemuxer, MixFormat, MixerHandle, MixerInputSink, RtspOptions, RtspSource,
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
#[cfg(feature = "cuda")]
pub use source::{
    CudaTextLayerHandle, CudaVideoCompositor, CudaVideoCompositorError, CudaVideoCompositorHandle,
    CudaVideoCompositorInput, CudaVideoCompositorInputSink, CudaVideoLayerHandle,
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
#[cfg(all(target_os = "windows", feature = "wgc-capture"))]
pub use source::{WgcCaptureOptions, WgcCaptureSource, WgcCaptureSourceError};
