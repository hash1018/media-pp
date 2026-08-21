//! Elements that produce buffers.
//!
//! Three shapes live here. Readers pull from something that already exists —
//! a container, an RTSP stream, a capture device, or the application itself
//! through [`AppSource`]. Generators synthesize
//! ([`TestVideoSource`], [`TestAudioSource`]). And fan-in elements
//! ([`AudioMixer`] and the video compositors) are sources with inputs: they
//! accept buffers from several upstream branches and emit one combined stream
//! on their own schedule.
//!
//! A pipeline is driven by exactly one of these, the one implementing
//! [`SourceElement`](crate::element::SourceElement) — its `run` loop is what
//! makes the whole graph move.

mod app_source;
mod audio_mixer;
mod capture;
mod compositor;
mod file_demuxer;
mod rtsp_source;
mod test;

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
#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-capture", feature = "wasapi-renderer")
))]
pub use crate::platform::windows::wasapi::{WasapiDevice, WasapiDeviceKind};
pub use app_source::{AppSource, AppSourceError, AppSourceHandle};
pub use audio_mixer::{
    AudioMixer, AudioMixerError, AudioMixerOptions, MixerHandle, MixerInputSink,
};
#[cfg(all(target_os = "windows", feature = "dxgi-capture"))]
pub use capture::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
pub use capture::{
    CaptureSourceKind, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
    PipeWireScreenCaptureSourceError,
};
#[cfg(all(target_os = "linux", feature = "pipewire-audio-capture"))]
pub use capture::{
    PipeWireAudioCaptureOptions, PipeWireAudioCaptureSource, PipeWireAudioCaptureSourceError,
};
#[cfg(all(target_os = "windows", feature = "wasapi-capture"))]
pub use capture::{WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError};
#[cfg(feature = "cuda")]
pub use compositor::{
    CudaTextLayerHandle, CudaVideoCompositor, CudaVideoCompositorError, CudaVideoCompositorHandle,
    CudaVideoCompositorInput, CudaVideoCompositorInputSink, CudaVideoLayerHandle,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use compositor::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
pub use compositor::{
    SwVideoCompositor, SwVideoCompositorError, SwVideoCompositorHandle, SwVideoCompositorInput,
    SwVideoCompositorInputSink, SwVideoLayerHandle, TextLayer, VideoCompositorOptions, VideoFit,
    VideoInputId, VideoLayer, VideoRect,
};
pub use file_demuxer::{FileDemuxError, FileDemuxer, StreamInfo};
pub use rtsp_source::{RtspOptions, RtspSource, RtspSourceError};
pub use test::{
    TestAudioOptions, TestAudioSource, TestAudioSourceError, TestVideoOptions, TestVideoSource,
    TestVideoSourceError,
};
