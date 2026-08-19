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
