mod app_source;
mod audio_mixer;
mod capture;
mod compositor;
mod file_demuxer;
mod rtsp_source;
mod test;

#[cfg(any(feature = "wasapi-capture", feature = "wasapi-renderer"))]
pub use crate::platform::windows::wasapi::{WasapiDevice, WasapiDeviceKind};
pub use app_source::{AppSource, AppSourceError, AppSourceHandle};
pub use audio_mixer::{
    AudioMixer, AudioMixerError, AudioMixerOptions, MixerHandle, MixerInputSink,
};
#[cfg(feature = "dxgi-capture")]
pub use capture::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(feature = "wasapi-capture")]
pub use capture::{WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError};
#[cfg(feature = "d3d11-renderer")]
pub use compositor::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
pub use compositor::{
    TextLayer, VideoCompositor, VideoCompositorError, VideoCompositorHandle, VideoCompositorInput,
    VideoCompositorInputSink, VideoCompositorOptions, VideoFit, VideoInputId, VideoLayer,
    VideoLayerHandle, VideoRect,
};
pub use file_demuxer::{FileDemuxError, FileDemuxer, StreamInfo};
pub use rtsp_source::{RtspOptions, RtspSource, RtspSourceError, RtspTransport};
pub use test::{
    TestAudioOptions, TestAudioSource, TestAudioSourceError, TestVideoOptions, TestVideoSource,
    TestVideoSourceError,
};
