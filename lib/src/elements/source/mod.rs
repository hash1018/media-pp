mod app_source;
mod audio_mixer;
mod capture;
mod file_demuxer;
mod rtsp_source;
mod test;

pub use app_source::{AppSource, AppSourceError, AppSourceHandle};
pub use audio_mixer::{
    AudioMixer, AudioMixerError, AudioMixerOptions, MixerHandle, MixerInputSink,
};
#[cfg(feature = "wasapi-capture")]
pub use capture::{
    AudioDevice, AudioDeviceKind, WasapiCaptureOptions, WasapiCaptureSource,
    WasapiCaptureSourceError,
};
#[cfg(feature = "dxgi-capture")]
pub use capture::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
pub use file_demuxer::{FileDemuxError, FileDemuxer, StreamInfo};
pub use rtsp_source::{RtspOptions, RtspSource, RtspSourceError, RtspTransport};
pub use test::{
    TestAudioOptions, TestAudioSource, TestAudioSourceError, TestVideoOptions, TestVideoSource,
    TestVideoSourceError,
};
