mod app_source;
#[cfg(feature = "wasapi-capture")]
mod audio_capture_source;
#[cfg(feature = "dxgi-capture")]
mod dxgi_screen_source;
mod file_demuxer;
mod rtsp_source;
mod test_video_source;

pub use app_source::{AppSource, AppSourceError, AppSourceHandle};
#[cfg(feature = "wasapi-capture")]
pub use audio_capture_source::{
    AudioCaptureOptions, AudioCaptureSource, AudioCaptureSourceError, AudioDevice, AudioDeviceKind,
};
#[cfg(feature = "dxgi-capture")]
pub use dxgi_screen_source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiScreenOptions, DxgiScreenSource,
    DxgiScreenSourceError,
};
pub use file_demuxer::{FileDemuxError, FileDemuxer, StreamInfo};
pub use rtsp_source::{RtspOptions, RtspSource, RtspSourceError, RtspTransport};
pub use test_video_source::{TestVideoOptions, TestVideoSource, TestVideoSourceError};
