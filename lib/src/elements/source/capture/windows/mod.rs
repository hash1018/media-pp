#[cfg(feature = "dxgi-capture")]
mod dxgi_capture_source;
#[cfg(feature = "mf-capture")]
mod mf_capture_source;
#[cfg(feature = "wasapi-capture")]
mod wasapi_capture_source;
#[cfg(feature = "wgc-capture")]
mod wgc_capture_source;

#[cfg(feature = "dxgi-capture")]
pub use dxgi_capture_source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(feature = "mf-capture")]
pub use mf_capture_source::{MfCaptureOptions, MfCaptureSource, MfCaptureSourceError};
#[cfg(feature = "wasapi-capture")]
pub use wasapi_capture_source::{
    WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError,
};
#[cfg(feature = "wgc-capture")]
pub use wgc_capture_source::{WgcCaptureOptions, WgcCaptureSource, WgcCaptureSourceError};
