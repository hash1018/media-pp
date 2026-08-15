#[cfg(feature = "dxgi-capture")]
mod dxgi_capture_source;
#[cfg(feature = "wasapi-capture")]
mod wasapi_capture_source;

#[cfg(feature = "dxgi-capture")]
pub use dxgi_capture_source::{
    CaptureArea, CaptureMode, CaptureRect, DxgiCaptureOptions, DxgiCaptureSource,
    DxgiCaptureSourceError,
};
#[cfg(feature = "wasapi-capture")]
pub use wasapi_capture_source::{
    WasapiCaptureOptions, WasapiCaptureSource, WasapiCaptureSourceError,
};
