//! Elements that terminate a branch.
//!
//! Muxers write to files or streams ([`Mp4Muxer`], [`SegmentedMp4Muxer`],
//! [`HlsMuxer`], [`RtspSink`]), renderers present to a device or window, and
//! [`AppSink`] hands buffers back to the application. [`FrameCounter`] and
//! [`PacketCounter`] are the trivial terminals that make a graph runnable
//! while something upstream is being tested.
//!
//! A sink is where [`Eos`](crate::buffer::MediaBuffer::Eos) stops travelling
//! and has to be acted on: anything holding delayed data flushes and finalizes
//! there. A sink's `consume` is also a plain synchronous call — it must bound
//! its own blocking, since the [`Queue`](crate::queue::Queue) in front of it
//! cannot reclaim a worker parked inside one.

mod app_sink;
mod frame_counter;
mod muxer;
#[cfg(feature = "ort")]
mod ort_detector;
mod packet_counter;
mod renderer;
mod rtsp_sink;

pub use app_sink::AppSink;
pub use frame_counter::FrameCounter;
pub use muxer::{
    HlsMode, HlsMuxer, HlsMuxerError, HlsMuxerStreamSink, HlsOptions, HlsSegmentFormat,
};
pub use muxer::{Mp4Muxer, Mp4MuxerError, Mp4MuxerStreamSink, SegmentPolicy, SegmentedMp4Muxer};
#[cfg(feature = "ort")]
pub use ort_detector::{COCO_CLASS_LABELS, Detection, OrtDetector, OrtDetectorError};
pub use packet_counter::PacketCounter;
pub use renderer::SubmitError;
#[cfg(feature = "cuda")]
pub use renderer::{CudaFrameRenderer, CudaRenderer, CudaRendererError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use renderer::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use renderer::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
pub use renderer::{
    PipeWireAudioRenderer, PipeWireAudioRendererError, PipeWireAudioRendererOptions,
};
#[cfg(all(target_os = "windows", feature = "wasapi-renderer"))]
pub use renderer::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
pub use rtsp_sink::{RtspSink, RtspSinkError};
