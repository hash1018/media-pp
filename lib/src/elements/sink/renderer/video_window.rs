//! A window that shows video on whichever renderer this platform has — the
//! GStreamer `autovideosink` of this crate.

use std::sync::Arc;

use thiserror::Error as ThisError;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
use crate::elements::{D3d11Gpu as Gpu, D3d11WindowRenderer as Backend};
#[cfg(all(target_os = "windows", feature = "d3d12", not(feature = "d3d11")))]
use crate::elements::{D3d12Gpu as Gpu, D3d12WindowRenderer as Backend};
#[cfg(all(target_os = "linux", feature = "vulkan"))]
use crate::elements::{VulkanGpu as Gpu, VulkanWindowRenderer as Backend};
use crate::{
    buffer::MediaBuffer,
    contract::InputContract,
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink},
    elements::{WindowControl, WindowEvents, WindowOptions},
    error::Result,
    pp_log::PpLog,
};

/// Why a [`VideoWindow`] could not be opened.
#[derive(Debug, ThisError)]
pub enum VideoWindowError {
    /// [`WindowOptions`] asked for a window with no area.
    #[error("a window of {width}x{height} has nothing to draw in")]
    EmptyWindow {
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
    },
    /// No GPU could be opened to draw with. The source is the platform
    /// renderer's own GPU error.
    #[error("no GPU to draw with: {0}")]
    Gpu(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The window, or presenting into it, could not be set up. The source is
    /// the platform renderer's own error.
    #[error("could not open the window: {0}")]
    Window(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// A terminal sink that shows video in a window of its own, on whichever
/// renderer this platform has — `D3d11WindowRenderer` on Windows (or
/// `D3d12WindowRenderer` in a build with only `d3d12`), `VulkanWindowRenderer`
/// on Linux — with a GPU of its own to draw it with. The GStreamer
/// `autovideosink` of this crate: the one video sink a program can name
/// without a `#[cfg]`.
///
/// What it takes is what every one of those takes: frames in system memory,
/// NV12, YUV420P or BGRA, uploaded and drawn each in its own colours — what a
/// software decode, a CPU capture or an application's own frames give, with
/// no scaler or upload in front. A software decode of a layout none of them
/// draws, a 10-bit or a 4:4:4 one, needs a `SwScaler` to YUV420P first; the
/// link check says so when the pipeline is built.
///
/// GPU frames are the platform renderer's business, not this one's: this
/// window's GPU is its own, and a decoder's or a capture's textures must be
/// on the renderer's device to be drawn. For those — zero-copy D3D11VA or
/// NVDEC playback — open the platform renderer itself with the GPU the rest
/// of the pipeline shares.
///
/// Everything else is its platform renderer's: the window, what it reports
/// as [`WindowEvents`], what [`Self::window_control`] can change, the log
/// and the graph, where it appears under the platform renderer's type.
pub struct VideoWindow {
    renderer: Backend,
    control: WindowControl,
}

impl VideoWindow {
    /// Opens a window of its own, on a thread of its own, and a GPU to draw
    /// into it with.
    pub fn open(
        name: impl Into<String>,
        options: WindowOptions,
    ) -> std::result::Result<(Self, WindowEvents), VideoWindowError> {
        if options.width == 0 || options.height == 0 {
            return Err(VideoWindowError::EmptyWindow {
                width: options.width,
                height: options.height,
            });
        }
        let gpu = Gpu::new().map_err(|error| VideoWindowError::Gpu(Box::new(error)))?;
        Self::open_on(name, &gpu, options)
    }

    /// [`Self::open`], on a GPU a decoder can put its pictures on too — and
    /// the [`DecodeTarget`](crate::elements::DecodeTarget) that puts them there, `downstream_hw_frames` being
    /// its surface budget. That is D3D11 on Windows (D3D12 in a build with
    /// only `d3d12`), and CUDA on Linux where the build has `cuda` and the
    /// machine an NVIDIA GPU; on Linux otherwise, system memory.
    #[cfg(any(
        all(target_os = "windows", feature = "wasapi-renderer"),
        all(target_os = "linux", feature = "pipewire-audio-renderer")
    ))]
    pub(crate) fn open_for_decoding(
        name: impl Into<String>,
        options: WindowOptions,
        downstream_hw_frames: i32,
    ) -> std::result::Result<(Self, WindowEvents, crate::elements::DecodeTarget), VideoWindowError>
    {
        if options.width == 0 || options.height == 0 {
            return Err(VideoWindowError::EmptyWindow {
                width: options.width,
                height: options.height,
            });
        }
        use crate::elements::DecodeTarget;

        let gpu_error = |error| VideoWindowError::Gpu(Box::new(error));
        #[cfg(all(target_os = "windows", feature = "d3d11"))]
        let (gpu, target) = {
            let gpu = Gpu::new().map_err(gpu_error)?;
            let target = DecodeTarget::D3d11 {
                gpu: gpu.clone(),
                downstream_hw_frames,
            };
            (gpu, target)
        };
        #[cfg(all(target_os = "windows", feature = "d3d12", not(feature = "d3d11")))]
        let (gpu, target) = {
            let _ = downstream_hw_frames;
            let gpu = Gpu::new().map_err(gpu_error)?;
            (gpu.clone(), DecodeTarget::D3d12 { gpu })
        };
        #[cfg(all(target_os = "linux", feature = "vulkan", feature = "cuda"))]
        let (gpu, target) = match crate::elements::CudaDevice::new()
            .ok()
            .and_then(|device| Some((Gpu::for_cuda(&device).ok()?, device)))
        {
            Some((gpu, device)) => (
                gpu,
                DecodeTarget::Cuda {
                    device,
                    downstream_hw_frames,
                },
            ),
            // No NVIDIA GPU, or no Vulkan device on it: pictures are
            // decoded in software, and drawn on whatever GPU draws.
            None => (Gpu::new().map_err(gpu_error)?, DecodeTarget::System),
        };
        #[cfg(all(target_os = "linux", feature = "vulkan", not(feature = "cuda")))]
        let (gpu, target) = {
            let _ = downstream_hw_frames;
            (Gpu::new().map_err(gpu_error)?, DecodeTarget::System)
        };
        let (window, events) = Self::open_on(name, &gpu, options)?;
        Ok((window, events, target))
    }

    fn open_on(
        name: impl Into<String>,
        gpu: &Gpu,
        options: WindowOptions,
    ) -> std::result::Result<(Self, WindowEvents), VideoWindowError> {
        let (renderer, events) = Backend::open(name, gpu, options)
            .map_err(|error| VideoWindowError::Window(Box::new(error)))?;
        let control = renderer.window_control().ok_or_else(|| {
            VideoWindowError::Window("the renderer opened no window of its own".into())
        })?;
        Ok((Self { renderer, control }, events))
    }

    /// What changes the window — its title, whether it fills the screen —
    /// while it is drawn into; taken before this goes into a pipeline.
    pub fn window_control(&self) -> WindowControl {
        self.control.clone()
    }
}

impl Element for VideoWindow {
    fn name(&self) -> Arc<str> {
        self.renderer.name()
    }

    fn element_type(&self) -> ElementType {
        self.renderer.element_type()
    }

    fn pp_log(&self) -> &PpLog {
        self.renderer.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.renderer.pp_log_mut()
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.renderer.attach_context(context);
    }
}

impl Sink for VideoWindow {
    fn input_contract(&self) -> InputContract {
        self.renderer.input_contract()
    }

    fn ready_consume(&mut self) -> bool {
        self.renderer.ready_consume()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.renderer.consume(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.renderer.control(msg)
    }
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{
        bus::BusEvent,
        contract::{
            MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet, PortContract,
            check_link,
        },
        elements::{TestVideoOptions, TestVideoSource},
        pipeline::Pipeline,
    };

    /// A synthetic source's YUV420P goes straight into it, with nothing in
    /// between, and is drawn.
    #[test]
    fn a_source_in_system_memory_goes_straight_in() {
        let (window, events) = match VideoWindow::open(
            "screen",
            WindowOptions {
                title: "media-pp video window test".into(),
                width: 320,
                height: 240,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: no video window here ({error})");
                return;
            }
        };
        let control = window.window_control();
        control
            .set_title("media-pp video window test, renamed")
            .unwrap();
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: 320,
                height: 240,
                frame_rate: ffmpeg_next::Rational::new(30, 1),
            },
        );
        let (pipeline, ()) = Pipeline::new("video-window", source, |source, ctx| {
            let branch = ctx.branch().queue("frames", 4).to(window)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("a software source links straight to it");
        pipeline.run().expect("run");
        std::thread::sleep(Duration::from_millis(500));
        let shown = pipeline
            .stats()
            .elements
            .iter()
            .filter(|element| element.element_type != ElementType::Queue)
            .find(|element| &*element.name == "screen")
            .map_or(0, |element| element.buffers_in);
        pipeline.stop();
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
        assert!(shown >= 5, "only {shown} frames reached the window");
        drop(pipeline);
        // What it reported before it went, then nothing: the window went with
        // the pipeline that owned it.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            match events.events.recv_timeout(Duration::from_millis(100)) {
                Ok(_) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    assert!(std::time::Instant::now() < deadline, "the window stayed");
                }
            }
        }
    }

    /// What its link check refuses is what no platform renderer draws.
    #[test]
    fn a_layout_nothing_draws_is_refused_when_linked() {
        let Ok((window, _events)) = VideoWindow::open(
            "screen",
            WindowOptions {
                width: 64,
                height: 64,
                ..WindowOptions::default()
            },
        ) else {
            return;
        };
        let decoded = |layout| {
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                    .with_layouts(PixelLayoutSet::of(layout)),
            )
        };
        let accepted = window.input_contract();
        assert!(!check_link(&decoded(PixelLayout::Yuv420p), &accepted).is_refused());
        assert!(check_link(&decoded(PixelLayout::P010), &accepted).is_refused());
    }

    #[test]
    fn a_window_with_nothing_to_draw_in_is_refused() {
        assert!(matches!(
            VideoWindow::open(
                "screen",
                WindowOptions {
                    height: 0,
                    ..WindowOptions::default()
                },
            ),
            Err(VideoWindowError::EmptyWindow { height: 0, .. })
        ));
    }
}
