//! A D3D12 renderer that brings its own window, or draws into one it is
//! given — the part a program otherwise writes for itself behind
//! [`D3d12FrameRenderer`](crate::elements::D3d12FrameRenderer).

use std::{
    any::Any,
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, HWND, RECT},
        Graphics::{
            Direct3D::{
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                Fxc::{D3D_BLOB_ROOT_SIGNATURE, D3DGetBlobPart},
                ID3DBlob,
            },
            Direct3D12::*,
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_DEVICE_HUNG,
                DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET, DXGI_MWA_NO_ALT_ENTER,
                DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory2,
                IDXGISwapChain3,
            },
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
        UI::WindowsAndMessaging::GetClientRect,
    },
    core::{Interface, s},
};

use super::{
    d3d12_renderer::{D3d12Picture, check_d3d12_frame},
    present_timing::PresentTiming,
    system_frame::{PlaneShape, SystemLayout, colour_rows, planes_of},
};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MediaKindSet, MemoryDomain, MemoryDomainSet, PixelLayout,
        PixelLayoutSet, PortContract,
    },
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::{
        D3d12Gpu, D3d12RendererError, SubmitError, WindowControl, WindowEvents, WindowOptions,
    },
    error::Result,
    platform::windows::{hlsl::compile_shader, window::OwnedWindow},
    pool::UnboundObjectPoolRef,
    pp_log::{PpLog, pp_error, pp_info},
};

const FRAME_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_frame.hlsl");
const NV12_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_nv12.hlsl");
const YUV420P_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_yuv420p.hlsl");
const BGRA_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_bgra.hlsl");

/// Double-buffered flip-model swap chain.
const FRAME_COUNT: usize = 2;

/// Why a [`D3d12WindowRenderer`] could not be set up.
#[derive(Debug, ThisError)]
pub enum D3d12WindowRendererError {
    /// [`WindowOptions`] asked for a window with no area.
    #[error("a window of {width}x{height} has nothing to draw in")]
    EmptyWindow {
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
    },
    /// The window given is not a Win32 window, or has no handle to give.
    #[error("the window given has no Win32 handle to draw into")]
    NotAWin32Window,
    /// Windows would not create the window.
    #[error("could not open a window: {0}")]
    Window(windows::core::Error),
    /// Direct3D would not set up presenting into the window.
    #[error("could not present into the window: {0}")]
    Present(windows::core::Error),
}

/// A terminal sink that shows D3D12 video frames in a window — one it opens
/// for itself, the way a GStreamer video sink does when nobody hands it one,
/// or one the application gives it. The D3D12 sibling of
/// `D3d11WindowRenderer`, with the same [`WindowOptions`] and [`WindowEvents`].
///
/// It takes two kinds of frame:
///
/// - What a [`D3d12Renderer`](crate::elements::D3d12Renderer) takes — the
///   NV12 textures `D3d12Decoder` and `D3d12Upload` make, drawn zero-copy
///   after a GPU wait on each frame's own fence — checked the same way: a
///   texture from another device is refused.
/// - A frame in system memory, NV12, YUV420P (or YUVJ420P) or BGRA — what a
///   software decode, a CPU capture or an application's own frames give —
///   copied into an upload buffer here and from there into textures made for
///   its layout and size, as part of the same draw. A software decode needs
///   no scaler or upload in front, as on Linux with `VulkanWindowRenderer`,
///   and needs no video support of the device.
///
/// A frame's error is a [`D3d12RendererError`]. It draws each frame itself
/// rather than through a presenter, and so knows it: a YUV frame is
/// converted with its own colour description, BT.709, BT.601 or BT.2020 and
/// limited or full range as it says, and where it says nothing, BT.709 for a
/// picture over 576 rows and BT.601 otherwise. A BGRA frame is drawn as it
/// is. The picture keeps its aspect ratio inside the window, with black bars
/// as needed, and the renderer follows the window's size on its own, reading
/// it before every frame.
///
/// It draws; it does not pace. Put a [`crate::elements::VideoSynchronizer`]
/// or [`crate::elements::Pacer`] in front for a picture shown at its own
/// time. Each frame is presented synchronized to the display's refresh, and
/// one frame is in flight at a time: each waits for the one before it to
/// finish on the GPU.
///
/// Handed a picture, it tells the pipeline's playback clock what putting it
/// on the screen takes — measured from the swap chain's frame statistics,
/// or two refreshes of the desktop's compositor until they say — so a
/// `VideoSynchronizer` in front hands each picture over that much early and
/// it is shown when its sound is heard.
///
/// Its GPU is the [`D3d12Gpu`] every other D3D12 element in the pipeline
/// shares — the frames it shows must come from that device.
pub struct D3d12WindowRenderer {
    name: Arc<str>,
    pp_log: PpLog,
    presenter: WindowPresenter,
    /// `Some` for a window it opened itself.
    control: Option<WindowControl>,
}

impl D3d12WindowRenderer {
    /// Opens a window of its own on a thread of its own and draws into it.
    ///
    /// What the window reports — keys, resizing, the user closing it —
    /// comes out of the [`WindowEvents`] returned beside it; the renderer
    /// acts on none of it but resizing. Closing only hides the window, and
    /// the window is destroyed when the renderer is dropped. It is a plain
    /// Win32 window, not a `winit` one, so it does not collide with an
    /// application's own event loop, and several can be open at once.
    pub fn open(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: WindowOptions,
    ) -> std::result::Result<(Self, WindowEvents), D3d12WindowRendererError> {
        if options.width == 0 || options.height == 0 {
            return Err(D3d12WindowRendererError::EmptyWindow {
                width: options.width,
                height: options.height,
            });
        }
        let (events_tx, events) = crossbeam_channel::unbounded();
        let window =
            OwnedWindow::open(&options, events_tx).map_err(D3d12WindowRendererError::Window)?;
        let hwnd = window.hwnd();
        let control = WindowControl {
            control: window.control(),
        };
        let presenter = WindowPresenter::new(gpu, hwnd, Keep::Owned(window))?;
        Ok((
            Self::around(name, presenter, Some(control)),
            WindowEvents { events },
        ))
    }

    /// Draws into `window`, which the application owns and runs the event
    /// loop of — a `winit` window, or anything else with a Win32 handle. The
    /// renderer keeps its `Arc`, so the window cannot be dropped while it is
    /// being drawn into. Keys and closing are the application's own events
    /// here; there is nothing for the renderer to report.
    pub fn for_window<W>(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        window: Arc<W>,
    ) -> std::result::Result<Self, D3d12WindowRendererError>
    where
        W: HasWindowHandle + Send + Sync + 'static,
    {
        let hwnd = match window.window_handle().map(|handle| handle.as_raw()) {
            Ok(RawWindowHandle::Win32(handle)) => HWND(handle.hwnd.get() as *mut c_void),
            _ => return Err(D3d12WindowRendererError::NotAWin32Window),
        };
        let presenter = WindowPresenter::new(gpu, hwnd, Keep::Given(window))?;
        Ok(Self::around(name, presenter, None))
    }

    /// What changes the window it opened — its title, whether it fills the
    /// screen — while it is drawn into; taken before the renderer goes into
    /// a pipeline. `None` for a window it was given, which is the
    /// application's to change. See [`WindowControl`].
    pub fn window_control(&self) -> Option<WindowControl> {
        self.control.clone()
    }

    fn around(
        name: impl Into<String>,
        presenter: WindowPresenter,
        control: Option<WindowControl>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12WindowRenderer, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            presenter,
            control,
        }
    }

    /// Where the centre pixel of each frame drawn is read back to.
    #[cfg(test)]
    fn probe(&self) -> Arc<Mutex<Option<[u8; 4]>>> {
        Arc::clone(&self.presenter.probe)
    }

    fn draw(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> std::result::Result<(), D3d12RendererError> {
        if frame.format() != Pixel::D3D12 {
            let layout = SystemLayout::of(frame.format())
                .ok_or(D3d12RendererError::UnsupportedFormat(frame.format()))?;
            return self
                .presenter
                .show_system(&frame, layout)
                .map_err(D3d12RendererError::Submit);
        }
        let picture = check_d3d12_frame(&frame, &self.presenter.device)?;
        let (width, height) = (frame.width(), frame.height());
        let rows = colour_rows(&frame);
        // SAFETY: `check_d3d12_frame` established the texture's device, and
        // the fence is its producer's own. `frame` keeps the pooled texture
        // alive until the GPU is done with it.
        unsafe {
            self.presenter
                .show(picture, (width, height), rows, Box::new(frame))
                .map_err(D3d12RendererError::Submit)
        }
    }
}

impl Element for D3d12WindowRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12WindowRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    /// Takes a place in the pipeline's presentation delay, which a
    /// `VideoSynchronizer` in front hands pictures over early by.
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        if let Ok(mut state) = self.presenter.state.lock() {
            state.timing.delay.registration = Some(context.playback_clock.register_presenter());
        }
    }
}

impl Sink for D3d12WindowRenderer {
    /// An NV12 D3D12 texture, as a
    /// [`D3d12Renderer`](crate::elements::D3d12Renderer) takes; or a frame in
    /// system memory, NV12, YUV420P or BGRA.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Frames(
            MediaKindSet::of(MediaKind::VideoFrame),
            MemoryDomainSet::from_slice(&[MemoryDomain::D3d12, MemoryDomain::System]),
            PixelLayoutSet::from_slice(&[
                PixelLayout::Nv12,
                PixelLayout::Yuv420p,
                PixelLayout::Bgra,
            ]),
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };
        self.draw(frame)
            .inspect_err(|error| pp_error!(self, "draw failed: {error}"))?;
        let change = self.presenter.state.lock().ok().and_then(|mut state| {
            let (delay, source) = state.timing.delay.take_change()?;
            Some((delay, source.to_owned()))
        });
        if let Some((delay, source)) = change {
            pp_info!(
                self,
                "a picture takes {delay:.1?} to reach the screen, {source}"
            );
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, with nothing to flush or forward.
        Ok(())
    }
}

/// What keeps the window alive for as long as the presenter draws into it.
enum Keep {
    /// Opened here, destroyed when this drops.
    Owned(#[allow(dead_code)] OwnedWindow),
    /// The application's, kept by its `Arc`.
    Given(#[allow(dead_code)] Arc<dyn Any + Send + Sync>),
}

struct PresentState {
    swap_chain: IDXGISwapChain3,
    command_allocator: ID3D12CommandAllocator,
    command_list: ID3D12GraphicsCommandList,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_size: usize,
    render_targets: Vec<ID3D12Resource>,
    srv_heap: ID3D12DescriptorHeap,
    srv_size: usize,
    /// The fence value the last frame submitted here signals — waited on
    /// before anything that frame may still be using is touched again.
    last_submitted: u64,
    /// What kept the last frame's texture alive, held until the wait above
    /// says the GPU is done reading it.
    pending_keep_alive: Option<Box<dyn Any + Send>>,
    width: u32,
    height: u32,
    /// The textures and upload buffer system-memory frames are drawn
    /// through, made for the layout and size of the last one and remade when
    /// either changes.
    system: Option<SystemPlanes>,
    /// When the frame being drawn was handed over, in performance-counter
    /// ticks — what a measured presentation delay counts from.
    handed: i64,
    /// What a picture takes to reach the screen, from the swap chain's frame
    /// statistics.
    timing: PresentTiming,
}

/// What system-memory frames of one layout and size are drawn through.
struct SystemPlanes {
    layout: SystemLayout,
    size: (u32, u32),
    planes: Vec<SystemPlane>,
    /// Every plane's rows, each at its placed footprint, copied into its
    /// texture as part of the draw.
    upload: ID3D12Resource,
}

/// One plane: its texture, between frames in `COPY_DEST`, and where its rows
/// sit in the upload buffer.
struct SystemPlane {
    shape: PlaneShape,
    texture: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

/// What a draw reads from.
enum DrawSource<'a> {
    /// A D3D12 frame's own NV12 texture.
    Texture(&'a D3d12Picture),
    /// A system-memory frame, its planes staged into `PresentState::system`.
    System(SystemLayout),
}

/// The swap chain, pipeline state and fence a window is drawn with — the
/// [`D3d12FrameRenderer`] a [`D3d12WindowRenderer`] is built around.
struct WindowPresenter {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    root_signature: ID3D12RootSignature,
    nv12_pipeline: ID3D12PipelineState,
    yuv420p_pipeline: ID3D12PipelineState,
    bgra_pipeline: ID3D12PipelineState,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    hwnd: isize,
    /// Set once a call reports the device removed, after which every draw
    /// fails fast rather than touching the GPU again.
    device_lost: AtomicBool,
    state: Mutex<PresentState>,
    /// The centre pixel of the last frame drawn, R, G, B, A — copied out of
    /// the back buffer before it is presented, for a test to check the
    /// colour it came out.
    #[cfg(test)]
    probe: Arc<Mutex<Option<[u8; 4]>>>,
    /// Where that pixel is copied to: one row of a readback buffer.
    #[cfg(test)]
    readback: ID3D12Resource,
    /// Last, so the swap chain above is released before the window goes.
    _keep: Keep,
}

// SAFETY: COM interfaces used from whichever thread the pipeline runs this
// sink on; the command list, allocator and swap chain are only touched under
// `state`'s lock, and the window handle and fence event are only read.
unsafe impl Send for WindowPresenter {}
// SAFETY: as for `Send`; every mutable part is behind a lock or an atomic.
unsafe impl Sync for WindowPresenter {}

impl WindowPresenter {
    fn new(
        gpu: &D3d12Gpu,
        hwnd: HWND,
        keep: Keep,
    ) -> std::result::Result<Self, D3d12WindowRendererError> {
        let present = D3d12WindowRendererError::Present;
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let (width, height) = client_size(hwnd).unwrap_or((1, 1));
        let (width, height) = (width.max(1), height.max(1));
        // SAFETY: COM calls on a live device and a window that `keep` keeps
        // alive; every out-pointer and description is a local.
        unsafe {
            let factory: IDXGIFactory2 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(present)?;
            let swap_chain: IDXGISwapChain3 = factory
                .CreateSwapChainForHwnd(
                    &queue,
                    hwnd,
                    &DXGI_SWAP_CHAIN_DESC1 {
                        Width: width,
                        Height: height,
                        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        BufferCount: FRAME_COUNT as u32,
                        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
                        ..Default::default()
                    },
                    None,
                    None,
                )
                .and_then(|chain| chain.cast())
                .map_err(present)?;
            factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)
                .map_err(present)?;

            let rtv_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: FRAME_COUNT as u32,
                    ..Default::default()
                })
                .map_err(present)?;
            let rtv_size =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
            let render_targets = create_render_targets(&device, &swap_chain, &rtv_heap, rtv_size)
                .map_err(present)?;
            let srv_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    // A frame's planes: three at most, YUV420P's.
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(present)?;
            let srv_size = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let command_allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(present)?;
            let command_list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocator, None)
                .map_err(present)?;
            // Created recording; closed once so the first frame's `Reset`
            // finds it in the state that expects.
            command_list.Close().map_err(present)?;

            let vertex = compile_shader(
                FRAME_SHADER,
                s!("present_frame.hlsl"),
                s!("vs_main"),
                s!("vs_5_1"),
            )
            .map_err(present)?;
            let nv12 = compile_shader(
                NV12_SHADER,
                s!("present_nv12.hlsl"),
                s!("ps_nv12"),
                s!("ps_5_1"),
            )
            .map_err(present)?;
            // The root signature is the one the vertex shader declares.
            let root_blob = D3DGetBlobPart(
                vertex.GetBufferPointer(),
                vertex.GetBufferSize(),
                D3D_BLOB_ROOT_SIGNATURE,
                0,
            )
            .map_err(present)?;
            let root_signature: ID3D12RootSignature = device
                .CreateRootSignature(0, blob_bytes(&root_blob))
                .map_err(present)?;
            let nv12_pipeline =
                create_pipeline(&device, &root_signature, &vertex, &nv12).map_err(present)?;
            let yuv420p = compile_shader(
                YUV420P_SHADER,
                s!("present_yuv420p.hlsl"),
                s!("ps_yuv420p"),
                s!("ps_5_1"),
            )
            .map_err(present)?;
            let yuv420p_pipeline =
                create_pipeline(&device, &root_signature, &vertex, &yuv420p).map_err(present)?;
            let bgra = compile_shader(
                BGRA_SHADER,
                s!("present_bgra.hlsl"),
                s!("ps_bgra"),
                s!("ps_5_1"),
            )
            .map_err(present)?;
            let bgra_pipeline =
                create_pipeline(&device, &root_signature, &vertex, &bgra).map_err(present)?;

            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(present)?;
            let fence_event = CreateEventW(None, false, false, None).map_err(present)?;
            #[cfg(test)]
            let readback = create_buffer(
                &device,
                D3D12_HEAP_TYPE_READBACK,
                u64::from(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT),
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .map_err(present)?;

            Ok(Self {
                device,
                queue,
                root_signature,
                nv12_pipeline,
                yuv420p_pipeline,
                bgra_pipeline,
                fence,
                fence_event,
                hwnd: hwnd.0 as isize,
                device_lost: AtomicBool::new(false),
                state: Mutex::new(PresentState {
                    swap_chain,
                    command_allocator,
                    command_list,
                    rtv_heap,
                    rtv_size,
                    render_targets,
                    srv_heap,
                    srv_size,
                    last_submitted: 0,
                    pending_keep_alive: None,
                    width,
                    height,
                    system: None,
                    handed: 0,
                    timing: PresentTiming::new(),
                }),
                #[cfg(test)]
                probe: Arc::default(),
                #[cfg(test)]
                readback,
                _keep: keep,
            })
        }
    }

    /// Records a copy of the back buffer's centre pixel into
    /// [`Self::readback`], leaving `target` in the state it returns.
    ///
    /// # Safety
    ///
    /// `list` is open, and `target` is the back buffer it has just drawn
    /// into, in `RENDER_TARGET`.
    #[cfg(test)]
    unsafe fn copy_probe(
        &self,
        list: &ID3D12GraphicsCommandList,
        target: &ID3D12Resource,
        state: &PresentState,
    ) -> D3D12_RESOURCE_STATES {
        let (x, y) = (state.width / 2, state.height / 2);
        // SAFETY: the caller's promise; the descriptions are locals, and the
        // references they take are released right after the call.
        unsafe {
            transition(
                list,
                target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let mut destination = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(self.readback.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: D3D12_TEXTURE_DATA_PITCH_ALIGNMENT,
                        },
                    },
                },
            };
            let mut source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(target.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(
                &destination,
                0,
                0,
                0,
                &source,
                Some(&D3D12_BOX {
                    left: x,
                    top: y,
                    front: 0,
                    right: x + 1,
                    bottom: y + 1,
                    back: 1,
                }),
            );
            ManuallyDrop::drop(&mut destination.pResource);
            ManuallyDrop::drop(&mut source.pResource);
        }
        D3D12_RESOURCE_STATE_COPY_SOURCE
    }

    /// Reads what [`Self::copy_probe`] copied, once the frame is done.
    #[cfg(test)]
    fn read_probe(&self, state: &PresentState) {
        if self.wait_for(state.last_submitted).is_err() {
            return;
        }
        let mut data = std::ptr::null_mut();
        // SAFETY: the readback buffer is this presenter's own, and the GPU's
        // write into it has finished; the four bytes read are inside it.
        unsafe {
            if self.readback.Map(0, None, Some(&mut data)).is_ok() {
                let bgra = std::slice::from_raw_parts(data.cast::<u8>(), 4);
                *self.probe.lock().unwrap() = Some([bgra[2], bgra[1], bgra[0], bgra[3]]);
                self.readback.Unmap(0, None);
            }
        }
    }

    /// Waits until the frame that signalled `value` has finished on the GPU.
    fn wait_for(&self, value: u64) -> windows::core::Result<()> {
        if value == 0 {
            return Ok(());
        }
        // SAFETY: the fence and its event are this presenter's own and live.
        unsafe {
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn checked<T>(&self, result: windows::core::Result<T>) -> std::result::Result<T, SubmitError> {
        result.map_err(|error| {
            if matches!(
                error.code(),
                DXGI_ERROR_DEVICE_REMOVED | DXGI_ERROR_DEVICE_RESET | DXGI_ERROR_DEVICE_HUNG
            ) {
                self.device_lost.store(true, Ordering::Relaxed);
                SubmitError::DeviceRemoved
            } else {
                SubmitError::RenderFailed
            }
        })
    }

    /// Brings the swap chain to `width`x`height`. The caller has waited for
    /// the last frame, so nothing still reads the old back buffers.
    fn resize_to(
        &self,
        state: &mut PresentState,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError> {
        state.render_targets.clear();
        // SAFETY: the swap chain and heaps are only touched under `state`'s
        // lock, which the caller holds.
        unsafe {
            self.checked(state.swap_chain.ResizeBuffers(
                FRAME_COUNT as u32,
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_SWAP_CHAIN_FLAG(0),
            ))?;
            state.render_targets = self.checked(create_render_targets(
                &self.device,
                &state.swap_chain,
                &state.rtv_heap,
                state.rtv_size,
            ))?;
        }
        state.width = width;
        state.height = height;
        state.timing.restart();
        Ok(())
    }

    /// Records the frame's draw — letterboxed, after a GPU wait on a D3D12
    /// frame's own fence, or after copying a system-memory frame's staged
    /// planes into their textures — executes it, presents, and signals this
    /// presenter's fence.
    ///
    /// # Safety
    ///
    /// A [`DrawSource::Texture`]'s texture is an NV12 resource on this
    /// presenter's device, in the `COMMON` state, fully written once its
    /// fence reaches its value. A [`DrawSource::System`] has had its planes
    /// staged into `state.system` by [`Self::stage`].
    unsafe fn draw(
        &self,
        state: &mut PresentState,
        source: DrawSource<'_>,
        (frame_width, frame_height): (u32, u32),
        rows: Option<&[[f32; 4]; 3]>,
    ) -> windows::core::Result<()> {
        let unexpected = || windows::core::Error::from(windows::Win32::Foundation::E_UNEXPECTED);
        // SAFETY: as the caller promises for the source; everything else is
        // this presenter's own, under `state`'s lock.
        unsafe {
            // The table's three slots: the frame's planes, and a null view
            // wherever its layout has none — so every slot is initialized.
            let mut views: [Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>; 3] =
                Default::default();
            let pipeline = match &source {
                DrawSource::Texture(picture) => {
                    views[0] = Some((
                        picture.texture.clone(),
                        plane_srv_desc(DXGI_FORMAT_R8_UNORM, 0),
                    ));
                    views[1] = Some((
                        picture.texture.clone(),
                        plane_srv_desc(DXGI_FORMAT_R8G8_UNORM, 1),
                    ));
                    &self.nv12_pipeline
                }
                DrawSource::System(layout) => {
                    let system = state.system.as_ref().ok_or_else(unexpected)?;
                    for (slot, plane) in views.iter_mut().zip(&system.planes) {
                        *slot =
                            Some((plane.texture.clone(), plane_srv_desc(plane.shape.format, 0)));
                    }
                    match layout {
                        SystemLayout::Nv12 => &self.nv12_pipeline,
                        SystemLayout::Yuv420p => &self.yuv420p_pipeline,
                        SystemLayout::Bgra => &self.bgra_pipeline,
                    }
                }
            };
            for (index, view) in views.iter().enumerate() {
                let handle = srv_handle(&state.srv_heap, state.srv_size, index);
                match view {
                    Some((resource, desc)) => {
                        self.device
                            .CreateShaderResourceView(resource, Some(desc), handle)
                    }
                    None => self.device.CreateShaderResourceView(
                        None::<&ID3D12Resource>,
                        Some(&plane_srv_desc(DXGI_FORMAT_R8_UNORM, 0)),
                        handle,
                    ),
                }
            }
            state.command_allocator.Reset()?;
            state.command_list.Reset(&state.command_allocator, None)?;
            let list = &state.command_list;
            match &source {
                DrawSource::Texture(picture) => transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ),
                DrawSource::System(_) => {
                    let system = state.system.as_ref().ok_or_else(unexpected)?;
                    for plane in &system.planes {
                        copy_plane(list, &system.upload, &plane.footprint, &plane.texture);
                        transition(
                            list,
                            &plane.texture,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        );
                    }
                }
            }

            let index = state.swap_chain.GetCurrentBackBufferIndex() as usize;
            let target = state.render_targets[index].clone();
            let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: state.rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr
                    + index * state.rtv_size,
            };
            transition(
                list,
                &target,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            list.SetPipelineState(pipeline);
            list.SetGraphicsRootSignature(&self.root_signature);
            list.SetDescriptorHeaps(&[Some(state.srv_heap.clone())]);
            list.SetGraphicsRootDescriptorTable(
                0,
                state.srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            // The three colour rows, as the root signature's twelve constants.
            if let Some(rows) = rows {
                list.SetGraphicsRoot32BitConstants(1, 12, rows.as_ptr().cast(), 0);
            }
            list.RSSetViewports(&[letterbox(
                frame_width,
                frame_height,
                state.width,
                state.height,
            )]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: state.width as i32,
                bottom: state.height as i32,
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
            match &source {
                // Back to `COMMON` before closing: the texture is its
                // producer's again once this list has run.
                DrawSource::Texture(picture) => transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                // Back to `COPY_DEST`, where the next frame's copy expects them.
                DrawSource::System(_) => {
                    let system = state.system.as_ref().ok_or_else(unexpected)?;
                    for plane in &system.planes {
                        transition(
                            list,
                            &plane.texture,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        );
                    }
                }
            }
            #[cfg(test)]
            let target_state = self.copy_probe(list, &target, state);
            #[cfg(not(test))]
            let target_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
            transition(list, &target, target_state, D3D12_RESOURCE_STATE_PRESENT);
            list.Close()?;

            // A GPU-side wait for whatever wrote the texture: sharing a device
            // does not order this queue after the decoder's or uploader's.
            if let DrawSource::Texture(picture) = &source {
                self.queue.Wait(&picture.fence, picture.fence_value)?;
            }
            self.queue
                .ExecuteCommandLists(&[Some(state.command_list.cast()?)]);
            // Presented before returning: a preroll counts this return as the
            // frame being on its way to the screen.
            state.swap_chain.Present(1, DXGI_PRESENT(0)).ok()?;
            let value = state.last_submitted + 1;
            self.queue.Signal(&self.fence, value)?;
            state.last_submitted = value;
        }
        state.timing.presented(&state.swap_chain, state.handed);
        Ok(())
    }

    /// What every frame waits for before it is drawn: the device still
    /// there, the last frame done on the GPU — so its heaps, list, back
    /// buffer and staging are free again, and its texture can be let go —
    /// and the swap chain at the window's size. `None` for a minimised
    /// window, which has nothing to draw into until it comes back.
    fn begin(
        &self,
        width: u32,
        height: u32,
    ) -> std::result::Result<Option<MutexGuard<'_, PresentState>>, SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }
        // What a measured presentation delay counts from: the moment the frame
        // is handed over, which is what a synchronizer schedules.
        let handed = PresentTiming::now();
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        state.handed = handed;
        self.checked(self.wait_for(state.last_submitted))?;
        state.pending_keep_alive = None;
        match client_size(HWND(self.hwnd as *mut c_void)) {
            Some((0, _)) | Some((_, 0)) => return Ok(None),
            Some((width, height)) if (width, height) != (state.width, state.height) => {
                self.resize_to(&mut state, width, height)?;
            }
            _ => {}
        }
        Ok(Some(state))
    }

    /// Draws one checked D3D12 frame, converted with `rows`, and holds
    /// `keep_alive` until the GPU is done reading its texture.
    ///
    /// # Safety
    ///
    /// `picture` is an NV12 texture on this presenter's device, in `COMMON`,
    /// with the fence its producer signals once it is written.
    unsafe fn show(
        &self,
        picture: D3d12Picture,
        (width, height): (u32, u32),
        rows: [[f32; 4]; 3],
        keep_alive: Box<dyn Any + Send>,
    ) -> std::result::Result<(), SubmitError> {
        let Some(mut state) = self.begin(width, height)? else {
            return Ok(());
        };
        // SAFETY: the caller's promise for the texture and its fence.
        self.checked(unsafe {
            self.draw(
                &mut state,
                DrawSource::Texture(&picture),
                (width, height),
                Some(&rows),
            )
        })?;
        state.pending_keep_alive = Some(keep_alive);
        #[cfg(test)]
        self.read_probe(&state);
        Ok(())
    }

    /// Draws a system-memory frame: its planes copied into an upload buffer
    /// here, and from there into textures on the GPU as part of the draw.
    /// Nothing of the frame is kept once this returns.
    fn show_system(
        &self,
        frame: &ffmpeg::frame::Video,
        layout: SystemLayout,
    ) -> std::result::Result<(), SubmitError> {
        let (width, height) = (frame.width(), frame.height());
        let planes = planes_of(frame, layout).ok_or(SubmitError::InvalidFrame)?;
        let Some(mut state) = self.begin(width, height)? else {
            return Ok(());
        };
        self.stage(&mut state, layout, (width, height), &planes)?;
        let rows = (layout != SystemLayout::Bgra).then(|| colour_rows(frame));
        // SAFETY: the planes were staged just above, into `state.system`.
        self.checked(unsafe {
            self.draw(
                &mut state,
                DrawSource::System(layout),
                (width, height),
                rows.as_ref(),
            )
        })?;
        #[cfg(test)]
        self.read_probe(&state);
        Ok(())
    }

    /// Copies `planes` into the upload buffer, at the placed footprints the
    /// copy into each plane's texture reads — making the buffer and the
    /// textures first when the layout or size is new. The last frame is done
    /// on the GPU (see [`Self::begin`]), so none of it is in use.
    fn stage(
        &self,
        state: &mut PresentState,
        layout: SystemLayout,
        size: (u32, u32),
        planes: &[(&[u8], usize)],
    ) -> std::result::Result<(), SubmitError> {
        let current = state
            .system
            .as_ref()
            .is_some_and(|system| system.layout == layout && system.size == size);
        if !current {
            state.system = None;
            state.system = Some(self.checked(self.system_planes(layout, size))?);
        }
        let system = state.system.as_ref().ok_or(SubmitError::RenderFailed)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the upload buffer is this presenter's own and not in use;
        // every write lands inside the footprint its plane was given, which
        // `GetCopyableFootprints` sized for exactly these rows.
        unsafe {
            self.checked(system.upload.Map(0, None, Some(&mut mapped)))?;
            let base = mapped.cast::<u8>();
            for (plane, (data, stride)) in system.planes.iter().zip(planes) {
                let footprint = &plane.footprint;
                let pitch = footprint.Footprint.RowPitch as usize;
                let row_bytes = plane.shape.row_bytes();
                for row in 0..plane.shape.height as usize {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(row * stride),
                        base.add(footprint.Offset as usize + row * pitch),
                        row_bytes,
                    );
                }
            }
            system.upload.Unmap(0, None);
        }
        Ok(())
    }

    /// Textures for each plane of a `layout` frame of `size`, left in
    /// `COPY_DEST`, and one upload buffer holding them all.
    fn system_planes(
        &self,
        layout: SystemLayout,
        (width, height): (u32, u32),
    ) -> windows::core::Result<SystemPlanes> {
        let mut planes = Vec::new();
        let mut offset = 0u64;
        for shape in layout.planes(width, height) {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(shape.width),
                Height: shape.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: shape.format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            // SAFETY: a texture on this presenter's device and a query of the
            // footprint its one subresource needs, into locals.
            unsafe {
                self.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut texture,
                )?;
                self.device.GetCopyableFootprints(
                    &desc,
                    0,
                    1,
                    offset,
                    Some(&mut footprint),
                    None,
                    None,
                    Some(&mut total),
                );
            }
            let texture = texture
                .ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_POINTER))?;
            offset = (footprint.Offset + total)
                .next_multiple_of(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
            planes.push(SystemPlane {
                shape,
                texture,
                footprint,
            });
        }
        let upload = create_buffer(
            &self.device,
            D3D12_HEAP_TYPE_UPLOAD,
            offset,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        Ok(SystemPlanes {
            layout,
            size: (width, height),
            planes,
            upload,
        })
    }
}

impl Drop for WindowPresenter {
    fn drop(&mut self) {
        if let Ok(state) = self.state.lock() {
            let _ = self.wait_for(state.last_submitted);
        }
        // SAFETY: the event this presenter created, closed once.
        unsafe {
            let _ = CloseHandle(self.fence_event);
        }
    }
}

/// The window's client area, or `None` if it cannot be read (a window gone).
fn client_size(hwnd: HWND) -> Option<(u32, u32)> {
    let mut rect = RECT::default();
    // SAFETY: reads a window's client rectangle into a local.
    unsafe { GetClientRect(hwnd, &mut rect) }.ok()?;
    Some((
        (rect.right - rect.left).max(0) as u32,
        (rect.bottom - rect.top).max(0) as u32,
    ))
}

/// # Safety
///
/// A live blob.
unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: a blob owns `GetBufferSize()` readable bytes at
    // `GetBufferPointer()` for as long as it lives.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

unsafe fn create_render_targets(
    device: &ID3D12Device,
    swap_chain: &IDXGISwapChain3,
    heap: &ID3D12DescriptorHeap,
    size: usize,
) -> windows::core::Result<Vec<ID3D12Resource>> {
    // SAFETY: views of the swap chain's own buffers in this presenter's heap.
    unsafe {
        let start = heap.GetCPUDescriptorHandleForHeapStart();
        (0..FRAME_COUNT)
            .map(|index| {
                let buffer: ID3D12Resource = swap_chain.GetBuffer(index as u32)?;
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + index * size,
                };
                device.CreateRenderTargetView(&buffer, None, handle);
                Ok(buffer)
            })
            .collect()
    }
}

unsafe fn create_pipeline(
    device: &ID3D12Device,
    root_signature: &ID3D12RootSignature,
    vertex: &ID3DBlob,
    pixel: &ID3DBlob,
) -> windows::core::Result<ID3D12PipelineState> {
    let blend = D3D12_RENDER_TARGET_BLEND_DESC {
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        ..Default::default()
    };
    let mut formats = [DXGI_FORMAT::default(); 8];
    formats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    // SAFETY: the blobs outlive this call, and the root signature reference
    // the description takes is released right after it.
    unsafe {
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vertex.GetBufferPointer(),
                BytecodeLength: vertex.GetBufferSize(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: pixel.GetBufferPointer(),
                BytecodeLength: pixel.GetBufferSize(),
            },
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [blend; 8],
                ..Default::default()
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: formats,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let pipeline = device.CreateGraphicsPipelineState(&desc);
        ManuallyDrop::drop(&mut desc.pRootSignature);
        pipeline
    }
}

/// A buffer of `size` bytes in a heap of `heap_type`, starting in `state`.
fn create_buffer(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    size: u64,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a committed resource on a live device, from local descriptions.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            state,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_POINTER))
}

/// Records a copy of one plane's rows, at `footprint` in `upload`, into
/// `texture`.
///
/// # Safety
///
/// `list` is open, `texture` is in `COPY_DEST`, and `footprint` is the one
/// `GetCopyableFootprints` gave for it inside `upload`.
unsafe fn copy_plane(
    list: &ID3D12GraphicsCommandList,
    upload: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    texture: &ID3D12Resource,
) {
    let mut destination = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let mut source = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(upload.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *footprint,
        },
    };
    // SAFETY: the caller's promise; the references the descriptions took are
    // released right after the call.
    unsafe {
        list.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
        ManuallyDrop::drop(&mut destination.pResource);
        ManuallyDrop::drop(&mut source.pResource);
    }
}

fn srv_handle(
    heap: &ID3D12DescriptorHeap,
    size: usize,
    index: usize,
) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    // SAFETY: reads the start of a live heap.
    let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: start.ptr + index * size,
    }
}

/// One plane of an NV12 texture: the format picks which, the plane slice
/// says where.
fn plane_srv_desc(format: DXGI_FORMAT, plane_slice: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: plane_slice,
                ResourceMinLODClamp: 0.0,
            },
        },
    }
}

unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: records one barrier on an open list, then releases the
    // resource reference the barrier description took.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let transition = &mut *barrier.Anonymous.Transition;
        ManuallyDrop::drop(&mut transition.pResource);
    }
}

/// The largest rectangle of the frame's aspect ratio that fits the window,
/// centred in it.
fn letterbox(
    frame_width: u32,
    frame_height: u32,
    window_width: u32,
    window_height: u32,
) -> D3D12_VIEWPORT {
    let scale =
        (window_width as f32 / frame_width as f32).min(window_height as f32 / frame_height as f32);
    let (width, height) = (frame_width as f32 * scale, frame_height as f32 * scale);
    D3D12_VIEWPORT {
        TopLeftX: (window_width as f32 - width) * 0.5,
        TopLeftY: (window_height as f32 - height) * 0.5,
        Width: width,
        Height: height,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_the_aspect_ratio_and_centres_it() {
        let near = |view: D3D12_VIEWPORT, expected: [f32; 4]| {
            let got = [view.TopLeftX, view.TopLeftY, view.Width, view.Height];
            assert!(
                got.iter()
                    .zip(expected)
                    .all(|(got, expected)| (got - expected).abs() < 0.01),
                "{got:?} is not {expected:?}"
            );
        };
        // A wide picture in a square window: full width, bars above and below.
        near(
            letterbox(1920, 1080, 1000, 1000),
            [0.0, 218.75, 1000.0, 562.5],
        );
        // A tall picture in a wide window: full height, bars at the sides.
        near(
            letterbox(720, 1280, 1000, 500),
            [359.375, 0.0, 281.25, 500.0],
        );
        // A window of the picture's own shape: it fills it.
        near(letterbox(1280, 720, 640, 360), [0.0, 0.0, 640.0, 360.0]);
    }

    use std::{num::NonZeroIsize, time::Duration};

    use raw_window_handle::{HandleError, Win32WindowHandle, WindowHandle};

    use crate::{
        bus::BusEvent,
        elements::{
            AppSource, D3d12Upload, D3d12UploadError, SwScaler, TestVideoOptions, TestVideoSource,
            WindowEvent,
        },
        pipeline::Pipeline,
    };

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;

    fn gpu() -> Option<D3d12Gpu> {
        D3d12Gpu::new()
            .inspect_err(|error| eprintln!("skipping: no Direct3D 12 device here ({error})"))
            .ok()
    }

    /// What puts a test picture on the GPU — which only a device that does
    /// video has: FFmpeg's D3D12 device context wants one, and a software
    /// adapter such as WARP, all a CI runner has, is not.
    fn upload(gpu: &D3d12Gpu) -> Option<D3d12Upload> {
        match D3d12Upload::new("upload", gpu) {
            Ok(upload) => Some(upload),
            Err(D3d12UploadError::HwDeviceInit(code)) => {
                eprintln!("skipping: this device does not do video (code {code})");
                None
            }
            Err(error) => panic!("the upload did not open: {error}"),
        }
    }

    /// Plays a test picture into `renderer` for half a second, and returns
    /// how many frames it took, every error the bus carried, and the
    /// presentation delay the renderer gave the pipeline's playback clock.
    fn show(upload: D3d12Upload, renderer: D3d12WindowRenderer) -> (u64, Vec<BusEvent>, Duration) {
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: WIDTH,
                height: HEIGHT,
                frame_rate: ffmpeg_next::Rational::new(30, 1),
            },
        );
        let (pipeline, ()) = Pipeline::new("window-renderer", source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(SwScaler::to_format(
                    "to-nv12",
                    ffmpeg_next::format::Pixel::NV12,
                    ffmpeg_next::software::scaling::Flags::BILINEAR,
                ))
                .pipe(upload)
                .queue("to-screen", 4)
                .to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pipeline.run().expect("run");
        std::thread::sleep(Duration::from_millis(500));
        let shown = pipeline
            .stats()
            .elements
            .iter()
            .find(|element| element.element_type == ElementType::D3d12WindowRenderer)
            .map_or(0, |element| element.buffers_in);
        let delay = pipeline.playback_clock().presentation_delay();
        pipeline.stop();
        let errors = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        (shown, errors, delay)
    }

    /// Opened on its own window, it shows what it is given, with nothing on
    /// the bus but the pipeline's own; the window reports its size, and once
    /// the renderer is gone so is the window, and the events with it.
    #[test]
    fn a_window_of_its_own_shows_the_frames() {
        let Some(gpu) = gpu() else { return };
        let Some(upload) = upload(&gpu) else { return };
        let (renderer, events) = match D3d12WindowRenderer::open(
            "screen",
            &gpu,
            WindowOptions {
                title: "media-pp window renderer test".into(),
                width: WIDTH,
                height: HEIGHT,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: no window can be opened here ({error})");
                return;
            }
        };
        let control = renderer
            .window_control()
            .expect("a window of its own is its to change");
        control
            .set_title("media-pp window renderer test, renamed")
            .expect("the window is there");
        let (shown, errors, delay) = show(upload, renderer);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(shown >= 5, "only {shown} frames reached the window");
        // Measured from the swap chain's frame statistics, or estimated from
        // the desktop's refresh — where there is a desktop to say it.
        if super::super::present_timing::desktop_refresh().is_some() {
            assert!(
                delay > Duration::ZERO && delay < Duration::from_millis(200),
                "the renderer says a picture takes {delay:?} to reach the screen"
            );
        }
        assert!(
            std::iter::from_fn(|| events.try_recv())
                .any(|event| matches!(event, WindowEvent::Resized { .. })),
            "the window says how big it is"
        );
        assert_eq!(events.recv(), None, "gone with the renderer");
    }

    /// A window the renderer was given, as an application's `winit` window
    /// would be: drawn into, and kept alive by the renderer's `Arc` until it
    /// is done.
    struct GivenWindow(OwnedWindow);

    impl HasWindowHandle for GivenWindow {
        fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
            let hwnd =
                NonZeroIsize::new(self.0.hwnd().0 as isize).ok_or(HandleError::Unavailable)?;
            // SAFETY: the handle of a window `self` keeps open.
            Ok(unsafe {
                WindowHandle::borrow_raw(RawWindowHandle::Win32(Win32WindowHandle::new(hwnd)))
            })
        }
    }

    #[test]
    fn a_window_it_is_given_shows_the_frames() {
        let Some(gpu) = gpu() else { return };
        let Some(upload) = upload(&gpu) else { return };
        let (events_tx, _events) = crossbeam_channel::unbounded();
        let options = WindowOptions {
            title: "media-pp given window test".into(),
            width: WIDTH,
            height: HEIGHT,
        };
        let window = match OwnedWindow::open(&options, events_tx) {
            Ok(window) => Arc::new(GivenWindow(window)),
            Err(error) => {
                eprintln!("skipping: no window can be opened here ({error})");
                return;
            }
        };
        let renderer = D3d12WindowRenderer::for_window("screen", &gpu, Arc::clone(&window))
            .expect("draw into the given window");
        assert_eq!(
            Arc::strong_count(&window),
            2,
            "the renderer holds the window"
        );
        assert!(
            renderer.window_control().is_none(),
            "a window it was given is the application's"
        );
        let (shown, errors, _) = show(upload, renderer);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(shown >= 5, "only {shown} frames reached the window");
    }

    /// No area and no Win32 handle are refused before anything is opened.
    #[test]
    fn a_window_with_nothing_to_draw_in_is_refused() {
        let Some(gpu) = gpu() else { return };
        assert!(matches!(
            D3d12WindowRenderer::open(
                "screen",
                &gpu,
                WindowOptions {
                    width: 0,
                    ..WindowOptions::default()
                },
            ),
            Err(D3d12WindowRendererError::EmptyWindow { width: 0, .. })
        ));

        struct Headless;
        impl HasWindowHandle for Headless {
            fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
                Err(HandleError::Unavailable)
            }
        }
        assert!(matches!(
            D3d12WindowRenderer::for_window("screen", &gpu, Arc::new(Headless)),
            Err(D3d12WindowRendererError::NotAWin32Window)
        ));
    }

    /// A solid NV12 picture of one Y'CbCr value, tagged with `space`, limited
    /// range.
    fn solid_nv12(ycbcr: [u8; 3], space: ffmpeg_next::color::Space) -> MediaBuffer {
        let mut frame = ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::NV12, 64, 64);
        frame.data_mut(0).fill(ycbcr[0]);
        for pair in frame.data_mut(1).as_chunks_mut::<2>().0 {
            pair.copy_from_slice(&ycbcr[1..]);
        }
        frame.set_color_space(space);
        frame.set_color_range(ffmpeg_next::color::Range::MPEG);
        frame.set_pts(Some(0));
        MediaBuffer::video(frame)
    }

    /// A solid YUV420P picture of one Y'CbCr value at an odd size, tagged
    /// with `space` — so its chroma planes round up, and FFmpeg pads its
    /// rows past the picture's width.
    fn solid_yuv420p(ycbcr: [u8; 3], space: ffmpeg_next::color::Space) -> MediaBuffer {
        let mut frame = ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::YUV420P, 63, 35);
        for (plane, value) in ycbcr.into_iter().enumerate() {
            frame.data_mut(plane).fill(value);
        }
        frame.set_color_space(space);
        frame.set_color_range(ffmpeg_next::color::Range::MPEG);
        frame.set_pts(Some(0));
        MediaBuffer::video(frame)
    }

    /// A solid BGRA picture of one R'G'B' value.
    fn solid_bgra([r, g, b]: [u8; 3]) -> MediaBuffer {
        let mut frame = ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::BGRA, 64, 64);
        for pixel in frame.data_mut(0).as_chunks_mut::<4>().0 {
            *pixel = [b, g, r, 255];
        }
        frame.set_pts(Some(0));
        MediaBuffer::video(frame)
    }

    /// Draws `frame` — through `upload` first, where there is one — and
    /// reads back the centre of the window.
    fn drawn(gpu: &D3d12Gpu, upload: Option<D3d12Upload>, frame: MediaBuffer) -> Option<[u8; 3]> {
        let (renderer, _events) = match D3d12WindowRenderer::open(
            "screen",
            gpu,
            WindowOptions {
                title: "media-pp window colour test".into(),
                width: WIDTH,
                height: HEIGHT,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: no window can be opened here ({error})");
                return None;
            }
        };
        let probe = renderer.probe();
        let (source, handle) = AppSource::new("frames", 4);
        let (pipeline, ()) = Pipeline::new("window-colour", source, |source, ctx| {
            let branch = match upload {
                Some(upload) => ctx.branch().pipe(upload).to(renderer)?,
                None => ctx.branch().to(renderer)?,
            };
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pipeline.run().expect("run");
        handle.push(frame).expect("push");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while probe.lock().unwrap().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        pipeline.stop();
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
        let [r, g, b, _] = probe.lock().unwrap().expect("the frame was drawn");
        Some([r, g, b])
    }

    fn near(got: [u8; 3], expected: [u8; 3]) -> bool {
        got.iter()
            .zip(expected)
            .all(|(&got, expected)| got.abs_diff(expected) <= 3)
    }

    /// Each frame is converted with its own matrix. (72, 107, 220) is R'G'B'
    /// (230, 20, 20) in BT.709; tagged BT.709 it comes out as that. Read
    /// with BT.601 — what this renderer used to do with every frame — the
    /// same numbers are (211, 0, 22), and a BT.601 frame of them is drawn as
    /// that.
    #[test]
    fn a_frame_is_drawn_with_its_own_colour_description() {
        use ffmpeg_next::color::Space;

        let Some(gpu) = gpu() else { return };
        let ycbcr = [72, 107, 220];
        let Some(first) = upload(&gpu) else { return };
        let Some(bt709) = drawn(&gpu, Some(first), solid_nv12(ycbcr, Space::BT709)) else {
            return;
        };
        assert!(near(bt709, [230, 20, 20]), "BT.709 drawn as {bt709:?}");
        let Some(second) = upload(&gpu) else { return };
        let Some(bt601) = drawn(&gpu, Some(second), solid_nv12(ycbcr, Space::BT470BG)) else {
            return;
        };
        assert!(near(bt601, [211, 0, 22]), "BT.601 drawn as {bt601:?}");
    }

    /// A frame in system memory is drawn as it comes — NV12, YUV420P, BGRA,
    /// no upload in front — each YUV one with its own colour description,
    /// and a YUV420P one at an odd size with its padded rows read by their
    /// stride. Needs no video support of the device, so it draws even on a
    /// software adapter.
    #[test]
    fn a_system_memory_frame_is_drawn_as_it_comes() {
        use ffmpeg_next::color::Space;

        let Some(gpu) = gpu() else { return };
        let ycbcr = [72, 107, 220];
        let Some(nv12) = drawn(&gpu, None, solid_nv12(ycbcr, Space::BT709)) else {
            return;
        };
        assert!(near(nv12, [230, 20, 20]), "NV12 drawn as {nv12:?}");
        let Some(yuv420p) = drawn(&gpu, None, solid_yuv420p(ycbcr, Space::BT709)) else {
            return;
        };
        assert!(near(yuv420p, [230, 20, 20]), "YUV420P drawn as {yuv420p:?}");
        let Some(bt601) = drawn(&gpu, None, solid_yuv420p(ycbcr, Space::BT470BG)) else {
            return;
        };
        assert!(
            near(bt601, [211, 0, 22]),
            "BT.601 YUV420P drawn as {bt601:?}"
        );
        let Some(bgra) = drawn(&gpu, None, solid_bgra([30, 140, 200])) else {
            return;
        };
        assert!(near(bgra, [30, 140, 200]), "BGRA drawn as {bgra:?}");
    }

    /// Frames of one layout and size after another reuse what the first was
    /// drawn through, and a new size is drawn through new textures — each
    /// still coming out as it should.
    #[test]
    fn frames_that_change_size_are_each_drawn() {
        use ffmpeg_next::color::Space;

        let Some(gpu) = gpu() else { return };
        let Ok((renderer, _events)) = D3d12WindowRenderer::open(
            "screen",
            &gpu,
            WindowOptions {
                width: WIDTH,
                height: HEIGHT,
                ..WindowOptions::default()
            },
        ) else {
            return;
        };
        let probe = renderer.probe();
        let (source, handle) = AppSource::new("frames", 8);
        let (pipeline, ()) = Pipeline::new("window-sizes", source, |source, ctx| {
            let branch = ctx.branch().to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pipeline.run().expect("run");
        for frame in [
            solid_nv12([72, 107, 220], Space::BT709),
            solid_nv12([72, 107, 220], Space::BT709),
            solid_yuv420p([72, 107, 220], Space::BT709),
            solid_bgra([30, 140, 200]),
        ] {
            *probe.lock().unwrap() = None;
            handle.push(frame).expect("push");
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while probe.lock().unwrap().is_none() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(probe.lock().unwrap().is_some(), "every frame is drawn");
        }
        let [r, g, b, _] = probe.lock().unwrap().expect("drawn");
        assert!(
            near([r, g, b], [30, 140, 200]),
            "the last drawn as {:?}",
            [r, g, b]
        );
        pipeline.stop();
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
    }
}
