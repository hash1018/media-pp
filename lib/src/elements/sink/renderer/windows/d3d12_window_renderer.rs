//! A D3D12 renderer that brings its own window, or draws into one it is
//! given — the part a program otherwise writes for itself behind
//! [`D3d12FrameRenderer`].

use std::{
    any::Any,
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

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

use crate::{
    buffer::MediaBuffer,
    contract::InputContract,
    control::ControlMsg,
    element::{Element, ElementType, Sink},
    elements::{
        D3d12FrameRenderer, D3d12Gpu, D3d12Renderer, SubmitError, WindowEvents, WindowOptions,
    },
    error::Result,
    platform::windows::{hlsl::compile_shader, window::OwnedWindow},
    pp_log::PpLog,
};

const FRAME_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_frame.hlsl");
const NV12_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/present_nv12.hlsl");

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
/// It is a [`D3d12Renderer`] with the window, swap chain and shaders built
/// in: frames go through the same checks — a texture from another device is
/// refused — and are drawn zero-copy from the NV12 textures `D3d12Decoder`
/// and `D3d12Upload` make, waiting on the GPU for each frame's own fence
/// before reading it. The picture keeps its aspect ratio inside the window,
/// with black bars as needed, and the renderer follows the window's size on
/// its own, reading it before every frame.
///
/// It draws; it does not pace. Put a [`crate::elements::VideoSynchronizer`]
/// or [`crate::elements::Pacer`] in front for a picture shown at its own
/// time. Each frame is presented synchronized to the display's refresh, and
/// one frame is in flight at a time: each waits for the one before it to
/// finish on the GPU.
///
/// Its GPU is the [`D3d12Gpu`] every other D3D12 element in the pipeline
/// shares — the frames it shows must come from that device.
pub struct D3d12WindowRenderer {
    inner: D3d12Renderer,
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
        let presenter = WindowPresenter::new(gpu, hwnd, Keep::Owned(window))?;
        Ok((Self::around(name, presenter), WindowEvents { events }))
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
        Ok(Self::around(name, presenter))
    }

    fn around(name: impl Into<String>, presenter: WindowPresenter) -> Self {
        Self {
            inner: D3d12Renderer::labelled(
                name,
                Box::new(presenter),
                ElementType::D3d12WindowRenderer,
            ),
        }
    }
}

impl Element for D3d12WindowRenderer {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }
}

impl Sink for D3d12WindowRenderer {
    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }

    fn ready_consume(&mut self) -> bool {
        self.inner.ready_consume()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.inner.consume(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.inner.control(msg)
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
}

/// The swap chain, pipeline state and fence a window is drawn with — the
/// [`D3d12FrameRenderer`] a [`D3d12WindowRenderer`] is built around.
struct WindowPresenter {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    root_signature: ID3D12RootSignature,
    nv12_pipeline: ID3D12PipelineState,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    hwnd: isize,
    /// Set once a call reports the device removed, after which every draw
    /// fails fast rather than touching the GPU again.
    device_lost: AtomicBool,
    state: Mutex<PresentState>,
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
                    NumDescriptors: 2,
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

            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(present)?;
            let fence_event = CreateEventW(None, false, false, None).map_err(present)?;

            Ok(Self {
                device,
                queue,
                root_signature,
                nv12_pipeline,
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
                }),
                _keep: keep,
            })
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
        Ok(())
    }

    /// Records the frame's draw — letterboxed, after a GPU wait on its own
    /// fence — executes it, presents, and signals this presenter's fence.
    ///
    /// # Safety
    ///
    /// `texture` is an NV12 resource on this presenter's device, in the
    /// `COMMON` state, fully written once `fence` reaches `fence_value`.
    unsafe fn draw(
        &self,
        state: &mut PresentState,
        texture: &ID3D12Resource,
        (fence, fence_value): (&ID3D12Fence, u64),
        (frame_width, frame_height): (u32, u32),
    ) -> windows::core::Result<()> {
        // SAFETY: as the caller promises for `texture`; everything else is
        // this presenter's own, under `state`'s lock.
        unsafe {
            let luma = srv_handle(&state.srv_heap, state.srv_size, 0);
            let chroma = srv_handle(&state.srv_heap, state.srv_size, 1);
            self.device.CreateShaderResourceView(
                texture,
                Some(&plane_srv_desc(DXGI_FORMAT_R8_UNORM, 0)),
                luma,
            );
            self.device.CreateShaderResourceView(
                texture,
                Some(&plane_srv_desc(DXGI_FORMAT_R8G8_UNORM, 1)),
                chroma,
            );
            state.command_allocator.Reset()?;
            state.command_list.Reset(&state.command_allocator, None)?;
            let list = &state.command_list;
            transition(
                list,
                texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );

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
            list.SetPipelineState(&self.nv12_pipeline);
            list.SetGraphicsRootSignature(&self.root_signature);
            list.SetDescriptorHeaps(&[Some(state.srv_heap.clone())]);
            list.SetGraphicsRootDescriptorTable(
                0,
                state.srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
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
            // Back to `COMMON` before closing: the texture is its producer's
            // again once this list has run.
            transition(
                list,
                texture,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            transition(
                list,
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            list.Close()?;

            // A GPU-side wait for whatever wrote the texture: sharing a device
            // does not order this queue after the decoder's or uploader's.
            self.queue.Wait(fence, fence_value)?;
            self.queue
                .ExecuteCommandLists(&[Some(state.command_list.cast()?)]);
            // Presented before returning: a preroll counts this return as the
            // frame being on its way to the screen.
            state.swap_chain.Present(1, DXGI_PRESENT(0)).ok()?;
            let value = state.last_submitted + 1;
            self.queue.Signal(&self.fence, value)?;
            state.last_submitted = value;
        }
        Ok(())
    }
}

impl D3d12FrameRenderer for WindowPresenter {
    fn device(&self) -> ID3D12Device {
        self.device.clone()
    }

    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D12Resource,
        fence: ID3D12Fence,
        fence_value: u64,
        width: u32,
        height: u32,
        keep_alive: Box<dyn Any + Send>,
    ) -> std::result::Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        // One frame in flight: the last one is done before its heaps, list
        // and back buffer are reused — and before its texture is let go.
        self.checked(self.wait_for(state.last_submitted))?;
        state.pending_keep_alive = None;
        // Follow the window: read its size before every frame. A minimised
        // window has no area, and there is nothing to draw into until it
        // comes back.
        match client_size(HWND(self.hwnd as *mut c_void)) {
            Some((0, _)) | Some((_, 0)) => return Ok(()),
            Some((width, height)) if (width, height) != (state.width, state.height) => {
                self.resize_to(&mut state, width, height)?;
            }
            _ => {}
        }
        // SAFETY: `D3d12Renderer` hands on an NV12 texture on this device, in
        // `COMMON`, with the fence its producer signals when it is written.
        self.checked(unsafe {
            self.draw(&mut state, &texture, (&fence, fence_value), (width, height))
        })?;
        state.pending_keep_alive = Some(keep_alive);
        Ok(())
    }

    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        self.checked(self.wait_for(state.last_submitted))?;
        state.pending_keep_alive = None;
        self.resize_to(&mut state, width, height)
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
            D3d12Upload, D3d12UploadError, SwScaler, TestVideoOptions, TestVideoSource, WindowEvent,
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
        match D3d12Upload::new("upload", gpu.device()) {
            Ok(upload) => Some(upload),
            Err(D3d12UploadError::HwDeviceInit(code)) => {
                eprintln!("skipping: this device does not do video (code {code})");
                None
            }
            Err(error) => panic!("the upload did not open: {error}"),
        }
    }

    /// Plays a test picture into `renderer` for half a second, and returns
    /// how many frames it took and every error the bus carried.
    fn show(upload: D3d12Upload, renderer: D3d12WindowRenderer) -> (u64, Vec<BusEvent>) {
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
        pipeline.stop();
        let errors = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        (shown, errors)
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
        let (shown, errors) = show(upload, renderer);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(shown >= 5, "only {shown} frames reached the window");
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
        let (shown, errors) = show(upload, renderer);
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
}
