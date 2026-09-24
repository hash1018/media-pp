//! A D3D11 renderer that brings its own window, or draws into one it is
//! given — the part a program otherwise writes for itself behind
//! [`D3d11FrameRenderer`](crate::elements::D3d11FrameRenderer).

use std::{
    any::Any,
    ffi::c_void,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use ffmpeg_next as ffmpeg;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY},
            Direct3D11::*,
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
                },
                DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
                DXGI_MWA_NO_ALT_ENTER, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIAdapter,
                IDXGIDevice, IDXGIFactory2, IDXGISwapChain1,
            },
        },
        UI::WindowsAndMessaging::GetClientRect,
    },
    core::{Interface, s},
};

use super::d3d11_renderer::{D3d11Picture, check_d3d11_frame};
use crate::{
    buffer::MediaBuffer,
    color::yuv_to_rgb_rows,
    contract::{InputContract, MediaKind, MemoryDomain, PixelLayoutSet, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::{D3d11Gpu, D3d11RendererError, SubmitError, WindowEvents, WindowOptions},
    error::Result,
    platform::windows::{d3d11::compile_shader, window::OwnedWindow},
    pp_log::{PpLog, pp_error, pp_info},
};

const BGRA_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d11/present_bgra.hlsl");
const NV12_SHADER: &[u8] = include_bytes!("../../../../shaders/d3d11/present_nv12.hlsl");

/// Double-buffered flip-model swap chain.
const FRAME_COUNT: u32 = 2;

/// Why a [`D3d11WindowRenderer`] could not be set up.
#[derive(Debug, ThisError)]
pub enum D3d11WindowRendererError {
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

/// A terminal sink that shows D3D11 video frames in a window — one it opens
/// for itself, the way a GStreamer video sink does when nobody hands it one,
/// or one the application gives it.
///
/// It takes what a [`D3d11Renderer`](crate::elements::D3d11Renderer) takes,
/// checked the same way — a texture from another device is refused, and a
/// frame's error is a [`D3d11RendererError`] — but draws it itself rather
/// than through a presenter, and so knows the frame it draws: an NV12 frame
/// is converted with its own colour description, BT.709, BT.601 or BT.2020
/// and limited or full range as it says, and where it says nothing, BT.709
/// for a picture over 576 rows and BT.601 otherwise. A BGRA frame is drawn
/// as it is. The picture keeps its aspect ratio inside the window, with
/// black bars as needed. It follows the window's size on its own, reading
/// it before every frame, so nothing has to tell it about a resize.
///
/// It draws; it does not pace. Put a [`crate::elements::VideoSynchronizer`]
/// or [`crate::elements::Pacer`] in front for a picture shown at its own
/// time rather than as fast as it is decoded. Each frame is presented
/// synchronized to the display's refresh.
///
/// Its GPU is the [`D3d11Gpu`] every other D3D11 element in the pipeline
/// shares — the frames it shows must come from that device.
pub struct D3d11WindowRenderer {
    name: Arc<str>,
    pp_log: PpLog,
    presenter: WindowPresenter,
}

impl D3d11WindowRenderer {
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
        gpu: &D3d11Gpu,
        options: WindowOptions,
    ) -> std::result::Result<(Self, WindowEvents), D3d11WindowRendererError> {
        if options.width == 0 || options.height == 0 {
            return Err(D3d11WindowRendererError::EmptyWindow {
                width: options.width,
                height: options.height,
            });
        }
        let (events_tx, events) = crossbeam_channel::unbounded();
        let window =
            OwnedWindow::open(&options, events_tx).map_err(D3d11WindowRendererError::Window)?;
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
        gpu: &D3d11Gpu,
        window: Arc<W>,
    ) -> std::result::Result<Self, D3d11WindowRendererError>
    where
        W: HasWindowHandle + Send + Sync + 'static,
    {
        let hwnd = match window.window_handle().map(|handle| handle.as_raw()) {
            Ok(RawWindowHandle::Win32(handle)) => HWND(handle.hwnd.get() as *mut c_void),
            _ => return Err(D3d11WindowRendererError::NotAWin32Window),
        };
        let presenter = WindowPresenter::new(gpu, hwnd, Keep::Given(window))?;
        Ok(Self::around(name, presenter))
    }

    fn around(name: impl Into<String>, presenter: WindowPresenter) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11WindowRenderer, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            presenter,
        }
    }

    /// Where the centre pixel of each frame drawn is read back to.
    #[cfg(test)]
    fn probe(&self) -> Arc<Mutex<Option<[u8; 4]>>> {
        Arc::clone(&self.presenter.probe)
    }

    fn draw(&self, frame: &ffmpeg::frame::Video) -> std::result::Result<(), D3d11RendererError> {
        let picture = check_d3d11_frame(frame, &self.presenter.device)?;
        self.presenter
            .show(&picture, frame)
            .map_err(D3d11RendererError::Submit)
    }
}

impl Element for D3d11WindowRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11WindowRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for D3d11WindowRenderer {
    /// What a [`D3d11Renderer`](crate::elements::D3d11Renderer) takes: a
    /// D3D11 texture, NV12 or BGRA.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };
        self.draw(&frame)
            .inspect_err(|error| pp_error!(self, "draw failed: {error}"))?;
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
    swap_chain: IDXGISwapChain1,
    /// `None` only inside a resize, between releasing the old back buffer's
    /// view and making the new one.
    render_target_view: Option<ID3D11RenderTargetView>,
    width: u32,
    height: u32,
}

/// The swap chain and shaders a [`D3d11WindowRenderer`] draws its window
/// with.
///
/// Every draw holds the shared context's lock for its whole
/// bind-draw-present sequence (see [`D3d11Gpu`]), so it cannot interleave
/// with another element's use of the same context.
struct WindowPresenter {
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    vertex_shader: ID3D11VertexShader,
    bgra_pixel_shader: ID3D11PixelShader,
    nv12_pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    /// The three rows `present_nv12.hlsl` turns Y'CbCr into R'G'B' with,
    /// written from each NV12 frame's own colour description.
    colour: ID3D11Buffer,
    hwnd: isize,
    /// Set once a call reports the device removed, after which every draw
    /// fails fast rather than touching the GPU again.
    device_lost: AtomicBool,
    state: Mutex<PresentState>,
    /// The centre pixel of the last frame drawn, R, G, B, A — read back
    /// before it is presented, for a test to check the colour it came out.
    #[cfg(test)]
    probe: Arc<Mutex<Option<[u8; 4]>>>,
    /// Last, so the swap chain above is released before the window goes.
    _keep: Keep,
}

// SAFETY: COM interfaces used from whichever thread the pipeline runs this
// sink on; the swap chain is only touched under `state`'s lock and the shared
// context under its own, and the window handle is only read.
unsafe impl Send for WindowPresenter {}
// SAFETY: as for `Send`; every mutable part is behind a lock or an atomic.
unsafe impl Sync for WindowPresenter {}

impl WindowPresenter {
    fn new(
        gpu: &D3d11Gpu,
        hwnd: HWND,
        keep: Keep,
    ) -> std::result::Result<Self, D3d11WindowRendererError> {
        let device = gpu.device().clone();
        let (width, height) = client_size(hwnd).unwrap_or((1, 1));
        let present = D3d11WindowRendererError::Present;
        // SAFETY: COM calls on a live device and a window that `keep` keeps
        // alive; every out-pointer is a local.
        unsafe {
            // The factory that made the device's own adapter, which is the
            // one a swap chain for that device has to come from.
            let adapter: IDXGIAdapter = device
                .cast::<IDXGIDevice>()
                .map_err(present)?
                .GetAdapter()
                .map_err(present)?;
            let factory: IDXGIFactory2 = adapter.GetParent().map_err(present)?;
            let swap_chain = factory
                .CreateSwapChainForHwnd(
                    &device,
                    hwnd,
                    &DXGI_SWAP_CHAIN_DESC1 {
                        Width: width.max(1),
                        Height: height.max(1),
                        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                        BufferCount: FRAME_COUNT,
                        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
                        ..Default::default()
                    },
                    None,
                    None,
                )
                .map_err(present)?;
            factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)
                .map_err(present)?;
            let render_target_view =
                create_render_target_view(&device, &swap_chain).map_err(present)?;

            let vertex = compile_shader(
                BGRA_SHADER,
                s!("present_bgra.hlsl"),
                s!("vs_main"),
                s!("vs_5_0"),
            )
            .map_err(present)?;
            let bgra = compile_shader(
                BGRA_SHADER,
                s!("present_bgra.hlsl"),
                s!("ps_bgra"),
                s!("ps_5_0"),
            )
            .map_err(present)?;
            let nv12 = compile_shader(
                NV12_SHADER,
                s!("present_nv12.hlsl"),
                s!("ps_nv12"),
                s!("ps_5_0"),
            )
            .map_err(present)?;
            let bytes = |blob: &windows::Win32::Graphics::Direct3D::ID3DBlob| {
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
            };
            let mut vertex_shader = None;
            device
                .CreateVertexShader(bytes(&vertex), None, Some(&mut vertex_shader))
                .map_err(present)?;
            let mut bgra_pixel_shader = None;
            device
                .CreatePixelShader(bytes(&bgra), None, Some(&mut bgra_pixel_shader))
                .map_err(present)?;
            let mut nv12_pixel_shader = None;
            device
                .CreatePixelShader(bytes(&nv12), None, Some(&mut nv12_pixel_shader))
                .map_err(present)?;
            let mut sampler = None;
            device
                .CreateSamplerState(
                    &D3D11_SAMPLER_DESC {
                        Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                        ComparisonFunc: D3D11_COMPARISON_NEVER,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    },
                    Some(&mut sampler),
                )
                .map_err(present)?;
            let mut colour = None;
            device
                .CreateBuffer(
                    &D3D11_BUFFER_DESC {
                        ByteWidth: size_of::<[[f32; 4]; 3]>() as u32,
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                        ..Default::default()
                    },
                    None,
                    Some(&mut colour),
                )
                .map_err(present)?;
            let missing = || {
                present(windows::core::Error::from(
                    windows::Win32::Foundation::E_POINTER,
                ))
            };
            Ok(Self {
                device,
                context: gpu.context(),
                vertex_shader: vertex_shader.ok_or_else(missing)?,
                bgra_pixel_shader: bgra_pixel_shader.ok_or_else(missing)?,
                nv12_pixel_shader: nv12_pixel_shader.ok_or_else(missing)?,
                sampler: sampler.ok_or_else(missing)?,
                colour: colour.ok_or_else(missing)?,
                hwnd: hwnd.0 as isize,
                device_lost: AtomicBool::new(false),
                state: Mutex::new(PresentState {
                    swap_chain,
                    render_target_view: Some(render_target_view),
                    width: width.max(1),
                    height: height.max(1),
                }),
                #[cfg(test)]
                probe: Arc::default(),
                _keep: keep,
            })
        }
    }

    fn checked<T>(&self, result: windows::core::Result<T>) -> std::result::Result<T, SubmitError> {
        result.map_err(|error| {
            if is_device_lost(&error) {
                self.device_lost.store(true, Ordering::Relaxed);
                SubmitError::DeviceRemoved
            } else {
                SubmitError::RenderFailed
            }
        })
    }

    /// Brings the swap chain to `width`x`height`.
    fn resize_to(
        &self,
        state: &mut PresentState,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError> {
        // The old back buffer's view goes first: `ResizeBuffers` fails while
        // anything still refers to a buffer it would replace.
        state.render_target_view = None;
        // SAFETY: the swap chain is only touched under `state`'s lock, which
        // the caller holds.
        unsafe {
            self.checked(state.swap_chain.ResizeBuffers(
                FRAME_COUNT,
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_SWAP_CHAIN_FLAG(0),
            ))?;
            state.render_target_view =
                Some(self.checked(create_render_target_view(&self.device, &state.swap_chain))?);
        }
        state.width = width;
        state.height = height;
        Ok(())
    }

    /// Binds, draws the frame letterboxed into the window, and presents — the
    /// whole sequence under the shared context's lock.
    fn draw(
        &self,
        pixel_shader: &ID3D11PixelShader,
        planes: &[Option<ID3D11ShaderResourceView>],
        colour: Option<[[f32; 4]; 3]>,
        frame_width: u32,
        frame_height: u32,
    ) -> std::result::Result<(), SubmitError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        // Follow the window: read its size before every frame, so a resize
        // needs no call from anyone. A minimised window has no area, and there
        // is nothing to draw into until it comes back.
        match client_size(HWND(self.hwnd as *mut c_void)) {
            Some((0, _)) | Some((_, 0)) => return Ok(()),
            Some((width, height)) if (width, height) != (state.width, state.height) => {
                let _context = self
                    .context
                    .lock()
                    .map_err(|_| SubmitError::RendererStopped)?;
                self.resize_to(&mut state, width, height)?;
            }
            _ => {}
        }
        let context = self
            .context
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        let viewport = letterbox(frame_width, frame_height, state.width, state.height);
        let scissor = RECT {
            left: 0,
            top: 0,
            right: state.width as i32,
            bottom: state.height as i32,
        };
        let target = state
            .render_target_view
            .clone()
            .ok_or(SubmitError::RenderFailed)?;
        // SAFETY: every object bound here is live and on the shared device,
        // and the shared context is held for the whole sequence.
        unsafe {
            context.ClearRenderTargetView(&target, &[0.0, 0.0, 0.0, 1.0]);
            context.OMSetRenderTargets(Some(&[Some(target)]), None);
            // The context is shared with GPU filters such as the compositor;
            // their blend and rasterizer state must not leak into this
            // opaque full-frame draw.
            context.OMSetBlendState(None, None, u32::MAX);
            context.RSSetState(None);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.RSSetViewports(Some(&[viewport]));
            context.RSSetScissorRects(Some(&[scissor]));
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(pixel_shader, None);
            context.PSSetShaderResources(0, Some(planes));
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            if let Some(rows) = colour {
                context.UpdateSubresource(&self.colour, 0, None, rows.as_ptr().cast(), 0, 0);
                context.PSSetConstantBuffers(0, Some(&[Some(self.colour.clone())]));
            }
            context.Draw(3, 0);
            // Unbound before anything else draws through this context, so the
            // frame's texture is not held by leftover state.
            context.PSSetShaderResources(0, Some(&vec![None; planes.len()]));
            context.PSSetConstantBuffers(0, Some(&[None]));
            context.OMSetRenderTargets(None, None);
        }
        #[cfg(test)]
        // SAFETY: the shared context is held, and the swap chain under
        // `state`'s lock.
        unsafe {
            self.probe_centre(&context, &state)
        };
        // Presented before returning: a preroll counts this return as the
        // frame being on its way to the screen.
        // SAFETY: the swap chain is only touched under `state`'s lock, held here.
        self.checked(unsafe { state.swap_chain.Present(1, DXGI_PRESENT(0)) }.ok())?;
        drop(context);
        Ok(())
    }

    /// Reads the centre pixel of the back buffer just drawn into
    /// [`Self::probe`] — what the window shows there once it is presented.
    ///
    /// # Safety
    ///
    /// `context` is the shared context, held, and `state` is under its lock.
    #[cfg(test)]
    unsafe fn probe_centre(&self, context: &ID3D11DeviceContext, state: &PresentState) {
        // SAFETY: the caller's promise; everything made here is a local.
        unsafe {
            let Ok(back) = state.swap_chain.GetBuffer::<ID3D11Texture2D>(0) else {
                return;
            };
            let mut staging = None;
            let made = self.device.CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: 1,
                    Height: 1,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_STAGING,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut staging),
            );
            let (Ok(()), Some(staging)) = (made, staging) else {
                return;
            };
            let (x, y) = (state.width / 2, state.height / 2);
            context.CopySubresourceRegion(
                &staging,
                0,
                0,
                0,
                0,
                &back,
                0,
                Some(&D3D11_BOX {
                    left: x,
                    top: y,
                    front: 0,
                    right: x + 1,
                    bottom: y + 1,
                    back: 1,
                }),
            );
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            if context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .is_ok()
            {
                let bgra = std::slice::from_raw_parts(mapped.pData.cast::<u8>(), 4);
                *self.probe.lock().unwrap() = Some([bgra[2], bgra[1], bgra[0], bgra[3]]);
                context.Unmap(&staging, 0);
            }
        }
    }

    fn plane(
        &self,
        texture: &ID3D11Texture2D,
        format: DXGI_FORMAT,
        array_index: u32,
    ) -> std::result::Result<Option<ID3D11ShaderResourceView>, SubmitError> {
        let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: array_index,
                    ArraySize: 1,
                },
            },
        };
        let mut view = None;
        // SAFETY: a view of a texture on this presenter's device, which
        // `D3d11Renderer` checks before submitting.
        self.checked(unsafe {
            self.device
                .CreateShaderResourceView(texture, Some(&desc), Some(&mut view))
        })?;
        Ok(view)
    }

    fn usable(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }
        Ok(())
    }

    /// Draws one checked frame: BGRA as it is, NV12 through the matrix its
    /// own colour description gives.
    fn show(
        &self,
        picture: &D3d11Picture,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), SubmitError> {
        let (width, height) = (frame.width(), frame.height());
        self.usable(width, height)?;
        let texture = &picture.texture;
        if picture.format == DXGI_FORMAT_B8G8R8A8_UNORM {
            let bgra = self.plane(texture, DXGI_FORMAT_B8G8R8A8_UNORM, picture.array_index)?;
            return self.draw(&self.bgra_pixel_shader, &[bgra], None, width, height);
        }
        // One NV12 texture read as two planes: the view's format picks which.
        let luma = self.plane(texture, DXGI_FORMAT_R8_UNORM, picture.array_index)?;
        let chroma = self.plane(texture, DXGI_FORMAT_R8G8_UNORM, picture.array_index)?;
        let rows = yuv_to_rgb_rows(frame.color_space(), frame.color_range(), height);
        self.draw(
            &self.nv12_pixel_shader,
            &[luma, chroma],
            Some(rows),
            width,
            height,
        )
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

fn is_device_lost(error: &windows::core::Error) -> bool {
    matches!(
        error.code(),
        DXGI_ERROR_DEVICE_REMOVED | DXGI_ERROR_DEVICE_RESET | DXGI_ERROR_DEVICE_HUNG
    )
}

unsafe fn create_render_target_view(
    device: &ID3D11Device,
    swap_chain: &IDXGISwapChain1,
) -> windows::core::Result<ID3D11RenderTargetView> {
    // SAFETY: the swap chain's first back buffer, viewed on its own device.
    unsafe {
        let back_buffer: ID3D11Texture2D = swap_chain.GetBuffer(0)?;
        let mut view = None;
        device.CreateRenderTargetView(&back_buffer, None, Some(&mut view))?;
        view.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_POINTER))
    }
}

/// The largest rectangle of the frame's aspect ratio that fits the window,
/// centred in it.
fn letterbox(
    frame_width: u32,
    frame_height: u32,
    window_width: u32,
    window_height: u32,
) -> D3D11_VIEWPORT {
    let scale =
        (window_width as f32 / frame_width as f32).min(window_height as f32 / frame_height as f32);
    let (width, height) = (frame_width as f32 * scale, frame_height as f32 * scale);
    D3D11_VIEWPORT {
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
        let near = |view: D3D11_VIEWPORT, expected: [f32; 4]| {
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
            AppSource, D3d11Upload, SwScaler, TestVideoOptions, TestVideoSource, WindowEvent,
        },
        pipeline::Pipeline,
    };

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;

    fn gpu() -> Option<D3d11Gpu> {
        D3d11Gpu::new()
            .inspect_err(|error| eprintln!("skipping: no Direct3D 11 device here ({error})"))
            .ok()
    }

    /// Plays a test picture into `renderer` for half a second, and returns
    /// how many frames it took and every error the bus carried.
    fn show(gpu: &D3d11Gpu, renderer: D3d11WindowRenderer) -> (u64, Vec<BusEvent>) {
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: WIDTH,
                height: HEIGHT,
                frame_rate: ffmpeg_next::Rational::new(30, 1),
            },
        );
        let device = gpu.device().clone();
        let (pipeline, ()) = Pipeline::new("window-renderer", source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(SwScaler::to_format(
                    "to-nv12",
                    ffmpeg_next::format::Pixel::NV12,
                    ffmpeg_next::software::scaling::Flags::BILINEAR,
                ))
                .pipe(D3d11Upload::new("upload", &device))
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
            .find(|element| element.element_type == ElementType::D3d11WindowRenderer)
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
        let (renderer, events) = match D3d11WindowRenderer::open(
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
        let (shown, errors) = show(&gpu, renderer);
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
        let renderer = D3d11WindowRenderer::for_window("screen", &gpu, Arc::clone(&window))
            .expect("draw into the given window");
        assert_eq!(
            Arc::strong_count(&window),
            2,
            "the renderer holds the window"
        );
        let (shown, errors) = show(&gpu, renderer);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(shown >= 5, "only {shown} frames reached the window");
    }

    /// No area and no Win32 handle are refused before anything is opened.
    #[test]
    fn a_window_with_nothing_to_draw_in_is_refused() {
        let Some(gpu) = gpu() else { return };
        assert!(matches!(
            D3d11WindowRenderer::open(
                "screen",
                &gpu,
                WindowOptions {
                    width: 0,
                    ..WindowOptions::default()
                },
            ),
            Err(D3d11WindowRendererError::EmptyWindow { width: 0, .. })
        ));

        struct Headless;
        impl HasWindowHandle for Headless {
            fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
                Err(HandleError::Unavailable)
            }
        }
        assert!(matches!(
            D3d11WindowRenderer::for_window("screen", &gpu, Arc::new(Headless)),
            Err(D3d11WindowRendererError::NotAWin32Window)
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

    /// Draws `frame` and reads back the centre of the window.
    fn drawn(gpu: &D3d11Gpu, frame: MediaBuffer) -> Option<[u8; 3]> {
        let (renderer, _events) = match D3d11WindowRenderer::open(
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
        let device = gpu.device().clone();
        let (pipeline, ()) = Pipeline::new("window-colour", source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(D3d11Upload::new("upload", &device))
                .to(renderer)?;
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

    /// Each NV12 frame is converted with its own matrix. (72, 107, 220) is
    /// R'G'B' (230, 20, 20) in BT.709; tagged BT.709 it comes out as that.
    /// Read with BT.601 — what this renderer used to do with every frame —
    /// the same numbers are (211, 0, 22), and a BT.601 frame of them is
    /// drawn as that.
    #[test]
    fn an_nv12_frame_is_drawn_with_its_own_colour_description() {
        use ffmpeg_next::color::Space;

        let Some(gpu) = gpu() else { return };
        let ycbcr = [72, 107, 220];
        let Some(bt709) = drawn(&gpu, solid_nv12(ycbcr, Space::BT709)) else {
            return;
        };
        assert!(near(bt709, [230, 20, 20]), "BT.709 drawn as {bt709:?}");
        let Some(bt601) = drawn(&gpu, solid_nv12(ycbcr, Space::BT470BG)) else {
            return;
        };
        assert!(near(bt601, [211, 0, 22]), "BT.601 drawn as {bt601:?}");
    }
}
