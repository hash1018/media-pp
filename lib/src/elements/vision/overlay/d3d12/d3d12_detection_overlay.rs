//! [`D3d12DetectionOverlay`]: what a detector found, drawn onto D3D12
//! pictures by compute shaders.

use std::collections::HashMap;
use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::*,
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R8_TYPELESS,
            DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_TYPELESS,
            DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R16_TYPELESS,
            DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_TYPELESS,
            DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_UNORM,
        },
    },
    core::{Interface, s},
};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Attached, ElementType, Identity},
    elements::Detections,
    error::Result,
    frame_size::ForSize,
    platform::ffmpeg::AvBufferRef,
    platform::windows::{
        d3d12_compute::{
            Commands, Kernel, buffer, copy_plane, plane_view, plane_write_view, transition,
            uav_barrier, upload_texture, writable_texture,
        },
        d3d12_gpu::D3d12Gpu,
        d3d12va::{
            create_hw_device_ctx, create_hw_frames_ctx_of, d3d12va_texture, set_d3d12va_fence_value,
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::{
    Canvas, Cut, DetectionOverlayOptions, DetectionOverlayOptionsError, Hide, LABEL_CACHE, Mark,
    MaskKey, Rect, bt709_limited, cover, hides, marks_turned, rasterize, ten_bit,
};
use crate::elements::Analytics;
use crate::elements::source::{TextFontError, load_font};
use crate::orientation::Orientations;

const PAINT: &[u8] = include_bytes!("../../../../shaders/d3d12/overlay.hlsl");
const REDACT: &[u8] = include_bytes!("../../../../shaders/d3d12/redact.hlsl");

/// What `redact.hlsl` reads as "none" for a table offset.
const NONE: u32 = u32::MAX;

/// The views of the copy's planes in the heap: UNORM luma and chroma, then
/// the same as integers; each stroke's mask after them.
const PLANE_VIEWS: usize = 4;

/// Errors specific to `D3d12DetectionOverlay`. Converts into the crate-wide
/// `Error` via `?`.
#[derive(Debug, ThisError)]
pub enum D3d12DetectionOverlayError {
    /// Something other than a decoded picture arrived.
    #[error("D3d12DetectionOverlay only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// Not an NV12 D3D12 picture, nor a P010 one to hide in with nothing
    /// drawn: what it is.
    #[error(
        "D3d12DetectionOverlay draws on NV12 D3D12 pictures, and hides in P010 where it draws nothing; got {0}"
    )]
    UnsupportedPicture(String),
    /// A D3D12 picture made on another device than the [`D3d12Gpu`] it was
    /// given.
    #[error("D3d12DetectionOverlay was handed a picture of another D3D12 device")]
    DeviceMismatch,
    /// A picture larger than the texture it is said to be in.
    #[error("a {picture:?} picture is larger than its {texture:?} texture")]
    PictureOutsideTexture {
        /// The picture's size.
        picture: (u32, u32),
        /// The texture's.
        texture: (u32, u32),
    },
    /// The GPU cannot read the formats it draws in through a shader's
    /// writable view, which every D3D12 GPU of feature level 12 can.
    #[error("the GPU cannot load R8, R8G8, R16 and R16G16 through a typed UAV")]
    NoTypedUavLoads,
    /// FFmpeg could not take the device for the frames the copies are
    /// drawn on — a device without D3D12 Video, as a software adapter is.
    #[error("failed to create the D3D12VA device context (code {0})")]
    HwDeviceInit(i32),
    /// Making the frames the copies are drawn on failed.
    #[error("could not make the frames to draw on (code {0})")]
    Frames(i32),
    /// Taking a frame from those frames failed.
    #[error("failed to take a frame from the D3D12 pool (code {0})")]
    FrameGet(i32),
    /// D3D12 failed.
    #[error(transparent)]
    D3d12(#[from] windows::core::Error),
    /// Options it cannot be made of.
    #[error(transparent)]
    Options(#[from] DetectionOverlayOptionsError),
    /// The label font is not a TrueType or OpenType font.
    #[error("the label font is not a TrueType or OpenType font")]
    LabelFont,
}

impl From<TextFontError> for D3d12DetectionOverlayError {
    fn from(error: TextFontError) -> Self {
        match error {
            TextFontError::Size(size) => DetectionOverlayOptionsError::LabelSize(size).into(),
            TextFontError::Font(_) => Self::LabelFont,
        }
    }
}

/// Draws the [`Detections`] each D3D12 picture carries onto a copy of it,
/// on the GPU with compute shaders — a box around each object, and with
/// [`BoxStyle::label`](super::super::BoxStyle::label) its class and score
/// above it — and hands the copy on, carrying the same `Detections`: what
/// [`SwDetectionOverlay`](super::super::SwDetectionOverlay) does in system
/// memory, so that a detector's pictures go on to an encoder or a renderer
/// without leaving the GPU, on any vendor's.
///
/// It takes NV12 D3D12 pictures of any size made on the [`D3d12Gpu`] it is
/// given, and P010 ones where it only hides, as the software overlay does,
/// and draws at that size. The copy is a frame from a pool of this
/// element's own, and is what keeps the boxes off the picture it was
/// handed, which a Tee in front of it shares with another branch. A picture
/// carrying nothing to draw is handed on as it came.
///
/// Lines and label bands are rectangle fills; a label's text is rasterized
/// on the CPU once per distinct string, uploaded as a coverage mask and
/// blended on the GPU. A picture is painted in BT.709 limited-range colour,
/// as the other overlays paint it.
///
/// What a rule hides is hidden before any box is drawn, by two shaders —
/// one for each cell's mean, one painting each sample — that read and
/// write the planes as integers. What the CPU works out with a division or
/// a root — each cell's reciprocal, where each sample sits among the cells,
/// how much of the cover it takes — the CPU works out here too and hands
/// over, as a shader's are not correctly rounded; the rest is done in the
/// CPU's operations and order, so that the bytes are `SwDetectionOverlay`'s.
///
/// # Requirements
///
/// The `d3d12` feature, on Windows, and a GPU that loads R8 and R16
/// formats through a typed UAV, as every one of feature level 12 does.
pub struct D3d12DetectionOverlay(FilterStage<Overlaying>);

filter_stage!(D3d12DetectionOverlay);

/// A label's coverage, uploaded.
struct Mask {
    texture: ID3D12Resource,
    size: (u32, u32),
}

/// The copy's planes as textures a shader writes, and what they were made
/// for: its size, and whether in ten bits.
struct Work {
    luma: ID3D12Resource,
    chroma: ID3D12Resource,
    size: (u32, u32),
    wide: bool,
}

/// What `redact.hlsl` is handed for one plane of one hide: its `Cells`
/// constants, and whether its means are to be taken first.
struct CellsDispatch {
    chroma: bool,
    constants: [u32; 16],
    take_means: bool,
}

/// What a [`D3d12DetectionOverlay`] does with each picture. The resources
/// the GPU may still be reading are kept until `Drop` has waited for it.
struct Overlaying {
    pp_log: PpLog,
    options: DetectionOverlayOptions,
    font: Option<ab_glyph::FontArc>,
    /// Masks already rasterized and uploaded — labels by their text, the
    /// pieces of zones and lines by where they are; `None` for one with
    /// nothing to draw.
    masks: HashMap<MaskKey, Option<Mask>>,
    /// Whether it has said that a rule names a class the detections carry
    /// no names for.
    warned_names: bool,
    /// How each picture is turned to be shown.
    orientations: Orientations,
    device: ID3D12Device,
    /// `overlay.hlsl`'s `paint_nv12`.
    paint: Kernel,
    /// `redact.hlsl`'s `cell_means1`, and under its root signature
    /// `cell_means2`, `cell_paint1` and `cell_paint2`.
    means1: Kernel,
    means2: ID3D12PipelineState,
    paint1: ID3D12PipelineState,
    paint2: ID3D12PipelineState,
    commands: Commands,
    /// The copy's planes' views and each stroke's mask's, shader-visible;
    /// made again, larger, for a picture with more strokes.
    heap: Option<(ID3D12DescriptorHeap, usize)>,
    heap_step: usize,
    work: Option<Work>,
    /// The cells' means, a `float4` each, grown to the most a picture has
    /// wanted.
    means: Option<(ID3D12Resource, usize)>,
    /// What the CPU works out for the hiding shaders, grown likewise.
    table: Option<(ID3D12Resource, usize)>,
    /// This element's own reference to the device its copies are made on.
    hw_device_ctx: AvBufferRef,
    /// The pools the copies come from, one per layout, each made for the
    /// size of the pictures arriving — see `ForSize`.
    nv12_frames: ForSize<AvBufferRef>,
    p010_frames: ForSize<AvBufferRef>,
    /// Reuses only the CPU-side `AVFrame` wrapper; each texture comes from a
    /// frames pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The picture the last draw copies from, held until the GPU has: let
    /// go of when it is handed on, its texture would go back to its
    /// producer's pool, to be written again under the copy.
    reading: Option<MediaBuffer>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the D3D12 objects
// are free-threaded and touched only through `&mut self`, by the one thread
// transforming at a time.
unsafe impl Send for Overlaying {}

impl D3d12DetectionOverlay {
    /// An overlay drawing as `options` say, on pictures made on `gpu` — the
    /// same [`D3d12Gpu`] every other D3D12 element of the pipeline was made
    /// with — its copies made there too.
    ///
    /// # Errors
    ///
    /// Options it cannot be made of — a label with no font, a class named
    /// twice — a font that cannot be read, a GPU that cannot load what it
    /// draws in through a typed UAV, and D3D12's where a shader will not
    /// compile.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: DetectionOverlayOptions,
    ) -> std::result::Result<Self, D3d12DetectionOverlayError> {
        let name: Arc<str> = name.into().into();
        options.check()?;
        let identity = Identity::new(ElementType::D3d12DetectionOverlay, &*name);
        let pp_log = identity.pp_log().clone();
        // Read once at any size: each label is rasterized at its own.
        let font = options
            .font
            .as_ref()
            .map(|font| load_font(font.clone(), 16.0))
            .transpose()?;
        let device = gpu.device().clone();
        let mut supported = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a live device, and a correctly sized out-parameter of the
        // feature asked about.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut supported).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
            )?;
        }
        if !supported.TypedUAVLoadAdditionalFormats.as_bool() {
            return Err(D3d12DetectionOverlayError::NoTypedUavLoads);
        }
        // SAFETY: every call is on a live device.
        let (paint, means1, means2, paint1, paint2, heap_step) = unsafe {
            let paint = Kernel::new(&device, PAINT, s!("overlay.hlsl"), s!("paint_nv12"))?;
            let means1 = Kernel::new(&device, REDACT, s!("redact.hlsl"), s!("cell_means1"))?;
            let sibling = |entry| means1.sibling(&device, REDACT, s!("redact.hlsl"), entry);
            let (means2, paint1, paint2) = (
                sibling(s!("cell_means2"))?,
                sibling(s!("cell_paint1"))?,
                sibling(s!("cell_paint2"))?,
            );
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            (paint, means1, means2, paint1, paint2, heap_step)
        };
        let commands = Commands::new(&device, gpu.queue())?;
        // SAFETY: a live device, whose context owns its own reference.
        let hw_device_ctx = unsafe { create_hw_device_ctx(&device) }
            .map_err(D3d12DetectionOverlayError::HwDeviceInit)?;
        pp_info!(
            pp_log: &pp_log,
            "opened: NV12, or P010 to hide in, {} rules, others {:?}, font={}",
            options.rules.len(),
            options.others,
            font.is_some()
        );
        Ok(Self(FilterStage::new(
            identity,
            Overlaying {
                pp_log,
                options,
                font,
                masks: HashMap::new(),
                warned_names: false,
                orientations: Orientations::default(),
                device,
                paint,
                means1,
                means2,
                paint1,
                paint2,
                commands,
                heap: None,
                heap_step,
                work: None,
                means: None,
                table: None,
                hw_device_ctx,
                nv12_frames: ForSize::new(),
                p010_frames: ForSize::new(),
                pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
                reading: None,
            },
        )))
    }
}

/// A colour as `paint_nv12` takes it: Y', Cb, Cr in BT.709's limited range,
/// each 0 to 1.
fn colour_of(color: crate::color::Color) -> [f32; 3] {
    let (y, cb, cr) = bt709_limited(color);
    [y, cb, cr].map(|value| f32::from(value) / 255.0)
}

/// `paint_nv12`'s twelve constants for `rect` in `colour`.
fn paint_constants(rect: Rect, colour: [f32; 3], masked: bool) -> [u32; 12] {
    [
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        colour[0].to_bits(),
        colour[1].to_bits(),
        colour[2].to_bits(),
        0,
        u32::from(masked),
        0,
        0,
        0,
    ]
}

/// Where a sample sits among the cells' centres along a side: the cells
/// either side and how far it is from the first's centre to the second's —
/// `SwDetectionOverlay`'s `between`, the same operations in the same order.
fn between(sample: u32, cells: u32, length: u32) -> [f32; 3] {
    let last = (cells - 1) as f32;
    let at = ((sample as f32 + 0.5) * cells as f32 / length as f32 - 0.5)
        .max(0.0)
        .min(last);
    let first = at.floor() as u32;
    [
        first as f32,
        (first + 1).min(cells - 1) as f32,
        at - first as f32,
    ]
}

/// What the CPU works out for one plane of one hide — the reciprocal of each
/// cell's count, the columns' and rows' places among the cells, each
/// sample's cover, a fill's colour — appended to `table`, and the
/// dispatch's constants: `means_at` is where its cells' means go.
#[allow(clippy::too_many_arguments)]
fn cells_dispatch(
    table: &mut Vec<f32>,
    chroma: bool,
    (origin, size): ((u32, u32), (u32, u32)),
    cells: (u32, u32),
    (smooth, cut, wide): (bool, Cut, bool),
    fill: Option<[f32; 4]>,
    means_at: &mut u32,
) -> CellsDispatch {
    let edge = super::super::cell_edge;
    let reciprocals_at = table.len() as u32;
    for cy in 0..cells.1 {
        let (y0, y1) = (edge(cy, cells.1, size.1), edge(cy + 1, cells.1, size.1));
        for cx in 0..cells.0 {
            let (x0, x1) = (edge(cx, cells.0, size.0), edge(cx + 1, cells.0, size.0));
            table.push(1.0f32 / ((x1 - x0) * (y1 - y0)) as f32);
        }
    }
    let (columns_at, rows_at) = if smooth {
        let columns_at = table.len() as u32;
        table.extend((0..size.0).flat_map(|x| between(x, cells.0, size.0)));
        let rows_at = table.len() as u32;
        table.extend((0..size.1).flat_map(|y| between(y, cells.1, size.1)));
        (columns_at, rows_at)
    } else {
        (0, 0)
    };
    let alpha_at = if cut.is_box() {
        NONE
    } else {
        let at = table.len() as u32;
        for y in 0..size.1 {
            for x in 0..size.0 {
                table.push(cover(x, y, size.0, size.1, cut).unwrap_or(-1.0));
            }
        }
        at
    };
    let fill_at = match fill {
        Some(values) => {
            let at = table.len() as u32;
            table.extend(values);
            at
        }
        None => NONE,
    };
    let constants = [
        origin.0,
        origin.1,
        size.0,
        size.1,
        cells.0,
        cells.1,
        u32::from(smooth),
        if wide { 6 } else { 0 },
        *means_at,
        reciprocals_at,
        columns_at,
        rows_at,
        alpha_at,
        fill_at,
        0,
        0,
    ];
    *means_at += cells.0 * cells.1;
    CellsDispatch {
        chroma,
        constants,
        take_means: fill.is_none(),
    }
}

impl Overlaying {
    /// The size of the mask `key` names, where it has one to draw,
    /// rasterizing and uploading it where it is not cached yet.
    fn mask(&mut self, key: &MaskKey) -> Option<(u32, u32)> {
        if !self.masks.contains_key(key) {
            let mask = match rasterize(key, self.font.as_ref()) {
                Ok(Some(raster)) => upload_texture(
                    &self.device,
                    &mut self.commands,
                    DXGI_FORMAT_R8_UNORM,
                    (raster.width, raster.height),
                    (
                        &raster.coverage,
                        raster.width as usize,
                        raster.width as usize,
                    ),
                )
                .map(|texture| Mask {
                    texture,
                    size: (raster.width, raster.height),
                })
                .inspect_err(|error| pp_error!(self, "{key:?} not drawn: {error}"))
                .ok(),
                Ok(None) => None,
                Err(error) => {
                    pp_error!(self, "{key:?} not drawn: {error:?}");
                    None
                }
            };
            self.masks.insert(key.clone(), mask);
        }
        self.masks.get(key)?.as_ref().map(|mask| mask.size)
    }

    /// A frame of `layout` at `width` x `height` from this element's pool.
    fn frame(
        &mut self,
        wide: bool,
        (width, height): (u32, u32),
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12DetectionOverlayError>
    {
        let (device, pp_log) = (&self.hw_device_ctx, &self.pp_log);
        let (pool, layout) = if wide {
            (&mut self.p010_frames, ffi::AVPixelFormat::AV_PIX_FMT_P010LE)
        } else {
            (&mut self.nv12_frames, ffi::AVPixelFormat::AV_PIX_FMT_NV12)
        };
        let frames_ctx = pool
            .try_get(width, height, |width, height| {
                pp_info!(pp_log: pp_log, "drawing {layout:?} {width}x{height}");
                // SAFETY: a live device context, which is what the owned
                // `AvBufferRef` is; the pool grows as frames are taken.
                unsafe { create_hw_frames_ctx_of(device, layout, width, height, 0) }
                    .map_err(D3d12DetectionOverlayError::Frames)
            })?
            .as_ptr();
        let mut destination = self.pool.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous texture
        // handed back first; the frames context is this element's own, held
        // for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                return Err(D3d12DetectionOverlayError::FrameGet(code));
            }
        }
        Ok(destination)
    }

    /// The copy's planes for a picture of `size`, in ten bits where `wide`,
    /// made again where the last were for another.
    fn work(&mut self, size: (u32, u32), wide: bool) -> windows::core::Result<&Work> {
        if self
            .work
            .as_ref()
            .is_none_or(|work| work.size != size || work.wide != wide)
        {
            let (luma, chroma): (DXGI_FORMAT, DXGI_FORMAT) = if wide {
                (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16G16_TYPELESS)
            } else {
                (DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8G8_TYPELESS)
            };
            let half = (size.0.div_ceil(2), size.1.div_ceil(2));
            // SAFETY: a live device.
            let (luma, chroma) = unsafe {
                (
                    writable_texture(&self.device, luma, size)?,
                    writable_texture(&self.device, chroma, half)?,
                )
            };
            self.work = Some(Work {
                luma,
                chroma,
                size,
                wide,
            });
        }
        Ok(self.work.as_ref().expect("made above"))
    }

    /// A copy of `source` with `detections` and `analytics` drawn on it.
    fn draw(
        &mut self,
        source: &ffmpeg::frame::Video,
        detections: Option<&Detections>,
        analytics: Option<&Analytics>,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12DetectionOverlayError>
    {
        let (texture, fence, fence_value) = d3d12va_texture(source)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
            .ok_or_else(|| {
                D3d12DetectionOverlayError::UnsupportedPicture(format!("{:?}", source.format()))
            })?;
        // SAFETY: the live frame owns these non-null objects; cloning the
        // borrowed wrappers takes references of the picture's own.
        let (texture, fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&texture)
                    .expect("not null")
                    .clone(),
                ID3D12Fence::from_raw_borrowed(&fence)
                    .expect("not null")
                    .clone(),
            )
        };
        let mut made_on: Option<ID3D12Device> = None;
        // SAFETY: a live resource and a correctly typed out-parameter.
        unsafe { texture.GetDevice(&mut made_on) }?;
        if made_on.is_none_or(|made_on| made_on.as_raw() != self.device.as_raw()) {
            return Err(D3d12DetectionOverlayError::DeviceMismatch);
        }
        // SAFETY: a live resource's plain description, returned by value.
        let desc = unsafe { texture.GetDesc() };
        let wide = match desc.Format {
            DXGI_FORMAT_NV12 => false,
            // Hidden in, never drawn on, as the CPU's overlay does.
            DXGI_FORMAT_P010 if !self.options.draws_any() => true,
            other => {
                return Err(D3d12DetectionOverlayError::UnsupportedPicture(
                    if other == DXGI_FORMAT_P010 {
                        "P010 with something to draw".to_owned()
                    } else {
                        format!("a texture of DXGI format {}", other.0)
                    },
                ));
            }
        };
        let (width, height) = (source.width(), source.height());
        let surface = (desc.Width as u32, desc.Height);
        if width == 0 || height == 0 || width > surface.0 || height > surface.1 {
            return Err(D3d12DetectionOverlayError::PictureOutsideTexture {
                picture: (width, height),
                texture: surface,
            });
        }
        // What the last picture queued still reads the masks, the table, the
        // views and that picture: it runs out before any is touched or let
        // go of.
        self.commands.wait_idle()?;
        self.reading = None;
        let canvas = Canvas {
            width,
            height,
            block: 2,
        };
        // The cache is emptied between pictures, never while one's marks are
        // placed: every mask made for this picture is read after all of them
        // are.
        if self.masks.len() >= LABEL_CACHE {
            self.masks.clear();
        }
        // The options out of `self` while the marks are placed, as placing
        // them makes masks through `self`; nothing returns in between.
        let options = std::mem::take(&mut self.options);
        let orientation = self.orientations.of(source, &self.pp_log);
        let strokes = marks_turned(
            canvas,
            orientation,
            &options,
            detections,
            analytics,
            &mut |key| self.mask(key),
        );
        // What is hidden, in the order the CPU hides it: before every mark,
        // so that a box and its label are drawn over it where asked for.
        let hidden = detections
            .map(|detections| hides(canvas, &options, detections))
            .unwrap_or_default();
        self.options = options;

        // Each hide as the cells of its two planes, what the CPU works out
        // for them in the table: NV12's colour cut into the same cells at
        // half the size, the rectangle being on whole blocks.
        let transfer = source.color_transfer_characteristic();
        let mut table = Vec::new();
        let mut cells_at = 0;
        let mut dispatches = Vec::new();
        for hide in &hidden {
            let (rect, count, smooth, cut, fill) = match *hide {
                Hide::Cells {
                    rect,
                    cells,
                    smooth,
                    cut,
                } => (rect, cells, smooth, cut, None),
                Hide::Fill { rect, color, cut } => {
                    let [y, cb, cr] = if wide {
                        ten_bit(color, transfer)
                    } else {
                        let (y, cb, cr) = bt709_limited(color);
                        [y, cb, cr].map(f32::from)
                    };
                    (
                        rect,
                        (1, 1),
                        false,
                        cut,
                        Some([[y, 0.0, 0.0, 0.0], [cb, cr, 0.0, 0.0]]),
                    )
                }
            };
            let whole = ((rect.x, rect.y), (rect.width, rect.height));
            let half = ((rect.x / 2, rect.y / 2), (rect.width / 2, rect.height / 2));
            for (chroma, area) in [(false, whole), (true, half)] {
                dispatches.push(cells_dispatch(
                    &mut table,
                    chroma,
                    area,
                    count,
                    (smooth, cut, wide),
                    fill.map(|values| values[usize::from(chroma)]),
                    &mut cells_at,
                ));
            }
        }
        let mut destination = self.frame(wide, (width, height))?;
        let (to, to_fence, to_value) = d3d12va_texture(&destination)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
            .ok_or_else(|| {
                D3d12DetectionOverlayError::UnsupportedPicture("a frame of its own pool".into())
            })?;
        // SAFETY: as for the source's, the pooled frame owns them.
        let (to, to_fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&to)
                    .expect("not null")
                    .clone(),
                ID3D12Fence::from_raw_borrowed(&to_fence)
                    .expect("not null")
                    .clone(),
            )
        };

        // Every resource made, and the table written, before anything is
        // recorded, so a failure leaves no list half written.
        let means_wanted = cells_at as usize;
        if means_wanted > 0
            && self
                .means
                .as_ref()
                .is_none_or(|(_, held)| *held < means_wanted)
        {
            // SAFETY: a live device.
            let made = unsafe {
                buffer(
                    &self.device,
                    (means_wanted * 4 * size_of::<f32>()) as u64,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )?
            };
            self.means = Some((made, means_wanted));
        }
        if !table.is_empty()
            && self
                .table
                .as_ref()
                .is_none_or(|(_, held)| *held < table.len())
        {
            // SAFETY: a live device.
            let made = unsafe {
                buffer(
                    &self.device,
                    (table.len() * size_of::<f32>()) as u64,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?
            };
            self.table = Some((made, table.len()));
        }
        if let Some((held, _)) = &self.table
            && !table.is_empty()
        {
            // SAFETY: an upload buffer of at least the table's floats, which
            // the GPU is not reading — waited for above — mapped for the copy.
            unsafe {
                let mut mapped = std::ptr::null_mut();
                held.Map(0, None, Some(&mut mapped))?;
                std::ptr::copy_nonoverlapping(table.as_ptr(), mapped.cast::<f32>(), table.len());
                held.Unmap(0, None);
            }
        }
        let views = PLANE_VIEWS + strokes.len();
        if self.heap.as_ref().is_none_or(|(_, held)| *held < views) {
            // SAFETY: a live device and a description built here.
            let heap: ID3D12DescriptorHeap = unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: views as u32,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?
            };
            self.heap = Some((heap, views));
        }
        let (luma_unorm, chroma_unorm, luma_uint, chroma_uint) = if wide {
            (
                DXGI_FORMAT_R16_UNORM,
                DXGI_FORMAT_R16G16_UNORM,
                DXGI_FORMAT_R16_UINT,
                DXGI_FORMAT_R16G16_UINT,
            )
        } else {
            (
                DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_R8G8_UNORM,
                DXGI_FORMAT_R8_UINT,
                DXGI_FORMAT_R8G8_UINT,
            )
        };
        let work = self.work((width, height), wide)?;
        let (work_luma, work_chroma) = (work.luma.clone(), work.chroma.clone());
        let (heap, _) = self.heap.as_ref().expect("made above");
        let heap = heap.clone();
        let step = self.heap_step;
        // SAFETY: a live heap's handles, each `step` apart, inside it.
        let (cpu_start, gpu_start) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart(),
                heap.GetGPUDescriptorHandleForHeapStart(),
            )
        };
        let cpu = |at: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu_start.ptr + at * step,
        };
        let gpu = |at: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_start.ptr + (at * step) as u64,
        };
        // A stroke's mask, as the cache holds it: a stroke whose mask is gone
        // is drawn not at all, rather than as a solid block.
        let masks: Vec<Option<Option<ID3D12Resource>>> = strokes
            .iter()
            .map(|stroke: &Mark| match &stroke.mask {
                None => Some(None),
                Some(key) => self
                    .masks
                    .get(key)
                    .and_then(Option::as_ref)
                    .map(|mask| Some(mask.texture.clone())),
            })
            .collect();
        // SAFETY: the GPU is idle on this element's work, waited for above,
        // so the heap's views may be written; each names a live resource.
        unsafe {
            for (at, (resource, format)) in [
                (&work_luma, luma_unorm),
                (&work_chroma, chroma_unorm),
                (&work_luma, luma_uint),
                (&work_chroma, chroma_uint),
            ]
            .into_iter()
            .enumerate()
            {
                self.device.CreateUnorderedAccessView(
                    resource,
                    None,
                    Some(&plane_write_view(format, 0)),
                    cpu(at),
                );
            }
            for (index, mask) in masks.iter().enumerate() {
                // No mask, or none to read, is a view of nothing: the shader
                // is told not to read it.
                let resource = mask.as_ref().and_then(Option::as_ref);
                self.device.CreateShaderResourceView(
                    resource,
                    Some(&plane_view(DXGI_FORMAT_R8_UNORM, 0)),
                    cpu(PLANE_VIEWS + index),
                );
            }
        }

        let list = self.commands.begin(None)?;
        // SAFETY: the list is open and the GPU idle on this element's work;
        // every view and barrier names a live resource this element, the
        // picture or the copy holds until the queue has run what reads it.
        unsafe {
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            // The picture into the copy's planes.
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            for plane in [&work_luma, &work_chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
            }
            copy_plane(&list, (&work_luma, 0), (&texture, 0));
            copy_plane(&list, (&work_chroma, 0), (&texture, 1));
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            for plane in [&work_luma, &work_chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
            // What is hidden, its cells' means taken, then each sample
            // painted.
            if let (false, Some((table, _)), Some((means, _))) =
                (dispatches.is_empty(), &self.table, &self.means)
            {
                list.SetComputeRootSignature(&self.means1.root_signature);
                list.SetComputeRootDescriptorTable(1, gpu(2));
                list.SetComputeRootShaderResourceView(2, table.GetGPUVirtualAddress());
                list.SetComputeRootUnorderedAccessView(3, means.GetGPUVirtualAddress());
                for dispatch in &dispatches {
                    let c = &dispatch.constants;
                    let (size, cells) = ((c[2], c[3]), (c[4], c[5]));
                    list.SetComputeRoot32BitConstants(0, 16, c.as_ptr().cast(), 0);
                    if dispatch.take_means {
                        list.SetPipelineState(if dispatch.chroma {
                            &self.means2
                        } else {
                            &self.means1.pipeline
                        });
                        list.Dispatch((cells.0 * cells.1).div_ceil(64), 1, 1);
                        uav_barrier(&list, means);
                    }
                    list.SetPipelineState(if dispatch.chroma {
                        &self.paint2
                    } else {
                        &self.paint1
                    });
                    list.Dispatch(size.0.div_ceil(8), size.1.div_ceil(8), 1);
                    uav_barrier(&list, &work_luma);
                    uav_barrier(&list, &work_chroma);
                }
            }
            // Each mark over what was hidden.
            if !strokes.is_empty() {
                list.SetComputeRootSignature(&self.paint.root_signature);
                list.SetPipelineState(&self.paint.pipeline);
                list.SetComputeRootDescriptorTable(1, gpu(0));
                for (index, (stroke, mask)) in strokes.iter().zip(&masks).enumerate() {
                    let Some(mask) = mask else {
                        pp_error!(
                            self,
                            "{:?} not drawn: its mask is gone from the cache",
                            stroke.mask
                        );
                        continue;
                    };
                    let constants =
                        paint_constants(stroke.rect, colour_of(stroke.color), mask.is_some());
                    list.SetComputeRootDescriptorTable(2, gpu(PLANE_VIEWS + index));
                    list.SetComputeRoot32BitConstants(0, 12, constants.as_ptr().cast(), 0);
                    list.Dispatch(
                        stroke.rect.width.div_ceil(8),
                        stroke.rect.height.div_ceil(8),
                        1,
                    );
                    uav_barrier(&list, &work_luma);
                    uav_barrier(&list, &work_chroma);
                }
            }
            // The copy's planes into the frame handed on.
            for plane in [&work_luma, &work_chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
            }
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            copy_plane(&list, (&to, 0), (&work_luma, 0));
            copy_plane(&list, (&to, 1), (&work_chroma, 0));
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
            for plane in [&work_luma, &work_chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
        }
        // After whatever wrote the picture, and whatever last wrote the frame
        // drawn on; the frame's own fence says when it is drawn.
        let drawn = to_value + 1;
        self.commands.submit(
            &[(&fence, fence_value), (&to_fence, to_value)],
            &[(&to_fence, drawn)],
        )?;
        set_d3d12va_fence_value(&mut destination, drawn);
        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers.
        unsafe {
            ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
        }
        Ok(destination)
    }
}

impl Drop for Overlaying {
    fn drop(&mut self) {
        // What is queued still reads the masks, the table and the planes:
        // it runs out before they go.
        let _ = self.commands.wait_idle();
    }
}

impl Filter for Overlaying {
    /// NV12 D3D12 pictures, or P010 ones where nothing is drawn.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::YUV420),
        )
    }

    /// The picture it was handed, or a copy of it in the same layout.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(D3d12DetectionOverlayError::UnsupportedBuffer(kind).into());
        };
        let metadata = buf.metadata_arc().cloned();
        let detections = metadata
            .as_deref()
            .and_then(|metadata| metadata.get::<Detections>());
        let analytics = metadata
            .as_deref()
            .and_then(|metadata| metadata.get::<Analytics>());
        if let Some(detections) = detections
            && !self.warned_names
            && self.options.names_unmatched(detections)
        {
            pp_warn!(
                self,
                "a rule names a class, and the detections carry no class names: it matches nothing"
            );
            self.warned_names = true;
        }
        if !self.options.draws(detections, analytics) {
            // Nothing to draw: the same picture, not a copy of it.
            out.push(buf);
            return Ok(());
        }
        let drawn = self
            .draw(frame, detections, analytics)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        // Held until the next draw, or `Drop`, has waited for the copy.
        self.reading = Some(buf.clone());
        let mut output = MediaBuffer::Video(Arc::new(drawn).into());
        output.set_metadata(metadata);
        out.push(output);
        Ok(())
    }

    fn reset(&mut self) {}

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        BoxColors, BoxStyle, HideShape, Hiding, LabelStyle, OverlayParts, RedactStyle, Treatment,
        tests::system_font,
    };
    use super::*;
    use crate::buffer::Metadata;
    use crate::color::Color;
    use crate::element::RawSink;
    use crate::elements::{Detection, SwDetectionOverlay};
    use crate::test_support::{capture, try_d3d12_gpu};

    /// `frame` — NV12 or P010 in system memory — as a D3D12 picture on
    /// `gpu`, with its timing and colour; `None` where FFmpeg cannot make
    /// one.
    fn upload(gpu: &D3d12Gpu, frame: &ffmpeg::frame::Video) -> Option<MediaBuffer> {
        // SAFETY: a live device; the frames context is wrapped before it is
        // filled in, and the frame is this function's own until returned.
        unsafe {
            let device = create_hw_device_ctx(gpu.device()).ok()?;
            let Ok(frames) = create_hw_frames_ctx_of(
                &device,
                frame.format().into(),
                frame.width(),
                frame.height(),
                0,
            ) else {
                eprintln!("skipping: FFmpeg could not make D3D12 frames");
                return None;
            };
            let mut on_gpu = ffmpeg::frame::Video::empty();
            assert!(ffi::av_hwframe_get_buffer(frames.as_ptr(), on_gpu.as_mut_ptr(), 0) >= 0);
            assert!(ffi::av_hwframe_transfer_data(on_gpu.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
            assert!(ffi::av_frame_copy_props(on_gpu.as_mut_ptr(), frame.as_ptr()) >= 0);
            Some(MediaBuffer::video(on_gpu))
        }
    }

    /// An overlay on `gpu`, or `None`, saying why, where FFmpeg cannot take
    /// the device for its frames — a software adapter's, as CI's is.
    fn open(gpu: &D3d12Gpu, options: DetectionOverlayOptions) -> Option<D3d12DetectionOverlay> {
        match D3d12DetectionOverlay::new("overlay", gpu, options) {
            Ok(overlay) => Some(overlay),
            Err(D3d12DetectionOverlayError::HwDeviceInit(code)) => {
                eprintln!("skipping: FFmpeg could not take this device for D3D12 frames ({code})");
                None
            }
            Err(error) => panic!("did not open: {error}"),
        }
    }

    /// The picture `buf` holds, copied to system memory.
    fn download(buf: &MediaBuffer) -> ffmpeg::frame::Video {
        let MediaBuffer::Video(frame) = buf else {
            panic!("a picture");
        };
        let mut down = ffmpeg::frame::Video::empty();
        // SAFETY: a live D3D12 frame read into a fresh one FFmpeg allocates.
        unsafe {
            assert!(ffi::av_hwframe_transfer_data(down.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
            assert!(ffi::av_frame_copy_props(down.as_mut_ptr(), frame.as_ptr()) >= 0);
        }
        down
    }

    /// One detection of class 0, from (16, 8) to (48, 24) of a 64 by 32
    /// picture.
    fn found() -> Detections {
        Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![Detection::new(0, 0.9, 0.25, 0.25, 0.5, 0.5)],
        )
    }

    /// A grey NV12 picture, `width` by `height`.
    fn grey(width: u32, height: u32) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(5));
        frame
    }

    /// A `pixel` picture 96 by 48 in which no byte is its neighbour's, so a
    /// cell that read the wrong samples would show it.
    fn noise(pixel: ffmpeg::format::Pixel) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(pixel, 96, 48);
        for plane in 0..frame.planes() {
            for (index, byte) in frame.data_mut(plane).iter_mut().enumerate() {
                let mut v = (index as u32).wrapping_mul(2_654_435_761) ^ (plane as u32 * 977);
                v ^= v >> 15;
                *byte = (v.wrapping_mul(2_246_822_519) >> 24) as u8;
            }
        }
        frame.set_pts(Some(7));
        frame
    }

    /// The bytes a picture shows, plane by plane, row by row, without the
    /// padding at the end of each row.
    fn shown(frame: &ffmpeg::frame::Video) -> Vec<Vec<u8>> {
        let (width, height) = (frame.width() as usize, frame.height() as usize);
        let bytes = if frame.format() == ffmpeg::format::Pixel::P010LE {
            2 * width
        } else {
            width
        };
        [(height, bytes), (height / 2, bytes)]
            .into_iter()
            .enumerate()
            .map(|(plane, (rows, bytes))| {
                let stride = frame.stride(plane);
                (0..rows)
                    .flat_map(|row| frame.data(plane)[row * stride..][..bytes].to_vec())
                    .collect()
            })
            .collect()
    }

    /// What `options` make of `picture` carrying `metadata`: on the CPU, and
    /// on the GPU, both in system memory — or `None` where FFmpeg cannot make
    /// D3D12 frames.
    fn both(
        gpu: &D3d12Gpu,
        options: &DetectionOverlayOptions,
        picture: &ffmpeg::frame::Video,
        metadata: &Metadata,
    ) -> Option<(ffmpeg::frame::Video, ffmpeg::frame::Video)> {
        let mut cpu = SwDetectionOverlay::new("cpu", options.clone()).expect("opens");
        let on_cpu = capture(&mut cpu);
        cpu.consume(MediaBuffer::video(picture.clone()).with_metadata(metadata.clone()))
            .expect("drawn on the CPU");
        let MediaBuffer::Video(expected) = on_cpu.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        let input = upload(gpu, picture)?.with_metadata(metadata.clone());
        let mut overlay = D3d12DetectionOverlay::new("gpu", gpu, options.clone()).expect("opens");
        let on_gpu = capture(&mut overlay);
        overlay.consume(input).expect("drawn on the GPU");
        let actual = download(&on_gpu.lock().unwrap().remove(0));
        Some(((**expected).clone(), actual))
    }

    /// The lines are painted in the box's colour, limited-range BT.709, on a
    /// copy with the picture's timing and its detections, and the picture
    /// handed in is untouched.
    #[test]
    fn boxes_are_drawn_on_a_copy() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let color = Color::new(255, 0, 0);
        let options = DetectionOverlayOptions {
            others: Treatment::boxes(BoxStyle {
                color: BoxColors::One(color),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let Some(mut overlay) = open(&gpu, options) else {
            return;
        };
        let kept = capture(&mut overlay);
        let Some(input) = upload(&gpu, &grey(64, 32)) else {
            return;
        };
        let input = input.with_metadata(Metadata::new().with(found()));
        overlay.consume(input.clone()).expect("drawn");
        let output = kept.lock().unwrap().remove(0);
        assert_eq!(
            output.metadata().and_then(|m| m.get::<Detections>()),
            Some(&found()),
            "the detections go on with the copy"
        );
        let (drawn, untouched) = (download(&output), download(&input));
        let pixel = |frame: &ffmpeg::frame::Video, x: usize, y: usize| -> Vec<u8> {
            let chroma = &frame.data(1)[(y / 2) * frame.stride(1) + (x / 2) * 2..];
            vec![frame.data(0)[y * frame.stride(0) + x], chroma[0], chroma[1]]
        };
        let (y, cb, cr) = bt709_limited(color);
        let line = vec![y, cb, cr];
        assert_eq!(pixel(&drawn, 16, 8), line, "top-left corner");
        assert_eq!(pixel(&drawn, 47, 23), line, "bottom-right corner");
        assert_eq!(pixel(&drawn, 32, 16), pixel(&untouched, 32, 16), "inside");
        assert_eq!(pixel(&drawn, 2, 2), pixel(&untouched, 2, 2), "outside");
        assert_eq!(drawn.pts(), Some(5), "the picture's timing");
        assert_ne!(pixel(&untouched, 16, 8), line, "the picture handed in");
    }

    #[test]
    fn a_picture_with_nothing_to_draw_is_handed_on_as_it_came() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(mut overlay) = open(&gpu, DetectionOverlayOptions::default()) else {
            return;
        };
        let kept = capture(&mut overlay);
        let Some(bare) = upload(&gpu, &grey(64, 32)) else {
            return;
        };
        let MediaBuffer::Video(input) = &bare else {
            unreachable!();
        };
        let id = crate::buffer::picture_id(input);
        overlay.consume(bare).expect("bare");
        let MediaBuffer::Video(output) = kept.lock().unwrap().remove(0) else {
            unreachable!();
        };
        assert_eq!(crate::buffer::picture_id(&output), id);
    }

    /// A picture in system memory carrying something to draw is refused,
    /// saying what it is, and the overlay draws the next one.
    #[test]
    fn a_picture_in_system_memory_is_refused() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(mut overlay) = open(&gpu, DetectionOverlayOptions::default()) else {
            return;
        };
        let kept = capture(&mut overlay);
        let error = overlay
            .consume(MediaBuffer::video(grey(64, 32)).with_metadata(Metadata::new().with(found())))
            .expect_err("refused");
        assert!(
            error.to_string().contains("NV12"),
            "says what it was: {error}"
        );
        let Some(input) = upload(&gpu, &grey(64, 32)) else {
            return;
        };
        overlay
            .consume(input.with_metadata(Metadata::new().with(found())))
            .expect("drawn after");
        assert_eq!(kept.lock().unwrap().len(), 1);
    }

    /// With a font, the band above the box is the box's colour and the text
    /// on it is not: written, and on the picture's top where the box is.
    #[test]
    fn a_label_is_drawn_above_its_box() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font");
            return;
        };
        let color = Color::new(255, 196, 0);
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                color: BoxColors::One(color),
                label: Some(LabelStyle::new(12.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let Some(mut overlay) = open(&gpu, options) else {
            return;
        };
        let kept = capture(&mut overlay);
        let mut detections = found();
        detections.items[0].y = 0.5;
        detections.items[0].height = 0.4;
        let Some(input) = upload(&gpu, &grey(128, 64)) else {
            return;
        };
        overlay
            .consume(input.with_metadata(Metadata::new().with(detections)))
            .expect("drawn");
        let drawn = download(&kept.lock().unwrap().remove(0));
        let luma = |x: usize, y: usize| drawn.data(0)[y * drawn.stride(0) + x];
        let band = bt709_limited(color).0;
        // The box starts at (32, 32); its band is above it, and its padding
        // row is the band's colour.
        let (top, bottom) = (0..32).fold((None, None), |(top, bottom), y| {
            if luma(32, y) == band {
                (top.or(Some(y)), Some(y))
            } else {
                (top, bottom)
            }
        });
        let (Some(top), Some(bottom)) = (top, bottom) else {
            panic!("no band above the box");
        };
        assert_eq!(bottom, 31, "the band ends where the box starts");
        let right = (32..128)
            .take_while(|&x| luma(x, top) == band)
            .last()
            .unwrap();
        let written = (top..=bottom)
            .flat_map(|y| (32..=right).map(move |x| (x, y)))
            .filter(|&(x, y)| luma(x, y) < band.saturating_sub(40))
            .count();
        assert!(written > 10, "the text is drawn ({written} dark pixels)");
    }

    /// Zones and lines are drawn on the GPU as the CPU draws them: the same
    /// marks through the same masks, no byte more than 1 apart.
    #[test]
    fn zones_and_lines_are_drawn_as_the_cpu_draws_them() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let options = DetectionOverlayOptions {
            parts: OverlayParts {
                line_width: 4,
                ..OverlayParts::all()
            },
            ..DetectionOverlayOptions::default()
        };
        let analytics: Analytics = super::super::super::tests::watched();
        let Some((expected, actual)) = both(
            &gpu,
            &options,
            &grey(128, 64),
            &Metadata::new().with(analytics),
        ) else {
            return;
        };
        let (expected, actual) = (shown(&expected), shown(&actual));
        let changed: usize = expected
            .iter()
            .flatten()
            .map(|&byte| usize::from(byte != 128))
            .sum();
        let differ: usize = expected
            .iter()
            .flatten()
            .zip(actual.iter().flatten())
            .map(|(a, b)| usize::from(a.abs_diff(*b) > 1))
            .sum();
        assert!(changed > 200, "something drawn ({changed} bytes)");
        assert_eq!(differ, 0, "bytes more than 1 apart");
    }

    /// A mosaic, a blur and a fill on the GPU write the same bytes as the
    /// CPU's overlay on the same picture, of the whole box or the ellipse
    /// inside it, hard-edged and faded — two faces, grown by the margin, cut
    /// into cells of uneven width, the colour at half the size — and a box
    /// drawn over a hidden face is drawn over it.
    #[test]
    fn a_box_is_hidden_on_the_gpu_as_on_the_cpu() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let faces = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("face")]),
            vec![
                Detection::new(0, 0.9, 0.1, 0.2, 0.37, 0.55),
                Detection::new(0, 0.8, 0.62, 0.05, 0.2, 0.3),
            ],
        );
        let few = std::num::NonZeroU32::new(5).unwrap();
        let styles = [
            RedactStyle::mosaic(),
            RedactStyle::blur(),
            RedactStyle::Mosaic {
                cells: few,
                min_cell: 2,
            },
            RedactStyle::Blur {
                cells: few,
                min_cell: 2,
            },
            RedactStyle::Fill(Color::new(10, 200, 30)),
        ];
        let shapes = [
            (HideShape::Rectangle, 0.0),
            (HideShape::Ellipse, 0.0),
            (HideShape::Rectangle, 0.3),
            (HideShape::Ellipse, 0.3),
        ];
        let picture = noise(ffmpeg::format::Pixel::NV12);
        for (style, (shape, feather)) in styles
            .into_iter()
            .flat_map(|style| shapes.map(|shape| (style, shape)))
        {
            for drawn in [false, true] {
                let treatment = Treatment {
                    hide: Some(Hiding {
                        shape,
                        feather,
                        ..Hiding::new(style)
                    }),
                    ..if drawn {
                        Treatment::default()
                    } else {
                        Treatment::hidden(style)
                    }
                };
                let options = DetectionOverlayOptions {
                    others: treatment,
                    ..DetectionOverlayOptions::default()
                };
                let Some((expected, actual)) = both(
                    &gpu,
                    &options,
                    &picture,
                    &Metadata::new().with(faces.clone()),
                ) else {
                    return;
                };
                let (expected, actual) = (shown(&expected), shown(&actual));
                assert_ne!(
                    expected,
                    shown(&picture),
                    "{style:?} {shape:?} {feather}: something hidden"
                );
                for (plane, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    let differ = expected
                        .iter()
                        .zip(actual)
                        .filter(|(expected, actual)| expected != actual)
                        .count();
                    assert_eq!(
                        differ, 0,
                        "{style:?} {shape:?} {feather}, boxes drawn {drawn}, plane {plane}: \
                         {differ} bytes differ"
                    );
                }
            }
        }
    }

    /// A picture stored turned is drawn on as it is shown: boxes and labels
    /// drawn on it and turned back are the bytes drawn on the picture stored
    /// the right way up, for each of the eight ways it can be turned.
    #[test]
    fn a_turned_picture_is_drawn_on_as_it_is_shown() {
        use super::super::super::tests::{every_orientation, turn};
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font to draw labels in");
            return;
        };
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                label: Some(LabelStyle::new(14.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let found = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![
                Detection::new(0, 0.9, 0.25, 0.5, 0.5, 0.375),
                Detection::new(0, 0.6, 0.625, 0.0, 0.25, 0.25),
            ],
        );
        let draw = |picture: ffmpeg::frame::Video, found: Detections| {
            let input = upload(&gpu, &picture)?.with_metadata(Metadata::new().with(found));
            let mut overlay =
                D3d12DetectionOverlay::new("gpu", &gpu, options.clone()).expect("opens");
            let on_gpu = capture(&mut overlay);
            overlay.consume(input).expect("drawn");
            Some(download(&on_gpu.lock().unwrap().remove(0)))
        };
        let upright = noise(ffmpeg::format::Pixel::NV12);
        let Some(drawn) = draw(upright.clone(), found.clone()) else {
            return;
        };
        let expected = shown(&drawn);
        for orientation in every_orientation() {
            let mut stored = found.clone();
            for item in &mut stored.items {
                [item.x, item.y, item.width, item.height] =
                    orientation.from_display([item.x, item.y, item.width, item.height]);
            }
            let drawn = draw(turn(&upright, orientation, false), stored).expect("drawn");
            assert_eq!(
                shown(&turn(&drawn, orientation, true)),
                expected,
                "{orientation:?}"
            );
        }
    }

    /// In a P010 picture — HLG, ten bits at the top of two bytes — a mosaic,
    /// a blur and a fill, of the box or the ellipse inside it, hard-edged and
    /// faded, write the CPU's bytes, the samples around left as they were to
    /// the last bit; a picture of it is refused where anything would be
    /// drawn.
    #[test]
    fn a_p010_picture_is_hidden_on_the_gpu_as_on_the_cpu() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let faces = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("face")]),
            vec![
                Detection::new(0, 0.9, 0.1, 0.2, 0.37, 0.55),
                Detection::new(0, 0.8, 0.62, 0.05, 0.2, 0.3),
            ],
        );
        let mut picture = noise(ffmpeg::format::Pixel::P010LE);
        crate::color::ColorDescription {
            space: ffmpeg::color::Space::BT2020NCL,
            range: ffmpeg::color::Range::MPEG,
            primaries: ffmpeg::color::Primaries::BT2020,
            transfer: ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
        }
        .describe(&mut picture);
        let few = std::num::NonZeroU32::new(5).unwrap();
        let styles = [
            RedactStyle::mosaic(),
            RedactStyle::blur(),
            RedactStyle::Mosaic {
                cells: few,
                min_cell: 2,
            },
            RedactStyle::Fill(Color::new(200, 30, 60)),
            RedactStyle::Fill(Color::WHITE),
        ];
        for style in styles {
            for (shape, feather) in [
                (HideShape::Rectangle, 0.0),
                (HideShape::Ellipse, 0.0),
                (HideShape::Rectangle, 0.3),
                (HideShape::Ellipse, 0.3),
            ] {
                let options = DetectionOverlayOptions {
                    others: Treatment {
                        hide: Some(Hiding {
                            shape,
                            feather,
                            ..Hiding::new(style)
                        }),
                        ..Treatment::none()
                    },
                    ..DetectionOverlayOptions::default()
                };
                let Some((expected, actual)) = both(
                    &gpu,
                    &options,
                    &picture,
                    &Metadata::new().with(faces.clone()),
                ) else {
                    return;
                };
                assert_eq!(actual.format(), ffmpeg::format::Pixel::P010LE);
                let (expected, actual) = (shown(&expected), shown(&actual));
                assert_ne!(
                    expected,
                    shown(&picture),
                    "{style:?} {shape:?} {feather}: something hidden"
                );
                for (plane, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    let differ = expected
                        .iter()
                        .zip(actual)
                        .filter(|(expected, actual)| expected != actual)
                        .count();
                    assert_eq!(
                        differ, 0,
                        "{style:?} {shape:?} {feather}, plane {plane}: {differ} bytes differ"
                    );
                }
            }
        }

        // Boxes would be drawn: refused, nothing handed on.
        let mut drawing =
            D3d12DetectionOverlay::new("gpu", &gpu, DetectionOverlayOptions::default())
                .expect("opens");
        let kept = capture(&mut drawing);
        let input = upload(&gpu, &picture)
            .expect("uploaded above")
            .with_metadata(Metadata::new().with(faces));
        assert!(drawing.consume(input).is_err(), "nothing is drawn on P010");
        assert!(kept.lock().unwrap().is_empty());
    }
}
