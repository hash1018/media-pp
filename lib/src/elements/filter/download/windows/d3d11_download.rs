use std::sync::{Arc, Mutex};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D11::{
            D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
            D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext,
            ID3D11Texture2D,
        },
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    },
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::windows::d3d11va::d3d11va_texture,
    pool::UnboundObjectPool,
};

/// Errors specific to `D3d11Download`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11DownloadError {
    /// Creating, copying, or mapping a D3D11 staging texture failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
    /// The input frame is not backed by a D3D11 texture.

    #[error("D3d11Download only accepts Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// A frame tagged as D3D11 contains no valid texture reference.

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode/D3d11VideoCompositor"
    )]
    InvalidD3d11Frame,
    /// The input texture uses a DXGI format this downloader cannot copy.

    #[error("D3d11Download only reads DXGI_FORMAT_B8G8R8A8_UNORM textures, got {0:?}")]
    UnsupportedTextureFormat(DXGI_FORMAT),
    /// The input texture belongs to another D3D11 device.

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11Download was created with — every D3D11 element in \
         one pipeline must share exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,
    /// Input dimensions differ from the fixed download dimensions.

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d11Download was \
         opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for this downloader.
        expected_width: u32,
        /// Height configured for this downloader.
        expected_height: u32,
    },
    /// The backing texture is smaller than the visible frame.

    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than {expected_width}x{expected_height}"
    )]
    TextureTooSmall {
        /// Backing texture width in pixels.
        actual_width: u32,
        /// Backing texture height in pixels.
        actual_height: u32,
        /// Visible frame width in pixels.
        expected_width: u32,
        /// Visible frame height in pixels.
        expected_height: u32,
    },
    /// The frame selects a texture-array slice outside the resource bounds.

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex {
        /// Invalid texture-array index.
        index: isize,
        /// Number of slices in the texture array.
        array_size: u32,
    },
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("D3d11Download only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Downloads GPU-resident `Pixel::D3D11` BGRA video frames (e.g. from
/// [`crate::elements::D3d11VideoCompositor`]) to CPU-resident `Pixel::BGRA`
/// frames — the mirror of [`crate::elements::D3d11Upload`]. Needed because
/// this crate's video encoder ([`crate::elements::SwEncoder`]) is
/// software-only and has no zero-copy GPU input path; chain a
/// [`crate::elements::SwScaler`] after this to convert to whatever pixel
/// format an encoder actually needs.
///
/// Unlike `D3d11Upload` (a plain `ID3D11Device::CreateTexture2D` call, safe
/// to issue from any thread), reading a texture back to the CPU needs
/// `ID3D11DeviceContext::CopySubresourceRegion`/`Map`/`Unmap` — context-level calls,
/// not device-level ones. `context` must be the exact same
/// `Arc<Mutex<ID3D11DeviceContext>>` every other context-touching D3D11
/// consumer in this pipeline shares (e.g.
/// `render_common::D3d11GpuContext::context()`) — see that type's own docs
/// on why a lone per-element context handle isn't enough to prevent two
/// unrelated bind/draw/copy sequences from interleaving on the one
/// underlying immediate context.
pub struct D3d11Download {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    width: u32,
    height: u32,
    /// Built once at construction (fixed `width`/`height` for this
    /// element's lifetime, same convention as `D3d11Upload`) and reused
    /// across every `consume` call rather than recreated per frame.
    staging: ID3D11Texture2D,
    pad: SrcPad,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper (free
// -threaded for the device-level/context-level calls used here, the latter
// behind `context`'s own `Mutex`) or plain data. `&mut self` on every
// method that touches non-`Arc`/`Mutex` state already rules out concurrent
// access to those parts from multiple threads.
unsafe impl Send for D3d11Download {}

impl D3d11Download {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared context, every other D3D11 element in this pipeline uses —
    /// see this type's own docs on why. `width`/`height` are fixed for
    /// this element's lifetime; every frame `consume` receives must match
    /// exactly.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, D3d11DownloadError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Download, &name, None);
        let staging = create_staging_texture(device, width, height)?;
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
            ),
        );
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        pp_info!(pp_log: &pp_log, "opened: {width}x{height}");
        Ok(Self {
            name,
            pp_log,
            device: device.clone(),
            context,
            width,
            height,
            staging,
            pad,
            pool,
        })
    }

    /// Copies one texture-array subresource into the cached staging texture,
    /// `Map`s it, and copies every row into a pooled CPU frame — under `context`'s
    /// lock for the whole sequence, since both calls touch the shared
    /// immediate context (see this type's own docs).
    fn download(
        &self,
        texture: &ID3D11Texture2D,
        source_subresource: u32,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        windows::core::Error,
    > {
        let mut output = self.pool.get();
        let context = self
            .context
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: validation established the source slice bounds and format;
        // the staging texture matches the visible dimensions. A successful
        // map keeps `pData` valid and row-pitched through the paired `Unmap`.
        unsafe {
            let source_box = D3D11_BOX {
                left: 0,
                top: 0,
                front: 0,
                right: self.width,
                bottom: self.height,
                back: 1,
            };
            context.CopySubresourceRegion(
                &self.staging,
                0,
                0,
                0,
                0,
                texture,
                source_subresource,
                Some(&source_box as *const D3D11_BOX),
            );

            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context.Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;

            let row_bytes = (self.width as usize) * 4;
            let src_stride = mapped.RowPitch as usize;
            // Overwrites the pooled slot's previous pixel contents in
            // place — same reasoning as `D3d11Upload`'s own pooled frame.
            let dst_stride = output.stride(0);
            let src = mapped.pData as *const u8;
            let dst = output.data_mut(0);
            for row in 0..self.height as usize {
                let src_row = std::slice::from_raw_parts(src.add(row * src_stride), row_bytes);
                dst[row * dst_stride..row * dst_stride + row_bytes].copy_from_slice(src_row);
            }

            context.Unmap(&self.staging, 0);
        }
        Ok(output)
    }
}

impl Element for D3d11Download {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Download
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11Download {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11Download {
    /// The mirror of D3d11Upload: only a device texture has anything to bring back.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::D3d11))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::D3D11 {
                    let format = frame.format();
                    pp_error!(self, "unsupported pixel format: {format:?}");
                    return Err(D3d11DownloadError::UnsupportedFormat(format).into());
                }
                if frame.width() != self.width || frame.height() != self.height {
                    let error = D3d11DownloadError::DimensionMismatch {
                        actual_width: frame.width(),
                        actual_height: frame.height(),
                        expected_width: self.width,
                        expected_height: self.height,
                    };
                    pp_error!(self, "{error}");
                    return Err(error.into());
                }

                let (texture_raw, index) =
                    d3d11va_texture(&frame).ok_or(D3d11DownloadError::InvalidD3d11Frame)?;
                // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*`
                // — still owned by `frame`'s own buffer reference, not by
                // us. `.clone()` (`AddRef`) gives us an independently
                // ref-counted handle, valid for the rest of this call.
                let texture = unsafe {
                    ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                        .expect("D3d11 frame's texture pointer must not be null")
                        .clone()
                };

                // SAFETY: `texture` is a live cloned COM interface and
                // `GetDevice` returns an owned reference to its creator.
                let texture_device =
                    unsafe { texture.GetDevice() }.map_err(D3d11DownloadError::from)?;
                if texture_device.as_raw() != self.device.as_raw() {
                    pp_error!(self, "device mismatch");
                    return Err(D3d11DownloadError::DeviceMismatch.into());
                }

                let mut desc = D3D11_TEXTURE2D_DESC::default();
                // SAFETY: `desc` is a live out-parameter for the live texture.
                unsafe { texture.GetDesc(&mut desc) };
                if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
                    pp_error!(self, "unsupported texture format: {:?}", desc.Format);
                    return Err(D3d11DownloadError::UnsupportedTextureFormat(desc.Format).into());
                }
                if desc.Width < self.width || desc.Height < self.height {
                    let error = D3d11DownloadError::TextureTooSmall {
                        actual_width: desc.Width,
                        actual_height: desc.Height,
                        expected_width: self.width,
                        expected_height: self.height,
                    };
                    pp_error!(self, "{error}");
                    return Err(error.into());
                }
                if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
                    let error = D3d11DownloadError::InvalidArrayIndex {
                        index,
                        array_size: desc.ArraySize,
                    };
                    pp_error!(self, "{error}");
                    return Err(error.into());
                }
                let source_subresource = (index as u32) * desc.MipLevels;

                let mut cpu_frame = self
                    .download(&texture, source_subresource)
                    .inspect_err(|error| pp_error!(self, "GPU download failed: {error}"))
                    .map_err(D3d11DownloadError::from)?;
                cpu_frame.set_pts(frame.pts());
                cpu_frame.set_color_space(frame.color_space());
                cpu_frame.set_color_range(frame.color_range());

                self.pad.push(MediaBuffer::Video(Arc::new(cpu_frame)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(D3d11DownloadError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(D3d11DownloadError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame GPU->CPU transfer,
        // same reasoning as `D3d11Upload::control`.
        self.pad.control(msg)
    }
}

fn create_staging_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> std::result::Result<ID3D11Texture2D, D3d11DownloadError> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
    };
    let mut texture: Option<ID3D11Texture2D> = None;
    // SAFETY: `desc` fully describes a staging allocation, no initial data is
    // supplied, and `texture` is a live COM out-parameter.
    unsafe {
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
    }
    Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use windows::Win32::Graphics::Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_SUBRESOURCE_DATA, D3D11_USAGE_DEFAULT,
    };

    use super::*;
    use crate::platform::windows::d3d11va::wrap_d3d11_texture;
    use crate::test_support::try_d3d11_device;

    /// Real hardware round trip: build a small BGRA texture directly (the
    /// same shape a `D3d11VideoCompositor` output would have), wrap it as
    /// a `Pixel::D3D11` frame, download it, and confirm the CPU bytes
    /// match what was uploaded.
    #[test]
    fn gpu_bgra_texture_round_trips_to_a_cpu_bgra_frame() {
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };

        let (width, height) = (4u32, 4u32);
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|index| [index as u8, (index * 2) as u8, (index * 3) as u8, 255])
            .collect();
        // SAFETY: the initial-data pointer addresses the live `pixels` buffer
        // with the exact row pitch and extent in the texture description; the
        // returned interface is captured in a live out-parameter.
        let source_texture = unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let initial = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr() as *const c_void,
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            };
            let mut texture = None;
            device
                .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
                .expect("CreateTexture2D failed");
            texture.expect("CreateTexture2D succeeded without producing a texture")
        };
        let source_frame = wrap_d3d11_texture(source_texture, width, height).unwrap();

        let mut download = D3d11Download::new("test-download", &device, context, width, height)
            .expect("D3d11Download::new should succeed");

        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();
        download.src_pads()[0].link(Box::new(CapturingSink {
            received: received_clone,
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        let source_pool =
            crate::pool::UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut source_pooled = source_pool.get();
        *source_pooled = source_frame;
        download
            .consume(MediaBuffer::Video(Arc::new(source_pooled)))
            .expect("download consume should succeed");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 1);
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
        let stride = frame.stride(0);
        for row in 0..height as usize {
            let row_bytes = &frame.data(0)[row * stride..row * stride + width as usize * 4];
            let expected = &pixels[row * width as usize * 4..(row + 1) * width as usize * 4];
            assert_eq!(row_bytes, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn downloads_only_the_selected_bgra_texture_array_slice() {
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };

        let (width, height) = (2u32, 2u32);
        let red = [0u8, 0, 255, 255].repeat((width * height) as usize);
        let blue = [255u8, 0, 0, 255].repeat((width * height) as usize);
        let initial = [
            D3D11_SUBRESOURCE_DATA {
                pSysMem: red.as_ptr().cast::<c_void>(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            },
            D3D11_SUBRESOURCE_DATA {
                pSysMem: blue.as_ptr().cast::<c_void>(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            },
        ];
        // SAFETY: both initial subresources point at live, correctly pitched
        // pixel buffers matching the two array slices declared below.
        let texture = unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 2,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut texture = None;
            device
                .CreateTexture2D(&desc, Some(initial.as_ptr()), Some(&mut texture))
                .expect("CreateTexture2D array failed");
            texture.expect("CreateTexture2D succeeded without producing a texture")
        };
        let mut source_frame = wrap_d3d11_texture(texture, width, height).unwrap();
        // SAFETY: this test exclusively owns the unpublished frame; address 1
        // encodes slice index 1 in `data[1]` and is never dereferenced.
        unsafe {
            (*source_frame.as_mut_ptr()).data[1] = std::ptr::dangling_mut::<u8>();
        }

        let mut download = D3d11Download::new("test-download", &device, context, width, height)
            .expect("D3d11Download::new should succeed");
        let received = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        let source_pool =
            crate::pool::UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut source_pooled = source_pool.get();
        *source_pooled = source_frame;
        download
            .consume(MediaBuffer::Video(Arc::new(source_pooled)))
            .expect("download consume should succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        let row = &frame.data(0)[..width as usize * 4];
        assert_eq!(row, &blue[..width as usize * 4]);
    }

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<std::sync::Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }
}
