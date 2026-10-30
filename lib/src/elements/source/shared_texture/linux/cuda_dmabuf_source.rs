use std::{
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
    time::Duration,
};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, SourceStage},
    elements::{AppSource, AppSourceHandle, source::app_source::Receiving},
    ffmpeg::ffi,
    platform::{
        cuda::{CudaDevice, CudaFrameFormat, frame::create_hw_frames_ctx},
        ffmpeg::AvBufferRef,
        linux::dmabuf_cuda::{
            CudaBgraSurface, DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888, DmaBufCudaError,
            DmaBufCudaImporter, DmaBufPlane,
        },
    },
    pool::UnboundObjectPool,
    pp_log::pp_info,
    produce::source_stage,
};

use super::dmabuf::{DmaBuf, DmaBufAlpha};

/// Errors specific to `CudaDmaBufSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaDmaBufSourceError {
    /// The DMA-BUF could not be read into CUDA: EGL would not import it, GL
    /// would not read it, or CUDA would not take what GL read.
    #[error(transparent)]
    Import(#[from] DmaBufCudaError),

    /// The pool the imported pictures land in could not be made or drawn
    /// from.
    #[error("the CUDA pool could not be used: {0}")]
    Pool(String),

    /// The thread that does the import is gone, which only happens when it
    /// could not start or panicked.
    #[error("the DMA-BUF import thread has stopped")]
    ImportStopped,

    /// A picture of another size than the source was made for.
    #[error(
        "a {width}x{height} DMA-BUF was pushed into a {expected_width}x{expected_height} source"
    )]
    WrongSize {
        width: u32,
        height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    /// The source has ended — its `Pipeline` finished, or end-of-stream was
    /// already submitted — so nothing more can be pushed into it.
    #[error("CudaDmaBufSource has already ended")]
    Closed,
}

/// Brings another process's DMA-BUFs into this pipeline, as `Pixel::CUDA`
/// BGRA frames on this pipeline's own device.
///
/// The Linux counterpart of `D3d11SharedTextureSource`, and in the same
/// shape: a pushed source, built on [`AppSource`], whose producer calls
/// [`CudaDmaBufHandle::push`] with a buffer per picture. It copies for the
/// same reason that one does — the producer reuses its buffers — and the
/// copy never leaves the GPU.
///
/// # How the copy is made
///
/// CUDA will not register an EGL image made from a DMA-BUF on desktop
/// NVIDIA drivers, so the import goes through OpenGL: the buffer becomes a
/// texture, GL reads it into a pixel buffer CUDA has registered, and CUDA
/// copies that into a frame of its own — see
/// `platform::linux::dmabuf_cuda`, which `PipeWireScreenCaptureSource` also
/// uses. For [`DmaBufAlpha::Premultiplied`] the texture is first drawn
/// through a shader that divides the alpha out.
///
/// All of that happens on a thread of this source's own, which owns the EGL
/// and CUDA contexts it needs current: a push waits for it, so the buffer
/// is still the producer's picture while it is read, but the producer's
/// thread — a browser engine's, typically, with GL state of its own — never
/// has another context made current on it.
///
/// # What it does not do
///
/// Resize, convert or pace, as `D3d11SharedTextureSource` does not. A
/// buffer of another size is refused, and a multi-plane format is not
/// something this can be given.
pub struct CudaDmaBufSource(SourceStage<Receiving>);

source_stage!(CudaDmaBufSource);

/// Pushes another process's DMA-BUFs into a [`CudaDmaBufSource`].
///
/// Cheap to clone, and every clone feeds the same source. Push
/// [`Self::finish`] when done, or drop every clone.
///
/// # Push while the buffer is still the producer's picture
///
/// [`Self::push`] imports before it returns, so it must be called while the
/// producer still holds that picture — inside its paint callback, not
/// queued for later.
#[derive(Clone)]
pub struct CudaDmaBufHandle {
    pusher: AppSourceHandle,
    import: Arc<ImportThread>,
}

/// The thread imports happen on, and the way to ask it for one.
struct ImportThread {
    requests: Mutex<Option<mpsc::Sender<Request>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    width: u32,
    height: u32,
    /// Reused for the small `AVFrame` wrapper only; the CUDA surface inside
    /// is the pool's own and goes back to it when downstream lets go.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

struct Request {
    buffer: DmaBuf,
    reply: mpsc::Sender<std::result::Result<ffmpeg::frame::Video, CudaDmaBufSourceError>>,
}

/// How long a push waits for the import thread before giving up on the
/// picture. An import is a few milliseconds; this only matters if the
/// thread has hung, and then the producer loses a picture rather than its
/// own thread.
const IMPORT_WAIT: Duration = Duration::from_secs(2);

impl CudaDmaBufSource {
    /// `device` must be the [`CudaDevice`] every other CUDA element in this
    /// pipeline was built from. `width`/`height` are what every pushed
    /// buffer must be and what the frames coming out are. `capacity` bounds
    /// how many frames may sit unconsumed, as [`AppSource::new`]'s does.
    ///
    /// Starts the import thread and makes its contexts before returning, so
    /// a machine that cannot import at all — no EGL device for the GPU, a
    /// driver without the DMA-BUF extensions — is an error here rather than
    /// at the first picture.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
        capacity: usize,
    ) -> std::result::Result<(Self, CudaDmaBufHandle), CudaDmaBufSourceError> {
        let (inner, pusher) = AppSource::receiving(
            name,
            capacity,
            ElementType::CudaDmaBufSource,
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
            // Live, as `D3d11SharedTextureSource` is: what it emits is
            // whatever the producer drew just now, and a paused pipeline
            // cannot ask it for a first picture.
            true,
        );
        let import = ImportThread::start(device.retain(), width, height, alpha)?;
        pp_info!(pp_log: inner.pp_log(), "opened: {width}x{height}, alpha {alpha:?}");
        Ok((
            Self(inner),
            CudaDmaBufHandle {
                pusher,
                import: Arc::new(import),
            },
        ))
    }
}

impl CudaDmaBufHandle {
    /// Imports `buffer` into this pipeline and pushes it downstream as one
    /// frame, stamped `pts` (`None` leaves it unstamped, which is what a
    /// compositor-driven graph wants).
    ///
    /// Blocks while the source's queue is full, as
    /// [`AppSourceHandle::push`] does.
    pub fn push(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<(), CudaDmaBufSourceError> {
        let frame = self.import.import(buffer, pts)?;
        self.pusher
            .push(frame)
            .map_err(|_| CudaDmaBufSourceError::Closed)
    }

    /// [`Self::push`] for a producer that cannot wait — a browser engine's
    /// paint callback, which is that engine's one thread. `Ok(false)` means
    /// the queue was full and this picture was dropped; `Err` means the
    /// import failed or the source has ended. As with
    /// `D3d11SharedTextureHandle::try_push`, the import still happens.
    pub fn try_push(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<bool, CudaDmaBufSourceError> {
        let frame = self.import.import(buffer, pts)?;
        self.pusher
            .try_push(frame)
            .map_err(|_| CudaDmaBufSourceError::Closed)
    }

    /// Ends the stream, as [`AppSourceHandle::finish`] does.
    pub fn finish(&self) -> std::result::Result<(), CudaDmaBufSourceError> {
        self.pusher
            .finish()
            .map_err(|_| CudaDmaBufSourceError::Closed)
    }
}

impl ImportThread {
    fn start(
        hw_device_ctx: Arc<AvBufferRef>,
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
    ) -> std::result::Result<Self, CudaDmaBufSourceError> {
        let (requests, incoming) = mpsc::channel::<Request>();
        let (ready_tx, ready) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("cuda-dmabuf-import".to_owned())
            .spawn(move || {
                let mut worker = match Worker::new(hw_device_ctx, width, height, alpha) {
                    Ok(worker) => {
                        let _ = ready_tx.send(Ok(()));
                        worker
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                for request in incoming {
                    let _ = request.reply.send(worker.import(request.buffer));
                }
            })
            .map_err(|_| CudaDmaBufSourceError::ImportStopped)?;
        match ready.recv() {
            Ok(Ok(())) => Ok(Self {
                requests: Mutex::new(Some(requests)),
                thread: Mutex::new(Some(thread)),
                width,
                height,
                wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = thread.join();
                Err(CudaDmaBufSourceError::ImportStopped)
            }
        }
    }

    fn import(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<MediaBuffer, CudaDmaBufSourceError> {
        if (buffer.width, buffer.height) != (self.width, self.height) {
            return Err(CudaDmaBufSourceError::WrongSize {
                width: buffer.width,
                height: buffer.height,
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        let (reply, answer) = mpsc::channel();
        {
            let requests = self
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            requests
                .as_ref()
                .ok_or(CudaDmaBufSourceError::ImportStopped)?
                .send(Request {
                    buffer: *buffer,
                    reply,
                })
                .map_err(|_| CudaDmaBufSourceError::ImportStopped)?;
        }
        let mut frame = answer
            .recv_timeout(IMPORT_WAIT)
            .map_err(|_| CudaDmaBufSourceError::ImportStopped)??;
        frame.set_pts(pts);
        let mut slot = self.wrappers.get();
        *slot = frame;
        Ok(MediaBuffer::Video(Arc::new(slot).into()))
    }
}

impl Drop for ImportThread {
    fn drop(&mut self) {
        // The sender first: the thread's loop ends when it goes, and it drops
        // its contexts on its own way out.
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = thread.join();
        }
    }
}

/// What lives on the import thread: the importer, whose contexts are
/// current there, and the pool its pictures land in.
struct Worker {
    importer: DmaBufCudaImporter,
    /// Held so the device outlives the pool made from it.
    _hw_device_ctx: Arc<AvBufferRef>,
    frames: AvBufferRef,
    width: u32,
    height: u32,
    alpha: DmaBufAlpha,
}

impl Worker {
    fn new(
        hw_device_ctx: Arc<AvBufferRef>,
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
    ) -> std::result::Result<Self, CudaDmaBufSourceError> {
        let importer = DmaBufCudaImporter::new()?;
        // SAFETY: `create_hw_frames_ctx`'s contract is a live device context,
        // which the owned `AvBufferRef` is.
        let frames =
            unsafe { create_hw_frames_ctx(&hw_device_ctx, CudaFrameFormat::Bgra, width, height) }
                .map_err(|error| CudaDmaBufSourceError::Pool(error.to_string()))?;
        Ok(Self {
            importer,
            _hw_device_ctx: hw_device_ctx,
            frames,
            width,
            height,
            alpha,
        })
    }

    fn import(
        &mut self,
        buffer: DmaBuf,
    ) -> std::result::Result<ffmpeg::frame::Video, CudaDmaBufSourceError> {
        let mut frame = ffmpeg::frame::Video::empty();
        // SAFETY: `self.frames` is this worker's own pool, and `frame` is the
        // empty local the surface is being allocated into.
        let code =
            unsafe { ffi::av_hwframe_get_buffer(self.frames.as_ptr(), frame.as_mut_ptr(), 0) };
        if code < 0 {
            return Err(CudaDmaBufSourceError::Pool(format!(
                "av_hwframe_get_buffer failed ({code})"
            )));
        }
        let surface = CudaBgraSurface::from_frame(&frame)
            .ok_or_else(|| CudaDmaBufSourceError::Pool("a surface with no memory".to_owned()))?;
        let plane = DmaBufPlane {
            fd: buffer.fd,
            offset: buffer.offset,
            stride: i32::try_from(buffer.stride).unwrap_or(i32::MAX),
            modifier: buffer.modifier,
            fourcc: match self.alpha {
                DmaBufAlpha::Ignored => DRM_FORMAT_XRGB8888,
                DmaBufAlpha::Straight | DmaBufAlpha::Premultiplied => DRM_FORMAT_ARGB8888,
            },
        };
        // Drawn, whatever the alpha: a producer's buffer may be one the
        // driver will only import as an external texture — on NVIDIA, every
        // `DRM_FORMAT_MOD_LINEAR` buffer is — which cannot be read directly.
        self.importer.draw_into(
            plane,
            self.width,
            self.height,
            surface,
            self.alpha == DmaBufAlpha::Premultiplied,
        )?;
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::{
        element::RawSink,
        elements::CudaDownload,
        test_support::{capture, try_cuda_device},
    };

    /// Imports `pixels` — `width` x `height` packed BGRA — and brings the
    /// imported frame back to the CPU as rows of BGRA. `None` means skip.
    fn round_trip(
        pixels: &[u8],
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
        modifier: u64,
    ) -> Option<Vec<u8>> {
        let (device, _cuda_lock) = try_cuda_device()?;
        let (buffer, stride) =
            super::super::dmabuf::tests::gbm_buffer_with(pixels, width, height, modifier)?;
        let (_source, handle) =
            match CudaDmaBufSource::new("dmabuf", &device, width, height, alpha, 1) {
                Ok(built) => built,
                Err(CudaDmaBufSourceError::Import(DmaBufCudaError::Library(library))) => {
                    eprintln!("skipping: {library} is missing");
                    return None;
                }
                Err(error) => panic!("the source did not build on a CUDA machine: {error}"),
            };
        let dmabuf = DmaBuf {
            fd: buffer.as_raw_fd(),
            offset: 0,
            stride,
            modifier,
            width,
            height,
        };
        let frame = match handle.import.import(&dmabuf, Some(5)) {
            Ok(frame) => frame,
            Err(error) => panic!("import failed: {error}"),
        };

        let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Bgra);
        let received = capture(&mut download);
        download.consume(frame).expect("download");
        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.pts(), Some(5), "the import dropped the pts");
        let stride = frame.stride(0);
        Some(
            (0..height as usize)
                .flat_map(|row| {
                    frame.data(0)[row * stride..row * stride + width as usize * 4].to_vec()
                })
                .collect(),
        )
    }

    /// Straight alpha comes through byte for byte — which also pins the row
    /// order and the channel order, since every pixel differs from its
    /// neighbours in both directions.
    #[test]
    fn a_straight_buffer_arrives_unchanged() {
        let (width, height) = (64u32, 32u32);
        let pixels: Vec<u8> = (0..height)
            .flat_map(|row| {
                (0..width).flat_map(move |column| {
                    [
                        column as u8 * 3,
                        row as u8 * 7,
                        200 - column as u8,
                        255 - row as u8,
                    ]
                })
            })
            .collect();
        let Some(out) = round_trip(&pixels, width, height, DmaBufAlpha::Straight, 0) else {
            return;
        };
        assert_eq!(out, pixels);
    }

    /// Premultiplied colour is divided back out, rounding to nearest, and a
    /// fully transparent pixel stays black rather than dividing by zero.
    #[test]
    fn a_premultiplied_buffer_arrives_straight() {
        let (width, height) = (4u32, 1u32);
        #[rustfmt::skip]
        let pixels = [
            100, 50, 25, 128,
            255, 128, 0, 255,
            0, 0, 0, 0,
            64, 64, 64, 64,
        ];
        let Some(out) = round_trip(&pixels, width, height, DmaBufAlpha::Premultiplied, 0) else {
            return;
        };
        #[rustfmt::skip]
        let expected = [
            199, 100, 50, 128,
            255, 128, 0, 255,
            0, 0, 0, 0,
            255, 255, 255, 64,
        ];
        assert_eq!(out, expected);
    }

    /// A buffer laid out in NVIDIA's tiles — what Chromium 156 hands over
    /// on its Wayland platform, `0x300000000606014` — comes through as the
    /// linear one does.
    #[test]
    fn a_block_linear_buffer_arrives_unchanged() {
        const NVIDIA_BLOCK_LINEAR: u64 = 0x0300_0000_0060_6014;
        let (width, height) = (64u32, 32u32);
        let pixels: Vec<u8> = (0..height)
            .flat_map(|row| {
                (0..width).flat_map(move |column| {
                    [
                        column as u8 * 3,
                        row as u8 * 7,
                        200 - column as u8,
                        255 - row as u8,
                    ]
                })
            })
            .collect();
        let Some(out) = round_trip(
            &pixels,
            width,
            height,
            DmaBufAlpha::Straight,
            NVIDIA_BLOCK_LINEAR,
        ) else {
            return;
        };
        assert_eq!(out, pixels);
    }
}
