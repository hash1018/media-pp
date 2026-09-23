use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
    time::Duration,
};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D11::{
                D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
                ID3D11Device, ID3D11Device1, ID3D11DeviceContext, ID3D11Texture2D,
            },
            Dxgi::{
                Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
                IDXGIKeyedMutex,
            },
        },
    },
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlReceiver,
    element::{Element, ElementType, Source, SourceElement},
    elements::{AppSource, AppSourceHandle},
    error::{D3d11FrameWrapError, D3d11SharedDeviceError, Result},
    pad::SrcPad,
    platform::windows::{d3d11::protect_shared_device, d3d11va::wrap_d3d11_texture},
    pool::UnboundObjectPool,
    pp_log::{PpLog, pp_info},
};

/// The one texture format taken here: what a compositing producer — a
/// browser engine, another process's renderer — hands over. NV12 and the
/// rest would each need a matching copy and their own validation, and
/// nothing asks for them yet.
const SHARED_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

/// How long to wait for a producer's keyed mutex before giving up on one
/// frame. Only a producer that guards its texture with one has it at all,
/// and a well-behaved one holds it for its own draw — long enough to
/// outlast that, short enough that a stuck producer costs a frame rather
/// than the calling thread.
const KEYED_MUTEX_WAIT: u32 = 100;

/// Errors specific to `D3d11SharedTextureSource`. Converts into the
/// crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11SharedTextureSourceError {
    /// The handle could not be opened on this pipeline's device: it does
    /// not name a shared texture, it was already closed, or it came from a
    /// device this one cannot share with.
    #[error("could not open shared texture handle {handle:#x}: {source}")]
    Open {
        /// The handle as it was passed in.
        handle: isize,
        /// What Direct3D said about it.
        #[source]
        source: windows::core::Error,
    },

    /// The texture behind the handle is not the size this source was
    /// opened for. Nothing here resizes.
    #[error(
        "the shared texture is {actual_width}x{actual_height}, but \
         D3d11SharedTextureSource was opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Width of the texture behind the handle.
        actual_width: u32,
        /// Height of the texture behind the handle.
        actual_height: u32,
        /// Width this source copies into.
        expected_width: u32,
        /// Height this source copies into.
        expected_height: u32,
    },

    /// The texture behind the handle is in a format this source does not
    /// copy — see [`D3d11SharedTextureSource`]'s own docs.
    #[error("D3d11SharedTextureSource only takes DXGI_FORMAT_B8G8R8A8_UNORM textures, got {0:?}")]
    UnsupportedFormat(DXGI_FORMAT),

    /// The producer guards its texture with a keyed mutex that could not be
    /// taken — most often because the producer is holding it, and this
    /// waited a bounded moment for it rather than for ever.
    #[error("could not take the shared texture's keyed mutex: {0}")]
    KeyedMutex(#[source] windows::core::Error),

    /// Direct3D refused this pipeline's own copy of the texture.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// The device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),

    /// FFmpeg could not wrap the copy as a frame.
    #[error(transparent)]
    FrameWrap(#[from] D3d11FrameWrapError),

    /// The source has ended — its `Pipeline` finished, or end-of-stream was
    /// already submitted — so nothing more can be pushed into it.
    #[error("D3d11SharedTextureSource has already ended")]
    Closed,
}

/// Brings another device's shared textures into this pipeline, as
/// `Pixel::D3D11` video frames on this pipeline's own device.
///
/// A pushed source rather than a reading one: whoever produces the pictures
/// calls [`D3d11SharedTextureHandle::push`] with a shared-texture handle per
/// picture, and this emits one frame downstream for each. That is
/// [`crate::elements::AppSource`]'s shape, which this is built on — the
/// difference is the copy, which is the whole point of the element.
///
/// # Why it copies
///
/// The texture behind the handle belongs to the producer, and a producer
/// reuses its textures: a browser engine draws its next picture into one it
/// handed over two pictures ago. What a pipeline holds has to outlive the
/// call — a compositor keeps the last frame of every input for as long as
/// that input is quiet — so each push copies into a texture of this
/// pipeline's own. The copy is GPU to GPU; no pixel crosses to the CPU.
///
/// The handle is opened afresh every push instead of being cached by its
/// value. A producer closes handles and the numbers come back, so a cache
/// keyed by one would eventually hand back a texture that is no longer the
/// one that number names.
///
/// # What it does not do
///
/// Resize, convert, pace, or interpret. A texture of another size or format
/// is refused rather than adapted — chain a
/// [`crate::elements::D3d11Scaler`] after this for either — and a producer
/// that paints only when something changed produces frames only then. That
/// is usually enough, since a compositor answers its own rate out of the
/// last frame each input gave it.
///
/// # Alpha
///
/// Whatever the producer put in the texture, byte for byte. A browser
/// engine composites its page with the alpha already multiplied into the
/// colour, and a layer drawn from such a frame has to say so — see
/// [`VideoLayer::premultiplied_alpha`](crate::elements::VideoLayer::premultiplied_alpha).
pub struct D3d11SharedTextureSource {
    inner: AppSource,
}

/// Pushes another device's textures into a [`D3d11SharedTextureSource`].
///
/// Cheap to clone — two refcount bumps and an `Arc` — and every clone feeds
/// the same source. Push [`Self::finish`] when done, or drop every clone,
/// which ends the source the same way [`AppSourceHandle`]'s own drop does.
///
/// # Push where the handle is still valid
///
/// [`Self::push`] opens the handle and copies before it returns, so it must
/// be called while the producer still holds that picture — inside its paint
/// callback, not queued for later. That is the entire reason this work
/// happens on the caller's thread instead of the source's own.
#[derive(Clone)]
pub struct D3d11SharedTextureHandle {
    pusher: AppSourceHandle,
    import: Arc<Import>,
}

/// What a push needs: this pipeline's device and shared context, the shape
/// it copies into, and the pool the outgoing frames come from.
struct Import {
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    width: u32,
    height: u32,
    /// Reused across pushes for the small `AVFrame` wrapper only — the GPU
    /// texture inside it is a fresh allocation per frame, since downstream
    /// may still hold the last one. Same arrangement, for the same reason,
    /// as [`crate::elements::D3d11Upload`]'s.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `device` is a `windows-rs` COM wrapper used only for device-level
// calls, which are free-threaded, and `context` is only ever touched under
// its own `Mutex` — the contract every other D3D11 element here holds to.
// Everything else is plain data.
unsafe impl Send for Import {}
// SAFETY: as above — and every method here takes `&self` and issues only
// those same calls, so two threads sharing one `Import` is the case that
// reasoning already covers. `D3d11SharedTextureHandle` is `Clone`, so this
// is what lets two producer threads push into the same source.
unsafe impl Sync for Import {}

impl D3d11SharedTextureSource {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses — the frames this produces are that device's, and the copy is a
    /// context-level call.
    ///
    /// `width`/`height` are what every pushed texture must be and what the
    /// frames coming out are. `capacity` bounds how many frames may sit
    /// unconsumed before [`D3d11SharedTextureHandle::push`] blocks, exactly
    /// as [`AppSource::new`]'s own does.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        width: u32,
        height: u32,
        capacity: usize,
    ) -> std::result::Result<(Self, D3d11SharedTextureHandle), D3d11SharedTextureSourceError> {
        // Pushed from the producer's thread and drained on the source's own,
        // so the device has to be usable from more than the thread that made
        // it before any command is issued.
        protect_shared_device(device)?;
        let (inner, pusher) = AppSource::typed(
            name,
            capacity,
            ElementType::D3d11SharedTextureSource,
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
        );
        pp_info!(pp_log: inner.pp_log(), "opened: {width}x{height}");
        Ok((
            Self { inner },
            D3d11SharedTextureHandle {
                pusher,
                import: Arc::new(Import {
                    device: device.clone(),
                    context,
                    width,
                    height,
                    pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
                }),
            },
        ))
    }
}

impl D3d11SharedTextureHandle {
    /// Copies the texture behind `handle` into this pipeline and pushes it
    /// downstream as one frame, stamped `pts` (in the caller's own time
    /// base — `None` leaves it unstamped, which is what a
    /// [`crate::elements::VideoCompositorOptions`]-driven graph wants,
    /// since the compositor sets its own).
    ///
    /// `handle` is a shared-texture handle belonging to another device —
    /// what `IDXGIResource1::CreateSharedHandle` produces, and what a
    /// producer holding one hands over. It is opened, copied from, and done
    /// with before this returns; nothing keeps it afterwards.
    ///
    /// What the producer owes in return is a flush: drawing it has queued
    /// but not submitted is not guaranteed to be in the texture when
    /// another device opens it. A producer that hands pictures over through
    /// a callback — a browser engine's accelerated paint — has already
    /// flushed by the time it calls; one that draws and pushes itself has
    /// to. Nothing here can check it, so a missing flush shows up as a
    /// stale or empty picture rather than an error.
    ///
    /// Blocks while the source's queue is full, as
    /// [`AppSourceHandle::push`] does. See this type's own docs on where
    /// this must be called from.
    pub fn push(
        &self,
        handle: isize,
        pts: Option<i64>,
    ) -> std::result::Result<(), D3d11SharedTextureSourceError> {
        let frame = self.import.copy(handle, pts)?;
        self.pusher
            .push(frame)
            .map_err(|_| D3d11SharedTextureSourceError::Closed)
    }

    /// [`Self::push`] for a producer that cannot wait — which a browser
    /// engine's paint callback cannot: it is that engine's own thread, and
    /// everything else it does happens there too.
    ///
    /// `Ok(false)` (not an error) means the source's queue was full and this
    /// picture was dropped; `Err` means the source itself has ended. The
    /// same trade-off [`AppSourceHandle::try_push`] offers, and the reason
    /// it matters more here: a pipeline this source feeds can be *paused*,
    /// and a paused pipeline consumes nothing at all. A blocking push then
    /// stops the producer for as long as the pause lasts, which for a
    /// browser engine means every one of its pages, not only this one.
    ///
    /// The copy still happens — whether there is room is only known once
    /// there is something to put there — so a source nobody is draining
    /// costs a copy per picture until the producer is told to stop drawing.
    pub fn try_push(
        &self,
        handle: isize,
        pts: Option<i64>,
    ) -> std::result::Result<bool, D3d11SharedTextureSourceError> {
        let frame = self.import.copy(handle, pts)?;
        self.pusher
            .try_push(frame)
            .map_err(|_| D3d11SharedTextureSourceError::Closed)
    }

    /// Ends the stream, the same as pushing `Eos` into an [`AppSource`] —
    /// or drop every clone of this, which does the same thing.
    pub fn finish(&self) -> std::result::Result<(), D3d11SharedTextureSourceError> {
        self.pusher
            .push(MediaBuffer::Eos)
            .map_err(|_| D3d11SharedTextureSourceError::Closed)
    }
}

impl Import {
    /// Opens `handle`, copies what is behind it into a texture of this
    /// pipeline's own, and wraps that as a frame.
    fn copy(
        &self,
        handle: isize,
        pts: Option<i64>,
    ) -> std::result::Result<MediaBuffer, D3d11SharedTextureSourceError> {
        let shared = self.open(handle)?;
        let mut description = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `shared` is the texture just opened and `description` a
        // live out-parameter.
        unsafe { shared.GetDesc(&mut description) };
        if description.Width != self.width || description.Height != self.height {
            return Err(D3d11SharedTextureSourceError::DimensionMismatch {
                actual_width: description.Width,
                actual_height: description.Height,
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        if description.Format != SHARED_FORMAT {
            return Err(D3d11SharedTextureSourceError::UnsupportedFormat(
                description.Format,
            ));
        }

        let copy = self.allocate()?;
        // A producer that guards its texture takes this around its own
        // drawing; one that does not has no such interface to cast to. Held
        // for exactly the copy, and released on the error path too — a
        // mutex left acquired would stall the producer for good.
        let keyed: Option<IDXGIKeyedMutex> = shared.cast().ok();
        {
            let context = self
                .context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(keyed) = &keyed {
                // SAFETY: `keyed` is the opened texture's own interface.
                unsafe { keyed.AcquireSync(0, KEYED_MUTEX_WAIT) }
                    .map_err(D3d11SharedTextureSourceError::KeyedMutex)?;
            }
            // SAFETY: both textures are this device's, validated above to be
            // the same size and format, and neither is mapped.
            unsafe { context.CopyResource(&copy, &shared) };
            if let Some(keyed) = &keyed {
                // SAFETY: acquired immediately above.
                unsafe { keyed.ReleaseSync(0) }
                    .map_err(D3d11SharedTextureSourceError::KeyedMutex)?;
            }
        }

        let mut pooled = self.pool.get();
        *pooled = wrap_d3d11_texture(copy, self.width, self.height)?;
        pooled.set_pts(pts);
        Ok(MediaBuffer::Video(Arc::new(pooled)))
    }

    /// Opens the producer's handle on this pipeline's device. See the
    /// element's own docs on why nothing caches the result.
    fn open(
        &self,
        handle: isize,
    ) -> std::result::Result<ID3D11Texture2D, D3d11SharedTextureSourceError> {
        let device: ID3D11Device1 = self.device.cast()?;
        // SAFETY: the handle is the caller's, valid for the duration of this
        // call by `D3d11SharedTextureHandle::push`'s contract, and
        // `OpenSharedResource1` only reads it.
        unsafe { device.OpenSharedResource1(HANDLE(handle as *mut c_void)) }
            .map_err(|source| D3d11SharedTextureSourceError::Open { handle, source })
    }

    /// This pipeline's own texture for one frame.
    fn allocate(&self) -> std::result::Result<ID3D11Texture2D, D3d11SharedTextureSourceError> {
        let description = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: SHARED_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            // Enough for whatever draws it next — a compositor layer, a
            // renderer, a scaler — and nothing decode-specific.
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        // SAFETY: `description` is fully initialized, no initial data is
        // supplied, and `texture` is a live out-parameter.
        unsafe {
            self.device
                .CreateTexture2D(&description, None, Some(&mut texture))?
        };
        Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
    }
}

impl Element for D3d11SharedTextureSource {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11SharedTextureSource
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }
}

impl Source for D3d11SharedTextureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        self.inner.src_pads()
    }
}

impl SourceElement for D3d11SharedTextureSource {
    /// Unlike the [`AppSource`] underneath, this one is live: what it emits
    /// is whatever another device drew just now, and a paused pipeline
    /// cannot ask it for a first picture.
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        self.inner.run(control, bus)
    }

    /// No-op, as [`AppSource::seek`] is: there is no timeline here to
    /// reposition — what comes next is whatever the producer paints next.
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.inner.seek(target)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::capture;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use windows::Win32::Graphics::{
        Direct3D11::{
            D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SUBRESOURCE_DATA,
        },
        Dxgi::{DXGI_SHARED_RESOURCE_READ, IDXGIResource1},
    };

    use super::*;
    use crate::{
        element::Sink, elements::D3d11Download, pipeline::Pipeline, test_support::try_d3d11_device,
    };

    /// A texture as another device would hand it over: shared through an NT
    /// handle, every texel `color`. Both it and its handle stay open for
    /// the rest of the test, the way a producer's do while it is painting.
    fn shared_texture(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        width: u32,
        height: u32,
        color: [u8; 4],
    ) -> (ID3D11Texture2D, isize) {
        let pixels = color.repeat((width * height) as usize);
        let description = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: SHARED_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0)
                as u32,
        };
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast::<c_void>(),
            SysMemPitch: width * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        // SAFETY: `initial` addresses a live, correctly pitched plane of
        // exactly the declared dimensions through the call.
        unsafe { device.CreateTexture2D(&description, Some(&initial), Some(&mut texture)) }
            .expect("CreateTexture2D(shared BGRA) failed");
        let texture = texture.expect("CreateTexture2D succeeded without producing a texture");
        // What a producer owes the other device: nothing it has not flushed is
        // guaranteed to be there when the other one opens the handle.
        // SAFETY: the context belongs to `device` and is held exclusively here.
        unsafe { context.lock().unwrap().Flush() };
        let resource: IDXGIResource1 = texture.cast().expect("a D3D11 texture is a DXGI resource");
        // SAFETY: the texture was created with the NT-handle share flags the
        // call requires; the returned handle is owned by this test.
        let handle =
            unsafe { resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ.0, None) }
                .expect("CreateSharedHandle failed");
        (texture, handle.0 as isize)
    }

    /// The point of the element: what one device wrote is what a frame on
    /// the other device holds.
    #[test]
    fn a_texture_from_another_device_arrives_as_this_pipelines_own_frame() {
        let Some((producer, producer_context)) = try_d3d11_device() else {
            return;
        };
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (_source, handle) =
            D3d11SharedTextureSource::new("shared", &device, context.clone(), 8, 8, 4)
                .expect("D3d11SharedTextureSource::new should succeed");

        // BGRA byte order, so this is a strong red at full alpha.
        let (_producer_texture, shared) =
            shared_texture(&producer, &producer_context, 8, 8, [20, 30, 230, 255]);
        let frame = handle
            .import
            .copy(shared, Some(7))
            .expect("importing the producer's texture must succeed");

        let MediaBuffer::Video(video) = &frame else {
            panic!("expected a Video buffer, got {}", frame.kind());
        };
        assert_eq!(video.format(), ffmpeg::format::Pixel::D3D11);
        assert_eq!((video.width(), video.height()), (8, 8));
        assert_eq!(video.pts(), Some(7), "the pushed timestamp is kept");

        let mut download = D3d11Download::new("download", &device, context)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        download
            .consume(frame)
            .expect("downloading the imported frame must succeed");
        let received = received.lock().unwrap();
        let MediaBuffer::Video(downloaded) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(
            &downloaded.data(0)[0..4],
            [20, 30, 230, 255],
            "the imported frame must hold what the other device wrote"
        );
    }

    /// A producer that cannot wait is told there was no room rather than
    /// held until there is. Nothing drains this source — its pipeline is
    /// never started — which is the state a paused one puts a live producer
    /// in.
    #[test]
    fn a_full_queue_costs_a_picture_rather_than_the_producers_thread() {
        let Some((producer, producer_context)) = try_d3d11_device() else {
            return;
        };
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (_source, handle) = D3d11SharedTextureSource::new("shared", &device, context, 8, 8, 1)
            .expect("D3d11SharedTextureSource::new should succeed");

        let (_producer_texture, shared) =
            shared_texture(&producer, &producer_context, 8, 8, [0, 0, 0, 255]);
        assert!(
            handle
                .try_push(shared, None)
                .expect("the source is running"),
            "the first picture takes the one place there is"
        );
        assert!(
            !handle
                .try_push(shared, None)
                .expect("the source is running"),
            "the second finds it full and is dropped"
        );
    }

    /// Nothing here resizes, so a texture of another shape is a caller
    /// mistake reported as one rather than something to adapt to.
    #[test]
    fn a_texture_of_another_size_is_refused() {
        let Some((producer, producer_context)) = try_d3d11_device() else {
            return;
        };
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (_source, handle) = D3d11SharedTextureSource::new("shared", &device, context, 8, 8, 4)
            .expect("D3d11SharedTextureSource::new should succeed");

        let (_producer_texture, shared) =
            shared_texture(&producer, &producer_context, 16, 16, [0, 0, 0, 255]);
        let error = handle
            .push(shared, None)
            .expect_err("a texture of another size must be refused");
        assert!(
            matches!(
                error,
                D3d11SharedTextureSourceError::DimensionMismatch {
                    actual_width: 16,
                    actual_height: 16,
                    expected_width: 8,
                    expected_height: 8,
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    /// A producer can close a handle at any moment, so one that names
    /// nothing must come back as an error rather than a panic.
    #[test]
    fn a_handle_that_names_no_shared_texture_is_refused() {
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (_source, handle) = D3d11SharedTextureSource::new("shared", &device, context, 8, 8, 4)
            .expect("D3d11SharedTextureSource::new should succeed");

        let error = handle
            .push(0x4, None)
            .expect_err("a handle that names nothing must be refused");
        assert!(
            matches!(
                error,
                D3d11SharedTextureSourceError::Open { handle: 0x4, .. }
            ),
            "expected Open, got {error:?}"
        );
    }

    /// The element as a pipeline sees it: pushes reach downstream, and
    /// `finish` ends the source.
    #[test]
    fn pushed_textures_reach_downstream_and_finish_ends_the_source() {
        let Some((producer, producer_context)) = try_d3d11_device() else {
            return;
        };
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (source, handle) = D3d11SharedTextureSource::new("shared", &device, context, 8, 8, 4)
            .expect("D3d11SharedTextureSource::new should succeed");

        let frames = Arc::new(AtomicUsize::new(0));
        let counted = frames.clone();
        let pipeline = Pipeline::new("shared-texture", source, move |source, ctx| {
            let branch = ctx.branch().to(crate::elements::AppSink::new(
                "count",
                move |buf: MediaBuffer| {
                    if !buf.is_eos() {
                        counted.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(())
                },
            ))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("the pipeline must wire up");
        pipeline.run().expect("the pipeline must start");

        let (_producer_texture, shared) =
            shared_texture(&producer, &producer_context, 8, 8, [0, 0, 0, 255]);
        for pts in 0..3 {
            handle
                .push(shared, Some(pts))
                .expect("pushing must succeed while the source runs");
        }
        handle.finish().expect("finishing must succeed");

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, crate::bus::BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert_eq!(frames.load(Ordering::SeqCst), 3);
    }
}
