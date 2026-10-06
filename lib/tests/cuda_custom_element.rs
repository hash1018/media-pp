//! An element of an application's own, working on CUDA pictures with CUDA
//! calls of its own between this crate's CUDA elements, the pictures never
//! leaving the GPU — built from the public API alone, as a crate outside
//! this one would build it.
//!
//! `Pictures -> CudaUpload -> Mark -> CudaDetectionOverlay -> CudaDownload
//! -> Collect`: `Mark` reads each uploaded picture through a
//! `CudaSurfaceView`, copies it into a picture of its own `CudaFramePool`,
//! paints a black square there with the driver, and says where a box is
//! with `Detections`; the overlay after it takes that picture as one of
//! its own device's and fills the box white. The test then reads both
//! back.

#![cfg(feature = "cuda")]

use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

use media_pp::{
    Error, Result,
    buffer::{MediaBuffer, set_time_base},
    bus::BusEvent,
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, Filter, Output, Produced, Sink, Source, Wait, element_pp_log},
    elements::{
        CudaDetectionOverlay, CudaDevice, CudaDownload, CudaFrameFormat, CudaFramePool,
        CudaSurfaceView, CudaUpload, Detection, DetectionOverlayOptions, Detections, Hiding,
        RedactStyle, Treatment,
    },
    ffmpeg::{self, ffi, format::Pixel, frame::Video},
    pipeline::Pipeline,
    pp_log::PpLog,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const PICTURES: i64 = 5;
/// The luma every picture is made with.
const GREY: u8 = 100;
/// Where `Mark` paints black, in pixels: x and y from, and the side.
const SQUARE: (usize, usize, usize) = (8, 8, 16);

/// The CUDA driver, as an application's element opens it: by name, for
/// the entry points it calls.
struct Driver {
    _library: libloading::Library,
    context: *mut c_void,
    device: i32,
    push: unsafe extern "C" fn(*mut c_void) -> i32,
    pop: unsafe extern "C" fn(*mut *mut c_void) -> i32,
    copy: unsafe extern "C" fn(*const Copy2d) -> i32,
    set: unsafe extern "C" fn(u64, usize, u8, usize, usize) -> i32,
    synchronize: unsafe extern "C" fn() -> i32,
    release: unsafe extern "C" fn(i32) -> i32,
}

// SAFETY: the context is the device's primary one, which the driver lets
// any thread make current; every call here pushes it first.
unsafe impl Send for Driver {}

/// `CUDA_MEMCPY2D`.
#[repr(C)]
struct Copy2d {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

/// `CU_MEMORYTYPE_DEVICE`.
const DEVICE_MEMORY: u32 = 2;

fn check(call: &str, result: i32) -> Result<()> {
    if result == 0 {
        Ok(())
    } else {
        Err(Error::Other(format!("{call} failed: CUresult {result}")))
    }
}

impl Driver {
    /// The primary context of `device`'s GPU, retained.
    fn open(device: &CudaDevice) -> Result<Self> {
        #[cfg(windows)]
        const LIBRARY: &str = "nvcuda.dll";
        #[cfg(not(windows))]
        const LIBRARY: &str = "libcuda.so.1";
        /// Entry point `name`, as the function type `T` it is declared as.
        ///
        /// # Safety
        ///
        /// `T` is that entry point's type, and is not called once
        /// `library` is gone.
        unsafe fn entry<T: Copy>(library: &libloading::Library, name: &[u8]) -> Result<T> {
            // SAFETY: the caller's promise.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| Error::Other(format!("{error}")))
        }
        // SAFETY: the NVIDIA driver's initialisers are its own; each entry
        // point is read as the type the driver API declares for it, and the
        // library is kept beside them.
        unsafe {
            let library = libloading::Library::new(LIBRARY)
                .map_err(|error| Error::Other(format!("{LIBRARY}: {error}")))?;
            let init = entry::<unsafe extern "C" fn(u32) -> i32>(&library, b"cuInit")?;
            let get =
                entry::<unsafe extern "C" fn(*mut i32, i32) -> i32>(&library, b"cuDeviceGet")?;
            let retain = entry::<unsafe extern "C" fn(*mut *mut c_void, i32) -> i32>(
                &library,
                b"cuDevicePrimaryCtxRetain",
            )?;
            check("cuInit", init(0))?;
            let mut handle = 0;
            check("cuDeviceGet", get(&mut handle, device.ordinal() as i32))?;
            let mut context = std::ptr::null_mut();
            check("cuDevicePrimaryCtxRetain", retain(&mut context, handle))?;
            Ok(Self {
                context,
                device: handle,
                push: entry(&library, b"cuCtxPushCurrent_v2")?,
                pop: entry(&library, b"cuCtxPopCurrent_v2")?,
                copy: entry(&library, b"cuMemcpy2D_v2")?,
                set: entry(&library, b"cuMemsetD2D8_v2")?,
                synchronize: entry(&library, b"cuCtxSynchronize")?,
                release: entry(&library, b"cuDevicePrimaryCtxRelease_v2")?,
                _library: library,
            })
        }
    }

    /// `work`, with the context current on this thread.
    fn current<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        // SAFETY: the context is retained for as long as `self` is.
        check("cuCtxPushCurrent", unsafe { (self.push)(self.context) })?;
        let done = work();
        let mut popped = std::ptr::null_mut();
        // SAFETY: pops what was pushed above.
        check("cuCtxPopCurrent", unsafe { (self.pop)(&mut popped) })?;
        done
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: releases the one reference `open` retained.
        unsafe { (self.release)(self.device) };
    }
}

fn identity(name: &str) -> (Arc<str>, PpLog) {
    (name.into(), element_pp_log(ElementType::Other, name, None))
}

macro_rules! element {
    ($type:ty) => {
        impl Element for $type {
            fn name(&self) -> Arc<str> {
                self.name.clone()
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
    };
}

/// `PICTURES` grey NV12 pictures in system memory.
struct Pictures {
    name: Arc<str>,
    pp_log: PpLog,
    next: i64,
}
element!(Pictures);

impl Source for Pictures {
    /// Made as fast as they are asked for, as a file is read.
    fn is_live(&self) -> bool {
        false
    }

    fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
        if self.next == PICTURES {
            return Ok(Produced::End);
        }
        let mut picture = Video::new(Pixel::NV12, WIDTH, HEIGHT);
        picture.data_mut(0).fill(GREY);
        picture.data_mut(1).fill(128);
        picture.set_pts(Some(self.next));
        set_time_base(&mut picture, ffmpeg::Rational(1, 30));
        self.next += 1;
        Ok(Produced::Buffer(MediaBuffer::video(picture)))
    }
}

/// The application's own element: a copy of each picture with a black
/// square on it, carrying a box for the overlay.
struct Mark {
    name: Arc<str>,
    pp_log: PpLog,
    device: CudaDevice,
    driver: Driver,
    pool: Option<CudaFramePool>,
}
element!(Mark);

impl Mark {
    fn mark(&mut self, picture: &Video) -> Result<MediaBuffer> {
        let from = CudaSurfaceView::new(&self.device, picture)?;
        if from.layout() != Pixel::NV12 {
            return Err(Error::Other(format!(
                "Mark takes NV12, got {:?}",
                from.layout()
            )));
        }
        let size = (from.width(), from.height());
        if self.pool.as_ref().map(|pool| (pool.width(), pool.height())) != Some(size) {
            self.pool = Some(CudaFramePool::new(
                &self.device,
                CudaFrameFormat::Nv12,
                size.0,
                size.1,
            )?);
        }
        let mut marked = self.pool.as_ref().expect("made above").get()?;
        let to = CudaSurfaceView::new(&self.device, &marked)?;
        let driver = &self.driver;
        driver.current(|| {
            for (from, to) in from.planes().iter().zip(to.planes()) {
                let copy = Copy2d {
                    src_x_in_bytes: 0,
                    src_y: 0,
                    src_memory_type: DEVICE_MEMORY,
                    src_host: std::ptr::null(),
                    src_device: from.pointer,
                    src_array: std::ptr::null_mut(),
                    src_pitch: from.pitch,
                    dst_x_in_bytes: 0,
                    dst_y: 0,
                    dst_memory_type: DEVICE_MEMORY,
                    dst_host: std::ptr::null_mut(),
                    dst_device: to.pointer,
                    dst_array: std::ptr::null_mut(),
                    dst_pitch: to.pitch,
                    width_in_bytes: from.row_bytes,
                    height: from.rows as usize,
                };
                // SAFETY: both planes are live device memory of at least
                // these rows, each `row_bytes` wide.
                check("cuMemcpy2D", unsafe { (driver.copy)(&copy) })?;
            }
            let luma = to.planes()[0];
            let (x, y, side) = SQUARE;
            // SAFETY: the square lies inside the picture's luma plane.
            check("cuMemsetD2D8", unsafe {
                (driver.set)(
                    luma.pointer + (y * luma.pitch + x) as u64,
                    luma.pitch,
                    16,
                    side,
                    side,
                )
            })?;
            // What the next element reads, finished first.
            // SAFETY: plain wait on the current context.
            check("cuCtxSynchronize", unsafe { (driver.synchronize)() })
        })?;
        // SAFETY: two live, distinct frames.
        unsafe { ffi::av_frame_copy_props(marked.as_mut_ptr(), picture.as_ptr()) };
        let found = Detections::new(
            self.name.clone(),
            Arc::from([Arc::from("thing")]),
            vec![Detection::new(0, 0.9, 0.5, 0.5, 0.25, 0.25)],
        );
        Ok(found.attach_to(MediaBuffer::Video(Arc::new(marked).into())))
    }
}

impl Filter for Mark {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(picture) = &buf else {
            return Err(Error::Other(format!(
                "Mark takes pictures, got {}",
                buf.kind()
            )));
        };
        out.push(self.mark(picture)?);
        Ok(())
    }

    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }
}

/// Each picture's PTS and luma plane, as it came back.
type Seen = Arc<Mutex<Vec<(Option<i64>, Vec<u8>, usize)>>>;

struct Collect {
    name: Arc<str>,
    pp_log: PpLog,
    seen: Seen,
}
element!(Collect);

impl Sink for Collect {
    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(picture) = &buf else {
            return Ok(());
        };
        self.seen.lock().unwrap().push((
            picture.pts(),
            picture.data(0).to_vec(),
            picture.stride(0),
        ));
        Ok(())
    }
}

#[test]
fn an_application_element_works_on_cuda_pictures_between_this_crates_own() {
    let device = match CudaDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no CUDA device ({error})");
            return;
        }
    };
    let driver = Driver::open(&device).expect("the CUDA driver beside the device");
    let overlay = CudaDetectionOverlay::new(
        "overlay",
        &device,
        DetectionOverlayOptions {
            others: Treatment {
                draw: None,
                hide: Some(Hiding {
                    style: RedactStyle::Fill(Color::WHITE),
                    margin: 0.0,
                }),
                min_score: 0.0,
            },
            ..DetectionOverlayOptions::default()
        },
    )
    .expect("overlay");
    let (name, pp_log) = identity("mark");
    let mark = Mark {
        name,
        pp_log,
        device: device.clone(),
        driver,
        pool: None,
    };
    let (name, pp_log) = identity("pictures");
    let pictures = Pictures {
        name,
        pp_log,
        next: 0,
    };
    let seen = Seen::default();
    let (name, pp_log) = identity("collect");
    let collect = Collect {
        name,
        pp_log,
        seen: seen.clone(),
    };

    let (pipeline, ()) = Pipeline::new("custom-cuda", pictures, |pictures, ctx| {
        let branch = ctx
            .branch()
            .pipe(CudaUpload::new("upload", &device, CudaFrameFormat::Nv12))
            .pipe(mark)
            .pipe(overlay)
            .pipe(CudaDownload::new(
                "download",
                &device,
                CudaFrameFormat::Nv12,
            ))
            .to(collect)?;
        ctx.attach(pictures, 0, branch)?;
        Ok(())
    })
    .expect("a pipeline whose links agree");
    pipeline.run().expect("run");
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Finished => break,
            BusEvent::Error { .. } => panic!("{event}"),
            _ => {}
        }
    }
    pipeline.stop();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), PICTURES as usize, "every picture came back");
    let (x, y, side) = SQUARE;
    // The overlay's box: half-way across and down, a quarter of each side.
    let boxed = (WIDTH as usize / 2, HEIGHT as usize / 2);
    for (index, (pts, luma, stride)) in seen.iter().enumerate() {
        assert_eq!(*pts, Some(index as i64), "picture {index}'s time");
        let at = |x: usize, y: usize| luma[y * stride + x];
        assert_eq!(at(x, y), 16, "picture {index}: Mark's square");
        assert_eq!(
            at(x + side - 1, y + side - 1),
            16,
            "picture {index}: its far corner"
        );
        assert_eq!(
            at(x + side, y + side),
            GREY,
            "picture {index}: past the square"
        );
        assert_eq!(
            at(boxed.0, boxed.1),
            235,
            "picture {index}: the overlay's fill"
        );
        assert_eq!(
            at(WIDTH as usize - 1, 0),
            GREY,
            "picture {index}: untouched"
        );
    }
}
