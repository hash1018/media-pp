//! [`MfVirtualCamera`]: a pipeline's pictures as a camera other
//! applications can open.

use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::{
    Foundation::{CloseHandle, HANDLE},
    Media::MediaFoundation::{
        IMFVirtualCamera, MF_VERSION, MFSTARTUP_NOSOCKET, MFShutdown, MFStartup,
        MFVirtualCameraAccess, MFVirtualCameraAccess_CurrentUser, MFVirtualCameraLifetime,
        MFVirtualCameraLifetime_Session, MFVirtualCameraType,
        MFVirtualCameraType_SoftwareCameraSource,
    },
    System::{
        LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW},
        Memory::{
            FILE_MAP_READ, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            OpenFileMappingW, UnmapViewOfFile,
        },
        Registry::{HKEY, HKEY_LOCAL_MACHINE, KEY_READ, RegCloseKey, RegOpenKeyExW},
        SystemInformation::GetTickCount64,
    },
};
use windows::core::{GUID, HRESULT, HSTRING, Interface, PCWSTR};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Element, ElementType, Sink, element_pp_log},
    elements::filter::scaler::{is_rgb, matrix},
    error::Result,
    platform::windows::com::ComApartment,
    render::{SinkStage, sink_stage},
};

use super::protocol::{self, Header};

/// How often a camera nobody has opened yet is looked for again: opening
/// the section is a system call, and while no application reads the camera
/// there is no section to find.
const LOOK_AGAIN_AFTER: Duration = Duration::from_millis(500);

/// Shows what reaches it as a camera — "Windows Virtual Camera" in every
/// application that lists cameras: Teams, Zoom, a browser, Windows' own
/// Camera app. The camera exists from construction until the element is
/// dropped.
///
/// It takes decoded video in system memory, any pixel format and size, and
/// hands each picture on as NV12 at the size the application reading the
/// camera chose — one of 1920x1080, 1280x720 and 640x360 at 30 frames a
/// second — converted to BT.709. Pictures arriving faster than the camera's
/// rate are fine: the camera takes the newest. While no application reads
/// the camera, nothing is converted or written.
///
/// # Requirements
///
/// Windows 11 (build 22000) or newer, and the camera's DLL,
/// `media_pp_vcam.dll` from this repository's `vcam` crate, registered once
/// from an elevated prompt: Windows loads it into its Frame Server service,
/// and that service reads only what is registered for the whole machine.
/// [`MfVirtualCamera::new`] says which of the two is missing rather than
/// creating a camera that shows nothing.
///
/// # Threads
///
/// The camera is created and removed on a thread of the element's own, in
/// the multithreaded COM apartment Media Foundation wants, so the thread
/// constructing the element may be in any apartment. Dropping the element
/// removes the camera and joins that thread.
pub struct MfVirtualCamera(SinkStage<Feeding>);

sink_stage!(MfVirtualCamera);

/// Why an [`MfVirtualCamera`] could not be made or could not take a buffer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MfVirtualCameraError {
    /// This Windows has no virtual camera support: it is older than
    /// Windows 11 (build 22000).
    #[error("virtual cameras need Windows 11 (build 22000) or newer")]
    Unsupported,
    /// The camera's DLL is not registered for the machine.
    #[error(
        "the virtual camera is not installed: register media_pp_vcam.dll (CLSID {clsid}) \
         with regsvr32 from an elevated prompt"
    )]
    NotInstalled {
        /// The CLSID the DLL registers.
        clsid: &'static str,
    },
    /// Windows refused to create or start the camera — for one, camera
    /// access is denied in Privacy settings.
    #[error("the virtual camera could not be started: {0}")]
    Start(windows::core::Error),
    /// Not a video frame in system memory.
    #[error("MfVirtualCamera takes video frames in system memory, got {0}")]
    UnsupportedBuffer(&'static str),
    /// Converting a picture to the camera's size and format failed.
    #[error("could not convert a picture for the camera: {0}")]
    Convert(#[from] ffmpeg::Error),
}

/// What an [`MfVirtualCamera`] does with each picture.
struct Feeding {
    name: Arc<str>,
    pp_log: PpLog,
    /// The thread keeping the camera registered; `None` only in tests,
    /// which play the camera's part themselves.
    _camera: Option<Camera>,
    section_name: &'static str,
    section: Option<Mapping>,
    looked_at: Option<Instant>,
    conversion: Option<Conversion>,
}

/// One picture shape's conversion to the camera's, and the frame it writes.
struct Conversion {
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    to: (u32, u32),
    context: ffmpeg::software::scaling::Context,
    frame: ffmpeg::frame::Video,
}

// SAFETY: the scaling context and frame are heap allocations owned solely
// by this conversion, used only by the one thread rendering at a time,
// exactly as `SwScaler` holds its own.
unsafe impl Send for Conversion {}

impl MfVirtualCamera {
    /// Creates the camera, named `friendly_name` — Windows adds "Windows
    /// Virtual Camera" — for the current user, for as long as the element
    /// lives.
    pub fn new(name: impl Into<String>, friendly_name: &str) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MfVirtualCamera, &name, None);
        if !installed() {
            return Err(MfVirtualCameraError::NotInstalled {
                clsid: protocol::CLSID,
            }
            .into());
        }
        let camera = Camera::start(friendly_name)?;
        pp_info!(pp_log: &pp_log, "created: camera \"{friendly_name}\" registered");
        Ok(Self(SinkStage::new(Feeding {
            name,
            pp_log,
            _camera: Some(camera),
            section_name: protocol::SECTION_NAME,
            section: None,
            looked_at: None,
            conversion: None,
        })))
    }

    /// The element without a camera, writing to the section `section_name`
    /// that a test creates.
    #[cfg(test)]
    fn for_section(section_name: &'static str) -> Self {
        let name: Arc<str> = "camera".into();
        let pp_log = element_pp_log(ElementType::MfVirtualCamera, &name, None);
        Self(SinkStage::new(Feeding {
            name,
            pp_log,
            _camera: None,
            section_name,
            section: None,
            looked_at: None,
            conversion: None,
        }))
    }
}

/// Whether the camera's DLL is registered where Frame Server looks.
fn installed() -> bool {
    let key_name = format!(
        "Software\\Classes\\CLSID\\{}\\InprocServer32",
        protocol::CLSID
    );
    let mut key = HKEY::default();
    // SAFETY: opens one key for reading, closed straight away.
    unsafe {
        let opened = RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            &HSTRING::from(key_name),
            None,
            KEY_READ,
            &mut key,
        );
        if opened.is_ok() {
            let _ = RegCloseKey(key);
        }
        opened.is_ok()
    }
}

/// `MFCreateVirtualCamera`'s signature, looked up at run time: linking it
/// would keep a program that has an `MfVirtualCamera` from even starting on
/// Windows 10, whose Media Foundation has no such function.
type CreateVirtualCamera = unsafe extern "system" fn(
    MFVirtualCameraType,
    MFVirtualCameraLifetime,
    MFVirtualCameraAccess,
    PCWSTR,
    PCWSTR,
    *const GUID,
    u32,
    *mut *mut std::ffi::c_void,
) -> HRESULT;

/// The thread holding the registered camera.
struct Camera {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Camera {
    /// Registers and starts the camera on a thread of its own, and waits
    /// until it has or could not.
    fn start(friendly_name: &str) -> Result<Self> {
        let friendly_name = HSTRING::from(friendly_name);
        let (stop, stopped) = mpsc::channel::<()>();
        let (done, started) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("virtual-camera".into())
            .spawn(move || {
                let _apartment = match ComApartment::new() {
                    Ok(apartment) => apartment,
                    Err(error) => {
                        let _ = done.send(Err(MfVirtualCameraError::Start(error)));
                        return;
                    }
                };
                // SAFETY: balanced by `MFShutdown` below, on this thread.
                if let Err(error) = unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) } {
                    let _ = done.send(Err(MfVirtualCameraError::Start(error)));
                    return;
                }
                match create(&friendly_name) {
                    Ok(camera) => {
                        let _ = done.send(Ok(()));
                        // Held until the element is dropped, which drops the
                        // sender and ends this wait.
                        let _ = stopped.recv();
                        // SAFETY: removes the camera this thread made.
                        let _ = unsafe { camera.Remove() };
                    }
                    Err(error) => {
                        let _ = done.send(Err(error));
                    }
                }
                // SAFETY: balances the `MFStartup` above.
                let _ = unsafe { MFShutdown() };
            })
            .map_err(|error| {
                crate::error::Error::Other(format!("virtual camera thread: {error}"))
            })?;
        let camera = Self {
            stop: Some(stop),
            thread: Some(thread),
        };
        match started.recv() {
            Ok(Ok(())) => Ok(camera),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(MfVirtualCameraError::Start(windows::core::Error::empty()).into()),
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Creates the camera and starts it, on a thread in the multithreaded
/// apartment with Media Foundation started.
fn create(friendly_name: &HSTRING) -> std::result::Result<IMFVirtualCamera, MfVirtualCameraError> {
    // SAFETY: loads a system DLL by name from System32 only, and looks up
    // a function whose signature is `CreateVirtualCamera`'s.
    let create: CreateVirtualCamera = unsafe {
        let module = LoadLibraryExW(
            &HSTRING::from("mfsensorgroup.dll"),
            None,
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map_err(|_| MfVirtualCameraError::Unsupported)?;
        let function = GetProcAddress(module, windows::core::s!("MFCreateVirtualCamera"))
            .ok_or(MfVirtualCameraError::Unsupported)?;
        std::mem::transmute::<unsafe extern "system" fn() -> isize, CreateVirtualCamera>(function)
    };
    let source_id = HSTRING::from(protocol::CLSID);
    let mut camera = std::ptr::null_mut();
    // SAFETY: every argument is valid for the call; `camera` receives an
    // owned reference on success.
    let camera = unsafe {
        create(
            MFVirtualCameraType_SoftwareCameraSource,
            MFVirtualCameraLifetime_Session,
            MFVirtualCameraAccess_CurrentUser,
            PCWSTR(friendly_name.as_ptr()),
            PCWSTR(source_id.as_ptr()),
            std::ptr::null(),
            0,
            &mut camera,
        )
        .ok()
        .map_err(MfVirtualCameraError::Start)?;
        IMFVirtualCamera::from_raw(camera)
    };
    // SAFETY: starts the camera just created, with no callback.
    unsafe { camera.Start(None) }.map_err(MfVirtualCameraError::Start)?;
    Ok(camera)
}

/// The section, opened as the producing side: read and write only, which
/// is what the camera grants.
struct Mapping {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
}

// SAFETY: a section handle and view may be used from any thread; what is in
// the view is touched only through `protocol`'s atomics and guarded copy.
unsafe impl Send for Mapping {}

impl Mapping {
    fn open(name: &str) -> Option<Self> {
        let access = FILE_MAP_READ | FILE_MAP_WRITE;
        // SAFETY: opens a named mapping and maps all of it; both are let go
        // of in `Drop`, or here when the map fails.
        unsafe {
            let handle = OpenFileMappingW(access.0, false, &HSTRING::from(name)).ok()?;
            let view = MapViewOfFile(handle, access, 0, 0, protocol::SECTION_BYTES);
            if view.Value.is_null() {
                let _ = CloseHandle(handle);
                return None;
            }
            Some(Self { handle, view })
        }
    }

    fn view(&self) -> *mut u8 {
        self.view.Value.cast()
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this mapping's own view and handle.
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.handle);
        }
    }
}

impl Feeding {
    /// The section, once the camera has made one — looked for at most every
    /// [`LOOK_AGAIN_AFTER`] until then.
    fn section(&mut self) -> Option<&Mapping> {
        if self.section.is_none() {
            let now = Instant::now();
            if self
                .looked_at
                .is_some_and(|looked| now.duration_since(looked) < LOOK_AGAIN_AFTER)
            {
                return None;
            }
            self.looked_at = Some(now);
            self.section = Mapping::open(self.section_name);
            if self.section.is_some() {
                pp_info!(self, "an application opened the camera");
            }
        }
        self.section.as_ref()
    }

    /// Converts `frame` to NV12 at `to`, reusing the conversion while the
    /// pictures keep their shape.
    fn convert(
        &mut self,
        frame: &ffmpeg::frame::Video,
        to: (u32, u32),
    ) -> Result<&ffmpeg::frame::Video> {
        let from = (
            frame.format(),
            frame.width(),
            frame.height(),
            frame.color_space(),
            frame.color_range(),
        );
        if self
            .conversion
            .as_ref()
            .is_none_or(|conversion| conversion.from != from || conversion.to != to)
        {
            self.conversion = Some(conversion(from, to)?);
        }
        let conversion = self.conversion.as_mut().expect("made above");
        conversion
            .context
            .run(frame, &mut conversion.frame)
            .map_err(MfVirtualCameraError::Convert)?;
        Ok(&conversion.frame)
    }
}

/// A conversion of pictures shaped `from` to BT.709 limited-range NV12 at
/// `to` — what the camera's media types say its frames are.
fn conversion(
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    to: (u32, u32),
) -> std::result::Result<Conversion, MfVirtualCameraError> {
    let (format, width, height, space, range) = from;
    let mut context = ffmpeg::software::scaling::Context::get(
        format,
        width,
        height,
        ffmpeg::format::Pixel::NV12,
        to.0,
        to.1,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;
    // SAFETY: `context` is a live `SwsContext` this conversion owns, and the
    // tables are swscale's own static ones.
    unsafe {
        let source = ffmpeg::ffi::sws_getCoefficients(matrix(space));
        let destination = ffmpeg::ffi::sws_getCoefficients(ffmpeg::ffi::SWS_CS_ITU709);
        let source_full = is_rgb(format) || range == ffmpeg::color::Range::JPEG;
        ffmpeg::ffi::sws_setColorspaceDetails(
            context.as_mut_ptr(),
            source,
            i32::from(source_full),
            destination,
            0,
            0,
            1 << 16,
            1 << 16,
        );
    }
    Ok(Conversion {
        from,
        to,
        context,
        frame: ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, to.0, to.1),
    })
}

/// Whether `format` is a hardware frame's, whose pixels are not in it.
fn is_hardware(format: ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors.
    unsafe {
        let descriptor = ffmpeg::ffi::av_pix_fmt_desc_get(format.into());
        !descriptor.is_null()
            && (*descriptor).flags & (ffmpeg::ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0
    }
}

/// Packs `frame`'s two planes, each `linesize` apart, into `into`.
fn pack_nv12(frame: &ffmpeg::frame::Video, into: &mut [u8]) {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let (luma, chroma) = into.split_at_mut(width * height);
    for (row, out) in luma.chunks_exact_mut(width).enumerate() {
        let stride = frame.stride(0);
        out.copy_from_slice(&frame.data(0)[row * stride..row * stride + width]);
    }
    for (row, out) in chroma.chunks_exact_mut(width).enumerate() {
        let stride = frame.stride(1);
        out.copy_from_slice(&frame.data(1)[row * stride..row * stride + width]);
    }
}

impl Element for Feeding {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MfVirtualCamera
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for Feeding {
    /// Decoded video in system memory, any pixel layout: each picture is
    /// converted to whatever the reading application asked for.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(MfVirtualCameraError::UnsupportedBuffer(buf.kind()).into());
        };
        if is_hardware(frame.format()) {
            return Err(MfVirtualCameraError::UnsupportedBuffer("a hardware video frame").into());
        }
        let Some(view) = self.section().map(Mapping::view) else {
            return Ok(());
        };
        // SAFETY: the section is mapped for as long as `self.section` holds
        // it, which outlives this call.
        let header = unsafe { Header::at(view) };
        if !header.is_current() {
            pp_warn!(
                self,
                "the camera's section is from another version; nothing is shown"
            );
            return Ok(());
        }
        let Some(size) = header.wanted() else {
            return Ok(());
        };
        let converted = self.convert(frame, size)?;
        // SAFETY: as above; this element is the section's one writer.
        unsafe {
            protocol::write_frame(view, size.0, size.1, GetTickCount64() as i64, |into| {
                pack_nv12(converted, into)
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use windows::Win32::{
        Foundation::INVALID_HANDLE_VALUE,
        System::Memory::{CreateFileMappingW, FILE_MAP_ALL_ACCESS, PAGE_READWRITE},
    };

    use super::*;
    use crate::element::RawSink;

    /// The camera's part: a section under a `Local\` name, which needs no
    /// privilege, opened at `size`.
    struct FakeCamera {
        handle: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
    }

    impl FakeCamera {
        fn new(name: &str, size: Option<(u32, u32)>) -> Self {
            // SAFETY: a pagefile-backed mapping of the section's size.
            unsafe {
                let handle = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    None,
                    PAGE_READWRITE,
                    0,
                    protocol::SECTION_BYTES as u32,
                    &HSTRING::from(name),
                )
                .expect("section");
                let view =
                    MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, protocol::SECTION_BYTES);
                let camera = Self { handle, view };
                let header = camera.header();
                header.magic.store(protocol::MAGIC, Ordering::Release);
                header.version.store(protocol::VERSION, Ordering::Release);
                if let Some((width, height)) = size {
                    header.width.store(width, Ordering::Release);
                    header.height.store(height, Ordering::Release);
                }
                camera
            }
        }

        fn header(&self) -> &Header {
            // SAFETY: mapped for as long as `self` lives.
            unsafe { Header::at(self.view.Value.cast()) }
        }

        fn frame(&self, width: u32, height: u32) -> Vec<u8> {
            let mut frame = vec![0; protocol::nv12_bytes(width, height)];
            // SAFETY: as above.
            let read = unsafe {
                protocol::read_frame(self.view.Value.cast(), width, height, 0, &mut frame)
            };
            assert!(read, "a whole frame was written");
            frame
        }
    }

    impl Drop for FakeCamera {
        fn drop(&mut self) {
            // SAFETY: this fake's own view and handle.
            unsafe {
                let _ = UnmapViewOfFile(self.view);
                let _ = CloseHandle(self.handle);
            }
        }
    }

    /// A flat picture in `format`, `width` by `height`.
    fn picture(format: ffmpeg::format::Pixel, width: u32, height: u32, fill: u8) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(fill);
        }
        MediaBuffer::video(frame)
    }

    #[test]
    fn each_picture_reaches_the_camera_as_nv12_at_the_size_it_was_opened_at() {
        let name = "Local\\media-pp-mf-virtual-camera-test-size";
        let camera = FakeCamera::new(name, Some((640, 360)));
        let mut element = MfVirtualCamera::for_section(name);
        element
            .consume(picture(ffmpeg::format::Pixel::YUV420P, 1920, 1080, 128))
            .expect("render");
        assert_eq!(
            camera.header().sequence.load(Ordering::Acquire),
            2,
            "one frame written"
        );
        let frame = camera.frame(640, 360);
        // Within swscale's rounding of a 1920 to 640 shrink.
        assert!(
            frame.iter().all(|&byte| byte.abs_diff(128) <= 2),
            "a flat grey stays flat grey"
        );
    }

    #[test]
    fn nothing_is_written_while_no_application_reads_the_camera() {
        let name = "Local\\media-pp-mf-virtual-camera-test-idle";
        let camera = FakeCamera::new(name, None);
        let mut element = MfVirtualCamera::for_section(name);
        element
            .consume(picture(ffmpeg::format::Pixel::BGRA, 320, 240, 255))
            .expect("render");
        assert_eq!(camera.header().sequence.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_camera_opened_later_is_found_and_a_bgra_picture_is_converted_for_it() {
        let name = "Local\\media-pp-mf-virtual-camera-test-later";
        let mut element = MfVirtualCamera::for_section(name);
        element
            .consume(picture(ffmpeg::format::Pixel::BGRA, 320, 240, 255))
            .expect("nobody reads the camera yet");
        let camera = FakeCamera::new(name, Some((640, 360)));
        std::thread::sleep(LOOK_AGAIN_AFTER);
        element
            .consume(picture(ffmpeg::format::Pixel::BGRA, 320, 240, 255))
            .expect("render");
        let frame = camera.frame(640, 360);
        let luma = 640 * 360;
        assert!(
            frame[..luma].iter().all(|&byte| byte == 235),
            "white is BT.709 limited white"
        );
        assert!(
            frame[luma..].iter().all(|&byte| byte.abs_diff(128) <= 1),
            "and neutral"
        );
    }

    #[test]
    fn a_sound_frame_is_refused_by_name() {
        let mut element =
            MfVirtualCamera::for_section("Local\\media-pp-mf-virtual-camera-test-audio");
        let sound = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            16,
            ffmpeg::ChannelLayout::STEREO,
        );
        let error = element
            .consume(MediaBuffer::Audio(Arc::new(sound)))
            .expect_err("sound is not a picture");
        assert!(error.to_string().contains("system memory"), "{error}");
    }
}
