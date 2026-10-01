//! The source driven in-process, as Frame Server drives it, with no
//! registration: the tests make the activator directly and play the
//! producer's part on the section themselves, under a `Local\` name of
//! their own — only a service may create a `Global\` one.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::{
    Media::MediaFoundation::*,
    System::{
        Com::{COINIT_MULTITHREADED, CoInitializeEx, StructuredStorage::PROPVARIANT},
        SystemInformation::GetTickCount64,
    },
};
use windows_core::{BOOL, ComObject, GUID, IUnknown, Interface};

use crate::{activator::Activator, protocol, stream::frame_size};

/// Media Foundation, started for the test that holds it.
struct Mf;

impl Mf {
    fn start() -> Self {
        // SAFETY: initializes COM and Media Foundation for this thread and
        // process; `Drop` balances the latter.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET).expect("MFStartup");
        }
        Self
    }
}

impl Drop for Mf {
    fn drop(&mut self) {
        // SAFETY: balances `start`.
        let _ = unsafe { MFShutdown() };
    }
}

/// A section name no other test uses at the same time.
fn section_name() -> String {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    format!(
        "Local\\media-pp-vcam-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn source(section: &str) -> IMFMediaSource {
    let activator: IMFActivate =
        ComObject::new(Activator::new(section).expect("activator")).to_interface();
    // SAFETY: activates the source the activator makes.
    unsafe {
        activator
            .ActivateObject::<IMFMediaSource>()
            .expect("activate")
    }
}

/// The next event of `kind` from `generator`, within a second.
fn next_event(generator: &IMFMediaEventGenerator, kind: MF_EVENT_TYPE) -> IMFMediaEvent {
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        // SAFETY: a non-blocking read of the generator's queue.
        match unsafe { generator.GetEvent(MF_EVENT_FLAG_NO_WAIT) } {
            Ok(event) => {
                // SAFETY: reads the event's own type.
                if unsafe { event.GetType() }.expect("type") == kind.0 as u32 {
                    return event;
                }
            }
            Err(_) => std::thread::sleep(Duration::from_millis(2)),
        }
    }
    panic!("no event {kind:?} within a second");
}

/// Starts `source` at `width` by `height`, and returns its stream.
fn start(source: &IMFMediaSource, width: u32, height: u32) -> IMFMediaStream {
    // SAFETY: selects the size on a copy of the descriptor and starts the
    // source with it, as a reader does.
    unsafe {
        let descriptor = source.CreatePresentationDescriptor().expect("descriptor");
        let (mut selected, mut stream) = (BOOL::default(), None);
        descriptor
            .GetStreamDescriptorByIndex(0, &mut selected, &mut stream)
            .expect("stream");
        let handler = stream
            .expect("stream")
            .GetMediaTypeHandler()
            .expect("handler");
        let wanted = (0..handler.GetMediaTypeCount().expect("count"))
            .map(|index| handler.GetMediaTypeByIndex(index).expect("type"))
            .find(|media_type| frame_size(media_type).expect("size") == (width, height))
            .expect("the size is offered");
        handler.SetCurrentMediaType(&wanted).expect("set type");
        descriptor.SelectStream(0).expect("select");
        source
            .Start(&descriptor, &GUID::zeroed(), &PROPVARIANT::default())
            .expect("start");
        let event = next_event(&source.cast().expect("generator"), MENewStream);
        let value = event.GetValue().expect("value");
        let unknown: IUnknown = IUnknown::try_from(&value).expect("the new stream");
        unknown.cast().expect("a stream")
    }
}

/// The next sample the stream hands out after asking for one.
fn sample(stream: &IMFMediaStream) -> IMFSample {
    // SAFETY: asks for one sample and reads the event that carries it.
    unsafe {
        stream.RequestSample(None).expect("request");
        let event = next_event(&stream.cast().expect("generator"), MEMediaSample);
        let value = event.GetValue().expect("value");
        IUnknown::try_from(&value)
            .expect("the sample")
            .cast()
            .expect("a sample")
    }
}

/// A sample's bytes, from a packed buffer.
fn bytes(sample: &IMFSample) -> Vec<u8> {
    // SAFETY: locks the sample's one buffer to read it, and unlocks it.
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer().expect("buffer");
        let (mut data, mut length) = (std::ptr::null_mut(), 0u32);
        buffer
            .Lock(&mut data, None, Some(&mut length))
            .expect("lock");
        let copy = std::slice::from_raw_parts(data, length as usize).to_vec();
        buffer.Unlock().expect("unlock");
        copy
    }
}

fn now_ms() -> i64 {
    // SAFETY: a plain call.
    unsafe { GetTickCount64() as i64 }
}

#[test]
fn the_source_offers_nv12_at_every_size_and_says_frame_server_may_share_it() {
    let _mf = Mf::start();
    let source = source(&section_name());
    // SAFETY: reads what the source describes itself as.
    unsafe {
        assert_eq!(
            source.GetCharacteristics().expect("characteristics"),
            MFMEDIASOURCE_IS_LIVE.0 as u32
        );
        let descriptor = source.CreatePresentationDescriptor().expect("descriptor");
        assert_eq!(descriptor.GetStreamDescriptorCount().expect("count"), 1);
        let (mut selected, mut stream) = (BOOL::default(), None);
        descriptor
            .GetStreamDescriptorByIndex(0, &mut selected, &mut stream)
            .expect("stream");
        let handler = stream
            .expect("stream")
            .GetMediaTypeHandler()
            .expect("handler");
        let sizes: Vec<_> = (0..handler.GetMediaTypeCount().expect("count"))
            .map(|index| {
                let media_type = handler.GetMediaTypeByIndex(index).expect("type");
                assert_eq!(
                    media_type.GetGUID(&MF_MT_SUBTYPE).expect("subtype"),
                    MFVideoFormat_NV12
                );
                frame_size(&media_type).expect("size")
            })
            .collect();
        assert_eq!(sizes, protocol::SIZES);
        let ex: IMFMediaSourceEx = source.cast().expect("IMFMediaSourceEx");
        let attributes = ex.GetStreamAttributes(0).expect("stream attributes");
        assert_eq!(
            attributes
                .GetUINT32(&MF_DEVICESTREAM_FRAMESERVER_SHARED)
                .expect("shared"),
            1
        );
        source.Shutdown().expect("shutdown");
    }
}

#[test]
fn a_started_stream_says_its_size_and_hands_on_the_producers_frame() {
    let _mf = Mf::start();
    let name = section_name();
    let source = source(&name);
    let (width, height) = protocol::SIZES[2];
    let stream = start(&source, width, height);

    // The producer's side: the section the stream made, and the size in it.
    let section = Producer::open(&name).expect("the stream made the section");
    assert_eq!(section.header().wanted(), Some((width, height)));

    let placeholder = bytes(&sample(&stream));
    assert_eq!(placeholder.len(), protocol::nv12_bytes(width, height));
    assert!(
        placeholder.iter().all(|&byte| byte == 32 || byte == 128),
        "the dark placeholder"
    );

    // SAFETY: the section is mapped for the rest of the test.
    unsafe {
        protocol::write_frame(section.view(), width, height, now_ms(), |frame| {
            for (index, byte) in frame.iter_mut().enumerate() {
                *byte = (index % 251) as u8;
            }
        })
    };
    let frame = bytes(&sample(&stream));
    assert!(
        frame
            .iter()
            .enumerate()
            .all(|(index, &byte)| byte == (index % 251) as u8)
    );

    // SAFETY: stops and shuts down the source.
    unsafe {
        source.Stop().expect("stop");
        assert_eq!(
            section.header().wanted(),
            None,
            "a stopped stream wants nothing"
        );
        source.Shutdown().expect("shutdown");
    }
}

#[test]
fn samples_come_at_the_cameras_rate_however_fast_they_are_asked_for() {
    let _mf = Mf::start();
    let source = source(&section_name());
    let stream = start(&source, protocol::SIZES[2].0, protocol::SIZES[2].1);
    let generator: IMFMediaEventGenerator = stream.cast().expect("generator");
    let count = 10;
    let started = Instant::now();
    // SAFETY: asks for `count` samples at once, then reads them.
    unsafe {
        for _ in 0..count {
            stream.RequestSample(None).expect("request");
        }
    }
    for _ in 0..count {
        next_event(&generator, MEMediaSample);
    }
    let took = started.elapsed();
    let interval = Duration::from_secs(1) / protocol::FRAME_RATE;
    assert!(
        took >= interval * (count - 1),
        "{count} samples in {took:?}"
    );
    // SAFETY: shuts the source down.
    unsafe { source.Shutdown().expect("shutdown") };
}

#[test]
fn a_frame_server_allocators_samples_are_filled_row_by_row() {
    let _mf = Mf::start();
    let name = section_name();
    let source = source(&name);
    // SAFETY: an allocator of Media Foundation's own, handed over as Frame
    // Server hands over its own.
    unsafe {
        let mut raw = std::ptr::null_mut();
        MFCreateVideoSampleAllocatorEx(&IMFVideoSampleAllocatorEx::IID, &mut raw)
            .expect("allocator");
        let allocator = IMFVideoSampleAllocatorEx::from_raw(raw);
        let control: IMFSampleAllocatorControl = source.cast().expect("allocator control");
        control
            .SetDefaultAllocator(0, &allocator)
            .expect("set allocator");
    }
    let (width, height) = protocol::SIZES[1];
    let stream = start(&source, width, height);
    let section = Producer::open(&name).expect("section");
    // SAFETY: the section is mapped for the rest of the test.
    unsafe {
        protocol::write_frame(section.view(), width, height, now_ms(), |frame| {
            let luma = width as usize * height as usize;
            frame[..luma].fill(200);
            frame[luma..].fill(90);
        })
    };
    let frame = bytes(&sample(&stream));
    let luma = width as usize * height as usize;
    assert!(frame[..luma].iter().all(|&byte| byte == 200), "luma");
    assert!(
        frame[luma..protocol::nv12_bytes(width, height)]
            .iter()
            .all(|&byte| byte == 90),
        "chroma"
    );
    // SAFETY: shuts the source down.
    unsafe { source.Shutdown().expect("shutdown") };
}

/// The section opened as the producing application opens it: read and
/// write, the only rights the camera grants a signed-in user.
struct Producer {
    handle: windows::Win32::Foundation::HANDLE,
    view: windows::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
}

impl Producer {
    fn open(name: &str) -> windows_core::Result<Self> {
        use windows::Win32::System::Memory::{
            FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, OpenFileMappingW,
        };
        let access = FILE_MAP_READ | FILE_MAP_WRITE;
        // SAFETY: opens a named mapping and maps all of it.
        unsafe {
            let handle = OpenFileMappingW(access.0, false, &windows_core::HSTRING::from(name))?;
            let view = MapViewOfFile(handle, access, 0, 0, protocol::SECTION_BYTES);
            if view.Value.is_null() {
                return Err(windows_core::Error::from_thread());
            }
            Ok(Self { handle, view })
        }
    }

    fn view(&self) -> *mut u8 {
        self.view.Value.cast()
    }

    fn header(&self) -> &protocol::Header {
        // SAFETY: mapped for as long as `self` lives.
        unsafe { protocol::Header::at(self.view()) }
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        // SAFETY: this producer's own view and handle.
        unsafe {
            let _ = windows::Win32::System::Memory::UnmapViewOfFile(self.view);
            let _ = windows::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}
