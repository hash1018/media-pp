//! What a [`MfVirtualCamera`](crate::elements::MfVirtualCamera) and the
//! camera it registers agree on — the one file both sides compile.
//!
//! The camera is a Media Foundation media source in a DLL of its own
//! (`media-pp-vcam`), which Windows' Frame Server loads into a service
//! process to serve every application reading the camera. The element runs
//! in the application producing the pictures. Between the two is one named
//! section of shared memory: a [`Header`] and room for one NV12 frame.
//!
//! The camera creates the section, when a stream starts, because only the
//! service it runs in may name an object in the `Global\` namespace an
//! ordinary process can then open; it writes the size the stream was opened
//! at. The element opens the section once it exists, writes each picture at
//! that size, and does nothing while there is no section or no size — no one
//! is watching. The frame is guarded by a sequence count, odd while it is
//! written, so the camera never hands on half of one.
//!
//! This file has no dependencies beyond `std`: the camera DLL includes it
//! with `#[path]` and links nothing of FFmpeg, which the service it is
//! loaded into would not find.

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering, fence};

/// The CLSID the camera's media source is registered under.
pub const CLSID: &str = "{51BEF1F3-9256-49A2-B965-EAA4D87CC2E6}";

/// [`CLSID`] as a number, for the side that needs a `GUID`.
pub const CLSID_U128: u128 = 0x51BEF1F3_9256_49A2_B965_EAA4D87CC2E6;

/// The section's name: in the global namespace, since the camera runs in
/// session 0 and the element in the user's session.
pub const SECTION_NAME: &str = "Global\\MediaPpVirtualCamera";

/// What a section begins with, so a stale or foreign one is not misread.
pub const MAGIC: u32 = u32::from_le_bytes(*b"MPVC");

/// Changed whenever [`Header`] or the frame layout changes, so that an
/// element and a camera built from different versions of this file refuse
/// each other rather than misread each other.
pub const VERSION: u32 = 1;

/// The rate the camera offers, in frames per second.
pub const FRAME_RATE: u32 = 30;

/// The sizes the camera offers, largest first; the application reading the
/// camera picks one, and the element scales to it.
pub const SIZES: [(u32, u32); 3] = [(1920, 1080), (1280, 720), (640, 360)];

/// Where the frame begins: the header, rounded up to a page.
pub const HEADER_BYTES: usize = 4096;

/// Room for one NV12 frame of the largest of [`SIZES`].
pub const MAX_FRAME_BYTES: usize = 1920 * 1080 * 3 / 2;

/// The section's whole size.
pub const SECTION_BYTES: usize = HEADER_BYTES + MAX_FRAME_BYTES;

/// How old the last picture may be before the camera shows its placeholder
/// instead: the producer has stopped or gone.
pub const STALE_AFTER_MS: u64 = 1000;

/// The bytes of an NV12 frame `width` by `height`, packed: a luma plane of
/// `width` bytes a row, then the interleaved chroma at half the height.
pub const fn nv12_bytes(width: u32, height: u32) -> usize {
    width as usize * height as usize * 3 / 2
}

/// The start of a section.
#[repr(C)]
pub struct Header {
    /// [`MAGIC`], written by the camera when it creates the section.
    pub magic: AtomicU32,
    /// [`VERSION`], likewise.
    pub version: AtomicU32,
    /// The size the camera's stream is open at; zero while no stream is.
    /// Written by the camera.
    pub width: AtomicU32,
    /// See [`Self::width`].
    pub height: AtomicU32,
    /// Odd while the element writes a frame, even once it has; zero until
    /// the first frame. Written by the element.
    pub sequence: AtomicU64,
    /// The size of the frame last written, packed and NV12. Written by the
    /// element inside the sequence.
    pub frame_width: AtomicU32,
    /// See [`Self::frame_width`].
    pub frame_height: AtomicU32,
    /// When the last frame was written, by `GetTickCount64`, which both
    /// processes read the same. Written by the element.
    pub written_at_ms: AtomicI64,
}

const _: () = assert!(std::mem::size_of::<Header>() <= HEADER_BYTES);

impl Header {
    /// The header at the start of a mapped section.
    ///
    /// # Safety
    ///
    /// `view` is the start of a mapping of at least [`SECTION_BYTES`] that
    /// outlives the returned reference.
    pub unsafe fn at<'a>(view: *mut u8) -> &'a Header {
        // SAFETY: the caller guarantees the mapping; the header is all
        // atomics, so any process may read and write it concurrently.
        unsafe { &*view.cast::<Header>() }
    }

    /// Whether this is a section of this version.
    pub fn is_current(&self) -> bool {
        self.magic.load(Ordering::Acquire) == MAGIC
            && self.version.load(Ordering::Acquire) == VERSION
    }

    /// The size the camera is open at, if it is.
    pub fn wanted(&self) -> Option<(u32, u32)> {
        let width = self.width.load(Ordering::Acquire);
        let height = self.height.load(Ordering::Acquire);
        (width > 0 && height > 0).then_some((width, height))
    }
}

/// Writes one frame into the section: `write` is handed the frame's bytes,
/// [`nv12_bytes`] long for `width` by `height`, and fills them.
///
/// # Safety
///
/// `view` is a mapping as for [`Header::at`], and only one writer at a time
/// writes to it.
pub unsafe fn write_frame(
    view: *mut u8,
    width: u32,
    height: u32,
    now_ms: i64,
    write: impl FnOnce(&mut [u8]),
) {
    let bytes = nv12_bytes(width, height);
    assert!(bytes <= MAX_FRAME_BYTES, "{width}x{height} does not fit");
    // SAFETY: as the caller guarantees.
    let header = unsafe { Header::at(view) };
    header.sequence.fetch_add(1, Ordering::AcqRel);
    fence(Ordering::Release);
    header.frame_width.store(width, Ordering::Relaxed);
    header.frame_height.store(height, Ordering::Relaxed);
    // SAFETY: the frame region begins after the header and is at least
    // `MAX_FRAME_BYTES` long; the sequence count keeps a reader from
    // trusting what it copied while this runs.
    write(unsafe { std::slice::from_raw_parts_mut(view.add(HEADER_BYTES), bytes) });
    header.written_at_ms.store(now_ms, Ordering::Relaxed);
    header.sequence.fetch_add(1, Ordering::Release);
}

/// Copies the last whole frame out of the section into `into`, if there is
/// one at `width` by `height` written no earlier than `stale_before_ms`.
/// Retries a few times if the writer is part way through; `false` means
/// there was nothing to copy.
///
/// # Safety
///
/// `view` is a mapping as for [`Header::at`].
pub unsafe fn read_frame(
    view: *mut u8,
    width: u32,
    height: u32,
    stale_before_ms: i64,
    into: &mut [u8],
) -> bool {
    let bytes = nv12_bytes(width, height);
    if bytes > MAX_FRAME_BYTES || into.len() < bytes {
        return false;
    }
    // SAFETY: as the caller guarantees.
    let header = unsafe { Header::at(view) };
    for _ in 0..4 {
        let before = header.sequence.load(Ordering::Acquire);
        if before == 0 {
            return false;
        }
        if before % 2 == 1 {
            std::thread::yield_now();
            continue;
        }
        let fresh = header.written_at_ms.load(Ordering::Relaxed) >= stale_before_ms;
        let fits = header.frame_width.load(Ordering::Relaxed) == width
            && header.frame_height.load(Ordering::Relaxed) == height;
        if !fresh || !fits {
            return false;
        }
        // SAFETY: the frame region is `MAX_FRAME_BYTES` long and `bytes`
        // fits in it; a torn copy is discarded below.
        unsafe { std::ptr::copy_nonoverlapping(view.add(HEADER_BYTES), into.as_mut_ptr(), bytes) };
        fence(Ordering::Acquire);
        if header.sequence.load(Ordering::Relaxed) == before {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section() -> Vec<u64> {
        vec![0u64; SECTION_BYTES / 8]
    }

    #[test]
    fn a_frame_written_is_read_back_whole() {
        let mut memory = section();
        let view = memory.as_mut_ptr().cast::<u8>();
        let (width, height) = SIZES[2];
        let mut into = vec![0u8; nv12_bytes(width, height)];
        // SAFETY: `memory` is a section's size and outlives every use.
        unsafe {
            assert!(
                !read_frame(view, width, height, 0, &mut into),
                "nothing yet"
            );
            write_frame(view, width, height, 100, |frame| frame.fill(7));
            assert!(read_frame(view, width, height, 100, &mut into));
        }
        assert!(into.iter().all(|&byte| byte == 7));
    }

    #[test]
    fn a_frame_of_another_size_or_too_old_is_not_read() {
        let mut memory = section();
        let view = memory.as_mut_ptr().cast::<u8>();
        let mut into = vec![0u8; MAX_FRAME_BYTES];
        // SAFETY: as above.
        unsafe {
            write_frame(view, 640, 360, 100, |frame| frame.fill(1));
            assert!(!read_frame(view, 1280, 720, 0, &mut into), "another size");
            assert!(!read_frame(view, 640, 360, 101, &mut into), "too old");
        }
    }
}
