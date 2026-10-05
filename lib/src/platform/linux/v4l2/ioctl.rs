//! What every V4L2 call here shares: how a request number is made, how one
//! is issued, and the one question asked of every node — what it is.
//!
//! # The request numbers
//!
//! An ioctl request encodes its direction, the size of the struct it carries,
//! a type letter and an ordinal. They are computed here from those parts
//! rather than pasted in as magic constants, so a struct whose layout is
//! wrong shows up as a mismatched request number in a test rather than as a
//! driver writing past the end of it.

use std::ffi::{OsStr, c_void};
use std::fs;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Issues `request` on `file` with `argument` as the struct it carries.
pub(super) fn call<T>(
    file: &std::fs::File,
    request: libc::c_ulong,
    argument: &mut T,
) -> std::io::Result<()> {
    // SAFETY: `argument` is a live value of the layout this request number was
    // computed from, and the descriptor belongs to `file` for the whole call.
    let code = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            request,
            std::ptr::from_mut(argument).cast::<c_void>(),
        )
    };
    if code < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// `_IOC`, as `linux/ioctl.h` defines it: direction, then the size of the
/// struct that travels with the request, then the type letter and the
/// ordinal.
pub(super) const fn request(direction: u32, ordinal: u32, size: usize) -> libc::c_ulong {
    ((direction << 30) | ((size as u32) << 16) | (b'V' as u32) << 8 | ordinal) as libc::c_ulong
}

/// The driver writes, the caller does not.
pub(super) const READ: u32 = 2;
/// Both: what goes in is read, and the answer comes back in the same struct.
pub(super) const READ_WRITE: u32 = 3;

pub(super) const VIDIOC_QUERYCAP: libc::c_ulong = request(READ, 0, size_of::<Capability>());

const CAP_DEVICE_CAPS: u32 = 0x8000_0000;

/// Every `/dev/videoN` node, in node order.
pub(super) fn video_nodes() -> std::io::Result<Vec<PathBuf>> {
    let mut nodes: Vec<PathBuf> = fs::read_dir("/dev")?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_video_node(path))
        .collect();
    nodes.sort();
    Ok(nodes)
}

fn is_video_node(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| {
            name.strip_prefix("video").is_some_and(|rest| {
                !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
            })
        })
}

/// What the node at `file` says it is.
pub(super) fn query_capability(file: &std::fs::File) -> std::io::Result<Capability> {
    let mut capability = Capability::default();
    // SAFETY: the file is a live V4L2 node and `capability` is a live local
    // of exactly the layout `VIDIOC_QUERYCAP`'s request number encodes — see
    // this module's own docs and `the_request_numbers_match_the_kernels`.
    call(file, VIDIOC_QUERYCAP, &mut capability)?;
    Ok(capability)
}

/// `struct v4l2_capability`.
#[repr(C)]
#[derive(Default)]
pub(super) struct Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

impl Capability {
    /// What *this node* does, which is not the same question as what the
    /// device does: a camera's metadata node belongs to a device whose
    /// `capabilities` says it captures, and answers nothing itself.
    /// `device_caps` is the per-node answer, offered since Linux 3.3 and
    /// flagged in `capabilities` when it is there.
    pub(super) fn node_caps(&self) -> u32 {
        if self.capabilities & CAP_DEVICE_CAPS != 0 {
            self.device_caps
        } else {
            self.capabilities
        }
    }

    /// What the driver calls the card, for a list to show.
    pub(super) fn card(&self) -> Option<String> {
        text(&self.card)
    }

    /// The driver's own name — `"v4l2 loopback"` for a loopback device.
    #[cfg(feature = "v4l2-virtual-camera")]
    pub(super) fn driver(&self) -> Option<String> {
        text(&self.driver)
    }
}

/// A NUL-terminated field of a V4L2 struct, as text.
fn text(field: &[u8]) -> Option<String> {
    let end = field.iter().position(|byte| *byte == 0)?;
    let name = OsStr::from_bytes(&field[..end])
        .to_string_lossy()
        .trim()
        .to_owned();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number computed here against the one the kernel's own headers
    /// publish — see the module docs for why that is the safety argument.
    #[test]
    fn the_request_numbers_match_the_kernels() {
        assert_eq!(VIDIOC_QUERYCAP, 0x8068_5600);
    }

    #[test]
    fn only_numbered_video_nodes_are_video_nodes() {
        assert!(is_video_node(Path::new("/dev/video0")));
        assert!(is_video_node(Path::new("/dev/video12")));
        assert!(!is_video_node(Path::new("/dev/video")));
        assert!(!is_video_node(Path::new("/dev/video-codec")));
        assert!(!is_video_node(Path::new("/dev/media0")));
    }
}
