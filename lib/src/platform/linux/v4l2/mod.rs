//! What V4L2 is asked directly: which cameras there are and what each
//! offers, which FFmpeg's own demuxer cannot answer — see [`device`] — and
//! the writing end of a loopback device, which FFmpeg has no muxer for —
//! see [`loopback`].

#[cfg(feature = "v4l2-capture")]
mod device;
mod ioctl;
#[cfg(feature = "v4l2-virtual-camera")]
mod loopback;

#[cfg(feature = "v4l2-capture")]
pub use device::{V4l2CaptureFormat, format_name_for, list_devices, list_formats};
#[cfg(feature = "v4l2-virtual-camera")]
pub(crate) use loopback::{LoopbackError, LoopbackOutput, list_loopback_devices};

/// One V4L2 device node the machine currently has: a camera, or a loopback
/// device a program writes pictures into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V4l2Device {
    /// The device node, which is what a caller stores and opens by —
    /// `/dev/video0` and the like.
    ///
    /// Not stable across replugging on its own: the kernel hands out the
    /// lowest free number, so unplugging one camera can renumber another.
    /// It is what V4L2 offers, and what every tool on the platform names a
    /// camera by.
    pub id: String,
    /// What the driver calls the card, for a picker to show — a loopback
    /// device's is the `card_label` it was loaded with. Falls back to the
    /// node when a driver reports nothing.
    pub name: String,
}
