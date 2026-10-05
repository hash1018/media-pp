//! What travels between elements.
//!
//! [`MediaBuffer`] is an enum rather than one opaque buffer type, because
//! `ffmpeg-next` already hands back strongly-typed packets and frames and
//! collapsing them would only mean unwrapping again downstream. Its own
//! documentation covers why each payload is shared rather than copied, and
//! why a video frame arrives through a pool reference.
//!
//! Every buffer can carry [`Metadata`]: what an element found out about it,
//! such as the objects a detector saw in a picture.

use std::any::{Any, TypeId};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::pool::{UnboundObjectPool, UnboundObjectPoolRef};

/// The unit of data that flows between elements.
///
/// Compressed and uncompressed data are kept as distinct variants (rather
/// than a single opaque `Buffer` type like GStreamer) because ffmpeg-next
/// already gives us strongly-typed `Packet`/`Frame` types — collapsing them
/// into one type would just mean unwrapping again downstream.
///
/// Payloads are `Arc`-wrapped so `MediaBuffer` is cheaply `Clone` —
/// duplicating a buffer (e.g. [`crate::elements::Tee`] fanning packets out
/// to a decode branch and a remux branch) is a refcount bump, never a copy
/// of the encoded/decoded data.
///
/// `Video` specifically wraps an [`UnboundObjectPoolRef`], not a plain
/// `ffmpeg::frame::Video` — that's what lets whichever element produced it
/// (see [`crate::pool::UnboundObjectPool`], owned as that element's own
/// struct field) get the underlying buffer back automatically once every
/// `Arc` clone downstream has been dropped, instead of it just being freed.
///
/// Each payload sits in a wrapper — [`PacketBuffer`], [`VideoBuffer`],
/// [`AudioBuffer`] — beside the buffer's [`Metadata`], and the wrapper
/// dereferences to the payload's `Arc`: `MediaBuffer::Video(frame) =>
/// frame.width()` reads the picture as it always has. A wrapper is made
/// from its `Arc` with `.into()`.
#[derive(Clone)]
pub enum MediaBuffer {
    /// Encoded media packet produced by a demuxer or encoder.
    ///
    /// Its PTS, DTS, duration, stream index, and time base remain part of the
    /// packet contract while it travels through packet-level elements.
    Packet(PacketBuffer),

    /// Decoded video frame whose backing storage returns to its producer's
    /// [`crate::pool::UnboundObjectPool`] after the last `Arc` clone drops.
    ///
    /// Treat the published frame as immutable. A transforming element creates
    /// a replacement frame and preserves PTS, its [time base](time_base),
    /// duration, and color metadata unless it intentionally establishes a new
    /// timeline — in which case it sets the time base of that timeline.
    Video(VideoBuffer),

    /// Decoded audio frame shared immutably between downstream branches.
    ///
    /// Sample format, sample rate, channel layout, PTS, its
    /// [time base](time_base), and duration describe the audio contract a
    /// transforming element must either preserve or deliberately replace.
    Audio(AudioBuffer),
}

impl MediaBuffer {
    /// A video frame made by hand — a still picture, a rendered caption, a
    /// test pattern — as a [`MediaBuffer::Video`], without the caller
    /// having to know that one carries a pooled frame.
    ///
    /// Nothing recycles it: its buffers are freed once the last clone
    /// downstream is dropped. Something that makes frames over and over
    /// keeps an [`UnboundObjectPool`] of its own and sends
    /// `MediaBuffer::Video(Arc::new(pool.get()).into())`, so each frame goes
    /// back to be drawn into again.
    pub fn video(frame: ffmpeg::frame::Video) -> Self {
        // A pool of zero keeps nothing: the one frame taken from it is
        // dropped, not returned, when its last reference goes.
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        MediaBuffer::Video(Arc::new(slot).into())
    }

    /// A packet as a [`MediaBuffer::Packet`].
    pub fn packet(packet: ffmpeg::Packet) -> Self {
        MediaBuffer::Packet(Arc::new(packet).into())
    }

    /// A sound as a [`MediaBuffer::Audio`].
    pub fn audio(frame: ffmpeg::frame::Audio) -> Self {
        MediaBuffer::Audio(Arc::new(frame).into())
    }

    /// Stable, human-readable variant name for diagnostics emitted when
    /// elements are wired to an incompatible media type.
    pub fn kind(&self) -> &'static str {
        match self {
            MediaBuffer::Packet(_) => "Packet",
            MediaBuffer::Video(_) => "Video",
            MediaBuffer::Audio(_) => "Audio",
        }
    }

    /// What has been found out about this buffer, if anything has.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata_arc().map(Arc::as_ref)
    }

    /// The same, shared — for an element handing it on to a buffer of its
    /// own making.
    pub fn metadata_arc(&self) -> Option<&Arc<Metadata>> {
        match self {
            MediaBuffer::Packet(buffer) => buffer.metadata.as_ref(),
            MediaBuffer::Video(buffer) => buffer.metadata.as_ref(),
            MediaBuffer::Audio(buffer) => buffer.metadata.as_ref(),
        }
    }

    /// This buffer carrying `metadata` in place of whatever it carried. The
    /// payload is shared, not copied.
    #[must_use]
    pub fn with_metadata(mut self, metadata: impl Into<Arc<Metadata>>) -> Self {
        self.set_metadata(Some(metadata.into()));
        self
    }

    /// Replaces what this buffer carries — `None` takes it away.
    pub fn set_metadata(&mut self, metadata: Option<Arc<Metadata>>) {
        match self {
            MediaBuffer::Packet(buffer) => buffer.metadata = metadata,
            MediaBuffer::Video(buffer) => buffer.metadata = metadata,
            MediaBuffer::Audio(buffer) => buffer.metadata = metadata,
        }
    }
}

/// Makes a payload wrapper: the `Arc` beside the buffer's metadata, read
/// through as the `Arc` it holds.
macro_rules! payload_buffer {
    ($(#[$doc:meta])* $name:ident, $payload:ty) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            payload: Arc<$payload>,
            metadata: Option<Arc<Metadata>>,
        }

        impl $name {
            /// `payload`, carrying nothing yet.
            pub fn new(payload: Arc<$payload>) -> Self {
                Self {
                    payload,
                    metadata: None,
                }
            }

            /// The payload's `Arc`, to keep or hand on.
            pub fn payload(&self) -> &Arc<$payload> {
                &self.payload
            }

            /// The payload's `Arc`, the wrapper let go of.
            pub fn into_payload(self) -> Arc<$payload> {
                self.payload
            }

            /// What has been found out about this buffer, if anything has.
            pub fn metadata(&self) -> Option<&Metadata> {
                self.metadata.as_deref()
            }

            /// The same, shared.
            pub fn metadata_arc(&self) -> Option<&Arc<Metadata>> {
                self.metadata.as_ref()
            }

            /// This buffer carrying `metadata` in place of whatever it
            /// carried.
            #[must_use]
            pub fn with_metadata(mut self, metadata: impl Into<Arc<Metadata>>) -> Self {
                self.metadata = Some(metadata.into());
                self
            }
        }

        impl Deref for $name {
            type Target = Arc<$payload>;

            fn deref(&self) -> &Self::Target {
                &self.payload
            }
        }

        /// Lets an element that holds the only reference write through it,
        /// with `Arc::get_mut`, or put another payload in its place; what
        /// the buffer carries stays.
        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.payload
            }
        }

        impl From<Arc<$payload>> for $name {
            fn from(payload: Arc<$payload>) -> Self {
                Self::new(payload)
            }
        }
    };
}

payload_buffer!(
    /// What [`MediaBuffer::Packet`] holds: the packet, and its
    /// [`Metadata`].
    PacketBuffer,
    ffmpeg::Packet
);

payload_buffer!(
    /// What [`MediaBuffer::Video`] holds: the pooled frame, and its
    /// [`Metadata`].
    VideoBuffer,
    UnboundObjectPoolRef<ffmpeg::frame::Video>
);

payload_buffer!(
    /// What [`MediaBuffer::Audio`] holds: the frame, and its [`Metadata`].
    AudioBuffer,
    ffmpeg::frame::Audio
);

/// What has been found out about a buffer, one value per type.
///
/// Open rather than a fixed set of fields, so an element defines the type
/// of what it finds — a detector its detections, a tracker its tracks — and
/// reads another's by asking for that type. One value of a type at a time:
/// putting a second replaces the first.
///
/// A buffer holds it behind an `Arc` and never changes it in place: an
/// element adding to it makes a copy with [`Self::with`] and puts that on
/// the buffer it hands on, so a buffer a [`crate::elements::Tee`] shared
/// keeps what it had in every other branch.
///
/// A [`crate::element::Filter`] that makes a buffer of the same sort from
/// the one it was handed — a picture from a picture — finds this carried
/// across by the framework; see that trait.
#[derive(Clone, Default)]
pub struct Metadata {
    entries: Vec<Entry>,
}

#[derive(Clone)]
struct Entry {
    type_id: TypeId,
    type_name: &'static str,
    value: Arc<dyn Any + Send + Sync>,
}

impl Metadata {
    /// Nothing found yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The value of type `T`, where there is one.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.entries
            .iter()
            .find(|entry| entry.type_id == TypeId::of::<T>())
            .and_then(|entry| entry.value.downcast_ref::<T>())
    }

    /// Whether there is a value of type `T`.
    pub fn contains<T: Any + Send + Sync>(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.type_id == TypeId::of::<T>())
    }

    /// Puts `value` in, in place of any value of its type, and says what it
    /// replaced.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) -> Option<Arc<dyn Any + Send + Sync>> {
        let entry = Entry {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            value: Arc::new(value),
        };
        match self
            .entries
            .iter_mut()
            .find(|existing| existing.type_id == entry.type_id)
        {
            Some(existing) => Some(std::mem::replace(existing, entry).value),
            None => {
                self.entries.push(entry);
                None
            }
        }
    }

    /// This, with `value` in place of any value of its type.
    #[must_use]
    pub fn with<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert(value);
        self
    }

    /// Takes the value of type `T` out, and says whether there was one.
    pub fn remove<T: Any + Send + Sync>(&mut self) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|entry| entry.type_id != TypeId::of::<T>());
        self.entries.len() != before
    }

    /// How many values there are.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Debug for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(self.entries.iter().map(|entry| entry.type_name))
            .finish()
    }
}

/// The unit a decoded frame's timestamps are in, or `None` where nothing has
/// said.
///
/// A frame's PTS is a count, and a count means nothing without its unit. A
/// packet carries its own; FFmpeg's decoders do not yet copy it onto the
/// frames they produce, so every element in this crate that makes a frame
/// sets it with [`set_time_base`], and every one that passes a frame's
/// timestamps on passes this with them. That is what lets a
/// [`crate::elements::Pacer`] read the unit off the frame instead of being
/// told it — told separately, it could be told the wrong stream's, and
/// playback would run at the wrong speed without a word.
///
/// Takes a video or an audio frame alike, through `Deref`.
pub fn time_base(frame: &ffmpeg::Frame) -> Option<ffmpeg::Rational> {
    // SAFETY: `frame` is a live `AVFrame`; `time_base` is a plain field.
    let time_base = unsafe { (*frame.as_ptr()).time_base };
    (time_base.num > 0 && time_base.den > 0).then(|| time_base.into())
}

/// Says what unit `frame`'s timestamps are in — see [`time_base`]. An
/// element that makes a frame of its own, or a caller pushing one through
/// an [`crate::elements::AppSource`], sets this beside the PTS.
pub fn set_time_base(frame: &mut ffmpeg::Frame, time_base: ffmpeg::Rational) {
    // SAFETY: `frame` is a live `AVFrame` borrowed mutably; `time_base` is
    // a plain field no other part of the frame depends on.
    unsafe { (*frame.as_mut_ptr()).time_base = time_base.into() };
}

/// Hands `source`'s timing to `destination`: its PTS, and the unit it is in.
///
/// For an element that makes its output frame itself rather than with
/// `av_frame_copy_props` — which already carries both, and every other
/// property besides.
pub(crate) fn carry_timing(destination: &mut ffmpeg::Frame, source: &ffmpeg::Frame) {
    destination.set_pts(source.pts());
    // SAFETY: both are live `AVFrame`s and distinct, `destination` borrowed
    // mutably; `time_base` is a plain field.
    unsafe { (*destination.as_mut_ptr()).time_base = (*source.as_ptr()).time_base };
}

/// Which buffer a video frame's pixels live in.
///
/// Not the frame: a producer with nothing new to show re-emits a fresh
/// `AVFrame` referencing the same picture every tick — a screen capture of a
/// still desktop does exactly that — so comparing frames, or the `Arc`s
/// around them, answers "changed" every time while the pixels have not
/// moved. The plane pointers do not.
///
/// The first two planes are enough for every layout this crate carries:
/// packed formats use one, and the semi-planar and planar ones this crate
/// composites in differ in the first two whenever they differ at all. A
/// hardware frame that keeps its picture in the fourth pointer instead —
/// VideoToolbox's `CVPixelBuffer` — has nothing in the first two, and is
/// told apart by that one.
///
/// Only sound as an identity while the frame it came from is still
/// referenced. A picture whose buffer has been released can be handed out
/// again at the same address, so every caller here holds that reference for
/// as long as it holds the identity.
pub(crate) fn picture_id(frame: &ffmpeg::frame::Video) -> (usize, usize) {
    // SAFETY: `as_ptr` is a live `AVFrame`. Only the values of its plane
    // pointers are read; nothing dereferences them, which for GPU memory
    // would not be valid from the host anyway.
    unsafe {
        let ptr = frame.as_ptr();
        match ((*ptr).data[0] as usize, (*ptr).data[1] as usize) {
            (0, 0) => ((*ptr).data[3] as usize, 0),
            planes => planes,
        }
    }
}

/// Lets go of the picture a pooled wrapper was pointing at, as it returns to
/// its pool.
///
/// The `release` an [`UnboundObjectPool`](crate::pool::UnboundObjectPool) of
/// *wrappers* wants: an empty frame is given a picture with `av_frame_ref` on
/// every checkout, and without this it would keep that reference until its
/// next one. Nothing reads those pixels in the meantime, but everything that
/// asks whether a picture is still in use — [`picture_is_referenced`] — would
/// go on answering yes, so a producer would keep a buffer, or a screen-sized
/// texture, alive for an idle wrapper.
///
/// Only for a pool whose items own no picture of their own. A pool of real
/// frames composited or decoded into would be emptied by this.
pub(crate) fn release_picture(frame: &mut ffmpeg::frame::Video) {
    // SAFETY: `as_mut_ptr` is this frame's own live `AVFrame`, and the pool
    // has taken it back, so nothing else refers to it.
    unsafe { ffi::av_frame_unref(frame.as_mut_ptr()) }
}

/// Whether anything besides this frame itself still points at its picture.
///
/// The companion to [`picture_id`], for an element that offers an unchanged
/// picture again by pointing an empty wrapper at it with `av_frame_ref`.
/// Such a wrapper shares the picture's *buffer*, not the pool slot the frame
/// came from, so an [`UnboundObjectPoolRef`] that has gone back to its pool
/// says nothing about whether a wrapper downstream is still showing those
/// pixels — the buffer's own reference count is the only record of it, and
/// a producer that recycles its frames has to keep one out of the pool until
/// this reads false for it.
///
/// Only the first buffer is examined, for the same reason [`picture_id`]
/// reads only the first plane pointers: a wrapper references either all of a
/// frame's buffers or none of them.
pub(crate) fn picture_is_referenced(frame: &ffmpeg::frame::Video) -> bool {
    // SAFETY: `as_ptr` is a live `AVFrame`. `buf[0]` is either null (a frame
    // that owns no picture) or a live `AVBufferRef`, and reading its count
    // does not touch the picture itself — which for GPU memory would not be
    // valid from the host anyway.
    unsafe {
        let buf = (*frame.as_ptr()).buf[0];
        !buf.is_null() && ffi::av_buffer_get_ref_count(buf) > 1
    }
}

#[cfg(test)]
mod tests {

    /// Two frames whose picture is in the fourth pointer — VideoToolbox's —
    /// are two pictures, not one because their first two are both empty.
    /// Taken for one, every VideoToolbox frame after the first was answered
    /// with the first's download.
    #[test]
    fn a_picture_in_the_fourth_pointer_is_told_apart() {
        let mut first = ffmpeg::frame::Video::empty();
        let mut second = ffmpeg::frame::Video::empty();
        // SAFETY: only the pointer values are set, and nothing reads through
        // them; both are cleared again before the frames are freed, which
        // then frees no picture.
        unsafe {
            (*first.as_mut_ptr()).data[3] = 0x1000 as *mut u8;
            (*second.as_mut_ptr()).data[3] = 0x2000 as *mut u8;
        }
        let ids = (picture_id(&first), picture_id(&second));
        // SAFETY: as above.
        unsafe {
            (*first.as_mut_ptr()).data[3] = std::ptr::null_mut();
            (*second.as_mut_ptr()).data[3] = std::ptr::null_mut();
        }
        assert_ne!(ids.0, ids.1);
        assert_eq!(ids.0, (0x1000, 0));
    }
    use super::*;

    /// A hand-made frame goes in as it is: the same picture and timing, as
    /// a `Video` buffer, with no pool of the caller's own behind it.
    #[test]
    fn a_hand_made_frame_is_carried_as_it_is() {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, 4, 2);
        frame.data_mut(0).fill(7);
        frame.set_pts(Some(42));
        set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));

        let MediaBuffer::Video(carried) = MediaBuffer::video(frame) else {
            panic!("a Video buffer");
        };
        assert_eq!((carried.width(), carried.height()), (4, 2));
        assert_eq!(carried.pts(), Some(42));
        assert_eq!(time_base(&carried), Some(ffmpeg::Rational::new(1, 30)));
        assert!(carried.data(0)[..4].iter().all(|&byte| byte == 7));
    }

    #[test]
    fn kind_reports_each_variant() {
        assert_eq!(
            MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty()).into()).kind(),
            "Packet"
        );
        assert_eq!(
            MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty()).into()).kind(),
            "Audio"
        );
    }

    /// A frame no one has described says nothing, rather than a unit of
    /// zero ticks that would read as one.
    #[test]
    fn a_new_frame_has_no_time_base_until_one_is_set() {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        assert_eq!(time_base(&frame), None);
        set_time_base(&mut frame, ffmpeg::Rational::new(1, 90_000));
        assert_eq!(time_base(&frame), Some(ffmpeg::Rational::new(1, 90_000)));
    }

    /// The two ways a frame's timing reaches another frame here both carry
    /// the unit with the count: FFmpeg's own property copy, which the
    /// elements that use it rely on, and `carry_timing` for the rest.
    #[test]
    fn copying_a_frames_timing_carries_its_time_base() {
        let mut source = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            32,
            ffmpeg::ChannelLayout::STEREO,
        );
        source.set_pts(Some(441));
        set_time_base(&mut source, ffmpeg::Rational::new(1, 44_100));

        let mut copied = ffmpeg::frame::Audio::empty();
        // SAFETY: both are live and distinct.
        unsafe { ffi::av_frame_copy_props(copied.as_mut_ptr(), source.as_ptr()) };
        assert_eq!(copied.pts(), Some(441));
        assert_eq!(time_base(&copied), Some(ffmpeg::Rational::new(1, 44_100)));

        let mut carried = ffmpeg::frame::Audio::empty();
        carry_timing(&mut carried, &source);
        assert_eq!(carried.pts(), Some(441));
        assert_eq!(time_base(&carried), Some(ffmpeg::Rational::new(1, 44_100)));
    }

    #[derive(Debug, PartialEq)]
    struct Detected(u32);

    #[derive(Debug, PartialEq)]
    struct Tracked(&'static str);

    #[test]
    fn metadata_holds_one_value_of_each_type() {
        let metadata = Metadata::new().with(Detected(1)).with(Tracked("car"));
        assert_eq!(metadata.get::<Detected>(), Some(&Detected(1)));
        assert_eq!(metadata.get::<Tracked>(), Some(&Tracked("car")));
        assert_eq!(metadata.get::<u8>(), None);
        assert_eq!(metadata.len(), 2);

        let metadata = metadata.with(Detected(2));
        assert_eq!(metadata.get::<Detected>(), Some(&Detected(2)), "replaced");
        assert_eq!(metadata.len(), 2);

        let mut metadata = metadata;
        assert!(metadata.remove::<Tracked>());
        assert!(!metadata.remove::<Tracked>());
        assert!(!metadata.contains::<Tracked>());
        assert!(format!("{metadata:?}").contains("Detected"), "{metadata:?}");
    }

    /// A buffer a `Tee` shared keeps what it carried in every other branch:
    /// putting metadata on one copy changes that copy alone, and the
    /// payload stays the one both share.
    #[test]
    fn metadata_put_on_a_copy_leaves_the_shared_buffer_as_it_was() {
        let shared = MediaBuffer::packet(ffmpeg::Packet::copy(&[7]));
        let copy = shared
            .clone()
            .with_metadata(Metadata::new().with(Detected(3)));
        assert!(shared.metadata().is_none());
        assert_eq!(
            copy.metadata()
                .and_then(|metadata| metadata.get::<Detected>()),
            Some(&Detected(3))
        );
        let (MediaBuffer::Packet(shared), MediaBuffer::Packet(copy)) = (&shared, &copy) else {
            unreachable!("both are packets");
        };
        assert!(Arc::ptr_eq(shared.payload(), copy.payload()), "one payload");
    }

    /// The wrapper reads as the payload it holds, which is what keeps
    /// `MediaBuffer::Video(frame) => frame.width()` working.
    #[test]
    fn a_wrapper_reads_as_its_payload() {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, 4, 2);
        frame.set_pts(Some(5));
        let buffer = MediaBuffer::video(frame).with_metadata(Metadata::new().with(Detected(1)));
        let MediaBuffer::Video(frame) = &buffer else {
            unreachable!("a picture");
        };
        assert_eq!(
            (frame.width(), frame.height(), frame.pts()),
            (4, 2, Some(5))
        );
        assert_eq!(buffer.kind(), "Video");
    }
}
