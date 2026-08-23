//! What a port promises about the buffers passing through it.
//!
//! This is a deliberately conservative link check, not caps negotiation.
//! It answers one question — "can these two elements possibly be wired
//! together?" — from information every element already knows when it is
//! constructed, and it answers "I don't know" whenever it isn't sure.
//!
//! What it does *not* do, on purpose: it never picks a codec, never
//! inserts a converter, never renegotiates mid-stream, and never
//! reallocates a pool. Pixel format, resolution, stride, color space, and
//! device identity stay where they already are — validated against the
//! real buffer when it arrives, by the element that is about to use it.
//! A contract only rules out wiring that could never have worked at all,
//! such as feeding encoded packets into an encoder that only accepts
//! decoded video, wiring a container's audio stream into a video decoder,
//! or handing a D3D11 texture to a CUDA filter.
//!
//! Declaring a contract is opt-in: both sides default to
//! [`InputContract::Unknown`] / [`OutputContract::Unknown`], which always
//! links, so an element outside this crate keeps working untouched. This
//! crate's own elements do declare one, with the deliberate exceptions of
//! [`AppSource`](crate::elements::AppSource) — only the application knows
//! what it will push — and a demuxer pad for a medium not modelled here.

use std::fmt;

use ffmpeg_next as ffmpeg;

/// Which [`MediaBuffer`](crate::buffer::MediaBuffer) payloads a port deals
/// in, split by medium as well as by encoding.
///
/// The medium is part of the kind because
/// [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet) alone does
/// not carry it: a demuxer's audio and video pads emit the same variant,
/// so without this split, wiring a container's audio stream into a video
/// decoder is a link the check cannot see. Every element that deals in
/// packets does know its own medium when it is constructed — from the
/// stream parameters it was opened with, or from being an audio encoder
/// rather than a video one — so the distinction costs nothing to state.
///
/// [`MediaBuffer::Eos`](crate::buffer::MediaBuffer::Eos) is deliberately
/// absent: every sink must accept EOS, so it is never part of what a
/// contract can rule out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    /// Encoded video, as [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet).
    VideoPacket,
    /// Encoded audio, as [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet).
    AudioPacket,
    /// Decoded [`MediaBuffer::Video`](crate::buffer::MediaBuffer::Video).
    VideoFrame,
    /// Decoded [`MediaBuffer::Audio`](crate::buffer::MediaBuffer::Audio).
    AudioFrame,
}

impl MediaKind {
    /// The encoded kind a stream of `medium` carries, or `None` for a
    /// medium none of this crate's elements handle (subtitles, data). A
    /// caller with no kind to state declares
    /// [`OutputContract::Unknown`]/[`InputContract::Unknown`] and leaves
    /// that pad to the runtime check, rather than guessing.
    pub fn packet_for(medium: ffmpeg::media::Type) -> Option<Self> {
        match medium {
            ffmpeg::media::Type::Video => Some(MediaKind::VideoPacket),
            ffmpeg::media::Type::Audio => Some(MediaKind::AudioPacket),
            _ => None,
        }
    }

    const fn bit(self) -> u8 {
        match self {
            MediaKind::VideoPacket => 1 << 0,
            MediaKind::AudioPacket => 1 << 1,
            MediaKind::VideoFrame => 1 << 2,
            MediaKind::AudioFrame => 1 << 3,
        }
    }

    const ALL: [MediaKind; 4] = [
        MediaKind::VideoPacket,
        MediaKind::AudioPacket,
        MediaKind::VideoFrame,
        MediaKind::AudioFrame,
    ];
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MediaKind::VideoPacket => "VideoPacket",
            MediaKind::AudioPacket => "AudioPacket",
            MediaKind::VideoFrame => "VideoFrame",
            MediaKind::AudioFrame => "AudioFrame",
        };
        f.write_str(name)
    }
}

/// A set of [`MediaKind`]s, as a port rarely deals in exactly one.
///
/// A set rather than a single kind because the two sides mean different
/// things: a producer's set is everything it *may* emit, a consumer's is
/// everything it *can* accept, and compatibility is the former being a
/// subset of the latter. A demuxer feeding a muxer may emit either
/// encoded kind, while a video decoder accepts only one of them — a
/// distinction a single kind could not express.|Audio` into a
/// video-only sink is a real mismatch for half its buffers, which a single
/// kind could not express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaKindSet(u8);

impl MediaKindSet {
    /// A set holding exactly `kind`.
    pub const fn of(kind: MediaKind) -> Self {
        Self(kind.bit())
    }

    /// A set holding every kind in `kinds`. Duplicates are harmless.
    pub const fn from_slice(kinds: &[MediaKind]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < kinds.len() {
            bits |= kinds[index].bit();
            index += 1;
        }
        Self(bits)
    }

    /// Both encoded kinds — what a muxer, a packet counter, or any other
    /// element that interleaves or forwards encoded media deals in.
    pub const PACKETS: Self = Self::from_slice(&[MediaKind::VideoPacket, MediaKind::AudioPacket]);

    /// Both decoded kinds — what an element that handles frames without
    /// caring which medium they are deals in.
    pub const FRAMES: Self = Self::from_slice(&[MediaKind::VideoFrame, MediaKind::AudioFrame]);

    /// Returns whether `kind` is in this set.
    pub const fn contains(self, kind: MediaKind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// Returns whether every kind in this set is also in `other` — the
    /// producer-into-consumer direction the link check asks about.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }
}

impl fmt::Display for MediaKindSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for kind in MediaKind::ALL {
            if !self.contains(kind) {
                continue;
            }
            if !first {
                f.write_str("|")?;
            }
            write!(f, "{kind}")?;
            first = false;
        }
        if first {
            f.write_str("nothing")
        } else {
            Ok(())
        }
    }
}

/// Where a decoded frame's pixels actually live.
///
/// [`MediaBuffer::Video`](crate::buffer::MediaBuffer::Video) is one variant
/// covering system memory, CUDA device memory, and D3D11/D3D12 textures
/// alike, so the buffer type alone cannot tell a CPU scaler that it was
/// handed a GPU texture. This is the part of the contract that catches
/// that — it says which backend owns the memory, and nothing about the
/// format, size, or specific device within that backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Host memory: an ordinary FFmpeg frame with CPU-readable planes.
    System,
    /// CUDA device memory bound to a CUDA context.
    Cuda,
    /// A D3D11 texture owned by an `ID3D11Device`.
    D3d11,
    /// A D3D12 resource owned by an `ID3D12Device`.
    D3d12,
}

impl MemoryDomain {
    const fn bit(self) -> u8 {
        match self {
            MemoryDomain::System => 1 << 0,
            MemoryDomain::Cuda => 1 << 1,
            MemoryDomain::D3d11 => 1 << 2,
            MemoryDomain::D3d12 => 1 << 3,
        }
    }

    const ALL: [MemoryDomain; 4] = [
        MemoryDomain::System,
        MemoryDomain::Cuda,
        MemoryDomain::D3d11,
        MemoryDomain::D3d12,
    ];
}

impl fmt::Display for MemoryDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MemoryDomain::System => "System",
            MemoryDomain::Cuda => "CUDA",
            MemoryDomain::D3d11 => "D3D11",
            MemoryDomain::D3d12 => "D3D12",
        };
        f.write_str(name)
    }
}

/// A set of [`MemoryDomain`]s — everywhere a port's frames may live.
///
/// A producer's set is what it may emit and a consumer's is what it can
/// take, so compatibility is the former being a subset of the latter,
/// exactly as for [`MediaKindSet`]. An element that genuinely does not
/// care — one that never reads the pixels — declares [`Self::ALL`], which
/// is a claim rather than an omission: there is no "unstated" domain to
/// forget, because [`PortContract::Frames`] has nowhere to leave it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryDomainSet(u8);

impl MemoryDomainSet {
    /// Every backend — for an element that passes frames through without
    /// reading them.
    pub const ALL: Self = Self::from_slice(&MemoryDomain::ALL);

    /// A set holding exactly `domain`.
    pub const fn of(domain: MemoryDomain) -> Self {
        Self(domain.bit())
    }

    /// A set holding every domain in `domains`. Duplicates are harmless.
    pub const fn from_slice(domains: &[MemoryDomain]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < domains.len() {
            bits |= domains[index].bit();
            index += 1;
        }
        Self(bits)
    }

    /// Returns whether `domain` is in this set.
    pub const fn contains(self, domain: MemoryDomain) -> bool {
        self.0 & domain.bit() != 0
    }

    /// Returns whether every domain in this set is also in `other` — the
    /// producer-into-consumer direction the link check asks about.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }
}

impl fmt::Display for MemoryDomainSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::ALL {
            return f.write_str("any memory");
        }
        let mut first = true;
        for domain in MemoryDomain::ALL {
            if !self.contains(domain) {
                continue;
            }
            if !first {
                f.write_str("|")?;
            }
            write!(f, "{domain}")?;
            first = false;
        }
        if first {
            f.write_str("nothing")
        } else {
            Ok(())
        }
    }
}

/// What one port deals in.
///
/// Split by encoding rather than carrying an optional domain, because the
/// two halves ask different questions. Encoded media is always host
/// memory, so [`Self::Packets`] has nowhere to put a domain and nowhere to
/// forget one. Decoded frames always live somewhere specific, so
/// [`Self::Frames`] always states it — an element that genuinely takes any
/// backend says [`MemoryDomainSet::ALL`], which reads as the deliberate
/// claim it is rather than as an omission.
///
/// The two never link to each other. That falls out of the shape, and it
/// matches [`MediaKind`]: no packet kind is a frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortContract {
    /// Encoded media — [`MediaKind::VideoPacket`]/[`MediaKind::AudioPacket`].
    Packets(MediaKindSet),
    /// Decoded frames — [`MediaKind::VideoFrame`]/[`MediaKind::AudioFrame`]
    /// — together with the backends their memory may live in.
    Frames(MediaKindSet, MemoryDomainSet),
}

impl PortContract {
    /// One encoded kind.
    pub const fn packet(kind: MediaKind) -> Self {
        Self::Packets(MediaKindSet::of(kind))
    }

    /// One decoded kind in one backend's memory.
    pub const fn frame(kind: MediaKind, memory: MemoryDomain) -> Self {
        Self::Frames(MediaKindSet::of(kind), MemoryDomainSet::of(memory))
    }

    /// One decoded kind, wherever it lives — for an element that forwards
    /// or counts frames without reading them.
    pub const fn any_frame(kind: MediaKind) -> Self {
        Self::Frames(MediaKindSet::of(kind), MemoryDomainSet::ALL)
    }

    /// Returns whether a producer emitting `produced` can feed a consumer
    /// accepting `self`.
    ///
    /// Every kind the producer may emit has to be accepted, and so does
    /// every domain its frames may live in. Encoded media and decoded
    /// frames never satisfy each other.
    pub fn accepts(&self, produced: &PortContract) -> bool {
        match (self, produced) {
            (PortContract::Packets(accepted), PortContract::Packets(produced)) => {
                produced.is_subset_of(*accepted)
            }
            (
                PortContract::Frames(accepted, accepted_memory),
                PortContract::Frames(produced, produced_memory),
            ) => produced.is_subset_of(*accepted) && produced_memory.is_subset_of(*accepted_memory),
            _ => false,
        }
    }
}

impl fmt::Display for PortContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortContract::Packets(kinds) => write!(f, "{kinds}"),
            PortContract::Frames(kinds, memory) => write!(f, "{kinds} ({memory})"),
        }
    }
}

/// What a [`Sink`](crate::element::Sink) can be fed.
///
/// [`Any`](Self::Any) and [`Unknown`](Self::Unknown) both link to
/// anything, but they mean opposite things and differ in what happens
/// *downstream* of the element — see [`OutputContract::Passthrough`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputContract {
    /// This element accepts exactly this and nothing else.
    Fixed(PortContract),

    /// A guarantee that every [`MediaKind`] is handled. A
    /// [`Queue`](crate::queue::Queue) forwards whatever it is given; an
    /// [`AppSink`](crate::elements::AppSink) hands every buffer to its
    /// closure. Note the scope: this promises the *element* passes each
    /// kind along, not that the application's own closure will succeed
    /// with it. A closure that only understands packets still returns its
    /// own error, which is outside what a link check can or should know.
    Any,

    /// No claim. Links to anything, and stops the check from continuing
    /// past this element, because nothing here knows what comes out.
    Unknown,
}

impl fmt::Display for InputContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputContract::Fixed(contract) => write!(f, "{contract}"),
            InputContract::Any => f.write_str("anything"),
            InputContract::Unknown => f.write_str("unknown"),
        }
    }
}

/// What a [`SrcPad`](crate::pad::SrcPad) emits.
///
/// Declared per pad rather than per element because
/// [`Tee`](crate::elements::Tee) and
/// [`FileDemuxer`](crate::elements::FileDemuxer) own several, and nothing
/// requires them to agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputContract {
    /// This pad emits exactly this and nothing else.
    Fixed(PortContract),

    /// Whatever arrived on the input leaves here unchanged — a
    /// [`Queue`](crate::queue::Queue), a [`Tee`](crate::elements::Tee), a
    /// [`Pacer`](crate::elements::Pacer). This is what keeps a check alive
    /// across the middle of a pipeline: the upstream contract is carried
    /// through, so a decoder's output still meets an encoder's input two
    /// queues later.
    Passthrough,

    /// No claim. The check goes dark from here on.
    Unknown,
}

impl fmt::Display for OutputContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputContract::Fixed(contract) => write!(f, "{contract}"),
            OutputContract::Passthrough => f.write_str("whatever it receives"),
            OutputContract::Unknown => f.write_str("unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_producers_kinds_must_all_be_accepted() {
        let video_only = PortContract::any_frame(MediaKind::VideoFrame);
        let both = PortContract::Frames(MediaKindSet::FRAMES, MemoryDomainSet::ALL);

        assert!(video_only.accepts(&video_only));
        assert!(both.accepts(&video_only));
        // The audio half of `both` has nowhere to go in a video-only sink.
        assert!(!video_only.accepts(&both));
    }

    /// The split the medium exists for: both are `MediaBuffer::Packet`, so
    /// nothing else separates a container's audio stream from its video one.
    #[test]
    fn encoded_audio_and_encoded_video_are_different_kinds() {
        let video = PortContract::packet(MediaKind::VideoPacket);
        let audio = PortContract::packet(MediaKind::AudioPacket);

        assert!(!video.accepts(&audio));
        assert!(!audio.accepts(&video));
        // A muxer takes either, and a demuxer pad of either kind fits it.
        let muxer = PortContract::Packets(MediaKindSet::PACKETS);
        assert!(muxer.accepts(&video));
        assert!(muxer.accepts(&audio));
    }

    /// Encoded and decoded ports never satisfy each other, and the shape
    /// is what says so — there is no domain to compare across them.
    #[test]
    fn packets_and_frames_never_link() {
        let packets = PortContract::Packets(MediaKindSet::PACKETS);
        let frames = PortContract::Frames(MediaKindSet::FRAMES, MemoryDomainSet::ALL);

        assert!(!packets.accepts(&frames));
        assert!(!frames.accepts(&packets));
    }

    #[test]
    fn a_medium_maps_to_its_encoded_kind_or_to_nothing() {
        assert_eq!(
            MediaKind::packet_for(ffmpeg::media::Type::Video),
            Some(MediaKind::VideoPacket)
        );
        assert_eq!(
            MediaKind::packet_for(ffmpeg::media::Type::Audio),
            Some(MediaKind::AudioPacket)
        );
        // Subtitles and data streams are not modelled, and a caller with
        // no kind to state leaves that pad Unknown rather than guessing.
        assert_eq!(MediaKind::packet_for(ffmpeg::media::Type::Subtitle), None);
    }

    /// Domains are a set on both sides now, so "takes any backend" is a
    /// claim an element makes rather than a field it left empty.
    #[test]
    fn every_domain_a_producer_may_emit_must_be_accepted() {
        let system = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System);
        let d3d11 = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11);
        let anywhere = PortContract::any_frame(MediaKind::VideoFrame);

        assert!(system.accepts(&system));
        assert!(!system.accepts(&d3d11));
        // A pass-through element takes either; neither takes everything.
        assert!(anywhere.accepts(&d3d11));
        assert!(anywhere.accepts(&system));
        assert!(!d3d11.accepts(&anywhere));
    }

    #[test]
    fn kinds_render_for_diagnostics() {
        assert_eq!(
            PortContract::packet(MediaKind::VideoPacket).to_string(),
            "VideoPacket"
        );
        assert_eq!(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11).to_string(),
            "VideoFrame (D3D11)"
        );
        assert_eq!(
            PortContract::any_frame(MediaKind::VideoFrame).to_string(),
            "VideoFrame (any memory)"
        );
        assert_eq!(MediaKindSet::PACKETS.to_string(), "VideoPacket|AudioPacket");
    }
}
