//! What a port promises about the buffers passing through it.
//!
//! This is a deliberately conservative link check, not caps negotiation.
//! It answers one question — "can these two elements possibly be wired
//! together?" — from information every element already knows when it is
//! constructed, and it answers "I don't know" whenever it isn't sure.
//!
//! What it does *not* do, on purpose: it never picks a codec, never
//! inserts a converter, never renegotiates mid-stream, and never
//! reallocates a pool. Resolution, stride, color space, device identity and
//! a format's finer points stay where they already are — validated against
//! the real buffer when it arrives, by the element that is about to use it.
//! A contract only rules out wiring that could never have worked at all,
//! such as feeding encoded packets into an encoder that only accepts
//! decoded video, wiring a container's audio stream into a video decoder,
//! handing a D3D11 texture to a CUDA filter, or presenting BGRA through a
//! renderer that only draws NV12.
//!
//! That last one is a frame's [`PixelLayout`], and it is stated only where
//! construction already settles it — a scaler built to put out NV12, a
//! renderer that presents NV12 — never guessed. A layout that depends on
//! the stream is stated as every one it may turn out to be, and one that
//! follows the input as [`OutputContract::SameLayout`]. Claiming a
//! narrower set than an element really handles would refuse a pipeline
//! that works, which is worse than the runtime error the check exists to
//! bring forward; so where in doubt, an element says
//! [`PixelLayoutSet::ALL`], which [`PortContract::frame`] starts from.
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
    /// Timed text, as [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet).
    ///
    /// There is no decoded counterpart and there is not meant to be. A
    /// subtitle is already text by the time it is a packet, so nothing in
    /// this crate decodes one into a different shape — see
    /// [`crate::subtitle`], which builds the packets a muxer writes.
    SubtitlePacket,
}

impl MediaKind {
    /// The encoded kind a stream of `medium` carries, or `None` for a
    /// medium none of this crate's elements handle (data, attachments). A
    /// caller with no kind to state declares
    /// [`OutputContract::Unknown`]/[`InputContract::Unknown`] and leaves
    /// that pad to the runtime check, rather than guessing.
    pub fn packet_for(medium: ffmpeg::media::Type) -> Option<Self> {
        match medium {
            ffmpeg::media::Type::Video => Some(MediaKind::VideoPacket),
            ffmpeg::media::Type::Audio => Some(MediaKind::AudioPacket),
            ffmpeg::media::Type::Subtitle => Some(MediaKind::SubtitlePacket),
            _ => None,
        }
    }

    const fn bit(self) -> u8 {
        match self {
            MediaKind::VideoPacket => 1 << 0,
            MediaKind::AudioPacket => 1 << 1,
            MediaKind::VideoFrame => 1 << 2,
            MediaKind::AudioFrame => 1 << 3,
            MediaKind::SubtitlePacket => 1 << 4,
        }
    }

    const ALL: [MediaKind; 5] = [
        MediaKind::VideoPacket,
        MediaKind::AudioPacket,
        MediaKind::VideoFrame,
        MediaKind::AudioFrame,
        MediaKind::SubtitlePacket,
    ];
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MediaKind::VideoPacket => "VideoPacket",
            MediaKind::AudioPacket => "AudioPacket",
            MediaKind::VideoFrame => "VideoFrame",
            MediaKind::AudioFrame => "AudioFrame",
            MediaKind::SubtitlePacket => "SubtitlePacket",
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
/// distinction a single kind could not express.
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
    /// The one domain in this set, where it holds exactly one.
    const fn only(self) -> Option<MemoryDomain> {
        let mut index = 0;
        while index < MemoryDomain::ALL.len() {
            if self.0 == MemoryDomain::ALL[index].bit() {
                return Some(MemoryDomain::ALL[index]);
            }
            index += 1;
        }
        None
    }

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

/// How a decoded video frame's pixels are laid out — the part of its format
/// an element that reads them is built for.
///
/// Only the layouts this crate's GPU paths deal in are told apart: NV12,
/// what hardware decoders and encoders trade in; P010, the same at 10 bits;
/// and BGRA, what captures produce and shaders write. Every other format —
/// planar YUV from a software decoder, a camera's YUY2, audio — is
/// [`Self::Other`], which is as much as a link check needs to know about it:
/// that it is none of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelLayout {
    /// 8-bit 4:2:0, a luma plane and an interleaved chroma plane.
    Nv12,
    /// NV12's layout at 16 bits a sample, 10 of them used.
    P010,
    /// 8-bit packed BGRA.
    Bgra,
    /// Any other format.
    Other,
}

impl PixelLayout {
    const fn bit(self) -> u8 {
        match self {
            PixelLayout::Nv12 => 1 << 0,
            PixelLayout::P010 => 1 << 1,
            PixelLayout::Bgra => 1 << 2,
            PixelLayout::Other => 1 << 3,
        }
    }

    const ALL: [PixelLayout; 4] = [
        PixelLayout::Nv12,
        PixelLayout::P010,
        PixelLayout::Bgra,
        PixelLayout::Other,
    ];

    /// The layout a frame in `format` has — for a system-memory frame, or
    /// the `sw_format` of a hardware one.
    pub fn of(format: ffmpeg::format::Pixel) -> Self {
        match format {
            ffmpeg::format::Pixel::NV12 => PixelLayout::Nv12,
            ffmpeg::format::Pixel::P010LE => PixelLayout::P010,
            ffmpeg::format::Pixel::BGRA => PixelLayout::Bgra,
            _ => PixelLayout::Other,
        }
    }
}

impl fmt::Display for PixelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PixelLayout::Nv12 => "NV12",
            PixelLayout::P010 => "P010",
            PixelLayout::Bgra => "BGRA",
            PixelLayout::Other => "other layouts",
        };
        f.write_str(name)
    }
}

/// A set of [`PixelLayout`]s — every layout a port's frames may be in.
///
/// Compared as the other two sets are: what a producer may emit has to be
/// a subset of what a consumer can take. [`Self::ALL`] is what an element
/// says when it does not read pixels, or reads whatever it is given, and is
/// what [`PortContract::frame`] starts from — so an element that states
/// nothing about layout links as it always has.
///
/// A layout belongs here only where construction already settles it: a
/// scaler built to put out NV12, an encoder configured for BGRA, a renderer
/// that presents NV12. Where it depends on the stream — a decoder handing on
/// NV12 or P010 as the stream turns out — the set holds every one it may
/// be, and where it follows the input frame by frame, the port says
/// [`OutputContract::SameLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelLayoutSet(u8);

impl PixelLayoutSet {
    /// Every layout.
    pub const ALL: Self = Self::from_slice(&PixelLayout::ALL);
    /// NV12 alone.
    pub const NV12: Self = Self::of(PixelLayout::Nv12);
    /// BGRA alone.
    pub const BGRA: Self = Self::of(PixelLayout::Bgra);
    /// Any other format alone.
    pub const OTHER: Self = Self::of(PixelLayout::Other);
    /// 4:2:0 at 8 or 10 bits — what a hardware video processor or scaler
    /// reads as Y'CbCr.
    pub const YUV420: Self = Self::from_slice(&[PixelLayout::Nv12, PixelLayout::P010]);
    /// NV12 or BGRA — what most of this crate's GPU elements take.
    pub const NV12_OR_BGRA: Self = Self::from_slice(&[PixelLayout::Nv12, PixelLayout::Bgra]);
    /// Every layout a GPU scaler here reads.
    pub const GPU_SCALABLE: Self =
        Self::from_slice(&[PixelLayout::Nv12, PixelLayout::P010, PixelLayout::Bgra]);
    /// What a hardware decoder opened for a stream of `format` hands on: NV12
    /// for 8-bit 4:2:0, P010 for 10-bit, a surface in some other layout for
    /// anything else — and, where the stream does not say, nothing stated.
    /// Settled at construction by the parameters it was opened with; a stream
    /// that changes layout part way is left to the runtime checks, like a
    /// stream that changes size.
    pub fn decoded_from(format: ffmpeg::format::Pixel) -> Self {
        use ffmpeg::format::Pixel;
        match format {
            Pixel::None => Self::ALL,
            Pixel::YUV420P | Pixel::YUVJ420P | Pixel::NV12 => Self::NV12,
            Pixel::YUV420P10LE | Pixel::YUV420P10BE | Pixel::P010LE | Pixel::P010BE => {
                Self::of(PixelLayout::P010)
            }
            _ => Self::OTHER,
        }
    }

    /// A set holding exactly `layout`.
    pub const fn of(layout: PixelLayout) -> Self {
        Self(layout.bit())
    }

    /// A set holding every layout in `layouts`. Duplicates are harmless.
    pub const fn from_slice(layouts: &[PixelLayout]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < layouts.len() {
            bits |= layouts[index].bit();
            index += 1;
        }
        Self(bits)
    }

    /// Returns whether `layout` is in this set.
    pub const fn contains(self, layout: PixelLayout) -> bool {
        self.0 & layout.bit() != 0
    }

    /// Returns whether every layout in this set is also in `other`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }
}

impl fmt::Display for PixelLayoutSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::ALL {
            return f.write_str("any layout");
        }
        let mut first = true;
        for layout in PixelLayout::ALL {
            if !self.contains(layout) {
                continue;
            }
            if !first {
                f.write_str("|")?;
            }
            write!(f, "{layout}")?;
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
/// claim it is rather than as an omission. The layouts ride along with the
/// domain, [`PixelLayoutSet::ALL`] unless an element says otherwise.
///
/// The two never link to each other. That falls out of the shape, and it
/// matches [`MediaKind`]: no packet kind is a frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortContract {
    /// Encoded media — [`MediaKind::VideoPacket`]/[`MediaKind::AudioPacket`].
    Packets(MediaKindSet),
    /// Decoded frames — [`MediaKind::VideoFrame`]/[`MediaKind::AudioFrame`]
    /// — together with the backends their memory may live in and the
    /// layouts their pixels may be in.
    Frames(MediaKindSet, MemoryDomainSet, PixelLayoutSet),
}

impl PortContract {
    /// One encoded kind.
    pub const fn packet(kind: MediaKind) -> Self {
        Self::Packets(MediaKindSet::of(kind))
    }

    /// One decoded kind in one backend's memory, in any layout — see
    /// [`Self::with_layouts`] for saying which.
    pub const fn frame(kind: MediaKind, memory: MemoryDomain) -> Self {
        Self::Frames(
            MediaKindSet::of(kind),
            MemoryDomainSet::of(memory),
            PixelLayoutSet::ALL,
        )
    }

    /// One decoded kind, wherever it lives — for an element that forwards
    /// or counts frames without reading them.
    pub const fn any_frame(kind: MediaKind) -> Self {
        Self::Frames(
            MediaKindSet::of(kind),
            MemoryDomainSet::ALL,
            PixelLayoutSet::ALL,
        )
    }

    /// The same, its frames only ever in `layouts`. Encoded media has no
    /// layout, and is returned as it is.
    pub const fn with_layouts(self, layouts: PixelLayoutSet) -> Self {
        match self {
            Self::Frames(kinds, memory, _) => Self::Frames(kinds, memory, layouts),
            packets => packets,
        }
    }

    /// The layouts its frames may be in — every one, for encoded media,
    /// which has none to rule out.
    pub const fn layouts(&self) -> PixelLayoutSet {
        match self {
            Self::Frames(_, _, layouts) => *layouts,
            Self::Packets(_) => PixelLayoutSet::ALL,
        }
    }

    /// Returns whether a producer emitting `produced` can feed a consumer
    /// accepting `self`.
    ///
    /// Every kind the producer may emit has to be accepted, and so does
    /// every domain its frames may live in. So does every layout they may be
    /// in, where the producer states any: [`PixelLayoutSet::ALL`] from a
    /// producer is its saying nothing, and is not checked. Encoded media and decoded
    /// frames never satisfy each other.
    pub fn accepts(&self, produced: &PortContract) -> bool {
        match (self, produced) {
            (PortContract::Packets(accepted), PortContract::Packets(produced)) => {
                produced.is_subset_of(*accepted)
            }
            (
                PortContract::Frames(accepted, accepted_memory, accepted_layouts),
                PortContract::Frames(produced, produced_memory, produced_layouts),
            ) => {
                // A producer that states no layout is not held to one: it is
                // a claim it did not make, not every layout at once.
                produced.is_subset_of(*accepted)
                    && produced_memory.is_subset_of(*accepted_memory)
                    && (*produced_layouts == PixelLayoutSet::ALL
                        || produced_layouts.is_subset_of(*accepted_layouts))
            }
            _ => false,
        }
    }
}

impl fmt::Display for PortContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortContract::Packets(kinds) => write!(f, "{kinds}"),
            PortContract::Frames(kinds, memory, layouts) if *layouts == PixelLayoutSet::ALL => {
                write!(f, "{kinds} ({memory})")
            }
            PortContract::Frames(kinds, memory, layouts) => {
                write!(f, "{kinds} ({memory}, {layouts})")
            }
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

    /// This, in the layout that arrived — for an element that changes what a
    /// frame is or where it lives but not how its pixels are laid out: a
    /// resize-only scaler. The contract's own layouts are every one it can
    /// be given. Where nothing upstream says what it will be given, nothing
    /// downstream is checked against it — assuming every layout would refuse
    /// a consumer that takes only the one it will turn out to pass on.
    SameLayout(PortContract),

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
            OutputContract::SameLayout(contract) => {
                write!(f, "{contract}, in the layout it receives")
            }
            OutputContract::Passthrough => f.write_str("whatever it receives"),
            OutputContract::Unknown => f.write_str("unknown"),
        }
    }
}

/// What to put between a producer and a consumer that cannot be linked, as
/// one sentence naming the elements — or `None` where nothing in this crate
/// bridges the two, such as audio into a video filter.
///
/// Read off where the frames are and where they need to be: memory first,
/// since no element changes a layout across backends in one step, then the
/// layout within one backend. Every element named is one that exists for
/// exactly that crossing; where it takes only some layouts, the sentence
/// says what to put before it.
pub fn remedy(produced: &PortContract, accepted: &PortContract) -> Option<&'static str> {
    use MemoryDomain::{Cuda, D3d11, D3d12, System};
    use PixelLayout::{Bgra, Nv12, P010};

    let (
        produced_kinds,
        produced_memory,
        produced_layouts,
        accepted_kinds,
        accepted_memory,
        accepted_layouts,
    ) = match (produced, accepted) {
        (PortContract::Packets(kinds), PortContract::Frames(..)) => {
            return if kinds.contains(MediaKind::VideoPacket) {
                Some(
                    "decode it first: a SwDecoder, or a VideoDecodeBin onto the device the consumer reads",
                )
            } else {
                Some("decode it first: a SwDecoder")
            };
        }
        (PortContract::Frames(..), PortContract::Packets(_)) => {
            return Some("encode it first: an encoder turns frames into packets");
        }
        (PortContract::Packets(_), PortContract::Packets(_)) => return None,
        (
            PortContract::Frames(produced_kinds, produced_memory, produced_layouts),
            PortContract::Frames(accepted_kinds, accepted_memory, accepted_layouts),
        ) => (
            *produced_kinds,
            *produced_memory,
            *produced_layouts,
            *accepted_kinds,
            *accepted_memory,
            *accepted_layouts,
        ),
    };
    if !produced_kinds.is_subset_of(accepted_kinds) {
        return None;
    }

    if !produced_memory.is_subset_of(accepted_memory) {
        let from = produced_memory.only()?;
        let to = accepted_memory.only()?;
        return match (from, to) {
            (System, D3d11) => Some(
                "upload it: a D3d11Upload, which takes NV12 or BGRA — a SwScaler::to_format to one of those first where the frames are in another layout",
            ),
            (System, D3d12) => Some(
                "upload it: a D3d12Upload, which takes NV12 — a SwScaler::to_format to NV12 first where the frames are in another layout",
            ),
            (System, Cuda) => Some(
                "upload it: a CudaUpload built for NV12 or BGRA — a SwScaler::to_format to that layout first where the frames are in another",
            ),
            (D3d11, System) => Some(
                "download it: a D3d11Download, which reads BGRA — a D3d11Scaler with D3d11ScalerFormat::Bgra first where the frames are in another layout",
            ),
            (D3d12, System) => Some("download it: a D3d12Download, which reads NV12"),
            (Cuda, System) => Some("download it: a CudaDownload built for NV12 or BGRA"),
            _ => Some(
                "no element here moves frames between two GPU backends directly: download them to system memory and upload them again",
            ),
        };
    }

    // The same memory; the layout is what differs.
    let to_nv12 = accepted_layouts.contains(Nv12);
    let to_bgra = accepted_layouts.contains(Bgra);
    let from_p010 = produced_layouts.contains(P010);
    let from_bgra = produced_layouts.contains(Bgra);
    match accepted_memory.only()? {
        System => Some("convert it: a SwScaler::to_format to a layout the consumer takes"),
        D3d11 if to_nv12 => Some("convert it: a D3d11Scaler with D3d11ScalerFormat::Nv12"),
        D3d11 if to_bgra => Some(
            "convert it: a D3d11Scaler with D3d11ScalerFormat::Bgra — or a D3d11ToneMap, for PQ or HLG video",
        ),
        D3d11 => None,
        D3d12 => Some("nothing here converts a D3D12 frame's layout: decode or upload it as NV12"),
        Cuda if to_nv12 && from_bgra => {
            Some("convert it: a CudaConverter built for CudaFrameFormat::Nv12")
        }
        Cuda if to_nv12 && from_p010 => {
            Some("bring it to 8 bits: a CudaScaler built with_format CudaFrameFormat::Nv12")
        }
        Cuda if to_bgra && from_p010 => Some(
            "convert it: a CudaScaler with_format CudaFrameFormat::Nv12, then a CudaConverter built for CudaFrameFormat::Bgra — or the CudaConverter alone, for PQ or HLG video",
        ),
        Cuda if to_bgra => Some("convert it: a CudaConverter built for CudaFrameFormat::Bgra"),
        Cuda => None,
    }
}

/// `remedy`'s answer as the tail of an error sentence — `"; convert it: …"` —
/// or nothing where it has none.
pub(crate) fn remedy_suffix(produced: &PortContract, accepted: &PortContract) -> String {
    remedy(produced, accepted)
        .map(|remedy| format!("; {remedy}"))
        .unwrap_or_default()
}

/// What [`check_link`] makes of a pad feeding a sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkCheck {
    /// Every buffer the pad may put out, the sink takes.
    Fits,
    /// The sink cannot take what the pad puts out — the link a pipeline
    /// refuses before anything runs.
    Refused {
        /// What the pad puts out, as far as it says.
        produced: PortContract,
        /// What the sink takes.
        accepted: PortContract,
    },
    /// One side says too little to tell. A pipeline links it, and leaves it
    /// to the frames themselves: a pad that passes on what it is given, or
    /// one that states nothing, or a sink that states nothing, or a pad
    /// passing on a layout it is not yet known to be given into a sink that
    /// takes only some.
    Unknown,
}

impl LinkCheck {
    /// What to put between the two, where this is [`Self::Refused`] and
    /// something in this crate bridges them — see [`remedy`].
    pub fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::Refused { produced, accepted } => remedy(produced, accepted),
            _ => None,
        }
    }

    /// Whether this is [`Self::Refused`].
    pub fn is_refused(&self) -> bool {
        matches!(self, Self::Refused { .. })
    }
}

impl fmt::Display for LinkCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkCheck::Fits => f.write_str("fits"),
            LinkCheck::Refused { produced, accepted } => {
                write!(f, "produces {produced}, which takes only {accepted}")?;
                f.write_str(&remedy_suffix(produced, accepted))
            }
            LinkCheck::Unknown => f.write_str("cannot be told before it runs"),
        }
    }
}

/// Whether `producer` can feed `consumer` — [`check_link`] asked of the two
/// elements themselves, with no pad or trait to reach for:
///
/// ```ignore
/// let fits = check_elements(&mut decoder, &renderer);
/// if fits.is_refused() {
///     println!("{fits}"); // what does not fit, and what goes between
/// }
/// ```
///
/// It asks about the producer's first output, which for every element
/// with one is the only one. A source with several — a demuxer, a `Tee` —
/// is asked about through [`check_link`] and the pad in question, and one
/// with none answers [`LinkCheck::Unknown`].
pub fn check_elements(
    producer: &mut (impl crate::element::Source + ?Sized),
    consumer: &(impl crate::element::Sink + ?Sized),
) -> LinkCheck {
    match producer.src_pads().first() {
        Some(pad) => check_link(&pad.contract(), &consumer.input_contract()),
        None => LinkCheck::Unknown,
    }
}

/// Whether a pad declaring `produced` can feed a sink declaring
/// `accepted` — asked of the two before they are linked, and by the same
/// rules a pipeline applies when they are:
///
/// ```ignore
/// let produced = decoder.src_pads()[0].contract();
/// if check_link(&produced, &renderer.input_contract()).is_refused() {
///     // put the converter the renderer needs between them
/// }
/// ```
///
/// This is one link. Where something that passes buffers through — a
/// `Queue`, a `Pacer` — sits between two elements, ask about the element
/// that produces what it passes on, since it passes that on unchanged; the
/// pipeline's own check follows the whole branch.
pub fn check_link(produced: &OutputContract, accepted: &InputContract) -> LinkCheck {
    let accepted = match accepted {
        InputContract::Fixed(accepted) => *accepted,
        InputContract::Any => return LinkCheck::Fits,
        InputContract::Unknown => return LinkCheck::Unknown,
    };
    let (produced, layout_known) = match produced {
        OutputContract::Fixed(produced) => (*produced, true),
        // Its kind and memory are settled; its layout is whatever reaches it.
        OutputContract::SameLayout(produced) => (produced.with_layouts(PixelLayoutSet::ALL), false),
        OutputContract::Passthrough | OutputContract::Unknown => return LinkCheck::Unknown,
    };
    if !accepted.accepts(&produced) {
        return LinkCheck::Refused { produced, accepted };
    }
    if !layout_known && accepted.layouts() != PixelLayoutSet::ALL {
        return LinkCheck::Unknown;
    }
    LinkCheck::Fits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_producers_kinds_must_all_be_accepted() {
        let video_only = PortContract::any_frame(MediaKind::VideoFrame);
        let both = PortContract::Frames(
            MediaKindSet::FRAMES,
            MemoryDomainSet::ALL,
            PixelLayoutSet::ALL,
        );

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
        let frames = PortContract::Frames(
            MediaKindSet::FRAMES,
            MemoryDomainSet::ALL,
            PixelLayoutSet::ALL,
        );

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
        assert_eq!(
            MediaKind::packet_for(ffmpeg::media::Type::Subtitle),
            Some(MediaKind::SubtitlePacket)
        );
        // Data streams and attachments still are not modelled, and a caller
        // with no kind to state leaves that pad Unknown rather than guessing.
        assert_eq!(MediaKind::packet_for(ffmpeg::media::Type::Data), None);
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

    /// The case layouts exist for: a CUDA decoder line that may put out
    /// BGRA cannot feed a renderer that presents NV12, though both are CUDA
    /// video frames — while one that puts out NV12 can, and a port that
    /// states no layout takes either.
    #[test]
    fn every_layout_a_producer_may_emit_must_be_accepted() {
        let cuda = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda);
        let renderer = cuda.with_layouts(PixelLayoutSet::of(PixelLayout::Nv12));
        let nv12 = cuda.with_layouts(PixelLayoutSet::of(PixelLayout::Nv12));
        let bgra = cuda.with_layouts(PixelLayoutSet::of(PixelLayout::Bgra));
        let either = cuda.with_layouts(PixelLayoutSet::from_slice(&[
            PixelLayout::Nv12,
            PixelLayout::P010,
        ]));

        assert!(renderer.accepts(&nv12));
        assert!(!renderer.accepts(&bgra));
        assert!(!renderer.accepts(&either), "P010 has nowhere to go");
        assert!(cuda.accepts(&bgra), "no layout stated takes any");
        assert!(
            renderer.accepts(&cuda),
            "a producer stating no layout is not held to one"
        );
    }

    #[test]
    fn layouts_render_for_diagnostics_only_when_narrowed() {
        let d3d11 = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11);
        assert_eq!(d3d11.to_string(), "VideoFrame (D3D11)");
        assert_eq!(
            d3d11
                .with_layouts(PixelLayoutSet::from_slice(&[
                    PixelLayout::Nv12,
                    PixelLayout::Bgra
                ]))
                .to_string(),
            "VideoFrame (D3D11, NV12|BGRA)"
        );
        assert_eq!(
            PortContract::packet(MediaKind::VideoPacket)
                .with_layouts(PixelLayoutSet::of(PixelLayout::Nv12)),
            PortContract::packet(MediaKind::VideoPacket),
            "encoded media has no layout to narrow"
        );
    }

    #[test]
    fn a_format_maps_to_its_layout() {
        use ffmpeg::format::Pixel;
        assert_eq!(PixelLayout::of(Pixel::NV12), PixelLayout::Nv12);
        assert_eq!(PixelLayout::of(Pixel::P010LE), PixelLayout::P010);
        assert_eq!(PixelLayout::of(Pixel::BGRA), PixelLayout::Bgra);
        assert_eq!(PixelLayout::of(Pixel::YUV420P), PixelLayout::Other);
    }

    /// The answer a caller gets before linking is the one the pipeline would
    /// give, for each shape a pad and a sink can declare.
    #[test]
    fn check_link_answers_as_a_pipeline_would() {
        let cuda = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda);
        let d3d11 = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11);
        let nv12_renderer = InputContract::Fixed(cuda.with_layouts(PixelLayoutSet::NV12));
        let bgra = OutputContract::Fixed(cuda.with_layouts(PixelLayoutSet::BGRA));
        let nv12 = OutputContract::Fixed(cuda.with_layouts(PixelLayoutSet::NV12));

        assert_eq!(check_link(&nv12, &nv12_renderer), LinkCheck::Fits);
        let refused = check_link(&bgra, &nv12_renderer);
        assert_eq!(
            refused,
            LinkCheck::Refused {
                produced: cuda.with_layouts(PixelLayoutSet::BGRA),
                accepted: cuda.with_layouts(PixelLayoutSet::NV12),
            }
        );
        assert!(refused.is_refused());
        assert_eq!(
            refused.to_string(),
            "produces VideoFrame (CUDA, BGRA), which takes only VideoFrame (CUDA, NV12); \
             convert it: a CudaConverter built for CudaFrameFormat::Nv12"
        );
        assert!(
            check_link(&OutputContract::Fixed(d3d11), &nv12_renderer).is_refused(),
            "the wrong memory is refused whatever the layout"
        );

        // A sink that takes anything, and ones that say nothing.
        assert_eq!(check_link(&bgra, &InputContract::Any), LinkCheck::Fits);
        assert_eq!(
            check_link(&bgra, &InputContract::Unknown),
            LinkCheck::Unknown
        );
        assert_eq!(
            check_link(&OutputContract::Passthrough, &nv12_renderer),
            LinkCheck::Unknown
        );
        assert_eq!(
            check_link(&OutputContract::Unknown, &nv12_renderer),
            LinkCheck::Unknown
        );

        // A pad passing on the layout it is given: its memory is checked,
        // its layout cannot be told where the sink takes only some.
        let resize = OutputContract::SameLayout(cuda.with_layouts(PixelLayoutSet::GPU_SCALABLE));
        assert_eq!(check_link(&resize, &nv12_renderer), LinkCheck::Unknown);
        assert_eq!(
            check_link(&resize, &InputContract::Fixed(cuda)),
            LinkCheck::Fits
        );
        assert!(check_link(&OutputContract::SameLayout(d3d11), &nv12_renderer).is_refused());
    }

    /// A refusal says what goes between, naming the element that makes that
    /// one crossing — and says nothing where no element here makes it.
    #[test]
    fn a_refusal_names_what_goes_between() {
        let frame = |memory, layout: PixelLayout| {
            PortContract::frame(MediaKind::VideoFrame, memory)
                .with_layouts(PixelLayoutSet::of(layout))
        };
        let says = |produced, accepted, element: &str| {
            let remedy = remedy(&produced, &accepted).unwrap_or("");
            assert!(
                remedy.contains(element),
                "{produced} into {accepted}: {remedy:?} names no {element}"
            );
        };
        use MemoryDomain::{Cuda, D3d11, System};
        use PixelLayout::{Bgra, Nv12, P010};

        says(frame(System, Nv12), frame(D3d11, Nv12), "D3d11Upload");
        says(frame(System, Bgra), frame(Cuda, Bgra), "CudaUpload");
        says(frame(D3d11, Bgra), frame(System, Bgra), "D3d11Download");
        says(frame(Cuda, Bgra), frame(Cuda, Nv12), "CudaConverter");
        says(frame(Cuda, P010), frame(Cuda, Nv12), "CudaScaler");
        says(frame(Cuda, Nv12), frame(Cuda, Bgra), "CudaConverter");
        says(frame(D3d11, Nv12), frame(D3d11, Bgra), "D3d11Scaler");
        says(frame(D3d11, P010), frame(D3d11, Nv12), "D3d11Scaler");
        says(frame(System, Nv12), frame(System, Bgra), "SwScaler");
        says(frame(D3d11, Nv12), frame(Cuda, Nv12), "system memory");
        says(
            PortContract::packet(MediaKind::VideoPacket),
            frame(System, Nv12),
            "decode",
        );
        assert_eq!(
            remedy(
                &PortContract::frame(MediaKind::AudioFrame, System),
                &frame(System, Nv12)
            ),
            None,
            "nothing turns audio into video"
        );
    }
}
