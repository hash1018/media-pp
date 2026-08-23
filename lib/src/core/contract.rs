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
//! such as feeding [`MediaBuffer::Packet`](crate::buffer::MediaBuffer)
//! into an encoder that only accepts decoded video, or handing a D3D11
//! texture to a CUDA filter.
//!
//! Most elements declare nothing and default to [`InputContract::Unknown`]
//! / [`OutputContract::Unknown`], which always links. Declaring a contract
//! is opt-in, so an element outside this crate keeps working untouched.

use std::fmt;

/// Which [`MediaBuffer`](crate::buffer::MediaBuffer) payloads a port deals in.
///
/// [`MediaBuffer::Eos`](crate::buffer::MediaBuffer::Eos) is deliberately
/// absent: every sink must accept EOS, so it is never part of what a
/// contract can rule out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    /// Encoded [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet).
    Packet,
    /// Decoded [`MediaBuffer::Video`](crate::buffer::MediaBuffer::Video).
    Video,
    /// Decoded [`MediaBuffer::Audio`](crate::buffer::MediaBuffer::Audio).
    Audio,
}

impl MediaKind {
    const fn bit(self) -> u8 {
        match self {
            MediaKind::Packet => 1 << 0,
            MediaKind::Video => 1 << 1,
            MediaKind::Audio => 1 << 2,
        }
    }

    const ALL: [MediaKind; 3] = [MediaKind::Packet, MediaKind::Video, MediaKind::Audio];
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MediaKind::Packet => "Packet",
            MediaKind::Video => "Video",
            MediaKind::Audio => "Audio",
        };
        f.write_str(name)
    }
}

/// A set of [`MediaKind`]s, as a port rarely deals in exactly one.
///
/// A set rather than a single kind because the two sides mean different
/// things: a producer's set is everything it *may* emit, a consumer's is
/// everything it *can* accept, and compatibility is the former being a
/// subset of the latter. A producer emitting `Video|Audio` into a
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

/// What one port deals in: which payloads, and where their memory lives.
///
/// `memory` is `None` when the question does not apply (a compressed
/// packet is always host memory) or when the element cannot promise an
/// answer at construction time. `None` on either side of a link means the
/// memory domain simply is not checked there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortContract {
    /// Payload variants this port deals in.
    pub media: MediaKindSet,
    /// Backend owning the memory, when that is both applicable and known.
    pub memory: Option<MemoryDomain>,
}

impl PortContract {
    /// A contract for exactly one [`MediaKind`], with no memory-domain claim.
    pub const fn of(kind: MediaKind) -> Self {
        Self {
            media: MediaKindSet::of(kind),
            memory: None,
        }
    }

    /// The same contract, narrowed to one [`MemoryDomain`].
    pub const fn in_memory(self, memory: MemoryDomain) -> Self {
        Self {
            memory: Some(memory),
            ..self
        }
    }

    /// Returns whether a producer emitting `produced` can feed a consumer
    /// accepting `self`.
    ///
    /// Every kind the producer may emit has to be accepted, and the memory
    /// domains must agree whenever both sides state one. An unstated domain
    /// on either side is not a mismatch — it is missing information, which
    /// this check always resolves in favor of allowing the link.
    pub fn accepts(&self, produced: &PortContract) -> bool {
        if !produced.media.is_subset_of(self.media) {
            return false;
        }
        match (self.memory, produced.memory) {
            (Some(accepted), Some(produced)) => accepted == produced,
            _ => true,
        }
    }
}

impl fmt::Display for PortContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.memory {
            Some(memory) => write!(f, "{} ({memory})", self.media),
            None => write!(f, "{}", self.media),
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
        let video_only = PortContract::of(MediaKind::Video);
        let both = PortContract {
            media: MediaKindSet::from_slice(&[MediaKind::Video, MediaKind::Audio]),
            memory: None,
        };

        assert!(video_only.accepts(&video_only));
        assert!(both.accepts(&video_only));
        // The audio half of `both` has nowhere to go in a video-only sink.
        assert!(!video_only.accepts(&both));
    }

    #[test]
    fn a_stated_memory_domain_must_match_but_an_absent_one_never_blocks() {
        let system = PortContract::of(MediaKind::Video).in_memory(MemoryDomain::System);
        let d3d11 = PortContract::of(MediaKind::Video).in_memory(MemoryDomain::D3d11);
        let unstated = PortContract::of(MediaKind::Video);

        assert!(system.accepts(&system));
        assert!(!system.accepts(&d3d11));
        assert!(unstated.accepts(&d3d11));
        assert!(d3d11.accepts(&unstated));
    }

    #[test]
    fn kinds_render_for_diagnostics() {
        assert_eq!(PortContract::of(MediaKind::Packet).to_string(), "Packet");
        assert_eq!(
            PortContract::of(MediaKind::Video)
                .in_memory(MemoryDomain::D3d11)
                .to_string(),
            "Video (D3D11)"
        );
        assert_eq!(
            MediaKindSet::from_slice(&[MediaKind::Audio, MediaKind::Packet]).to_string(),
            "Packet|Audio"
        );
    }
}
