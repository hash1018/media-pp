//! Fragmented MP4 through FFmpeg's `mp4` muxer, written into memory and cut
//! where the caller says: the init section once, then one `moof` + `mdat`
//! fragment each time it is asked for what was written since the last.
//!
//! The muxer writes no edit list, so two runs of the same tracks have the
//! same init section; without one it counts each track's decode times from
//! its first packet's, which each fragment's `tfdt` is given back here, so
//! what is served is on the timeline the packets were written on.
//!
//! The muxer writes through an I/O context of our own that collects bytes
//! rather than writing a file, and is told to write a fragment only when
//! asked (`frag_custom`). Its track list (`moov`) waits for the first
//! fragment (`delay_moov`), so a stream that says its parameter sets only in
//! band — an RTSP camera's — is described from its first keyframe; the init
//! section is what comes before the first `moof`.

use std::ptr;

use ffmpeg_next::{self as ffmpeg, Rational, ffi, packet::Mut as _};

use crate::elements::StreamInfo;

const ENOMEM: i32 = 12;

/// How much the I/O context holds before it hands bytes to `collect`.
const BUFFER: usize = 64 * 1024;

/// What the muxer writes, collected in order.
#[derive(Default)]
struct Collected {
    bytes: Vec<u8>,
}

unsafe extern "C" fn collect(opaque: *mut std::ffi::c_void, buf: *const u8, size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    // SAFETY: `opaque` is the `Collected` the fragmenter boxed for this
    // context and keeps until after the context is freed; `buf` holds
    // `size` bytes for the length of this call.
    unsafe {
        let collected = &mut *opaque.cast::<Collected>();
        collected
            .bytes
            .extend_from_slice(std::slice::from_raw_parts(buf, size as usize));
    }
    size
}

/// One run's fragmented MP4, every track in each fragment.
pub(super) struct Fragmenter {
    ctx: *mut ffi::AVFormatContext,
    collected: Box<Collected>,
    /// The unit each track's packets are read in where they say none.
    units: Vec<Rational>,
    /// The unit the muxer counts each track in.
    out: Vec<Rational>,
    /// The init section, once the first fragment has been cut off it and
    /// until it is taken.
    init: Option<Vec<u8>>,
    /// Whether the init section has been written and cut off.
    described: bool,
    /// Each track's first decode time, in its unit in the muxer: what the
    /// muxer took off its fragments' decode times.
    first: Vec<Option<i64>>,
}

// SAFETY: the muxer context, its I/O and what it collects into are this
// value's alone, used through `&mut self` only; FFmpeg keeps no thread-local
// state for them.
unsafe impl Send for Fragmenter {}

impl Fragmenter {
    /// A fragmenter of `tracks`, in their order, its header written.
    pub(super) fn new(tracks: &[StreamInfo]) -> Result<Self, ffmpeg::Error> {
        let mut collected = Box::<Collected>::default();
        // SAFETY: each pointer made here is checked before use, and freed
        // in `Drop` — the context's I/O with it — or right here on failure;
        // `collected` outlives the context since `Drop` frees the context
        // first.
        unsafe {
            let mut ctx = ptr::null_mut();
            let ret = ffi::avformat_alloc_output_context2(
                &mut ctx,
                ptr::null(),
                c"mp4".as_ptr(),
                ptr::null(),
            );
            if ret < 0 || ctx.is_null() {
                return Err(ffmpeg::Error::from(ret.min(-1)));
            }
            let buffer = ffi::av_malloc(BUFFER).cast::<u8>();
            let pb = if buffer.is_null() {
                ptr::null_mut()
            } else {
                ffi::avio_alloc_context(
                    buffer,
                    BUFFER as i32,
                    1,
                    ptr::from_mut(collected.as_mut()).cast(),
                    None,
                    Some(collect),
                    None,
                )
            };
            if pb.is_null() {
                ffi::av_free(buffer.cast());
                ffi::avformat_free_context(ctx);
                return Err(ffmpeg::Error::Other { errno: ENOMEM });
            }
            (*ctx).pb = pb;
            (*ctx).flags |= ffi::AVFMT_FLAG_CUSTOM_IO;
            let mut this = Self {
                ctx,
                collected,
                units: tracks.iter().map(StreamInfo::time_base).collect(),
                out: Vec::new(),
                init: None,
                described: false,
                first: vec![None; tracks.len()],
            };
            for track in tracks {
                let stream = ffi::avformat_new_stream(ctx, ptr::null());
                if stream.is_null() {
                    return Err(ffmpeg::Error::Other { errno: ENOMEM });
                }
                let ret =
                    ffi::avcodec_parameters_copy((*stream).codecpar, track.parameters().as_ptr());
                if ret < 0 {
                    return Err(ffmpeg::Error::from(ret));
                }
                // The tag the source container gave it means nothing here;
                // the muxer picks its own.
                (*(*stream).codecpar).codec_tag = 0;
                (*stream).time_base = track.time_base().into();
            }
            let mut options = ffmpeg::Dictionary::new();
            options.set(
                "movflags",
                "frag_custom+empty_moov+delay_moov+default_base_moof",
            );
            // A run that begins past zero — one going on where the last
            // ended — would get an edit list, and an init section other than
            // the last run's for the same tracks.
            options.set("use_editlist", "0");
            let mut raw = options.disown();
            let ret = ffi::avformat_write_header(ctx, &mut raw);
            ffi::av_dict_free(&mut raw);
            if ret < 0 {
                return Err(ffmpeg::Error::from(ret));
            }
            this.out = (0..tracks.len())
                .map(|index| (*(*(*ctx).streams.add(index))).time_base.into())
                .collect();
            Ok(this)
        }
    }

    /// The unit track `index`'s timestamps are written in.
    pub(super) fn unit_out(&self, index: usize) -> Rational {
        self.out[index]
    }

    /// The unit `packet` of track `index` counts in: its own, or the
    /// track's where it says none.
    pub(super) fn unit_in(&self, index: usize, packet: &ffmpeg::Packet) -> Rational {
        let unit = packet.time_base();
        if unit.numerator() > 0 && unit.denominator() > 0 {
            unit
        } else {
            self.units[index]
        }
    }

    /// Writes `packet`, its timestamps and duration already in
    /// [`Self::unit_out`] of track `index`.
    pub(super) fn write(
        &mut self,
        index: usize,
        mut packet: ffmpeg::Packet,
    ) -> Result<(), ffmpeg::Error> {
        packet.set_stream(index);
        packet.set_position(-1);
        if self.first[index].is_none() {
            self.first[index] = packet.dts().or(packet.pts());
        }
        // SAFETY: the context is open, and the packet ours, for the call.
        let ret = unsafe { ffi::av_write_frame(self.ctx, packet.as_mut_ptr()) };
        if ret < 0 {
            return Err(ffmpeg::Error::from(ret));
        }
        Ok(())
    }

    /// A fragment of everything written since the last one: empty where
    /// nothing was.
    pub(super) fn cut(&mut self) -> Result<Vec<u8>, ffmpeg::Error> {
        self.flush()?;
        if self.described {
            let mut fragment = std::mem::take(&mut self.collected.bytes);
            self.restore_decode_times(&mut fragment);
            return Ok(fragment);
        }
        // The muxer's first flush writes the track list it delayed and
        // keeps the samples for the next: asked again, it writes them.
        if first_box(&self.collected.bytes, b"moof").is_none() {
            self.flush()?;
        }
        let mut bytes = std::mem::take(&mut self.collected.bytes);
        let Some(at) = first_box(&bytes, b"moof") else {
            // Nothing written yet: what is held stays for the next.
            self.collected.bytes = bytes;
            return Ok(Vec::new());
        };
        let mut fragment = bytes.split_off(at);
        self.init = Some(bytes);
        self.described = true;
        self.restore_decode_times(&mut fragment);
        Ok(fragment)
    }

    /// Adds back to each track's `tfdt` in `fragment` the first decode
    /// time the muxer took off it.
    fn restore_decode_times(&self, fragment: &mut [u8]) {
        for (moof, size) in boxes(fragment, 0, fragment.len(), b"moof") {
            for (traf, size) in boxes(fragment, moof + 8, moof + size, b"traf") {
                let end = traf + size;
                let track = boxes(fragment, traf + 8, end, b"tfhd")
                    .first()
                    .and_then(|&(tfhd, _)| read_u32(fragment, tfhd + 12))
                    .and_then(|id| usize::try_from(id).ok()?.checked_sub(1));
                let Some(first) = track.and_then(|track| self.first.get(track).copied().flatten())
                else {
                    continue;
                };
                for (tfdt, _) in boxes(fragment, traf + 8, end, b"tfdt") {
                    let version = fragment[tfdt + 8];
                    let at = tfdt + 12;
                    if version == 1 && at + 8 <= fragment.len() {
                        let was = u64::from_be_bytes(fragment[at..at + 8].try_into().unwrap());
                        let now = was.wrapping_add_signed(first);
                        fragment[at..at + 8].copy_from_slice(&now.to_be_bytes());
                    } else if at + 4 <= fragment.len() {
                        let was = u32::from_be_bytes(fragment[at..at + 4].try_into().unwrap());
                        let now = i64::from(was) + first;
                        if let Ok(now) = u32::try_from(now) {
                            fragment[at..at + 4].copy_from_slice(&now.to_be_bytes());
                        }
                    }
                }
            }
        }
    }

    /// Asks the muxer for the fragment it holds, and the I/O for what it
    /// buffered.
    fn flush(&mut self) -> Result<(), ffmpeg::Error> {
        // SAFETY: the context and its I/O are open; a null packet asks the
        // muxer for the fragment it holds.
        let ret = unsafe {
            let ret = ffi::av_write_frame(self.ctx, ptr::null_mut());
            ffi::avio_flush((*self.ctx).pb);
            ret
        };
        if ret < 0 {
            return Err(ffmpeg::Error::from(ret));
        }
        Ok(())
    }

    /// The init section — `ftyp` and `moov` — once the first fragment has
    /// been cut.
    pub(super) fn take_init(&mut self) -> Option<Vec<u8>> {
        self.init.take()
    }
}

impl Drop for Fragmenter {
    fn drop(&mut self) {
        // SAFETY: the context and its I/O, made in `new`; the buffer is the
        // I/O's own, which it may have replaced.
        unsafe {
            let mut pb = (*self.ctx).pb;
            (*self.ctx).pb = ptr::null_mut();
            ffi::avformat_free_context(self.ctx);
            if !pb.is_null() {
                ffi::av_freep(ptr::addr_of_mut!((*pb).buffer).cast());
                ffi::avio_context_free(&mut pb);
            }
        }
    }
}

/// The boxes of type `kind` among those from `from` to `to` in `bytes`:
/// where each begins, and its size.
fn boxes(bytes: &[u8], from: usize, to: usize, kind: &[u8; 4]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut at = from;
    while at + 8 <= to.min(bytes.len()) {
        let Some(size) = read_u32(bytes, at).map(|size| size as usize) else {
            break;
        };
        if size < 8 || at + size > to {
            break;
        }
        if &bytes[at + 4..at + 8] == kind {
            found.push((at, size));
        }
        at += size;
    }
    found
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// Where the first top-level box of type `kind` begins in `bytes`, walking
/// the boxes from the start.
fn first_box(bytes: &[u8], kind: &[u8; 4]) -> Option<usize> {
    let mut at = 0;
    while at + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[at..at + 4].try_into().ok()?) as usize;
        if &bytes[at + 4..at + 8] == kind {
            return Some(at);
        }
        let size = match size {
            // A 64-bit size follows the type.
            1 if at + 16 <= bytes.len() => {
                u64::from_be_bytes(bytes[at + 8..at + 16].try_into().ok()?) as usize
            }
            // To the end, or not a box.
            0 | 1 => return None,
            size => size,
        };
        if size < 8 {
            return None;
        }
        at = at.checked_add(size)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], body: usize) -> Vec<u8> {
        let mut bytes = ((8 + body) as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(kind);
        bytes.extend(std::iter::repeat_n(0, body));
        bytes
    }

    #[test]
    fn the_first_box_of_a_kind_is_found_among_the_top_level_ones() {
        let mut bytes = boxed(b"ftyp", 16);
        bytes.extend(boxed(b"moov", 100));
        let moof = bytes.len();
        bytes.extend(boxed(b"moof", 20));
        bytes.extend(boxed(b"mdat", 300));
        assert_eq!(first_box(&bytes, b"moof"), Some(moof));
        assert_eq!(first_box(&bytes, b"ftyp"), Some(0));
        assert_eq!(first_box(&bytes, b"sidx"), None);
        assert_eq!(first_box(&bytes[..10], b"moof"), None, "cut short");
    }
}
