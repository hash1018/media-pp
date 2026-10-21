//! RTP packetization through FFmpeg's `rtp` muxer, and the SDP describing
//! what it sends.
//!
//! The muxer knows how every codec it carries is cut into RTP payloads —
//! H.264 and H.265 into fragmentation units, AAC with its AU headers, Opus,
//! VP8, VP9 and the rest — and how each is described in SDP. It writes
//! through an I/O context of our own, which collects what it writes rather
//! than sending it: one RTP or RTCP packet per write, since it flushes after
//! each one and its buffer is no larger than a packet.

use std::ffi::{CStr, CString};
use std::ptr;

use ffmpeg_next::{self as ffmpeg, Rational, ffi, packet::Mut as _};

use crate::elements::StreamInfo;

/// What an RTP packet is let grow to, header included: a UDP datagram that
/// crosses an Ethernet link whole.
pub(crate) const MTU: usize = 1400;

/// The payload type track `index` is sent as: a dynamic one each.
pub(crate) fn payload_type(index: usize) -> u8 {
    96 + (index % 32) as u8
}

/// Whether FFmpeg's RTP muxer can carry `track`: the codecs its
/// packetizers cut — `avformat_query_codec` cannot say, since the muxer
/// keeps no list of them where it would look.
pub(crate) fn carries(track: &StreamInfo) -> bool {
    use ffmpeg::codec::Id;
    matches!(
        track.parameters.id(),
        Id::H264
            | Id::HEVC
            | Id::VP8
            | Id::VP9
            | Id::MPEG4
            | Id::MJPEG
            | Id::H263
            | Id::MPEG1VIDEO
            | Id::MPEG2VIDEO
            | Id::THEORA
            | Id::AAC
            | Id::OPUS
            | Id::MP2
            | Id::MP3
            | Id::VORBIS
            | Id::SPEEX
            | Id::PCM_MULAW
            | Id::PCM_ALAW
            | Id::PCM_S16BE
            | Id::ADPCM_G722
            | Id::ADPCM_G726
            | Id::AMR_NB
            | Id::AMR_WB
    )
}

fn rtp_format() -> *const ffi::AVOutputFormat {
    // SAFETY: a lookup by name; a null answer is checked by the callers.
    unsafe { ffi::av_guess_format(c"rtp".as_ptr(), ptr::null(), ptr::null()) }
}

/// An `rtp` muxer context of one stream, `track`, sending it as `payload`:
/// nothing written yet, no I/O.
///
/// # Safety
///
/// The caller frees what it answers with `avformat_free_context`.
unsafe fn context(
    track: &StreamInfo,
    payload: u8,
) -> Result<*mut ffi::AVFormatContext, ffmpeg::Error> {
    // SAFETY: every pointer is checked as it is made; on failure what was
    // made is freed before returning, and on success the caller frees it.
    unsafe {
        let mut ctx = ptr::null_mut();
        let ret =
            ffi::avformat_alloc_output_context2(&mut ctx, rtp_format(), ptr::null(), ptr::null());
        if ret < 0 || ctx.is_null() {
            return Err(ffmpeg::Error::from(ret.min(-1)));
        }
        let stream = ffi::avformat_new_stream(ctx, ptr::null());
        if stream.is_null() {
            ffi::avformat_free_context(ctx);
            return Err(ffmpeg::Error::Other { errno: ENOMEM });
        }
        let ret = ffi::avcodec_parameters_copy((*stream).codecpar, track.parameters.as_ptr());
        if ret < 0 {
            ffi::avformat_free_context(ctx);
            return Err(ffmpeg::Error::from(ret));
        }
        (*stream).time_base = track.time_base.into();
        // Read both by the muxer as it writes and by `av_sdp_create`, so
        // what is described is what is sent.
        let ret = ffi::av_opt_set_int(
            (*ctx).priv_data,
            c"payload_type".as_ptr(),
            i64::from(payload),
            0,
        );
        if ret < 0 {
            ffi::avformat_free_context(ctx);
            return Err(ffmpeg::Error::from(ret));
        }
        Ok(ctx)
    }
}

/// Out of memory, as FFmpeg reports it.
const ENOMEM: i32 = 12;

/// The SDP describing `tracks` as a server sends them, each its own media
/// section with a control URL of `streamid=<index>`, named `name`.
pub(crate) fn sdp(name: &str, tracks: &[StreamInfo]) -> Result<String, ffmpeg::Error> {
    let mut contexts = Vec::with_capacity(tracks.len());
    let result = (|| {
        for (index, track) in tracks.iter().enumerate() {
            // SAFETY: freed below, whatever happens.
            contexts.push(unsafe { context(track, payload_type(index))? });
        }
        let title = CString::new(name.replace('\0', "")).unwrap_or_default();
        // SAFETY: the first context, just made; the dictionary copies the
        // strings.
        unsafe {
            ffi::av_dict_set(
                &mut (*contexts[0]).metadata,
                c"title".as_ptr(),
                title.as_ptr(),
                0,
            );
        }
        let mut buffer = vec![0u8; 16 * 1024];
        // SAFETY: as many contexts as said, each with its one stream; the
        // buffer is as long as said.
        let ret = unsafe {
            ffi::av_sdp_create(
                contexts.as_mut_ptr(),
                contexts.len() as i32,
                buffer.as_mut_ptr().cast(),
                buffer.len() as i32,
            )
        };
        if ret < 0 {
            return Err(ffmpeg::Error::from(ret));
        }
        let text = CStr::from_bytes_until_nul(&buffer)
            .map_err(|_| ffmpeg::Error::InvalidData)?
            .to_string_lossy()
            .into_owned();
        Ok(with_connection_line(&text))
    })();
    for ctx in contexts {
        // SAFETY: made by `context`, never handed out.
        unsafe { ffi::avformat_free_context(ctx) };
    }
    result
}

/// `sdp` with a session-level `c=` line where it has none: FFmpeg leaves it
/// out for a description of several streams with nowhere to send them,
/// and some players refuse a description without one.
fn with_connection_line(sdp: &str) -> String {
    if sdp.lines().any(|line| line.starts_with("c=")) {
        return sdp.to_owned();
    }
    let mut out = String::with_capacity(sdp.len() + 24);
    for line in sdp.split_inclusive('\n') {
        out.push_str(line);
        if line.starts_with("s=") {
            out.push_str("c=IN IP4 0.0.0.0\r\n");
        }
    }
    out
}

/// What the muxer writes, collected: each write one RTP or RTCP packet.
#[derive(Default)]
struct Collected {
    packets: Vec<Vec<u8>>,
}

unsafe extern "C" fn collect(opaque: *mut std::ffi::c_void, buf: *const u8, size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    // SAFETY: `opaque` is the `Collected` the packetizer boxed for this
    // context and keeps until after the context is freed; `buf` holds
    // `size` bytes for the length of this call.
    unsafe {
        let collected = &mut *opaque.cast::<Collected>();
        collected
            .packets
            .push(std::slice::from_raw_parts(buf, size as usize).to_vec());
    }
    size
}

/// Cuts one track's packets into RTP packets, with the RTCP sender reports
/// the muxer adds every few seconds among them — one viewer's, since each
/// starts its own sequence.
pub(crate) struct Packetizer {
    ctx: *mut ffi::AVFormatContext,
    collected: Box<Collected>,
    /// The unit what it is handed counts in.
    unit: Rational,
    /// The unit the muxer counts in, its clock rate's.
    clock: Rational,
}

// SAFETY: the muxer context, its I/O and what it collects into are this
// value's alone, used through `&mut self` only, never shared between
// threads; FFmpeg keeps no thread-local state for them.
unsafe impl Send for Packetizer {}

impl Packetizer {
    /// A packetizer of `track`, sent as payload type `payload`, whose
    /// sender reports put timestamp zero at `start_us` — microseconds since
    /// the Unix epoch, the same for every track of a viewer, so the player
    /// lines the tracks up.
    pub(crate) fn new(
        track: &StreamInfo,
        payload: u8,
        start_us: i64,
    ) -> Result<Self, ffmpeg::Error> {
        let mut collected = Box::<Collected>::default();
        // SAFETY: each pointer made here is checked before use and freed in
        // `Drop`, or right here on failure; `collected` outlives the context
        // since `Drop` frees the context first.
        unsafe {
            let ctx = context(track, payload)?;
            let buffer = ffi::av_malloc(MTU).cast::<u8>();
            let pb = if buffer.is_null() {
                ptr::null_mut()
            } else {
                ffi::avio_alloc_context(
                    buffer,
                    MTU as i32,
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
            (*pb).max_packet_size = MTU as i32;
            (*ctx).pb = pb;
            (*ctx).flags |= ffi::AVFMT_FLAG_CUSTOM_IO;
            (*ctx).packet_size = MTU as u32;
            (*ctx).start_time_realtime = start_us;
            let mut this = Self {
                ctx,
                collected,
                unit: track.time_base,
                clock: Rational(1, 90_000),
            };
            let ret = ffi::avformat_write_header(ctx, ptr::null_mut());
            if ret < 0 {
                return Err(ffmpeg::Error::from(ret));
            }
            this.clock = (*(*(*ctx).streams)).time_base.into();
            // Nothing is sent before the first packet; drop anything the
            // header wrote.
            this.collected.packets.clear();
            Ok(this)
        }
    }

    /// `packet`'s RTP packets, and a sender report before them where one is
    /// due. Its timestamps count in the track's unit and must not go back.
    pub(crate) fn packetize(
        &mut self,
        packet: &ffmpeg::Packet,
    ) -> Result<Vec<Vec<u8>>, ffmpeg::Error> {
        if packet.pts().is_none() {
            return Ok(Vec::new());
        }
        let mut packet = packet.clone();
        packet.set_stream(0);
        packet.rescale_ts(self.unit, self.clock);
        // SAFETY: the context is open, and the packet ours, for the call.
        let ret = unsafe { ffi::av_write_frame(self.ctx, packet.as_mut_ptr()) };
        let packets = std::mem::take(&mut self.collected.packets);
        if ret < 0 {
            return Err(ffmpeg::Error::from(ret));
        }
        Ok(packets)
    }
}

impl Drop for Packetizer {
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

/// Whether `packet`, as a packetizer writes it, is RTCP rather than RTP:
/// its payload type byte one of RTCP's (200 to 204).
pub(crate) fn is_rtcp(packet: &[u8]) -> bool {
    packet.len() >= 2 && (200..=204).contains(&packet[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(
        medium: ffmpeg::media::Type,
        codec: ffmpeg::codec::Id,
    ) -> ffmpeg::codec::Parameters {
        let mut parameters = ffmpeg::codec::Parameters::new();
        // SAFETY: parameters this test owns alone; plain fields.
        unsafe {
            (*parameters.as_mut_ptr()).codec_type = medium.into();
            (*parameters.as_mut_ptr()).codec_id = codec.into();
            if medium == ffmpeg::media::Type::Audio {
                (*parameters.as_mut_ptr()).sample_rate = 48_000;
                ffi::av_channel_layout_default(&mut (*parameters.as_mut_ptr()).ch_layout, 2);
            }
        }
        parameters
    }

    fn track(
        index: usize,
        medium: ffmpeg::media::Type,
        codec: ffmpeg::codec::Id,
        unit: Rational,
    ) -> StreamInfo {
        StreamInfo {
            index,
            kind: medium,
            parameters: parameters(medium, codec),
            time_base: unit,
            frame_rate: None,
        }
    }

    #[test]
    fn a_description_has_a_section_per_track_with_its_control() {
        let tracks = [
            track(
                0,
                ffmpeg::media::Type::Video,
                ffmpeg::codec::Id::H264,
                Rational(1, 90_000),
            ),
            track(
                1,
                ffmpeg::media::Type::Audio,
                ffmpeg::codec::Id::OPUS,
                Rational(1, 48_000),
            ),
        ];
        let sdp = sdp("cam", &tracks).unwrap();
        assert!(sdp.contains("s=cam"), "{sdp}");
        assert!(sdp.contains("c=IN IP4"), "{sdp}");
        assert!(sdp.contains("m=video 0 RTP/AVP 96"), "{sdp}");
        assert!(sdp.contains("a=rtpmap:96 H264/90000"), "{sdp}");
        assert!(sdp.contains("m=audio 0 RTP/AVP 97"), "{sdp}");
        assert!(sdp.contains("a=control:streamid=0"), "{sdp}");
        assert!(sdp.contains("a=control:streamid=1"), "{sdp}");
    }

    #[test]
    fn a_large_picture_is_cut_into_packets_no_larger_than_the_mtu() {
        let video = track(
            0,
            ffmpeg::media::Type::Video,
            ffmpeg::codec::Id::H264,
            Rational(1, 1_000),
        );
        let mut packetizer = Packetizer::new(&video, 96, 0).unwrap();
        // One IDR slice NAL unit of 10 000 bytes, in Annex B.
        let mut data = vec![0, 0, 0, 1, 0x65];
        data.extend(std::iter::repeat_n(0xAB, 10_000));
        let mut packet = ffmpeg::Packet::copy(&data);
        packet.set_pts(Some(40));
        packet.set_dts(Some(40));
        packet.set_flags(ffmpeg::packet::Flags::KEY);
        let sent = packetizer.packetize(&packet).unwrap();
        let rtp: Vec<_> = sent.iter().filter(|p| !is_rtcp(p)).collect();
        assert!(
            sent.iter().any(|p| is_rtcp(p)),
            "a sender report goes with the first packet"
        );
        assert!(rtp.len() >= 8, "{} packets", rtp.len());
        assert!(rtp.iter().all(|p| p.len() <= MTU));
        for (n, p) in rtp.iter().enumerate() {
            assert_eq!(p[0] >> 6, 2, "RTP version 2");
            assert_eq!(p[1] & 0x7f, 96, "the payload type asked for");
            assert_eq!(
                p[1] & 0x80 != 0,
                n == rtp.len() - 1,
                "the marker on the last packet only"
            );
        }
        let timestamp = |p: &[u8]| u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        let first = timestamp(rtp[0]);
        assert!(
            rtp.iter().all(|p| timestamp(p) == first),
            "one picture, one timestamp"
        );

        let mut later = packet.clone();
        later.set_pts(Some(80));
        later.set_dts(Some(80));
        let next = packetizer.packetize(&later).unwrap();
        let next = next.iter().find(|p| !is_rtcp(p)).unwrap();
        assert_eq!(
            timestamp(next).wrapping_sub(first),
            40 * 90,
            "40 ms later on a 90 kHz clock"
        );
        let sequence = |p: &[u8]| u16::from_be_bytes([p[2], p[3]]);
        assert_eq!(
            sequence(next),
            sequence(rtp.last().unwrap()).wrapping_add(1)
        );
    }
}
