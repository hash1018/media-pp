use std::{fmt, ptr};

use ffmpeg_next as ffmpeg;
use str0m::format::{Codec, CodecSpec};

use crate::error::Result;

use super::command::WebRtcError;

/// What an inbound WebRTC track carries, confirmed from its first packets:
/// what a [`WebRtcSource`](crate::elements::WebRtcSource) waits for before
/// it says what each of its outputs carries, from
/// [`Self::codec_parameters`].
///
/// SDP exposes several possible codecs, while this value identifies the one
/// the remote sender actually used. Audio payload metadata is sufficient as
/// soon as the first payload arrives. H.264 is different: construction waits
/// until both SPS and PPS have arrived, so [`Self::codec_parameters`] can
/// describe the stream without borrowing the remote encoder's configuration.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct WebRtcStreamInfo {
    codec_spec: CodecSpec,
    h264: Option<H264Config>,
}

impl fmt::Debug for WebRtcStreamInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebRtcStreamInfo")
            .field("codec_spec", &self.codec_spec)
            .field("video_dimensions", &self.video_dimensions())
            .field("codec_parameters_ready", &self.codec_parameters_ready())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct H264Config {
    sps: Vec<u8>,
    pps: Vec<u8>,
    width: u32,
    height: u32,
}

impl WebRtcStreamInfo {
    /// Returns dimensions parsed from received codec configuration, currently
    /// `Some` for H.264 information.
    pub fn video_dimensions(&self) -> Option<(u32, u32)> {
        self.h264.as_ref().map(|h264| (h264.width, h264.height))
    }

    /// Returns the RTP timestamp time base derived from the codec clock rate.
    pub fn time_base(&self) -> Result<ffmpeg::Rational> {
        let clock_rate = clock_rate_i32(self.codec_spec)?;
        Ok(ffmpeg::Rational::new(1, clock_rate))
    }

    /// Builds FFmpeg codec parameters for this depayloaded stream.
    ///
    /// The result describes the compressed stream independently of its next
    /// consumer and can be passed to a decoder or container muxer. H.264
    /// includes received SPS/PPS and dimensions, while Opus includes the
    /// negotiated channel layout and `OpusHead`. Whether a particular
    /// container accepts the codec remains the muxer's responsibility.
    pub fn codec_parameters(&self) -> Result<ffmpeg::codec::Parameters> {
        let (medium, id) = ffmpeg_codec(self.codec_spec.codec).ok_or(
            WebRtcError::UnsupportedCodecParameters(self.codec_spec.codec),
        )?;
        let mut parameters = base_parameters(self.codec_spec, medium, id)?;
        match self.codec_spec.codec {
            Codec::H264 => {
                let h264 = self
                    .h264
                    .as_ref()
                    .ok_or(WebRtcError::H264ParameterSetsNotReceived)?;
                set_h264_parameters(&mut parameters, h264)?;
            }
            Codec::Opus => {
                let channels = audio_channels(self.codec_spec)?;
                set_extradata(
                    &mut parameters,
                    &crate::elements::rtsp::sdp::opus_head(channels),
                )?;
            }
            _ => {}
        }
        Ok(parameters)
    }

    /// What a source of this track says it carries: its
    /// [`Self::codec_parameters`], timed in its [`Self::time_base`]. Fails
    /// where either of those would.
    pub(crate) fn stream_info(&self) -> Result<crate::elements::StreamInfo> {
        Ok(crate::elements::StreamInfo::new(
            self.codec_parameters()?,
            self.time_base()?,
        ))
    }

    fn codec_parameters_ready(&self) -> bool {
        match self.codec_spec.codec {
            Codec::H264 => self.h264.is_some(),
            Codec::Opus => self
                .codec_spec
                .channels
                .is_some_and(|channels| (1..=2).contains(&channels)),
            codec => ffmpeg_codec(codec).is_some(),
        }
    }
}

impl From<CodecSpec> for WebRtcStreamInfo {
    fn from(codec_spec: CodecSpec) -> Self {
        Self {
            codec_spec,
            h264: None,
        }
    }
}

/// Per-track payload observer owned by the `WebRtcPeer` thread.
pub(super) struct StreamInfoProbe {
    codec_spec: Option<CodecSpec>,
    h264_sps: Option<Vec<u8>>,
    h264_pps: Option<Vec<u8>>,
}

impl StreamInfoProbe {
    pub(super) fn new() -> Self {
        Self {
            codec_spec: None,
            h264_sps: None,
            h264_pps: None,
        }
    }

    /// Returns once the observed codec has enough information for the public
    /// stream-info contract. H.264 may span several frames.
    pub(super) fn observe(
        &mut self,
        codec_spec: CodecSpec,
        payload: &[u8],
    ) -> Option<WebRtcStreamInfo> {
        if self.codec_spec != Some(codec_spec) {
            self.codec_spec = Some(codec_spec);
            self.h264_sps = None;
            self.h264_pps = None;
        }
        if codec_spec.codec != Codec::H264 {
            return Some(codec_spec.into());
        }

        for nalu in annex_b_nalus(payload) {
            match nalu.first().map(|byte| byte & 0x1f) {
                Some(7) => self.h264_sps = Some(nalu.to_vec()),
                Some(8) => self.h264_pps = Some(nalu.to_vec()),
                _ => {}
            }
        }
        let (Some(sps), Some(pps)) = (&self.h264_sps, &self.h264_pps) else {
            return None;
        };
        let (width, height) = parse_h264_dimensions(sps)?;
        Some(WebRtcStreamInfo {
            codec_spec,
            h264: Some(H264Config {
                sps: sps.clone(),
                pps: pps.clone(),
                width,
                height,
            }),
        })
    }
}

fn base_parameters(
    codec_spec: CodecSpec,
    medium: ffmpeg::media::Type,
    id: ffmpeg::codec::Id,
) -> Result<ffmpeg::codec::Parameters> {
    let audio = medium == ffmpeg::media::Type::Audio;
    let sample_rate = audio.then(|| clock_rate_i32(codec_spec)).transpose()?;
    let channels = if audio {
        match codec_spec.channels {
            Some(0) => return Err(invalid_channel_count(codec_spec).into()),
            channels => channels,
        }
    } else {
        None
    };
    let mut parameters = ffmpeg::codec::Parameters::new();

    // SAFETY: `parameters` is exclusively owned. These are plain fields, and
    // `av_channel_layout_default` initializes its owned channel layout.
    unsafe {
        let raw = parameters.as_mut_ptr();
        (*raw).codec_type = medium.into();
        (*raw).codec_id = id.into();
        if let Some(sample_rate) = sample_rate {
            (*raw).sample_rate = sample_rate;
        }
        if let Some(channels) = channels {
            ffmpeg::ffi::av_channel_layout_default(&mut (*raw).ch_layout, i32::from(channels));
        }
    }
    Ok(parameters)
}

fn set_h264_parameters(
    parameters: &mut ffmpeg::codec::Parameters,
    h264: &H264Config,
) -> Result<()> {
    // Keep FFmpeg-facing configuration in Annex-B form. Besides being valid
    // decoder extradata, this tells the MP4 muxer that incoming access units
    // use Annex-B too, so it converts both the header to avcC and packet NALUs
    // to length-prefixed samples while writing the container.
    let extradata = annex_b_h264_extradata(&h264.sps, &h264.pps)?;
    // SAFETY: `parameters` is exclusively borrowed and SPS validation
    // guarantees the indexed profile bytes exist.
    unsafe {
        let raw = parameters.as_mut_ptr();
        (*raw).width = h264.width as i32;
        (*raw).height = h264.height as i32;
        (*raw).profile = i32::from(h264.sps[1]);
        (*raw).level = i32::from(h264.sps[3]);
    }
    set_extradata(parameters, &extradata)
}

fn set_extradata(parameters: &mut ffmpeg::codec::Parameters, bytes: &[u8]) -> Result<()> {
    let size = i32::try_from(bytes.len())
        .map_err(|_| WebRtcError::CodecConfigurationTooLarge { size: bytes.len() })?;
    let padded = bytes
        .len()
        .checked_add(ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize)
        .ok_or(WebRtcError::CodecConfigurationTooLarge { size: bytes.len() })?;

    // SAFETY: FFmpeg owns and frees extradata allocated with `av_mallocz`.
    // The required padding stays zero and the copy fits the allocation.
    unsafe {
        let allocation = ffmpeg::ffi::av_mallocz(padded) as *mut u8;
        if allocation.is_null() {
            return Err(WebRtcError::CodecParametersAllocationFailed { size: padded }.into());
        }
        ptr::copy_nonoverlapping(bytes.as_ptr(), allocation, bytes.len());
        let raw = parameters.as_mut_ptr();
        (*raw).extradata = allocation;
        (*raw).extradata_size = size;
    }
    Ok(())
}

fn annex_b_h264_extradata(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>> {
    if sps.len() < 4 || sps.first().map(|byte| byte & 0x1f) != Some(7) {
        return Err(WebRtcError::InvalidH264ParameterSet("SPS").into());
    }
    if pps.is_empty() || pps.first().map(|byte| byte & 0x1f) != Some(8) {
        return Err(WebRtcError::InvalidH264ParameterSet("PPS").into());
    }
    let mut result = Vec::with_capacity(8 + sps.len() + pps.len());
    result.extend_from_slice(&[0, 0, 0, 1]);
    result.extend_from_slice(sps);
    result.extend_from_slice(&[0, 0, 0, 1]);
    result.extend_from_slice(pps);
    Ok(result)
}

fn clock_rate_i32(codec_spec: CodecSpec) -> std::result::Result<i32, WebRtcError> {
    let clock_rate = codec_spec.clock_rate.get();
    i32::try_from(clock_rate).map_err(|_| WebRtcError::InvalidStreamClockRate {
        codec: codec_spec.codec,
        clock_rate,
    })
}

fn audio_channels(codec_spec: CodecSpec) -> std::result::Result<u8, WebRtcError> {
    match codec_spec.channels {
        // Opus mapping family 0 (the only mapping WebRTC negotiates here) is
        // defined for mono and stereo only.
        Some(channels @ 1..=2) => Ok(channels),
        _ => Err(invalid_channel_count(codec_spec)),
    }
}

fn invalid_channel_count(codec_spec: CodecSpec) -> WebRtcError {
    WebRtcError::InvalidStreamChannelCount {
        codec: codec_spec.codec,
        channels: codec_spec.channels.unwrap_or(0),
    }
}

fn ffmpeg_codec(codec: Codec) -> Option<(ffmpeg::media::Type, ffmpeg::codec::Id)> {
    use ffmpeg::{codec::Id, media::Type};
    match codec {
        Codec::Opus => Some((Type::Audio, Id::OPUS)),
        Codec::PCMU => Some((Type::Audio, Id::PCM_MULAW)),
        Codec::PCMA => Some((Type::Audio, Id::PCM_ALAW)),
        Codec::H264 => Some((Type::Video, Id::H264)),
        Codec::H265 => Some((Type::Video, Id::HEVC)),
        Codec::H266 => Some((Type::Video, Id::VVC)),
        Codec::Vp8 => Some((Type::Video, Id::VP8)),
        Codec::Vp9 => Some((Type::Video, Id::VP9)),
        Codec::Av1 => Some((Type::Video, Id::AV1)),
        _ => None,
    }
}

/// The str0m codec an encoder or demuxer describing itself with `id` feeds.
/// The inverse of [`ffmpeg_codec`], and deliberately its mirror image: a
/// codec added to one has to be added to the other.
pub(super) fn str0m_codec(id: ffmpeg::codec::Id) -> Option<Codec> {
    use ffmpeg::codec::Id;
    match id {
        Id::OPUS => Some(Codec::Opus),
        Id::PCM_MULAW => Some(Codec::PCMU),
        Id::PCM_ALAW => Some(Codec::PCMA),
        Id::H264 => Some(Codec::H264),
        Id::HEVC => Some(Codec::H265),
        Id::VVC => Some(Codec::H266),
        Id::VP8 => Some(Codec::Vp8),
        Id::VP9 => Some(Codec::Vp9),
        Id::AV1 => Some(Codec::Av1),
        _ => None,
    }
}

pub(super) use crate::elements::parameter_sets::annex_b as annex_b_nalus;

fn parse_h264_dimensions(sps: &[u8]) -> Option<(u32, u32)> {
    crate::elements::parameter_sets::h264_shape(sps).map(|shape| (shape.width, shape.height))
}

#[cfg(test)]
mod tests {
    use ffmpeg_next as ffmpeg;
    use str0m::{
        format::{Codec, CodecSpec, FormatParams},
        media::Frequency,
    };

    use super::{StreamInfoProbe, WebRtcStreamInfo};
    use crate::elements::WebRtcError;

    const SPS: &[u8] = &[
        0x67, 0x42, 0xc0, 0x1f, 0x1a, 0x32, 0x35, 0x01, 0x40, 0x7a, 0x40, 0x3c, 0x22, 0x11, 0xa8,
    ];
    const PPS: &[u8] = &[0x68, 0x1a, 0x34, 0xe3, 0xc8];

    fn spec(codec: Codec, clock_rate: Frequency, channels: Option<u8>) -> CodecSpec {
        CodecSpec {
            codec,
            clock_rate,
            channels,
            format: FormatParams::default(),
        }
    }

    #[test]
    fn h264_waits_for_both_parameter_sets_and_derives_codec_parameters() {
        let codec = spec(Codec::H264, Frequency::NINETY_KHZ, None);
        let mut probe = StreamInfoProbe::new();
        let mut sps_payload = vec![0, 0, 0, 1];
        sps_payload.extend_from_slice(SPS);
        assert!(probe.observe(codec, &sps_payload).is_none());

        let mut pps_payload = vec![0, 0, 1];
        pps_payload.extend_from_slice(PPS);
        let info = probe
            .observe(codec, &pps_payload)
            .expect("PPS completes H.264 stream info");
        let parameters = info.codec_parameters().expect("H.264 codec parameters");

        assert_eq!(info.video_dimensions(), Some((640, 480)));
        assert_eq!(info.time_base().unwrap(), ffmpeg::Rational::new(1, 90_000));
        assert_eq!(parameters.id(), ffmpeg::codec::Id::H264);
        // SAFETY: read-only access to live parameters.
        unsafe {
            assert_eq!((*parameters.as_ptr()).width, 640);
            assert_eq!((*parameters.as_ptr()).height, 480);
            assert!((*parameters.as_ptr()).extradata_size > 0);
        }
    }

    #[test]
    fn opus_derives_complete_codec_parameters() {
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, Some(2)));
        let parameters = info.codec_parameters().expect("Opus codec parameters");
        assert_eq!(parameters.id(), ffmpeg::codec::Id::OPUS);
        // SAFETY: read-only access to live parameters.
        unsafe {
            assert_eq!((*parameters.as_ptr()).sample_rate, 48_000);
            assert_eq!((*parameters.as_ptr()).ch_layout.nb_channels, 2);
            let extra = std::slice::from_raw_parts(
                (*parameters.as_ptr()).extradata,
                (*parameters.as_ptr()).extradata_size as usize,
            );
            assert_eq!(&extra[..8], b"OpusHead");
        }
    }

    #[test]
    fn vp8_derives_codec_parameters_without_container_policy() {
        let info = WebRtcStreamInfo::from(spec(Codec::Vp8, Frequency::NINETY_KHZ, None));
        let parameters = info.codec_parameters().expect("VP8 codec parameters");

        assert_eq!(parameters.id(), ffmpeg::codec::Id::VP8);
        assert_eq!(info.time_base().unwrap(), ffmpeg::Rational::new(1, 90_000));
    }

    /// A received track records as what it is: its own parameters, timed in
    /// its own RTP clock — here Opus at 48 kHz, not some default.
    #[test]
    fn a_track_format_is_the_streams_own_parameters_and_clock() {
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, Some(2)));
        let format = info.stream_info().expect("Opus track format");
        assert_eq!(format.parameters().id(), ffmpeg::codec::Id::OPUS);
        assert_eq!(format.time_base(), info.time_base().unwrap());
        assert_eq!(format.time_base(), ffmpeg::Rational::new(1, 48_000));
    }

    /// And fails the way its parameters would, rather than recording a
    /// track described by nothing.
    #[test]
    fn a_stream_without_codec_parameters_has_no_track_format() {
        let info = WebRtcStreamInfo::from(spec(Codec::Rtx, Frequency::NINETY_KHZ, None));
        assert!(matches!(
            info.stream_info().expect_err("RTX track-format rejection"),
            crate::Error::WebRtcError(WebRtcError::UnsupportedCodecParameters(Codec::Rtx))
        ));
    }

    #[test]
    fn rtx_cannot_form_codec_parameters() {
        let info = WebRtcStreamInfo::from(spec(Codec::Rtx, Frequency::NINETY_KHZ, None));
        assert!(matches!(
            info.codec_parameters()
                .err()
                .expect("RTX codec-parameter rejection"),
            crate::Error::WebRtcError(WebRtcError::UnsupportedCodecParameters(Codec::Rtx))
        ));
    }

    #[test]
    fn invalid_clock_rate_and_channels_are_typed_errors() {
        let clock_rate = Frequency::new(u32::MAX).unwrap();
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, clock_rate, Some(2)));
        assert!(matches!(
            info.time_base().unwrap_err(),
            crate::Error::WebRtcError(WebRtcError::InvalidStreamClockRate {
                codec: Codec::Opus,
                clock_rate: u32::MAX,
            })
        ));

        let no_channels =
            WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, None));
        assert!(matches!(
            no_channels
                .codec_parameters()
                .err()
                .expect("complete Opus parameters require a channel count"),
            crate::Error::WebRtcError(WebRtcError::InvalidStreamChannelCount {
                codec: Codec::Opus,
                channels: 0,
            })
        ));

        let zero_channels =
            WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, Some(0)));
        assert!(matches!(
            zero_channels
                .codec_parameters()
                .err()
                .expect("zero channels are invalid"),
            crate::Error::WebRtcError(WebRtcError::InvalidStreamChannelCount {
                codec: Codec::Opus,
                channels: 0,
            })
        ));
    }

    /// A payload that never completes the parameter sets leaves the probe
    /// unconfirmed rather than panicking: this runs on `WebRtcPeer`'s own
    /// ICE/DTLS thread, directly on bytes a remote peer chose, so the only
    /// two acceptable outcomes are "confirmed" and "not yet". The caller
    /// sees the second as a track that has not said what it carries yet.
    ///
    /// `parse_h264_dimensions` and `BitReader` return `Option` throughout
    /// today, and `annex_b_h264_extradata`'s own validation is unreachable
    /// behind that — these cases exist so a later rewrite reaching for
    /// indexing or `unwrap` fails here instead of taking the connection
    /// down.
    #[test]
    fn malformed_h264_payloads_never_confirm_and_never_panic() {
        let codec = spec(Codec::H264, Frequency::NINETY_KHZ, None);
        let annex_b = |nalu: &[u8]| {
            let mut payload = vec![0, 0, 0, 1];
            payload.extend_from_slice(nalu);
            payload
        };

        // Nothing at all, and a start code with no NAL behind it.
        for payload in [vec![], vec![0, 0, 0, 1], vec![0, 0, 1]] {
            let mut probe = StreamInfoProbe::new();
            assert!(
                probe.observe(codec, &payload).is_none(),
                "an empty payload cannot confirm a stream"
            );
        }

        // NAL types that are neither SPS (7) nor PPS (8) — an IDR slice and
        // an SEI, both of which a real sender emits constantly.
        for nal_type in [1u8, 5, 6, 9, 31] {
            let mut probe = StreamInfoProbe::new();
            let payload = annex_b(&[0x60 | nal_type, 0x42, 0xc0, 0x1f]);
            assert!(
                probe.observe(codec, &payload).is_none(),
                "NAL type {nal_type} must not be mistaken for a parameter set"
            );
        }

        // Every truncation of a real SPS, each followed by a valid PPS, so a
        // parameter set that cannot be parsed is never rescued by the other
        // one arriving intact.
        //
        // Not every truncation fails to parse: an SPS carries its dimension
        // fields well before its end, so cutting off only the trailing VUI
        // still yields the real width and height rather than garbage. What
        // has to hold for *all* of them is that the outcome is one of the
        // two the caller can act on — unconfirmed, or confirmed with
        // parameters that actually build — and never a panic on this
        // thread. Anything shorter than the NAL header plus profile bytes
        // is rejected outright by `parse_h264_dimensions`' own guard.
        for length in 0..SPS.len() {
            let mut probe = StreamInfoProbe::new();
            let _ = probe.observe(codec, &annex_b(&SPS[..length]));
            let confirmed = probe.observe(codec, &annex_b(PPS));
            if length < 4 {
                assert!(
                    confirmed.is_none(),
                    "a {length}-byte SPS is too short to describe anything"
                );
            }
            if let Some(info) = confirmed {
                info.codec_parameters().unwrap_or_else(|error| {
                    panic!("a {length}-byte SPS confirmed but then failed to build: {error}")
                });
            }
        }

        // Exp-Golomb with more leading zeroes than `read_ue` accepts: a
        // valid SPS NAL header and profile bytes, then a run of zero bits
        // long enough to overflow the shift the suffix is built with.
        let mut sps = vec![0x67, 0x42, 0xc0, 0x1f];
        sps.extend(std::iter::repeat_n(0x00, 16));
        sps.push(0x01);
        let mut probe = StreamInfoProbe::new();
        let _ = probe.observe(codec, &annex_b(&sps));
        assert!(
            probe.observe(codec, &annex_b(PPS)).is_none(),
            "an over-long Exp-Golomb code must fail the parse, not the process"
        );
    }
}
