//! [`StreamInfo`]: what one stream is — what a source reports of each it
//! hands out, what a muxer's track and a broadcast are opened from, and what
//! an encoder says of its own output.

use ffmpeg_next as ffmpeg;

/// One stream: its codec parameters, the unit its timestamps are in, how
/// many pictures a second it has where it says, and which of its source's
/// outputs it comes out of.
///
/// A source reports one for each stream before the pipeline runs —
/// [`FileDemuxer::open`](crate::elements::FileDemuxer::open),
/// [`RtspSource::open`](crate::elements::RtspSource::open),
/// `WhepSource::open` (with `webrtc`), a
/// [`Broadcast`](crate::elements::Broadcast)'s tracks — so that what is
/// built downstream is built from what the stream is, without asking again
/// by index. A muxer's `add_stream` and [`Broadcast::publish`] take one,
/// and an encoder of this crate gives one of what it makes: `&encoder`
/// turns into it. [`StreamInfo::new`] is for a stream described by hand.
///
/// It is the stream as it began. A stream whose parameters change partway —
/// a camera reconfigured, a WebRTC sender that changes its picture size —
/// no longer matches it from there. A decoder reads the new parameter sets
/// in band and follows; what keeps the first in a header, as an MP4 does,
/// refuses the stream from there.
///
/// [`Broadcast::publish`]: crate::elements::Broadcast::publish
#[derive(Clone)]
pub struct StreamInfo {
    index: usize,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    frame_rate: Option<ffmpeg::Rational>,
    /// Whether the stream's parameter sets are the same from its first
    /// packet to its last, so an MP4 may keep them in the sample entry
    /// alone even where `parameters` carries none: what an encoder of this
    /// crate knows of its own output, and nothing can know of a stream
    /// copied from elsewhere, whose sets may change partway.
    fixed_parameter_sets: bool,
}

impl StreamInfo {
    /// A stream whose packets carry `parameters` and are timed in
    /// `time_base`: the first of its source's outputs, with no frame rate
    /// said.
    pub fn new(parameters: ffmpeg::codec::Parameters, time_base: ffmpeg::Rational) -> Self {
        Self {
            index: 0,
            parameters,
            time_base,
            frame_rate: None,
            fixed_parameter_sets: false,
        }
    }

    /// A picture stream received from outside FFmpeg — an RTSP or RTP
    /// client of the application's own, say — described by its codec, its
    /// configuration (`extradata`: Annex B parameter sets for H.264 and
    /// H.265, as FFmpeg's own RTSP client gives them), its size, and the unit
    /// its timestamps count in.
    ///
    /// Its packets are then what this crate's sources hand on for the same
    /// codec: H.264 and H.265 in Annex B, keyframes carrying their parameter
    /// sets where a picture is to be joined at them.
    pub fn video(
        codec: ffmpeg::codec::Id,
        extradata: &[u8],
        (width, height): (u32, u32),
        time_base: ffmpeg::Rational,
    ) -> Result<Self, StreamInfoError> {
        let mut parameters = described(codec, ffmpeg::media::Type::Video, extradata)?;
        // SAFETY: parameters made here and owned alone; plain fields.
        unsafe {
            let par = parameters.as_mut_ptr();
            (*par).width = width as i32;
            (*par).height = height as i32;
        }
        Ok(Self::new(parameters, time_base))
    }

    /// A sound stream received from outside FFmpeg, described by its codec,
    /// its configuration (`extradata`: AAC's AudioSpecificConfig, nothing
    /// for G.711), its rate and channels, and the unit its timestamps count
    /// in — see [`StreamInfo::video`]. AAC's packets are raw, without ADTS
    /// headers, as FFmpeg's own RTSP client hands them on.
    pub fn audio(
        codec: ffmpeg::codec::Id,
        extradata: &[u8],
        sample_rate: u32,
        channels: u16,
        time_base: ffmpeg::Rational,
    ) -> Result<Self, StreamInfoError> {
        let mut parameters = described(codec, ffmpeg::media::Type::Audio, extradata)?;
        // SAFETY: parameters made here and owned alone; plain fields, and
        // the channel layout FFmpeg fills in itself.
        unsafe {
            let par = parameters.as_mut_ptr();
            (*par).sample_rate = sample_rate as i32;
            ffmpeg::ffi::av_channel_layout_default(&mut (*par).ch_layout, i32::from(channels));
            if codec == ffmpeg::codec::Id::AAC {
                (*par).frame_size = 1024;
            }
        }
        Ok(Self::new(parameters, time_base))
    }

    /// What one of a container's streams says about itself.
    pub(crate) fn of(stream: &ffmpeg::format::stream::Stream<'_>) -> Self {
        // A copy of its own. What a stream hands out shares the whole
        // input's ownership through an `Rc` — so it kept the file (or the
        // RTSP connection) open for as long as the caller held this, and,
        // this going to one thread and the source to another, counted that
        // `Rc` from two threads at once.
        let parameters = stream.parameters().clone();
        let rate = stream.avg_frame_rate();
        let video = parameters.medium() == ffmpeg::media::Type::Video;
        Self {
            index: stream.index(),
            frame_rate: (video && rate.numerator() > 0 && rate.denominator() > 0).then_some(rate),
            parameters,
            time_base: stream.time_base(),
            fixed_parameter_sets: false,
        }
    }

    /// The same stream, coming out of its source's output `index`.
    #[must_use]
    pub(crate) fn with_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }

    /// The same stream, said to have `frame_rate` pictures a second — what
    /// an encoder re-encoding it is opened at.
    #[must_use]
    pub fn with_frame_rate(mut self, frame_rate: ffmpeg::Rational) -> Self {
        self.frame_rate = Some(frame_rate);
        self
    }

    /// The same stream, said to keep one set of parameter sets throughout —
    /// for an encoder of this crate whose FFmpeg encoder hands its sets only
    /// in band (`hevc_d3d12va`), so that a muxer of MP4 still writes its
    /// HEVC as `hvc1`.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    pub(crate) fn with_fixed_parameter_sets(mut self) -> Self {
        self.fixed_parameter_sets = true;
        self
    }

    /// The same stream, its pictures shown turned as `orientation` says:
    /// the display matrix a container keeps beside a video track — what a
    /// re-encoded phone recording has to carry on, given
    /// [`Self::orientation`] of the stream it came from. An upright one
    /// takes away any matrix the stream carried. Only a container that
    /// keeps one writes it — MP4 and MOV do.
    #[must_use]
    pub fn with_orientation(mut self, orientation: crate::orientation::Orientation) -> Self {
        let kind = ffmpeg::ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX;
        // SAFETY: the parameters are this value's own; FFmpeg's side-data
        // functions keep its list and count in step, and a new entry's
        // `data` is the `size` bytes asked for, which the matrix fills.
        unsafe {
            let raw = self.parameters.as_mut_ptr();
            ffmpeg::ffi::av_packet_side_data_remove(
                (*raw).coded_side_data,
                &mut (*raw).nb_coded_side_data,
                kind,
            );
            if !orientation.is_upright() {
                let matrix = orientation.matrix();
                let size = size_of_val(&matrix);
                let entry = ffmpeg::ffi::av_packet_side_data_new(
                    &mut (*raw).coded_side_data,
                    &mut (*raw).nb_coded_side_data,
                    kind,
                    size,
                    0,
                );
                if entry.is_null() {
                    std::alloc::handle_alloc_error(
                        std::alloc::Layout::array::<i32>(matrix.len()).expect("nine i32s"),
                    );
                }
                std::ptr::copy_nonoverlapping(matrix.as_ptr().cast::<u8>(), (*entry).data, size);
            }
        }
        self
    }

    /// Which of its source's outputs it comes out of: the stream's index in
    /// a file or an RTSP session, its place among a broadcast's tracks. Zero
    /// for one an encoder or a caller made.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Whether it is sound, video or something else, as its parameters say.
    pub fn kind(&self) -> ffmpeg::media::Type {
        self.parameters.medium()
    }

    /// What a decoder for it is built from —
    /// [`SwDecoder`](crate::elements::SwDecoder),
    /// [`VideoDecodeBin`](crate::elements::VideoDecodeBin), and the rest.
    pub fn parameters(&self) -> &ffmpeg::codec::Parameters {
        &self.parameters
    }

    /// The unit its packets' timestamps are in — what a
    /// [`Pacer`](crate::elements::Pacer) for it is built with.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.time_base
    }

    /// How many pictures a second a video stream has on average, as its
    /// source says — what an encoder re-encoding it is opened at. `None`
    /// for sound, and for a video stream whose source does not say.
    pub fn frame_rate(&self) -> Option<ffmpeg::Rational> {
        self.frame_rate
    }

    /// The parameters, kept.
    pub(crate) fn into_parameters(self) -> ffmpeg::codec::Parameters {
        self.parameters
    }

    /// The parameters, to put what was found out about the stream in them.
    pub(crate) fn parameters_mut(&mut self) -> &mut ffmpeg::codec::Parameters {
        &mut self.parameters
    }

    /// Whether an encoder of this crate said its parameter sets stay the
    /// same throughout — see [`Self::with_fixed_parameter_sets`].
    pub(crate) fn fixed_parameter_sets(&self) -> bool {
        self.fixed_parameter_sets
    }

    /// What a video stream says its Y'CbCr is — matrix, range, primaries
    /// and transfer — for an encoder re-encoding its decoded pictures to
    /// write into its own stream. `None` for sound, and for a video stream
    /// that names no matrix.
    pub fn color(&self) -> Option<crate::color::ColorDescription> {
        // SAFETY: plain fields of parameters this value owns.
        let raw = unsafe { &*self.parameters.as_ptr() };
        let space = ffmpeg::color::Space::from(raw.color_space);
        (self.kind() == ffmpeg::media::Type::Video && space != ffmpeg::color::Space::Unspecified)
            .then(|| crate::color::ColorDescription {
                space,
                range: raw.color_range.into(),
                primaries: raw.color_primaries.into(),
                transfer: raw.color_trc.into(),
            })
    }

    /// Which way up a video stream's pictures are shown — the display
    /// matrix its container keeps beside them, which a phone's portrait
    /// recording carries — and upright where it keeps none, or for sound.
    /// Each decoded picture carries it too, read by
    /// [`Orientation::of`](crate::orientation::Orientation::of); what a file
    /// its pictures are re-encoded to is to say is this, given to
    /// [`Self::with_orientation`] of the encoder's stream.
    ///
    /// # Errors
    ///
    /// Where the matrix turns the pictures by other than quarter turns.
    pub fn orientation(
        &self,
    ) -> Result<crate::orientation::Orientation, crate::orientation::UnsupportedOrientation> {
        // SAFETY: plain fields of parameters this value owns; the side data
        // is FFmpeg's own, a display matrix being nine `i32`s by its
        // definition, which the size check makes sure of.
        unsafe {
            let raw = self.parameters.as_ptr();
            let data = ffmpeg::ffi::av_packet_side_data_get(
                (*raw).coded_side_data,
                (*raw).nb_coded_side_data,
                ffmpeg::ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            );
            if data.is_null() || (*data).size < 9 * size_of::<i32>() {
                return Ok(crate::orientation::Orientation::UPRIGHT);
            }
            crate::orientation::Orientation::of_matrix(&*((*data).data as *const [i32; 9]))
        }
    }

    /// What a video stream's pictures are in, decoded in software — 4:2:0
    /// at eight bits or ten, say — as its parameters say. A hardware
    /// decoder puts the same in its own layout: a 10-bit stream is P010 out
    /// of NVDEC, which [`crate::contract::PixelLayoutSet::decoded_from`]
    /// says. `None` for sound, and for a video stream that does not say.
    pub fn pixel_format(&self) -> Option<ffmpeg::format::Pixel> {
        // SAFETY: a plain field of parameters this value owns.
        let format = unsafe { (*self.parameters.as_ptr()).format };
        let known = 0..ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NB as i32;
        if self.kind() != ffmpeg::media::Type::Video || !known.contains(&format) {
            return None;
        }
        // SAFETY: every value from naught to `AV_PIX_FMT_NB` names one of
        // FFmpeg's pixel formats, which the enum has a variant for each of.
        let format = unsafe { std::mem::transmute::<i32, ffmpeg::ffi::AVPixelFormat>(format) };
        Some(format.into())
    }

    /// A video stream's picture size, width then height, as its parameters
    /// say — what an encoder re-encoding it is opened at. `None` for sound,
    /// and for a video stream that does not say.
    pub fn size(&self) -> Option<(u32, u32)> {
        // SAFETY: plain fields of parameters this value owns.
        let (width, height) = unsafe {
            let raw = self.parameters.as_ptr();
            ((*raw).width, (*raw).height)
        };
        (self.kind() == ffmpeg::media::Type::Video && width > 0 && height > 0)
            .then_some((width as u32, height as u32))
    }

    /// Whether a video stream's pictures come out of the order they are
    /// shown in — B-frames — as its parameters say: what an encoder said of
    /// its own output, or what probing the stream found. WebRTC carries no
    /// such stream to a browser. False for sound, and for a stream described
    /// by hand, which says nothing of it.
    pub fn reorders(&self) -> bool {
        // SAFETY: a plain field of parameters this value owns.
        let delay = unsafe { (*self.parameters.as_ptr()).video_delay };
        self.kind() == ffmpeg::media::Type::Video && delay > 0
    }
}

/// Why a [`StreamInfo`] could not be described by hand.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamInfoError {
    /// The codec is not one of the medium asked for: sound described as a
    /// picture, or a picture as sound.
    #[error("{codec:?} is not a {expected:?} codec")]
    WrongMedium {
        /// The codec given.
        codec: ffmpeg::codec::Id,
        /// What it was to be.
        expected: ffmpeg::media::Type,
    },
}

/// Codec parameters of `codec`, a codec of `medium`, carrying `extradata`.
fn described(
    codec: ffmpeg::codec::Id,
    medium: ffmpeg::media::Type,
    extradata: &[u8],
) -> Result<ffmpeg::codec::Parameters, StreamInfoError> {
    if codec.medium() != medium {
        return Err(StreamInfoError::WrongMedium {
            codec,
            expected: medium,
        });
    }
    let mut parameters = ffmpeg::codec::Parameters::new();
    // SAFETY: parameters made here and owned alone; plain fields, and an
    // extradata buffer from FFmpeg's allocator with its padding, which
    // FFmpeg frees with the parameters.
    unsafe {
        let par = parameters.as_mut_ptr();
        (*par).codec_type = medium.into();
        (*par).codec_id = codec.into();
        if !extradata.is_empty() {
            let size = extradata.len() + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
            let data = ffmpeg::ffi::av_mallocz(size).cast::<u8>();
            if !data.is_null() {
                std::ptr::copy_nonoverlapping(extradata.as_ptr(), data, extradata.len());
                (*par).extradata = data;
                (*par).extradata_size = extradata.len() as i32;
            }
        }
    }
    Ok(parameters)
}

/// A copy, for whatever takes one by value: a muxer's `add_stream` given
/// `&stream` of a demuxed stream copied through unchanged.
impl From<&StreamInfo> for StreamInfo {
    fn from(stream: &StreamInfo) -> Self {
        stream.clone()
    }
}

/// By hand, since FFmpeg's parameters have no `Debug` of their own: the
/// codec stands in for them.
impl std::fmt::Debug for StreamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamInfo")
            .field("index", &self.index)
            .field("kind", &self.kind())
            .field("codec", &self.parameters.id())
            .field("time_base", &self.time_base)
            .field("frame_rate", &self.frame_rate)
            .field("size", &self.size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANNEX_B_SETS: [u8; 12] = [0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1e, 0, 0, 0, 1];

    #[test]
    fn a_picture_stream_described_by_hand_is_what_was_said() {
        let info = StreamInfo::video(
            ffmpeg::codec::Id::H264,
            &ANNEX_B_SETS,
            (1920, 1080),
            ffmpeg::Rational::new(1, 90_000),
        )
        .unwrap();
        assert_eq!(info.kind(), ffmpeg::media::Type::Video);
        assert_eq!(info.parameters().id(), ffmpeg::codec::Id::H264);
        assert_eq!(info.size(), Some((1920, 1080)));
        assert_eq!(info.time_base(), ffmpeg::Rational::new(1, 90_000));
        // SAFETY: reading the parameters' own extradata, as long as it says.
        let extradata = unsafe {
            let par = info.parameters().as_ptr();
            std::slice::from_raw_parts((*par).extradata, (*par).extradata_size as usize)
        };
        assert_eq!(extradata, ANNEX_B_SETS);
    }

    #[test]
    fn a_sound_stream_described_by_hand_is_what_was_said() {
        // AudioSpecificConfig: AAC LC, 48 kHz, one channel.
        let config = [0x11, 0x88];
        let info = StreamInfo::audio(
            ffmpeg::codec::Id::AAC,
            &config,
            48_000,
            1,
            ffmpeg::Rational::new(1, 48_000),
        )
        .unwrap();
        assert_eq!(info.kind(), ffmpeg::media::Type::Audio);
        // SAFETY: plain fields of parameters this test owns.
        let (rate, channels, frame_size) = unsafe {
            let par = info.parameters().as_ptr();
            (
                (*par).sample_rate,
                (*par).ch_layout.nb_channels,
                (*par).frame_size,
            )
        };
        assert_eq!((rate, channels, frame_size), (48_000, 1, 1024));
        let g711 = StreamInfo::audio(
            ffmpeg::codec::Id::PCM_MULAW,
            &[],
            8_000,
            1,
            ffmpeg::Rational::new(1, 8_000),
        )
        .unwrap();
        assert_eq!(g711.parameters().id(), ffmpeg::codec::Id::PCM_MULAW);
    }

    #[test]
    fn a_codec_of_the_other_medium_is_refused() {
        let unit = ffmpeg::Rational::new(1, 90_000);
        assert_eq!(
            StreamInfo::video(ffmpeg::codec::Id::AAC, &[], (1, 1), unit).unwrap_err(),
            StreamInfoError::WrongMedium {
                codec: ffmpeg::codec::Id::AAC,
                expected: ffmpeg::media::Type::Video,
            }
        );
        assert!(StreamInfo::audio(ffmpeg::codec::Id::H264, &[], 8_000, 1, unit).is_err());
    }

    #[test]
    fn a_stream_reorders_where_its_encoder_puts_b_frames_in() {
        use crate::elements::{SwEncoder, SwEncoderOptions, VideoCodec};
        let encoder = |codec, max_b_frames| {
            SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec,
                    width: 320,
                    height: 240,
                    pixel_format: ffmpeg::format::Pixel::YUV420P,
                    frame_rate: ffmpeg::Rational::new(25, 1),
                    bit_rate: 300_000,
                    gop_size: 25,
                    max_b_frames,
                },
            )
        };
        let plain = encoder(VideoCodec::OpenH264, None).expect("OpenH264 is always there");
        assert!(!StreamInfo::from(&plain).reorders());
        let by_hand = StreamInfo::video(
            ffmpeg::codec::Id::H264,
            &ANNEX_B_SETS,
            (320, 240),
            ffmpeg::Rational::new(1, 90_000),
        )
        .unwrap();
        assert!(!by_hand.reorders(), "nothing said of it");
        let Ok(b_frames) = encoder(VideoCodec::H264, Some(2)) else {
            eprintln!("skipping the B-frame half: no libx264 in this FFmpeg");
            return;
        };
        assert!(StreamInfo::from(&b_frames).reorders());
    }
}
