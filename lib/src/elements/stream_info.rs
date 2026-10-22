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
