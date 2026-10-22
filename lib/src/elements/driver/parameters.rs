//! Completing a received track's codec parameters — its extradata, and a
//! picture's size read from its first keyframe — for the servers that take
//! publishers' streams in: RTSP and RTMP.

use std::ptr;

use ffmpeg_next::{self as ffmpeg, ffi};

/// Replaces `parameters`' extradata with `extradata`.
pub(crate) fn set_extradata(parameters: &mut ffmpeg::codec::Parameters, extradata: &[u8]) {
    // SAFETY: the parameters' own extradata, replaced with a buffer from
    // FFmpeg's allocator with its padding, as FFmpeg frees it.
    unsafe {
        let par = parameters.as_mut_ptr();
        let data = ffi::av_mallocz(extradata.len() + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize)
            .cast::<u8>();
        if data.is_null() {
            return;
        }
        ptr::copy_nonoverlapping(extradata.as_ptr(), data, extradata.len());
        ffi::av_freep(ptr::addr_of_mut!((*par).extradata).cast());
        (*par).extradata = data;
        (*par).extradata_size = extradata.len() as i32;
    }
}

/// A picture's size, as FFmpeg's parser of `codec` reads it from `frame`
/// given `extradata`: Annex B parameter sets and a keyframe with no
/// extradata, or an `avcC`/`hvcC` record and a length-prefixed keyframe.
pub(crate) fn picture_size(
    codec: ffmpeg::codec::Id,
    extradata: &[u8],
    frame: &[u8],
) -> Option<(i32, i32)> {
    let mut padded = frame.to_vec();
    padded.resize(frame.len() + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize, 0);
    // SAFETY: a parser and a codec context made, used and freed here; the
    // input and the extradata are padded as the parser reads past their
    // ends, and the extradata is allocated by FFmpeg, which frees it with
    // the context.
    unsafe {
        let parser = ffi::av_parser_init(ffi::AVCodecID::from(codec) as i32);
        if parser.is_null() {
            return None;
        }
        let mut context = ffi::avcodec_alloc_context3(ptr::null());
        if context.is_null() {
            ffi::av_parser_close(parser);
            return None;
        }
        if !extradata.is_empty() {
            let data =
                ffi::av_mallocz(extradata.len() + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize)
                    .cast::<u8>();
            if !data.is_null() {
                ptr::copy_nonoverlapping(extradata.as_ptr(), data, extradata.len());
                (*context).extradata = data;
                (*context).extradata_size = extradata.len() as i32;
            }
        }
        (*parser).flags |= ffi::PARSER_FLAG_COMPLETE_FRAMES;
        let mut out = ptr::null_mut();
        let mut out_size = 0;
        ffi::av_parser_parse2(
            parser,
            context,
            &mut out,
            &mut out_size,
            padded.as_ptr(),
            frame.len() as i32,
            ffi::AV_NOPTS_VALUE,
            ffi::AV_NOPTS_VALUE,
            0,
        );
        let size = ((*parser).width, (*parser).height);
        ffi::av_parser_close(parser);
        ffi::avcodec_free_context(&mut context);
        (size.0 > 0 && size.1 > 0).then_some(size)
    }
}
