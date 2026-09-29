//! Core Video pixel buffers — what a camera hands over through AVFoundation,
//! and a screen through ScreenCaptureKit — as the frames the rest of this
//! crate takes: NV12 or BGRA copied into system memory, or, with
//! `videotoolbox`, the buffer itself as a VideoToolbox frame, nothing
//! copied.

use ffmpeg_next as ffmpeg;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};

/// Why a pixel buffer could not be made a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PixelBufferError {
    /// It holds a layout other than the one asked for — its four-character
    /// code.
    Unexpected(u32),
    /// Core Video would not lock it to be read.
    Lock(i32),
    /// Its planes are smaller than its size says.
    Truncated,
}

/// What a pixel buffer holds, of the layouts this crate reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// NV12, `420v` or `420f`: the range it is in.
    Nv12(ffmpeg::color::Range),
    /// `BGRA`, full-range RGB.
    Bgra,
}

impl Layout {
    fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Nv12(_) => ffmpeg::format::Pixel::NV12,
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
        }
    }

    /// Says on `frame` what its samples mean.
    fn describe(self, frame: &mut ffmpeg::frame::Video) {
        match self {
            Self::Nv12(range) => frame.set_color_range(range),
            Self::Bgra => {
                frame.set_color_space(ffmpeg::color::Space::RGB);
                frame.set_color_range(ffmpeg::color::Range::JPEG);
            }
        }
    }
}

/// One pixel buffer, held. A clone holds it too.
#[derive(Clone)]
pub(crate) struct PixelBuffer(CFRetained<CVPixelBuffer>);

// SAFETY: a Core Video buffer's reference count is atomic, and a pixel
// buffer is safe to read from any thread; this crate only ever reads one,
// under its own lock, once its producer has handed it over.
unsafe impl Send for PixelBuffer {}

impl PixelBuffer {
    pub(crate) fn new(buffer: CFRetained<CVPixelBuffer>) -> Self {
        Self(buffer)
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (
            CVPixelBufferGetWidth(&self.0) as u32,
            CVPixelBufferGetHeight(&self.0) as u32,
        )
    }

    /// What it holds, where it is `want` — NV12 or BGRA, as FFmpeg names
    /// them.
    fn layout(&self, want: ffmpeg::format::Pixel) -> Result<Layout, PixelBufferError> {
        let format = CVPixelBufferGetPixelFormatType(&self.0);
        let layout = if format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            Some(Layout::Nv12(ffmpeg::color::Range::MPEG))
        } else if format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            Some(Layout::Nv12(ffmpeg::color::Range::JPEG))
        } else if format == kCVPixelFormatType_32BGRA {
            Some(Layout::Bgra)
        } else {
            None
        };
        layout
            .filter(|layout| layout.pixel() == want)
            .ok_or(PixelBufferError::Unexpected(format))
    }

    /// Copies the picture, which must be `want`, into `frame`, which is made
    /// `want` of its size where it is not already, and says what its samples
    /// mean.
    pub(crate) fn copy_to(
        &self,
        frame: &mut ffmpeg::frame::Video,
        want: ffmpeg::format::Pixel,
    ) -> Result<(), PixelBufferError> {
        let layout = self.layout(want)?;
        let (width, height) = self.size();
        if frame.format() != want || frame.width() != width || frame.height() != height {
            *frame = ffmpeg::frame::Video::new(want, width, height);
        }
        // SAFETY: a live pixel buffer, locked for reading here and unlocked
        // below with the same flags, whatever the copy did.
        let status =
            unsafe { CVPixelBufferLockBaseAddress(&self.0, CVPixelBufferLockFlags::ReadOnly) };
        if status != 0 {
            return Err(PixelBufferError::Lock(status));
        }
        let copied = match layout {
            Layout::Nv12(_) => self.copy_planes(frame, width as usize),
            Layout::Bgra => self.copy_packed(frame, width as usize * 4),
        };
        // SAFETY: as above.
        unsafe { CVPixelBufferUnlockBaseAddress(&self.0, CVPixelBufferLockFlags::ReadOnly) };
        copied?;
        layout.describe(frame);
        Ok(())
    }

    /// Both NV12 planes, a row at a time: luma `width` bytes a row, and
    /// chroma as many, its Cb and Cr interleaved at half the rows. Called
    /// locked.
    fn copy_planes(
        &self,
        frame: &mut ffmpeg::frame::Video,
        width: usize,
    ) -> Result<(), PixelBufferError> {
        if CVPixelBufferGetPlaneCount(&self.0) != 2 {
            return Err(PixelBufferError::Truncated);
        }
        for plane in 0..2 {
            let source = CVPixelBufferGetBaseAddressOfPlane(&self.0, plane).cast::<u8>();
            let source_stride = CVPixelBufferGetBytesPerRowOfPlane(&self.0, plane);
            let rows = CVPixelBufferGetHeightOfPlane(&self.0, plane);
            copy_rows(frame, plane, source, source_stride, rows, width)?;
        }
        Ok(())
    }

    /// The one plane of a packed layout, `row_bytes` a row. Called locked.
    fn copy_packed(
        &self,
        frame: &mut ffmpeg::frame::Video,
        row_bytes: usize,
    ) -> Result<(), PixelBufferError> {
        let source = CVPixelBufferGetBaseAddress(&self.0).cast::<u8>();
        let source_stride = CVPixelBufferGetBytesPerRow(&self.0);
        let rows = CVPixelBufferGetHeight(&self.0);
        copy_rows(frame, 0, source, source_stride, rows, row_bytes)
    }

    /// The pixel buffer itself as a VideoToolbox frame, with `frames_ctx` —
    /// of its layout and size — as the context it says it came from: what a
    /// VideoToolbox decoder hands on, and what a VideoToolbox encoder or
    /// download reads. The frame holds the buffer until it is freed.
    #[cfg(feature = "videotoolbox")]
    pub(crate) fn into_videotoolbox_frame(
        self,
        frames_ctx: &crate::platform::ffmpeg::AvBufferRef,
    ) -> Result<ffmpeg::frame::Video, PixelBufferError> {
        use ffmpeg::ffi;

        // SAFETY: a live frames context, whose `data` is an
        // `AVHWFramesContext` by FFmpeg's own definition.
        let sw_format = unsafe {
            let frames = (*frames_ctx.as_ptr()).data as *const ffi::AVHWFramesContext;
            ffmpeg::format::Pixel::from((*frames).sw_format)
        };
        let layout = self.layout(sw_format)?;
        let (width, height) = self.size();

        /// Gives back the reference `into_videotoolbox_frame` took.
        unsafe extern "C" fn release(_opaque: *mut std::ffi::c_void, data: *mut u8) {
            // SAFETY: `data` is the pixel buffer whose reference was handed
            // to the frame's buffer below, given back exactly once.
            drop(unsafe {
                CFRetained::from_raw(std::ptr::NonNull::new_unchecked(
                    data.cast::<CVPixelBuffer>(),
                ))
            });
        }

        let mut frame = ffmpeg::frame::Video::empty();
        let pixel_buffer = CFRetained::into_raw(self.0).as_ptr();
        // SAFETY: `frame` is a fresh frame this owns. Its buffer takes the
        // pixel buffer's reference and gives it back through `release`, as
        // FFmpeg's own VideoToolbox decoder does; `data[3]` is where a
        // VideoToolbox frame keeps it; the frames context reference is a new
        // one of the caller's, which the frame frees.
        unsafe {
            let ptr = frame.as_mut_ptr();
            let buffer = ffi::av_buffer_create(
                pixel_buffer.cast::<u8>(),
                std::mem::size_of::<*mut CVPixelBuffer>(),
                Some(release),
                std::ptr::null_mut(),
                ffi::AV_BUFFER_FLAG_READONLY,
            );
            if buffer.is_null() {
                release(std::ptr::null_mut(), pixel_buffer.cast::<u8>());
                return Err(PixelBufferError::Truncated);
            }
            (*ptr).buf[0] = buffer;
            (*ptr).data[3] = pixel_buffer.cast::<u8>();
            (*ptr).format = ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX as i32;
            (*ptr).width = width as i32;
            (*ptr).height = height as i32;
            (*ptr).hw_frames_ctx = ffi::av_buffer_ref(frames_ctx.as_ptr());
        }
        layout.describe(&mut frame);
        Ok(frame)
    }
}

/// `rows_needed` of `frame`'s plane `plane`, `row_bytes` each, from `rows`
/// rows of `source_stride` bytes at `source`, where there are that many.
fn copy_rows(
    frame: &mut ffmpeg::frame::Video,
    plane: usize,
    source: *const u8,
    source_stride: usize,
    rows: usize,
    row_bytes: usize,
) -> Result<(), PixelBufferError> {
    let stride = frame.stride(plane);
    let needed = frame.plane_height(plane) as usize;
    if source.is_null() || source_stride < row_bytes || rows < needed {
        return Err(PixelBufferError::Truncated);
    }
    let destination = frame.data_mut(plane);
    for row in 0..needed {
        // SAFETY: the caller has the buffer locked, and the plane holds
        // `rows` rows of `source_stride` bytes from `source`, of which the
        // first `row_bytes` of this row are read — checked just above.
        let bytes =
            unsafe { std::slice::from_raw_parts(source.add(row * source_stride), row_bytes) };
        destination[row * stride..][..row_bytes].copy_from_slice(bytes);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use objc2_core_foundation::CFDictionary;
    use objc2_core_video::{CVPixelBufferCreate, kCVPixelBufferIOSurfacePropertiesKey};
    use objc2_foundation::{NSDictionary, NSString};

    /// An IOSurface-backed pixel buffer of `width` by `height` in `format`,
    /// its contents unset.
    fn create(width: usize, height: usize, format: u32) -> CFRetained<CVPixelBuffer> {
        let io_surface: objc2::rc::Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::new();
        // SAFETY: the key is Core Video's own constant, toll-free bridged to
        // the NSString the dictionary is keyed by, and the dictionary is as
        // well to the CFDictionary Core Video reads.
        let attributes: objc2::rc::Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> = unsafe {
            let key = &*(kCVPixelBufferIOSurfacePropertiesKey as *const _ as *const NSString);
            NSDictionary::from_retained_objects(
                &[key],
                &[objc2::rc::Retained::into_super(
                    objc2::rc::Retained::into_super(io_surface),
                )],
            )
        };
        let mut buffer: *mut CVPixelBuffer = std::ptr::null_mut();
        // SAFETY: `buffer` is a live out-param and the attributes a live
        // dictionary for the call.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                format,
                Some(&*(objc2::rc::Retained::as_ptr(&attributes) as *const CFDictionary)),
                std::ptr::NonNull::from(&mut buffer),
            )
        };
        assert_eq!(status, 0, "a pixel buffer is made");
        // SAFETY: Core Video handed over one reference, which this takes.
        unsafe { CFRetained::from_raw(std::ptr::NonNull::new(buffer).unwrap()) }
    }

    /// An IOSurface-backed BGRA pixel buffer of `width` by `height`, every
    /// pixel `pixel` — what a screen hands over.
    pub(crate) fn bgra_buffer(width: usize, height: usize, pixel: [u8; 4]) -> PixelBuffer {
        let buffer = create(width, height, kCVPixelFormatType_32BGRA);
        // SAFETY: a live pixel buffer, locked for writing and unlocked below.
        let locked = unsafe { CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        assert_eq!(locked, 0);
        let base = CVPixelBufferGetBaseAddress(&buffer).cast::<u8>();
        let stride = CVPixelBufferGetBytesPerRow(&buffer);
        for row in 0..height {
            // SAFETY: the buffer is locked for writing, and each row is
            // `stride` bytes, of which `width` pixels are the picture.
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(base.add(row * stride), width * 4) };
            for sample in bytes.chunks_mut(4) {
                sample.copy_from_slice(&pixel);
            }
        }
        // SAFETY: as above.
        unsafe { CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        PixelBuffer::new(buffer)
    }

    /// An IOSurface-backed NV12 pixel buffer of `width` by `height`, luma
    /// `luma` and chroma `cb`, `cr` — what a camera hands over.
    pub(crate) fn nv12_buffer(
        width: usize,
        height: usize,
        luma: u8,
        cb: u8,
        cr: u8,
    ) -> PixelBuffer {
        let buffer = create(
            width,
            height,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        );
        // SAFETY: a live pixel buffer, locked for writing and unlocked below.
        let locked = unsafe { CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        assert_eq!(locked, 0);
        for (plane, rows) in [(0, height), (1, height / 2)] {
            let base = CVPixelBufferGetBaseAddressOfPlane(&buffer, plane).cast::<u8>();
            let stride = CVPixelBufferGetBytesPerRowOfPlane(&buffer, plane);
            for row in 0..rows {
                // SAFETY: the buffer is locked for writing, and each row is
                // `stride` bytes of its plane.
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(base.add(row * stride), width) };
                if plane == 0 {
                    bytes.fill(luma);
                } else {
                    for pair in bytes.chunks_mut(2) {
                        pair[0] = cb;
                        pair[1] = cr;
                    }
                }
            }
        }
        // SAFETY: as above.
        unsafe { CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        PixelBuffer::new(buffer)
    }

    /// A camera's pixel buffer comes out an NV12 frame of its size, every
    /// sample where it was, at the range its layout says.
    #[test]
    fn a_pixel_buffer_is_copied_into_nv12() {
        let buffer = nv12_buffer(64, 48, 90, 100, 200);
        let mut frame = ffmpeg::frame::Video::empty();
        buffer
            .copy_to(&mut frame, ffmpeg::format::Pixel::NV12)
            .expect("NV12 copies");
        assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!((frame.width(), frame.height()), (64, 48));
        assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
        assert_eq!(
            frame.data(0)[frame.stride(0) * 47 + 63],
            90,
            "the last luma"
        );
        let chroma = frame.stride(1) * 23;
        assert_eq!(&frame.data(1)[chroma..chroma + 4], &[100, 200, 100, 200]);
    }

    /// A screen's pixel buffer comes out a full-range RGB BGRA frame of its
    /// size, every pixel where it was — and is refused where NV12 is wanted.
    #[test]
    fn a_pixel_buffer_is_copied_into_bgra() {
        let buffer = bgra_buffer(66, 40, [10, 20, 30, 255]);
        let mut frame = ffmpeg::frame::Video::empty();
        assert_eq!(
            buffer.copy_to(&mut frame, ffmpeg::format::Pixel::NV12),
            Err(PixelBufferError::Unexpected(kCVPixelFormatType_32BGRA))
        );
        buffer
            .copy_to(&mut frame, ffmpeg::format::Pixel::BGRA)
            .expect("BGRA copies");
        assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
        assert_eq!((frame.width(), frame.height()), (66, 40));
        assert_eq!(frame.color_space(), ffmpeg::color::Space::RGB);
        assert_eq!(frame.color_range(), ffmpeg::color::Range::JPEG);
        let last = frame.stride(0) * 39 + 65 * 4;
        assert_eq!(&frame.data(0)[last..last + 4], &[10, 20, 30, 255]);
    }

    /// The pixel buffer itself goes on as a VideoToolbox frame that reads
    /// back as the picture it holds, and that the media engine's encoder
    /// takes.
    #[cfg(feature = "videotoolbox")]
    #[test]
    fn a_pixel_buffer_is_a_videotoolbox_frame_as_it_is() {
        use crate::element::Sink;
        use crate::elements::{
            VideoToolboxCodec, VideoToolboxDownload, VideoToolboxEncoder,
            VideoToolboxEncoderOptions,
        };
        let Some(device) = crate::test_support::try_videotoolbox_device() else {
            return;
        };
        // SAFETY: the device's own live context.
        let frames = unsafe {
            crate::platform::macos::videotoolbox::create_frames_ctx(
                &device.retain(),
                ffmpeg::format::Pixel::NV12,
                64,
                48,
            )
        }
        .expect("a frames context");
        let mut frame = nv12_buffer(64, 48, 90, 100, 200)
            .into_videotoolbox_frame(&frames)
            .expect("a VideoToolbox frame");
        assert_eq!(frame.format(), ffmpeg::format::Pixel::VIDEOTOOLBOX);
        frame.set_pts(Some(0));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        let frame = crate::buffer::MediaBuffer::video(frame);

        let mut download = VideoToolboxDownload::new("read-back");
        let back =
            crate::test_support::one_frame(&mut download, frame.clone()).expect("it reads back");
        assert_eq!(back.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(back.data(0)[back.stride(0) * 47 + 63], 90);
        assert_eq!(&back.data(1)[..4], &[100, 200, 100, 200]);

        let _session = crate::test_support::encoder_session();
        let mut encoder = VideoToolboxEncoder::new(
            "encoder",
            &device,
            VideoToolboxEncoderOptions {
                codec: VideoToolboxCodec::H264,
                format: crate::elements::VideoToolboxFrameFormat::Nv12,
                width: 64,
                height: 48,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 1_000_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("an encoder");
        let packets = crate::test_support::capture(&mut encoder);
        encoder.consume(frame).expect("the encoder takes it");
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("eos");
        assert!(!packets.lock().unwrap().is_empty(), "and encodes it");
    }
}
