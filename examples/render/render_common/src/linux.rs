//! What only Linux needs besides: fitting a software decode to
//! `VulkanWindowRenderer`'s input.

/// What a software decode of `params` needs in front of `renderer`: nothing
/// where it decodes to a layout the renderer draws — YUV420P, as most
/// streams do — and a scaler to YUV420P at the stream's own size where it
/// does not, a 10-bit or a 4:4:4 one.
///
/// Asked of the renderer's own input contract rather than written out
/// here, so what it draws is said in one place: the library.
pub fn to_drawable(
    params: &media_pp::ffmpeg::codec::Parameters,
    renderer: &media_pp::elements::VulkanWindowRenderer,
) -> media_pp::Result<Option<media_pp::elements::SwScaler>> {
    use media_pp::{
        contract::{
            MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet, PortContract,
            check_link,
        },
        element::Sink,
        elements::SwScaler,
        ffmpeg,
    };

    let decoder = ffmpeg::codec::context::Context::from_parameters(params.clone())?
        .decoder()
        .video()?;
    let decoded = OutputContract::Fixed(
        PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
            .with_layouts(PixelLayoutSet::of(PixelLayout::of(decoder.format()))),
    );
    if !check_link(&decoded, &renderer.input_contract()).is_refused() {
        return Ok(None);
    }
    println!("decodes to {:?}; converting to YUV420P", decoder.format());
    Ok(Some(SwScaler::new(
        "to-yuv420p",
        ffmpeg::format::Pixel::YUV420P,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )))
}
