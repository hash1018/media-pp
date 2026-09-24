//! On Linux the library's own `VulkanWindowRenderer` opens and draws the
//! window, so what is left here is ending the work when it is closed.

use std::{sync::Arc, thread};

use media_pp::elements::{Key, WindowEvent, WindowEvents};

use crate::Shutdown;

/// A [`Shutdown`] that closing any of `windows`, or pressing Escape in one,
/// sets off: whatever has been published by then is stopped.
///
/// Each window is watched on a thread of its own, which ends when its window
/// does — when the renderer that opened it is dropped — so nothing here has
/// to be joined. Stopping happens on that thread, not the window's own, so a
/// renderer waiting for its window never waits on the stop.
pub fn stop_on_close(windows: impl IntoIterator<Item = WindowEvents>) -> Arc<Shutdown> {
    let shutdown = Arc::new(Shutdown::default());
    for window in windows {
        let shutdown = Arc::clone(&shutdown);
        thread::spawn(move || {
            while let Some(event) = window.recv() {
                if matches!(event, WindowEvent::Closed | WindowEvent::Key(Key::Escape)) {
                    for pipeline in shutdown.request() {
                        pipeline.stop();
                    }
                }
            }
        });
    }
    shutdown
}

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
