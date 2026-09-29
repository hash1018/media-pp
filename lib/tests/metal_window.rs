//! `MetalWindowRenderer` and `VideoWindow` against real windows on macOS.
//!
//! AppKit makes windows only on the main thread's event loop, which the test
//! harness does not run: its tests run on threads of their own while the
//! main thread waits for them. So this has a `main` of its own, which runs
//! the checks inside `run_with_windows` as a program would.
//!
//! Where this program may record the screen — run as the child of an
//! application the user has allowed — each picture is also captured from
//! its window and its colour checked: built with `--features
//! screencapturekit-capture` and run with `MEDIA_PP_SCREEN_CHECK=1`. Only
//! then, since asking whether it may shows the user a prompt.

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    checks::run();
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod checks {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use media_pp::{
        buffer::MediaBuffer,
        element::Sink,
        elements::{
            AppSource, MetalWindowRenderer, MetalWindowRendererError, VideoToolboxDevice,
            VideoToolboxUpload, VideoWindow, WindowGone, WindowOptions, run_with_windows,
        },
        ffmpeg,
        pipeline::Pipeline,
    };

    pub(super) fn run() {
        no_main_loop_is_an_error();
        run_with_windows(|| {
            every_kind_of_frame_is_drawn();
            a_dropped_window_is_gone();
            a_video_window_opens();
            #[cfg(feature = "coreaudio-renderer")]
            a_file_plays_to_its_end();
        });
        println!("metal_window: ok");
    }

    fn options(title: &str) -> WindowOptions {
        WindowOptions {
            title: title.into(),
            width: 320,
            height: 240,
        }
    }

    /// Opening a window from a thread of its own while the main thread runs
    /// no event loop fails after a few seconds, rather than hanging — and
    /// the window it asked for is never made later.
    fn no_main_loop_is_an_error() {
        let started = Instant::now();
        let opened = thread::spawn(|| MetalWindowRenderer::open("nowhere", options("nowhere")))
            .join()
            .unwrap();
        assert!(
            matches!(opened, Err(MetalWindowRendererError::Window(_))),
            "no main loop to open a window on"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        println!("no_main_loop_is_an_error: ok");
    }

    /// A frame of `width` by `height` in `format`, every pixel of it made by
    /// `fill` from its planes.
    fn frame(
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
        fill: impl Fn(&mut ffmpeg::frame::Video),
    ) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        fill(&mut frame);
        frame.set_pts(Some(0));
        media_pp::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        frame
    }

    fn bgra(pixel: [u8; 4]) -> ffmpeg::frame::Video {
        frame(ffmpeg::format::Pixel::BGRA, 64, 48, |frame| {
            for chunk in frame.data_mut(0).chunks_mut(4) {
                chunk.copy_from_slice(&pixel);
            }
        })
    }

    fn nv12(y: u8, cb: u8, cr: u8) -> ffmpeg::frame::Video {
        frame(ffmpeg::format::Pixel::NV12, 64, 48, |frame| {
            frame.data_mut(0).fill(y);
            for pair in frame.data_mut(1).chunks_mut(2) {
                pair[0] = cb;
                pair[1] = cr;
            }
        })
    }

    fn yuv420p(y: u8, cb: u8, cr: u8) -> ffmpeg::frame::Video {
        frame(ffmpeg::format::Pixel::YUV420P, 64, 48, |frame| {
            frame.data_mut(0).fill(y);
            frame.data_mut(1).fill(cb);
            frame.data_mut(2).fill(cr);
        })
    }

    /// Every kind of frame the renderer takes is drawn in its colours: in
    /// system memory, NV12, YUV420P and BGRA; as VideoToolbox frames, NV12
    /// and BGRA. The colour is checked where the screen may be recorded.
    fn every_kind_of_frame_is_drawn() {
        let device = VideoToolboxDevice::new().expect("a VideoToolbox device");
        // BT.601 at limited range, as a picture of 48 rows with no colour
        // description is read: red, grey and blue.
        let red = [255, 0, 0];
        let scenes: [(&str, ffmpeg::frame::Video, bool, [u8; 3]); 5] = [
            ("system-bgra", bgra([0, 0, 255, 255]), false, red),
            ("system-nv12", nv12(126, 128, 128), false, [128, 128, 128]),
            ("system-yuv420p", yuv420p(81, 90, 240), false, red),
            ("videotoolbox-nv12", nv12(81, 90, 240), true, red),
            (
                "videotoolbox-bgra",
                bgra([255, 0, 0, 255]),
                true,
                [0, 0, 255],
            ),
        ];
        for (title, picture, upload, expected) in scenes {
            let title = format!("media-pp {title}");
            let (renderer, events) =
                MetalWindowRenderer::open(title.clone(), options(&title)).expect("a window");
            let control = renderer.window_control().expect("its own window");
            let (source, pusher) = AppSource::new("frames", 8);
            let device = device.clone();
            let (pipeline, ()) = Pipeline::new(title.clone(), source, move |source, ctx| {
                let branch = if upload {
                    ctx.branch()
                        .pipe(VideoToolboxUpload::new("upload", &device))
                        .to(renderer)?
                } else {
                    ctx.branch().to(renderer)?
                };
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("wiring");
            pipeline.run().expect("the pipeline runs");
            // A few, so the picture is on the screen by the time it is looked
            // at, whatever the display's refresh.
            for pts in 0..5 {
                let mut picture = picture.clone();
                picture.set_pts(Some(pts));
                pusher.push(MediaBuffer::video(picture)).expect("pushed");
                thread::sleep(Duration::from_millis(50));
            }
            thread::sleep(Duration::from_millis(300));
            if let Some(shown) = captured(&title) {
                assert!(
                    shown
                        .iter()
                        .zip(expected)
                        .all(|(got, want)| got.abs_diff(want) <= 24),
                    "{title}: shown as {shown:?}, expected about {expected:?}"
                );
                println!("{title}: shown as {shown:?}");
            }
            control.set_title("renamed").expect("the window is there");
            pipeline.stop();
            if let Some(error) = pipeline.bus().iter().find_map(|event| match event {
                media_pp::bus::BusEvent::Error { error, .. } => Some(error),
                _ => None,
            }) {
                panic!("{title}: {error}");
            }
            drop(pipeline);
            drop(events);
        }
        println!("every_kind_of_frame_is_drawn: ok");
    }

    /// Once its renderer is gone, a window's events end and its control
    /// says so.
    fn a_dropped_window_is_gone() {
        let (renderer, events) =
            MetalWindowRenderer::open("gone", options("media-pp gone")).expect("a window");
        let control = renderer.window_control().expect("its own window");
        control.set_fullscreen(false).expect("there");
        drop(renderer);
        assert_eq!(events.recv(), None, "no more events");
        assert_eq!(control.set_title("late"), Err(WindowGone));
        println!("a_dropped_window_is_gone: ok");
    }

    /// `VideoWindow` opens on Metal and draws a software frame.
    fn a_video_window_opens() {
        let (mut window, _events) =
            VideoWindow::open("video-window", options("media-pp video window"))
                .expect("a video window");
        window
            .consume(MediaBuffer::video(bgra([0, 255, 0, 255])))
            .expect("drawn");
        println!("a_video_window_opens: ok");
    }

    /// A file with a picture and sound — made here, two seconds of a test
    /// pattern and a tone — plays in a `Player` to its end, its picture
    /// decoded on VideoToolbox and drawn in the player's window.
    #[cfg(feature = "coreaudio-renderer")]
    fn a_file_plays_to_its_end() {
        use media_pp::{
            elements::{DecodePath, WindowOptions},
            player::{Player, PlayerEvent, PlayerOptions},
        };

        let path = std::env::temp_dir().join(format!("metal_window_{}.mp4", std::process::id()));
        make_media(&path, Duration::from_secs(2));
        let player = Player::open(
            &path,
            PlayerOptions {
                window: WindowOptions {
                    title: "media-pp player".into(),
                    width: 320,
                    height: 180,
                },
                ..PlayerOptions::default()
            },
        )
        .expect("the file opens");
        assert!(
            matches!(player.decoding(), Some(DecodePath::Hardware)),
            "decoded on VideoToolbox: {:?}",
            player.decoding()
        );
        player.play().expect("it plays");
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut ended = false;
        while Instant::now() < deadline {
            match player.next_event_timeout(Duration::from_millis(250)) {
                Some(PlayerEvent::Ended) => {
                    ended = true;
                    break;
                }
                Some(PlayerEvent::Error { name, error }) => panic!("[{name}] {error}"),
                _ => {}
            }
        }
        let position = player.position();
        drop(player);
        let _ = std::fs::remove_file(&path);
        assert!(ended, "the file played to its end");
        assert!(
            position.is_some_and(|position| position >= Duration::from_millis(1500)),
            "{position:?}"
        );
        println!("a_file_plays_to_its_end: ok");
    }

    /// Writes `length` of a test pattern and a tone, H.264 and AAC, to
    /// `path`.
    #[cfg(feature = "coreaudio-renderer")]
    fn make_media(path: &std::path::Path, length: Duration) {
        use media_pp::{
            bus::BusEvent,
            elements::{
                AudioCodec, FileMuxer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
                SwEncoderOptions, SwScaler, TestAudioOptions, TestAudioSource, TestVideoOptions,
                TestVideoSource, VideoCodec,
            },
        };

        let (width, height, rate) = (640, 360, ffmpeg::Rational::new(30, 1));
        let video = SwEncoder::new(
            "video",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: rate,
                bit_rate: 1_000_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("an H.264 encoder");
        let audio = SwAudioEncoder::new(
            "audio",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48_000,
                channels: 2,
                bit_rate: 128_000,
            },
        )
        .expect("an AAC encoder");
        let mut muxer = FileMuxer::create(path).expect("the file");
        let video_track = muxer.add_stream("video", &video).expect("a picture track");
        let audio_track = muxer.add_stream("audio", &audio).expect("a sound track");
        let mut sinks = muxer.open().expect("the file opens");
        let (video_sink, audio_sink) = (
            sinks.take(video_track).expect("its sink"),
            sinks.take(audio_track).expect("its sink"),
        );
        let pattern = TestVideoSource::new(
            "pattern",
            TestVideoOptions {
                width,
                height,
                frame_rate: rate,
            },
        );
        let (pictures, ()) = Pipeline::new("pictures", pattern, |source, ctx| {
            let scaler = SwScaler::new(
                "to-yuv",
                ffmpeg::format::Pixel::YUV420P,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let branch = ctx.branch().pipe(scaler).pipe(video).to(video_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        let tone = TestAudioSource::new(
            "tone",
            TestAudioOptions {
                sample_rate: 48_000,
                channels: 2,
                frequency: 440.0,
            },
        );
        let (sound, ()) = Pipeline::new("sound", tone, |source, ctx| {
            let branch = ctx.branch().pipe(audio).to(audio_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pictures.run().expect("runs");
        sound.run().expect("runs");
        thread::sleep(length);
        pictures.finish();
        sound.finish();
        for pipeline in [&pictures, &sound] {
            let finished = pipeline
                .bus()
                .iter()
                .any(|event| matches!(event, BusEvent::Finished));
            assert!(finished, "the file is written to its end");
        }
    }

    /// The colour at the middle of the window titled `title`, as captured
    /// from the screen — `None` where this program may not record it.
    #[cfg(feature = "screencapturekit-capture")]
    fn captured(title: &str) -> Option<[u8; 3]> {
        use media_pp::elements::{
            AppSink, ScreenCaptureKitOptions, ScreenCaptureKitSource, ScreenCaptureKitTarget,
        };
        use std::sync::{Arc, Mutex};

        if std::env::var_os("MEDIA_PP_SCREEN_CHECK").is_none() {
            println!("{title}: not checked on the screen (MEDIA_PP_SCREEN_CHECK is not set)");
            return None;
        }
        let windows = match ScreenCaptureKitSource::list_windows() {
            Ok(windows) => windows,
            Err(error) => {
                println!("{title}: not checked on the screen ({error})");
                return None;
            }
        };
        let window = windows
            .into_iter()
            .find(|window| window.title == title && window.pid == std::process::id())
            .expect("the window is listed");
        let (source, _) = ScreenCaptureKitSource::open(
            "look",
            ScreenCaptureKitOptions::new(ScreenCaptureKitTarget::Window(window.id)),
        )
        .expect("the window is captured");
        let seen: Arc<Mutex<Option<[u8; 3]>>> = Arc::default();
        let sink = AppSink::new("look", {
            let seen = Arc::clone(&seen);
            move |buffer| {
                if let MediaBuffer::Video(frame) = &buffer {
                    let (width, height) = (frame.width() as usize, frame.height() as usize);
                    let at = frame.stride(0) * (height / 2 + height / 8) + width / 2 * 4;
                    let [b, g, r] = [
                        frame.data(0)[at],
                        frame.data(0)[at + 1],
                        frame.data(0)[at + 2],
                    ];
                    *seen.lock().unwrap() = Some([r, g, b]);
                }
                Ok(())
            }
        });
        let (pipeline, ()) = Pipeline::new("look", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pipeline.run().expect("the capture runs");
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && seen.lock().unwrap().is_none() {
            thread::sleep(Duration::from_millis(50));
        }
        thread::sleep(Duration::from_millis(200));
        pipeline.stop();
        *seen.lock().unwrap()
    }

    #[cfg(not(feature = "screencapturekit-capture"))]
    fn captured(title: &str) -> Option<[u8; 3]> {
        println!("{title}: not checked on the screen (built without screencapturekit-capture)");
        None
    }
}
