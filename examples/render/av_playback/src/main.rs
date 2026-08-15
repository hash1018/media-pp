#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        io::{self, BufRead},
        sync::Arc,
        thread,
        time::Duration,
    };

    use ffmpeg_next::{Rational, codec::Parameters, media};
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{
            AudioResampler, FileDemuxer, SwDecoder, TeeBuilder, TeeHandle, VideoSynchronizer,
            WasapiRenderer, WasapiRendererOptions,
        },
        graph::BranchId,
        pipeline::Pipeline,
    };
    use render_common::D3d12GpuContext;
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
        raw_window_handle::{HasWindowHandle, RawWindowHandle},
        window::{Window, WindowId},
    };

    /// Starts with video only, then lets the terminal attach/detach a decoded
    /// WASAPI audio branch at runtime. VideoSynchronizer uses wall time while
    /// audio is absent and automatically hands scheduling to WASAPI's played-
    /// sample position while the branch is attached.
    ///
    ///     cargo run -p av_playback -- path/to/video-with-audio.mp4
    ///     audio on
    ///     audio off
    ///     pause
    ///     resume
    ///     seek 30
    ///     seek 1:15
    ///     q
    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: av_playback <video-with-audio.mp4>");
            return;
        };

        let event_loop = EventLoop::<PlaybackDone>::with_user_event()
            .build()
            .expect("failed to create event loop");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            path,
            proxy,
            window: None,
            playback: None,
        };
        event_loop.run_app(&mut app).expect("event loop failed");
    }

    struct PlaybackDone;

    struct App {
        path: String,
        proxy: EventLoopProxy<PlaybackDone>,
        window: Option<Window>,
        playback: Option<thread::JoinHandle<()>>,
    }

    impl ApplicationHandler<PlaybackDone> for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let window = event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("media-pp A/V playback")
                        .with_inner_size(LogicalSize::new(1280, 720))
                        .with_resizable(false),
                )
                .expect("failed to create window");
            let hwnd = match window
                .window_handle()
                .expect("failed to get window handle")
                .as_raw()
            {
                RawWindowHandle::Win32(handle) => handle.hwnd.get(),
                _ => panic!("av_playback only supports Windows"),
            };
            let size = window.inner_size();
            let path = self.path.clone();
            let proxy = self.proxy.clone();
            self.playback = Some(thread::spawn(move || {
                if let Err(error) = play(&path, hwnd, size.width, size.height) {
                    eprintln!("playback failed: {error}");
                }
                let _ = proxy.send_event(PlaybackDone);
            }));
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            if self.window.as_ref().map(Window::id) == Some(window_id)
                && matches!(event, WindowEvent::CloseRequested)
            {
                event_loop.exit();
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
            event_loop.exit();
        }
    }

    fn play(path: &str, hwnd: isize, width: u32, height: u32) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = FileDemuxer::open("demux", path)?;
        let video = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let audio = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Audio)
            .ok_or_else(|| Error::Other("no audio stream in file".into()))?;
        let video_params = source
            .stream_parameters(video.index)
            .ok_or_else(|| Error::Other("video stream disappeared".into()))?;
        let video_time_base = source
            .stream_time_base(video.index)
            .ok_or_else(|| Error::Other("video stream disappeared".into()))?;
        let audio_params = source
            .stream_parameters(audio.index)
            .ok_or_else(|| Error::Other("audio stream disappeared".into()))?;
        let audio_time_base = source
            .stream_time_base(audio.index)
            .ok_or_else(|| Error::Other("audio stream disappeared".into()))?;

        let gpu = D3d12GpuContext::new().map_err(|error| Error::Other(format!("{error:?}")))?;
        let mut audio_tee_handle = None;

        let pipeline = Pipeline::new("av-playback", source, |source, context| {
            let video_branch = context
                .branch()
                .pipe(SwDecoder::new("video-decoder", video_params)?)
                .queue("video-frames", 32)
                .pipe(VideoSynchronizer::new(
                    "video-sync",
                    video_time_base,
                    context.playback_clock.clone(),
                )?)
                .to(Box::new(
                    render_common::d3d12_window_renderer(
                        "video-renderer",
                        &gpu,
                        hwnd,
                        width,
                        height,
                    )
                    .map_err(|error| Error::Other(format!("{error:?}")))?,
                ))?;
            context.attach(source, video.index, video_branch)?;

            // Keep a stable insertion point on the demuxer's audio pad. With no
            // branches attached the Tee cheaply drops packets, so playback starts
            // video-only without decoding audio.
            let (audio_tee, handle) =
                TeeBuilder::new("audio-tee", context.clone()).build_dynamic()?;
            context.attach(source, audio.index, audio_tee)?;
            audio_tee_handle = Some(handle);
            Ok(())
        })?;
        let audio_tee_handle =
            audio_tee_handle.ok_or_else(|| Error::Other("audio Tee was not initialized".into()))?;

        pipeline.run();
        {
            let pipeline = pipeline.clone();
            thread::spawn(move || {
                read_commands(pipeline, audio_tee_handle, audio_params, audio_time_base);
            });
        }

        for event in pipeline.bus().iter() {
            match event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => {
                    eprintln!("[{name}] error: {error}");
                }
                BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer"),
                BusEvent::Seeked {
                    name,
                    requested,
                    landed,
                    ..
                } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
            }
        }
        Ok(())
    }

    fn read_commands(
        pipeline: Arc<Pipeline>,
        audio_tee: TeeHandle,
        audio_params: Parameters,
        audio_time_base: Rational,
    ) {
        let mut audio_branch = None;
        print_help();

        for line in io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let command = line.trim().to_ascii_lowercase();
            let words = command.split_whitespace().collect::<Vec<_>>();

            match words.as_slice() {
                [] => {}
                ["audio", "on"] => {
                    if audio_branch.is_some() {
                        println!("audio is already on");
                        continue;
                    }
                    match attach_audio(&pipeline, &audio_tee, &audio_params, audio_time_base) {
                        Ok(branch_id) => audio_branch = Some(branch_id),
                        Err(error) => eprintln!("could not enable audio: {error}"),
                    }
                }
                ["audio", "off"] => {
                    let Some(branch_id) = audio_branch else {
                        println!("audio is already off");
                        continue;
                    };
                    match audio_tee.detach(branch_id) {
                        Ok(()) => {
                            audio_branch = None;
                            println!("audio off; video returned to wall-clock pacing");
                        }
                        Err(error) => eprintln!("could not disable audio: {error}"),
                    }
                }
                ["pause"] => {
                    pipeline.pause();
                    println!("paused");
                }
                ["resume"] => {
                    pipeline.resume();
                    println!("resumed");
                }
                ["seek", target] => match parse_timestamp(target) {
                    Some(target) => {
                        let clock = if audio_branch.is_some() {
                            "audio master"
                        } else {
                            "wall clock"
                        };
                        println!("seeking to {target:.2?} ({clock})...");
                        pipeline.seek(target);
                    }
                    None => eprintln!(
                        "could not parse {target:?}; use seconds (`seek 30`) or mm:ss (`seek 1:15`)"
                    ),
                },
                ["help"] => print_help(),
                ["q"] | ["quit"] => {
                    pipeline.stop();
                    break;
                }
                _ => eprintln!("unknown command; type `help` for the command list"),
            }
        }
    }

    fn attach_audio(
        pipeline: &Pipeline,
        audio_tee: &TeeHandle,
        audio_params: &Parameters,
        audio_time_base: Rational,
    ) -> media_pp::Result<BranchId> {
        let device = WasapiRenderer::list_devices()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default WASAPI render endpoint".into()))?;
        let device_name = device.name.clone();
        let (mut audio_renderer, output_format) =
            WasapiRenderer::open("speakers", WasapiRendererOptions { device })?;
        audio_renderer.bind_playback_clock_deferred(pipeline.playback_clock().clone())?;

        let branch = audio_tee
            .branch()
            .ok_or_else(|| Error::Other("audio Tee is no longer available".into()))?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new(
                "audio-resampler",
                output_format,
                audio_time_base,
            )?)
            .queue("audio-output", 8)
            .to(Box::new(audio_renderer))?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }

    fn print_help() {
        println!("commands:");
        println!("  audio on          attach the default WASAPI output");
        println!("  audio off         detach audio and keep video playing");
        println!("  pause             pause playback");
        println!("  resume            resume playback");
        println!("  seek <seconds>    seek, for example `seek 30` or `seek 1:15`");
        println!("  help              print this help");
        println!("  q                 stop playback");
    }

    /// `"90"` (plain seconds) or `"1:30"` (mm:ss) -> `Duration`.
    fn parse_timestamp(value: &str) -> Option<Duration> {
        let seconds = match value.split_once(':') {
            Some((minutes, seconds)) => {
                minutes.parse::<f64>().ok()? * 60.0 + seconds.parse::<f64>().ok()?
            }
            None => value.parse::<f64>().ok()?,
        };
        if seconds.is_finite() && seconds >= 0.0 {
            Some(Duration::from_secs_f64(seconds))
        } else {
            None
        }
    }
}
