//! Elements of your own, one of each kind, in a pipeline with nothing else
//! in it: `Ramp -> Invert -> Brightness`.
//!
//! - `Ramp` is a [`Source`]: asked for the next thing, it makes a grey
//!   picture whose brightness climbs from one to the next, thirty a second,
//!   waiting only through the `Wait` it is handed — so a pause or a stop
//!   never waits on it — and ends its stream after the count.
//! - `Invert` is a [`Filter`]: each picture becomes its negative. It makes
//!   a new frame rather than changing the one it was handed, which may
//!   still be read elsewhere, and carries its timing and colour across.
//! - `Brightness` is a [`Sink`]: it reads each picture's average luma and
//!   keeps it where `main` can read it once the pipeline is done.
//!
//! Each declares what it takes and hands on, so a wrong link is refused
//! before the pipeline starts. None of them sees a control message: the
//! framework runs the source's loop, hands the end of the stream on after
//! the filter's `drain`, and holds the terminal while paused. `main` checks
//! that every picture arrived, in order, inverted.
//!
//!     cargo run -p custom_element -- [pictures]
//!
//! [`Source`]: media_pp::element::Source
//! [`Filter`]: media_pp::element::Filter
//! [`Sink`]: media_pp::element::Sink

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use media_pp::{
        Error, Result,
        buffer::{MediaBuffer, set_time_base, time_base},
        bus::BusEvent,
        contract::{
            InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
            PortContract,
        },
        element::{
            Element, ElementType, Filter, Output, Produced, Sink, Source, Wait, element_pp_log,
        },
        ffmpeg::{self, format::Pixel, frame::Video},
        pipeline::Pipeline,
        pp_log::PpLog,
    };

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const RATE: i32 = 30;

    /// What every element here takes or hands on: 8-bit 4:2:0 pictures in
    /// three planes, in system memory.
    const PICTURES: PortContract = PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
        .with_layouts(PixelLayoutSet::of(PixelLayout::Yuv420p));

    /// Picture `index`'s brightness, as `Ramp` makes it: studio-range
    /// luma, 16 to 235, across the stream.
    fn level(index: u64, count: u64) -> u8 {
        (16 + index * 219 / count.saturating_sub(1).max(1)) as u8
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let count = match std::env::args().nth(1) {
            None => 90,
            Some(arg) => match arg.parse::<u64>() {
                Ok(count) if count > 0 => count,
                _ => {
                    eprintln!("usage: custom_element [pictures]");
                    std::process::exit(1);
                }
            },
        };

        let (brightness, seen) = Brightness::new("brightness");
        let (pipeline, ()) = Pipeline::new("custom", Ramp::new("ramp", count), |ramp, ctx| {
            let branch = ctx.branch().pipe(Invert::new("invert")).to(brightness)?;
            ctx.attach(ramp, 0, branch)?;
            Ok(())
        })?;
        pipeline.run()?;
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        let seen = seen.lock().unwrap();
        println!("pictures: {} of {count}", seen.len());
        for (index, &(pts, luma)) in seen.iter().enumerate() {
            let expected = 255 - level(index as u64, count);
            if pts != Some(index as i64) || luma.round() as u8 != expected {
                return Err(Error::Other(format!(
                    "picture {index}: pts {pts:?}, luma {luma:.1}, expected {expected}"
                )));
            }
        }
        if seen.len() as u64 != count {
            return Err(Error::Other(format!(
                "{} pictures arrived of {count}",
                seen.len()
            )));
        }
        println!("every picture arrived in order, inverted");
        Ok(())
    }

    /// A source: `count` grey pictures at [`RATE`] a second.
    struct Ramp {
        name: Arc<str>,
        pp_log: PpLog,
        count: u64,
        next: u64,
        /// When the first picture was due, on the clock [`Wait::now`]
        /// reads, which stands still while the pipeline is paused — so a
        /// resume carries on where it left off rather than catching up.
        start: Option<Instant>,
    }

    impl Ramp {
        fn new(name: &str, count: u64) -> Self {
            Self {
                name: name.into(),
                pp_log: element_pp_log(ElementType::Other, name, None),
                count,
                next: 0,
                start: None,
            }
        }
    }

    impl Element for Ramp {
        fn name(&self) -> Arc<str> {
            self.name.clone()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Source for Ramp {
        /// It makes pictures at a rate of its own, as a camera does.
        fn is_live(&self) -> bool {
            true
        }

        fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
            if self.next == self.count {
                return Ok(Produced::End);
            }
            let start = *self.start.get_or_insert_with(|| wait.now());
            let due = start + Duration::from_secs(self.next) / RATE as u32;
            if !wait.until(due) {
                // The pipeline has something for this thread; asked again
                // after.
                return Ok(Produced::Nothing);
            }

            let mut picture = Video::new(Pixel::YUV420P, WIDTH, HEIGHT);
            picture.data_mut(0).fill(level(self.next, self.count));
            picture.data_mut(1).fill(128);
            picture.data_mut(2).fill(128);
            picture.set_pts(Some(self.next as i64));
            set_time_base(&mut picture, ffmpeg::Rational(1, RATE));
            self.next += 1;
            // A source making pictures over and over would keep a pool of
            // its own to draw into again — see `MediaBuffer::video`.
            Ok(Produced::Buffer(MediaBuffer::video(picture)))
        }

        fn output_contract(&self) -> OutputContract {
            OutputContract::Fixed(PICTURES)
        }
    }

    /// A filter: each picture's negative.
    struct Invert {
        name: Arc<str>,
        pp_log: PpLog,
    }

    impl Invert {
        fn new(name: &str) -> Self {
            Self {
                name: name.into(),
                pp_log: element_pp_log(ElementType::Other, name, None),
            }
        }
    }

    impl Element for Invert {
        fn name(&self) -> Arc<str> {
            self.name.clone()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Filter for Invert {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
            // The contract keeps anything else from being linked here, but
            // a buffer is still checked before it is read.
            let MediaBuffer::Video(picture) = &buf else {
                return Err(Error::Other(format!(
                    "{} takes pictures, got {}",
                    self.name,
                    buf.kind()
                )));
            };
            if picture.format() != Pixel::YUV420P {
                return Err(Error::Other(format!(
                    "{} takes YUV420P, got {:?}",
                    self.name,
                    picture.format()
                )));
            }

            let mut negative = Video::new(Pixel::YUV420P, picture.width(), picture.height());
            for plane in 0..3 {
                let (from, to) = (picture.stride(plane), negative.stride(plane));
                let (width, rows) = (
                    picture.plane_width(plane) as usize,
                    picture.plane_height(plane) as usize,
                );
                let source = picture.data(plane);
                let destination = negative.data_mut(plane);
                for row in 0..rows {
                    let line = &source[row * from..row * from + width];
                    let made = &mut destination[row * to..row * to + width];
                    for (made, &sample) in made.iter_mut().zip(line) {
                        *made = 255 - sample;
                    }
                }
            }
            // What the picture says about itself goes on with it.
            negative.set_pts(picture.pts());
            if let Some(unit) = time_base(picture) {
                set_time_base(&mut negative, unit);
            }
            negative.set_color_space(picture.color_space());
            negative.set_color_range(picture.color_range());
            out.push(MediaBuffer::video(negative));
            Ok(())
        }

        fn input_contract(&self) -> InputContract {
            InputContract::Fixed(PICTURES)
        }

        fn output_contract(&self) -> OutputContract {
            OutputContract::Fixed(PICTURES)
        }
    }

    /// Each picture's PTS and average luma, in the order they arrived.
    type Seen = Arc<Mutex<Vec<(Option<i64>, f64)>>>;

    /// A terminal: each picture's average luma.
    struct Brightness {
        name: Arc<str>,
        pp_log: PpLog,
        seen: Seen,
    }

    impl Brightness {
        fn new(name: &str) -> (Self, Seen) {
            let seen = Seen::default();
            let sink = Self {
                name: name.into(),
                pp_log: element_pp_log(ElementType::Other, name, None),
                seen: seen.clone(),
            };
            (sink, seen)
        }
    }

    impl Element for Brightness {
        fn name(&self) -> Arc<str> {
            self.name.clone()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for Brightness {
        fn render(&mut self, buf: MediaBuffer) -> Result<()> {
            let MediaBuffer::Video(picture) = &buf else {
                return Err(Error::Other(format!(
                    "{} takes pictures, got {}",
                    self.name,
                    buf.kind()
                )));
            };
            let (stride, width, height) = (
                picture.stride(0),
                picture.width() as usize,
                picture.height() as usize,
            );
            let luma = picture.data(0);
            let total: u64 = (0..height)
                .flat_map(|row| &luma[row * stride..row * stride + width])
                .map(|&sample| u64::from(sample))
                .sum();
            let mean = total as f64 / (width * height) as f64;
            self.seen.lock().unwrap().push((picture.pts(), mean));
            Ok(())
        }

        /// A seek starts the stream again: what was seen before it is no
        /// longer what the pipeline shows.
        fn reset(&mut self) -> Result<()> {
            self.seen.lock().unwrap().clear();
            Ok(())
        }

        /// A stop does what `reset` does unless told otherwise; this one
        /// keeps what was seen, for `main` to read once it has stopped.
        fn stopping(&mut self) -> Result<()> {
            Ok(())
        }

        fn input_contract(&self) -> InputContract {
            InputContract::Fixed(PICTURES)
        }
    }
}
