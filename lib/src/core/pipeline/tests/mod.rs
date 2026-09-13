use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use super::*;
use ffmpeg_next::{self as ffmpeg, Rescale};

use crate::contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract};
use crate::elements::{
    FileDemuxer, Pacer, SwDecoder, TeeBuilder, TestAudioOptions, TestAudioSource, TestVideoOptions,
    TestVideoSource,
};
use crate::graph::GraphError;
use crate::test_support::try_test_video;
use crate::{
    control::{ControlReceiver, drain_control},
    element::{Source, SourceElement},
    pad::SrcPad,
};

mod contracts;
mod graph;
mod lifecycle;
mod seek;
mod stats;

struct BurstSource {
    pp_log: PpLog,
    pad: SrcPad,
    ready: Arc<AtomicBool>,
    buffers: usize,
}

impl Element for BurstSource {
    fn name(&self) -> Arc<str> {
        "burst".into()
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

impl Source for BurstSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for BurstSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        for _ in 0..self.buffers {
            self.pad
                .push(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))?;
        }
        self.ready.store(true, Ordering::Release);
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            thread::yield_now();
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

struct SlowEosSink {
    pp_log: PpLog,
    count: Arc<AtomicUsize>,
    saw_eos: Arc<AtomicBool>,
}

impl Element for SlowEosSink {
    fn name(&self) -> Arc<str> {
        "slow-eos".into()
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

impl Sink for SlowEosSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if buf.is_eos() {
            self.saw_eos.store(true, Ordering::Release);
        } else {
            thread::sleep(Duration::from_millis(5));
            self.count.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

struct NoOpSink {
    name: Arc<str>,
    pp_log: PpLog,
}
impl Element for NoOpSink {
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
impl Sink for NoOpSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

struct CountingSink {
    pp_log: PpLog,
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}
impl Element for CountingSink {
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
impl Sink for CountingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // Anything but `Eos` counts — covers `FileDemuxer`'s `Packet`s
        // (what every other test using this sink actually sends) and
        // `TestVideoSource`/`TestAudioSource`'s `Video`/`Audio` frames
        // (what `multi_source_pipeline_stops_every_source_from_one_stop_call`
        // sends) alike.
        if !buf.is_eos() {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}
