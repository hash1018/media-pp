//! Polling driven pipelines on a few threads — see [`DrivenRunner`].

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context as TaskContext, Wake, Waker},
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::error::{Result, ThreadSpawnError};

use super::DrivenPipeline;

/// Polls [`DrivenPipeline`]s on a few threads of its own — for an
/// application with no async executor, and for a server with a pipeline
/// for each of its players.
///
/// Each pipeline goes to the thread with the fewest, and is polled there
/// whenever it is woken — by a request through its handle, by what its
/// source waits on, or by the time it asked for — until it answers
/// [`Driven::Finished`](super::Driven::Finished); then it is dropped. Its
/// [`DrivenPipelineHandle`](super::DrivenPipelineHandle), taken before it
/// is handed over, is how it is controlled and heard from after.
///
/// Dropping the runner stops every pipeline it still has, on its thread,
/// and joins the threads.
pub struct DrivenRunner {
    lanes: Vec<Lane>,
}

/// One of the runner's threads.
struct Lane {
    tx: Sender<Message>,
    /// How many pipelines it has.
    live: Arc<AtomicUsize>,
    thread: Option<JoinHandle<()>>,
}

enum Message {
    Add(DrivenPipeline),
    Ready(u64),
    Stop,
}

impl DrivenRunner {
    /// Starts `threads` threads — at least one — with no pipelines yet.
    ///
    /// Fails if a thread cannot be started; those already started are
    /// stopped and joined first.
    pub fn new(threads: usize) -> Result<Self> {
        let mut runner = Self { lanes: Vec::new() };
        for index in 0..threads.max(1) {
            let (tx, rx) = unbounded();
            let live = Arc::new(AtomicUsize::new(0));
            let name = format!("media-pp:driven-{index}");
            let thread = thread::Builder::new().name(name.clone()).spawn({
                let (tx, live) = (tx.clone(), Arc::clone(&live));
                move || run(&rx, &tx, &live)
            });
            match thread {
                Ok(thread) => runner.lanes.push(Lane {
                    tx,
                    live,
                    thread: Some(thread),
                }),
                // `runner` is dropped here, which stops and joins the rest.
                Err(error) => return Err(ThreadSpawnError::new(name, error).into()),
            }
        }
        Ok(runner)
    }

    /// Hands `pipeline` to the thread with the fewest, which polls it from
    /// now on.
    pub fn spawn(&self, pipeline: DrivenPipeline) {
        let Some(lane) = self
            .lanes
            .iter()
            .min_by_key(|lane| lane.live.load(Ordering::Acquire))
        else {
            return;
        };
        lane.live.fetch_add(1, Ordering::AcqRel);
        // A lane ends only when the runner is dropped, so this is never
        // refused; if it were, the pipeline would be dropped, and stopped.
        let _ = lane.tx.send(Message::Add(pipeline));
    }

    /// How many pipelines it is polling: handed to it and not yet
    /// finished.
    pub fn pipelines(&self) -> usize {
        self.lanes
            .iter()
            .map(|lane| lane.live.load(Ordering::Acquire))
            .sum()
    }
}

impl Drop for DrivenRunner {
    fn drop(&mut self) {
        for lane in &self.lanes {
            let _ = lane.tx.send(Message::Stop);
        }
        let current = thread::current().id();
        for lane in &mut self.lanes {
            if let Some(thread) = lane.thread.take()
                && thread.thread().id() != current
            {
                let _ = thread.join();
            }
        }
    }
}

/// A pipeline on a lane, and the waker its polls are handed.
struct Task {
    pipeline: DrivenPipeline,
    wake: Arc<TaskWake>,
    waker: Waker,
}

/// Puts its pipeline back in its lane's line to be polled — once, however
/// often it is woken before it is.
struct TaskWake {
    id: u64,
    queued: AtomicBool,
    lane: Sender<Message>,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            let _ = self.lane.send(Message::Ready(self.id));
        }
    }
}

/// A lane's thread: polls each pipeline as it arrives and whenever it is
/// woken, until the runner is dropped — and then drops, and so stops,
/// every pipeline it still has.
fn run(rx: &Receiver<Message>, tx: &Sender<Message>, live: &AtomicUsize) {
    let mut tasks = HashMap::<u64, Task>::new();
    let mut next = 0;
    while let Ok(message) = rx.recv() {
        let id = match message {
            Message::Add(pipeline) => {
                let id = next;
                next += 1;
                let wake = Arc::new(TaskWake {
                    id,
                    queued: AtomicBool::new(false),
                    lane: tx.clone(),
                });
                let waker = Waker::from(Arc::clone(&wake));
                tasks.insert(
                    id,
                    Task {
                        pipeline,
                        wake,
                        waker,
                    },
                );
                id
            }
            Message::Ready(id) => id,
            Message::Stop => break,
        };
        let Some(task) = tasks.get_mut(&id) else {
            continue;
        };
        // Taken off the line before the poll, so a wake during it puts the
        // pipeline back on.
        task.wake.queued.store(false, Ordering::Release);
        let mut cx = TaskContext::from_waker(&task.waker);
        if Pin::new(&mut task.pipeline).poll(&mut cx).is_ready() {
            tasks.remove(&id);
            live.fetch_sub(1, Ordering::AcqRel);
        }
    }
    let left = tasks.len();
    drop(tasks);
    live.fetch_sub(left, Ordering::AcqRel);
}
