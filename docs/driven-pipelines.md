# Driven pipelines: many small pipelines on a few threads

Status: **accepted**, 2026-10-22, with the decisions in §8. Stage 1 of §7
is built — `DrivenPipeline`, its handle, `DrivenRunner`, and the test
sources drivable; §7 says what is left, and §4.8 where what was built
differs from what was first written here.

File references are relative to `lib/src/`.

## Contents

1. [The problem](#1-the-problem)
2. [How a pipeline runs today](#2-how-a-pipeline-runs-today)
3. [What was considered](#3-what-was-considered)
4. [The design](#4-the-design)
5. [The servers on driven pipelines](#5-the-servers-on-driven-pipelines)
6. [What this does not do](#6-what-this-does-not-do)
7. [Migration](#7-migration)
8. [Decisions](#8-decisions)

---

## 1. The problem

`RtspServer` and `WebRtcServer` each read a `Broadcast` themselves for every
player (`elements/driver/rtsp_server/session.rs`,
`elements/driver/webrtc/server/session.rs`): a `BroadcastReader`, a
`Timeline` and pacing of their own, inside the server, with nothing between
the broadcast and the network. Everything else in the crate moves media
through a pipeline: elements with pads, a bus, logs under each element's
identity, statistics, control that reaches every element the same way. The
servers are the one place media flows past all of that, and each server
carries its own copy of what `BroadcastSource` already does.

The obvious way to remove the exception — a pipeline for each player,
`BroadcastSource → (filters) → a sink of the player's session` — costs a
thread for every player, since a pipeline runs each source on a thread of
its own (§2). Three hundred players would be three hundred threads, each
woken for every packet; the servers now serve them on two to six.

What is wanted is the uniform model at the cost of the current one: **a
pipeline per player, run without a thread of its own**, so that a few
threads run hundreds of them.

## 2. How a pipeline runs today

- **A thread per source.** `Pipeline::run` (`core/pipeline/lifecycle.rs`)
  spawns one thread for each source, which calls `RawSource::run`. For a
  `Source` that is `SourceStage::ask` (`core/produce.rs`): take what the
  pipeline asks, hand on what is held back, then `produce` the next thing
  and push it down the chain.
- **The chain runs on the source's thread** up to the first `Queue`. Each
  `Queue` (`core/queue.rs`) is a thread boundary: everything after it runs
  on its worker thread. Elements that wait on the clock inside `consume` —
  `Pacer`, `VideoSynchronizer` — run behind a queue of their own
  (`RawSink::own_queue`).
- **Waiting blocks the thread.** A `Source` waits only through `Wait`
  (`Wait::until`, and inside the crate `Wait::recv`), which blocks on the
  clock or a channel until the pipeline has something for the thread.
- **Control is a message with an acknowledgement.** Each source and each
  queue worker has a control channel (`core/control.rs`). `Pipeline::pause`
  enqueues `Pause` on each and waits for each to acknowledge. A source takes
  it between two things it makes (`handle_request`), passes it down its
  chain (`RawSink::flow`), and stays blocked in `wait_out_pause` until
  `Resume`, `Stop` or a finish.
- **What restricts flow is state; what releases it is a message.**
  `PlaybackState` (`core/playback_state.rs`) is written by the pipeline
  before the message that announces a change. This rule, from
  `docs/stream-events.md`, stays.

Every one of these assumes a thread that may block. A pipeline without a
thread cannot block anywhere: not in `Wait`, not in a pause, not on a full
queue.

## 3. What was considered

**A. The server reads the broadcast itself** — what is built. Cheapest; one
reader, one timeline and the RTP of each player, on the server's threads.
The exception §1 is about.

**B. A pipeline per player, as pipelines run today.** Uniform, and works
now with a session sink. A thread per player, woken per packet: at three
hundred players of a 12 Mbit/s stream, some 330,000 context switches a
second besides the work.

**C. One pipeline per path, fanned out with a `Tee`.** One thread per path:
`BroadcastSource → (filters) → Tee → a sink per player`. Cheap, but a `Tee`
holds nothing: a player that joins waits for the next keyframe, losing the
instant start the broadcast gives each reader, and a slow player can only be
dropped from, not moved on to a keyframe. Both would have to be built into
the `Tee` or its sinks — the broadcast's work, again.

**D. Driven pipelines** — this document. A pipeline that never blocks and is
*polled*: its owner calls it when there is something to do, it does what it
can without waiting, and says when to call it next. A server's thread polls
the pipelines of its players. Uniform, a pipeline per player, and threads
only as many as the server has.

D is chosen because it is the only one that removes the exception without
giving up either the per-player reader (instant start, moving a slow
player on) or the thread count.

## 4. The design

### 4.1 One pipeline, two ways to run it

A pipeline is wired as it is today — `Pipeline::new`'s `wire` closure, the
same chain builder and elements — and then either **run**, as today, on
threads of its own, or **driven**:

```rust
let (mut session, ()) = DrivenPipeline::new("player-17", source, |source, ctx| {
    let branch = ctx.branch().to(sink)?;   // no queue: refused here
    ctx.attach(source, 0, branch)?;
    Ok(())
})?;

// The owner's loop, on whatever thread it has:
match session.poll(&mut cx) {               // a std::task::Context
    Driven::Pending { at } => { /* call again when woken, or at `at` */ }
    Driven::Finished => { /* done — what ended it is on the bus; drop it */ }
}
```

`DrivenPipeline` keeps what a pipeline is — its graph, bus, log identity
(`pipeline_id`), statistics, `PlaybackState` — and has no threads. It is
`Send`, so an owner may move it between threads between polls, but it is
only ever polled by one thread at a time (`&mut self`).

### 4.2 Polling

`poll` does, on the caller's thread:

1. what was asked of the pipeline since the last poll (§4.4);
2. what the source holds back for an output that can now take it — as
   `SourceStage::ask` does today with `Parking`;
3. `produce` the next thing and push it down the chain, synchronously, to
   the sinks — **repeatedly**, until the source answers that it has nothing
   now (`Produced::Nothing`), up to a budget of work per poll (a number of
   buffers, of the order of 32), so that one pipeline cannot keep the rest
   of its thread's waiting;
4. return `Pending { at }`: when it wants to be polled again whether woken
   or not — the earliest deadline anything in it asked for — or
   `Finished`, once its stream has ended or it was stopped.

A poll never blocks. A pipeline that stopped at the budget with more to do
answers `Pending { at: now }`.

### 4.3 Waiting without a thread: `Wait` in a driven pipeline

A `Source` already waits only through `Wait`. Driven, `Wait` does not block:

- `Wait::until(at)` answers `true` once `at` has come, and otherwise records
  `at` as the pipeline's next deadline and answers `false` — which every
  source already treats as "answer `Produced::Nothing`, and be asked again".
- `Wait::recv(rx)` (crate-private) answers with what is in the channel now,
  and otherwise `Received::LetGo`. A crossbeam channel cannot wake anyone,
  so this alone would leave the pipeline asleep: see the next point.
- `Wait::waker()` — new — gives the source the waker of the poll it is
  running in, for whatever it waits on that can wake: a
  `BroadcastReader` registers it through `poll_next`, which it already
  supports.

A source runs driven only where it says it can: `Source::drivable() -> bool`,
false by default, true for a source that waits only on the clock or on
something it hands the waker to. `DrivenPipeline::new` refuses any other
with a typed error. To start with: `BroadcastSource` (the waker, through
its reader) and `TestVideoSource`, `TestAudioSource` (the clock). A
`RawSource` written by hand, with its own loop, is never drivable.

### 4.4 Control without messages

A driven pipeline has no thread to send a message to: its owner polls it.
So control is applied **by the poll**, in the same order as today, through
the same functions:

- `pause`, `resume`, `stop`, `finish` and — for a seekable source —
  `seek` on a `DrivenPipeline` write `PlaybackState` first, as today, then
  queue the request and wake the pipeline. The next poll takes it exactly
  as `SourceStage::ask` takes one today (`handle_request`, `apply_one`,
  `RawSink::flow` down the chain, `pausing`/`resuming`/`stopping` hooks),
  so every element sees what it sees now, in the same order.
- **Paused** is a state, not a blocked thread: a paused driven pipeline's
  poll takes requests and returns `Pending { at: never }` without producing,
  until `Resume`, `Stop` or a finish — `wait_out_pause` as a state instead
  of a loop. `Wait::now` leaves the paused time out, as today.
- The calls do not wait for the request to be taken — there is no other
  thread to wait for, and the caller may be the very thread that polls.
  What is done is said on the bus as it is today, and `poll` answers
  `Finished` once a stop or the end has gone through.
- `Pipeline`'s waiting calls (`pause` returning only once every thread has
  paused, `seek` waiting for its preroll) have no driven counterpart.
  `seek` is applied on the next poll without a preroll wait: a driven
  pipeline's terminals are its own sinks, which a server reads as they go.

The rule that restrictions are state and releases are messages holds: the
state is written by the call, before anything else; the release is the
request the poll takes.

### 4.5 What may be in a driven pipeline

Checked when it is wired — `DrivenPipeline::new` fails with a typed error
naming the element — so that nothing in it can block a thread that is not
its own:

- **No `Queue`.** A queue is a thread. `ChainBuilder::queue` in a driven
  pipeline's wiring is an error.
- **No element with a queue of its own** — `RawSink::own_queue` is `Some`:
  `Pacer`, `VideoSynchronizer`.
- **A `Tee`** only with branches that have no queue.
- **Sinks and filters that do not wait.** This cannot be checked: an
  element's `consume` that blocks on a device or a socket would block the
  owner's thread. The rule is documented on `DrivenPipeline`, and a poll
  that takes longer than a bound (of the order of 20 ms) is logged at
  `Warn` with the pipeline's identity, so the element is found.
- **Heavy work belongs elsewhere.** Decoding and encoding are allowed — they
  do not wait — but run on the owner's thread, so a server's players on that
  thread wait for them. A transform for every player of a path belongs in a
  pipeline of its own that writes another `Broadcast` (§5.4).

Driven pipelines have one source. Several, as `PipelineBuilder` allows, can
follow later; nothing here needs them.

### 4.6 Failure and teardown

- A panic inside a poll — the source or anything on its chain — is caught,
  posted on the pipeline's bus as today's source thread does, and the
  pipeline is finished with its branches stopped; the owner's other
  pipelines go on.
- Dropping a `DrivenPipeline` stops it on the dropping thread: the same
  `Stop` cascade, the sources' and sinks' `stopping`, nothing to join.

### 4.7 Logging and statistics

Unchanged: the `run` record with the topology diagram when it is first
polled, elements' records under their identities, `Pipeline::stats`'s
counterpart. The thread an element's record names is the owner's thread,
which is the truth.

### 4.8 As built

Stage 1 settled four things this design left open or had otherwise:

- **Control is through a handle.** `DrivenPipeline::handle()` gives a
  `DrivenPipelineHandle` — cheap to clone, `Send + Sync`, with `pause`,
  `resume`, `stop`, `finish`, the bus, `stats` and `topology` — because the
  owner hands the pipeline itself to whatever polls it (`DrivenRunner`, an
  executor) and keeps only the handle. The handle keeps the bus and the
  control channel, never an element.
- **`Driven::Finished` carries nothing.** What ended a pipeline — its
  stream, a stop, a finish, a failure — is on its bus, as for a running
  pipeline; the error itself cannot be both posted and returned.
- **A drivable source keeps the time it waits for.** Driven, `Wait::until`
  answers `false` before its time and the source is asked again later, so
  `wait.until(wait.now() + tick)`, worked out anew at each ask, is never
  reached. `TestAudioSource` did that and made no sound driven; it keeps
  its next tick now, and `Source::drivable` says so. The driven conformance
  sequences found it.
- **The conformance sequences are a set of their own**
  (`core::pipeline::tests::conformance::driven`): a picture or a sound
  source, alone, behind a filter, or fanned out; three side by side,
  polled by a `DrivenRunner`, as a `Future` on an executor of their own, or
  by their owner's loop; pause, resume and waits in random order, ended by
  a stop, a finish or a drop. They run under the same variables as the
  threaded ones, and CI's `-- conformance` runs them.

## 5. The servers on driven pipelines

### 5.1 A sink for a session

`WebRtcTrackSink` writes to a `WebRtcPeer` today, through the peer's command
channel (`elements/driver/webrtc/track.rs`). It gains a second target: a
server session's **outbox** — what the session is to send, with when — which
the server's thread drains into the session's `str0m` connection after it
polls the session's pipeline. Same public type, same codec checks, `avcC`
to Annex B and parameter sets ahead of keyframes, which the server now does
in `server/annex_b.rs` and which then go away.

For `RtspServer`, an `RtspTrackSink` does the same into an RTSP session: its
packetizer and its TCP or UDP delivery.

### 5.2 What a session is

A session becomes its connection and a `DrivenPipeline`:

```text
server thread (one of N)
 ├─ session 17: str0m connection ← outbox ← [BroadcastSource → (filters) → WebRtcTrackSink ×tracks]
 ├─ session 18: …
 └─ the thread's loop: poll the pipelines that were woken or are due, then
    write each session's outbox into its connection, then the socket
```

The server's loop polls only the pipelines woken since the last turn
(their wakers mark them ready) or whose deadline has come, so idle players
cost nothing. `RtspServer`, whose connections are tokio tasks, polls a
session's pipeline from the session's task: `DrivenPipeline::poll` takes a
`std::task::Context`, so the task's own waker serves, and a deadline becomes
a `sleep_until`.

### 5.3 What moves where

| Today, in the server | With driven pipelines |
|---|---|
| a `BroadcastReader` per session | `BroadcastSource`, which has one |
| `Timeline` restamping per session | `BroadcastSource`, which does it |
| pacing what a UDP or WebRTC player catches up on at twice the speed | the session sink: due times in the outbox |
| `avcC` to Annex B, parameter sets (WebRTC) | `WebRtcTrackSink`, which already does it |
| which tracks a player is sent | the pipeline: a branch per track the player takes |

The servers keep connections, signalling, packetization and sockets — the
transport — and nothing of the media.

### 5.4 What a path is

A path stops being "a broadcast the server reads" and becomes "how to make
a player's pipeline":

```rust
handle.add_path("cam1", RtspPath::broadcast(&camera))?;            // BroadcastSource → sinks
handle.add_path("cam1/small", RtspPath::pipeline(|sinks| { … }))?; // the application's own
```

The default is the pipeline every player has today. The second form gives
the application the player's sinks to wire its own pipeline, still driven
— for the per-player filter the uniform model makes room for. A transform
every player of a path shares — a smaller picture, AAC made Opus for
browsers — stays one ordinary pipeline from one broadcast into another,
whose path the players then take.

## 6. What this does not do

- **It does not make a pipeline async.** `DrivenPipeline::poll` fits a
  `std::task` executor, tokio's among them, but elements stay synchronous:
  a `consume` runs to completion.
- **It does not drive every element.** Queues, pacers, a source with a loop
  of its own and an element that blocks stay with threaded pipelines.
- **It does not make the servers cheaper.** The aim is the same cost with
  the uniform model; §7 stage 5 measures it against the server reading the
  broadcast itself, and the stage does not land if a player costs much more.
- **Off-thread control is asynchronous.** A driven pipeline's control calls
  return before the request is taken; nothing waits for a pause to reach
  every element as `Pipeline::pause` does.

### 6.1 Left for later

Settled 2026-10-22: the first stages refuse queues, pacers and queued `Tee`
branches (§4.5). Two ways to lift that were discussed, and are to be
thought over again before any stage takes them up:

- **A queue as a task boundary.** In a driven pipeline a `Queue` would start
  no thread: it would split the pipeline into segments, each polled on its
  own — the upstream one answering `Pending` where the queue is full, woken
  when it has room, the downstream one `Pending` where it is empty, woken
  when something arrives. Handed to an executor as separate tasks
  (`spawn_with(|segment| …)`), the segments run in parallel on its threads,
  so a queue keeps its buffering, its backpressure and its parallelism with
  no thread of the pipeline's own. Control reaches each segment at its next
  poll, as it reaches the source's. Without this, a queue in a driven
  pipeline would bring an OS thread back, even in a pipeline spawned as a
  tokio task.
- **Pacing by deadline.** `Pacer` and `VideoSynchronizer` would hold a
  buffer and give the pipeline a deadline instead of sleeping in `consume`,
  as `Wait::until` does for a source — no queue of their own, no thread. The
  servers' catch-up pacing (§5.3) could then be such an element rather than
  code in the session sink.

## 7. Migration

Each stage lands on its own, with its tests, as the streaming stages did.

1. **Core** — done. `DrivenPipeline`: wiring and its checks (§4.5), `poll` (§4.2),
   `Wait` driven (§4.3), control by poll (§4.4), failure and teardown
   (§4.6). `Source::drivable`. The `Future`, the timer and `DrivenRunner`
   (§8, 6). The control conformance sequences gain
   driven shapes — a live source, and the broadcast shape, with pause,
   resume, finish and stop in random order — run pinned to two cores, alone
   and loaded, a few hundred times (`CONTRIBUTING.md`).
2. **Drivable sources.** `BroadcastSource` (the reader's waker).
   `TestVideoSource` and `TestAudioSource` (the clock) were made drivable
   in stage 1, which its tests needed.
3. **WebRTC.** `WebRtcTrackSink`'s outbox target; `WebRtcServer`'s sessions
   on driven pipelines, polled by the thread that owns them; the server's
   own reader, timeline, pacing and Annex B conversion removed; paths as
   §5.4. The server's tests, unchanged in what they assert, pass.
4. **RTSP.** `RtspTrackSink`; `RtspServer`'s sessions likewise, polled from
   their tasks.
5. **Measured.** `rtsp_load` and `webrtc_load` against the same figures as
   `docs/benchmarks/streaming`. A stage-3 or stage-4 change that costs a
   player more than a quarter again of the processor it costs now is
   reworked or reverted, not landed.

## 8. Decisions

Settled 2026-10-22:

1. **No seek in the first stages.** Every source the servers drive is live.
   A driven pipeline answers a seek with `PipelineError::NotSeekable`-like
   refusal — the same check `Pipeline::check_seek` makes — until a stage
   needs it.
2. **Names:** `DrivenPipeline`, `Driven::{Pending, Finished}`,
   `Source::drivable`.
3. **The work budget per poll** (32 buffers) and **the slow-poll warning**
   (20 ms) are constants, to be made options only if a caller needs them.
4. **Paths:** `add_path` keeps taking a plain `Broadcast`, as a convenience
   for the default pipeline; a path that wants its own names how its
   players' pipelines are made.
5. **Acceptance:** stage 5 rejects a change that costs a player more than a
   quarter again of the processor it costs now.
6. **Where a driven pipeline runs:** no tokio option in the core. Deadlines
   wake a pipeline through its waker too — one small timer thread for the
   process calls the waker of a pipeline whose deadline has come — so a
   `DrivenPipeline` is a plain `Future`: a tokio application
   `tokio::spawn`s it, with no adapter and no tokio dependency in the
   crate. For an application without an async runtime, and for the
   servers, the crate has a small executor of its own (`DrivenRunner`, N
   threads, polling the pipelines that were woken); `WebRtcServer`'s threads
   are one, and `RtspServer` awaits a session's pipeline in its task. Stage
   1 builds the `Future`, the timer and `DrivenRunner`.
7. **Queues, pacers and queued `Tee` branches are refused** in the first
   stages (§4.5); §6.1 has what is to be thought over before lifting that.
