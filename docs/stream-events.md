# Streams, flow and feedback: a redesign of how elements talk

Status: **accepted, in progress**, 2026-09-27. The design and the decisions
in [§7](#7-decisions) are settled; the version it ships in is decided once
the refactor is done. It describes where the pipeline core should end up and
a staged way to get there; each stage builds and passes on its own, and the
first two are worth doing whatever happens to the rest.

Evidence is from the code at `e0a6aef` and from every lifecycle-related `fix`
commit since 2026-06-01. File references are relative to `lib/src/`.

## Contents

1. [The problem](#1-the-problem)
2. [How it works today](#2-how-it-works-today)
3. [What the evidence says](#3-what-the-evidence-says)
4. [The design](#4-the-design)
5. [What this does not fix](#5-what-this-does-not-fix)
6. [Migration](#6-migration)
7. [Decisions](#7-decisions)

---

## 1. The problem

A pipeline carries one stream per pad, but what an element needs to know
about that stream arrives on four channels, each with its own ordering:

- the data itself (`MediaBuffer`, including `Eos`);
- a control cascade (`ControlMsg`), delivered between buffers and, at a
  `Queue`, overtaking whatever is backed up;
- a shared object any thread reads at any moment (`PlaybackState`);
- a timeline number that exists only on the thread that made a buffer and at
  the `Queue` that carries it to another.

Facts that belong at a point *in the data* travel on the channels that are
not ordered against it. Examples: "the new timeline starts here, and the seek
asked for this position"; "the preroll is over"; "from here the media runs
backwards"; "a new stretch starts here". So every element re-derives the order
from what reached it when.

The rules that make this work are real. They live in prose (`AGENTS.md`, the
core module docs) and are followed by hand in each of about 85 `control()`
implementations and 17 source loops. Each feature that adds a new fact adds a
new cross-channel rule: seek, preroll, rate, reverse, step, QoS, offline
rendering, looping backwards. The elements written before that rule are where
the next side effect shows up.

The fix is not in any one element. Three things are missing from the core:

- **a stream that carries its own facts**, so an element reads them where the
  data is;
- **a flow plane the pipeline drives directly**, so pausing, stopping and
  unblocking do not depend on a message making its way down through threads
  that may be busy or gone;
- **a default behaviour written once**, so an element implements its media
  work and nothing else.

---

## 2. How it works today

### 2.1 The channels

| Channel | Carries | Ordered against data? | Read by |
|---|---|---|---|
| Data push | `Packet`, `Video`, `Audio`, `Eos` | yes | every `Sink::consume` |
| Control cascade | `Pause`, `Resume`, `Stop`, `Flush`, `Preroll(ctx)`, `Seek(t)`, and `Finish` (a request, not a message) | no — handled between buffers; at a `Queue` it jumps the backlog | `Sink::control`, source loops via `drain_control` |
| `PlaybackState` | phase (see below), timeline number, interrupt and settle epochs, `backwards`, pictures shown per terminal, picture lateness | no — any thread, any time | about 15 call sites across sources, queues, tracers, decoders, pacers, the demuxer and `AudioTempo` |
| Timeline number | which seek's worth of media a buffer is from | implicitly (thread-local; handed over by `Queue`) | `Queue` alone |
| Direct calls | `ReversibleDecoder::begin_stretch` / `end_stretch` | inferred — `mark_stretch` (`core/pipeline/chain.rs`) guesses stretch boundaries from packet DTS going backwards | reversible decoders |

The phase is one of playing, paused (keeping the last preroll's holds),
prerolling, or stopped.

A control message travels as follows (`core/control.rs`,
`core/pipeline/lifecycle.rs`):

1. The pipeline writes the state first.
2. It enqueues the request on every source's channel, raises an interrupt,
   waits for every acknowledgement, then settles the interrupt.
3. Each source thread picks the request up between buffers and forwards it
   depth-first through the graph.
4. A `Queue` crosses the thread with a rendezvous. Its acknowledgement means
   the whole subtree below it has handled the message.
5. A `Tee` fans the message out to its branches.

Control that does not come from that broadcast follows none of its rules:

- the `Stop` after a source error (no phase change, no interrupt);
- `PipelineBridge`'s injected `Flush` (no interrupt, no new timeline);
- `SegmentedFileMuxer` using `Stop` to finalise a segment.

### 2.2 What a seek is

`Pipeline::seek` (`core/pipeline/seek.rs`):

1. `Pause`.
2. Raise an interrupt. Reset the playback clock's anchor. Write `backwards`,
   zero lateness and a new timeline number into the state. Broadcast `Flush`,
   then `Seek(target)`.
3. Enter the prerolling phase and broadcast `Preroll(ctx)`. The context holds
   the accurate-seek target, the step count and which terminals are silenced.
   Wait until every expected terminal reports a first sample (5 s timeout).
4. `Pause` again, always: ending a preroll by playing on directly let a queue
   hand a terminal data before it had been told to take it. Then `Resume` if
   it was playing.

`set_rate` builds a turn of direction from the same stages, and `step` builds
a step from them.

### 2.3 What elements do with it

A catalogue of every element's reactions (control, `Eos`, state reads, holds)
is summarised here.

**About 40 elements implement `control` as the same thing:** reset on
`Flush`/`Stop`, and ignore the rest. Among them:

- every per-frame transform with a repeat cache (scalers, converters,
  effects, keys, uploads, downloads, tone map);
- the audio filters;
- `ChangeGate`, `FrameRateLimiter`, `PauseGate`;
- decoders (which also reset QoS, the preroll gate and the stretch state).

Encoders deliberately do nothing. `TimestampOrigin` deliberately ignores
`Flush`. `CudaScaler` has no `control` at all, although its libavfilter graph
holds frames.

**Five separate hold-and-release mechanisms** do the same job:

| Element | Holds | Released by a message | Released by the next buffer | Dropped on |
|---|---|---|---|---|
| `Queue` | channel backlog | `Resume` / `Preroll`; the settle bell | — (waits on readiness: bell or 20 ms poll) | `Flush` |
| `Tee` | `owed`, per branch | `Resume`, `Preroll` | yes | `Flush`, `Stop` |
| `Pacer`, `VideoSynchronizer` | `pending` | **no** | yes; at once if an `Eos` is pending | `Flush`, `Stop` |
| `PrerollGate` (every decoder) | `candidate`, `after`; not ready while holding | **no** — only the phase changing | yes (the next decoded frame) or `Eos` | `Flush` |
| `FileDemuxer` | parked packets per pad | — | re-checked each loop and on a 1 ms poll | `Flush`, a new stretch or seek |

**Each element forms its own answer** where the protocol leaves room:

- The two audio renderers each keep a `paused` flag of their own instead of
  reading the phase, and they disagree. WASAPI drops a paused seek's preroll
  sample; PipeWire queues it and plays it on `Resume`.
- The Sw and D3D11 compositor inputs clear their latest frame on `Flush` and
  detach on a live `Eos`. The Vulkan and CUDA inputs do neither. (Since
  made one answer on every backend: `Flush` and a live `Eos` both clear the
  frame and keep the input, since a detached one could not be shown again
  after a seek back.)
- The playback clock is reset for a seek in four places (`reposition`, `step`,
  `Pacer` on `Seek`, the WASAPI renderer on `Flush`). `VideoSynchronizer`
  resets it nowhere.
- A control error on the source's own thread ends that thread. Behind a
  `Queue` the same error becomes a bus event. A `Tee` never returns one.
  (Since fixed: on the source's own thread it is a bus event too, and the
  source goes on.)
- `Rack` and `VideoEncodeBin` do not pass their contents' `ready_consume` on,
  and `Rack` does not pass `as_reversible` on. (Since fixed for
  `ready_consume`; no rack holds a decoder, so `as_reversible` waits for one
  that does.)
- `ReplayTrackSink::accepts_seek` answers true, against the trait's own
  documentation. (Since fixed: it refused seeks until b9b9ea9 moved the
  question to the wiring and lost its answer.)
- A `Finish` reaching `FileDemuxer` while it lingers at the end, paused, would
  push `Eos` twice. Only `Pipeline::finish` resuming first prevents it.
  (Since fixed where every source's would be: a pad hands on one `Eos` per
  stream.)

**The rules an element has to know**, collected from `AGENTS.md` and the
module docs. Each is enforced by hand.

- A source drains control with `drain_control` or `handle_request` and never
  applies a request itself (208af56).
- `Sink::control` is the element's own reaction; only `Queue`, `Tee` and bins
  forward.
- Restrictions go into the state before their message. Releases travel only
  as a message, passed on before any data. A preroll ends in a pause.
- Whatever an element holds back in a phase it hands on, in order, when it
  sees the phase is over — "from the next buffer as much as from the message
  behind it".
- `Eos` is forwarded, after stateful elements drain; `Stop` abandons.
- Wall-clock sources add `paused_for` back into their schedule.
- A `Queue` drops buffers from a timeline the pipeline has left. A buffer
  never carries a number itself (c0e76e8).

---

## 3. What the evidence says

Every `fix` commit since 2026-06-01 that touches seek, preroll, pause, stop,
EOS, flush, rate, reverse, step, loop, QoS, timeline, queue, tee or control
was read and classified by root cause: 61 in media-pp and 17 in obs-rs.

| Root cause | media-pp | obs-rs |
|---|---|---|
| A. Ordering between channels: something that had to be ordered against the data travelled beside it | 6 (11 counting secondary causes) | 1 |
| B. An element did not know a protocol rule | 9 (14) | 1 |
| C. `Eos` doubling as a lifecycle signal: a thread ending with its stream | 11 (15) | 4 |
| D. Stale data from a timeline the pipeline had left | 3 (6) | 3 |
| E. Fan-out and fan-in: one branch's state reaching its siblings | 7 (10) | 2 |
| G. Control-path liveness: a request that cannot reach a blocked or exited thread | 4 (7) | 0 |
| F. Local, nothing to do with the protocol | 21 | 6 |

Five conclusions shape the design.

- **Ordering is real, and most of it was fixed outside `fix` commits.** The
  control refactor of 2026-09-25 (bf1d2fe, ec4b8a4, 1ff9566) fixed a run of
  pure ordering bugs on its way:
  - a request reaching a `Pacer` before its interrupt, so it handed two
    samples to a terminal that takes one;
  - "playing" written to the state before the `Resume` that goes with it;
  - a `Tee` learning its hold was over from the next packet, and sending that
    packet ahead of the ones it kept.

  The rules the refactor left in `AGENTS.md` are a hand-written protocol for
  ordering two channels against each other. Other bugs have the same shape:
  - ae3feba: a pause drained queue backlogs into an interrupted `Pacer`;
  - 6d3c161: an `Eos` was stranded behind an interrupted wait;
  - 69b2f00 and ce40124: a renderer's command channel and data channel
    disagreed about a flush.
- **Elements need the timeline's facts next to the data.**
  - 2ca1ecf: a decoded buffer carries a pts and nothing to judge it against.
    So the accurate-seek gate had to live in the pacers, leaving audio
    branches ungated, and then in every decoder.
  - 92ef428: `AudioTempo` had to read the state to learn that a preroll was
    running.
- **Stale data is the weakest case.** Before the timeline numbers were built,
  the author measured whether they would catch anything. In thousands of
  pinned conformance sequences the check never fired, while every bug found
  was a stream *losing* data (fd9f756, c0e76e8). The real stale-data bugs were
  in buffers private to one element, which no queue-level check can see: the
  demuxer's parked packets (9aa9fdd) and PipeWire's internal queue (ce40124).
  This design argues from ordering, loss and duplicated behaviour, not from
  stale data.
- **The largest group is `Eos` as lifecycle (C).** A source thread returning
  at the end of its stream, or a `Queue` worker returning after forwarding
  `Eos`, ended both the stream and the only route control could take
  (9e53192, 06a8180, ec3fd88, d817bac). The cause here is that threads end
  with their stream and control is routed through them. In-band events alone
  do not fix it.
- **In-band events do not fix liveness (G).** A serialized event waits behind
  a blocked push exactly as data does. eb235e3 already shows a `Pause` and
  `Resume` stuck five seconds behind a blocking EOS drain. Whatever the
  design, something out of band must still reach a blocked thread.

About a third of the history (F) is local, and the demuxer's single-read-cursor
bugs (8e23512, 3e3e82b, 8d3ca52) are an interleaving problem. The design does
not claim those.

### 3.1 What the matrix found

Stage 0's conformance matrix (§6) found three bugs the hand-picked shapes
never had, in its first hours. Each is reproduced by a test in
`core/pipeline/tests/conformance.rs`, ignored and listed in its
`KNOWN_BROKEN` for as long as the bug is not fixed:

- **An accurate seek on a later lap of a looping file shows the keyframe
  before its target**
  (`an_accurate_seek_on_a_later_lap_shows_its_target`). A caller's seek is a
  position in the file; the source stays on the lap playback is on, so the
  pictures come stamped a lap further on; the decoders' preroll gate holds
  the target as the file position and lets the first picture through. The
  target and the data are in two different coordinates, and nothing next to
  the data says how they relate. Fixed as proposed: the `Segment` a seek
  begins carries its `position` in the media and its `start` on the
  timeline, from the source (`SeekableSource::on_timeline`, crate-private
  for now), and the gate puts the target on the timeline by them.
- **Packets fanned out and played backwards: the faster branch never
  steps** (`packets_fanned_out_backwards_step_on_in_every_branch`). The
  faster branch's decoder needs the rest of the stretch the source is
  reading; the slower branch, which has its sample, takes nothing more, and
  the source waited inside its push to that branch, where a `Tee` that
  keeps what comes for a prerolled branch could not let it go. A
  `StreamEvent::Stretch` from the source, as this document proposed, was
  tried first; the step still hung, since what the faster branch lacked
  was not the word that its stretch had ended but the stretch. Fixed on
  its own: a source reading backwards holds a packet until its pad can
  take it, as forwards it parks one, and the `Tee` answers ready once the
  branch has its sample. With that, the marker made no difference either
  way, so it is not in the design any more.
- **A paused seek to the end, with one-deep queues and sound, never
  prerolls the sound**
  (`a_paused_seek_to_the_end_prerolls_the_sound_through_one_deep_queues`).
  The demuxer parked packets per pad but pushed each pad's `Eos` with a
  wait for room, the picture's first, and the sound's `Eos` — which a seek
  past its last sample needs — waited behind it. The single read cursor,
  §5, which this design does not fix; so it was fixed on its own, each pad
  handed its end as it can take one.

---

## 4. The design

### 4.1 Three planes

| Plane | Direction | Ordered with data | Carries | Delivered by |
|---|---|---|---|---|
| **Stream** | downstream | yes — it *is* the data | buffers, and stream events: `Segment`, `Eos` | the pads, in order, through every element and thread boundary |
| **Flow** | pipeline → threads | no | `Pause`, `Resume`, `Stop`, flush start, and the interrupt that unblocks any wait | the pipeline, **directly to every thread-owning element** (each source, each `Queue` worker) and to every element with a device hook — never cascaded |
| **Feedback** | upstream | no | readiness (backpressure), picture lateness (QoS), "prerolled" | readiness through `ready_consume` as now; QoS and preroll completion through the pipeline |

The rule becomes: **what describes the stream travels in the stream; what
controls the flow comes from the pipeline to each thread directly; nothing
else crosses between elements.** That rule replaces every ordering rule in
§2.3.

### 4.2 The stream plane

```rust
pub enum Item {
    Buffer(MediaBuffer),          // Packet, Video, Audio — no Eos
    Event(StreamEvent),
}

pub enum StreamEvent {
    /// A run of buffers on one timeline begins; everything after it, up to
    /// the next Segment, belongs to it.
    Segment(Arc<Segment>),
    /// The segment's stream ends. Nothing about threads or lifetimes.
    Eos,
}

pub struct Segment {
    pub id: SegmentId,              // unique within the pipeline
    pub flushed: bool,              // follows a flush: drop what is held from before
    pub start: Duration,            // where the segment starts, on the stream's timeline
    pub show_from: Option<Duration>, // accurate seek: decode from start, show from here — on the timeline
    pub position: Duration,         // `start` as a position in the media, for a caller
    pub backwards: bool,            // which way; how fast is the playback clock's — a rate change without a turn begins no segment
    pub step: Option<usize>,        // a frame step's count, where this segment is one
}
```

Every time in a segment that an element compares a buffer against is on the
stream's timeline — the one the buffers are stamped on — and the source that
emits the segment is what converts a caller's position into it. A looping
file's timeline runs on across laps while a caller seeks within the file;
today the seek target and the pictures are in those two different
coordinates, and the matrix found what that costs (§3.1). `position` is the
same point as the caller meant it, for what reports progress.

Sources emit a `Segment`:
- when they start — every source, live ones included, so every stream begins
  with one and nothing downstream handles a stream without;
- after every seek, including a turn of direction;
- at each lap of a loop;
- from a live source, where its timestamps break — a camera plugged back in,
  a stream that restarts — as a `Segment` that is not flushed. obs-rs
  6d658be is this bug: a `Pacer` waited out a camera's timestamp jump
  because nothing said a new timeline had begun.

Compositors and mixers are sources, and emit their output's segments.

A stretch played backwards is still told to its decoder where a decode time
goes back, by the stage in front of it (`mark_stretch`). A `Stretch` event
from the source was tried and taken out again: §3.1.

What moves out of `PlaybackState` and the control messages:

| Today | Becomes |
|---|---|
| timeline number, thread-local, checked at `Queue` | done: `Segment.id` from the pipeline's count, and every pad and `Queue` worker the pipeline's own `Flush` reaches refusing what arrives until the flushed `Segment` passes (§4.3). A simplification, not a fix; `what_was_read_before_a_seek_does_not_arrive_after_it` fails with neither and passes with either. |
| `PlaybackState::backwards` | done for the elements: `Segment.backwards`, read by the stage that marks a decoder's stretches and by the decoders' preroll gate. A bool, not a rate: a rate that changes without a turn begins no segment, so a rate there would go stale. The pipeline and a source still read the state to decide how to seek. |
| `PrerollContext` target and step count | `Segment.show_from`, `Segment.step` |
| `Flush` resetting per-element state | the flushed `Segment`, arriving exactly where the old data ends |
| "first sample of the new timeline" | first buffer a terminal takes after the flushed `Segment` — exact, per terminal |
| `Completion`: `Eos` "not flushed since" | `Eos` of the current segment id |
| clock reset for a seek in four places | the one place that turns a `Segment` into a playback anchor (the playback clock, fed by the terminal that shows the segment's first picture) |

This reverses a rule `AGENTS.md` states outright (c0e76e8): *do not add a
number to `MediaBuffer` or thread it through elements*. The rule's reason
survives: no element has to know. A segment is not a field every buffer
carries and every element copies. It is an item the default handling
forwards, and an element with nothing to do about it never sees it. What the
rule could not provide is what the segment adds: the start, target, rate and
direction of a timeline next to its data, which is what 2ca1ecf and 92ef428
needed.

### 4.3 The flow plane

The pipeline owns the flow. It knows every thread-owning element: each
source, each `Queue` worker, each element with its own worker. It delivers
flow requests to each of them **directly**, not through the thread upstream.
It delivers device hooks (`pause`, `resume`, `stop`) directly to the elements
that have a device: capture sources and audio renderers.

- **Pause**: close every flow gate. A gate is where a thread pulls: a source
  asking for its next item, a `Queue` worker taking from its channel. Closing
  needs no order, because nothing is released by closing. Device hooks run
  alongside.
- **Resume**: open every gate. Each thread opens the elements on its own
  thread before it pulls again, and a `Queue` whose worker has not opened yet
  holds what it is handed, so no element is handed data before it is open,
  in whatever order the threads take the request. Stage 1 found that this is
  enough; an order computed from the graph — terminals' side first, sources
  last, as this section first proposed — is not needed.
- **Flush start**: raise the interrupt, empty every queue, and put every pad
  into **flushing**, which refuses data. Flushing ends when the flushed
  `Segment` passes the pad. A buffer mid-flight during the flush is refused
  because its pad is flushing; no number is needed.
- **Stop**: end every thread directly. A queue whose source has exited is
  reached like any other (d817bac, 06a8180).
- **The interrupt**: one primitive, `Wait`, used by every blocking wait in the
  framework: queue send, readiness, clock waits, preroll. Every flow request
  raises it. An element never blocks except through it. A device wait that
  cannot be interrupted, such as PipeWire's drain, runs off the element's
  thread or with a bounded timeout, and says so.

Threads live as long as the pipeline, not as long as their stream. A source
that reaches its end emits `Eos` and waits at its flow gate: seekable, stop-
able and pausable like any other. That is what `FileDemuxer::linger` does
today by hand. `Finish` asks each source to end its segment with `Eos` at its
next item.

### 4.4 Preroll

A preroll is:

1. Flush start (the pads are flushing).
2. Sources reposition and emit a flushed `Segment`.
3. Gates open in reverse topological order.
4. Each terminal takes buffers until it has the one the segment asks for:
   - the first buffer at or after `show_from`;
   - the `step`-th buffer, for a step;
   - `Eos`.

   It then reports *prerolled* on the feedback plane and becomes **not
   ready**. Backpressure holds everything upstream of it.
5. When every terminal has prerolled, the pipeline closes the gates (paused)
   or leaves them open (playing).

So the "preroll is over" fact never travels at all. A terminal knows it from
its own input, and everything else is simply held by backpressure. Four of the
five hold mechanisms exist to fake this backpressure and go away:
- `Tee.owed`;
- `PrerollGate.after`;
- `Pacer.pending` held across phases;
- `FileDemuxer`'s preroll parking.

Two cases need care:

- **Fan-out.** A `Tee` whose branches preroll at different times cannot block
  on the finished branch without starving the other (366c248). So a branch
  of a `Tee` that has to preroll begins with a `Queue`, and the `Tee` itself
  only forwards. Nothing inserts one: a `Queue` is this crate's explicit
  thread and error boundary, and its depth and overflow policy are the
  branch's to choose — a Preview drops, a recording blocks. A graph with a
  `Tee` branch that has no `Queue` refuses a seek, a step and a turn in
  `check_seek` / `check_reverse`, naming the `Tee`, as a recording muxer
  refuses them today. A graph that never seeks is unaffected: obs-rs's
  compositor counts every frame on a deliberately synchronous branch.
- **One packet, several frames.** A decoder can produce more frames from one
  packet than a prerolled terminal will take. The framework's output stash
  (§4.5) keeps them. This is the one generic hold that remains, written once.

Clipping to `show_from` happens in one place: a framework stage after every
decoder, which is today's `PrerollGate` reading the segment instead of the
phase. It applies to audio and video alike, which 2ca1ecf wanted.

### 4.5 One default behaviour

Elements implement their media work. The framework implements the protocol.

```rust
/// One frame or packet in, zero or more out — the shape of about forty
/// elements today.
trait Transform: Send {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()>;
    /// Delayed output, handed on before Eos. Default: nothing.
    fn drain(&mut self, out: &mut Output) -> Result<()> { Ok(()) }
    /// State from the timeline being left. Default: nothing.
    fn reset(&mut self) {}
    /// A new segment, for the few that care (a tempo stretcher reading its
    /// rate, a muxer with a timeline). Default: nothing.
    fn segment(&mut self, _segment: &Segment) {}
}
```

The framework's wrapper around a `Transform`:
- forwards events in order;
- calls `drain` before forwarding `Eos`, and `reset` on a flushed `Segment`;
- clears the repeat cache;
- keeps whatever the element produced that downstream is not ready for in
  the **output stash**, handed on first when readiness returns, and reports
  itself not ready meanwhile;
- stamps errors with the element's identity.

The element never sees a control message, the phase, a preroll or a seek.

```rust
/// A terminal. The framework delivers events and runs the preroll counting.
trait Render: Send {
    fn render(&mut self, buf: MediaBuffer) -> Result<()>;
    fn drain(&mut self) -> Result<()> { Ok(()) }          // before Eos is accepted
    fn reset(&mut self) {}                                // flushed Segment
    fn segment(&mut self, _segment: &Segment) {}
}

/// A source. The framework runs the loop, the gate, the interrupt, Finish,
/// linger-at-end and Segment emission; the source produces.
trait Produce: Send {
    fn is_live(&self) -> bool;
    /// The next item, blocking through the framework's `Wait`.
    fn produce(&mut self, wait: &Wait) -> Result<Produced>;
    fn seek(&mut self, _to: SeekRequest) -> Result<Landed> { Err(not_seekable()) }
}

/// For elements that own a device.
trait Device {
    fn pause(&mut self) -> Result<()> { Ok(()) }
    fn resume(&mut self) -> Result<()> { Ok(()) }
    fn stop(&mut self) {}
}
```

The raw `Sink` / `Source` pads remain for the few elements that route the
stream themselves: `Queue`, `Tee`, bins and racks, muxers with several
tracks, and compositor and mixer inputs. These are core or near-core, and
each is written against the protocol once.

What disappears from elements:

- about 85 `control()` implementations: the reset ones become `reset()`, and
  the device ones become `Device`;
- 17 hand-written source loops with `drain_control`, `handle_request`,
  `paused_for` accounting and `select!`: the loop is the framework's;
- the audio renderers' private `paused` flags and their disagreement: pause
  is a device hook, and the paused-preroll sample is the framework's to hold;
- four of five hold mechanisms (§4.4);
- every `PlaybackState` read outside the core.

### 4.6 Fan-in

A compositor or mixer input is a terminal of its feeding pipeline.

- It receives that pipeline's `Segment`s.
- In offline mode it maps them into the output's timeline, as the offline
  inputs already do with each input's pts.
- Its `Eos` ends that input only.
- The output's own segments are the compositor's to emit.

Writing this once, in the input helper, removes the divergence between the
Sw/D3D11 and Vulkan/CUDA inputs.

---

## 5. What this does not fix

- **Blocking outside the framework.** The design makes every framework wait
  interruptible. A device call that blocks by itself stays the element's
  problem, bounded by a timeout. eb235e3 was such a call: PipeWire's drain of
  a paused stream.
- **The single read cursor (E).** One demuxer feeding branches with separate
  backpressure is an interleaving problem, GStreamer's `multiqueue`. The
  per-branch queue after a `Tee` helps a `Tee`, not a demuxer. It is a
  separate piece of work, after this one.
- **Local bugs (F)**, about a third of the history: devices, rate arithmetic,
  shutdown joins, error attribution.
- **Rate changes without a turn** stay a property of the playback clock, as
  they are now: out of band, and not a new segment.

---

## 6. Migration

Each stage builds and passes the suite on its own. Stages 0 and 1 stand on
their own even if nothing after them is done. The version is decided once the
refactor is done; the A2 timeline engine waits for stage 4, since a timeline
of clips is segments, and built on today's core it would add one more set of
cross-channel rules to take apart again.

| Stage | What | Breaks |
|---|---|---|
| **0. A generated conformance matrix** — done | Six axes of file shape — fan-out, decoder, filter, pacing, sound, queue depth — in 16 shapes covering every pair of choices, beside the live and offline shapes; random sequences of pause, resume, seek (at and past the end too), step, rate both ways, looping, finish, stop, and stop while a call is under way, judged by what each call promised rather than by the messages that carried it, so the judge stands through every later stage. Found three bugs in its first hours (§3.1). | nothing |
| **1. The flow plane** — done | The pipeline delivers every request directly to every source and every `Queue` worker, and each thread passes it on only to the elements on its own thread (`control::Direct`); a queue carries across only what was not sent to every thread, such as a source's own `Stop` after an error. A worker that ends delivers what is still waiting for it first. `stop` returns once the threads have ended. Resume needs no order (§4.3). This addresses the C and G clusters, which in-band events cannot. Threads living until `Stop` with the framework's own "linger at the end", and the one `Wait` primitive, move to stage 5, where the framework owns the source loop. | nothing public; `stop` waits for the threads |
| **2. The stream plane** — done | `crate::stream`: `Item`, `StreamEvent`, and a hook on `Sink` that only this crate can override, since its argument is a type nothing outside can name. An event goes as control does on one thread — the element's reaction, which pushes what it answers the event with ahead of it, then the framework on through its pads — but in order with the data: a `Queue` carries it in its channel, a `Tee` hands it to each branch and keeps it with what a preroll holds back, a bin or a rack sends it down its line. It is never dropped for room and never waited for, since a source begins its stream before it looks at its control. What joins a stream under way — a branch attached to a `Tee`, a line filled anew — is handed its last segment first. The one event is a `Segment` saying which timeline it opens and whether a flush came before it, emitted by the framework as each source's thread starts and after each seek it applies — a turn included; the conformance matrix checks that every buffer a terminal is handed is on the timeline of the segment before it. `Eos` stays `MediaBuffer::Eos` until stage 6, which moves it in one step rather than mirroring it. | nothing public |
| **3. Sources emit the rest** | A `Segment` at each lap of a loop, and from a live source where its timestamps break, unflushed; a `PipelineBridge` passing its feeding side's flush on as a flushed `Segment`. The segment's `start` and `position` are in, which the decoders' preroll gate already reads to hold a seek's target against the pictures (§3.1); `show_from`, rate and step to come. A caller's seek on a looping file stays on the lap the source has read to, which near a lap's end is already the next one rather than the one shown; the segment is where the lap shown can come from. Done: pads and queue workers refuse what arrives between a seek's `Flush` and its segment, and the thread-local timeline number is gone; only the pipeline's own `Flush` flushes pads, so an element driven by hand behaves as before. Left: the lap, live and bridge segments, and `show_from`, rate and step, each with what reads it. | nothing public |
| **4. Readers move to the segment** | `backwards`, the seek target, the step count, clipping, completion, the clock anchor. `PlaybackState` shrinks to the flow phase, the interrupt and the preroll bookkeeping. | `crate`-internal only |
| **5. `Transform` / `Render` / `Produce` / `Device`** | Every in-tree element migrated, one family at a time, each family deleting its `control()` and its holds. Sink-side preroll (§4.4) lands with the renderers. With the source loop the framework's, threads live until `Stop` and "linger at the end" is the framework's; every blocking wait goes through one `Wait` (§4.3). | custom elements: they keep compiling against raw `Sink`/`Source` |
| **6. Remove the old surface** | `MediaBuffer::Eos` becomes `StreamEvent::Eos`, and the stream plane is made public. `ControlMsg` in `Sink`, `drain_control`, `handle_request` and the `pub` preroll types go. | **breaking**: custom elements, obs-rs |

obs-rs impact: it implements no element of its own — no `Sink`, no
`SourceElement`, no `control`. Stage 6 touches the one place it pushes
`MediaBuffer::Eos` through an `AppSource` (`engine/source/shared.rs`), any
`AppSink` closure that matches `Eos`, and the places that read `BusEvent`s.

---

## 7. Decisions

Settled 2026-09-27.

1. **Events are a separate `Item`, not `MediaBuffer` variants.** Today 37
   elements answer an unknown buffer with `UnsupportedBuffer` and 19 swallow
   it with `let MediaBuffer::Video(frame) = buf else { return Ok(()) }`. As a
   variant, a `Segment` would be an error in the first and silently dropped
   in the second — and a filter that drops a segment leaves everything after
   it without the new timeline, unnoticed: the side effect this redesign is
   for. As a separate type, `Transform` and `Render` receive `MediaBuffer`
   only, and that an element never loses an event is the compiler's to
   check. `Item` exists at the pads, in `Queue`, `Tee`, bins, racks and
   multi-track muxers — code rewritten for this protocol anyway — and
   `MediaBuffer` loses `Eos` and becomes data only.
2. **Sink-side preroll** (§4.4) is the end state. Segment-keyed holds are a
   waypoint only: stages 3–4 bring segments in without changing how a
   preroll works, and stage 5 switches it, behind the stage 0 matrix. The
   holds are where the preroll bugs were (fd9f756, 366c248, 6d3c161,
   546f8a9, ab7f577, dd527d9, 2ca1ecf, 9aa9fdd, 8d3ca52); keying them to an
   id would keep every one of those places. The fixed hardware decode pools
   (D3D11VA, NVDEC's 32) hold no more than today: `PrerollGate` and the
   queues already hold those surfaces.
3. **Every source emits a `Segment`**: at start, and a live one again, not
   flushed, where its timestamps break (§4.2).
4. **No `Queue` is inserted after a `Tee`.** A `Tee` branch without one makes
   the graph refuse seeks, steps and turns, naming the `Tee` (§4.4).
5. **The version is decided after the refactor.** The A2 timeline engine
   waits for stage 4.
