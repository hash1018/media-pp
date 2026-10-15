# How media-pp works

The ideas you need before writing anything bigger than the first example.
Each section is short and links to where the details are. The crate
documentation's first page ([docs.rs], or the [Windows] and [macOS] API
pages) covers the same ground in full.

- [Pipelines, sources, filters and sinks](#pipelines-sources-filters-and-sinks)
- [Building a branch](#building-a-branch)
- [Threads and queues](#threads-and-queues)
- [Several sources in one pipeline](#several-sources-in-one-pipeline)
- [Errors and the bus](#errors-and-the-bus)
- [Ending a pipeline](#ending-a-pipeline)
- [Pausing, seeking, speed and stepping](#pausing-seeking-speed-and-stepping)
- [Playing in time](#playing-in-time)
- [Buffers, timestamps and metadata](#buffers-timestamps-and-metadata)
- [Where pictures live: system memory and GPUs](#where-pictures-live-system-memory-and-gpus)
- [Live and offline](#live-and-offline)
- [Changing a pipeline while it runs](#changing-a-pipeline-while-it-runs)
- [Watching a running pipeline](#watching-a-running-pipeline)
- [Writing your own element](#writing-your-own-element)
- [Logging](#logging)

## Pipelines, sources, filters and sinks

A **pipeline** moves media from where it comes from to where it goes. It is
made of **elements** of three kinds:

| Kind | What it does | Examples |
|---|---|---|
| **Source** | Makes buffers: reads a file, captures a screen, receives a stream | `FileDemuxer`, `DxgiCaptureSource`, `RtspSource`, `TestVideoSource` |
| **Filter** | Turns each buffer into none, one or several | `SwDecoder`, `SwScaler`, `SwEncoder`, `SwOrtDetector` |
| **Sink** | Is where buffers end: a file, a window, a speaker, your code | `FileMuxer`, `D3d11WindowRenderer`, `WasapiRenderer`, `AppSink` |

```text
FileDemuxer ──> SwDecoder ──> Queue ──> SwScaler ──> AppSink
  source         filter       thread     filter        sink
```

A source can have several outputs — a file has a video and an audio
stream — and each output feeds its own **branch**. A pipeline can have
several sources.

## Building a branch

A branch is written in the order buffers flow, then attached to one output
of a source:

```rust,ignore
let (pipeline, ()) = Pipeline::new("play", source, |source, ctx| {
    let branch = ctx
        .branch()
        .pipe(decoder)          // a filter
        .queue("frames", 8)     // a thread boundary, 8 buffers deep
        .to(renderer)?;         // the sink it ends in
    ctx.attach(source, video.index, branch)?;   // onto the video stream
    Ok(())
})?;
pipeline.run()?;
```

`Pipeline::new` hands back the pipeline and whatever the closure returns —
handles you want to keep, for instance.

**Links are checked before anything runs.** Connecting a decoder's frames
straight to a muxer that wants packets is refused when the branch is built,
with a message saying what is missing and what to put between:

```text
decoder produces VideoFrame (System), which rec cannot accept
(it takes VideoPacket|AudioPacket); encode it first: an encoder turns
frames into packets
```

Nothing is converted for you: media-pp does not negotiate formats the way
GStreamer's caps do. What each element accepts is in its documentation, and
the [elements list](../elements.md) groups them by job and backend.

## Threads and queues

- **Each source runs on its own thread.** It pushes each buffer through its
  branch with a plain function call: the decoder runs, then the scaler, then
  the sink, on that one thread.
- **A `Queue` starts a new thread.** Put `.queue(name, depth)` in a branch
  wherever the work before and after it should run side by side — typically
  after a decoder, so decoding is not held up by rendering or encoding. The
  depth bounds how many buffers wait between the two.
- **What waits on a clock gets its own queue.** A `Pacer` or a
  `VideoSynchronizer` sleeps until each buffer is due; the branch puts a
  queue in front of it if you did not.

Nothing else creates threads behind your back, so where work runs is
always visible in the branch you wrote.

**When a queue is full** it waits for room by default, which is right for a
file: nothing is lost. For a live preview, where being on time matters more
than every picture, drop instead:

```rust,ignore
use media_pp::queue::OverflowPolicy;

ctx.branch()
    .pipe(decoder)
    .queue_with_policy("preview", 4, OverflowPolicy::DropNewest)
    .to(renderer)?
```

Each dropped buffer is reported as `BusEvent::Dropped`.

## Several sources in one pipeline

`Pipeline::new` takes one source. For several that belong together — a
screen and a microphone recorded into one file — use `PipelineBuilder`, so
they start, pause and stop as one:

```rust,ignore
let builder = PipelineBuilder::new("record");
let (builder, ()) = builder.add_source(screen, |screen, ctx| { /* wire it */ Ok(()) })?;
let (builder, ()) = builder.add_source(microphone, |mic, ctx| { /* wire it */ Ok(()) })?;
let pipeline = builder.build();
pipeline.run()?;
```

The [`screen_record_av`](../../examples/render/screen_record_av) example
records a screen and its sound this way.

## Errors and the bus

Two ways a failure reaches you:

1. **Building** a pipeline or a branch returns `Err` straight away: a link
   that cannot carry data, a file that will not open, a codec that is not
   there.
2. **While running**, failures are posted on the pipeline's **bus** as
   `BusEvent::Error`, naming the element that failed. The element after a
   failed buffer carries on with the next one: one bad frame does not stop
   a recording.

The bus is a channel you read:

```rust,ignore
for event in pipeline.bus().iter() {
    match event {
        BusEvent::Finished => break,
        BusEvent::Error { name, error, .. } => eprintln!("{name}: {error}"),
        _ => {}
    }
}
```

| Event | Means |
|---|---|
| `Error` | An element failed on one buffer, or a source could not go on |
| `Eos` | One element has finished its stream |
| `Finished` | Every sink has finished: the pipeline played to its end |
| `Dropped` | A queue was full and its policy dropped a buffer |
| `Seeked` | A seek landed, and where |

## Ending a pipeline

| Call | What happens |
|---|---|
| `pipeline.finish()` | Ends the stream from the source: encoders flush what they hold, files are finalized. Use it to end a recording. |
| `pipeline.stop()` | Abandons everything at once. Use it once you have what you need, or on an error. |

**A file does not end by itself.** At the end of a file its source waits,
because a seek could still take it back. Wait for `BusEvent::Finished` —
every sink has everything — and then call `stop()`. Dropping a `Pipeline`
stops it too.

## Pausing, seeking, speed and stepping

| Call | Does |
|---|---|
| `pause()` / `resume()` | Holds the pipeline and goes on |
| `seek(position, SeekMode::Accurate)` | Shows exactly that time; `SeekMode::Keyframe` is faster and shows the nearest keyframe |
| `set_rate(2.0)` | Plays faster or slower, the sound at its own pitch; a negative rate plays backwards, picture only |
| `step(1)` / `step(-1)` | Moves one picture forward or back and stays paused there |
| `check_seek()` | Asks whether this pipeline can seek at all, before trying |

A pipeline that is recording to a file refuses to seek: a recording is a
record of what happened. Live sources — cameras, screens, network streams —
cannot be sought either.

## Playing in time

Nothing in a pipeline waits for the clock unless you ask. A file is read as
fast as it decodes — right for converting it, wrong for watching it.

- **`Pacer`** hands each picture on when it is due. Use it for pictures
  alone.
- **`VideoSynchronizer`** does the same but follows the sound once an audio
  output is playing, so picture and sound stay together. Use it in place of
  a `Pacer` when there is sound — not both.
- Live sources — cameras, screens, network streams — arrive in time
  already and need neither.

## Buffers, timestamps and metadata

Everything that travels is a `MediaBuffer`:

| Variant | Holds |
|---|---|
| `Packet` | Encoded data, from a demuxer or an encoder |
| `Video` | A decoded picture — in system memory or on a GPU |
| `Audio` | Decoded sound |

The media inside is shared, not copied: sending one picture to three
branches costs three reference counts. Timestamps, time bases and colour
information pass through every element that does not deliberately make a
new timeline.

A buffer also carries **metadata**: what a detector found in a picture
(`Detections`), which stream of a batch it came from, a cut detected there.
Elements that do not know a piece of metadata hand it on untouched, so a
tracker after a scaler still sees the detector's boxes.

Your own types travel the same way: any `Send + Sync` value can be attached
and read back by type.

```rust,ignore
// Add to what the buffer already carries — `with_metadata` replaces all of it.
let metadata = buf.metadata().cloned().unwrap_or_default().with(CameraName("door".into()));
let buf = buf.with_metadata(metadata);
// … later, in another element:
if let Some(name) = buf.metadata().and_then(|m| m.get::<CameraName>()) { /* … */ }
```

## Where pictures live: system memory and GPUs

A decoded picture lives in one place, its **memory domain**: system memory,
or a GPU of one backend — D3D11, D3D12, CUDA, Vulkan, or VideoToolbox and
Metal. Each backend's elements take that backend's pictures and carry its
prefix (`D3d12Scaler`, `CudaEncoder`, `MetalVideoCompositor`); unprefixed
elements (`SwScaler`, `SwEncoder`) work on system memory, on any platform.

- To keep a whole pipeline on the GPU, use one backend's elements all the
  way through, all made from the **same device** (`D3d12Gpu`,
  `CudaDevice`, …). Pictures never come back to the CPU.
- To cross over, an `…Upload` element puts pictures onto a GPU and a
  `…Download` element brings them back.
- `VideoDecodeBin` and `VideoEncodeBin` choose a backend for you: hardware
  where it opens, software where it does not.

Which backend to pick on which machine is in
[choosing a backend](choosing-a-backend.md).

## Live and offline

Elements that combine several inputs — `AudioMixer` and the video
compositors — run in one of two modes, `RenderMode`:

- **Live**: produce at a steady rate, showing each input's latest picture.
  For a screen recorder or a broadcast.
- **Offline**: produce as fast as the inputs allow, placing each input by
  its timestamps, frame by frame. For exporting an edit, where every frame
  must be exact and nothing should wait for the clock.

## Changing a pipeline while it runs

A branch keeps the shape it was built with. These elements are where a
running pipeline changes anyway:

| Element | Lets you |
|---|---|
| `Tee` and its `TeeHandle` | Add a branch while buffers flow, finish one cleanly (a recording finalized while the preview keeps going), or drop one |
| `AudioMixer`, the video compositors | Add and remove inputs and layers, move and restack them |
| `Rack` | Replace the filters in a stretch of branch — add an effect — without reopening the source |
| `PipelineBridge` | Feed one pipeline from another, so a source that fails takes only its own pipeline with it |
| `FileDemuxerHandle` | Loop a file, its timeline carried across each lap |

## Watching a running pipeline

| Call | Gives |
|---|---|
| `pipeline.topology()` | A diagram of the graph, every element with its id — the same one the log writes at start |
| `pipeline.stats()` | What has passed each element so far, and how far behind live sources and compositors are |
| `pipeline.position()` | Where playback is |
| `pipeline.bus()` | Every error, end and dropped buffer, as it happens |

## Writing your own element

Implement one trait:

| You want | Implement | You write |
|---|---|---|
| Something that makes buffers | `Source` | `produce`: make the next buffer, or wait for one |
| Something that changes buffers | `Filter` | `transform`: take one buffer, push out what it becomes; `drain` for what it holds at the end |
| Something buffers end in | `Sink` | `render`: do something with each buffer |

The framework handles the rest: pausing, seeking, ending, threads. The
[`custom_element`](../../examples/core/custom_element) example writes one of
each and runs them together.

## Logging

media-pp writes its own log file rather than going through a global
logger, so it never clashes with your application's:

```rust,ignore
let _guard = media_pp::log::init("my-app", "logs", media_pp::log::Level::Info, 7)?;
```

Keep the guard for as long as you want logs written; it keeps 7 days of
files in `./logs`. Each line says which pipeline, element and thread it
came from, and every pipeline's start logs a diagram of its graph.

[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
