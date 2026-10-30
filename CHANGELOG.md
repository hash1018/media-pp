# Changelog

Notable changes to `media-pp`. Versions before 0.2.0 have no entry here — this
file starts where the first breaking release did, and the git history is the
record for everything earlier.

The crate is pre-1.0, so a `0.x` bump is where breaking changes land. Each one
below says what to write instead, because a rename with no migration line is a
compile error with no explanation.

## Unreleased

### Breaking

- **`TrackFormat` is gone; muxers and `Broadcast::publish` take a
  `StreamInfo`.** The two said the same thing — a stream's codec
  parameters and time base — and a demuxed stream had to be turned from
  one into the other. Write `StreamInfo` where you wrote `TrackFormat`:
  `StreamInfo::new(parameters, time_base)` for a stream described by hand,
  `&encoder` and `&stream` as before, `.with_orientation(..)` as before.

- **`StreamInfo`'s fields are read through methods.** Write
  `stream.index()`, `stream.kind()`, `stream.parameters()`,
  `stream.time_base()` and `stream.frame_rate()` for the fields of the same
  names; `stream.parameters().clone()` where the parameters were moved out.
  `StreamInfo::new` and `with_frame_rate` make one. A field can now be added
  without breaking anything that reads it.

- **`MuxerTrackError` has a `FormatChanged` variant**, so a `match` on it
  needs an arm for it — see below.

- **A `WebRtcPeer` sends through a `WebRtcMuxer` and receives through a
  `WebRtcSource`**, the two a WHIP or a WHEP connection is used through.
  Every media section goes one way. What this side sends,
  `handle.muxer(name)` makes a muxer of — given its tracks with
  `add_stream` and opened as one to a WHIP server is, a sink per track out
  of its `MuxerSinks` — which offers its tracks in one renegotiation, in
  the order it was given them: the offer goes to `on_offer`, the answer
  comes back through `set_answer`. What the remote peer adds comes out of
  `handle.next_source(name)` as one source of the tracks one of its offers
  adds, an output each, with what each carries — waited for, by its first
  packets, as `wait_stream_info` was. A section the remote peer adds asking
  to be sent to is where a muxer's track of its kind goes, as it opens, with
  no renegotiation, where the section took the track's codec. `add_track`, `next_track`, `AttachedTrack`,
  `TrackEndpoints`, `WebRtcTrackSink`, `WebRtcTrackSource`,
  `WebRtcStreamInfo` and `WebRtcError::StreamInfoTimeout` are gone, and
  `on_keyframe_request` is handed the track's name from `add_stream` where
  it was handed its `TrackId`. Instead of

  ```rust
  handle.add_track(MediaKind::Video, Direction::SendOnly, Codec::H264)?;
  let TrackEndpoints::Send(mut sink) = handle.next_track()?.endpoints else { .. };
  sink.set_source_parameters(&encoder.parameters())?;
  ```

  write

  ```rust
  let mut muxer = handle.muxer("camera");
  let track = muxer.add_stream(
      "video",
      StreamInfo::new(encoder.parameters(), encoder.time_base()),
  )?;
  let sink = muxer.open()?.take(track)?;
  ```

  and `let (source, streams) = handle.next_source("remote")?;` where a
  track's `WebRtcTrackSource` and its `wait_stream_info` were. The four
  WebRTC examples show both sides.

- **A server hands each publisher out as a source.** `RtspServer`,
  `WebRtcServer` and `RtmpServer` no longer write a publisher into a
  `Broadcast`: `take_publishers(path)` answers the path's `Publishers`,
  whose `next()` — or `next_async()`, for an async runtime — answers each
  publisher as it begins, an `RtspSource`, a `WebRtcSource` or the new
  `RtmpSource` of what it sends, with its streams, to be wired as any
  source is. Its source ends as the publisher leaves, and dropping it ends
  the publisher's connection; dropping the `Publishers` takes the path
  away. A publisher is taken apart from what players play under the same
  name. `RtspPathMode` and `WebRtcPathMode` are gone, and `add_path` takes
  the path and its broadcast alone; `RtspServerHandle::add_publish_path`
  and `remove_publish_path`, and `RtmpServerHandle::add_path` and
  `remove_path`, are gone. Where a path was added for a publisher —
  `RtspPathMode::Record`, `WebRtcPathMode::Publish`, an RTMP path — write
  its publishers into the broadcast its players read:

  ```rust
  handle.add_path("live", &live)?;
  let publishers = handle.take_publishers("live")?;
  while let Some((source, streams)) = publishers.next() {
      let sinks = live.publish(&streams)?;
      let (pipeline, ()) = Pipeline::new("live", source, |source, ctx| {
          for (stream, sink) in streams.iter().zip(sinks) {
              ctx.attach(source, stream.index(), ctx.branch().to(sink)?)?;
          }
          Ok(())
      })?;
      pipeline.run()?;
      // ... until its bus says `Finished`: the publisher has gone.
  }
  ```

  `ElementType` has an `RtmpSource` variant, so a `match` on it needs an
  arm for it. The `rtsp_server` and `webrtc_server` examples relay their
  publishers this way.

- **What a server plays is written through a muxer of its path.**
  `RtspServerHandle::path(name)` and `WebRtcServerHandle::path(name)`
  answer an `RtspServerPath` or a `WebRtcServerPath`, served for as long as
  it lives — dropped, its players' sessions end — whose `muxer()` is an
  `RtspMuxer` or a `WebRtcMuxer`, made, given its tracks and opened as one
  to a remote server is: each opened begins a run of the path's stream,
  which its players go on across. `add_path` and `remove_path` are gone,
  and a `Broadcast` is no longer handed to a server. Instead of

  ```rust
  let camera = Broadcast::new("camera", BroadcastOptions::default());
  handle.add_path("cam1", &camera)?;
  let sinks = camera.publish(&streams)?;
  ```

  write

  ```rust
  let cam1 = handle.path("cam1")?; // kept for as long as it is served
  let mut muxer = cam1.muxer();
  let tracks = streams
      .iter()
      .map(|stream| muxer.add_stream(format!("track-{}", stream.index()), stream))
      .collect::<media_pp::Result<Vec<_>>>()?;
  let mut sinks = muxer.open()?; // `sinks.take(track)` for each
  ```

  A `WebRtcServerPath`'s muxer refuses a track WebRTC does not carry, as
  any `WebRtcMuxer` does; check `WebRtcServer::carries` first. A
  `Broadcast` of the application's own, read by several outputs, is
  written into a path by a `BroadcastSource` of it into the path's muxer,
  as the `rtsp_server` and `webrtc_server` examples do.

  A path's `players()` says how many play it, and `players_change(from)`
  waits, on any async runtime, until other than `from` do: so what writes
  into it can do so only while it is played. Between runs a path still
  describes its last one, so a player plays it, is counted, and goes on
  with the next run its muxer opens.

### Changed

- **A muxer of MP4 refuses a track whose pictures change partway.** An
  H.264 or H.265 stream that goes on with other parameter sets — another
  size, profile, chroma format or bit depth — was written into a file
  whose header kept the first, which players show wrongly or not at all.
  `FileMuxer` to an MP4 or MOV, `HlsMuxer` of fMP4 and `DashMuxer` now
  answer the first keyframe of the new pictures with
  `MuxerTrackError::FormatChanged`, write nothing more of the track, and
  finish the output with what came before. Matroska, MPEG-TS, RTMP and
  RTSP write on as before.

- **A WebRTC player, or an RTSP one over UDP, starts at most a second
  behind.** A player joins at the newest keyframe its path's broadcast
  holds and is sent what came after at twice the stream's speed; a group
  of pictures of large pictures is thousands of packets, and one lost in
  them is past what a WebRTC connection keeps to send again — or, over UDP,
  anything can ask for again — so the picture broke until the next
  keyframe. Where the newest keyframe is more than a second behind, such a
  player now waits for the next one: its first picture comes up to a group
  of pictures later, but whole. An RTSP player over TCP, which its
  connection paces and which loses nothing, still starts at once.

- **A `WebRtcPeer` that falls behind goes on from a keyframe.** Its track
  sinks share one outbox; a peer backed up as far as it keeps lets go of
  what it holds and sends each track from its next keyframe, where it used
  to drop the newest packet each time — leaving a picture broken until the
  next keyframe anyway, and the peer still behind.

- **Speech is a family of elements, as vision is: `elements::speech`.**
  `WhisperTranscriber`, a terminal sink handing each line to a callback, is
  gone; a transcriber is now a filter per backend — `SwWhisperTranscriber`,
  and `CudaWhisperTranscriber`, `VulkanWhisperTranscriber` and
  `MetalWhisperTranscriber` on a GPU — that hands the sound on, holding it
  until its words are settled, each buffer carrying the lines that begin in
  it as a `Transcript`. `SubtitleEncoder` makes those lines a subtitle track,
  which took a hundred lines of the `transcribe` example before.

  ```rust
  // before
  let transcriber = WhisperTranscriber::new(name, model, policy, |segment| { /* … */ Ok(()) })?
      .with_language("ko")?;
  // after
  let options = WhisperOptions { chunk: policy, language: Some("ko".into()), ..Default::default() };
  let transcriber = SwWhisperTranscriber::new(name, model, options)?; // GPUs: …::new(name, model, options, 0)
  // then `.pipe(transcriber).pipe(SubtitleEncoder::new("text", subtitle::Codec::MovText))`,
  // or read `buf.metadata().and_then(|m| m.get::<Transcript>())` in an `AppSink`.
  ```

  `Segment` is `Utterance`, which carries its `words` with their times too,
  and no longer shares its name with `stream::Segment`. `WhisperOptions`
  also gives the model a `prompt` — the names and terms to spell as given —
  and can write the lines in English (`translate`). A GPU that is not there
  is now an error rather than whisper.cpp's silent fall back to the CPU.
  `ElementType::WhisperTranscriber` is one variant per backend, and
  `SubtitleEncoder`. The `transcribe` example is in `examples/speech` now,
  and picks its backend with `--gpu cuda|vulkan|metal`.

- **`ChunkPolicy::live_edge_ms` is gone.** Holding back the newest second
  was how a round decided what to trust; two rounds agreeing does that now
  (below). Leave the field out: `ChunkPolicy { chunk_ms, token_timing }`.

- **`RtspSourceError` has `InvalidListenUrl`, `NoPublisher` and
  `PublisherSilent` variants**, for `RtspSource::listen` — see Added. A
  `match` on `RtspSourceError` needs an arm for each.

### Added

- **`CudaDmaBufSource`: another process's DMA-BUFs as CUDA frames, on
  Linux with `cuda`.** What `D3d11SharedTextureSource` is on Windows and
  `MetalSharedTextureSource` on macOS: a pushed source whose producer hands
  `CudaDmaBufHandle::push` (or `try_push`, for a thread that cannot wait) a
  `DmaBuf` — one plane of BGRA, its stride, offset and modifier — per
  picture, which is copied on the GPU, through OpenGL, into a BGRA frame of
  the pipeline's own before the call returns. `DmaBufAlpha` says what the
  fourth byte is: ignored, straight, or premultiplied, which is divided
  back out. Linear buffers and NVIDIA's block-linear ones alike — the
  second is what Chromium 156 hands over drawing a page off-screen on its
  Wayland platform. `ElementType` has a `CudaDmaBufSource` variant.

- **`VulkanDmaBufSource`: the same, as Vulkan frames, on Linux with
  `vulkan`.** `VulkanDmaBufHandle::push` and `try_push` take the same
  `DmaBuf` and `DmaBufAlpha`, which `vulkan` alone now brings too. The
  buffer's memory is imported as an image laid out as its modifier says —
  where FFmpeg made the device with `VK_EXT_external_memory_dma_buf`,
  `VK_KHR_external_memory_fd` and `VK_EXT_image_drm_format_modifier`, and
  else `new` says so with `NoImport` — polled until the producer has drawn
  it, and copied by a compute kernel into a BGRA frame of the pipeline's
  own before the push returns. Not FFmpeg's mapping of a `DRM_PRIME`
  frame, which a build against kernel headers older than 6.0 makes without
  waiting for the producer and warns of at every picture. `vulkan` now
  takes `libc`, for that poll. `ElementType` has a `VulkanDmaBufSource`
  variant.

- **`heap::keep_small()`: glibc's heap held near what a server uses.** A
  process running many pipelines has a thread or more for each, and glibc
  an arena for each thread; left to itself it lets every arena keep 10 to
  14 MB free at its end. relay-pp's soak came to 1.8 GB of heap with a
  tenth of it in use; with this, called first in `main`, it held about
  550 MB for 2% more processor. It holds glibc's mmap and trim thresholds
  at 32 MB and 1 MB and trims every arena every ten seconds; it does
  nothing off Linux or without glibc, and nothing in the crate calls it —
  how a whole process allocates is the program's choice.

- **Low-Latency HLS: `LlHlsStream` and `LlHlsMuxer`.** An `LlHlsStream` is
  what a server hands its HLS players: fragmented MP4 cut into parts of a
  fifth of a second as it is written, kept in memory for a window of whole
  segments. `playlist(at)` answers a player's `_HLS_msn` and `_HLS_part`
  once what it names is in, and `file(name)` hands out an init section, a
  part or a segment — the part the playlist says is coming once it is in.
  Each run of the stream — a camera's connection — is written by a muxer
  `stream.muxer()` makes, with `add_stream` and `open` as the other muxers
  here; one whose init section is the last run's goes on as the same
  stream, its timestamps on from the last, with no discontinuity, so a
  player plays on across a camera that dropped and came back. Both waits
  may be awaited on any async runtime. `ElementType` has an `LlHlsMuxer`
  variant.

- **`WebRtcServerOptions::ice_servers`: STUN and TURN servers told to
  WHIP and WHEP clients.** As RFC 9725 tells them — a `Link` header each,
  with a TURN server's credentials — on the answer to an offer, and on the
  answer to an `OPTIONS` whose credentials `Authorize` lets through, so
  that a client behind a network that blocks UDP can ask for a relay before
  it offers; the server's own page hands them to the browser. A TURN
  server's credentials are told no one `Authorize` refuses, and
  `IceServer`'s `Debug` never shows them.

- **`RtmpServer`: what OBS and most encoders publish, taken in.** Under
  the `rtmp-server` feature, with no dependency beyond FFmpeg: a publisher
  pushing to `rtmp://<host>/<app>/<stream>` is written into the
  `Broadcast` added at `app/stream`, for an `RtspServer`, a
  `WebRtcServer` or a recording to read. H.264 and AAC as RTMP has always
  carried them, and H.265, AV1, VP9 and Opus as Enhanced RTMP does, go in
  as they come, nothing decoded; a picture's size comes from the metadata
  or the first keyframe, and a new configuration mid-stream publishes the
  broadcast again. A publisher that drops and comes back goes on into the
  same broadcast. `Authorize` decides who may publish, from `?user=&pass=`
  or `?token=` in the stream name.

- **`WebRtcServerHandle::session_stats`**: every open session's figures,
  measured about once a second — where its client is, the round trip, the
  share of packets lost, the bytes each way, resends and keyframes asked
  for, and how many times a player fell behind and skipped to a keyframe.

- **Who may play and publish: `Authorize`.** Given to `RtspServer` or
  `WebRtcServer` through their options, it is asked for each client before
  anything else about its request, with the path, whether it would play or
  publish, where it connects from and the credentials it presented —
  `Digest` (or `Basic`) from `rtsp://user:password@…`, `Bearer` from a
  WHIP or WHEP client. Refused, a client is answered `401`. A Digest
  response carries no password: `RtspDigest::verifies` checks it against
  the user's. The WHEP page plays
  `/<path>?token=…` with that token. Without one, everybody may, as before.

- **`Broadcast::stats`**: whether it is live and since when, how many
  times it was published, how many readers it has, the packets and bytes
  written into it and when the last was, and how many times a reader fell
  behind and was moved on — for a dashboard or a list of what a server
  serves.

- **`DashMuxer`: MPEG-DASH, and HLS over the same segments.** Made, given
  its tracks and opened as `HlsMuxer` is, it writes an `.mpd` manifest and
  fMP4 segments cut at keyframes — a sliding live window
  (`DashMode::Live`) or every segment, finished as a static manifest at the
  end (`DashMode::Event`). `DashOptions::hls_playlist` adds HLS playlists
  over those very segments, so that one recording plays in DASH players and
  Safari alike. Reading a manifest back takes an FFmpeg built with libxml2.

- **`WebRtcMuxer` and `WebRtcSource`: WebRTC to and from any WHIP or WHEP
  server.** `WebRtcMuxer` is made, given its tracks and opened as
  `RtspMuxer` is, handing out a sink per track out of a `MuxerSinks`;
  `WebRtcSource` is
  opened as `RtspSource` is, answering with its streams, an output each, and
  can be driven. Each is one HTTP exchange — `Authorization: Bearer` where a
  token is given — and a `DELETE` as it goes; what the server refuses is an
  error at open (`WebRtcError::Refused`, `Request`, `NotConnected`,
  `NothingDescribed`, `NoTracks`, `InvalidUrl` are new — a
  `match` on `WebRtcError` needs an arm for each). `https` comes with the
  `webrtc` feature through `ureq` and `rustls`. Tried against `WebRtcServer`
  and MediaMTX; `lib/tests/streaming.rs` carries a stream across WHIP,
  WHEP, RTSP and a recording each way. See `docs/whip-whep-clients.md`.

- **A WebRTC track converts HEVC's `hvcC` as it does H.264's `avcC`.** A
  `WebRtcMuxer` given an MP4 demuxer's HEVC stream puts the record's
  parameter sets in front of each keyframe and rewrites every
  length-prefixed packet as Annex-B; it used to refuse them with
  `ParameterSetsNotSupported`, which VVC's `vvcC` still is.

- **`DrivenPipeline`: a pipeline with no thread of its own.** Wired as a
  `Pipeline` is, then polled by whoever owns it — many of them on a few
  threads, as a server's players will be. `DrivenRunner` polls them for an
  application with no executor; as a `Future`, one goes to any executor,
  its deadlines kept by one timer thread for the process. Control is
  through `DrivenPipelineHandle` (pause, resume, stop, finish — none of
  them waits), and dropping one stops it. Its wiring refuses what has a
  thread of its own — a queue, a `Pacer` or `VideoSynchronizer`, a source
  that is not `Source::drivable` — with the new
  `GraphError::ThreadInDrivenPipeline` and `GraphError::SourceNotDrivable`;
  a `match` on `GraphError` needs an arm for each. `TestVideoSource`,
  `TestAudioSource` and `BroadcastSource` are drivable. See
  `docs/driven-pipelines.md`.

- **`WebRtcServer`: `Broadcast`s to WebRTC players, browsers among them.**
  One UDP socket serves every session, each a `str0m` connection run on the
  server's thread. A player is sent its path's broadcast as it is, from the
  newest keyframe, what it catches up on at twice the stream's speed — in
  the codecs WebRTC carries (H.264, H.265, VP8, VP9, AV1, Opus, G.711),
  H.264 from an MP4 turned into Annex B with its parameter sets ahead of each
  keyframe. A path in `WebRtcPathMode::Publish` takes one publisher, whose
  tracks are published into the broadcast once they have said what they are.
  Its sessions are spread over `WebRtcServerOptions::threads` threads, two by
  default, each with a UDP socket of its own; a thread carries some 70
  players of a 12 Mbit/s 1080p stream, and `webrtc_load` measures more — see
  [the streaming benchmarks](docs/benchmarks/streaming/).
  The server answers offers and leaves how they come to the application —
  `WebRtcServerHandle::play` and `publish` — and `serve_http` speaks WHEP and
  WHIP over HTTP, with a page at `/<path>` that plays the path in a browser.
  Data channels a client opens come out of `next_event`, and `send` answers
  on them. It runs under the `webrtc` feature, which now brings `socket2`
  for the socket's buffers. The `webrtc_server` example serves a file and a
  publisher's path; Chrome played it, and published to it from a canvas.

  ```rust
  let (server, handle) = WebRtcServer::bind("server", "192.168.0.5:8189", WebRtcServerOptions::default())?;
  handle.add_path("cam1", &camera, WebRtcPathMode::Play)?; // camera: a Broadcast
  DriverRunner::new(server).run()?;
  let _http = handle.serve_http("0.0.0.0:8080")?; // http://192.168.0.5:8080/cam1
  ```

- **`RtspServer`, behind `rtsp-server`: an RTSP server serving
  `Broadcast`s.** Each path is a broadcast, written by whatever pipeline
  the application has; any number of players play it — FFmpeg, VLC, a
  camera viewer — over TCP interleaved or UDP, each sent its own copy of
  the RTP packets and nothing decoded, from the newest keyframe, a slow one
  moved on to a newer keyframe rather than holding anyone back. A path in
  `RtspPathMode::Record` also takes one publisher's push (`ANNOUNCE` and
  `RECORD`, TCP or UDP; H.264, H.265, AAC, Opus, G.711) into its broadcast,
  which its players play on; one that goes silent for
  `RtspServerOptions::publisher_timeout` is let go of. It is a `Driver`, run
  with a `DriverRunner`, on a tokio runtime of its own inside its thread.
  The `rtsp_server` example serves a file as a camera and a publisher's
  path, and `rtsp_load` plays hundreds of players against it: on a Core
  i5-12400F it served 800 players of a 12 Mbit/s 1080p stream over TCP on
  1.75 cores and 71 MB, losing nothing, and UDP players at four to five
  times the processor each — see
  [the streaming benchmarks](docs/benchmarks/streaming/).

  ```rust
  let (server, handle) = RtspServer::bind("server", "0.0.0.0:8554", RtspServerOptions::default())?;
  handle.add_path("cam1", &camera, RtspPathMode::Play)?; // camera: a Broadcast
  DriverRunner::new(server).run()?;
  ```

- **`Broadcast`: one stream of packets, written once and read by many.**
  `Broadcast::publish` takes an input's tracks and hands out a
  `BroadcastSink` for each, the end of the branch carrying it; what they
  are handed is kept, nothing decoded or copied, and read by any number of
  `BroadcastSource`s — one at the start of a pipeline of its own, a
  recording, say — and `BroadcastReader`s, for code of your own, which wait
  on a thread or as a future. The writer never waits: a reader that falls
  behind further than `BroadcastOptions::retain` is moved on to the newest
  keyframe and told, and loses nothing else. A reader starts at the newest
  keyframe held, so it can show a picture at once; whole groups of
  pictures spanning at least `retain` are kept. A broadcast outlives its
  input: published again, after the input dropped, its readers are told one
  run ended and another began, and a `BroadcastSource` puts the next run
  after the last on one timeline. `Error::BroadcastError` and the
  `ElementType`s `Broadcast`, `BroadcastSink` and `BroadcastSource` come
  with it, and the `broadcast` example records and counts a file played
  twice into one.

  ```rust
  let camera = Broadcast::new("camera", BroadcastOptions::default());
  let sinks = camera.publish(&tracks)?; // one per track, for the input's pipeline
  let (recorder, tracks) = BroadcastSource::new("recorder", &camera)?;
  let mut viewer = camera.subscribe();
  ```

- **`RtspSource::listen`, receiving an RTSP push.** Rather than asking a
  server for a stream, the source takes the port in its URL and waits for
  one publisher — an encoder, `ffmpeg -f rtsp`, an `RtspMuxer` — to
  announce its streams and record into it, then hands on their packets as
  `RtspSource::open` does. The publisher picks TCP or UDP.
  `RtspListenOptions` bounds the wait for a publisher (`NoPublisher`) and
  how long one may go silent (`PublisherSilent`); its closing the
  connection ends the stream. It is FFmpeg's listener: one publisher per
  call, no authentication, no viewers. The `rtsp_listen` example counts
  what a publisher sends.

  ```rust
  let (source, streams) = RtspSource::listen("ingest", "rtsp://0.0.0.0:8554/live", RtspListenOptions::default())?;
  ```

- **`whisper-cuda` and `whisper-metal`.** whisper.cpp on an NVIDIA GPU
  with CUDA, and on a Mac's GPU, beside `whisper-vulkan`; see
  [features](docs/features.md) for what `whisper-cuda` needs to build.

- **`resize`, a saved software resize with sound.**
  `cargo run -p resize -- input.mp4 output.mp4 --width 160` decodes, scales
  and encodes SDR video to H.264, deriving an even height from the input's
  aspect ratio and copying its best audio stream without resetting time.

### Fixed

- **A peer cannot grow what a server holds of it without end.** A
  `WebRtcServer`'s WHIP and WHEP endpoint read a request's line and each
  header to its newline however long, so a client sending one without a
  newline grew it without bound; a line is now at most 8 KiB and a request
  has at most 100 headers, past which it is answered `431`. RTSP's blank
  lines between messages were skipped but never taken, so an `RtspServer`'s
  client — or the server an `RtspSource` reads — sending nothing else grew
  what was held without end; they now count toward a header block's limit.

- **What fuzzing found in the parsers of what comes over the network.**
  An `RtmpServer` that read a message header on a chunk stream in the
  middle of an unfinished message took the new message for the old one's
  rest: a shorter one's length less what was read overflowed. A header now
  begins a message, letting go of what was read of the other. An H.264
  sequence parameter set — what a camera or a publisher sends in band —
  whose sides each fit but whose picture in macroblocks did not, or whose
  scaling list had a `delta_scale` out of its range, overflowed working
  them out; the one is read, keeping no reordered pictures, the other is
  refused. Each overflow panicked a debug build and wrapped in a release
  one.

- **On Windows, nothing that waits on a UDP socket loses a datagram that
  comes as the wait ends.** A receive with a read timeout drops a datagram
  arriving as the timeout runs out, which neither it nor the next receive
  hands over. Every UDP socket here was read that way: the WebRTC server's,
  a `WebRtcPeer`'s, the WHIP and WHEP clients', and an `RtspSource`'s over
  UDP. A socket read every few milliseconds with little arriving between —
  a stream of thirty pictures a second at a quarter of a megabit — lost a
  packet every few seconds on an idle machine; a resend made most good, and
  under load, waiting on a resend, a WebRTC player stalled for a second and
  skipped a picture. The wait is now `WSAPoll`'s, and the receive after it
  has the datagram waiting. The `webrtc_load` example reads the same way,
  having counted the losses as the server's gaps.

- **On Windows, an `RtspSource` over TCP loses none of its stream as a wait
  ends.** A read timeout of a TCP stream drops what arrives as it runs out
  too — a write's worth taken from the stream and never handed over, once
  in some ten thousand waits — and an `RtspSource` read its connection that
  way while it played, interleaved RTP and all, and while it waited for an
  answer or a publisher. It waits as its UDP sockets do now. The
  `rtsp_load` example reads TCP without waiting, as it read UDP.

- **A live picture source keeps its time with the clock across dropped
  pictures.** A tick a slow downstream overran was dropped and the next
  picture stamped one after the last made, so every drop put the pictures
  a tick further behind the clock — and behind sound beside them, whose
  time is the clock's: six hours of a busy machine left an RTMP publish of
  a `TestVideoSource` and a `TestAudioSource` 7.4 s apart, and a recording
  of a compositor or a screen capture beside a microphone drifted the same
  way. Every video compositor (software, D3D11, D3D12, CUDA, Vulkan,
  Metal), every screen capture (DXGI, WGC, PipeWire, ScreenCaptureKit) and
  `TestVideoSource` now stamps the next picture after the ticks it
  dropped, and after an overrun the next tick is the first still ahead on
  the same cadence rather than one interval from when the overrun ended,
  so what was dropped is a whole number of ticks.

- **A WebRTC player catching up is paced on each track's own timeline.**
  `WebRtcServer` sends a joining player what the broadcast held at twice
  its speed, timed from the first packet sent; a track whose timestamps
  were ahead of the other's had its packets due that much later, and the
  first of them held up every packet after it, then let them all go at
  once. Each track is now timed from its own first packet.

- **A muxer gives a packet that says no duration the step from the last.**
  RTP carries no duration, so neither did a packet from an `RtspSource` or
  a `WebRtcSource`; written into an `HlsMuxer`, FFmpeg warned of every
  one — a log line per picture of each recording, some 2 GB a day of a
  server recording a dozen cameras — and an MP4 guessed each fragment's
  last. Every muxer's track now gives such a packet the step from the one
  before it on its track, as FFmpeg would guess, without holding a packet
  back; one that says its duration keeps it.

- **A `VulkanDecoder` goes on where the pictures change size, instead of
  crashing the process.** FFmpeg 8.0's Vulkan decoding, with the one
  decoding thread this opens it with, crashes inside libavcodec changing a
  context's size in place — `ffmpeg -threads 1 -hwaccel vulkan` does the
  same — which a camera reconfigured or a WebRTC sender lowering its
  resolution sets off. Where an H.264 or H.265 keyframe's parameter sets
  say other pictures, the decoder now hands on what it still holds and
  opens its codec context anew. Other codecs on Vulkan are not watched.

- **A `Broadcast` keeps its tracks' frame rate.** `Broadcast::tracks`, and
  every `BroadcastSource` and `BroadcastReader` of it, said `None` even
  for a track published from a stream that says its rate — a demuxed
  file's — so an encoder re-encoding what a broadcast carries had none to
  open at.

- **A live `DashMuxer` lets its old segments go on every FFmpeg.** FFmpeg's
  DASH muxer deletes through its file protocol, which an FFmpeg built with
  MSVC — vcpkg's for Windows — has no delete in: the manifest moved on and
  every segment stayed until the disk was full. The muxer now deletes them
  itself, keeping one more than FFmpeg would, so that where FFmpeg deletes
  it finds nothing to. (`HlsMuxer` was never affected: FFmpeg's HLS muxer
  deletes with `unlink` itself.)

- **A WebRTC client connects to a `WebRtcServer` at once, not a second
  later.** A client's DTLS ClientHello follows its nominating STUN at once,
  and the server, reading both in one batch, refused it: the session had
  not been polled since the nomination. Every connection waited a second
  for the client to send it again, and under load — retransmissions lost
  too, a second, two, four — a client gave up after ten. What comes from
  an address a session has taken STUN from now goes to that session, and
  a session that took STUN is driven before anything more is read, since
  str0m begins DTLS only as it is polled and drops what comes before.

- **`RtmpMuxer` takes a track from an MP4.** It kept the track's `avc1`
  tag, which FLV's muxer refuses as incompatible; it now writes its own.

- **A `WebRtcServer` session's id cannot be guessed.** The URL a WHEP or
  WHIP session is ended at — `DELETE /session/<id>` — was a count, so
  anyone could end anyone's session by counting. Half of its id is now
  drawn from a key the server alone knows, and a `SessionId` is shown as
  thirty-two hex digits where it was sixteen.

- **A `BroadcastSource` copied every packet it handed on.** Restamping a
  packet onto its own timeline cloned it, and `ffmpeg::Packet::clone`
  makes its copy writable — a full copy of a payload the broadcast still
  holds. Every reader paid it for every packet: a server's every player,
  and the memory of whatever such a reader held at once. It now shares the
  broadcast's payload, as `RtspServer`'s packetizer does. So do the muxers
  (`FileMuxer`, `Mp4Muxer` and the rest, and `SegmentedFileMuxer` before
  them, which copied twice), `ReplayBuffer` saving what it holds — a
  replay's worth of copies at once — and a `WebRtcTrackSink` shifting a
  negative first timestamp: each copied every packet only to change its
  timestamps or stream. What they write is byte for byte as before.

- **`WebRtcTrackSink` sent some of an MP4's H.264 pictures unconverted.**
  Given a demuxer's `avcC` parameters, it took any payload opening with a
  start code for one already Annex-B — and a NAL unit of 256 to 511 bytes
  has a four-byte length that opens `00 00 01`. Such a picture left with its
  length in front, which a receiver read as a NAL unit of type 25. A
  payload is now read as the record says first, and taken for Annex-B only
  where it does not read so.

- **A `WebRtcPeer` sends faster than a thousand packets a second.** Its loop
  took the deadline `str0m`'s pacer gives after every packet — one already
  past — as the end of its turn, and so sent one packet a turn; it now
  answers such a deadline and goes on sending.

- **A transcriber no longer loses words and repeats them where its rounds
  meet.** Each round heard the one before it again and kept the words the
  model timed inside its own window; the model times the same word hundreds
  of milliseconds apart in two rounds, so a word at the seam fell in
  neither window or in both — "재입국 충[격은 신혼 단계가 없]기", "빠르게 /
  빠르게". `large-v3-turbo` on FLEURS's Korean, streamed, missed 25% of the
  characters where the whole clip heard at once missed 3%. A word is now
  reported once two rounds in a row agree on the words up to it, by their
  text (LocalAgreement, after whisper_streaming), what is reported staying
  in the audio heard until it is cut away and given as the prompt; and
  silence is believed only once two rounds hear it, since whisper now and
  then passes over a sentence. The audio is cut at the quietest moment near
  where the model says two words part, and the model repeating itself — a
  line it made up, said again every round because the prompt held it — is
  passed over and the prompt dropped. Silence is believed only where the
  audio is quiet as well, a tail of sound the model left unheard is heard
  again on its own, and the last round keeps whichever of its hearing and
  the one before it holds more. The same clip misses 3.5%, at 6x real time
  where it was 7x; `medium`, which a loop had taken to 40% on Vulkan,
  misses 8%. See [the speech benchmarks](docs/benchmarks/speech/).

- **`AudioResampler` resamples a WAV with no channel mask.** A sound that
  says how many channels it has but not which — what such a WAV decodes to
  — was refused buffer after buffer as an input that had changed, so a mono
  float WAV made nothing at all. Such a layout is now taken as the default
  for that many channels, as libswresample itself takes it. `SwAudioEncoder`
  and `AudioMixer`'s inputs, which resample the same way, are fixed with it.

- **File-export examples protect their inputs and existing outputs.**
  `remux`, `transcode` and `resize` refuse input/output aliases and require
  `--overwrite` for an existing destination. They write beside it to a
  temporary file and publish only a completed, closed export; failure leaves
  the destination unchanged. The library's `FileMuxer` API is unchanged.
- **File-export examples fail on bus errors.** They return a failure exit
  status without `wrote ...`, rather than reporting partial output as success.
  The library still reports per-buffer errors and continues by its own policy.

## 0.4.1

`Report`, which the getting-started guide's programs return from `main` and
0.4.0 did not have, so that the guide builds against the release it tells
a reader to install; a dead branch reported once, and a failure logged as
the element that failed. crates.io's README now links the documentation of
the version it belongs to.

### Added

- **`media_pp::Report`, for `main` to return.** A `main` that returns a
  `Result` prints its error in `Debug`, which for `Error` is its structure;
  a `main` returning `Result<(), media_pp::Report>` prints what the error's
  `Display` says instead — `Error: could not open video.mp4: No such file
  or directory`, or a refused link naming the element to put between. `?`
  turns any of this crate's errors into one, so the body of `main` is
  written as before. Getting started and the examples' `main`s use it.

### Changed

- **`MediaKindSet`, `MemoryDomainSet` and `PixelLayoutSet` debug by name** —
  `MediaKindSet(VideoPacket)` and `PixelLayoutSet(any layout)` rather than
  `MediaKindSet(1)` and `PixelLayoutSet(63)` — so a refused link read in
  `Debug` says what did not fit.
- **crates.io's README links the documentation of its own version.** Its
  links to the guides, the changelog and the rest of the repository named
  `main`, which can be ahead of the release: 0.4.0's sent a reader to a
  guide using `Report` before it was published. They now name the
  release's tag, and its docs.rs link the release's own version rather than
  the newest (`scripts/crate-readme.sh`); the repository's README links
  relatively, to the branch being read.

### Fixed

- **A dead branch says so once.** A `Queue` whose worker a panic had ended
  answered every buffer after it with the closed channel, and a `Tee` in
  front posted each one — thirty errors a second from one branch, for as
  long as its siblings ran. The panic is posted once, by the worker; the
  first buffer after it is answered with the closed channel and the rest
  are dropped quietly.
- **A failure is logged as the element that failed.** An error a queue, a
  tee or a demuxer posted for an element after it went to the bus under
  that element's name but into the log under the poster's alone; the record
  now carries the failing element's identity and names the one that
  reported it.

## 0.4.0

macOS, every GPU, and video analysis. The crate runs on macOS —
VideoToolbox decoding and encoding, Metal scaling, compositing and
rendering, ScreenCaptureKit, AVFoundation and Core Audio, and `Player` —
and on any GPU through Vulkan Video on Linux and Windows, with D3D12 now
end to end on Windows. Detectors, classifiers, embedders, a tracker,
counting in zones and across lines, and finding cuts run on the CPU, CUDA,
Core ML and DirectML, and what they find is drawn or hidden. Most of what
breaks is how an element of your own is written: `Source`, `Filter` and
`Sink` are the traits you implement, the stage the framework makes of one
keeps who it is, and the end of a stream is an event, not a buffer.

### Breaking

- **A `Source`, `Filter` or `Sink` is no longer an `Element`.** The stage
  the framework makes of one keeps its name and log identity, so an element
  of your own is its work alone: delete its `impl Element` — `name`,
  `element_type`, `pp_log`, `pp_log_mut` — and the `name` and `pp_log`
  fields only that impl read. It goes by its type's name, or by the one
  its new `named` gives it where it is wired:

  ```rust
  ctx.branch().pipe(Invert.named("invert")).to(Seen.named("seen"))?;
  ```

  What its `attach_context` did goes into its new `attached`, which is
  handed the same `Context` (`cx.context()`) and its log identity in that
  pipeline (`cx.pp_log()`), to keep for its `pp_*!` records. `IntoFilter`,
  `IntoTerminal` and `IntoSource` take the `Named` one as they take the
  element. `Source::outputs` returns `Option<Vec<SrcPad>>`: `None`, the
  default, is the one output named for the source, which a source no
  longer has a name to make itself. `SeekableSource` asks only `Send` of
  what implements it.

- **`WasapiRendererError` and `PipeWireAudioRendererError` lose
  `PlaybackClockAlreadyBound` and `PlaybackClockBoundAfterStart`.** Nothing
  has raised them since the pipeline hands a renderer its playback clock as
  it is wired; an arm that names them goes.

- **`ReplayBufferError` has a `Replace` variant**: a clip written whole
  that could not take the place of the file at its path — see Fixed. A
  `match` on `ReplayBufferError` needs an arm for it.

- **`ElementType` has `D3d12OrtDetector`, `D3d12OrtClassifier`,
  `D3d12OrtEmbedder`, `D3d12CutDetector`, `D3d12DetectionOverlay`,
  `D3d12ToneMap`, `D3d12VideoEncoder` and `D3d12VideoCompositor`
  variants**, for the D3D12 detector, classifier, embedder, cut detector,
  overlay, tone map, encoder and compositor — see Added. A `match` on `ElementType` needs an arm for
  each.

- **`MetalOrtEmbedder::new` takes `MetalOrtEmbedderOptions`, and Core ML
  keeps what it compiles.** Beside the `OrtEmbedderOptions` every embedder
  takes, it says where the model Core ML compiles is kept (`model_cache`),
  as `MetalOrtDetectorOptions::model_cache` now does for a detector — the
  Metal counterpart of `engine_cache`. `None`, the default, keeps it in
  `~/Library/Caches/media-pp/coreml`; the classifier keeps its there too.
  Without one every session converted and compiled its model again: four
  models opened in 2.0 s that now open in 0.4 s from the cache. A model
  fixed to one batch is kept apart from the same model fixed to another.

  | Before | Now |
  |---|---|
  | `MetalOrtEmbedder::new(name, model, options)` | `MetalOrtEmbedder::new(name, model, MetalOrtEmbedderOptions { embedder: options, ..MetalOrtEmbedderOptions::default() })` |

- **`Hiding` has a `feather`.** A struct written out in full names it;
  one built from `Hiding::new` and `..` gets 0, the hard edge it had.

- **`CudaOrtEmbedder::new` takes `CudaOrtEmbedderOptions`.** Beside the
  `OrtEmbedderOptions` every embedder takes, it says with `ort-tensorrt`
  where TensorRT keeps the engine it builds (`engine_cache`), as
  `CudaOrtDetectorOptions::engine_cache` does for a detector; `None`, the
  default, keeps it where it was.

  | Before | Now |
  |---|---|
  | `CudaOrtEmbedder::new(name, device, model, options)` | `CudaOrtEmbedder::new(name, device, model, CudaOrtEmbedderOptions { embedder: options, ..CudaOrtEmbedderOptions::default() })` |

- **`OrtDetectorOptions` has a `model`, and `Detection` has
  `landmarks`.** `DetectorModel::Yolo`, the default, is what every
  detector read before. `DetectorModel::Custom { input, decoder }` runs a
  model this crate does not read: `ModelInput` says the values it wants —
  RGB or BGR, each times a scale plus a bias — and the application's own
  `DetectorDecoder` reads one picture's outputs into `ModelBox`es in pixels
  of the model's input, which the detector maps back onto the picture, turn
  and letterbox and all, on the CPU, CUDA and Metal alike;
  `non_max_suppression` is there for it. A `Detection`'s `landmarks` are
  the points a model found on the object, as fractions of the picture —
  none for YOLO.

  | Before | Now |
  |---|---|
  | `OrtDetectorOptions { conf_threshold, iou_threshold, labels, interval }` | the same with `model: DetectorModel::Yolo`, or `..OrtDetectorOptions::default()` |

- **`Hiding` has a `shape`.** `HideShape::Rectangle` covers the box as
  before; `HideShape::Ellipse` covers only the ellipse inside it, a pixel —
  or a 2x2 block of NV12 colour — where its centre is inside. The CPU and
  CUDA overlays cover the same pixels with the same bytes, for a mosaic, a
  blur and a fill, and `MetalDetectionOverlay` the same bytes as the CPU's
  too. `cuda_detect`, `cuda_track` and `metal_detect` take `--hide
  face=mosaic,ellipse`.

  | Before | Now |
  |---|---|
  | `Hiding { style, margin }` | `Hiding { style, margin, shape: HideShape::Rectangle }`, or `Hiding { margin, ..Hiding::new(style) }` |

- **`OrtDetector` is `SwOrtDetector`, a filter that hands each picture on
  with what it found.** Renamed as the scalers are named, beside the new
  `CudaOrtDetector` (see Added). It was a sink that called a closure; it
  now passes every picture through
  unchanged, carrying a `Detections` as its metadata (see Added), so what
  it found reaches whatever comes after it. It takes system-memory video of
  any format and size and fits each picture to the model itself, so no
  `SwScaler` is needed in front of it, and it reads YOLOv10 and YOLO26
  outputs (`[1, boxes, 6]`) as well as YOLOv8 and YOLO11's. A
  `Detection`'s box is now fractions of the picture rather than pixels of
  the model's input. It moved from `elements::sink` to `elements::filter`,
  under the same `elements::` path.

  | 0.3 | Now |
  |---|---|
  | `OrtDetector::new(name, model, conf, iou, \|frame, found\| ..)` as the branch's end | `SwOrtDetector::new(name, model, OrtDetectorOptions { conf_threshold, iou_threshold, labels: None })` piped, then read `buffer.metadata()?.get::<Detections>()` downstream — an `AppSink` does what the closure did |
  | `COCO_CLASS_LABELS[detection.class_id]` | `detections.label(&detection)` |
  | `SwScaler` to 640x640 RGB24 before it | nothing |
  | `detection.x` in model pixels | `detection.x * picture_width` |


  `MediaBuffer::Packet`, `Video` and `Audio` hold `PacketBuffer`,
  `VideoBuffer` and `AudioBuffer` in place of the bare `Arc`, each the
  `Arc` beside the buffer's `Metadata` (see Added). The wrapper
  dereferences to the `Arc`, so matching and reading are unchanged —
  `MediaBuffer::Video(frame) => frame.width()` still compiles — and what
  changes is making one and taking the `Arc` out:

  | 0.3 | Now |
  |---|---|
  | `MediaBuffer::Video(Arc::new(frame))` | `MediaBuffer::Video(Arc::new(frame).into())`, or `MediaBuffer::video(frame)` |
  | `MediaBuffer::Packet(Arc::new(packet))` | `MediaBuffer::packet(packet)` |
  | `MediaBuffer::Audio(Arc::new(frame))` | `MediaBuffer::audio(frame)` |
  | `frame.clone()` for the `Arc` | `frame.payload().clone()`, or `frame.into_payload()` |
  | `&*packet` for the `ffmpeg::Packet` | `&**packet` |

  A wrapper is `DerefMut` too, so `Arc::get_mut(frame)` on a buffer this
  element alone holds still writes through it.

- **The traits an element implements are renamed, so the plain names are
  the ones you write.** `Source`, `Filter` and `Sink` are now the traits an
  element of your own implements — a source asked for the next thing, a
  filter that makes buffers of buffers, a terminal that does something
  with each one; each is described under Added. The traits those names had
  in 0.3 are the ones an element implements only to route the stream
  itself, and carry `Raw`:

  | 0.3 | Now |
  |---|---|
  | `Sink` | `RawSink` |
  | `Source` | `SrcPads` |
  | `Filter` | `RawFilter` |
  | `SourceElement` | `RawSource` |

  So `impl Sink for` an element that routes the stream is
  `impl RawSink for`, and `impl Source for` is `impl SrcPads for`. A
  terminal or a filter of your own is shorter written as the new `Sink`
  or `Filter`; `custom_element` writes one of each kind. The entries below
  use the new names.

- **A terminal or a filter whose kind is forgotten is a `BoxSink` or a
  `BoxFilter`, not a bare `Box<dyn …>`.** What was a `Box<dyn Sink>` — a renderer picked
  at run time, a muxer's track from `MuxerSinks::take`, an input from
  `AudioMixer::add_source` or a compositor's `CompositorInput::sink`, a
  bridge's feeding end from `PipelineBridgeHandle::connect` — is a
  `BoxSink`, and what was a `Box<dyn Filter>` a `BoxFilter`, which is
  what `RackHandle::replace` takes. `BoxSink::new` and `BoxFilter::new`
  make one from a `Sink` or a `Filter` of your own and from any element of
  this crate alike, `.to()` and `.pipe()` take one as they take what it
  holds, and each derefs to the element inside. So
  `let renderer: Box<dyn Sink> = Box::new(window)` is
  `let renderer = BoxSink::new(window)`, and
  `rack.replace(vec![Box::new(effect)])` is
  `rack.replace(vec![BoxFilter::new(effect)])`. `IntoTerminal::into_terminal`
  and `IntoFilter::into_filter` return one too.

- **The end of a stream is `StreamEvent::Eos`, not a buffer.**
  `MediaBuffer` is data only: `MediaBuffer::Eos` and `is_eos` are gone,
  and the end travels as an event in the stream, in order with the
  buffers, as a segment already did. The `stream` module is public, and a
  `RawSink` hears the end through `stream_event(&StreamEvent)` — a muxer
  track, `FrameCounter` and a sink of your own alike — never through
  `consume`. An element in the middle of a chain never passes the end on
  itself: it reacts, pushing whatever it still holds, and the pipeline
  passes the end on after it. So:
  - `handle.push(MediaBuffer::Eos)` on an `AppSourceHandle` is
    `handle.finish()`;
  - `sink.consume(MediaBuffer::Eos)` on a sink driven by hand is
    `sink.stream_event(&StreamEvent::Eos)`;
  - a `consume` that matched `MediaBuffer::Eos` moves that arm into
    `stream_event`, and a filter of your own that pushed `Eos` on after
    draining now only drains. `StreamEvent` is `#[non_exhaustive]`, so the
    `match` there ends in a `_ => {}` arm:
    `match event { StreamEvent::Eos => self.finish()?, _ => {} }`.

- **`RawSink::control` is gone, and control stays inside the crate.** A sink
  of your own hears a pause, a resume and a stop through the new
  `RawSink::pausing`, `RawSink::resuming` and `RawSink::stopping` hooks, which do
  nothing by default, and a seek through the flushed
  `StreamEvent::Segment` its `stream_event` is handed — where it lets go
  of what it held from before the seek. `ControlMsg`, `PrerollContext`,
  `ControlSender`, `ControlReceiver`, `control::channel`,
  `ControlOutcome`, `drain_control` and `control::deliver` are no longer
  public; `control` keeps `SeekError`, `SeekRejection`,
  `SeekRejectReason` and `PrerollError`. A `consume`-only sink needs
  nothing; one that matched `ControlMsg::Pause`/`Resume`/`Stop` in
  `control` writes those arms as the hooks, and one that matched
  `ControlMsg::Flush` or `Seek` reacts to a segment with `flushed` set.

- **`RawSource` is implemented only by this crate's sources.** Its
  loop answers the pipeline through a channel that is no longer public.
  A source of your own is a `Source`, which `Pipeline::new` and
  `add_source` take as they take any source — with several outputs, a
  segment of its own, or seeking and playing backwards where it can (see
  Added). `RawSource` stays nameable, as a bound.

- **`AppSink::with_control` is `AppSink::with_events`**, whose closure is
  handed each `&StreamEvent` — the end, and each segment, a seek's one
  flushed — instead of each `ControlMsg`. A closure that finalized on
  `MediaBuffer::Eos` in `consume` does it on `StreamEvent::Eos` there; one
  that reset on `ControlMsg::Seek` resets on a flushed
  `StreamEvent::Segment`. `AppSink::new` lets every event by, as it let
  every control message by.

- **`VideoCompositorOptions` has a `mode`.** Every video compositor —
  `SwVideoCompositor`, `D3d11VideoCompositor`, `CudaVideoCompositor` —
  takes it, and `RenderMode::Live` is what they all did before. A struct
  literal naming every field adds `mode: RenderMode::Live`; one ending in
  `..Default::default()` needs nothing. Each compositor's error enum gains
  `FixedFrameRate` and `UntimedFrame`, which a `match` on it needs arms for.

- **`AudioMixerOptions` has a `mode` too**, for the same choice:
  `RenderMode::Live` is what the mixer did. `AudioMixerError` gains
  `FixedMixFormat` and `UntimedFrame`.

- **A `CudaVideoCompositor` text layer stacks by `z_index`**, as the other
  compositors' do, where it used to be drawn over every video layer. It
  starts at zero, and a video layer of the same `z_index` is drawn under
  it, so a text layer is still over every video layer at zero or below;
  one over a video layer raised above zero calls
  `CudaTextLayerHandle::set_z_index`.

- **The `vulkan` feature needs an FFmpeg built with Vulkan**, and the
  Vulkan headers, to build: its new elements are FFmpeg's Vulkan frames,
  and the build script generates the bindings to FFmpeg's Vulkan hardware
  context from the installed `hwcontext_vulkan.h` — see the `vulkan`
  feature's requirements in [`docs/features.md`](docs/features.md).
  vcpkg's `ffmpeg[vulkan]` brings both. A program using only
  `VulkanWindowRenderer` builds against such an FFmpeg as before.

- **`MemoryDomain` has a `Vulkan` variant**, for frames in Vulkan images.
  A `match` on `MemoryDomain` needs an arm for it.

- **`DecodeTarget`, `EncodeInput` and `EncodePath` have Vulkan variants**
  (`DecodeTarget::Vulkan` and `EncodeInput::Vulkan` with the `vulkan`
  feature, `EncodePath::Vulkan` always). A `match` on any of them needs an
  arm for it.

- **`Produced` has `On` and `Segment` variants**, for a source with
  several outputs and one that begins a segment of its own — see Added.
  A `match` on `Produced` needs arms for them.

- **`ReversibleSource::finish_stretch(bus)` is
  `ReversibleSource::stretch_complete()`**: a source says whether the
  stretch under way has been read to its end, and the framework asks it
  for more until it has, as a `Finish` arrives while playing backwards.
  `SeekableSource` asks only `Send` of what implements it now, so a
  `Source` can — see Added.

- **`RackError` has a `Waits` variant**: a `Rack` refuses a `Pacer` or a
  `VideoSynchronizer`, which it could only run on the thread feeding it —
  see Added. A `match` on `RackError` needs an arm for it.

- **`ElementType` has `CoreAudioRenderer` and `CoreAudioCaptureSource`
  variants**, for the macOS audio elements, `VideoToolboxDecoder`,
  `VideoToolboxEncoder`, `VideoToolboxUpload` and `VideoToolboxDownload`,
  `AvFoundationCaptureSource`, `ScreenCaptureKitSource`,
  `MetalVideoCompositor`, `MetalWindowRenderer`, `MetalScaler`,
  `MetalConverter`, `MetalChromaKey` and `MetalVideoEffect`, and
  `MfVirtualCamera` — see Added. A `match` on `ElementType` needs an arm
  for each.

- **`MemoryDomain` has a `VideoToolbox` variant**, for frames in Core Video
  pixel buffers — see Added. A `match` on `MemoryDomain` needs an arm for
  it.

- **`DecodeTarget`, `EncodeInput` and `EncodePath` have VideoToolbox
  variants** (`DecodeTarget::VideoToolbox` and `EncodeInput::VideoToolbox`
  with the `videotoolbox` feature, `EncodePath::VideoToolbox` always). A
  `match` on any of them needs an arm for it.

- **`FileMuxer::create` and `SegmentedFileMuxer::create` take an
  `Mp4Layout`**, how an MP4 or MOV file keeps its index — see Added.
  `Mp4Layout::Standard` is the layout every file had; a file of another
  container takes it alone, and asked for another is refused as the new
  `FileMuxerError::NotMp4` before anything is created. A `match` on
  `FileMuxerError` needs an arm for it.

  | Before | Now |
  |---|---|
  | `FileMuxer::create(path)` | `FileMuxer::create(path, Mp4Layout::Standard)` |
  | `SegmentedFileMuxer::create(policy, naming)` | `SegmentedFileMuxer::create(policy, Mp4Layout::Standard, naming)` |

### Added

- **`MediaBuffer::pts` and `MediaBuffer::time_base`** say a buffer's
  timestamp and the unit it counts in, a packet's, a picture's or a
  sound's alike, without matching on which it is.

- **`D3d12VideoEncoder::for_stream`** says the colour of the stream it
  re-encodes, as `SwEncoder`'s, `CudaEncoder`'s and
  `VideoToolboxEncoder`'s do, in place of choosing between `new` and
  `with_color` by hand.

- **`scripts/setup-ffmpeg` builds the FFmpeg media-pp needs.** FFmpeg 8.0
  exactly, through vcpkg at the revision CI uses, on Windows
  (`setup-ffmpeg.ps1`), Linux and macOS (`setup-ffmpeg.sh`), and points
  `FFMPEG_DIR` and the loader's path at it — since no prebuilt 8.0 is
  published any more and 8.1's headers do not build.

- **The `doctor` example says what a machine gives media-pp**: the FFmpeg
  a build finds and whether it is 8.0, its encoders and hardware device
  types, and whether each GPU backend opens a device there, each problem
  with what to do about it — `cargo run -p doctor`.

- **A user's guide**: [`docs/README.md`](docs/README.md) indexes the
  documentation, and `docs/guide/` explains how a pipeline works, which
  backend to choose, what to use for a common job, and how to fix what
  goes wrong. Getting started, the element list, the features and the
  examples were rewritten to be read in that order.

- **`D3d12VideoCompositor` composites D3D12 pictures**, with the `d3d12`
  feature: what the CUDA, Metal and Vulkan compositors do, on any
  vendor's GPU, through the same `VideoLayer`, text layers, handles and
  `VideoCompositorControl` traits, live or offline. Each layer is sampled
  from its NV12 picture straight into its place by compute shaders of this
  crate's own — scaled, cropped, read in its own colour and mixed at its
  opacity in one pass a plane — onto an NV12 BT.709 canvas, handed on as
  an ordinary D3D12VA frame; an unchanged scene hands its last frame on
  again. A layer it cannot draw is said on the bus under its own name,
  and the rest of the frame is drawn.

- **`ObjectTracker` follows D3D12 pictures by look on the GPU**, with the
  `d3d12` feature: what `cuda-visual-tracking` does for CUDA pictures and
  `metal` for VideoToolbox ones, on any vendor's GPU. Every followed
  object's filter is sampled from the NV12 or P010 picture where it is,
  transformed and matched at once by kernels of this crate's own, on a
  queue of the pictures' own device; four numbers an object come back.
  Before, a tracker could not read a D3D12 picture at all and followed
  its objects by motion alone. Where the kernels cannot be made, it says
  so once and goes on by motion.

- **`D3d12VideoEncoder` encodes D3D12 pictures through D3D12 Video**, with
  the `d3d12` feature, on any vendor's encode block: NV12 frames — a
  decoder's, an upload's, an overlay's or a tone map's — handed to FFmpeg's
  `hevc_d3d12va` as they are, so a D3D12 pipeline decodes, detects, hides
  and encodes without a picture leaving the GPU. H.264 (`h264_d3d12va`)
  opens where the linked FFmpeg has it, which 8.0 does not. No B-frames
  unless asked for: FFmpeg's default inserts them, and NVIDIA's D3D12
  encode fails at the first frame given them.

- **`D3d12ToneMap` brings HDR D3D12 pictures to SDR**, with the `d3d12`
  feature: what `D3d11ToneMap` draws on D3D11, by `core/tone_map.rs`'s
  definition, on any vendor's GPU. A P010 or NV12 picture tagged HLG or
  PQ comes out SDR NV12 in BT.709 limited range — NV12 rather than the
  BGRA of the D3D11 and CUDA elements, as every D3D12 element here reads
  NV12 — each sample within 2.5 of 255 of the definition.

- **`D3d12DetectionOverlay` draws and hides detections on D3D12
  pictures**, with the `d3d12` feature: what `CudaDetectionOverlay` and
  `MetalDetectionOverlay` do, on any vendor's GPU — boxes, labels, zones
  and lines painted on a copy of each NV12 picture, and faces hidden by a
  mosaic, a blur or a fill, of the box or the ellipse inside it, faded at
  the edge where asked, in NV12 and in HDR P010. What the CPU works out
  with a division or a square root it works out here too and hands to the
  shaders, as HLSL's are not correctly rounded, so the bytes are
  `SwDetectionOverlay`'s exactly.

- **`D3d12CutDetector` finds cuts in D3D12 pictures**, with the `d3d12`
  feature: what `CudaCutDetector` and `MetalCutDetector` do, on any
  vendor's GPU. A compute shader sums each cell of an NV12 or P010
  picture's thumbnail where it is, and only the sums come down to be
  divided as the CPU divides its own, so the thumbnail is
  `SwCutDetector`'s float for float.

- **`D3d12OrtDetector` detects on D3D12 pictures through DirectML**, with
  the `ort-directml` feature: object detection on the GPU of a Windows PC
  without NVIDIA — any vendor's, through its own D3D12 driver. A compute
  shader fits each NV12 picture, whole and in tiles, into a buffer on the
  `D3d12Gpu` it is given, and ONNX Runtime's DirectML provider, handed that
  GPU's own device and queue, reads it there; the outputs are decoded as
  every detector's are. On an RTX 3050 a RetinaFace R50 in half precision
  ran 6.7 ms an input, 1.2-1.4x TensorRT and many times the CPU. A model
  runs fastest at opset 20 or lower with half-precision weights — see
  `docs/d3d12-inference.md`. `D3d12OrtEmbedder` beside it makes what a
  detector found into vectors the same way, each object read through its
  five points or its box by a second shader; on DirectML, SFace and OSNet
  made vectors 0.9975 and 0.9999 alike the CPU's. `D3d12OrtClassifier`
  names each object with a second model, its box cut by the first shader;
  through DirectML, MobileNetV2 named ten boxes of a picture as the CPU
  did, the scores within 0.06. All three take HDR P010 pictures, HLG or
  PQ, as the CUDA and Metal ones do: through an SDR copy a compute shader
  makes by the same tone map, within 2 of 255 of it.

- **An MP4 or MOV file can be laid out fast-start or in fragments**
  (`Mp4Layout`, given to `FileMuxer::create` and, for every segment,
  `SegmentedFileMuxer::create`).
  - `FastStart` moves the index in front of the samples as the file is
    finished, so one streamed — in a browser, a messenger's preview, a cloud
    drive's player — plays as its first bytes arrive. Finishing reads and
    writes the file once more.
  - `Fragmented` puts the track list first, then a fragment (`moof` +
    `mdat`) at each video keyframe — every second for sound alone — each
    handed to the file as it is cut. A recording cut off before its
    trailer, by a process that dies or a machine that loses power, still
    opens with every fragment written before the one under way, where a
    standard MP4 opens not at all; it is also the shape a browser streams
    from.
  - `ReplayBuffer` saves an MP4 or MOV clip fast-start: a clip saved is one
    shared as it is, and moving a short clip's index costs next to nothing.

- **The window renderers show a phone's portrait recording upright.**
  `D3d11WindowRenderer`, `D3d12WindowRenderer`, `VulkanWindowRenderer` and
  `MetalWindowRenderer` — and so `VideoWindow` and `Player` — read each
  picture's display matrix (`Orientation::of`), fit the window to the
  picture as it is shown, and sample it turned. The elements that make a
  picture anew with their own frame rather than FFmpeg's copy of its
  properties — `SwScaler`, `D3d11Upload`, `D3d11Scaler`, `D3d11Download`,
  the D3D11 tone map, chroma key and video effect, and the software chroma
  key and video effect — carry the matrix on, where before a picture
  through them was taken as stored. On Windows each of a quarter turn
  either way and none is drawn and read back; Metal is not run here.

- **`for_stream` on `CudaEncoder`, `SwEncoder` and `VideoToolboxEncoder`
  keeps an HDR10 file's mastering display and light levels.** Re-encoding
  a PQ file's pictures, the encoder takes what the source stream says:
  its colour, as `with_color(stream.color())` did, and the mastering
  display and content light levels a display dimmer than the master
  tone-maps by. `FileMuxer` writes them beside the track (`mdcv`, `clli`)
  from the encoder's parameters, and NVENC and `libsvtav1` into the
  bitstream too; FFmpeg's VideoToolbox encoder does not. Before, they
  were dropped, and each player guessed. A PQ file whose container keeps
  neither, its bitstream does, has them read from its first picture when
  `FileDemuxer` opens it — decoded in software, that stream only — so its
  `StreamInfo` says them as one whose container kept them would. Dolby
  Vision's per-scene metadata is still not kept.

- **DETR-family detectors in every ORT detector (`DetectorModel::Detr`).**
  D-FINE, RT-DETR and RF-DETR as transformers exports them: `logits`, each
  class's logit for each of the queries, and `pred_boxes`, each box's
  centre and size as fractions of the input — by those names or by their
  shapes. Each query is its best class at that logit's sigmoid; the input
  is filled as the variant's `ModelInput` says (RGB from 0 to 1 for D-FINE
  and RT-DETR, less ImageNet's mean and over its deviation for RF-DETR).
  These are Apache-2.0 detectors where Ultralytics' YOLO is AGPL. On an
  RTX 3050 through TensorRT, D-FINE nano runs a 1080p file at 255 pictures
  a second where YOLO11n runs 375, decode included; it scores the same
  people lower than YOLO does — a person YOLO scores 0.5 at about 0.19 —
  so thresholds tuned to YOLO need tuning anew.

- **A hidden box can fade into the picture (`Hiding::feather`).** From the
  edge of the box, or of the ellipse inside it, the mosaic, blur or fill
  is mixed over what was there less and less toward the edge, by a smooth
  step, over that fraction of the way to the middle; a rectangle's corners
  are rounded as far. The CPU's, CUDA's and Metal's overlays write the
  same bytes, NV12, BGRA and P010 alike.

- **`MetalConverter` takes 10-bit, and brings HDR to SDR.** A P010
  VideoToolbox frame — what `VideoToolboxDecoder` makes of 10-bit video —
  comes out BGRA as NV12 does. One tagged PQ or HLG is brought to SDR
  BT.709 on the way by `core/tone_map.rs`'s definition, the one
  `CudaConverter` and `D3d11ToneMap` draw with, and is tagged BT.709. So a
  10-bit file, HDR or not, plays through `MetalRenderer` behind a
  `MetalConverter`.

- **Detectors look in tiles.** `OrtDetectorOptions::tiles: Some(Tiles)`
  looks at each picture in overlapping tiles — three across and two down a
  wide one by default, two across and three down a tall one, a quarter
  overlapping — as well as whole, each fitted to the model as a picture of
  its own, on the CPU, CUDA and Metal. A face found both whole and in a
  tile keeps the whole picture's box with the higher score, since a tile
  can cut a large face in two; what only tiles found is kept. The tiles
  run as a batch beside their picture, `max_batch` inputs a run. On eleven
  public videos RetinaFace ResNet50 missed 19% of 976 faces looking in
  tiles where it missed 46% without, and found more than half of those
  under 24 pixels where it found none. On an RTX 3050 through TensorRT,
  YOLO11n-face ran 112 pictures a second with tiles and `max_batch` 7
  against 362 without; `cuda_detect --tiles`. On an M5 through Core ML,
  YOLO11n ran 42 pictures a second with tiles and `max_batch` 7, 34 with
  `max_batch` 1, against 210 without, and the tiles find through Core ML
  where they find on the CPU; `metal_detect --tiles`. `Tiles::choose`, a
  `TileChooser` of the application's own, picks the tiles each picture is
  looked in from what it carries in — an earlier detector's
  `Detections` — through a `TiledPicture`; the whole picture is always
  looked at. Looking for faces only in the tiles a small person's head lies
  in missed the same faces as all six tiles on the eleven videos, at five
  runs of RetinaFace a picture instead of seven, three where a few people
  are.

- **A box followed by look says how sure the look was.** On a picture no
  detector looked at, `ObjectTracker` with `visual` puts each object where
  its correlation filter finds it; `Detection::look` now carries that
  filter's peak-to-sidelobe ratio — over about 7 the box is where the look
  found the object, under it where its motion would have taken it — on the
  CPU, CUDA and Metal alike, `None` on objects detected. A block followed
  scored 19 to 26 while there and under 4 once gone. What follows one
  object forward can end where its look is lost rather than after a fixed
  time.

- **Embedders: what a detector found, made into vectors.** `SwOrtEmbedder`
  and `CudaOrtEmbedder` run an embedding model on each object and put its
  vector, of length 1, in the object's `Detection::embeddings` — vectors
  that lie near each other where two objects look alike, compared with
  `Embedding::similarity`, for telling which boxes across a video are the
  same thing. `OrtEmbedderOptions::align` says how an object is cut:
  `Alignment::FivePoints` straightens a face by the five points a
  RetinaFace detector finds onto the ArcFace template a face recognition
  model is trained on, `Alignment::Box` stretches a whole object's box.
  The model's values are a `ModelInput`, as a detector's are. A followed
  object is embedded once and kept, as the classifiers keep their answers
  (`reembed`). On CUDA a new `warp_nv12`/`warp_bgra` kernel reads each
  object through its map, between the four pixels around each point, and
  the picture never leaves the GPU; its vectors are the CPU's to 0.998 of
  a cosine on NV12 and 0.99998 on BGRA. On five people of Tears of Steel,
  AdaFace IR-18's vectors put each of 34 faces nearest one of its own
  person, as a reference alignment in Python did. `MetalOrtEmbedder` does
  the same on VideoToolbox pictures through Core ML, a Metal kernel reading
  each object through its map; on three faces of a NASA crew portrait found
  by RetinaFace, SFace's vectors were the CPU's to a cosine of 0.9998 on
  NV12 and 0.99999 on BGRA.

- **RetinaFace, a face detector, in every ORT detector.**
  `DetectorModel::RetinaFace` reads biubug6's `Pytorch_Retinaface` exports,
  MobileNet or ResNet: BGR from 0 to 255 less a mean, made on the GPU by a
  new `swap_planes` kernel and `scale_planes`; three outputs of a row per
  anchor, told apart by their shape, whatever an export names them. Each
  face is class 0, named `face`, with its eyes, nose and mouth corners as
  `Detection::landmarks`. On a Tears of Steel clip it finds the faces a
  reference decoding in Python does, in the same places; MobileNet 0.25
  runs about 440 pictures a second on an RTX 3050 through TensorRT, decode
  included. `cuda_detect --retinaface --hide face=mosaic,ellipse` hides
  every face, and `metal_detect --retinaface` the same on a Mac, where it
  finds the CPU's faces and points through Core ML.

- **An iPhone's HDR goes through in ten bits.** `CudaFrameFormat::P010` is
  what NVDEC decodes 10-bit video to. `CudaEncoder` takes it and encodes
  HEVC Main 10 — `with_color` saying BT.2020 and HLG or PQ, as
  `StreamInfo::color` reads them — and refuses H.264, which NVENC has no
  ten bits of, with `TenBitCodec`. `CudaUpload`, `CudaDownload` and
  `CudaScaler` take it; `CudaConverter` refuses to make it
  (`UnsupportedOutput`) and `CudaVideoCompositor` to compose on it
  (`UnsupportedCanvas`). The detection overlays hide in a P010 picture —
  mosaic, blur and fill, box or ellipse, in its ten bits, the CPU and CUDA
  ones writing the same bytes — where they draw nothing; a fill's SDR
  colour is put at BT.2408's reference white, HLG's 75% or PQ's 203 nits.
  `CudaOrtDetector` and `CudaOrtClassifier` take an HLG or PQ P010
  picture and hand the model an SDR copy of it, tone-mapped as
  `CudaConverter` maps it: on an HLG and a PQ version of the people clip
  YOLO11n found the SDR clip's boxes again, at an IoU of 0.99. What they
  find goes on the ten-bit picture. `StreamInfo::pixel_format` says what
  a stream decodes to, and `cuda_detect` records a 10-bit file in ten
  bits, in its own colours, hiding what `--hide` says.

  The same on a Mac. `VideoToolboxFrameFormat::P010` is what
  `VideoToolboxDecoder` decodes 10-bit video to; `VideoToolboxEncoder`
  encodes it as HEVC Main 10 in the colours `with_color` says, and refuses
  it for H.264 with `TenBitsNeedHevc`, and `MetalVideoCompositor` refuses
  to compose on it (`UnsupportedCanvas`). `MetalOrtDetector`,
  `MetalOrtClassifier` and `MetalOrtEmbedder` look at an HLG or PQ P010
  picture through an SDR copy a Metal kernel makes by the same
  definition, to within 2 of 255 of it; `MetalDetectionOverlay` hides in
  the ten bits, the CPU's bytes, where it draws nothing, and gives its
  copies' pixel buffers the picture's colour, which VideoToolbox's HLG
  encoder reads there; `MetalCutDetector` reads P010 too. `metal_detect`
  records a 10-bit file in ten bits, in its own colours, as `hvc1`.

  What does not survive re-encoding: an iPhone's Dolby Vision metadata —
  what comes out is its HLG — and a PQ file's mastering display and
  light levels, which NVENC does not write.

- **Where each shot of an edited video begins.** `SwCutDetector` and
  `CudaCutDetector` hand each picture on, the first of each shot after the
  stream's first carrying a `SceneCut`; an `ObjectTracker` starts over on
  it, numbering what the new shot holds anew, and an `ObjectAnalytics`
  forgets where each object was. Each picture is made a thumbnail — luma
  in 64 by 36 cells, chroma in 32 by 18 — and a cut is a picture far from
  the last, and further than the pictures before it moved, as
  `CutDetectorOptions` says, or very far and of another shape altogether;
  a light over the same shot keeps its shape and is none; a flash, the blur running up to a cut and a
  fade through black are told apart by holding `lookahead` pictures. The
  CUDA one averages the cells on the GPU with the overlay's mosaic kernel
  and copies the thumbnail down, the same thumbnail the CPU makes, so the
  two find the same cuts. On 222 cuts marked by hand in three Blender
  films the defaults found 220 and called 11 that were none, where ffmpeg's
  `scdet` at its best found 191 for 18
  (`docs/benchmarks/vision/cuts.md`). `cut_detect` prints a file's cuts.
  `MetalCutDetector` does the same on VideoToolbox pictures, NV12 or P010,
  averaging the cells with the Metal overlay's mosaic kernel into memory
  the CPU reads: the CPU's thumbnail float for float, and the same cuts at
  the same scores (`cut_detect --metal`).

- **A phone's portrait recording is analysed the right way up, and stays
  turned when re-encoded.** `media_pp::orientation::Orientation` is which
  way up a picture is shown — a quarter-turn `Rotation` and whether it is
  mirrored — read from a picture with `Orientation::of` and from a stream
  with `StreamInfo::orientation`, and written into a re-encoded track with
  `TrackFormat::with_orientation`; `Orientation::to_display` and
  `from_display` move a box between the picture as stored and as shown. A
  matrix that turns by other than quarter turns is
  `UnsupportedOrientation`, which `Error` converts from. The pictures, and
  every box, track, zone and mosaic, stay as stored. What changes with a
  turned picture:

  - The detectors and classifiers — on the CPU, CUDA and Metal — fit it
    into the model the right way up, and read what the model found back
    onto the stored picture: a portrait recording's people had been handed
    to the model lying down. On a test clip turned a quarter, CUDA and the
    CPU each found the boxes of the upright clip again, at an IoU of 0.99.
  - The detection overlays lay labels, and the zones' and lines' labels,
    out as the picture is shown: upright, and above their boxes as seen.
  - `ObjectAnalytics` stands an object on the bottom of its box as shown.
  - `cuda_detect`, `cuda_track`, `metal_detect` and `transcode` keep the
    file's orientation in what they record.

  The window renderers came later, in an entry of their own.

- **An element of your own can work on CUDA pictures without them
  leaving the GPU.** `CudaSurfaceView::new(&device, &frame)` asks of a
  picture what the CUDA elements here ask — CUDA, from this device, NV12,
  P010 or BGRA — and gives each plane's device pointer, pitch and the bytes
  and rows that are picture. `CudaFramePool` hands out pictures of one
  format and size on the device to write into: the elements after yours
  take them as their own, where one allocated through another FFmpeg device
  context, even on the same GPU, is refused as `ForeignContext`.
  `CudaDevice::ordinal` names the GPU whose primary context every picture
  lives in, for retaining it with your own CUDA calls. `CudaFrameError`
  gains `NoSurface`, for a CUDA frame with no device pointer behind it.

- **And on VideoToolbox pictures with Metal of its own.**
  `MetalSurfaceView::new(&mtl_device, &frame)` asks of a picture what the
  Metal elements here ask — VideoToolbox, NV12 or BGRA, on a pixel buffer
  at least its size — and gives its planes as `MetalFramePlanes`, textures
  made over the pixel buffer's `IOSurface` on whichever Metal device you
  draw with, to read, write or render to. `VideoToolboxFramePool` hands out
  VideoToolbox pictures of one format and size to write into, which the
  elements after yours take as from their own upload, and
  `VideoToolboxFramePool::describe` makes one's pixel buffer say the colour
  its frame says — where VideoToolbox reads it, an HLG encoder failing a
  picture whose buffer says nothing — refusing a picture of another pool's.
  `MetalSurfaceError` and `VideoToolboxFramePoolError` say why either
  fails.

- **The detection overlays do with each class what its rule says** —
  draw its box, hide it, both or neither. `DetectionOverlayOptions::rules`
  holds a `ClassRule` per class, named as the model names it (`"face"`) or
  by number, and `others` covers the classes no rule names. A rule's
  `Treatment` gives its `BoxStyle` — line width, colour, label — or none,
  its `Hiding` or none, and the score below which the detection is left
  alone, so that faces can be hidden from 0.2 while boxes are drawn from
  0.5. The font is given once, `DetectionOverlayOptions::font`; a label
  asked for without one, or a class two rules name, is refused with
  `DetectionOverlayOptionsError`. A rule naming a class where the model
  names none is warned of once. By default every box is drawn as before.

  Hiding is DeepStream's redaction: the box grown by a margin and covered
  by a mosaic, a blur or a fill (`RedactStyle`), on the copy the overlay
  draws on, so that a branch beside it keeps the picture as it was. A
  mosaic's or a blur's cells are sized by the box — its shorter side cut
  into `cells`, none smaller than `min_cell` pixels — so that a face close
  to the camera is hidden as well as one far off; the blur blends those
  cells' means rather than blurring lightly, which can be partly undone.
  All three overlays hide and write the same bytes: `SwDetectionOverlay` on
  the CPU, `CudaDetectionOverlay` with two kernels of its own, and
  `MetalDetectionOverlay` with their Metal counterparts, which read and
  write the planes as integers and are compiled without fast math. The
  `cuda_detect`, `cuda_track` and `metal_detect` examples hide a class
  with `--hide person=blur`.


- **`StreamMux`: several streams through one model,** DeepStream's
  `nvstreammux` and `nvstreamdemux` — see docs/stream-mux.md. Inputs come
  from other pipelines through `StreamMuxHandle::add_source`; every picture
  goes on with `StreamOrigin` and `BatchSlot` metadata, a batch being its
  pictures one after another. Live, a batch waits for every input or
  `batch_timeout`; offline, for every input that has not ended.
  `StreamMuxHandle::demux` makes a `Tee` that hands each stream to a branch
  of its own and ends a stream's branch when the stream ends. A detector's
  `interval`, `ObjectTracker`, `ObjectAnalytics` and the classifiers' memory
  keep what they keep for each stream, and a seek of one stream starts only
  that one over; `AnalyticsOptions::streams` gives a stream zones and lines
  of its own. The tracker's numbers are unique across the streams, and go
  on after a seek rather than starting from 1 again.
  `CudaOrtDetectorOptions::max_batch` runs the pictures of a batch through
  the model at once — with TensorRT, an engine built for every batch up to
  it; a picture with no `BatchSlot` is a batch of its own, as before. The
  `cuda_multi_detect` example runs several files through one detector: on
  an RTX 3050, eight streams went from 395 pictures a second with a
  detector each to about 750 batched. A YOLOv8 or YOLO11 output stays on
  the GPU, where each box's best class is found, and six floats a box are
  copied down rather than every class's score.
  `MetalOrtDetectorOptions::max_batch` does the same through Core ML, the
  batch's pictures fitted in one Metal pass and the model's open batch
  fixed at that size; `MetalOrtDetector::new` takes those options in place
  of `OrtDetectorOptions`. An open height and width — an Ultralytics export
  with `dynamic=True` leaves both open beside the batch — are fixed at the
  model's size too, where Core ML would otherwise hand much of the model
  back to the CPU: YOLO11n went from 38 pictures a second to 232. The
  `metal_multi_detect` example is `cuda_multi_detect` on macOS: on an M5,
  eight streams went from 142 pictures a second one at a time to 310
  batched. Each box's best class of a YOLOv8 or YOLO11 output is found on
  the GPU there too, from a copy in memory the GPU shares — a batch of
  eight read in 0.9 ms rather than 1.9.

- **`CudaOrtDetector`: detection on CUDA pictures, on CUDA or through
  TensorRT** (features `ort-cuda` and `ort-tensorrt`). What `SwOrtDetector`
  does, with the picture never leaving the GPU: a kernel fits each NV12 or
  BGRA CUDA picture into the model's input in device memory, and ONNX
  Runtime's CUDA provider reads it there — or with `ort-tensorrt` its
  TensorRT provider, with CUDA for what TensorRT cannot run. The engine
  TensorRT builds for a model and GPU — minutes, once — is kept in the
  user's cache directory (`CudaOrtDetectorOptions::engine_cache`), an
  engine built for a range of batches in a directory of that range's own,
  `batch-1-4-4`: ONNX Runtime names an engine after its model alone, and
  two detectors of one model with different `max_batch` rebuilt each
  other's on every start — 162 seconds where loading takes 0.3. On an
  RTX 3050 a 1080p file is decoded by NVDEC and run through YOLOv10n at
  about 540 pictures a second through TensorRT, and about 190 on CUDA
  alone. The `cuda_detect` example runs it over a file.

  The NVIDIA libraries the providers need are linked into the program:
  CUDA 13's runtime, cuBLAS and cuRAND, and cuDNN 9 with `ort-cuda`, and
  TensorRT 10 and its ONNX parser besides with `ort-tensorrt`. An
  application built with them is one for NVIDIA GPUs, and like a missing
  FFmpeg a missing library stops it before it starts, by name. Building
  needs them where the linker finds them — `MEDIA_PP_NVIDIA_LIB_DIRS`,
  `LD_LIBRARY_PATH` or `LIB`, the CUDA installation or the system's
  directories — and the build script warns of any it could not find; a
  check, which never links, needs none. Where the program finds them at run
  time is the application's to arrange, as with FFmpeg: installed, on the
  loader's path, or beside it under an `$ORIGIN` run path.

  What linking cannot say is age. `CudaOrtDetector::runtime` asks each
  linked library and the driver, which is opened rather than linked, for
  its version, and says what this machine can run a detector on —
  `CudaRuntime::TensorRt`, `CudaOnly` or `Unavailable` — with a
  `RuntimeShortfall` for what falls short: the driver `Missing`, or a
  library `Outdated`, with the `LibraryVersion` it reports and the one
  needed, or TensorRT `NotBuilt`. The versions taken are those ONNX
  Runtime's build was made against: a driver for CUDA 13.0, the CUDA 13.2
  runtime, cuDNN 9.23.2 and TensorRT 10.15.1. `new` asks it first,
  refusing with `OrtDetectorError::CudaRuntimeUnavailable` where the CUDA
  side falls short. Where TensorRT does, `CudaOrtDetectorOptions::tensorrt`
  decides: the default `UseTensorRtPolicy::Preferred` runs on CUDA alone
  with a warning, `Required` refuses with
  `OrtDetectorError::TensorRtUnavailable` — and with the provider's own
  error where TensorRT will not start, rather than ONNX Runtime passing
  over it to CUDA — and `Off` never uses it.

- **`MetalOrtDetector`: detection on VideoToolbox pictures, through Core
  ML** (feature `ort-coreml`, Apple silicon). What `SwOrtDetector` does,
  with each NV12 or BGRA picture — a decoder's, a camera's, a screen's —
  fitted into the model's input by a Metal kernel, in memory the CPU and
  GPU share, and the model run by ONNX Runtime's Core ML provider, which
  reads it there. Core ML chooses between the Neural Engine, the GPU and
  the CPU layer by layer, and ONNX Runtime gives the CPU what Core ML does
  not take; what it refuses is running without Core ML at all — `new` fails
  where the provider will not start rather than run the whole model on the
  CPU, which is `SwOrtDetector`'s job. On an M5 YOLOv10n
  looks at a 720p picture in about 7 ms through Core ML, against about 19
  ms on the CPU, and loads in under half a second. The `metal_detect`
  example runs it over a file. `MetalOrtDetectorOptions::compute_units`, a
  `CoreMlComputeUnits`, holds Core ML to the GPU or to the Neural Engine
  rather than letting it choose; on an M5 the Neural Engine ran YOLO11n at
  a quarter of the GPU's rate.

- **`Detections`: what a detector found in a picture, as its metadata.**
  `SwOrtDetector` puts one on every picture it hands on — the detector's name,
  the model's class names, and a `Detection` per object, most confident
  first, each box in fractions of the picture so it fits the picture at any
  size it is scaled to; `Detections::label` names a detection's class. An
  empty list means it looked and found nothing. `OrtDetectorOptions` holds
  the confidence and overlap thresholds and the class names, which by
  default are read from the model's own `names` metadata, as Ultralytics
  exports carry them, so a detector of any classes names them without the
  caller knowing what they are.

  `Detections` and `Detection` need no feature: they are what any detector
  writes, this crate's or an application's, which makes them with
  `Detections::new` and `Detection::new` and puts them on a picture with
  `Detections::attach_to`. Both are non-exhaustive, so that what a tracker
  or a second model learns of an object can be added to them later.

- **`SwOrtClassifier` and `CudaOrtClassifier`: what a detector found,
  named by a second model** — DeepStream's secondary inference. Each
  object's box is cut from the picture and stretched to an image
  classifier's input, objects classified together where the model's batch
  is open, and each answer goes in the object's `Detection::classes` as a
  `Classification`: the classifier, its class names, the class and the
  score. After an `ObjectTracker`, a numbered object is classified once and
  its answer kept on every picture it is followed through, those a detector
  let by too, until `OrtClassifierOptions::reclassify` pictures have passed
  — DeepStream's asynchronous mode; `classes` limits it to some of the
  detector's classes, `min_size` to boxes big enough,
  `min_detection_score` (0.25) to boxes the detector was sure enough of —
  the unsure ones a tracker is handed and hands on unnumbered were
  otherwise classified again on every picture — `min_score` to answers
  sure enough, and `InputScale` says whether the model wants 0 to 1
  or ImageNet's normalisation. Raw scores are put through a softmax, and
  probabilities taken as they are. The CUDA one cuts the boxes with the
  detector's fitting kernels, pointed at each box, into one batch tensor in
  device memory, normalised there by a kernel of its own, and runs through
  TensorRT in half precision with `ort-tensorrt` — its engine built once
  for every batch from 1 to 32, as many as a picture has objects — and on
  ONNX Runtime's CUDA provider where TensorRT cannot run. On CUDA's
  provider beside a TensorRT detector, its many small kernels queued
  behind the detector's: one 1080p stream with a classifier ran at 228
  pictures a second where it ran at 545 without, and runs at 514 through
  TensorRT. Both cut each input pixel as the mean of the picture's pixels
  it covers, the CPU as the kernels do; where the CPU is sure of a box,
  MobileNetV2 on the GPU names the same class, its score within 0.02 of
  the CPU's on a BGRA picture and about 0.1 on NV12, whose chroma
  swscale interpolates and the kernels do not. The overlay labels an
  object with what was said of it, `car #5 0.48 | minivan`, and
  `cuda_track --classifier` runs the whole of detection, tracking,
  classification and drawing on the GPU.
  `OrtDetectorError` is `OrtError`, the old name kept as an alias, since
  the classifiers' errors are the same.

- **`vision_bench`, and the vision benchmarks it measured.** An example
  that builds the pipeline its flags describe — CPU, CUDA or TensorRT,
  streams batched or not, an interval, a tracker, a classifier, drawing
  and encoding — plays a file through it as fast as it goes and prints one
  line a script can collect, beside `model_only`, the model alone through
  the same TensorRT, and `bench.py`, which runs a matrix of them in turns
  and compares builds inside one run. `docs/benchmarks/vision` has what
  they measured on an RTX 3050, read against the model alone, and what a
  Mac needs to measure the same. On macOS `vision_bench` has a `metal`
  backend — VideoToolbox, Core ML on the compute units `--compute-units`
  allows, Metal — and `model_only` a Core ML provider; the matrix measured
  on an M5 MacBook Air, at the fanless laptop's sustained rate, is in
  `macos-m5-air.md`: YOLO11n through Core ML five times the CPU, the
  Neural Engine a quarter of the GPU, and a detector per stream ahead of
  one batching them.

- **`MetalOrtClassifier`: what a detector found, named by a second model on
  VideoToolbox pictures** (feature `ort-coreml`). What the other two
  classifiers do, with each object's box cut from the picture and
  stretched into the model's input by the Metal kernel the detector fits
  with — now able to take a rectangle of the picture, write any input of a
  batch, and scale each channel as ImageNet's normalisation wants — and
  the model run through Core ML. A model that leaves its batch open is
  fixed to take four through Core ML (`with_dimension_override`): open, it
  was compiled again for every new number of objects, 0.6 seconds a time,
  ran no faster than the CPU, and took a batch of one through an error it
  recovered from; fixed, it compiles once and runs on the Neural Engine.
  On an M5, MobileNetV2 says what it says on the CPU of the same input to
  within 2·10⁻⁵, names the zebra and the macaw of a test clip from the
  boxes YOLOv10n put round them, and behind YOLOv10n and a tracker keeps
  Intel's walking people at 195 pictures a second. `metal_detect
  --classifier` runs it.

- **`ObjectAnalytics`: objects counted in zones and across lines.**
  DeepStream's `nvdsanalytics`, on the `Detections` a picture carries: for
  each `Zone` — a polygon in fractions of the picture, of some classes or
  all — the objects standing in it, and whether as many as its
  `crowded_at` are; for each `Line` the objects followed across it, each
  way apart, with running totals and the crossings of each picture. It
  puts what it counted beside them as `Analytics`, with a `ZoneCount` per
  zone and a `LineCount` and its `Crossing`s per line. An object is where
  the bottom middle of its box is, and a line is crossed by an object an
  `ObjectTracker` before it has numbered; the pictures a tracker filled in
  count as well. An object is on a side of a line only once it is
  `Line::margin` past it — a tenth of its own box's height by default — so
  a box trembling about the line is not counted crossing back and forth.
  On Intel's `people-detection.mp4`, seven people walking away from the
  camera, a line across the middle of the floor counts the seven with or
  without it; one near the far wall, where the boxes tremble as people turn
  for the doors, counted a crossing back that never happened without it,
  and one along the wall's foot, which nobody crosses, four. It reads metadata alone, so it takes pictures wherever
  they live, and needs no feature. `metal_detect --line` counts with it.
  Each `ZoneCount` carries its zone's corners and each `LineCount` its
  line's ends, so that what draws them draws what was counted.

- **What a detection overlay draws, chosen.** `DetectionOverlayOptions::parts`
  (`OverlayParts`) turns an `ObjectAnalytics`'s zones and its lines on or
  off, and sets how thick and how large their labels are. A zone is outlined
  in amber, red while it is crowded, and labelled with how many objects are
  in it; a line is drawn across the picture in cyan, slanting or not, and
  labelled with its crossings each way, `gate 12 / 3`. All three overlays
  draw them alike, a slanting line in pieces through coverage masks made
  once and kept. `LabelStyle` says what a box's label says — `class`,
  `track_id`, `score` and `classes`, all by default — and a box whose label
  would say nothing has none. `metal_detect --line --out` draws its lines.

- **`ObjectTracker`: what a detector found, numbered across pictures.**
  ByteTrack — a Kalman filter per object, and matching by overlap with the
  Hungarian method, confident detections first and then the unconfident a
  partly hidden object still gets — puts the same `Detection::track_id` on
  an object in every picture it is followed through. It reads and writes
  `Detections` alone, never a pixel, so one element takes pictures wherever
  they live, and needs no feature. `TrackerOptions` holds ByteTrack's
  thresholds, how many sightings confirm a new object, how long a lost one
  is kept, and whether classes are matched apart.

  With it a detector need not look at every picture:
  `OrtDetectorOptions::interval` lets pictures by unlooked-at, carrying no
  `Detections`, and the tracker puts on each where it expects the objects
  to be, marked `Detections::predicted` — DeepStream's `interval`. Gaps are
  told from the timestamps, so pictures a dropping queue removed are
  carried through too. The filter follows a box's width and height rather
  than ByteTrack's aspect ratio, which its fixed uncertainty all but freezes:
  carried between detections on people walking, those boxes were half again
  too wide, and worse than holding the last box still. The `cuda_track`
  example records the tracking and measures the filled-in pictures against
  detecting every one: at a detection every third picture of 12 fps
  walking people, the expected boxes overlap the detector's by 0.78 on
  average, against 0.73 for holding still; at every fifth, 0.71 against
  0.63. On macOS, `metal_detect --track` and `--interval N` put it behind
  `MetalOrtDetector`.

  `TrackerOptions::visual` follows each object by how it looks as well —
  a correlation filter learned from the pixels around it, as DeepStream's
  NvDCF has (MOSSE, searched at three scales). Between detections it finds
  the object where it went rather than where its speed would have taken
  it; on a detection's picture it moves the track there before matching.
  It reads only the pixels around each object: 8-bit YUV, grey and RGB
  pictures in system memory in place, and with `cuda` NV12 and BGRA CUDA
  pictures by copying down those regions alone (`cuMemcpy2D`), with no
  kernel, and with `metal` NV12 and BGRA VideoToolbox pictures in place,
  their pixel buffers locked for the CPU, which on Apple silicon shares the
  GPU's memory; pictures it cannot read are followed by motion. What it finds is
  weighed against the motion in the Kalman filter, as a measurement three
  times as uncertain as a detection: taken at its word, the boxes were
  worse than motion alone over short gaps. Weighed, on 12 fps walking
  people it matches motion alone a picture after a detection and beats it
  further on — detecting every fifth picture, 91% of objects covered by
  half or more against 87%; every tenth, 74% against 51%, and on people,
  bicycles and cars 73% against 29%. It costs CPU per object and picture:
  a 1080p concert with a hundred people went from 8 to 24 seconds. On an
  M5, `metal_detect --visual --interval 9` over the same walking people
  counts the seven crossing a line across the floor, net, where motion
  alone counts four, and numbers 11 objects where motion alone numbers 19.

  With the `cuda-visual-tracking` feature, CUDA pictures are followed by
  look on the GPU: every followed object's neighbourhoods are sampled,
  normalised, transformed by cuFFT, matched against its filter, and their
  peaks found there at once, by kernels of their own, and four numbers an
  object come back; the filters live on the device. cuFFT 12 is linked, as
  `ort-cuda` links its libraries, so the program does not start without it.
  It follows as the CPU does — the same boxes, to the third decimal, on
  both walking clips — and costs nothing to see: on the 1080p concert, 8.1
  seconds against 8.2 for motion alone, where the CPU took 24.

  With `metal`, VideoToolbox pictures are followed by look on the GPU the
  same way, by Metal kernels of this crate's own — the 64-point transforms
  among them, rows then columns, as Metal has no cuFFT — and nothing more
  is linked, Metal being part of macOS; where Metal cannot, the CPU reads
  the pictures where they are. What a picture asks of the GPU goes in one
  pass, run when the peaks are wanted and before the picture is let go of:
  two round trips a picture followed, one a picture detected, where asking
  each step apart took nine. On an M5 it finds what the CPU finds to the
  sixth decimal, and with 1080p pictures of 25 objects takes 1.7 ms a
  picture against the CPU's 9.3, of 100, 2.9 against 37; with a few, the
  round trips cost it about a millisecond a picture more than the CPU.

  `BoxColors::ByTrack` colours each object by its number, and a label shows
  the number — `person #7 0.87`; a class the model names none of is
  labelled `class 3` rather than `#3`.

- **`SwDetectionOverlay` and `CudaDetectionOverlay`: what was found, drawn
  onto the picture.** Each picture carrying `Detections` is handed on as a
  copy with a box around each object — and, given a font
  (`DetectionOverlayOptions::font`), its class and score on a band above
  it — and the same `Detections` still on it. The software one draws on
  NV12, YUV 4:2:0, RGB24 and BGRA pictures; the CUDA one on NV12 and BGRA
  CUDA pictures without them leaving the GPU, so a detector's pictures go
  on to an encoder, a renderer or a compositor with their boxes. Each draws
  on a copy, never on the picture it was handed, so a Tee in front of one
  shares no boxes with its other branches, and a picture with nothing to
  draw is handed on as it came. Colours follow the class, or one colour for
  all (`BoxColors`); lines and labels fall on whole 2x2 blocks of an NV12
  or YUV picture, so their colour does not bleed. Neither needs a feature
  beyond `cuda` for the CUDA one. On the RTX 3050, `cuda_detect --out`
  decodes, detects, draws and encodes a 1080p file at about 420 pictures a
  second.

- **`MetalDetectionOverlay`: what was found, drawn onto VideoToolbox
  pictures** (feature `metal`). What the other two overlays do, on NV12
  and BGRA VideoToolbox pictures with Metal: a copy from a pool of its own,
  rectangle fills for the lines and bands, and each label rasterized once
  and blended from a coverage mask, so `MetalOrtDetector`'s pictures go on
  to `VideoToolboxEncoder` or a Metal renderer with their boxes. It passes
  its input's contract through, as they do. On an M5 it draws on a 720p
  picture in under a millisecond, and `metal_detect --out` decodes,
  detects, draws and encodes a 720p file at 150 to 200 pictures a second.

- **`Metadata`: what an element found out about a buffer, carried with
  it.** Every buffer can carry one — `MediaBuffer::metadata`,
  `with_metadata`, `set_metadata` — holding one value per type, so an
  element defines the type of what it finds and another reads it by that
  type (`metadata.get::<T>()`). It sits behind an `Arc` and is never
  changed in place: a buffer a `Tee` shared keeps what it had in every
  other branch. A `Filter` needs no code for it: its outputs of the
  input's sort at the input's timestamp — a picture scaled, converted,
  uploaded or downloaded — are given what the input carried, and nothing
  is carried onto another sort or timestamp, where it would describe the
  wrong picture. `FrameRateLimiter`, which stamps its outputs anew,
  carries it by hand. This is what a detector's results will travel in.

- **`V4l2VirtualCamera`: a pipeline's pictures as a Linux camera**
  (feature `v4l2-virtual-camera`). It writes into a v4l2loopback device,
  which every application reading cameras through V4L2 — a browser, Zoom,
  Discord, OBS Studio — lists under the label the module was loaded with,
  for as long as the element holds the device. Each picture is written as
  I420 at the size the element was made with, converted to BT.709 limited
  range from any system-memory video, and the device is told the rate.
  `V4l2VirtualCamera::list_devices` lists the loopback devices free to
  write to — none where the module is not loaded, which is the system's
  to do (`modprobe v4l2loopback exclusive_caps=1 card_label=…`); a device
  another program writes to answers `V4l2VirtualCameraError::InUse`. The
  `virtual_camera` example now runs on Linux too. `V4l2Device` is shared
  with `v4l2-capture`, and is exported with either feature.

- **`MfVirtualCamera`: a pipeline's pictures as a Windows camera**
  (feature `mf-virtual-camera`, Windows 11). Every application that lists
  cameras — Teams, Zoom, a browser, the Camera app — sees "… Windows
  Virtual Camera" for as long as the element lives, and gets each picture
  as NV12 at the size it chose (1920x1080, 1280x720 or 640x360, 30 frames
  a second), converted to BT.709 from any system-memory video. The camera
  itself is a DLL of its own, the new `vcam` crate, which Windows' Frame
  Server loads and which is installed once per machine from an elevated
  prompt (`vcam/install.ps1`); without it `MfVirtualCamera::new` answers
  `MfVirtualCameraError::NotInstalled`, and before Windows 11
  `Unsupported`. The `virtual_camera` example shows a test pattern or a
  looping file.

- **`MetalRenderer`: VideoToolbox frames drawn by an application's own
  Metal code** (macOS, `metal`), the sibling of `D3d11Renderer` and
  `CudaRenderer`, for a UI's own layer or an offscreen target rather than
  a window of this crate's. The application implements
  `MetalFrameRenderer` — its `device`, `submit` and `resize` — and is
  handed each picture as a `MetalFrame`: `MetalFramePlanes::Bgra` or
  `Nv12 { luma, chroma }`, textures on its own device made over the
  frame's `IOSurface` with nothing copied, the picture's size and its
  colour. The `MetalFrame` holds the frame, so its pixel buffer is not
  handed out again until the application lets go of it, once the GPU has
  read it. A frame in system memory is refused when the branch is built;
  `VideoToolboxUpload` is what brings one to it. New: `MetalTexture`,
  `MetalRendererError`, `Error::MetalRendererError` and
  `ElementType::MetalRenderer`.

- **`MetalSharedTextureSource`: another producer's `IOSurface`s, as this
  pipeline's own frames** (macOS, `metal`), the sibling of
  `D3d11SharedTextureSource` with its shape:

  ```rust
  let (source, handle) = MetalSharedTextureSource::new("browser", &device, 1920, 1080, 2)?;
  // ...from the producer's own callback, while the surface is its picture:
  handle.push(&surface, None)?;
  ```

  Each surface pushed — a browser engine's accelerated paint, or one
  another process sent, looked up from its Mach port — is copied with a
  Metal blit into a pixel buffer of this source's own and sent downstream
  as a BGRA VideoToolbox frame, since the producer draws its next picture
  into the same surface. A surface of another size, or anything but BGRA,
  is refused rather than adapted; the producer owes a finished picture,
  not a submitted one. `try_push` drops a picture rather than stall a
  producer the source's paused pipeline is not draining. New:
  `MetalSharedTextureHandle`, `MetalSharedTextureSourceError`,
  `Error::MetalSharedTextureSourceError` and
  `ElementType::MetalSharedTextureSource`.

- **`screen_record_av`, `screen_record_nvenc` and `screen_record_overlay`
  run on macOS**: the desktop and the output device's sound through
  `ScreenCaptureKitSource` and `CoreAudioCaptureSource`; ScreenCaptureKit's
  own pixel buffers encoded by `VideoToolboxEncoder` with nothing copied;
  and those under a live text layer of `MetalVideoCompositor`. The new
  **`metal_window_render`** example draws into a `winit` window of its own
  through `MetalWindowRenderer::for_window`, following it as it resizes; the
  renderer logs the size it draws at, at `Debug`, whenever that changes.

- **`MetalScaler`, `MetalConverter`, `MetalChromaKey`, `MetalVideoEffect`**:
  the rest of the GPU filters on macOS, behind `metal`, each the Metal
  sibling of its Vulkan counterpart with its API and contract, on
  VideoToolbox frames read and written through their `IOSurface`s. The
  scaler resizes NV12 and BGRA with the same four kernels
  (`MetalScalerInterp`), averaging what it covers when it shrinks; the
  converter brings NV12 to BGRA by each frame's own colour description; the
  key and the effect take BGRA and a `ChromaKeyHandle` or
  `VideoEffectHandle`, computing what the software ones compute, which
  their tests check against. Their kernels are Metal Shading Language
  compiled at run time, one kernel serving every plane of the scaler since
  a texture view of a plane's own format puts its channels where the plane
  keeps them.

- **`MetalWindowRenderer`, and `VideoWindow` and `Player` on macOS**,
  behind `metal` (`Player` with `coreaudio-renderer` too). The renderer is
  the Metal sibling of `D3d11WindowRenderer` and `VulkanWindowRenderer`,
  with their API — `open` for a window of its own with `WindowEvents` and a
  `WindowControl`, `for_window` for an application's AppKit view — and
  their contract: VideoToolbox frames, NV12 or BGRA, drawn from their
  pixel buffers as they are, and frames in system memory, NV12, YUV420P or
  BGRA, uploaded; each in its own colours, letterboxed, presented at the
  display's refresh into an sRGB layer. What a picture takes to reach the
  screen is told to the playback clock — two refreshes of the display,
  since the WindowServer gives a composited window no presented time.
  AppKit serves every window from the main thread's event loop only, so a
  window is made there from whichever thread opens it, and
  **`run_with_windows`** runs a program's work beside that loop for one
  that has none of its own; opening a window with no loop running fails
  after a few seconds (`WindowError::NoMainLoop`) rather than hanging.
  `VideoWindow` draws on it, and `Player` decodes on VideoToolbox and plays
  through `CoreAudioRenderer`. `Key::Other` carries AppKit's key code on
  macOS. Its checks open real windows, in a test with a `main` of its own
  (`tests/metal_window.rs`), which also plays a file through `Player`.
  `player`, `sw_decode_render`, `hw_decode_render`, `av_playback`,
  `seek_render`, `test_video`, `transcode_render`, `gpu_video_compositor`,
  `screen_preview_cpu`, `screen_preview_gpu` and `webrtc_video_call` run on
  macOS.

- **`MetalVideoCompositor`: compositing on the GPU on macOS**, behind the
  new `metal` feature (which enables `videotoolbox`), the Metal sibling of
  the D3D11, CUDA and Vulkan compositors with their API —
  `MetalVideoCompositorHandle`, `MetalVideoLayerHandle`,
  `MetalTextLayerHandle`, `VideoCompositorControl`, live or
  `RenderMode::Offline`. It composes VideoToolbox frames, NV12 or BGRA,
  into VideoToolbox frames, BGRA or NV12 (`VideoToolboxFrameFormat`), so a
  decoder's, an upload's, a camera's or a screen's pictures go in and a
  `VideoToolboxEncoder` takes what comes out, with no new memory domain
  and no copy: each picture is read, and each output written, through a
  Metal texture made over the `IOSurface` its pixel buffer is in. Its
  kernels are Metal Shading Language, compiled by Metal when it is made,
  so the build needs no shader toolchain; they draw as the Vulkan
  compositor's do — each layer's own colour description, alpha straight or
  premultiplied, text from coverage masks, NV12 out at BT.709 limited
  range. A pixel buffer belongs to no device, so it takes frames from any
  `VideoToolboxDevice`. `MetalError` says what Metal refused.

- **`ScreenCaptureKitSource`: display and window capture on macOS**,
  behind the new `screencapturekit-capture` feature (macOS 12.3+), in the
  shape `DxgiCaptureSource`, `WgcCaptureSource` and
  `PipeWireScreenCaptureSource` have, one source for both:
  `ScreenCaptureKitTarget::Display` or `Window`, from `list_displays`
  (`ScreenCaptureKitDisplay`, its size in pixels) and `list_windows`
  (`ScreenCaptureKitWindow`, an application's windows at the normal level,
  on this Space or another). It emits BGRA at a fixed rate
  (`ScreenCaptureKitOptions::frame_rate`, a `FrameRateHandle` at run time),
  `pts` counting ticks, a tick with nothing new offering the picture it
  already has without copying it; sRGB, full range, the pointer drawn where
  `include_cursor` says. `open` copies into system memory, and with
  `videotoolbox` `open_videotoolbox` hands on ScreenCaptureKit's own pixel
  buffers as VideoToolbox frames, which `VideoToolboxEncoder` — and so
  `VideoEncodeBin` — encodes on the media engine with nothing copied. A
  window resized is scaled into the size it was opened at; one closed, or
  a display gone, ends the source with `SourceGone`. Recording the screen
  needs the user's permission, which `open` asks for, returning
  `PermissionDenied` until it is given and the program started again.

- **`AvFoundationCaptureSource`: camera capture on macOS**, behind the new
  `avfoundation-capture` feature, on AVFoundation directly and in the shape
  `MfCaptureSource` and `V4l2CaptureSource` have: `list_devices` lists
  built-in, external, Continuity and Desk View cameras as
  `AvFoundationDevice`s, `list_formats` their modes as
  `AvFoundationCaptureFormat`s, and `open` answers with the source and the
  geometry the camera delivers — NV12 in system memory, or with
  `videotoolbox` `open_videotoolbox` hands on the pixel buffers the camera
  filled as VideoToolbox frames, which `VideoToolboxEncoder` takes with
  nothing copied. With no mode asked for, the camera stays in the one it is
  in — the system's default, 1080p on a MacBook, for a camera this process
  has not configured. Whatever the mode, the source holds it: a macOS
  capture session puts a camera in its preset's format each time it
  starts, so the camera is locked for configuration across every start,
  and its frame rate is fixed. `pts` counts the host clock in microseconds
  from zero, a pause collapsing to one frame interval. The camera needs the
  user's permission, which `open` asks for and waits on, and which macOS
  asks only for a program whose `Info.plist` has
  `NSCameraUsageDescription` (`PermissionDenied` where it is refused); a
  camera unplugged ends the source with `DeviceGone`.

- **VideoToolbox frames on macOS: `VideoToolboxDevice`,
  `VideoToolboxDecoder`, `VideoToolboxUpload`, `VideoToolboxDownload`**,
  behind the new `videotoolbox` feature — the first stage of a macOS video
  backend, the way `vulkan` began. FFmpeg does all of it: its VideoToolbox
  context (`VideoToolboxDevice`, shared by a pipeline's VideoToolbox
  elements), its hwaccel, which `VideoToolboxDecoder` decodes through with
  preroll, QoS and playing backwards, and its transfers, which
  `VideoToolboxUpload` puts NV12, P010 and BGRA — and YUV420P as NV12 —
  into pixel buffers with and `VideoToolboxDownload` brings any back with.
  `MemoryDomain::VideoToolbox` says where such frames live, and the link
  check names the upload or download that goes between. A pixel buffer
  belongs to no context, so the download takes no device and refuses no
  frame for having come from another; Core Video's pool grows, so the
  decoder is given no budget of frames to hold. Frames whose picture is in
  their fourth plane pointer, as a VideoToolbox frame's is, are now told
  apart by it wherever an element recognises a repeated picture — before,
  every one of them read as the first.

- **`VideoToolboxEncoder`, and the bins on VideoToolbox**: H.264 and H.265
  (`VideoToolboxCodec`) on the Mac's media engine through FFmpeg's
  `h264_videotoolbox` and `hevc_videotoolbox`, from NV12 or BGRA
  VideoToolbox frames (`VideoToolboxEncoderOptions::format`) — a
  `VideoToolboxDecoder`'s, a `VideoToolboxUpload`'s or a capture's — with
  `VulkanEncoder`'s options and contract: opened eagerly, packets drained
  after each frame and at `Eos`, a CPU frame or another layout refused
  before anything is encoded. It reads each frame's own pixel buffer, so
  it takes frames from any VideoToolbox element. BGRA the media engine
  makes YUV itself, always with BT.709's matrix at limited range whatever
  the session is told, so the stream says exactly that and `with_color`
  refuses to say otherwise of BGRA (`ColorOfBgra`). `DecodeTarget::VideoToolbox`
  decodes on the media engine where the stream allows and otherwise in
  software through `VideoToolboxUpload`, as on the other targets, 10-bit
  among the latter; `EncodeInput::VideoToolbox`, whose frames are NV12 or
  BGRA (`VideoToolboxFrameFormat`, as `VulkanFrameFormat` is on Vulkan),
  encodes both with `h264_videotoolbox`, and in software after a
  `VideoToolboxDownload` where it does not open. So `transcode` decodes and encodes a file on
  macOS without its pictures reaching the CPU: five seconds of 1080p in
  0.7 s on a MacBook Air, 0.07 s of it on the CPU.

- **`CoreAudioRenderer`: audio playback on macOS**, behind the new
  `coreaudio-renderer` feature. It is the Core Audio counterpart of
  `WasapiRenderer` and `PipeWireAudioRenderer` and keeps their contract:
  `CoreAudioRenderer::list_devices` lists the devices with output
  channels as `CoreAudioDevice`s, the default marked; `open(name,
  CoreAudioRendererOptions { device })` returns the format the device
  takes — its own rate and channels, 32-bit float interleaved — for an
  `AudioResampler` in front of it, and a frame in any other format is a
  `FormatMismatch`; it masters the playback clock from its first frame,
  stretches its sound to the playback rate, plays out what it holds at the
  end of the stream and stops the device for a pause. It plays through an
  AUHAL output unit, whose real-time callback reads a 100 ms ring the
  renderer writes, so where playback is comes from the device's own
  timestamps and latency rather than from what was handed over.
  `audio_playback` plays on macOS with it.

- **`CoreAudioCaptureSource`: audio capture on macOS**, behind the new
  `coreaudio-capture` feature, in the shape `WasapiCaptureSource` and
  `PipeWireAudioCaptureSource` have: `list_devices` lists inputs and
  outputs as `CoreAudioDevice`s whose `CoreAudioDeviceKind` says which, and
  `open` records an input or captures what the system plays to an output;
  `list_processes` and `open_process` capture what one application — and
  the helpers it has started — plays, mixed to stereo. What is played is
  captured through a Core Audio process tap and a private aggregate device,
  macOS 14.2 and newer (`ProcessTapsUnsupported` before it, the functions
  looked up rather than linked so older systems still start), and needs the
  "System Audio Recording" permission of an application whose `Info.plist`
  has `NSAudioCaptureUsageDescription`; without it a tap records silence.
  `open` returns the device's own rate and channels, 32-bit float
  interleaved. `pts` counts what the device captured, so a pause costs
  nothing and a packet dropped because downstream fell behind leaves its gap
  where it happened, reported as `BusEvent::Dropped`; a tap, which hands
  over nothing while nothing plays, has its silence made up once it has
  been quiet for 100 ms, and nothing inserted while it plays. An unplugged
  input ends the source with `DeviceGone`. The renderer refuses an input
  device with `NotAnOutputDevice`. `audio_capture` captures on macOS, the
  system's sound by default.

- **A `Source` can have several outputs, and begin a segment.** It says
  what its pads are in `Source::outputs` — one, named for it, by
  default — and which each buffer is for with `Produced::On(n, buffer)`;
  its end goes out through every one. `Produced::Segment` begins a new run
  of what it makes through every output, on the timeline its pipeline is
  on — another input's stream, another lap — behind a flush where it is
  `flushed`. A buffer for an output it does not have ends the source with
  `ProduceError::NoOutput`. With several outputs, what one cannot take
  yet is held back, within bounds, so the one read cursor keeps the others
  fed — what `FileDemuxer` did for itself, now any such source's — and
  each output has its end as soon as it owes nothing. A `Source` that can
  be sought says so through `Source::as_seekable` and `as_reversible`:
  its seeks are taken between one thing made and the next, and it stays
  at its end until stopped or sought back into what it reads.
  `RtspSource`, `PipelineBridge`, `WebRtcTrackSource` and `FileDemuxer`
  are written on it now, their API unchanged.
- **`Sink::stopping`** tells a stop from a seek's flush, as
  `Filter::stopping` does: `reset` by default. `PipeWireAudioRenderer`
  is a `Sink` now and uses it — a stop deactivates its stream and hands
  back the position it masters, a flush only empties its queue — and the
  V4L2 and PipeWire captures are `Source`s, so every source of this crate
  runs the framework's loop. Their API is unchanged.
- **A `Pacer` or `VideoSynchronizer` always runs behind a queue.** Each
  waits on the clock inside `consume`, and in a chain with nothing queued
  straight in front of it, those waits held up the demuxer or decoder
  feeding it, whose thread then waited on the clock too. Every example
  and `Player` already queued in front of one; nothing made a caller
  who did not know to. Now `ChainBuilder::pipe` puts a queue of eight
  in front of one, named `<name>-queue`, where no `.queue()` stands
  straight before it; one that does is the queue it runs behind, so
  nothing changes for a chain that already had it, and putting it there
  yourself is how to choose its depth — for a hardware decoder's fixed
  pool, say.
- **`Filter::stopping`** tells a stop from a seek's flush: it lets go
  of what the run as a whole kept, beside what `reset` does, and does
  what `reset` does by default. `TimestampOrigin` and `FrameRateLimiter`
  are transforms now, their output timeline begun again by a stop and
  kept across a seek as before.

- **`Sink` writes a terminal as what it does with each buffer.**
  `render` for each one, `drain` before its end is taken, `reset` for a
  seek or a stop, and `pausing` and `resuming` for a device it stops while
  paused; the framework turns the control messages into those, and the
  element never sees one. `ChainBuilder::to` and `build` now take
  `impl IntoTerminal<M>`, which a `RawSink` and a `Sink` both are.
  `WasapiRenderer`, `WhisperTranscriber`, `FrameCounter` and
  `PacketCounter` are written this way now.

- **`Source` writes a source as what it makes alone.** Asked for the next
  thing, it makes it — a buffer, nothing yet, or the end of its stream —
  waiting where it has to only through the `Wait` it is handed, which lets
  go as soon as the pipeline has something for the thread, and whose clock
  (`Wait::now`) stands still while the pipeline is paused, so a schedule
  kept on it owes nothing after a pause. The framework runs the loop, and
  calls its `starting` and `stopping` around it on the source's own thread,
  for what a device sets up there — an apartment joined, a capture started.
  `Pipeline::new` and `PipelineBuilder::add_source` now take
  `impl IntoSource<M>`, which a `RawSource` and a `Source` both are;
  the wiring closure is handed the source itself, as before, or the
  `SourceStage` a `Source` is made into. `TestVideoSource`,
  `TestAudioSource`, `AppSource`, `AudioMixer`, the four video
  compositors, the Windows captures (`DxgiCaptureSource`,
  `WgcCaptureSource`, `MfCaptureSource`, `WasapiCaptureSource`) and
  `D3d11SharedTextureSource` are written this way now; a pause no longer
  waits out `TestVideoSource`'s current frame interval, nor a request a
  live mixer's tick or a WASAPI capture's poll.

- **`Filter` writes a filter as its media work alone.** Its `transform`
  makes what each buffer answers to — nothing, one buffer or several — into
  an `Output`; `drain` hands on what it still holds before the end of the
  stream, and `reset` lets go of what a seek or a stop leaves behind. The
  framework gives it its one pad, forwards `Eos` after the drain, resets it
  on a `Flush` or a `Stop`, and answers whether it can take a buffer; it
  never sees a control message. `ChainBuilder::pipe` takes one as it takes
  any filter — it now takes `impl IntoFilter<M>`, which a `RawFilter` and a
  `Filter` both are, with the marker worked out by the compiler — and
  `BoxFilter::new` makes one of it for a list, as a `Rack` takes. A filter
  written as `RawSink` and `SrcPads` is unchanged. The scalers, uploads,
  downloads, converters, chroma keys, video effects and encoders of every
  backend, `D3d11ToneMap`, the audio filters, `AudioWaveform` and
  `ChangeGate` are written this way now, with their names, constructors,
  handles and accessors as they were, but for two edges the stage settles:
  a `D3d12Scaler` whose wait for the GPU fails at the end still ends the
  stream, and a `CudaScaler` hands on every frame its graph made even where
  downstream refuses one.

- **`VulkanScaler` resizes NV12 and BGRA Vulkan frames on the GPU**, the
  Vulkan counterpart of `CudaScaler`: `VulkanDecoder -> VulkanScaler ->
  VulkanEncoder`, or a compositor into a smaller encode, stays on the
  device, where it went `VulkanDownload -> SwScaler -> VulkanUpload`. Its
  kernels are this crate's own WGSL, not FFmpeg's `scale_vulkan`, which
  only an FFmpeg built with a GLSL compiler has — the pinned vcpkg one is
  not. `VulkanScalerInterp` chooses nearest, bilinear, bicubic or Lanczos,
  each spread over the source it covers when shrinking, so a picture made
  smaller is averaged rather than sampled. One 1080p frame of a real video
  made 720p measures 65 dB PSNR against libswscale's Lanczos.

- **Vulkan frames, on Windows and Linux: `VulkanDevice`, `VulkanDecoder`,
  `VulkanUpload` and `VulkanDownload`.** `VulkanDevice` is the one Vulkan
  device a pipeline's Vulkan elements share, made by FFmpeg with whatever
  its Vulkan decoders and encoders need; `VulkanDecoder` decodes H.264,
  HEVC and AV1 with Vulkan Video on any GPU whose driver does — NVIDIA's,
  and AMD's and Intel's with Mesa — backwards too; `VulkanUpload` takes
  NV12, P010, BGRA and YUV420P (as NV12) up, and `VulkanDownload` brings
  them back. The `vulkan` feature now builds on Windows as well; the
  window renderer stays Linux's.

- **`VulkanVideoCompositor`**, the fourth compositor, on any GPU with a
  Vulkan driver, on Windows and Linux: the same `VideoLayer` API, text
  layers, live and offline, and `VideoCompositorControl`, composing NV12 and
  BGRA Vulkan frames into BGRA (`VulkanFrameFormat::Bgra`, the default, which
  keeps a translucent background) or NV12 for an encoder. It draws with
  compute shaders of its own, compiled from WGSL at construction, and uses
  FFmpeg's frames under FFmpeg's own lock and semaphores, so a decoder's
  picture is drawn once written and the output read once drawn. Two 1080p
  layers composite in about 0.35 ms on an RTX 3050.

- **`VulkanConverter`**: NV12 Vulkan frames as BGRA ones, each read by its
  own colour description — what a BGRA-only Vulkan element is put behind
  when a decoder's or a camera's NV12 arrives.

- **`VulkanVideoEffect` and `VulkanChromaKey`**: the colour correction,
  luma key and chroma key on BGRA Vulkan frames, evaluating the numbers
  every other backend's evaluates — to within one step of the software
  element's — and tuned while they run through the same
  `VideoEffectHandle` and `ChromaKeyHandle`.

- **`VideoDecodeBin` and `VideoEncodeBin` choose Vulkan too.**
  `DecodeTarget::Vulkan` decodes with Vulkan Video where the stream allows —
  alpha, a codec it has no decoder for and 10-bit going to software and up
  through `VulkanUpload`, as on the other targets — and `EncodeInput::Vulkan`
  encodes NV12 with `h264_vulkan`, falling back to software after a
  `VulkanDownload`. `EncodeInput::for_decoded` pairs the two. Nothing on
  Vulkan converts colour yet: BT.2020 and HDR stay NV12 with their own tags,
  and BGRA is encoded in software.

- **`VulkanEncoder`**: H.264, H.265 and AV1 with Vulkan Video
  (`VulkanCodec`), from NV12 Vulkan frames — a `VulkanDecoder`'s, an NV12
  `VulkanVideoCompositor`'s or a `VulkanUpload`'s — with the options
  `CudaEncoder` takes. A codec the GPU does not encode fails to open, by
  name. So a whole edit — decode, composite, encode — runs on the GPU on
  any vendor's hardware whose driver has Vulkan Video.

- **One API for every video compositor: `VideoCompositorControl`,
  `VideoLayerControl` and `TextLayerControl`.** Each compositor's handles
  had the same methods, each answering its own error type; the traits are
  those methods once, answering the crate's `Error`, so what builds a
  composition is written once and handed whichever compositor the
  platform has. `CompositorInput<L>` is what `add_source` returns on every
  backend — `SwVideoCompositorInput` and its siblings are now aliases of
  it, with the same two fields. To make the three agree,
  `SwVideoCompositorHandle::add_layer` and `CudaVideoCompositorHandle::add_layer`
  register an input whose picture is set through the new
  `set_frame` on its layer handle, as `D3d11VideoCompositorHandle` already
  could — a still image or a title card with no pipeline behind it — and
  `CudaTextLayerHandle` has `set_z_index`.

- **A video compositor can render offline:
  `RenderMode::Offline { end }`.** Live, a compositor emits at its own rate
  by the wall clock and draws whatever each input last handed over.
  Offline it emits the frame for each output time as soon as every input
  has said what it shows then, and goes straight on to the next — as fast
  as the inputs decode, and the same output however fast that is. What an
  export wants, where the live mode is what a preview wants.

  Each input frame is placed by its own `pts`, read in its own time base
  and compared exactly (not through floating-point seconds), so inputs at
  different rates and in different time bases line up to the frame. A
  frame is shown until its duration runs out or the next begins. An input
  holds its pipeline back once it has a frame past the output time being
  made — a `Queue` somewhere in that pipeline is what waits — and is
  woken the moment the compositor moves on. The render ends at `end`, or,
  without one, once every input fed through a sink has ended and been
  shown to its last frame's end. An offline compositor is not live, and
  refuses `set_frame_rate` with `FixedFrameRate`: every output time is a
  count of its frames.

- **An `AudioMixer` can mix offline**, so an export's sound is made the way
  its picture is. Each input's sound is placed where its `pts` says, in its
  own time base and at its own sample rate, with silence where no input
  has anything; each twenty-millisecond stretch is mixed once every input
  has what belongs in it; and the mix ends at `end` or once every input has
  ended and been mixed to its last sample. Where an input's timestamps and
  the sound it has already given disagree by more than twenty milliseconds,
  the difference is a gap, filled with silence, or an overlap, dropped. Its
  format is fixed:
  `set_mix_format` answers `FixedMixFormat`.

- **`FileDemuxerHandle::in_lap` reads a timestamp back as a position in the
  file, on its own lap.** `lap_offset` is the lap the demuxer is reading,
  which is ahead of the picture by what the queues and a decoder hold — the
  next lap at the end of one played forwards, the one before at the start
  of one played backwards; at four times the speed that was two seconds of
  a progress bar gone blank at the end of every lap. `Player::position`
  reads its lap this way now.

### Changed

- **One `MfVirtualCamera` feeds the camera at a time.** Two at once — in
  one process or two, for one user or two — wrote over each other's
  pictures in the one section the camera reads. The second
  `MfVirtualCamera::new` now answers `MfVirtualCameraError::InUse`, a new
  variant of that `non_exhaustive` enum, until the first is dropped or its
  process ends.

- **The API documentation on GitHub Pages covers CUDA and the GPU
  inference providers.** The Windows pages were built without `cuda`,
  `cuda-visual-tracking`, `ort-cuda`, `ort-tensorrt` and `ort-directml`,
  and the macOS pages without `ort-coreml`, so no CUDA type, and no
  detector, classifier or embedder on a GPU, was documented anywhere. The
  build now checks a page of each is there.

- **`CudaOrtDetector` keeps a RetinaFace model's faces on the GPU.** Its
  outputs are a row for every anchor — 16,800 of a 640 input, a megabyte
  an input — of which a handful are faces; all of them came down to be
  read on the CPU. A kernel now keeps the anchors sure enough of a face
  where they are and only those come down, read to exactly what reading
  every anchor does. RetinaFace R50 through TensorRT on an RTX 3050, seven
  inputs a picture: 34.4 ms a picture before, 33.1 ms after.

- **`D3d11Decoder` hands on copies rather than running its pool out.**
  Downstream holding more pictures than `downstream_hw_frames` failed the
  stream with `SurfacePoolExhausted`, the fixed D3D11VA pool having no
  surface left to decode into. A picture that would leave downstream
  holding more than the surfaces asked for now goes on as a copy in a
  texture of its own, made on the GPU, its surface back to the pool, until
  downstream lets go; the log says so once. Within the budget nothing is
  copied. `SurfacePoolExhausted` stays for a codec keeping more references
  than the decoder counts on. A copy of a picture of an odd size, which
  playing backwards makes too, is now made at the even size D3D11 takes.

- **`MetalOrtDetector` runs a single picture at the cost of one.** A model
  whose batch is left open by name was fixed at `max_batch` for Core ML
  and every run computed that many inputs: a face detector of a picture
  and its six tiles ran a close-up with no tiles at the cost of seven. A
  run of one input now goes through a session fixed at one, made the
  first time one comes and kept in the model cache. On a music show 56%
  of the face detector's runs were one picture, and twenty seconds of it
  analysed in gamchum in 73 s where it took 174 s, with the same tracks;
  the second session takes 165 MB. Sessions at 2 and 4 as well cost
  440 MB in all and saved no more time.

- **`SwOrtDetector` and `SwOrtClassifier` read an untagged HD picture as
  BT.709,** as the CUDA and Metal detectors and every renderer already did:
  a picture that names no matrix is BT.709 above 576 lines and BT.601 at
  or below (`color::matrix_of`). swscale's own reading was BT.601 at any
  size, so the same 1080p picture was handed to a model in other colours
  on the CPU than on the GPU — its scores differed by about 0.05.

- **A YOLOv8 or YOLO11 detector reads its model's output about six times
  faster.** Each box's best class was found by walking a column of the
  output through a view whose dimension was known only at run time — about
  1.8 ms a picture for YOLO11n's 8400 boxes, measured on a desktop CPU; now
  a row at a time through a three-axis view, about 0.3 ms. Every
  detector reads its output the same way, so `SwOrtDetector`,
  `CudaOrtDetector` and `MetalOrtDetector` all gain; YOLOv10 and YOLO26
  outputs, already small, are unchanged.

- **The video analysis elements are a module of their own,
  `elements::vision`,** no longer inside `elements::filter`: detectors,
  classifiers, the tracker, analytics and the overlays are grouped by what
  they are for, as a stream mux for batched inference is to join them. The
  flat paths — `media_pp::elements::ObjectTracker` and the rest — are
  unchanged; only a path through `elements::filter::` has to become
  `elements::vision::`.

- **`ort-cuda`, `ort-tensorrt` and `cuda-visual-tracking` build on Windows
  with nothing of NVIDIA's installed.** The few calls this crate makes into
  CUDA's runtime, cuBLAS, cuRAND, cuFFT, cuDNN and TensorRT name each DLL
  themselves (`raw-dylib`), so no import library is looked for — NVIDIA's
  pip wheels, which is how TensorRT comes without an account, carry none.
  Running needs the DLLs on `PATH`, as it needed them before. Linux is
  unchanged. The `cuda_detect` and `cuda_track` examples run on Windows
  too.

- **The macOS API is documented too, beside the Windows one.** docs.rs
  builds on Linux only, so both are built on a runner of their own platform
  and published together: the Windows documentation moved from
  `hash1018.github.io/media-pp/media_pp/` to `.../windows/media_pp/` (the
  old address sends a visitor on), and the macOS one is at
  `.../macos/media_pp/`.

### Fixed

- **The `ort` features build in a new crate again.** media-pp asked for
  `ort` `2.0.0-rc.13` as a caret requirement, so a crate without a lockfile
  of its own resolved `2.0.0-rc.14`, whose `Session::builder` takes an
  argument, and media-pp failed to compile in it — after its build script
  had downloaded ONNX Runtime anew. It now asks for `=2.0.0-rc.13`. With
  0.3.1, `cargo update -p ort --precise 2.0.0-rc.13` holds it back.

- **HEVC from `D3d12VideoEncoder` goes into an MP4 or MOV as `hvc1`.**
  FFmpeg 8.0's `hevc_d3d12va` makes no extradata, so `FileMuxer` left the
  track FFmpeg's `hev1`, which Apple's players refuse. The encoder's
  parameter sets never change, which its track now says, so the muxer
  writes `hvc1` with them in the sample entry, as for every other encoder
  of this crate; a stream copied from elsewhere with its sets in band stays
  `hev1`. `SegmentedFileMuxer` and `ReplayBuffer` follow.

- **A failure downstream of a `Source` is reported as the element that
  failed.** The framework's source loop posted what a decoder or a filter
  behind it refused under the source's own name, so `BusEvent::Error`
  named a `FileDemuxer` or a WebRTC track for a decoder's error; it now
  names the element the error traces back to, as the control path
  already did.

- **A GPU compositor's input says once that its compositor stopped.** An
  input of `CudaVideoCompositor`, `VulkanVideoCompositor`,
  `MetalVideoCompositor` or `D3d12VideoCompositor` refused every picture
  after its compositor was gone, so a pipeline still feeding one — an
  offline export's sources after its end — posted the same
  `BusEvent::Error` once a picture. It now refuses the first and drops the
  rest, as the software and D3D11 compositors' inputs take them without a
  word.

- **A D3D12 detector, classifier or embedder warns of a model past opset
  20.** DirectML hands the operators of a later opset to the CPU, which made
  a YOLO11 exported at opset 22 five times slower than at 13 without a word
  from ONNX Runtime. The element now reads the model's opset and warns when
  it is made.

- **A replay buffer's failed save no longer deletes the file at its
  path.** `ReplayBufferHandle::save` removed the path on any failure,
  including one before anything was written — a path naming no container
  deleted the clip saved there before. The clip is now written beside the
  path and put in its place once whole, so a file there is replaced by a
  finished clip or kept as it was.

- **What a `RackHandle` or a `TeeHandle` lets go of is dropped outside
  their locks.** A replacement overtaken before the rack took it up was
  dropped holding the handle's slot, and a branch a `Tee` refused holding
  its branch list; a filter or sink whose drop reached for the same handle
  then waited on itself forever.

- **Turning playback around, and stepping, go on from the picture last
  shown.** `Pipeline::set_rate` turning backwards into forwards, or the
  other way, and `Pipeline::step` read which picture was shown before
  pausing, so a picture shown in between was not the one gone on from:
  going forwards again a picture was skipped, seen on Linux CI under load
  (shown 2.333 s, forwards again from 2.4 s). They pause first now, then
  read it.

- **`CudaOrtDetector`, `CudaOrtClassifier` and `CudaOrtEmbedder` hand
  the model their input where it is.** `ort`'s `TensorRefMut::from_raw`
  locates every tensor it makes in host memory, whatever memory it is
  told, so ONNX Runtime took the fitted input on the GPU for host memory
  and copied the whole of it to the device once more every run — which
  unified addressing let work, and hid. The input is now made a tensor
  through ONNX Runtime itself, on the device it is on. RetinaFace R50
  through TensorRT on an RTX 3050, seven inputs a picture: 37.5 ms a
  picture before, 36.8 ms after.

- **`SwDecoder` stamps a sound trimmed at its start where it plays.** A
  file cut out of a longer one, or whose edit list starts partway into a
  frame, has its demuxer say how many samples of the first frame not to
  play; FFmpeg drops them, but moves the frame's timestamp past them only
  when the codec knows the packets' time base, which nothing told it. The
  rest of that frame came out stamped where the undropped frame began, a
  hundredth of a second or so early, with a gap behind it. The decoder now
  takes the time base from the packets. Checked against FFmpeg's own
  reading of seven files with edit lists (a stream copy cut at a
  non-keyframe, an empty edit, B-frame delays, H.264 and HEVC): every
  picture's time and pixels, through `SwDecoder` and `CudaDecoder`, and
  where an accurate seek lands, already agreed; only this sound did not.

- **`CudaDecoder` decodes a stream with many reference frames whatever
  `downstream_hw_frames` is.** NVDEC caps its pool at 32 surfaces, the
  stream's reference frames and FFmpeg's working ones among them; an HEVC
  encoded with B-frames as references and a downstream of 20 came to 33,
  and `cuvidCreateDecoder` failed the stream at its first picture
  (`HwAccelUnavailable`). The pool now opens with as many extra surfaces
  as fit once the stream says how many reference frames it keeps. What
  downstream holds never took them: FFmpeg copies each picture out into a
  pool that grows, so a downstream of any depth decodes. The docs said
  otherwise and are corrected.

- **`VideoToolboxEncoder` with B-frames writes an MP4.** VideoToolbox's
  H.264 reorders pictures two deep, even where one B-frame is allowed, and
  FFmpeg's `h264_videotoolbox` dates them one deep: a packet then decoded
  after it was shown, and `FileMuxer` failed with `Invalid argument`. With
  `max_b_frames` above 0 the encoder now dates each packet itself, the
  earliest presentation time come out four packets before; on H.264 and
  HEVC with one and two B-frames every picture comes back out of the file,
  in order.

- **HEVC in MP4 and MOV plays on Apple devices.** `FileMuxer` writes an
  HEVC track as `hvc1` — its parameter sets in the sample entry, as an
  encoder of this crate puts them out of band — where FFmpeg wrote `hev1`,
  which an iPhone, QuickTime and Safari refuse to play. A track that brings
  a tag of its own, copied from another file, keeps it.

- **A stream that appears after a file or RTSP session was opened is
  passed over.** `FileDemuxer` and `RtspSource` have an output for each
  stream there was when they opened; MPEG-TS and FLV, which have no header
  to list their streams, may announce another later. A packet of one was
  handed to an output that does not exist, which ended the source with
  `ProduceError::NoOutput`, a whole recording or session lost to a stream
  nothing could have been linked to. Its packets are now dropped.

- **A panic on a pipeline's thread is told on the bus.** A queue's worker
  that panicked — an element behind it, or something it called — ended
  its thread with nothing posted, and so did the source's thread: the
  branch went quiet, no `Finished` ever came, and a caller waiting for it
  waited for good. Both now catch it and post `BusEvent::Error` with the
  new `Error::Panicked`, naming the thread and what the panic said — as
  the element a queue feeds, or as the source — and the source's thread
  stops its branch as it does on a returned error. A detection overlay
  that panicked took a benchmark run three hours to notice.

- **`Player::position` after a step is where the picture is.** With sound,
  the sound renderer sets the clock from the sample it has reached, and
  after a step that was up to a buffer away from the picture shown — 550 ms
  for a picture at 566.7 ms. Paused after a seek or a step, the position
  is now where they put it, as the docs said.

- **A playing seek behind a `Tee` shows where it landed.** With every
  branch of a `Tee` held by a seek's preroll, the `Tee` still took what
  came, and everything decoded meanwhile went on at once, past the pacer,
  as playback did: the picture a second or two past the target, and the
  sound before it let through at once. A `Tee` with every branch held now
  waits.

- **A finish right after an accurate seek plays nothing from before the
  target.** Ending the stream before the sound had reached the target used
  to hand on the last sample before it, as the end of the media would.

- **A preroll's terminal is handed only what it asks for.** A transform
  that makes several buffers of one — a tempo stretch, `CudaScaler`, a
  transform of your own — pushed them all into a terminal that already had
  its sample, where nothing asked the pad between them. What its pad
  cannot take while a preroll, or the pause after one, holds the graph is
  now kept in order and handed on first once it can, the stage answering
  not ready meanwhile; outside a preroll nothing changes. The decoders,
  which held theirs back by hand after one sample, keep them the same way
  now, and so do `VideoDecodeBin` and `Rack` for what their line hands
  them: the decoders only drop what comes before an accurate seek's
  target, whatever the phase, and the terminal's own readiness holds the
  rest.

- **A seek to the last pictures of a file whose sound ends sooner no
  longer times out.** The picture's branch took its sample and nothing
  more, so `FileDemuxer` parked the packets it could not hand on; the
  sound had nothing at or after the target and could only preroll on its
  end of stream. At the end of the file the demuxer waited for every parked
  packet to go before ending any pad, which was until the preroll ended,
  so the sound's end never came and the seek failed after five seconds.
  A pad that owes nothing parked now has its end at once; another's parked
  packets and end follow as it takes them. Rare — once in a few hundred of
  the conformance matrix's sequences, with one-deep queues — which is how
  it was found, and traced.

- **After a paused seek, `WasapiRenderer` plays from the seek's target.**
  The one sample a paused seek hands every terminal reached it while it
  was paused, and it dropped it, so once playback resumed the sound began
  a frame after the picture. It now waits in the device's buffer, the
  device still stopped — no blip — and plays on `Resume`, as
  `PipeWireAudioRenderer`'s already did.

- **An element refusing a pipeline's control no longer ends its source.**
  A sink on a source's own thread — no `Queue` between them — that
  failed a `Pause`, say, ended that source's thread, while the same sink
  behind a queue was a bus event and nothing more. It is a bus event
  wherever the element is now, under the element's own name, and the
  source goes on; every pad is still told, whatever the one before
  answered.

- **A pipeline with a `ReplayBuffer` refuses a seek again.** Its tracks
  refused one while the question was asked of the running graph, and
  answered yes once 0.3 took the answer from the wiring, so a seek went
  ahead and the window measured its length across the jump.

- **A `Rack` or a `VideoEncodeBin` says it cannot take a buffer when what
  is in it cannot**, as that element would in its place, instead of taking
  it and leaving the thread in front waiting inside the push.

- **A stream ends once.** An `Eos` pushed after one that went, with
  nothing in between, goes no further than the pad: a source asked to
  finish while it waited, paused, at the end it had already handed on sent
  a second one, which a muxer would finish its file on twice.

- **Every video compositor does the same with an input that ends, or is
  sought.** A live input's `Eos` removed it from `SwVideoCompositor` and
  `D3d11VideoCompositor`, so a file played to its end and then sought
  back was never shown again; `VulkanVideoCompositor` and
  `CudaVideoCompositor` kept its input, and its last picture on screen,
  and kept the picture from before a seek until the new one came. Now on
  every one an `Eos` and a seek's `Flush` leave the input showing nothing
  until its next picture, and only a `Stop` or `remove_source` removes
  it — so `source_count` counts an ended input on the first two, as it
  already did on the others.

- **A pause, seek or stop reaches every queue, whatever the thread before
  it is doing.** A request went from each source down its line, one thread
  handing it to the next, so a queue whose upstream thread was blocked
  handing it data, or had ended, heard nothing and held the whole request
  up. The pipeline now asks every source and every queue worker itself,
  and each passes the request on only to the elements on its own thread.
  `Pipeline::stop` now also returns once every source thread and the queue
  workers behind it have ended, as `finish` does, rather than leaving that
  to be read from the bus.

- **An accurate seek on a looping file shows its target on every lap.**
  A seek is a place in the file, and `FileDemuxer` stays on the lap
  playback is on, so the pictures after it are stamped a lap or more
  further on; the decoders held the target as the place in the file, took
  the first picture as past it, and showed the keyframe before the target
  instead. The stream now says, as a seek begins it, where its target is
  on the timeline the pictures are stamped on, and the decoders hold the
  target there. Found by the conformance matrix.

- **A file whose packets are fanned out to several branches steps
  backwards in every branch.** Played backwards, `FileDemuxer` handed each
  packet over without asking whether its pad could take it, and a branch
  that had its step's picture took nothing more: the demuxer waited inside
  that push, which a `Tee` keeping what comes for such a branch could not
  let go, and its sibling, a stretch short of its next picture, showed
  nothing until the step timed out. A packet read backwards is now held
  until its pad can take it, as one read forwards is. Found by the
  conformance matrix, which now plays that shape backwards too.

- **A paused seek to the end of a file prerolls every branch, however
  shallow its queues.** At the end of a file `FileDemuxer` handed each pad
  its `Eos` in turn, waiting for room — the picture's first. In a preroll a
  branch that has its sample takes nothing more, so with one-deep queues
  the push waited there, and the sound's `Eos`, which answers a seek past
  its last sample with the sample before it, waited behind it until the
  seek timed out. Each pad is now handed its end as soon as it can take
  one. Found by the conformance matrix.

- **`Player` turns round to play backwards at once on Linux.** Turning
  round pauses and flushes the sound, and as the picture is read back to
  the start of its file the sound's branch gets its `Eos` — which
  `PipeWireAudioRenderer` drained, asking PipeWire to report when a stream
  it had just paused had played out. PipeWire reports that only of a
  stream the graph is running, so it waited out its five-second limit, and
  the pipeline's next control reached nothing behind it until it had. With
  nothing handed to the stream since it was last emptied, an `Eos` now has
  nothing to wait for.

- **A packet from `VulkanEncoder` can outlive the encoder.** FFmpeg's
  Vulkan encoders hand out a view of a buffer from the encoder's own pool,
  whose free callback reaches through the encoder's Vulkan context; one
  let go of after the encoder was — in a muxer's queue as a recording
  stops — faulted inside libavcodec. Each packet is now copied into a
  buffer of its own before it leaves the element. Found as a crash in this
  crate's own tests one run in two on an RTX 3050 with FFmpeg 8.1.

- **A looping file turns round, and plays backwards, on any lap.** Where
  playback turns round is the picture shown, on the timeline looping carries
  past the file's end; `FileDemuxer` read it as a position in the file, so
  turned round a lap or more in it went back from the file's end, stamped a
  lap short, and the picture stood still until the clock came down to it.
  A position past the end of a lap is now put back into its lap, what is
  read backwards is stamped on that lap, and at the start of a lap it goes
  on from the end of the one before — down to the start of the first, where
  the stream ends as it always has.

- **Theora, not ProRes, is what `VulkanDecoder::supports` is tested to
  refuse**: FFmpeg 8.1 decodes ProRes on Vulkan's compute queues, and the
  test failed there.

- **The `vulkan` feature says what to install where Linux has no Vulkan
  headers**, instead of ending in a bindgen panic.

- **A program built with `cuda` starts without NVIDIA's driver.** The CUDA
  driver calls this crate makes itself linked `libcuda.so.1` — `nvcuda.dll`
  on Windows — so a machine without the driver could not load the program
  at all, and one meant to look for a GPU and use Vulkan where there is no
  NVIDIA one never got to look. The driver is now opened when first called,
  as FFmpeg opens it for its own CUDA code, and where it is missing the call
  fails with an error — `CudaDevice::new` already does, first. Nothing to
  link either: a build machine needs no driver or stub library.

## 0.3.1

One fix: a pipeline stopped just as its source ended could hang for good.

### Fixed

- **A pipeline stopped as its source ends no longer hangs.** A source that
  ended drops the `Queue` behind it with its `Eos` still inside, and a
  dropped queue stays until that `Eos` is handed on. Stopped a moment
  later, the pipeline's terminals hold for good, and the `Stop` message
  never reaches a queue whose source has already gone — so the queue waited
  on a terminal that would never take anything, the source's thread waited
  on the queue, and `Pipeline::stop` and its drop waited on the source. A
  stopped pipeline is now owed no end: the queue abandons what it holds, as
  a stop means to.

## 0.3.0

Playback, mostly. A file plays in a window with its sound (`Player`,
`VideoWindow`, window renderers on D3D11, D3D12 and Vulkan), and can be
sought, stepped, and played faster, slower or backwards. Most of what breaks
is the wiring saying more before anything runs: link contracts that name a
pixel layout, a seek refused from the graph rather than by the source,
frames that carry their own time base, and a pipeline that hands back what
its wiring made. Also new: filters that clean up a microphone, speech turned
into subtitles, a replay buffer, RTMP publishing, and HDR brought to SDR on
the GPU.

### Breaking

- **A source that can be sought is a `SeekableSource`;
  `SourceElement::is_seekable` and `SourceElement::seek` are gone.** Every
  source had to write both, and all but `FileDemuxer` answered `false` and
  wrote a `seek` that refused, each with a `SeekUnsupported` error of its
  own. `seek` is now the one method of `SeekableSource`, and a source says
  it is one from `SourceElement::as_seekable`, `None` by default — as it
  says it is a `ReversibleSource`, which extends it, for playing
  backwards. To migrate, a source that answered `is_seekable() == false`
  deletes `is_seekable` and `seek`; one that answered `true` deletes
  `is_seekable`, moves `seek` into `impl SeekableSource for YourSource`,
  and adds `fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
  Some(self) }`. `SeekUnsupported` is gone from `AudioMixerError`,
  `RtspSourceError`, `PipelineBridgeError`, the capture errors
  (`DxgiCaptureSourceError`, `WgcCaptureSourceError`, `MfCaptureSourceError`,
  `WasapiCaptureSourceError`, `PipeWireAudioCaptureSourceError`,
  `PipeWireScreenCaptureSourceError`) and the compositor errors (`SwVideoCompositorError`,
  `D3d11VideoCompositorError`, `CudaVideoCompositorError`);
  `TestVideoSourceError` and `TestAudioSourceError`, which had nothing
  else, are gone with their `Error` variants. A pipeline refuses to seek
  such a source as before, from `Pipeline::check_seek`; one driven by hand
  and sent a `Seek` answers a `SeekError` naming `SourceNotSeekable`.

- **`D3d11DecoderError` and `D3d12DecoderError` have a `Copy` variant.**
  It is a copy of a decoded picture failing on the device, which playing
  backwards makes — see `Pipeline::set_rate` under Added. A `match` on
  either needs the arm.

- **`SeekRejectReason` has two more variants: `SourceNotReversible` and
  `DecoderNotReversible`.** They are why `Pipeline::check_reverse` refuses
  to play backwards — see `Pipeline::set_rate` under Added. A `match` on
  `SeekRejectReason` needs the two arms; `check_seek` never answers with
  them.

- **A sink that cannot follow a seek says so with `Sink::accepts_seek`;
  `ControlMsg::CheckSeek` and `SeekCheckContext` are gone.** Whether a seek
  would be refused was asked by sending `CheckSeek` down the running graph
  and collecting refusals in a `SeekCheckContext`. Everything that refuses
  decides it as it is built — a live or non-seekable source, a recording —
  so the answer is now taken as each element is wired and kept with the
  graph; see `Pipeline::check_seek` under Added. A sink of your own that
  refused `CheckSeek` returns `false` from `accepts_seek` instead, and a
  `match` on `ControlMsg` loses the arm.

- **`Sink::control` is an element's own reaction; the graph passes the
  message on.** It took a `ControlMsg`, had no default, and every filter
  ended it by forwarding to its own pad — so a filter that forgot, or
  returned early on an error, stopped the message there, and a message
  added later had to be taught to every element. It now takes
  `&ControlMsg`, does nothing by default, and does not forward: whatever
  delivers a message to a filter hands it on through the filter's
  `src_pads()` afterwards — to every pad even where one fails, and even
  where the filter's own reaction failed. `SrcPad::control` is no longer
  public, since a filter still calling it would hand everything after it
  each message twice. To migrate, take `msg: &ControlMsg`, end with
  `Ok(())` where you called `self.pad.control(msg)`, and delete a
  `control` that did nothing else. Code that drives a filter by hand,
  outside a pipeline, calls `control::deliver(&mut filter, &msg)` for the
  old forwarding. `AppSink::with_control`'s callback is unchanged.

- **`RtspMuxer::open` gives up on a server that does not answer.** It
  waited without limit: publishing to an address with nothing listening,
  or to a server that took the connection and never replied, never
  returned. `RtspMuxer::create` takes the `RtspOptions` `RtspSource::open`
  already did — transport and a timeout, 5 seconds by default — and the
  timeout bounds the handshake and every write after it. Write
  `RtspMuxer::create(url, RtspOptions::default())`, or
  `RtspOptions { transport, ..Default::default() }` for a transport of your
  own. `RtspOptions` is still `media_pp::elements::RtspOptions`; the
  duplicate path through `elements::source` is gone.

- **`FileMuxer::create` names the file it could not create.** A name with
  no container in it — `out`, `out.unknownext` — was
  `FileMuxerError::Ffmpeg(Invalid argument)`, and a directory that is not
  there `No such file or directory`, neither saying which file. They are
  now `FileMuxerError::UnknownContainer { path }` and
  `FileMuxerError::Create { path, source }`. A match on `FileMuxerError`
  needs the two arms.

- **A presenter of your own is told what colour an NV12 frame is.**
  `D3d11FrameRenderer::submit_nv12_texture`,
  `D3d12FrameRenderer::submit_nv12_texture` and
  `CudaFrameRenderer::submit_nv12` take a last `color: ColorDescription`
  parameter: what the frame says of its matrix, range, primaries and
  transfer. Without it a presenter could only guess, and the usual guess —
  one fixed BT.601 matrix — draws a decoded HD stream, which is BT.709,
  with visibly wrong colours. Add the parameter to an implementation;
  `color.yuv_to_rgb_rows(height)`, also new, gives the three rows a shader
  converts with, filling in what the frame leaves unsaid the way this
  crate's own renderers do. `ColorDescription::of(frame)` reads the same
  description from any frame.

- **`FileDemuxer::open` names the file it could not open, and refuses one
  with nothing in it.** A missing or unreadable file was
  `FileDemuxerError::Ffmpeg("No such file or directory")`, with no word of
  which; it is now `FileDemuxerError::Open { path, source }`, which says. A
  file that opens with no streams at all — a recording finished before
  anything was written into it — was opened with an empty stream list for a
  caller to index into; it is now `FileDemuxerError::NoStreams { path }`.
  Code matching `FileDemuxerError::Ffmpeg` for an open failure matches
  `Open` instead.

- **`StreamInfo` has a `frame_rate` field**, so a literal of one needs it
  — see Added.

- **A file's source waits at its end instead of ending.** `FileDemuxer`
  sent `Eos` at the end of the file and ended its thread, and with it the
  queues it owns, a second or more of the file still in them. A `Pause` from
  then on reached nothing, so the last second played on through one; a
  paused seek near the end lost its picture and its `Eos` and never
  finished; and a file could not be sought back once it had ended —
  `PipelineError::NotRunning`. It now stays at the end, passing control on,
  until the pipeline is stopped, and a seek from there plays on. So a
  pipeline reading a file no longer ends by itself: `Pipeline::is_running`
  stays `true` and `BusReceiver::iter` does not end until it is stopped.
  Stop it on `BusEvent::Finished`, which says everything read has reached
  its terminals — what the examples and `Player` already did. A
  `FileDemuxer` driven by hand returns from `run` at the end once its
  control sender is gone.

- **D3D elements take their `D3d11Gpu` or `D3d12Gpu`.** Every D3D11
  element took an `&ID3D11Device` and, where it draws or copies, a separate
  `Arc<Mutex<ID3D11DeviceContext>>` — two values that had to belong to the
  same device, read out of a `D3d11Gpu` one at a time, next to a
  `D3d11WindowRenderer::open(&gpu)` that already took the GPU whole. Each
  now takes `gpu: &D3d11Gpu` where the device went, and the context
  argument is gone: the element reads both from the one GPU, so a context
  from another device can no longer be handed over. The D3D12 elements take
  `&D3d12Gpu` the same way. That is `D3d11ChromaKey::new`,
  `D3d11Decoder::new`, `D3d11Download::new`, `D3d11Scaler::new` and
  `to_format`, `D3d11ToneMap::new`, `D3d11Upload::new`,
  `D3d11VideoCompositor::new`, `D3d11VideoEffect::new`,
  `D3d11VideoEncoder::new` and `with_color`,
  `D3d11SharedTextureSource::new`, `DxgiCaptureSource::open_with_device`,
  `WgcCaptureSource::open_with_device`, `D3d12Decoder::new`,
  `D3d12Scaler::new` and `D3d12Upload::new`; and `DecodeTarget::D3d11`
  holds `gpu` in place of `device` and `context`, `DecodeTarget::D3d12`
  `gpu` in place of `device`.

  A capture that makes its own device hands it back ready to share:
  `DxgiCaptureSource::open` returns an `Option<D3d11Gpu>` where it returned
  an `Option<ID3D11Device>`, and `WgcCaptureSource::open` a `D3d11Gpu`
  where it returned an `ID3D11Device`, each with a new `Gpu` variant on its
  error for the one way that can fail. The `ContextDeviceMismatch` variants
  of `D3d11ChromaKeyError`, `D3d11ScalerError`, `D3d11ToneMapError`,
  `D3d11VideoEffectError` and `D3d11VideoEncoderError` are gone, there
  being no context left to mismatch; a single-threaded device is refused by
  `D3d11Gpu::from_device`, before any element has it.

  ```rust
  // before
  let upload = D3d11Upload::new("upload", gpu.device());
  let scaler = D3d11Scaler::new("scale", gpu.device(), gpu.context(), format, 1280, 720)?;
  let decoder = D3d11Decoder::new("decoder", params, gpu.device(), 8)?;
  let (capture, _format) = DxgiCaptureSource::open_with_device("screen", options, gpu.device())?;
  let target = DecodeTarget::D3d11 {
      device: gpu.device().clone(),
      context: gpu.context(),
      downstream_hw_frames: 8,
  };
  let (source, format, device) = DxgiCaptureSource::open("screen", options)?;
  let gpu = D3d11Gpu::from_device(device.expect("GPU mode returns a device"))?;
  let upload = D3d12Upload::new("upload", d3d12.device())?;

  // after
  let upload = D3d11Upload::new("upload", &gpu);
  let scaler = D3d11Scaler::new("scale", &gpu, format, 1280, 720)?;
  let decoder = D3d11Decoder::new("decoder", params, &gpu, 8)?;
  let (capture, _format) = DxgiCaptureSource::open_with_device("screen", options, &gpu)?;
  let target = DecodeTarget::D3d11 { gpu: gpu.clone(), downstream_hw_frames: 8 };
  let (source, format, gpu) = DxgiCaptureSource::open("screen", options)?;
  let gpu = gpu.expect("GPU mode returns a GPU");
  let upload = D3d12Upload::new("upload", &d3d12)?;
  ```

- **`PixelLayout::Yuv420p` and `Rgb24`: a link check tells them from other
  layouts.** YUV420P was `PixelLayout::Other`, so a consumer that draws it
  had to accept every other layout too, and a pipeline feeding it RGB24 or
  4:4:4 linked and failed on its first frame. `PixelLayout::of` maps
  `YUV420P` and `YUVJ420P` to the new `Yuv420p`, which `SwScaler` and
  `SwEncoder` state through it with no change of their own, and
  `TestVideoSource` states it too; `PixelLayoutSet::YUV420P` is the set of
  it alone. `PixelLayout::Rgb24` is added beside it for the same reason:
  `OrtDetector` takes RGB24 alone and said `Other`, which linked a YUV420P
  source to it; it says `Rgb24` now. `Other` is left for producers — a
  format with no layout of its own, which a consumer naming its layouts
  refuses — and no consumer here states it. A `match` on `PixelLayout`
  needs arms for both, and a contract written with `Other` to take YUV420P
  or RGB24 needs the new layout instead.

- **A preroll timeout names what it waited on.** `PrerollError::TimedOut`
  carried `pending: Vec<ElementId>`, so a failed seek said
  `pending terminals [ElementId(8)]` and left the caller to work out which
  branch that was. It carries `Vec<NodeInfo>` — each terminal's id, type
  and name, from the pipeline's graph — and reads `preroll timed out
  waiting on audio (Other #8)`.

- **`SwEncoder` is opened for one pixel format, and refuses any other.**
  It encoded whatever frame it was handed as if it were `YUV420P`: a BGRA
  screen capture wired straight in linked, ran without an error, and wrote
  a file that played as solid green. `SwEncoderOptions` has a new required
  `pixel_format` — `YUV420P` for what every example here does — which the
  encoder is opened for after checking that the codec takes it
  (`SwEncoderError::UnsupportedPixelFormat` names the ones it does), so a
  codec such as `libx264` can take NV12 or 10-bit frames with no conversion
  in front. A frame in another format or size is refused with
  `SwEncoderError::FrameMismatch`, and a wiring that can only deliver
  another layout no longer links. `YUVJ420P` fits a `YUV420P` encoder, being
  the same planes.

- **`FileDemuxError` is `FileDemuxerError`.** Every other element's error
  is named after the element — `FileMuxerError`, `SwDecoderError` — and a
  search for `FileDemuxerError` found nothing. The crate `Error` variant is
  renamed with it.

- **Screen captures take a `frame_rate`, not `fps`.** `DxgiCaptureOptions`,
  `WgcCaptureOptions` and `PipeWireScreenCaptureOptions` took `fps: u32`,
  while their own runtime `FrameRateHandle::set`, every encoder, the
  compositors, the test sources and the webcam all take an
  `ffmpeg::Rational`. A recording wrote the same rate twice in two shapes,
  and `30000/1001` could not be asked for at all. Each now has
  `frame_rate: ffmpeg::Rational` (`30/1` by default), so one value goes to
  the capture and the encoder alike. A rate that is not positive is refused
  by `open` with the new `InvalidFrameRate` on `DxgiCaptureSourceError` and
  `PipeWireScreenCaptureSourceError` — DXGI and PipeWire used to run a `0`
  at 1 fps — and `WgcCaptureSourceError::InvalidFps` becomes
  `InvalidFrameRate` carrying the rate. PipeWire refuses it before the
  portal shows a dialog.

- **A `Pipeline` says when `run` or `seek` did nothing.** Calling `run`
  on a pipeline that had already been run returned `Ok` and did nothing,
  so a second play-through showed neither an error nor any playback; it
  fails with the new `PipelineError::AlreadyStarted` and leaves the first
  run alone. `seek` before `run`, or once every source has stopped,
  returned `Ok` without moving anything; it fails with
  `PipelineError::NotRunning`. A source parked at the end of its stream
  still counts as running, so seeking back into a finished file works as
  it did. `PipelineError` is in `media_pp::pipeline` and converts into the
  crate's `Error`. `pause`, `resume` and `stop` stay quiet no-ops outside a
  run, since asking for a state a pipeline is already in is harmless.

- **`log::init` takes its directory as a path.** `log_path` was a `&str`,
  the one file location in the crate that was, so a caller holding a
  `PathBuf` had to convert it with `to_string_lossy` — which turns a name
  that is not UTF-8 into a different one, and logged somewhere other than
  where it was asked to. It is `impl AsRef<Path>` now: a `&str` or
  `&String` still works as it did, and a `Path` or `PathBuf` goes in
  directly. What no longer compiles is what relied on coercing to `&str`,
  such as `&path.to_string_lossy()`; pass `&path` itself.

- **`CudaUpload::new` no longer returns a `Result`.** Drop the `?`:
  `CudaUpload::new(name, &device, format)?` is
  `CudaUpload::new(name, &device, format)`. It never had a failure to
  report — it takes a reference to the device and builds its pads, and the
  surfaces it uploads into are allocated when the first frame says what
  size they are, which is where a CUDA error has always come from. The two
  other uploads are unchanged: `D3d11Upload::new` does not fail either and
  never said it did, and `D3d12Upload::new` does, when it allocates up
  front.

- **A muxer's `add_stream` takes the encoder or stream it describes.**
  `add_stream(name, parameters, time_base)` on `FileMuxer`,
  `SegmentedFileMuxer`, `HlsMuxer`, `RtmpMuxer`, `RtspMuxer` and
  `ReplayBuffer` is now `add_stream(name, format)`, where `format` is
  anything that converts into the new `TrackFormat`: `&encoder` for any of
  the four encoders, or a demuxed `&StreamInfo`. The two values described
  one stream and were passed separately, so one encoder's time base could
  go in beside another's parameters; now they come from the same place.
  `TrackFormat::new(parameters, time_base)` is left for a stream described
  by hand.

- **A Tee starts from the pipeline's context: `ctx.tee(name)`.**
  `TeeBuilder::new(name, ctx.clone())` is now `ctx.tee(name)`, beside
  `ctx.branch()`; the rest of the builder (`branch`, `build`,
  `build_dynamic`) is unchanged. A Tee's branches can only belong to the
  pipeline it is wired in, so there was never another context to hand it,
  and `TeeBuilder::new` is no longer public.

- **`FrameCounter` and `PacketCounter` hand back a `CounterHandle`.** Their
  `new` returned the `Arc<AtomicUsize>` they counted into, so reading it
  took an `Ordering` import and `.load(Ordering::Relaxed)`, and a caller
  could overwrite the count. `count.get()` reads it now; the handle is
  read-only, cheap to clone, and keeps reading the final number after its
  sink is dropped.

- **`media_pp::init` is gone; the library readies FFmpeg itself.** Delete
  the `media_pp::init()?;` line. It registered FFmpeg's error descriptions
  and its input devices, and forgetting it was silent: every FFmpeg error
  displayed as `ffmpeg error: ` with nothing after it, and a camera opened
  through FFmpeg was not found. Every element that reaches FFmpeg now does
  this once, on its way in, so there is nothing to remember.

- **A setter that refuses says why.** `set_frame_rate` on the three
  compositor handles, `MixerHandle::set_mix_format` and
  `rate::FrameRateHandle::set` (the captures' rate handles) returned
  `false` both for a value they could not take and for an element that had
  already stopped, so a caller could not tell which. Each returns a
  `Result`: `InvalidFrameRate` or `Stopped` from the compositor's own
  error, the new `AudioMixerError::InvalidMixFormat` or `Stopped` from the
  mixer, and the new `rate::FrameRateError` (`Invalid` or `Stopped`) from a
  capture. The running value is left alone either way, as before.

- **`PipelineBridgeHandle::connect` returns a `Result`.** It returned
  `None` once the bridge's pipeline had finished; it returns
  `PipelineBridgeError::Disconnected`, the bridge's own word for that, as
  the other handles now answer with an error of their own.

- **The demuxers' per-stream lookups are gone; `best` and `open`'s list
  carry the same.** `FileDemuxer` and `RtspSource` lose `best_stream`,
  `stream_parameters` and `stream_time_base`. `best(kind)` returns the
  `StreamInfo` of the stream to play — its `index`, `parameters` and
  `time_base` together — and `open` already returns one for every stream,
  each at its own index:

  ```rust
  // before
  let index = source.best_stream(media::Type::Video).ok_or(..)?;
  let parameters = source.stream_parameters(index).ok_or(..)?;
  let time_base = source.stream_time_base(index).ok_or(..)?;

  // after
  let video = source.best(media::Type::Video)?;
  // video.index, video.parameters, video.time_base
  ```

- **`TeeHandle::branch` returns a `Result`, like the rest of the handle.**
  It returned `None` once its `Tee` was gone, so every caller wrote its own
  `.ok_or("the Tee is gone")?`, while `attach` and `detach` on the same
  handle already returned `GraphError::ParentNotAttached`. It returns that
  error too now: `tee.branch()?` is the whole call.

- **`framerate` is `frame_rate` everywhere.** `TestVideoOptions::framerate`,
  `MfCaptureFormat::framerate`, `V4l2CaptureFormat::framerate` and the
  `framerate` field of `MfCaptureSourceError::FormatNotOffered` are renamed to
  `frame_rate`, the name the encoder and compositor options already used —
  so a test source's rate is handed to an encoder as `frame_rate:
  options.frame_rate` rather than across two spellings. The screen
  captures' `fps: u32` stays as it is: it is a cap in whole frames, not a
  rate.

- **`Pipeline::new` hands back what its wiring returns.** It returns
  `(Arc<Pipeline>, T)`, where `T` is whatever the `wire` closure returned —
  so something only the wiring can make, a `TeeHandle` from
  `build_dynamic` or a routing to keep, comes back out as a value rather
  than through a variable the closure fills in and the caller then has to
  `expect`. Several at once are a tuple. A closure with nothing to hand
  back returns `Ok(())` as before, and the call binds `()`:

  ```rust
  // before
  let mut tee = None;
  let pipeline = Pipeline::new("fan", source, |source, ctx| {
      let (branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
      ctx.attach(source, 0, branch)?;
      tee = Some(handle);
      Ok(())
  })?;
  let tee = tee.expect("wire ran");

  // after
  let (pipeline, tee) = Pipeline::new("fan", source, |source, ctx| {
      let (branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
      ctx.attach(source, 0, branch)?;
      Ok(handle)
  })?;

  // and where the closure returns nothing
  let (pipeline, ()) = Pipeline::new("play", source, |source, ctx| { /* ... */ Ok(()) })?;
  ```

  `PipelineBuilder::add_source` does the same: it returns `(builder, T)`,
  so a multi-source pipeline is built a statement at a time rather than as
  one chain:

  ```rust
  let builder = PipelineBuilder::new("record");
  let (builder, ()) = builder.add_source(video, |source, ctx| { /* ... */ Ok(()) })?;
  let (builder, routing) = builder.add_source(audio, |source, ctx| { /* ... */ Ok(routing) })?;
  let pipeline = builder.build();
  ```

- **Adding an input to a stopped compositor or mixer is an error, the same
  on every backend.** `D3d11VideoCompositorHandle::add_source`,
  `add_layer` and `add_text_layer` and `SwVideoCompositorHandle::add_source`
  returned `Ok(None)` once their compositor was gone, and
  `MixerHandle::add_source` returned `None`, so every caller wrote its own
  `.ok_or("the compositor is gone")?` — while the CUDA compositor already
  returned an error. Each now returns its input directly, or a new
  `Stopped` variant of its own error (`D3d11VideoCompositorError`,
  `SwVideoCompositorError`, `CudaVideoCompositorError`, `AudioMixerError`):
  drop the `Option` handling, `handle.add_source(name, layer)?` is the whole
  call. The CUDA compositor reports `Stopped` there too, where it used to
  borrow `SourceRemoved`, whose message is about a removed input.

- **`Error` is `#[non_exhaustive]`, and every public error converts into
  it.** A `match` on `media_pp::Error` needs a `_` arm; in return, an
  element added later — and each one adds a variant — no longer breaks
  it. `CudaDeviceError`, `CudaDriverError`, `CudaFrameError`, `RackError`,
  `PlaybackClockError`, `PipeWireDeviceError` and `DmaBufCudaError` gain
  variants of their own, so each passes through `?` where it used to have
  to be turned into a string first:

  ```rust
  // before
  let cuda = CudaDevice::new().map_err(|e| Error::Other(e.to_string()))?;

  // after
  let cuda = CudaDevice::new()?;
  ```

  A string keeps nothing to match on, and the examples taught it: 83 of
  them did it, most to errors that already converted. They no longer do.
  `SubmitError`, which a renderer implementation returns, is now a
  `std::error::Error` with messages of its own.

- **An upload, a download and a converter take the size from the frames.**
  `CudaUpload::new`, `D3d11Upload::new`, `D3d12Upload::new`,
  `CudaDownload::new`, `D3d11Download::new`, `D3d12Download::new` and
  `CudaConverter::new` no longer take `width` and `height`: drop the last
  two arguments. None of these elements changes a frame's size, so the
  size they were told had to agree with whatever the upstream element
  produced, and a caller that got it wrong learned so one frame later
  through a `DimensionMismatch` error. That error variant is gone from all
  seven.

  ```rust
  // before
  let converter = CudaConverter::new("to-nv12", &cuda, CudaFrameFormat::Nv12, width, height)?;

  // after
  let converter = CudaConverter::new("to-nv12", &cuda, CudaFrameFormat::Nv12)?;
  ```

  This is what the element a refused link names now costs to build: the
  remedy in the error message is the whole of it, with no stream
  parameters to open for a size — `hw_decode_render` opened a codec
  context solely for that and no longer does. The texture, staging buffer
  or `AVHWFramesContext` each element needs is made for the first frame's
  size and made again when that changes, so **a source that changes
  resolution mid-stream is now followed rather than refused** — an RTSP
  camera switching profile, a window capture being resized. A failed
  allocation for a new size leaves the element serving the size it already
  had. `CudaConverter`'s even-dimension requirement is now checked per
  frame, so `CudaConverterError::OddDimensions` comes back from `consume`
  rather than from `new`.

  `CudaChromaKey::new` and `CudaVideoEffect::new` lose the same two
  arguments, for the same reason and with the same consequences — keying and
  colour correction are per-pixel, and their D3D11 siblings never took a
  size. `CudaChromaKeyError::DimensionMismatch` and
  `CudaVideoEffectError::DimensionMismatch` are gone with them.

  A scaler, a compositor and an encoder still take a size: it is what they
  are for, not something they have to be told twice.

- **A CUDA element refuses a frame through one error, `CudaFrameError`.**
  `CudaConverter`, `CudaChromaKey`, `CudaVideoEffect`, `CudaDownload`,
  `CudaEncoder`, `CudaRenderer`, `CudaScaler` and `CudaVideoCompositor`
  each carried their own copy of the same four checks — a CUDA frame, with a
  frames context, from this element's device, in a layout it reads — as
  their own four error variants, and their own copy of the unsafe code
  asking them. Those variants (`UnsupportedFormat`, `MissingFramesContext`,
  `ForeignContext`, `UnsupportedSurfaceFormat`, and the renderer's
  `UnsupportedSoftwareFormat`) are replaced in every one of them by
  `Frame(CudaFrameError)`, and the checks by one function.

  ```rust
  // before
  Err(Error::CudaDownloadError(CudaDownloadError::ForeignContext)) => …

  // after
  Err(Error::CudaDownloadError(CudaDownloadError::Frame(
      CudaFrameError::ForeignContext { .. },
  ))) => …
  ```

  The messages still name the element — `CudaDownload reads NV12 surfaces,
  got BGRA` — and a refused layout now says what would have been taken,
  through `CudaSurfaces`, the same value that made the decision.

- **A frame's link contract states its pixel layout.**
  `PortContract::Frames` has a third field, a `PixelLayoutSet` — NV12, P010,
  BGRA or other — and `OutputContract` a new variant, `SameLayout`, for a
  port that passes on the layout it was given. Where a `Frames` value was
  built by hand, add `PixelLayoutSet::ALL`, which is what
  `PortContract::frame` and `any_frame` still start from; where
  `OutputContract` was matched, add an arm for `SameLayout`.

  This crate's elements state their layouts where construction settles
  them, so a branch that could only fail frame by frame is refused when it
  is built: a `VideoDecodeBin` that will put out BGRA wired straight into a
  `CudaRenderer`, which presents NV12, now fails at `Pipeline::new` naming
  both rather than erroring on every frame. A producer that states no
  layout is not checked against one, and a resize-only scaler
  (`D3d11Scaler` with `Preserve`, `CudaScaler::new`) or `D3d11Upload`
  passes on the layout of what reaches it. Every example and obs-rs's
  own pipelines link as before; one test did not — `TestVideoSource`'s
  YUV420P wired into a `D3d11Upload`, which only takes NV12 or BGRA and
  would have refused every frame.

- **`StreamInfo` carries each stream's parameters and time base, and is no
  longer `Copy`.** `FileDemuxer::open` and `RtspSource::open` report
  `parameters` and `time_base` beside `index` and `kind`, so a branch is
  built from what `open` returned without asking again by index. Holding
  FFmpeg's parameters makes it `Clone` only; where a `StreamInfo` was
  copied, clone it or take a reference. `stream_parameters` and
  `stream_time_base` still answer the same.

- **A muxer's sinks are taken by track, not by position.** Every muxer's
  `add_stream` returns a `MuxerTrack`, and `open` returns `MuxerSinks`
  instead of a `Vec<Box<dyn Sink>>`: each track's sink is taken out with the
  track its `add_stream` returned. This applies to `FileMuxer`,
  `SegmentedFileMuxer`, `HlsMuxer`, `RtmpMuxer` and `RtspMuxer` alike.

  ```rust
  // before
  muxer.add_stream("video", video_params, video_time_base)?;
  muxer.add_stream("audio", audio_params, audio_time_base)?;
  let mut sinks = muxer.open()?;
  let audio_sink = sinks.pop().expect("audio was added second");
  let video_sink = sinks.pop().expect("video was added first");

  // after
  let video = muxer.add_stream("video", video_params, video_time_base)?;
  let audio = muxer.add_stream("audio", audio_params, audio_time_base)?;
  let mut sinks = muxer.open()?;
  let video_sink = sinks.take(video)?;
  let audio_sink = sinks.take(audio)?;
  ```

  The order of a `Vec` was the only thing saying which sink was which, so
  every caller repeated the order it had added tracks in, and a track added
  only sometimes — audio when the source has any — made that easy to get
  wrong in a way that still compiled and sent one medium's packets to the
  other's track. `SegmentedFileMuxer::add_stream` still cannot fail, and
  returns the `MuxerTrack` directly.

  `MuxerTrack` is neither `Clone` nor `Copy` and `take` consumes it, so a
  sink cannot be taken twice. A track from a different muxer is refused with
  `MuxerTrackError::ForeignTrack` (through `Error::MuxerTrackError`) rather
  than a panic. `MuxerTrack` is `#[must_use]`: a track whose sink is never
  taken never reports itself finished, and the output is never finalized.

- **`FileMuxerStreamSink`, `HlsMuxerStreamSink`, `RtmpMuxerStreamSink` and
  `RtspMuxerStreamSink` are gone.** Nothing could reach one: every muxer's
  sinks are handed out as `Box<dyn Sink>`, so the names were exports with
  no way to hold a value of them — `redacted_url` on the two publishing
  ones included, which `RtmpMuxer::redacted_url` and
  `RtspMuxer::redacted_url` still answer before `open`. The four were one
  implementation copied four times, and are now one type inside the crate.

- **`CudaConverter` converts either way round, so its constructor is told
  which.** `CudaConverter::new(name, device, width, height)` becomes
  `CudaConverter::new(name, device, output)` — the size going the way of the
  entry above; pass `CudaFrameFormat::Nv12` for what it used to do.

  The new direction exists because a camera hands over NV12 and a chroma key
  takes BGRA, so a green screen had nowhere to be keyed on this backend.
  `CudaScaler` cannot fill the gap — it refuses a YUV/RGB pair either way
  and says so in its own docs. On Windows nothing was needed: `D3d11Scaler`
  already takes an output format, and `D3d11ScalerFormat::Bgra` names the
  chroma key among the things it is for.

  The kernel is the exact inverse of the one beside it, written in the order
  that undoes it, and the test asserts a round trip against that inverse
  computed in Rust rather than against a tolerance.

- **Both chroma keys hand out a `ChromaKeyHandle`, so their constructors
  return a tuple.** Keying is tuned by eye, and rebuilding the element for
  each nudge of a threshold means reopening whatever produces its frames —
  a visible stall for a camera, a portal dialog for a Wayland capture.

  ```rust
  // before
  let key = SwChromaKey::new("key", options);
  let key = D3d11ChromaKey::new("key", &device, context, options)?;

  // after
  let (key, handle) = SwChromaKey::new("key", options);
  let (key, handle) = D3d11ChromaKey::new("key", &device, context, options)?;
  ```

  Discard the handle with `let (key, _) = ...` where the settings never
  change. To retune, read, adjust and write back — one lock, and no window
  where half a change is live:

  ```rust
  let mut options = handle.options();
  options.threshold = 0.25;
  handle.set_options(options);
  ```

  `ChromaKeyHandle::set_enabled` turns keying off without disturbing the
  settings, and a disabled element hands each frame straight through — the
  same picture, not a copy. It is kept out of `ChromaKeyOptions` on purpose:
  that struct says how to key and this says whether to, the split
  `AudioVolume` has between its gain and its mute. It also means no
  construction site can produce a chroma key that silently does nothing by
  forgetting a field.

  `ChromaKeyOptions` now derives `PartialEq`, which is how each element
  notices a change and retires the frame its repeat cache was holding. That
  cache is why this is not purely additive: a picture keyed green is not an
  answer to the same picture once the key turned blue, and a still capture
  re-emitting one texture forever would otherwise stay frozen at the old
  settings with nothing to dislodge it.

- **`RtspSink` is now `RtspMuxer`, and carries more than one track.** It
  could only ever publish a single stream, so video and audio could not share
  one RTSP session — which is what the rename is about: every other muxer in
  this crate registers its tracks and then hands out one `Sink` each, because
  a header has to describe them all before the first packet, and this one now
  does the same. A type that returns sinks rather than being one is a
  `*Muxer` here.

  Rename `RtspSink` -> `RtspMuxer`, `RtspSinkError` -> `RtspMuxerError`,
  `Error::RtspSinkError` -> `Error::RtspMuxerError`, and
  `ElementType::RtspSink` -> `ElementType::RtspMuxer`. Deliberately without a
  compatibility alias.

  The call changes shape with the name:

  ```rust
  // before
  let sink = RtspSink::open("rtsp", url, RtspTransport::Tcp, params, time_base)?;

  // after
  let mut muxer = RtspMuxer::create(url, RtspTransport::Tcp)?;
  let video = muxer.add_stream("video", params, time_base)?; // and "audio", if there is one
  let mut sinks = muxer.open()?;                             // the RTSP handshake happens here
  let sink = sinks.take(video)?;
  ```

  The element name moves from `open` to each `add_stream`, so the tracks are
  told apart in logs and `BusEvent`s. `create` no longer touches the network:
  RTSP announces its streams in the header, so the
  `ANNOUNCE`/`SETUP`/`RECORD` handshake is `open`'s to perform and to fail
  at.

  `RtspSink::url` is gone; `RtspMuxer::redacted_url` replaces it. An RTSP URL
  can carry a password in its authority, and the old type logged the URL it
  was given.

  Unchanged: it still remuxes rather than encodes, still keeps a published
  timeline monotonic across an upstream seek (now per track), and still
  finalizes on `Eos` alone rather than on `Stop`.

- **`CudaDecoderError::UnsupportedCodec`, and the same on `D3d11DecoderError`
  and `D3d12DecoderError`.** Each hardware decoder's `new` refuses a codec no
  decoder can decode on its device with it (see Fixed), so a `match` over
  any of the three needs an arm for it.

- **`Pacer` reads the time base off each buffer instead of being told it.**
  `Pacer::new(name, time_base)?` is `Pacer::new(name)`, and
  `Pacer::with_discontinuity_limit(name, time_base, limit)?` is
  `Pacer::with_discontinuity_limit(name, limit)`; neither can fail any more,
  so drop the `?`. Told separately, a pacer could be told another stream's
  unit — the video's handed to the audio branch — and play at the wrong
  speed with nothing to say so; a packet carries its own, and now every
  frame this crate makes does too (see Added). `PacerError::InvalidTimeBase`
  is gone, and `PacerError::NoTimeBase` refuses a timed frame that does not
  say what unit its `pts` is in — one pushed through an `AppSource` or made
  by an element of your own — rather than guessing. Stamp such a frame with
  `buffer::set_time_base`.

- **`AudioResampler`, `FrameRateLimiter` and `VideoSynchronizer` read it
  off the frame too.** `AudioResampler::new(name, target, input_time_base)?`
  is `AudioResampler::new(name, target)`,
  `VideoSynchronizer::new(name, time_base)?` is `VideoSynchronizer::new(name)`
  — neither can fail any more — and
  `FrameRateLimiter::new(name, input_time_base, rate)` is
  `FrameRateLimiter::new(name, rate)`. Each was the same value that had to
  agree with the stream. `AudioResamplerError::InvalidTimeBase` and
  `VideoSynchronizerError::InvalidTimeBase` give way to a `NoTimeBase`
  variant on each, and the new `FrameRateLimiterError` has the one variant
  `NoTimeBase` — all refuse a timed frame that does not say its unit, as
  `Pacer` does. No element takes a stream's time base any more except a
  muxer's `add_stream`, which writes it into a header before the first
  packet arrives.

- **The encoders count in a unit of their own and read the frames'.**
  `time_base` is gone from `SwEncoderOptions`, `SwAudioEncoderOptions`,
  `CudaEncoderOptions` and `D3d11VideoEncoderOptions`: drop the field. It
  had to match the unit of the frames' timestamps, and a wrong one
  encoded a file that played at the wrong speed. A video encoder now
  counts in 1/90000 (1/60000 for `VideoCodec::Mpeg4`, whose bitstream
  holds no more) and converts each frame's `pts` from the unit the frame
  carries; the audio encoder counts samples at its output rate, as it
  already did whatever its option said. Each says its unit through
  `time_base()` — new on the three video encoders — which is what to
  hand a muxer's `add_stream`:

  ```rust
  // before
  let encoder = SwEncoder::new("encode", SwEncoderOptions { time_base, .. })?;
  let track = muxer.add_stream("video", encoder.parameters(), time_base)?;

  // after
  let encoder = SwEncoder::new("encode", SwEncoderOptions { .. })?;
  let track = muxer.add_stream("video", encoder.parameters(), encoder.time_base())?;
  ```

  A timed frame with no unit is refused with a new `NoTimeBase` variant
  on `SwEncoderError`, `CudaEncoderError` and `D3d11VideoEncoderError`.
  And a muxer track now reads each packet in the unit the packet carries,
  falling back on the one `add_stream` was given only for a packet that
  carries none — so a track registered with the old frame unit still
  writes its packets at the right times.

### Added

- **`AudioTempo` stretches sound to the playback rate, for what takes it
  at the wall clock's pace.** Played at twice the rate, a `Pacer` hands on
  two seconds of sound each second; an `AudioMixer` input or an encoder
  after it takes one. `AudioTempo`, after the pacer, makes it one again
  without changing its pitch — the same FFmpeg `atempo` stretch the audio
  renderers use — reading the rate from the pipeline's playback clock, so
  `Pipeline::set_rate` reaches it with nothing to set. At the file's own
  speed a frame goes on as it came. A stretched frame's `pts` is where its
  first sample is in the media, in its own sample rate; the format goes on
  as it came, planar or packed. New: `Error::AudioTempoError` and
  `ElementType::AudioTempo`.

- **`SwScaler::if_needed`: a software decode fitted to what takes it.**
  Given a stream's parameters and the sink it is to reach, it answers
  nothing where the sink takes what the stream decodes to, and otherwise a
  scaler to the first of YUV420P, NV12 and BGRA the sink does take, at the
  frames' own size — asked of the sink's input contract, so what a sink
  takes is said in one place. `Player` and the examples each had a copy of
  this that always chose YUV420P at the stream's opening size; they use
  this one now, which also follows a stream that changes size.

- **A video decoder decodes less while its pictures come too late.** A
  `Pacer` or `VideoSynchronizer` says how late each picture it hands on
  is, and every video decoder here reads that before each packet: while
  pictures keep coming late it stops decoding pictures nothing refers
  to — the B-frames, commonly — then all but keyframes, and it goes back a
  step once they have come on time for a while. What is shown stays at
  the right place in time. A preroll decodes everything, since a seek or
  a step asks for one particular picture. Playing backwards is where it
  counts where a group of pictures is longer than a stretch, since each
  stretch in it is decoded from the group's start: forced to one-second
  stretches on a 1080p recording with a keyframe every five seconds, a
  software decoder fell to 2.3 times the speed at -4, and with this held
  4, showing fewer pictures. One that keeps up is not touched.

- **A negative `Pipeline::set_rate` plays backwards.** From the picture
  shown, from a quarter of the speed to four times it
  (`Pipeline::REVERSE_RATE` is the media's own): turning round, everything
  is sought there and prerolled, as a seek is, and the same turning
  forwards again; faster or slower the same way round is only a change of
  speed. Backwards each stretch is decoded from the keyframe before it, so
  a fast rate asks more of the decoder than playing as fast forwards; what
  it cannot decode in time a pacer shows late. The sound is not played — nothing reaches it, and the
  playback clock goes on the wall, so a `Pacer` or `VideoSynchronizer` paces
  the picture as it does forwards. At the start of the file the stream ends
  and the pipeline says `Finished`. A seek or a step while backwards stays
  backwards; a step on is the picture before.

  Two new traits say what playing backwards asks of an element, and
  `Pipeline::check_reverse` answers, as the pipeline is wired, whether its
  elements have them. A source implements `ReversibleSource` —
  `seek_backwards`, and `finish_stretch` for a `finish` that comes in the
  middle of a stretch — and hands the picture on in *stretches*: a span of
  the media read from the keyframe before it, what is outside the span
  marked `AV_PKT_FLAG_DISCARD`, each span the one before the last, down to
  the start — `ReversibleSource` extends `SeekableSource`, since a
  turn is a seek. An element that turns a picture's packets into pictures
  implements `ReversibleDecoder` — `begin_stretch` and `end_stretch`, between
  which it holds its pictures and at the end hands them on last first; the
  pipeline tells it where each stretch begins and ends. Each says it is one
  from `as_reversible`, on `SourceElement` and on `Sink`, which is `None` by
  default: a source that is not refuses to play backwards, and so does an
  element whose contracts take a picture's packets and hand on pictures.
  `FileDemuxer` is a `ReversibleSource` for a file with a picture, and
  `SwDecoder`, `D3d11Decoder`, `D3d12Decoder` and `CudaDecoder` are
  `ReversibleDecoder`s, and so `VideoDecodeBin` on every path.
  `FileDemuxer` begins a stretch at a keyframe, so that a group of
  pictures is decoded once, as long as the pictures a decoder holds for
  it fit in 512 MB — a whole five-second group at 1080p and 30 a second,
  two thirds of a second at 4K and 60 — and a longer group is read in
  stretches that long from its end, decoding its start again for each.
  On a 1080p recording with a keyframe every five seconds every decode
  path, software included, plays it at -4 without leaving anything out.
  A hardware decoder holds copies, since its surfaces are a pool the rest of the stretch is decoded
  into, made on the device from a frames context of its own. In
  `Player`'s `respond_to` R turns round at the same speed, and the minus
  and plus keys change the speed either way round.

- **`Pipeline::set_rate` plays faster or slower, the sound at its own
  pitch.** From a quarter of the speed to four times it
  (`Pipeline::MIN_RATE`..=`MAX_RATE`), taken from where playback is with
  nothing sought or flushed: the playback clock covers that much more or
  less media each second, so a `Pacer` or `VideoSynchronizer` paces the
  picture to it unchanged. `WasapiRenderer` and `PipeWireAudioRenderer`
  stretch their sound to the rate with FFmpeg's `atempo` and say where
  playback is from what the device has played, each sample standing for
  `rate` samples of the media, so the picture stays with the sound. A seek,
  a step and a pause keep the rate; it is refused where a seek is, and
  outside the range with `PipelineError::UnsupportedRate`. `Player` has
  `set_rate` and `rate`, and in `respond_to` the minus and plus keys play a
  quarter slower and faster and Backspace at the file's own speed. `Key`
  gains `Backspace`, and `Char('+')` for the plus key (`=` unshifted on a
  US keyboard).

- **`Pipeline::step` moves the picture by frames, forward and back.**
  `step(n)` pauses and shows the picture `n` on, or `-n` back, and answers
  where it now is. Forward asks each terminal showing pictures for `n`
  more from wherever its decoder is — nothing sought, nothing decoded
  twice; back is an accurate seek to just before the picture shown,
  counted in the spacing of the pictures, so it lands on the one before
  however they are spaced. The sound is not played while stepping, and
  `resume` after a step first seeks to the picture shown, so the sound
  goes on in line with it. Refused where a seek is, and with
  `PipelineError::NoPicture` before anything has shown a picture. A
  `PrerollContext` now says how many samples it asks each terminal for,
  `samples`: one for a seek's, the frames asked for in a step's.
  `Player::step` does the same for a player, and the full stop and the
  comma step a picture on and back in `respond_to`.

- **`Pipeline::check_seek` says whether a seek would be refused, and by
  what, without seeking.** Answered from the graph as it stands, not by
  asking anything running: it works before `run` and after the sources
  have stopped, costs a lock, and follows branches as they come and go — a
  recording attached to a `Tee` refuses from its attach to its detach.
  What a player needs to decide whether to offer a seek bar. `seek` asks
  it first, as it asked the `CheckSeek` cascade before.

- **`SourceElement::pausing` and `SourceElement::resuming`**, called as a
  pause begins — before the `Pause` goes downstream — and as it ends —
  after the `Resume` has, before it is acknowledged. What a source does to
  stop and restart its own input goes there, and `drain_control` does the
  rest; the WASAPI, Media Foundation and PipeWire audio captures each kept
  a copy of the pause wait to do it, and now use these. Both default to
  doing nothing, so no source has to change.

- **`DxgiCaptureSource::outputs`** lists what `CaptureArea::Output` can
  name, by the index that names it, with each monitor's name, place on the
  desktop, and which one is primary — output 0 need not be.
  **`MfCaptureSource::frame_rate`** says the rate a camera was opened at,
  which an encoder after it needs and `open` with no format chosen did not
  say.

- **Re-encoding a file needs nothing outside the crate.** `StreamInfo`
  says a video stream's `frame_rate` and, through `size` and `color`, its
  picture size and colour description — what an encoder re-encoding it is
  opened with, which took reaching into FFmpeg before.
  `EncodeInput::for_decoded` turns a decode bin's target and output format
  into the input an encode bin takes, and the new `transcode` example puts
  them together: `FileDemuxer -> VideoDecodeBin -> VideoEncodeBin ->
  FileMuxer`, decoded onto and encoded from the GPU, the sound copied as it
  is. The README shows a file pipeline watched to `Finished` and stopped.

- **`Player` plays sound on its own, and has a volume, a loop and a choice
  of track.** A file with no picture was refused as one with nothing to
  play; it now plays, with its waveform in the window — drawn by the new
  `AudioWaveform` from the same decoded sound and shown by a
  `VideoSynchronizer` as it is heard — and the keys arrive there as ever.
  `AudioWaveform` is an element of its own: `f32` sound in, a BGRA
  oscilloscope trace out at a steady rate, each picture stamped with the
  moment of the sound it ends at. `set_volume`, `set_muted` and the up and down arrows and M in
  `respond_to` turn the sound, through an `AudioVolume` in its branch;
  `set_looping` plays the file again from the start instead of ending, with
  `position` read within the lap; and `PlayerOptions::audio_stream` picks
  one of several sound tracks by its stream index. `decoding` is `None` for
  a file with no picture.

- **`VideoEncodeBin`: H.264 by whichever encoder opens.** Encoding took
  choosing between `SwEncoder`, `D3d11VideoEncoder` and `CudaEncoder` by
  hand, with the download, conversion and colour description each one
  needs in front — obs-rs had grown its own probe for it. The bin takes
  frames from system memory, D3D11 or CUDA (`EncodeInput`) and opens
  `h264_nvenc`, then `h264_mf` for D3D11, then software (`libx264`, or
  `libopenh264`), keeping the first that opens; `path` says which, and a
  muxer's `add_stream` takes the bin itself. Every path says in the stream
  what its YUV is — from RGB, what it converted with; from YUV,
  `VideoEncodeOptions::color`. Chosen once, when it opens: a muxer writes
  its headers from that encoder. `D3d11Download` reads NV12 textures as
  well as BGRA, into NV12 frames with their colour description, so the
  software path from a decoder's or an upload's textures needs no video
  processor — which a device with none, such as a CI runner's, lacks.

- **A picture is shown when its sound is heard, not a screen's delay
  later.** `VideoSynchronizer` handed each picture over when the audio
  position reached it, and the audio position is what the listener hears,
  device latency and all; the renderer then took its draw, the wait for the
  display's next refresh and a compositor's repaint to show it — 22 to 38 ms
  behind the sound, measured with a flash and a beep. A video renderer now
  tells the pipeline's playback clock what showing a picture takes, and the
  synchronizer hands each over that much early; the flash now reaches
  XWayland 10 ms before the beep is heard, the compositor's repaint landing
  it within a few milliseconds of it, with no picture dropped for it.
  `VulkanWindowRenderer` measures it on Wayland, waiting now and then for a
  present to be shown (`VK_KHR_present_wait`, turned on by `VulkanGpu`
  where the device has it), and on X11, where that wait returns once
  XWayland has the picture, estimates two refreshes of the monitor the
  window is on, read from RandR. `D3d11WindowRenderer` and
  `D3d12WindowRenderer` measure it from the swap chain's frame statistics,
  which say when each present was shown without waiting for it — 27 ms on
  a 60 Hz desktop here — and until they have, estimate two refreshes of the
  desktop's compositor. Each logs what it found.

- **`Player` decodes on the GPU.** It decoded every file in software, which
  is what a 4K file cannot afford. Its picture now goes through a
  `VideoDecodeBin` onto the window's own GPU — D3D11VA on Windows, NVDEC on
  Linux with `cuda` and an NVIDIA GPU — and stays there to be drawn, in
  software and uploaded where the GPU does not take the stream, and in
  software into system memory where there is no GPU to decode onto.
  `Player::decoding` says which. Nothing to change in a program using it.
  Where it decodes in software, it does so on every thread, several
  pictures at once: on the one thread FFmpeg opens a decoder with, 4K60
  HEVC fell behind its rate, and with the picture behind, the demuxer could
  read no further and the sound broke up — a hundred gaps in twelve seconds,
  none now.
- **The software compositor draws text.** A text overlay was on the D3D11
  and CUDA compositors only, so a program that followed the README to
  `VideoWindow` and the software path had none. `SwVideoCompositorHandle`
  has `add_text_layer`, taking the same `TextLayer`, and its
  `SwTextLayerHandle` sets, moves, fades, restacks and hides the text as
  the others' do; the text is rasterized on the calling thread into a
  straight-alpha BGRA picture the compositor blends like any layer.
  `ab_glyph`, which rasterizes, is now always a dependency rather than one
  the `cuda` and `d3d11` features bring.

- **`D3d11Upload` and `CudaUpload` take YUV420P.** A software decode
  needed a `SwScaler` to NV12 in front of either. Both now take YUV420P (and
  YUVJ420P, tagged full range) and put it up as NV12, interleaving its
  chroma planes on the way — the same samples, so no scaler is needed.
  `CudaUpload` does this when built for `CudaFrameFormat::Nv12`, and adds
  `CudaUploadError::PlaneTooSmall` for a plane too short for its rows. The
  link check follows: what takes NV12 on the GPU links after an upload of
  YUV420P, and the suggestion it makes for a missing upload says so. A
  compositor input still takes frames on the compositor's device only —
  the upload in front is where the copy shows, in the topology and in the
  stats.

- **`VulkanGpu` and `VulkanWindowRenderer`: video in a window on Linux,
  from system memory or CUDA.** A new `vulkan` feature, and the Linux
  counterpart of `D3d11WindowRenderer`. `open` opens a window of its own on
  a thread of its own and returns the same `WindowOptions` and
  `WindowEvents` as on Windows — keys, resizing, closing, which hides the
  window — as a plain X11 window through libxcb, not `winit`, so it does
  not collide with an application's event loop; on a Wayland desktop it is
  an XWayland window. `for_window` draws into a window the application
  gives it instead, X11 or Wayland, anything with a `raw-window-handle`,
  kept alive by the renderer. It takes NV12, YUV420P and BGRA frames in
  system memory, each drawn by a shader of its own with no conversion in
  front, and NV12 and BGRA CUDA frames too where its `VulkanGpu` was made
  with `for_cuda`, which pairs it with the GPU CUDA decodes on by UUID. A
  CUDA frame is copied device to device into memory the renderer's device
  allocated and CUDA imported — what `render_common`'s `CudaWindowRenderer`
  did for the examples, in the library. Each YUV frame is drawn by its own
  colour description — BT.709, BT.601 or BT.2020, limited or full range,
  and by height where it says nothing — and every frame letterboxed, and
  presented in step with the display. On X11 it follows the window's size
  by itself; on a Wayland window, where a client sets its own size, the
  application passes it on through `WindowSize`. Named for what draws
  rather than for the frames it takes, like a GStreamer sink, because it
  takes more than one memory domain. `cuda_decode_render` shows `open`,
  with NVDEC and no `winit`; `vulkan_window_render` shows `for_window`.

- **`D3d11Gpu` and `D3d11WindowRenderer`: D3D11 with a window of its own.**
  `D3d11Gpu::new()` makes the one device a pipeline's D3D11 elements share,
  with the flags they need, protected for use across threads, and its
  immediate context behind the one `Arc<Mutex<_>>` they all lock — what a
  program had to write for itself with the `windows` crate; `from_device`
  shares a device made elsewhere. `D3d11WindowRenderer` is a renderer that
  brings its window: `open` opens one on a thread of its own, the way a
  GStreamer video sink does, and returns `WindowEvents` — keys, resizing,
  closing — for the application to act on; `for_window` draws into a window
  the application owns, anything with a `raw-window-handle`, such as a
  `winit` window, kept alive by the renderer. Either way it follows the
  window's size on its own and keeps the picture's aspect ratio. It is a
  plain Win32 window, not a `winit` one, so it does not collide with an
  application's event loop. `d3d11_decode_render` uses both and no longer
  needs `render_common` or `winit`.

- **`Player`: a file played in a window with its sound.** The `playbin` of
  this crate: `Player::open(path, options)` builds `FileDemuxer`, a software
  decode of each stream, a `VideoSynchronizer` in front of a `VideoWindow`,
  and an `AudioResampler` in front of the default output
  (`WasapiRenderer`, `PipeWireAudioRenderer`), the picture following the
  samples played. `play`, `pause`, `seek`, `position`, `duration` and
  `window_control` drive it, `next_event` reports the window, the end and
  an element's failure as `PlayerEvent`s — `next_event_timeout` returns
  `None` only when its time is up, and `Stopped` once playback is over — and `respond_to` does what a
  player usually does with a window event — Space, the arrows, F or a double
  click, and `false` for Escape or a close. After a seek, `position` says
  where it went until playback moves on from there. `FileDemuxer::duration`
  is new with it. The `player` example is a whole player in one screen of
  code, for both platforms.

- **`VideoWindow`: a video window on either platform, no `#[cfg]`.** The
  `autovideosink` of this crate: `VideoWindow::open(name, options)` opens
  whichever window renderer the platform has — `D3d11WindowRenderer` on
  Windows (`D3d12WindowRenderer` with only `d3d12`), `VulkanWindowRenderer`
  on Linux — on a GPU of its own, and returns it with its `WindowEvents`;
  `window_control()` changes the window. It takes what they all take,
  system-memory NV12, YUV420P and BGRA, so a software decode goes straight
  in. GPU frames stay with the platform renderers, whose GPU the rest of a
  pipeline can share. `sw_decode_render` is one program for both platforms
  through it.

- **A window renderer's own window can be changed, and reports clicks.**
  `D3d11WindowRenderer`, `D3d12WindowRenderer` and `VulkanWindowRenderer`
  have `window_control()`, taken before the renderer goes into a pipeline:
  a `WindowControl` that sets the window's title and fills the screen with
  it or puts it back — the saved style and placement on Windows, EWMH's
  `_NET_WM_STATE_FULLSCREEN` on X11 — and says `WindowGone` once the window
  is. It is `None` for a window the application gave the renderer, which is
  the application's to change. `WindowEvent` gains `MouseDown` and
  `DoubleClick`, each with its `MouseButton` and where in the picture area
  it was; a double click is the desktop's own on Windows, and two presses
  within half a second on X11, which has none. `d3d11_decode_render` shows
  where playback is in its title, and F or a double click fills the screen.

- **The D3D window renderers take frames in system memory.**
  `D3d11WindowRenderer` and `D3d12WindowRenderer` draw NV12, YUV420P (and
  YUVJ420P) and BGRA from system memory as well as their backend's own
  textures, uploading each frame themselves into textures made for its
  layout and size — as `VulkanWindowRenderer` already did on Linux. A
  software decode, a CPU capture or an application's own frames go straight
  into one; on Windows they needed a `SwScaler` to NV12 and a
  `D3d11Upload`/`D3d12Upload` in front, which was also where the two
  platforms' graphs parted. The system-memory path needs no video support
  of the device, so a software adapter draws it too.

- **`D3d12Gpu` and `D3d12WindowRenderer`: the same for D3D12.** `D3d12Gpu`
  makes the one device a pipeline's D3D12 elements share, and the one queue
  its windows present through; `from_device` shares a device made
  elsewhere. `D3d12WindowRenderer` has `open` and `for_window` as the D3D11
  one does, with the same `WindowOptions` and `WindowEvents`, and draws the
  NV12 frames `D3d12Decoder` and `D3d12Upload` make zero-copy, waiting on
  the GPU for each frame's own fence. `d3d12_upload` uses both and no
  longer needs `render_common` or `winit`.

- **`Pipeline::position`.** Where playback is — the media time the
  playback master has reached, from the audio renderer's played samples or
  the wall clock a `Pacer` or `VideoSynchronizer` keeps — for a progress
  bar. It holds still while paused, moves on from where a seek put it, and
  is `None` when nothing paces the pipeline. A player used to rebuild it
  from the timestamps of frames going past.

- **`D3d11DecoderError::SurfacePoolExhausted`.** Frames held downstream past
  what the fixed D3D11VA surface pool can spare failed decoding with
  FFmpeg's `Invalid data found when processing input`, the pool's own
  complaint reaching only FFmpeg's log. That failure now says how many
  decoded frames were held and how big the pool is, and what to change.
  `D3d11Decoder::new` and `DecodeTarget::D3d11` document that FFmpeg caps
  the pool at 64 surfaces in all, references included.

- **`log::init` records FFmpeg's own messages.** An encoder's closing
  statistics and a codec library's warnings (`[aac @ 0x…] Qavg: …`,
  libopenh264's) went to stderr, with no time, no thread, and nothing
  tying them to the pipeline. While the logger runs they are recorded in
  its file instead, as element `FFmpeg` named after the codec or format
  that said them, at the matching level. FFmpeg's own threshold (`INFO`
  unless changed) still decides what it reports. This sets FFmpeg's
  process-wide log callback, and dropping the `LogGuard` restores FFmpeg's
  own; an application with its own callback should install it after
  `init`. A program that never calls `init` is unaffected.

- **A pipeline can start paused, and so take one frame from anywhere.**
  `pause` before `run` was ignored, and the run started playing. It now
  makes the run start paused: every source stops before producing
  anything, and `run` returns once they all have. A `seek` from there puts
  exactly one sample through every terminal — with `SeekMode::Accurate`,
  the one covering the requested position — and returns once it has
  arrived:

  ```rust
  pipeline.pause();
  pipeline.run()?;
  pipeline.seek(Duration::from_secs(600), SeekMode::Accurate)?; // the sink has the frame at 10:00
  ```

  which is a thumbnail from part way into a file without decoding
  everything before it. `resume` before `run` undoes it.

- **`BusReceiver::recv_timeout`.** Waits a bounded time for the next event,
  for a loop with something else to watch too, and says whether it came
  back empty because nothing was posted in time or because the bus has
  ended — which `try_recv` cannot tell apart.

- **`WebRtcStreamInfo::track_format`.** A received WebRTC track records
  with `muxer.add_stream(name, info.track_format()?)`, instead of
  assembling `TrackFormat::new(info.codec_parameters()?, info.time_base()?)`
  by hand.

- **`MediaBuffer::video` wraps a hand-made frame.** A `Video` buffer carries
  a pooled frame, so a frame made by hand — a still picture, a caption, a
  test pattern — had to go through a pool of zero first, four lines every
  caller copied along with a comment explaining the trick.
  `MediaBuffer::video(frame)` does it. Nothing recycles such a frame; what
  makes frames over and over keeps its own `UnboundObjectPool` as before.

- **A `BusEvent` prints as one line.** `BusEvent` implements `Display`
  as `[name] eos`, `[name] error: ...`, `[name] dropped a buffer (queue
  full)`, `[name] seeked: requested ..., landed ...` and `finished` — the
  lines `BusReceiver::log_events` always printed — so a loop that only
  reports its bus is `println!("{event}")` rather than a `match` with an
  arm per variant. The examples' 34 such `match`es are one line each now.

- **`SubmitError` converts into `Error`.** The one public error left that
  had no conversion, so a renderer that could not be created had to be
  unwrapped or turned into a string. With it, the examples' constructors
  pass their errors through `?` rather than `.expect()` — 86 of the 146
  `.expect()`s are gone; the rest are on an `Option`, a lock, a joined
  thread or another library's error, where `?` has nothing to convert.

- **`BusEvent::Finished`: the pipeline says when it has ended.** Every
  element that completes end-of-stream posts an `Eos` — a `Queue` as well
  as the muxer after it — so the first `Eos` on the bus is only the first
  thing to end, and a pipeline stopped on it cuts off whatever was still
  draining: the second track of a file, the other side of a `Tee`.
  `Finished` is posted once every terminal sink has accepted `Eos`, after
  the last one's own, and again only after a seek sends a new stream
  through. A branch a `Tee` detaches is no longer waited for; one it
  finishes is counted when its trailer is written. A terminal that fails
  its `Eos` does not count as ended — that failure is the `Error` to act
  on. Stop on it instead of counting `Eos` events:

  ```rust
  // before
  if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
      pipeline.stop();
  }

  // after
  if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
      pipeline.stop();
  }
  ```

  `BusEvent` is `#[non_exhaustive]`, so a `match` with a `_` arm keeps
  compiling. The examples that stopped on the first `Eos`, or counted them,
  now stop on `Finished`.

- **A decoded frame says what unit its timestamps are in.**
  `buffer::time_base` reads it and `buffer::set_time_base` sets it. FFmpeg's
  decoders leave `AVFrame::time_base` unset, so every element in this crate
  that makes a frame — decoders, captures, synthetic sources, mixers,
  compositors, the resampler and the rate limiter — sets it, and every one
  that transforms a frame carries it over with the `pts`.

- **`FileDemuxer::best` and `RtspSource::best` find the stream to play in
  one `?`.** Each returns the `StreamInfo` of FFmpeg's own choice for a
  kind — the same one `best_stream` names by index — or `NoStream(kind)`
  where there is none, so the three lines every example opened with are
  one:

  ```rust
  // before
  let video = source
      .best_stream(media::Type::Video)
      .and_then(|index| streams.get(index))
      .ok_or_else(|| Error::Other("no video stream in file".into()))?;

  // after
  let video = source.best(media::Type::Video)?;
  ```

  A stream that may be absent is `source.best(media::Type::Audio).ok()`.
  `best_stream` is unchanged.

- **Every scaler can be asked for a layout instead of a size:
  `SwScaler::to_format`, `D3d11Scaler::to_format`,
  `CudaScaler::to_format`.** The layout is what a refused link asks for —
  "a SwScaler to NV12 first" in front of an upload that takes NV12, "a
  D3d11Scaler with D3d11ScalerFormat::Bgra first" in front of a download —
  and the size the frames happen to be is no part of that answer, so it is
  no longer asked for. The remedies name these constructors.

  ```rust
  // before: a size fetched from somewhere just to say "the same"
  SwScaler::new("to-nv12", Pixel::NV12, width, height, Flags::BILINEAR)
  D3d11Scaler::new("to-bgra", &device, ctx, D3d11ScalerFormat::Bgra, width, height)?
  CudaScaler::with_format("to-nv12", &cuda, width, height, interp, CudaFrameFormat::Nv12)

  // after
  SwScaler::to_format("to-nv12", Pixel::NV12, Flags::BILINEAR)
  D3d11Scaler::to_format("to-bgra", &device, ctx, D3d11ScalerFormat::Bgra)?
  CudaScaler::to_format("to-nv12", &cuda, interp, CudaFrameFormat::Nv12)
  ```

  It also changes what a mid-stream resolution change does. A scaler given
  a size absorbs one by scaling the picture back to it, which is what keeps
  a fixed-geometry encoder downstream working and a broken aspect ratio
  anywhere else; `to_format` passes the new size on, so what is downstream
  has to be able to take one. The size-taking constructors are unchanged
  and still the right ones in front of an encoder, a muxer's stream or a
  model.

- **A refused link says what goes between, and elements can be asked
  directly.** A pipeline's refusal and `LinkCheck` both end with the
  element that makes the crossing where one does — `; convert it: a
  CudaConverter built for CudaFrameFormat::Nv12`, `; upload it: a
  D3d11Upload, which takes NV12 or BGRA — a SwScaler::to_format to one of
  those first…`
  — through `contract::remedy`. `contract::check_elements(&mut producer,
  &consumer)` asks about two elements with no pad index or trait import. The
  crate documentation's first page has a section on connecting elements:
  what is caught before a pipeline runs and what only a frame shows, asking
  before linking, and a table of what goes between which backends and
  layouts.

- **`contract::check_link` answers whether two elements fit before they are
  linked.** It takes a pad's `SrcPad::contract()` and a sink's
  `Sink::input_contract()` and returns a `LinkCheck` — `Fits`,
  `Refused { produced, accepted }`, or `Unknown` where one side says too
  little — by the same rules a pipeline refuses a link with, which now asks
  it too. Trying the link to find out cost the elements: a refused branch
  drops what it was given. `hw_decode_render` asks it to decide whether a
  `CudaConverter` goes between the decode bin and the renderer, where it
  used to compare `output_format()` against what it knew the renderer took.

- **HDR video is brought to SDR on the GPU.** A PQ or HLG stream came out
  washed out: nothing read its transfer. `VideoDecodeBin` now brings one to
  SDR BT.709 BGRA as soon as it is decoded, on `D3d11` and `Cuda` — by the
  new `D3d11ToneMap`, a pixel shader over P010 or NV12, and by
  `CudaConverter`, which now takes P010 for this. Both evaluate one
  definition: BT.2020's matrix, ST 2084's EOTF or HLG's with a 1000-nit
  OOTF, BT.2020 primaries into BT.709's, BT.2390's EETF on the largest
  channel from 1000 nits down to 203, and gamma 2.2. A D3D11 video processor
  would have been the place, and the RTX 3050's offers no conversion from
  PQ or HLG to SDR. A 4K PQ stream decoded this way ran at 463 frames a
  second on D3D11 and 289 on CUDA. The stream's own MaxCLL and mastering
  metadata are not read, and the software path does not tone map.

- **`FileDemuxer::best_stream` and `RtspSource::best_stream`**: the stream of a
  kind FFmpeg judges the one to play (`av_find_best_stream`). The first
  video stream is not always it — cover art or a thumbnail can come first as
  a still picture — and this passes over a stream marked as an attached
  picture and prefers one with more than a single frame.

- **`DecodeThreading`: how many threads a software video decoder has, and
  how they share the work.** `SwDecoder::with_threading` takes one, and
  `VideoDecodeBin::open` an `Option` of one for its software path;
  `SwDecoder::new`, and `None`, set nothing and decode on one thread, as
  before. It holds `threads` (`None` for as many as the machine has) and a
  `DecodeThreadKind`. `Frame` decodes several pictures at once — at 1080p on
  twelve threads, H.264 about five times one thread and HEVC about four —
  and hands each out a picture per thread later. `Slice` works within one
  picture, holds nothing back, and is for a live source: ProRes and VP9,
  which split a picture into slices or tiles, still gain about six and two
  times — and a codec whose every picture stands alone gains nothing from
  `Frame`: at 1080p ProRes decoded no faster and took 335 MB against 93.
  Which suits a stream depends on its codec and size, so the caller says;
  nothing chooses for it.

- **`VideoDecodeBin`: a video stream decoded onto a GPU by whichever path
  can take it.** `VideoDecodeBin::open(name, params, target, threading)` takes a
  stream and a `DecodeTarget` — `D3d11`, `D3d12` or `Cuda`, owning its device, or
  `System` — and is one element holding the line for it: the target's
  own hardware decoder where it has one for the codec and the pictures are
  4:2:0, and otherwise `SwDecoder` → `SwScaler` → the target's upload,
  so the stream reaches the same device either way. A 10-bit stream is
  decoded by the hardware to P010 and brought down to NV12 on the GPU after
  it — by a `D3d11Scaler` on `D3d11`, whose target carries the pipeline's
  shared immediate context for it, and a `CudaScaler` on `Cuda`; `D3d12`
  decodes it in software. At 4K, HEVC 10-bit decoded this way on CUDA ran
  at 277 frames a second with next to no CPU, against 104 on four
  software threads. A BT.2020 stream is put out as BT.709 BGRA instead —
  see the fix below. Nothing maps HDR to SDR. Pictures with alpha
  take the software path and arrive as BGRA with their alpha intact, since
  no hardware decoder keeps it. Should the hardware open for a stream
  and then refuse it at a frame — a profile the GPU lacks — the bin puts the
  software line in its place, re-arms any preroll in progress and feeds it
  again from the last keyframe; downstream sees the same device and layout
  throughout. `path()` and `output_format()` answer at open, and
  `VideoDecodeBinHandle::path` later, with the reason software was chosen
  (`SoftwareReason::Alpha`, `NoHardwareDecoder`, `PixelFormat`,
  `HardwareRefused`, `SystemMemory`). `System` is `SwDecoder` alone, so the
  same code builds a decoder in a build without any GPU backend; on `D3d12`,
  whose frames here are NV12 only, alpha is not kept.

- **10-bit surfaces come down to 8 bits on the GPU.** `D3d11Scaler` takes
  a P010 texture as input — what D3D11VA decodes 10-bit HEVC to — and
  `D3d11ScalerFormat::Nv12` brings it down to NV12, keeping its colour
  tags. `CudaScaler::with_format(name, device, width, height, interp,
  format)` puts out `format` whatever comes in, and takes NVDEC's P010 for
  `CudaFrameFormat::Nv12`; a conversion `scale_cuda` has no kernel for is
  refused as `CudaScalerError::UnsupportedConversion`. `CudaScaler::new`
  still keeps the input's layout, and now lets P010 through as well.

- **`CudaDevice` is `Clone`.** It was one reference-counted FFmpeg device
  context already; a clone is another reference to the same device.

- **`D3d11SharedTextureSource`: another device's textures, as this
  pipeline's own frames** (Windows, `d3d11`). Whoever produces the pictures
  pushes a shared-texture handle per picture, and each one is opened on the
  pipeline's device, copied into a texture of its own, and sent downstream
  as a `Pixel::D3D11` frame:

  ```rust
  let (source, handle) =
      D3d11SharedTextureSource::new("browser", &device, context.clone(), 1920, 1080, 2)?;
  // ...from the producer's own callback, while its handle is still valid:
  handle.push(shared_handle, None)?;
  ```

  It copies because the texture belongs to the producer and a producer
  reuses its textures, while a pipeline — a compositor holding an input's
  last frame, say — has to outlive the call. It only copies: a texture of
  another size, or anything but `DXGI_FORMAT_B8G8R8A8_UNORM`, is refused
  rather than adapted, and the handle is opened afresh each push rather than
  cached by its value, which a producer is free to reuse for a different
  texture. The producer owes a flush before it hands a picture over.

  `push` blocks while the source's queue is full and `try_push` drops the
  picture instead, which is what a producer whose own thread cannot stall
  wants: a pipeline this feeds can be paused, and a paused one consumes
  nothing at all — so a blocking push stops the producer for as long as the
  pause lasts, and for a browser engine that means every one of its pages.

  New: `D3d11SharedTextureHandle`, `D3d11SharedTextureSourceError`,
  `Error::D3d11SharedTextureSourceError` and
  `ElementType::D3d11SharedTextureSource`.

- **`VideoLayer::premultiplied_alpha`**: whether an input's colour already
  has its alpha multiplied into it. Default `false`, which is every source
  that was here before; a browser engine's output is the case this exists
  for, since it composites its page that way. `D3d11VideoCompositor` blends
  such a layer by what it already holds rather than multiplying the alpha in
  a second time — which is what turned a half-transparent picture dark —
  and scales its colour by the layer's opacity alongside its alpha, so
  opacity still works on it. The software and CUDA compositors take the
  field and ignore it for now.

- **`CudaDecoder::supports(codec)`**: whether this FFmpeg build has a decoder
  for the codec that decodes on CUDA — whether `CudaDecoder::new` gets past
  choosing one. It needs no device, so a stream can be sent to `SwDecoder`
  and `CudaUpload` before anything is built. It speaks for FFmpeg, not the
  GPU: a profile NVDEC lacks still fails at the first frame with
  `HwAccelUnavailable`. `D3d11Decoder::supports` and
  `D3d12Decoder::supports` answer the same for D3D11VA and D3D12VA.

- **`ReplayBuffer`: the last stretch of an encode, saved on request.** A
  muxer in shape — `add_stream` per track, `open` for one sink per track —
  that keeps what its tracks are handed instead of writing it, letting go of
  whatever has fallen out of its window:

  ```rust
  let mut replay = ReplayBuffer::create(Duration::from_secs(30));
  let video = replay.add_stream("video", video_params, video_time_base);
  let audio = replay.add_stream("audio", audio_params, audio_time_base);
  let (mut sinks, handle) = replay.open()?;
  // ...wire the sinks behind their encoders; later, from a hotkey:
  let length = handle.save("replay.mp4")?;
  ```

  A saved clip opens on a keyframe of the video track and starts at zero,
  every track moved by that one origin, so the window is let go of a GOP at
  a time: it holds at most its length and at least that less one keyframe
  interval. `ReplayBufferHandle` is cheap to clone and holds the window
  weakly — once the sinks are dropped a save answers
  `ReplayBufferError::Stopped` — and a save blocks for the write, holding
  the tracks up only while it gathers references to what is held. New:
  `Error::ReplayBufferError` and `ElementType::ReplayBuffer`.

- **A compositor layer hands back the frame it will draw.**
  `SwVideoLayerHandle::latest_frame`, `D3d11VideoLayerHandle::latest_frame`
  and `CudaVideoLayerHandle::latest_frame` answer the last frame the input
  was handed — the pooled reference itself — or `None` before the first
  and once the input is removed. It is how a still picture of one input is
  taken: the compositor keeps that frame however long ago it arrived, so it
  is there for a producer that has paused or pushed a single picture,
  where a branch on the producer's own pipeline would wait for a frame
  that may never come.

- **Colour correction and a luma key: `VideoEffect`, on every backend.**
  `SwVideoEffect`, `D3d11VideoEffect` and `CudaVideoEffect` take a BGRA
  frame and apply one `VideoEffect`:

  ```rust
  let (effect, handle) = D3d11VideoEffect::new(
      "look", &device, context,
      VideoEffect::ColorCorrection(ColorCorrection {
          contrast: 1.2, saturation: 1.1, ..ColorCorrection::default()
      }),
  )?;
  // later, from a slider — or switch it to the other kind entirely
  handle.set_effect(VideoEffect::LumaKey(LumaKey { min: 0.1, ..LumaKey::default() }));
  ```

  `ColorCorrection` is brightness, contrast, saturation, hue, gamma and
  opacity; `LumaKey` cuts out what is darker than `min` or brighter than
  `max`, fading over a smoothing distance past each, and multiplies the
  alpha a pixel already has so it can follow a chroma key. Both defaults
  change nothing, and an element whose effect changes nothing — or that is
  turned off through `VideoEffectHandle::set_enabled` — hands each frame
  straight through.

  One element per backend rather than one per effect, because every effect
  resolves to the same small set of numbers: a colour matrix, an exponent,
  an opacity and a luma mask. One D3D11 shader, one CUDA kernel (PTX, so
  still no toolkit) and one software loop evaluate that, and an effect is
  added once. The GPU backends' tests compare their output with the
  software element's pixel for pixel: exact where the effect is linear,
  within one step where a gamma is set.

- **Speech becomes subtitles: `WhisperTranscriber`, and `subtitle` to carry
  what it says.** Two halves of one feature, kept apart on purpose.

  `WhisperTranscriber` is a terminal sink, like `OrtDetector` and for the
  same reason — what comes out is not media. It takes 16 kHz mono f32 (an
  `AudioResampler` in front is how audio reaches that shape) and hands
  `Segment { start_ms, end_ms, text }` to a callback.

  ```rust
  let transcriber = WhisperTranscriber::new(
      "transcribe", model_path, ChunkPolicy::default(),
      |segment| { println!("{}", segment.text); Ok(()) },
  )?
  .with_language("ko")?;
  ```

  The language is detected unless `with_language` names it, and naming it
  is both faster — detection is run again for every chunk — and steadier,
  since a chunk of music or two words can be heard as another language.
  Detection rather than whisper.cpp's own default, which is English: handed
  Korean under that, it does not fail but writes an English paraphrase.

  Lines arrive in order and never overlap. A line ends no later than the
  audio it was heard in, and one that overlaps the last starts where the
  last stopped — an MP4 text track cannot hold overlapping samples, and its
  muxer wrote negative durations for them.

  What a round reports is cut from what it heard by token, not by segment:
  the words starting inside its window are its own, the ones before were
  reported already, the ones at the edge are left for the next round. A
  segment the model opens in the context and runs into the new window is
  therefore neither lost nor said twice. `ChunkPolicy::token_timing` says
  how those times are found — `TokenTiming::Estimated`, whisper.cpp's own
  estimate, or `TokenTiming::Aligned`, dynamic time warping over the
  model's alignment heads, which are picked from the model file's own
  dimensions. On the minute of Korean below, aligned times left one piece
  said twice at the seams where estimated ones left several, for 6.1x real
  time against 6.4x.

  Whisper's encoder takes exactly 30 seconds, so transcribing a stream is a
  loop, and `ChunkPolicy` is what the loop is made of. `chunk_ms` (4000) is
  how much to gather before each inference; `live_edge_ms` (1000) is how
  much of the newest audio to distrust. The model's output nearest the edge
  is unstable — a word half-heard is guessed at, and the guess changes when
  the rest arrives — so that stretch is left for the next round instead of
  being reported and corrected. **A line, once reported, never changes**,
  which is what lets a caller write it straight to a file. Each inference
  is also given the previous chunk as context, so a word across the seam is
  heard whole; that costs a second pass over audio already transcribed.

  The delay is the sum of the two, so text runs about five seconds behind
  the speech. For a recording that is nothing, because a subtitle's place
  in a file is its timestamp and not its arrival.

  `crate::subtitle` is the other half: `subtitle::Codec` describes a text
  track (`parameters()`) and builds one line of it (`packet()`), in the
  codec its destination takes. `MovText` is for MP4, which takes 3GPP Timed
  Text and nothing else — not SRT, not ASS — and it writes the `tx3g`
  sample entry FFmpeg's own encoder writes, so a track says how its lines
  should look rather than leaving it to the player. `SubRip` and `WebVtt`
  are for an `.srt` or `.vtt` file, or a Matroska track: a sidecar file is
  a `FileMuxer` of its own, FFmpeg picking the writer from the extension,
  and it is written as the lines arrive — a recording that dies halfway
  leaves every line up to that moment. `MediaKind` gains `SubtitlePacket`
  so such a track states its contract like any other.

  Gaps need no packets: FFmpeg's muxer fills them with empty samples, so a
  caller pushes a line when there is something to say and nothing when
  there is not.

  Features: `whisper` builds whisper.cpp for the CPU, `whisper-vulkan` for
  any GPU Vulkan reaches. Vulkan rather than CUDA because CUDA needs a 3 GB
  toolkit to build and serves only NVIDIA, while Vulkan's runtime ships
  with every driver and only its shader compiler is a build requirement.
  Measured on an RTX 3050 against a minute of dense Korean speech, in the
  streaming loop above: `large-v3-turbo` at 6.5x real time and `small` at
  9.7x, where `large-v3-turbo` on the CPU had not finished after fourteen
  minutes. The loop costs more than transcribing a file in one pass — every
  inference pays for a whole 30-second encoder window to hear eight seconds.

  See `examples/core/transcribe`, which writes a copy of a file carrying
  video, audio and the transcription as three tracks.

- **`Rack` holds a stretch of chain whose contents can be replaced while
  frames are flowing.** Everything else in this crate settles its graph
  before the pipeline runs, so changing one element in the middle means
  building the branch again — which restarts whatever is at the top of it.
  For a camera that is a visible stall; on Wayland it is a portal dialog.

  ```rust
  let (rack, rack_handle) = Rack::new(
      "filters",
      InputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)),
      OutputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)),
  );
  // ...pipeline running...
  rack_handle.replace(vec![Box::new(key)])?;   // takes effect on the next buffer
  rack_handle.replace(Vec::new())?;            // and an empty rack is a wire
  ```

  What is in one is still a straight line, so an element with more than one
  output is refused at `RackHandle::replace` rather than a frame later. A
  replacement drops what the outgoing elements were holding, which makes a
  rack right for elements whose output depends on the buffer in front of them
  and nothing else — a scaler, a converter, a chroma key — and wrong for an
  encoder or a muxer. Draining the old line would put its last frames after
  the new line's first ones, and an element in the middle of a graph cannot
  put that timeline back in order.

  Its contracts are declared by the caller rather than derived from what it
  holds, because the caller putting one between two fixed elements knows
  both, and deriving would answer "unknown" across exactly the stretch most
  likely to be wired up wrong.

  Being inside one changes nothing else about an element. On its way in it
  is given what `ChainBuilder` gives a stage it builds: log records naming
  the pipeline it is in, the `Context` if it needs one, and the same tracer,
  so a failure it raises reaches a `Queue` naming *it* rather than naming
  the rack that was holding it. `Box<dyn Filter>` implements `Element`,
  `Source` and `Sink` for that last part — a rack's contents arrive already
  boxed, and the wrapper is written against a type that implements the
  three.

- **`CudaChromaKey` keys a green screen on the GPU under CUDA.** Chroma
  keying existed for the CPU and for D3D11, which left Linux without one at
  all: the compositor there is CUDA, and this crate refuses to wire a branch
  whose memory domains do not match, so the software element could only have
  gone in behind a `CudaDownload` and back out through a `CudaUpload` — two
  PCIe crossings per frame around a per-pixel transform.

  BGRA in, BGRA out, like both siblings, and it keeps PTS, duration and the
  colour tags: keying writes alpha and leaves the colour alone. Odd
  dimensions are fine, unlike `CudaConverter` — BGRA has no subsampled
  plane to halve. `CudaChromaKey::new` returns a `ChromaKeyHandle` beside
  the element.

  The kernel is hand-written PTX carried in the existing BGRA module, so a
  build needs no CUDA toolkit — the driver JIT-compiles it when the module
  loads, exactly as it already did for the conversion and blend kernels. It
  computes the same normalized BGR distance and the same feather ramp
  `SwChromaKey` does, from the same resolved band the D3D11 shader reads,
  which is now one shared `feather_band` rather than a copy per backend.

  New: `Error::CudaChromaKeyError` and `ElementType::CudaChromaKey`.

- **`RtmpMuxer` publishes a live broadcast to an RTMP server** — Twitch,
  YouTube, or a local MediaMTX. It is the publishing half only: nothing here
  runs a server, and the address and stream key come from whoever receives.
  Shaped like `FileMuxer` rather than `RtspSink`, because a broadcast is video
  *and* audio in one FLV container and the header has to describe both up
  front: `create` connects, `add_stream` registers each track, and `open`
  writes the header and returns one `Sink` per track. It remuxes and does not
  encode, so H.264 and AAC come from the encoders upstream.

  A publish URL ends in a credential, so nothing logs the URL it was given —
  `redacted_url` is what reaches a log and what a caller should display. It
  does not reconnect: a connection lost mid-broadcast is a write error, and
  recovering means a new `RtmpMuxer` and so a fresh keyframe.

  New: `Error::RtmpMuxerError`, `ElementType::RtmpMuxer`, and the
  `rtmp_publish` example.

- **`AudioGate` and `NoiseSuppressor` clean up a microphone.** The two
  audio filters a streaming application reaches for first, working on the
  signal rather than its format: both take `f32` audio, packed or planar,
  and pass it on with its timestamps and layout unchanged.

  `AudioGate` silences what is below a level and passes what is above it,
  with a streaming application's five settings — open and close thresholds
  in dBFS, attack, hold and release — and their meaning. Two thresholds so a
  level hovering at the line does not open and close the gate on every
  syllable; a hold so a pause between words is not cut out. One gain for
  every channel, taken from the loudest, so a stereo image never leans.
  `AudioGateHandle::set_options` replaces all five at once from the next
  frame. No delay.

  ```rust
  let (gate, handle) = AudioGate::new("mic-gate");
  handle.set_options(AudioGateOptions { open_threshold_db: -30.0, ..handle.options() })?;
  ```

  `NoiseSuppressor`, behind the new `rnnoise` feature, takes steady
  background noise — a fan, a room — out of speech with RNNoise, through
  `nnnoiseless`: pure Rust, the weights compiled in, no model file to ship.
  48 kHz only, as the network was trained; about 0.4% of one core per
  channel. RNNoise gives each 10 ms block back one block late, and that
  block is taken off the front rather than left to push the sound late
  against its timestamps: frames come out one for one, a block after they
  arrive, each carrying its own samples denoised. `Eos` drains the block it
  holds.

  New: `Error::AudioGateError`, `Error::NoiseSuppressorError`,
  `ElementType::AudioGate`, `ElementType::NoiseSuppressor`, and the
  `rnnoise` feature.

- **`AudioCompressor` and `AudioLimiter` even a voice out.** The next two
  after a gate, with its format, its linked gain and its handle.

  `AudioCompressor` turns down what goes over a threshold by a ratio — at
  4:1 a level 12 dB over comes out 3 dB over — with a streaming
  application's five settings: threshold, ratio, attack, release and an
  output gain to make back up what it took off. The level it reacts to is
  followed per sample, rising over the attack and falling over the release,
  so it neither pumps on each waveform peak nor lets a word start at full
  volume.

  `AudioLimiter` is the last thing on a channel: nothing comes out louder
  than its threshold. It does not look ahead, so it reacts in the sample a
  peak arrives, turning down to exactly what brings that sample to the
  ceiling; that is what makes the threshold a guarantee rather than a
  target. It lets go over its release.

  ```rust
  let (compressor, _) = AudioCompressor::new("mic-compressor");
  let (limiter, handle) = AudioLimiter::new("mic-limiter");
  handle.set_options(AudioLimiterOptions { threshold_db: -3.0, ..handle.options() })?;
  ```

  New: `Error::AudioCompressorError`, `Error::AudioLimiterError`,
  `ElementType::AudioCompressor`, `ElementType::AudioLimiter`.

- **`Pipeline::stats` says what every element is doing while it runs.** The
  graph says what a pipeline looks like; this says whether anything moves
  through it — a capture that stopped delivering, a queue that is full and
  dropping, a branch still draining after it was finished. Each used to be
  found in a log afterwards.

  ```rust
  for element in pipeline.stats().elements {
      println!("#{} {} in={} busy={:?} idle={:?}",
          element.id, element.name, element.buffers_in,
          element.busy, element.idle_for);
  }
  ```

  Per element: buffers taken, time spent inside `consume`, how long since it
  last took or pushed a buffer, errors, and whether it has seen `Eos`; per
  output pad, what went through it, and the bytes of the packets among it —
  an encoder's bitrate; for a `Queue`, how full it is, what it dropped and
  how long its upstream waited on it; for a video compositor, the frames it
  drew, the ticks of its frame rate it missed because the one before ran
  late, and the time spent drawing — not handing the frame on, so a slow
  encoder downstream shows up as missed ticks with little drawing time,
  and is told apart from a compositor that is itself too slow. Running totals rather than
  rates — two readings and the time between them give the rate, matched by
  `ElementStats::id`. `busy` includes every stage after it on the same
  thread, up to the next `Queue`, because a chain runs as nested calls.

  Runtime `Tee` branches are covered as they come and go. An attached branch
  is reported from the moment it joins; a detached one disappears at once;
  one ended with `finish_branch` goes on appearing, as
  `ElementState::Finishing`, while its `Eos` drains, and disappears once it
  has been dropped. A re-attached branch is new elements with new ids.

  Always on, and counted in the wrappers every stage and pad already has,
  so an element written outside this crate is counted like one inside it.
  The cost is about a tenth of a microsecond per buffer per stage, nearly
  all of it reading the clock.

  New: the `stats` module (`PipelineStats`, `ElementStats`, `ElementState`,
  `PadStats`, `QueueStats`, `TickStats`).

- **Encoders can say what colour their stream holds.** `CudaEncoder` and
  `SwEncoder` gain `with_color`, which is `new` plus a `ColorDescription` —
  matrix, range, primaries, transfer — told to the encoder before it opens,
  which is the only time it reads one. It ends up in the stream's headers
  and the container's, where every player finds it; `new` still says
  nothing. Nothing is converted: it names what the frames already are.

  ```rust
  let encoder = CudaEncoder::with_color(
      "encode", &device, options, ColorDescription::BT709_LIMITED,
  )?;
  ```

  Untagged, a player guesses, and FFmpeg guesses BT.601 at any size: a
  BT.709 recording's (230, 20, 20) decoded as (211, 0, 22).
  `ColorDescription::describe` says the same about a frame.

  `D3d11VideoEncoder` gains the same `with_color`, for NV12 input only:
  given BGRA the encoder converts for itself, by a matrix of its own —
  `h264_nvenc` BT.601, which it tags; `h264_mf` BT.601 at 320x240 and
  BT.709 at 1080p, which it does not — so describing BGRA is refused with
  `D3d11VideoEncoderError::DescribedBgraInput`. Convert with
  `D3d11Scaler` to `D3d11ScalerFormat::Nv12`, which states BT.709 limited,
  and describe that.

  New: `color::ColorDescription`.

### Changed

- **`Player` decodes each stream on a thread of its own.** A queue of 64
  packets now sits between the demuxer and each decoder, so a slow picture
  decode no longer holds up the sound, and a file muxed with one stream a
  second or so ahead of the other waits in those queues rather than in the
  demuxer. A program building the same graph itself does the same with a
  `queue` in front of each decoder.

- **A seek made while playing pauses before it plays on.** Its preroll now
  ends in a `Pause`, and the `Resume` follows it, where it used to play on
  with the `Resume` alone. An element counting control messages sees one
  more `Pause` per seek. It is what makes sure no element is handed data
  before the `Resume` that lets it take any has reached it.

- **An idle or paused `Queue` sleeps until something happens.** Its worker
  woke every 20 ms to see whether it had been dropped, idle or paused
  alike, and a thread waiting to hand it a buffer looked every 5 ms for a
  request on its way — a graph with many queues, paused, kept every one of
  them waking fifty times a second. They are woken now by what they wait
  for: a request, the pipeline's state moving on, the queue being dropped,
  room being made. Only a worker whose downstream is not ready still looks
  again on a timer, for the downstream that becomes ready without saying so
  — a device playing out what it holds.

- **`ChainBuilder::to` takes any sink, boxed or not.** `.to(counter)` rather
  than `.to(Box::new(counter))`, and the same for `build`. A boxed element is
  now the element it holds — `Box<T>` is an `Element`, `Sink` and `Source`
  wherever `T` is — so a sink already boxed, as a muxer hands its sinks
  over, is still one, and `.to(Box::new(x))` compiles as it did.

- **A chroma key multiplies the alpha it is given instead of replacing it.**
  `SwChromaKey`, `D3d11ChromaKey` and `CudaChromaKey` used to write the key's
  coverage straight into alpha, so a key placed after a luma key put back
  everything that one had taken out. An opaque input — every capture and
  decoder in this crate — keys exactly as before, byte for byte.

- **`VulkanWindowRenderer` copies a CUDA frame once, not twice.** CUDA used
  to copy each frame into a Vulkan buffer, and Vulkan then copied that into
  the images it samples. The images are now imported into CUDA as arrays and
  written directly, and the second copy is gone. A 4K60 NVDEC stream
  measured 8% of an RTX 3050's SMs and memory bandwidth before, and 6%
  after. What is drawn is the same byte for byte.

### Fixed

- **`VideoDecodeBin` on a device that decodes no video decodes in
  software.** A software adapter — what a machine or a virtual machine
  without a GPU has — cannot make a D3D11VA or D3D12VA decoder, and the bin
  failed to open with the decoder's `HwDeviceInit`, where a stream the
  hardware refuses at its first picture was already gone on with in
  software. It now does the same from the start: the stream is decoded in
  software and put on the device, and `VideoDecodeBinHandle::path` reads
  `SoftwareReason::HardwareRefused`.

- **A file whose picture is muxed ahead of its sound no longer freezes when
  both branches are full at once.** `FileDemuxer` held a full picture
  branch's packets back only while the sound's branch had room at that
  moment; a packet read while both were full was pushed into the picture's
  queue and waited on there. Once the sound drained, the demuxer was still
  waiting on a picture that waited on sound nobody was reading, and
  playback stopped for good a few seconds in — found with `Player` on a
  recording muxed a second ahead, whatever keys were being pressed. A full
  branch's packets are now held back whenever the read cursor is within
  `MAX_INTERLEAVE` of what is owed, and reading waits while no branch can
  take anything, as before.

- **The full stop and the comma step a `Player`'s picture from the
  keyboard.** `respond_to` steps on `Key::Char('.')` and `Key::Char(',')`,
  and no window renderer produced them: Windows' virtual-key codes and X11's
  keysyms mapped letters and digits to `Char` and everything else to
  `Other`, so the keys did nothing while the tests, which hand `respond_to`
  the `Char` directly, passed. Both now map the full stop, the comma and
  the minus — the three punctuation keys Windows defines as one key on
  every layout — to `Char`, and no more, so a program reads the same keys
  on both.

- **An accurate seek to just before a keyframe shows the picture there, not
  the keyframe.** A file is sought by when its keyframes are decoded, and
  with B-frames that is before they are shown: the pictures shown just
  before a keyframe belong to the group before it, and a target among them
  landed on the keyframe, where nothing covered it — the keyframe was shown,
  or with open groups a picture after it. `FileDemuxer` now reads where a
  seek landed and, where the first picture starts after the target, seeks
  again from before it is decoded. The target is also no longer rounded up
  to the stream's next tick on its way to FFmpeg, which landed an instant
  before a keyframe on the keyframe in files without B-frames too.

- **An accurate seek no longer drops the pictures after the one it shows.**
  To pick the picture covering the target, a decoder has to decode the one
  after it, and it dropped that one — with whatever else the same packet
  decoded — once the seek had its picture. Playing on after an accurate
  seek skipped a picture or more straight after the one it had shown. What
  follows the seek's picture is now kept and handed on after it.

- **The end of a stream is not lost when a file is paused, sought or
  finished at its last picture.** A `Pacer` or `VideoSynchronizer` waiting
  on that picture's time lets go of it when a request comes and keeps it,
  with whatever follows, for its next call — and with the `Eos` already
  behind it there is none: the thread feeding it has nothing left to hand
  over. What it kept was dropped with it, so the stream never ended and no
  `Finished` came. Once the end of the stream is in, they now hand
  everything on at once. And a queue being dropped waits for the
  pipeline to settle an interrupt before it drains, as it does while
  running, so it does not feed a pacer that is letting go.

- **A `Pacer` puts one sample through a seek, not two.** A seek's request
  can reach a `Pacer` before the interrupt the pipeline raises for it, and
  for that moment the pacer took itself for interrupted: it kept the
  preroll's buffer and handed it on together with the next, two samples
  into a terminal that takes one. A seek made while paused could then show
  a picture past the one it landed on. A preroll waits for nothing, so the
  pacer now lets its buffers straight through whatever the interrupt says.

- **A source tells every one of its pads, whatever one of them answers.**
  A control message went to a source's pads in turn and stopped at the
  first that failed: a file's picture branch refusing a `Pause` left the
  sound branch beside it playing. Every pad is told now, and the first
  failure is still what comes back.

- **A queue drops what a seek has left behind, however late it gets
  there.** A seek empties every queue with a `Flush` and then repositions
  the sources, and that relied on every source being paused and every
  thread's timing going its way. A source of your own that read on while
  paused — handed the `Flush` and the `Seek` as two requests, and pushing
  between them — put media from the old position into a queue the flush
  had just emptied, and it was delivered ahead of the new position's.
  Every buffer now carries the timeline it was read on, from the thread
  that read it across each `Queue`, and a queue drops what is from one a
  seek has left. Nothing in the public API changes, and a buffer made on
  a thread no pipeline numbered — an element's own worker, a caller
  driving elements by hand — is never dropped for it.

- **A seek no longer loses the stream that prerolls first.** After handing
  its branch the seek's one sample, a decoder suppressed what it decoded
  until the preroll ended — and went on taking packets meanwhile, since
  nothing said it was full. With a picture slow to preroll, the sound's
  decoder was fed the whole rest of the file in that time and threw it
  away: a player resumed after a seek with no sound at all. A decoder now
  says it is not ready while its preroll has its sample and runs on, so
  the queue in front of it holds the packets and the demuxer keeps them.
  `VideoDecodeBin` passes the answer on from the decoder inside it.

- **Every control request lets go of a thread blocked handing data on.**
  Only pause, stop, finish and a seek's opening check used to raise the
  interrupt that makes a `Queue` take such a buffer as held over and a
  `Pacer` let go of its wait; the rest were sent on the assumption that
  the graph was already paused and nothing could be blocked. A source that
  was not — as `FileDemuxer` at the end of its file was until 208af56 —
  then left a seek waiting on it for good. Every request raises one now,
  and a paced wait wakes the moment it is raised rather than at the end of
  a polling slice.

- **A `Tee` loses nothing while a seek's other branches catch up.** While
  a seek's preroll runs, a `Tee` stops feeding a branch that has taken its
  sample, so the branches stay level — and it dropped what came for that
  branch meanwhile rather than keeping it. In front of the decoders that
  was packets: a branch lost a third of a second of them, and the pictures
  that depended on them, whenever a sibling was slower to preroll. And a
  seek past the end of a file, whose last picture and `Eos` arrive
  together, left every branch without an end. What arrives for a held
  branch is now kept, in order, and handed on when playback resumes.

- **`Pipeline::finish` of a playing file hands its terminals their `Eos`.**
  Two things lost it. `finish` interrupts the clock, and a `Pacer` or
  `VideoSynchronizer` in a wait gave up and kept its picture — then took
  the interrupt as answered only by a control message of its own, which
  `finish` sends none of, so every wait after it gave up too and the
  `Eos` stayed behind the picture for good. And the queues a finished
  source owns are dropped as it ends: a worker that found its downstream
  busy at that moment — a queue in front of a `Pacer` usually is — ended
  there, dropping what it held and the `Eos` with it. A waiting element
  now lets go only while an interrupt is outstanding, and a dropped queue
  still holding an `Eos` hands everything up to it on first; one holding
  none, as a detached branch's does, is abandoned as before. A muxer at the
  end of a paced branch had never written its trailer on `finish`. Found by
  the new control conformance sequences — see `CONTRIBUTING.md`.

- **A seek the moment a file has ended no longer hangs or plays nothing.**
  A `FileDemuxer` waiting at the end of its file passed a `Pause` on
  without pausing itself, and a pipeline pauses around every seek — so a
  seek from the end read the file on at once, into the paused queue
  behind. Once that queue was full the source was stuck handing it a
  packet and never took the `Preroll` the seek waited on: `Pipeline::seek`
  did not return. Behind a queue that drops on a timeout, as `Player`'s
  do, it returned instead with the start of the file dropped, and nothing
  played — `Player::play` after `Ended` stayed at zero. It showed only
  where the queue filled before the seek's next step, as on a slow CI
  runner. Paused at the end, it now waits like anywhere else.

- **The streams `FileDemuxer::open` and `RtspSource::open` describe no
  longer hold the source open.** Each `StreamInfo`'s parameters shared the
  ownership of the whole input with the source, so a file stayed open —
  and on Windows could not be deleted or replaced — for as long as the
  caller kept the stream list, even after the pipeline had ended; an RTSP
  session stayed connected the same way. That sharing is counted by a
  non-atomic `Rc`, and with the source on its own thread and the list on
  the caller's, the two could drop it at once. They are copies of their
  own now.

- **`SegmentedFileMuxer` cuts the sound where it cuts the picture.** A
  picture reaches a muxer later than its sound — through a queue, and an
  encoder holding frames back — and the cut happened when the keyframe
  arrived, so the sound already there for the next segment ended the one
  before: 0.4 s of it at every cut behind NVENC, playing on past that
  file's last picture and missing from the start of its own. Another track's
  packets now wait for the picture to pass them, and at a cut, what comes
  before the keyframe's time ends the outgoing file and the rest opens the
  next. Sound that arrives before the first picture no longer makes a
  segment of its own either.

- **A `ReplayBuffer` clip ends with its picture.** For the same reason its
  sound ran on 0.46 s past the last frame; what the other tracks hold past
  the picture's end stays out of a clip.

- **`RtspSource` no longer logs the password in a camera's address.** It
  logged the URL it opened as given; it now logs it with the credentials
  removed, as `RtspMuxer` already did.

- **The docs match what the examples do.** `screen_record_software` and
  `screen_record_av` end with `finish()`, which keeps the last frames the
  encoder holds, and their READMEs said `stop()`, which drops them; the
  `rtsp_source` example took a developer's camera address when given
  none, and asks for one now; the README says RTSP output goes to a server
  such as MediaMTX, which this crate does not provide; and the muxers' doc
  examples use `VideoCodec::OpenH264`, which every build has, not
  `VideoCodec::H264`, which many do not.

- **`D3d11Upload` keeps a frame's primaries and transfer.** It passed on
  the matrix and range and dropped the other two, so an uploaded BT.2020
  PQ or HLG picture reached anything reading them after it looking like
  SDR. The other uploads already copied all four.

- **`Player` stays at the end once it gets there.** Its `position` went on
  counting past the file's length after `Ended`, a second a second; it now
  reads the file's length there. `play` once `Ended` has been reported
  starts the file again from the start, as a player's play button does.

- **`VideoEncodeBin` refuses a frame in another layout than
  `EncodeInput::System` said** — `VideoEncodeBinError::FormatMismatch` —
  rather than encoding it under a colour description made for the layout
  it was told.

- **The `fanout` and `tee` examples end at the end of their file.** They
  waited for the bus to close, which a file's pipeline no longer does by
  itself; they stop on `Finished` now.

- **The D3D11 and D3D12 window renderers draw a frame in its own colours.**
  Their NV12 shaders had BT.601 limited range written into them, carried
  over from the examples' presenters, so every HD picture — BT.709, what
  nearly every decoded stream is — was converted with the wrong matrix:
  the R'G'B' (230, 20, 20) a BT.709 frame holds came out as (212, 0, 23).
  `D3d11WindowRenderer` and `D3d12WindowRenderer` now draw their frames
  themselves rather than through a presenter, so they read each frame's
  colour description and convert with its own matrix and range — BT.709,
  BT.601 or BT.2020, limited or full, and for an untagged frame BT.709 over
  576 rows and BT.601 otherwise — as `D3d11VideoCompositor` already did.
  They check frames exactly as `D3d11Renderer` and `D3d12Renderer` do, and
  report the same errors. Those two, and their presenter traits, are
  unchanged: a presenter of one's own is still handed no colour
  description.

- **Pausing leaves a queue's backlog in the queue.** `Pipeline::pause`
  interrupts every paced wait before its request has reached the queues,
  and an interrupted `Pacer` or `VideoSynchronizer` takes whatever it is
  handed without waiting. A queue's worker went on handing it buffers in
  that time, and a source blocked on the full queue could only pass the
  pause on once the worker had made room — so the whole backlog moved
  into the pacer: a 12-frame queue lost 14 frames to it on nine pauses in
  ten. Held there until resume, they were out of their pool while the
  queue filled again, which a fixed D3D11VA or NVDEC pool had to be sized
  twice over for. A queue built by a pipeline now takes nothing but
  control from the interrupt until every source has acknowledged the
  request, and a buffer that finds its channel full meanwhile is held
  over rather than blocking the thread with the request to pass on; the
  pacer holds the one frame it was waiting on. `d3d11_decode_render`'s
  decoder budget is back to its queue and a few more.

- **A file whose picture is muxed ahead of its sound plays with any video
  queue.** `FileDemuxer` waited on a pad whose branch was full, which
  stopped its one read cursor. When the full branch was the picture —
  frames the audio had not reached yet — the audio packets that would let
  it reach them lay further on in the file, so the audio clock stopped and
  the picture waited on it for good: an 8-frame video queue froze a file
  whose picture leads its sound by a second at 0.667 s, with no error. A
  full pad's packets are now held back while another branch is still
  waiting on the cursor, for up to 5 s of file time, and go out in order
  once it has room; past that, or with nothing else waiting, the full pad
  is waited on as before.

- **A pipeline with several sources no longer loses one to a pause, seek
  or finish.** Control went to one source at a time, after the clock had
  been interrupted. A source waiting its turn behind another's cascade was
  woken with nothing to take: its interrupted `Pacer` handed each buffer
  straight back, so it read on unpaced — to the end of its file within
  milliseconds, where its thread ended. Two demuxers on one file, one for
  the picture and one for the sound, failed their seek's preroll on three
  runs in four. Every request is now queued on every source before
  anything is woken, and the sources handle it together rather than in
  turn.

- **`seek` no longer hangs a player that seeks while the video waits on
  the sound.** `seek` first asks every branch whether it can seek, and it
  interrupted the clock only after that. A `VideoSynchronizer` holding a
  frame the audio had not reached yet — the usual state of a file whose
  picture is muxed ahead of its sound — sat inside a `Queue` worker that
  takes control only between buffers; the audio it waited for came from
  the demuxer, which had stopped to ask that very question. None of them
  moved again, with no error and no timeout. A newcomer's player hung this
  way on four seeks in six right after resuming. The clock is interrupted
  before the question now.

- **The recording examples end with `finish`.** `audio_record`, `hls`,
  `rtmp_publish`, `screen_record_software`, `screen_record_av` and the
  compositor examples ended with `stop`, which finalizes a playable file
  but abandons what the encoders still hold — the last frame of video, two
  frames of AAC. The Linux halves of `screen_record_av`, `_software` and
  `_nvenc` also passed encoders a `time_base` they no longer take, and did
  not compile.

- **A pipeline played to its end can be sought through a `Queue`.** A
  queue's worker ended with the `Eos` it forwarded, so once a `FileDemuxer`
  had reached the end of its file, a seek repositioned it and its new
  stream stopped at the first `Queue`: the terminal behind it never saw a
  sample and `Pipeline::seek` failed with `PrerollError::TimedOut` five
  seconds later. The worker now goes back to waiting after `Eos`, and only
  `Stop` or dropping the `Queue` ends it, as before. What changes to see is
  that elements behind a queue now receive the `Stop` that ends a pipeline
  after their `Eos`, as those in a branch without one always did, and are
  released at that `Stop` or when their source is dropped, rather than at
  `Eos`.

- **`RtspSource`'s packets carry their stream's time base.** FFmpeg does not
  promise to fill a demuxed packet's, and `FileDemuxer` already stamped it;
  `RtspSource` did not, so a packet-level element downstream could read a
  `pts` in no unit at all.

- **`D3d11SharedTextureSource` says what a producer really owes it.** Its
  docs said a flush was enough before a push; it is not. A flush submits
  the producer's drawing without waiting for it, and this element copies on
  its own device's queue, which cannot wait on another device's — so under
  a loaded GPU a flushed picture was copied before it existed, as an empty
  frame, in about one push in forty. A producer either guards the texture
  with a keyed mutex released with key 0, which this element already takes
  around its copy, or waits for its drawing to complete before pushing.
  Nothing in the element changed; its own test had the same mistake and
  failed intermittently for it, and now waits, and the keyed-mutex path
  has a test of its own.

- **A 10-bit stream that did not say so no longer leaks P010.** A stream
  described only by its session has no layout in its parameters, so
  `VideoDecodeBin` sent it to the hardware as if it were 8-bit, and a 10-bit
  one reached whatever was downstream as P010, which nothing there reads.
  The first P010 frame now puts the target's converter to NV12 after the
  decoder, on `D3d11` and `Cuda`, and the frames come out NV12 as
  `output_format` said at open.

- **BT.2020 video comes out in the right colours.** `D3d11Scaler` described
  colour to the video processor with a bitfield whose one matrix bit says
  BT.601 or BT.709, and read BT.2020 as BT.601; `CudaConverter` and
  `CudaVideoCompositor` read every NV12 frame as BT.709, BT.601 included. A
  red (200, 40, 40) encoded with BT.2020's matrix came out (190, 32, 40)
  and (204, 48, 38). `D3d11Scaler` now hands the processor a
  `DXGI_COLOR_SPACE_TYPE`, and the CUDA kernel takes the matrix and range of
  the frame it reads, as `D3d11VideoCompositor` already did; a CUDA NV12
  layer not in the canvas's own BT.709 is converted on its way in rather
  than copied. BT.2020's primaries are brought into BT.709's too, which
  a D3D11 video processor does only from P010 to RGB and writes no BT.2020
  Y'CbCr at all — so `VideoDecodeBin` puts a BT.2020 stream out as BT.709
  BGRA on `D3d11` and `Cuda`, straight from the decoder, and every element
  after it reads it right. `output_format` says so at open.

- **A GPU refusing a stream is reported as that.** `D3d11Decoder`,
  `D3d12Decoder` and `CudaDecoder` returned FFmpeg's `EPERM` and then
  `AVERROR_INVALIDDATA` for a stream whose profile the GPU lacks — the same
  errors a damaged stream gives. They now return `HwAccelUnavailable` for
  every packet after the hardware turned the stream down.

- **The CUDA compositor says its canvas is BT.709, limited range** — which is
  what every fill and blend into it converts with. It said nothing, so
  whatever read it later picked its own answer; swscale's was BT.601. The
  D3D11 compositor's BGRA canvas already said `RGB`, full range.
- **`SwScaler` reads a YUV frame by what it says it is.** It used swscale's
  default whatever the frame said, which is BT.601 limited, so a BT.709
  picture scaled to RGB came out with its colours shifted — (230, 20, 20) as
  (211, 0, 22). Going YUV to YUV it keeps the matrix and range it was given
  rather than converting into the default, and every output says what it is.
  A frame that says nothing is read exactly as before.
- **`CudaDecoder` opens a decoder that decodes on CUDA, not FFmpeg's default
  one.** With `libdav1d` built in, FFmpeg picks it for AV1 ahead of its own
  `av1` decoder, and only the latter reaches NVDEC — so AV1 opened, as
  software, and failed at the first frame with `HwAccelUnavailable`. It now
  opens `av1` and decodes on the GPU. A codec with no such decoder at all,
  ProRes say, is refused by `new` instead of opening and failing the same
  way later. `D3d11Decoder` and `D3d12Decoder` opened FFmpeg's default too
  and failed on AV1 the same way; they now choose the same way, for D3D11VA
  and D3D12VA, and every frame of an AV1 stream comes out on the GPU.

## 0.2.0

Two renames, a camera source on both platforms, and a good deal of runtime
control that used to be fixed at construction.

### Breaking

- **`Mp4Muxer` is now `FileMuxer`.** `format::output` has always guessed the
  container from the file name, so this type has always written whatever the
  path asked for; MP4 was the only thing still claiming otherwise. Rename
  `Mp4Muxer` → `FileMuxer`, `Mp4MuxerError` → `FileMuxerError`,
  `Mp4MuxerStreamSink` → `FileMuxerStreamSink`, `SegmentedMp4Muxer` →
  `SegmentedFileMuxer`. `Error::Mp4MuxerError` → `Error::FileMuxerError`. The
  `ElementType` variants carry the same names into logs and change with them.
  Deliberately without a compatibility alias.

- **`D3d11NvencEncoder` is now `D3d11VideoEncoder`**, and it reaches Intel and
  AMD hardware as well as NVIDIA's. The old element opened `h264_nvenc`, so a
  machine without an NVIDIA GPU fell back to encoding on the CPU — the
  expensive path the element existed to replace. It now asks Media Foundation
  for whichever hardware H.264/HEVC transform the installed driver registers.
  Rename `D3d11NvencEncoder` → `D3d11VideoEncoder`, `D3d11NvencEncoderOptions`
  → `D3d11VideoEncoderOptions`, `D3d11NvencEncoderError` →
  `D3d11VideoEncoderError`, `D3d11NvencCodec` → `D3d11VideoCodec`,
  `D3d11NvencInputFormat` → `D3d11VideoInputFormat`.
  `Error::D3d11NvencEncoderError` → `Error::D3d11VideoEncoderError`.

- **Elements are handed their clocks instead of being given them.**
  `Pacer::bind_playback_clock` and `bind_playback_clock_deferred` are gone.
  Nothing checked that the clock a caller passed was the one the pipeline
  actually runs on, and at least one example passed a `Clock::new()` the
  pipeline never paused, reset or interrupted. Construct a `Pacer` with only
  its name, time base and options; the pipeline supplies the rest through
  `Element::attach_context` when the element is wired. The same applies to an
  audio renderer claiming the playback clock as master.

- **`SegmentPolicy` has a second variant, `Size(u64)`** — an exhaustive match
  over it no longer compiles. See *Added*.

### Added

- **Camera capture, on both platforms.** `MfCaptureSource` (Windows, feature
  `mf-capture`) goes through Media Foundation's source reader;
  `V4l2CaptureSource` (Linux, feature `v4l2-capture`) goes through FFmpeg's
  own `video4linux2` demuxer, with four read-only ioctls behind the picker.
  Both push CPU-resident NV12 frames, so `D3d11Upload` and `CudaUpload` take
  them directly. A caller picks a device and a picture shape — `MfDevice` /
  `MfCaptureFormat`, `V4l2Device` / `V4l2CaptureFormat` — and never a subtype:
  which of MJPEG, YUY2 or NV12 a mode is natively is the element's business.

- **`PipelineBridge`**, which carries buffers from one pipeline into another.
  With `PipelineBridgeHandle`, `PipelineBridgeOptions` and
  `PipelineBridgeError`.

- **A settable frame rate.** The new `rate` module publishes `FrameRate` and
  `FrameRateHandle`, and the compositors and captures take one, so an output
  rate can change while the graph runs instead of being fixed when it was
  built.

- **`MixFormat`**, and an `AudioMixer` whose mix format can change while it
  runs.

- **`VideoSourceRect`** on a layer, so a compositor input can draw only part
  of its picture. Both compositors honour it; the CUDA one copies the region
  before scaling.

- **`FileDemuxerHandle`**, which plays a file again when it reaches the end
  and reports how far the loop has carried the timeline.

- **`Pipeline::is_running`**, so a caller can ask whether anything is still
  on a thread of its own without draining the bus for it.

- **`SegmentPolicy::Size`**, cutting a recording by bytes as well as by
  duration. Both wait for the video track's next keyframe, so a segment
  overruns by about a GOP either way.

- **B-frames.** `SwEncoder` and both hardware encoders take a count, and
  `SwEncoder` can now use FFmpeg's own H.264 encoder beside OpenH264.

- **A CUDA compositor layer that brings its own transparency**, blended under
  its per-pixel alpha.

- **`WebRtcTrackSink::set_source_parameters`**, one declaration that tells a
  track sink what feeds it, so a peer is sent SPS/PPS.

- **`Pacer::with_discontinuity_limit`**, for a source whose timeline may
  restart underneath it.

- New `ElementType` variants for everything above, plus `AudioMixerInput` so
  a mixer input says what it is in a topology diagram.

### Changed

- **One place holds where the media timeline sits.** `Clock` went back to
  being the monotonic control and pause clock; `PlaybackClock` owns the media
  origin and answers the mapping both ways. A pipeline used to carry two
  independent origins that could disagree; no graph mixed `Pacer` and
  `VideoSynchronizer` hard enough to expose it, which is why nothing broke.

- The test fixture is now synthesized from this crate's own sources and
  encoders rather than asked for through an environment variable, so the
  library's tests run everywhere against the same file.
  `MEDIA_PP_TEST_VIDEO` is read by `tests/soak.rs` alone.

### Fixed

- Encoders put their codec headers in extradata, so a non-MP4 container gets
  them; WebRTC puts them back in front of every keyframe.
- `AudioMixer` reads the mix in chunks rather than through a constant-sized
  window, and resamples an input through the engine that sizes its output.
- `ChangeGate` times its rate limit from a deadline rather than from the last
  frame it let through.
- A capture's tick loop follows a rate change instead of only reporting one.
- Each segment of a segmented recording starts its timeline at zero.
- WebRTC reads VVC's own NAL header, and accepts an avcC that carries no
  parameter sets.
