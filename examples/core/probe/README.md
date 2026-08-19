# probe

`FileDemuxer -> Queue -> PacketCounter`. Smoke test for the architecture: open
a file, inspect its streams, *then* decide how to wire the pipeline — only the
video src pad gets linked, so any other stream's packets are simply dropped
unlinked. Demuxes on the source thread, hops across an explicit `Queue`
thread boundary, and counts packets on the queue's worker thread.

```sh
cargo run -p probe -- path/to/video.mp4
```
