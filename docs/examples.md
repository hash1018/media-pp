# Examples

Each directory under `examples/` is a crate of its own, so platform
dependencies stay out of the library. Run one with

```sh
cargo run -p <name> -- <args>
```

One that needs a file prints its usage when run without one; no media is
checked in. A platform-only example compiles elsewhere as a stub that says
so. Each has a README with its pipeline drawn out.

| Directory | What is there |
|---|---|
| [`core`](../examples/core) | Decoding, queues, fan-out, dynamic tees, app sources and sinks, elements of your own, audio capture and playback, remuxing, GPU transcoding, HLS, RTMP, transcription, CPU compositing, a virtual camera |
| [`cuda`](../examples/cuda) | Headless CUDA recording, and GPU text compositing |
| [`render`](../examples/render/README.md) | `Player`, playback on every renderer, seeking, screen preview and recording, GPU scaling, keying, hardware encoding — with an index of its own |
| [`rtsp`](../examples/rtsp) | Publishing to an RTSP server such as MediaMTX, seeking what is published, and receiving from one — this crate does not serve RTSP itself |
| [`vision`](../examples/vision) | Scaling, and ONNX object detection with its boxes drawn — on the CPU; on CUDA pictures through TensorRT, with objects tracked between detections; and on macOS through Core ML and Metal |
| [`webrtc`](../examples/webrtc) | Loopback, a two-way video call, and recording received tracks |

Some to start with:

```sh
cargo run -p player -- path/to/video.mp4      # a file with its sound, in a window
cargo run -p probe -- path/to/video.mp4       # what a file holds, through a queue
cargo run -p transcode -- in.mp4 out.mp4      # decoded and encoded on the GPU where it can
cargo run -p screen_record_software           # the desktop to an MP4
```
