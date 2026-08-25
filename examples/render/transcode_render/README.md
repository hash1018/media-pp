# transcode_render

`TestVideoSource -> SwEncoder -> SwDecoder -> Pacer -> SwScaler -> GPU upload
-> Renderer`: encodes a synthetic moving-gradient stream (via `libopenh264`) and decodes it straight
back — no file, camera, or container/mux involved at all — presented in a
native window at real playback speed. Proves `SwEncoder`'s `Packet`s are
actually valid, decodable H.264 (not just "avcodec_open2 succeeded"): if the
round trip corrupted anything, the gradient would visibly glitch or freeze
instead of scrolling smoothly.

This example keeps a `Pacer` after the encode/decode round trip. The source
itself is already paced accurately enough for direct rendering, but the
encoder and decoder add their own buffering and per-frame variance; this
particular chain has not been validated without the final clock-anchored
pacing stage. Windows presents through D3D12; Linux uploads the decoded frames
to CUDA and presents through Vulkan.

```sh
cargo run -p transcode_render
```
