# screen_record_nvenc

`DxgiCaptureSource` (GPU mode) `-> D3d11NvencEncoder -> Mp4Muxer`: records the
desktop into a playable `.mp4` without the pixels ever touching the CPU or
going through a separate CPU color-conversion stage.

The contrast with `screen_record` is the whole point. That example runs
`DxgiCaptureSource` (CPU mode) `-> SwScaler -> SwEncoder`: every frame is mapped
back to system memory, converted BGRA->YUV420P by libswscale, and encoded on
the CPU. Here capture writes a GPU-resident BGRA texture, NVENC consumes that
texture directly — `D3d11NvencInputFormat::Bgra`, since NVENC does its own
color conversion inside the encode block — and only the compressed packets
ever reach the CPU. There is no `SwScaler` and no `D3d11Download` in this graph
at all.

Needs an NVIDIA GPU and an ffmpeg build with NVENC. `Pipeline::finish` sends
ordered EOS through the encoder and muxer so delayed frames are drained
before the MP4 trailer is finalized. Windows only.

```sh
cargo run -p screen_record_nvenc -- <output.mp4> [seconds]
```
