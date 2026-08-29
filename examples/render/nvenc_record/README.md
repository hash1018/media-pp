# nvenc_record

`AppSource -> SwScaler(NV12) -> upload -> NVENC -> Mp4Muxer`: encodes
GPU-resident frames on the GPU's own NVENC block straight into a playable
`.mp4`, with no CPU readback anywhere after the upload.

The contrast with a software tail is the point: `gpu_video_compositor`'s
recording branch has to run `Download -> SwScaler -> SwEncoder`, pulling every
frame back over PCIe and converting and encoding it on the CPU, because
`SwEncoder` has no GPU input path. Here the frame stays on the GPU from the
upload onward.

`cuda_record` is the same graph on the CUDA backend, in its own crate because
CUDA is a vendor backend rather than a platform one and runs on Windows too.

Needs an NVIDIA GPU and an ffmpeg build with NVENC; `D3d11VideoEncoder` reports
a typed error rather than panicking on anything else. No window and no media
file are involved, so this runs headless.

```sh
cargo run -p nvenc_record -- [output.mp4] [seconds]
```
