# cuda_record

`AppSource -> SwScaler(NV12) -> CudaUpload -> Queue -> CudaEncoder ->
FileMuxer`: encodes GPU-resident frames on the GPU's own NVENC block straight
into a playable `.mp4`, with no CPU readback after the upload.

The contrast with a software tail is the point: a recording branch ending in
`SwEncoder` has to run `CudaDownload -> SwScaler -> SwEncoder`, pulling every
frame back over PCIe and converting and encoding it on the CPU.

CUDA is a vendor backend, not a platform one, so this builds and runs the
same way on Windows and Linux; `nvenc_record` is the D3D11 counterpart. It
needs an NVIDIA GPU and an FFmpeg with NVENC — `CudaEncoder` reports a typed
error otherwise — and no window or media file, so it runs headless.

```sh
cargo run -p cuda_record -- [output.mp4] [seconds]
```

It records 1280x720 at 30 fps to `cuda_record.mp4` for 5 seconds by default.
