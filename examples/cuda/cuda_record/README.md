# cuda_record

`AppSource -> SwScaler(NV12) -> CudaUpload -> CudaEncoder -> FileMuxer`: encodes
GPU-resident frames on the GPU's own NVENC block straight into a playable
`.mp4`, with no CPU readback anywhere after the upload.

The contrast with a software tail is the point: a recording branch that ends in
`SwEncoder` has to run `CudaDownload -> SwScaler -> SwEncoder`, pulling every
frame back over PCIe and converting and encoding it on the CPU. Here the frame
stays on the GPU from the upload onward.

Nothing here is platform-specific. CUDA is a vendor backend rather than a
platform one, so this crate has no per-target dependency table and no `cfg`
switch — it builds and runs the same way on Windows and Linux. `nvenc_record`
is the D3D11 counterpart for the same graph.

Needs an NVIDIA GPU and an ffmpeg build with NVENC; `CudaEncoder` reports a
typed error rather than panicking on anything else. No window and no media file
are involved, so this runs headless.

```sh
cargo run -p cuda_record -- [output.mp4] [seconds]
```
