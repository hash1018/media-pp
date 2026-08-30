# d3d11_chroma_key

A green screen keyed out on the GPU and composited over live video, with the
frame never leaving video memory between the upload and the recording branch's
download.

Three pipelines meet at one compositor:

```text
AppSource(BGRA) -> D3d11Upload -> D3d11ChromaKey -> "keyed" layer
TestVideoSource -> SwScaler(NV12) -> D3d11Upload -> "background" layer
D3d11VideoCompositor -> D3d11Download -> SwScaler(YUV420P)
    -> SwEncoder -> FileMuxer
```

`AppSource` stands in for a real external producer — a camera or capture SDK's
callback — handing over `BGRA` frames of a figure on a green backdrop.
`D3d11Upload` puts those on the GPU as BGRA rather than NV12, which is what
keeps the backdrop exactly the color `ChromaKeyMethod::Green` keys: a YUV round
trip would quantize it and leave the threshold covering for the drift.

`D3d11ChromaKey` writes that green into alpha, and the compositor blends the
result over its background layer. The keyed layer walks across the canvas as it
goes, so the recording shows the background passing behind a figure with no
green around it — where, without the key, an opaque green rectangle would cover
the background instead.

The background layer takes the NV12 route through the same `D3d11Upload`, which
is the shape a decoder-fed input has. Both routes end up as `Pixel::D3D11`
frames on one shared `ID3D11Device` and immediate context; every D3D11 element
here rejects a texture that came from a different device.

`video_compositor` is the same graph on the CPU, with `SwChromaKey` and
`SwVideoCompositor`.

No window and no media file are involved, so this runs headless.

```sh
cargo run -p d3d11_chroma_key -- [output.mp4] [seconds]
```
