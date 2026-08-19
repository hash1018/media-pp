# d3d11_decode_render

`Demux -> D3d11Decoder -> Queue -> Pacer -> Renderer`: decodes on the GPU via
D3D11VA hardware acceleration and presents the frames in a native window at
real playback speed, without ever copying the decoded pixels back to system
memory — `D3d11Renderer` draws straight from the decoder's own D3D11 texture.
The D3D11 sibling of `hw_decode_render` (which does the same thing via
D3D12VA instead).

`D3d11Decoder` never touches FFmpeg's `hw_frames_ctx`/`AVD3D11VAFramesContext`
itself — only `hw_device_ctx` and `get_format` — so libavcodec's own internal
D3D11VA hwaccel init handles frames-context allocation entirely inside
already-correct C code, unlike the hand-mirrored struct path that crashed
when this project tried to drive it manually. This example is what actually
proves that's safe on real hardware.

```sh
cargo run -p d3d11_decode_render -- path/to/video.mp4
```
