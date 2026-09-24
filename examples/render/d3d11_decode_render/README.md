# d3d11_decode_render

`FileDemuxer -> D3d11Decoder -> Queue -> Pacer -> D3d11WindowRenderer`:
decodes on the GPU via D3D11VA hardware acceleration and shows the frames in
a window at real playback speed, without ever copying the decoded pixels back
to system memory — the renderer draws straight from the decoder's own D3D11
texture. The D3D11 sibling of `hw_decode_render` (which does the same thing
via D3D12VA instead).

The window is the renderer's own: `D3d11WindowRenderer::open` opens it on a
thread of its own, the way a GStreamer video sink does, and reports what
happens to it. Space pauses and resumes; F or a double click fills the screen
and puts it back; Escape or closing the window stops. Its `WindowControl`
changes it too: the title shows where playback is. The one device every
element shares is a `D3d11Gpu`.

`D3d11Decoder` never touches FFmpeg's `hw_frames_ctx`/`AVD3D11VAFramesContext`
itself — only `hw_device_ctx` and `get_format` — so libavcodec's own internal
D3D11VA hwaccel init handles frames-context allocation entirely inside
already-correct C code, unlike the hand-mirrored struct path that crashed
when this project tried to drive it manually. This example is what actually
proves that's safe on real hardware.

```sh
cargo run -p d3d11_decode_render -- path/to/video.mp4
```
