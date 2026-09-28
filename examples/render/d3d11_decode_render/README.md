# d3d11_decode_render

`FileDemuxer -> D3d11Decoder -> Queue -> Pacer -> D3d11WindowRenderer`:
decodes on the GPU via D3D11VA hardware acceleration and shows the frames in
a window at real playback speed, without ever copying the decoded pixels back
to system memory — the renderer draws straight from the decoder's own D3D11
texture. The D3D11 sibling of `hw_decode_render`, which decodes through
`VideoDecodeBin` onto D3D12 with a software fallback.

The window is the renderer's own: `D3d11WindowRenderer::open` opens it on a
thread of its own, the way a GStreamer video sink does, and reports what
happens to it. Space pauses and resumes; F or a double click fills the screen
and puts it back; Escape or closing the window stops, and so does the end of
the file or an element's failure. Its `WindowControl` changes it too: the
title shows where playback is. The one device every element shares is a
`D3d11Gpu`.

```sh
cargo run -p d3d11_decode_render -- path/to/video.mp4
```
