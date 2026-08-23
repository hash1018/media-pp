# seek_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler -> GPU upload ->
Renderer`, same chain as `sw_decode_render`, plus a terminal prompt that
reads timestamps and calls
`Pipeline::seek` with them while the window is open — proves `seek` actually
changes what's on screen, not just that it compiles. The same prompt also
exposes `pause`/`resume`. Windows uses D3D12; Linux uses CUDA/Vulkan.

```sh
cargo run -p seek_render -- path/to/video.mp4
```

Once the window is open, type commands on stdin:

```text
pause
resume
seek 30
seek 1:15
q
```
