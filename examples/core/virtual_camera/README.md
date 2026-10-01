# virtual_camera

Shows a pipeline's pictures as a camera other applications open — Teams,
Zoom, a browser and Windows' Camera app list it as "media-pp Windows Virtual
Camera" — until `q` + Enter:

- with no file: `TestVideoSource -> Queue -> MfVirtualCamera`;
- with one: `FileDemuxer -> SwDecoder -> Queue -> Pacer -> MfVirtualCamera`,
  looping, so the file plays at its own speed for as long as this runs.

`MfVirtualCamera` converts each picture to the size the reading application
picked. Windows 11 only, and the camera's DLL must be installed once for the
machine — see [`vcam`](../../../vcam/README.md); without it this says so and
exits.

```sh
cargo run -p virtual_camera -- [video.mp4]
```
