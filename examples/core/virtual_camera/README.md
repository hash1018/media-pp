# virtual_camera

Shows a pipeline's pictures as a camera other applications open — a
browser, Zoom, Teams, Windows' Camera app — until `q` + Enter:

- with no file: `TestVideoSource -> Queue -> camera`;
- with one: `FileDemuxer -> SwDecoder -> Queue -> Pacer -> camera`,
  looping, so the file plays at its own speed for as long as this runs.

The camera is the platform's:

- **Windows**: `MfVirtualCamera`, listed as "media-pp Windows Virtual
  Camera", which converts each picture to the size the reading application
  picked. Windows 11 only, and the camera's DLL must be installed once for
  the machine — see [`vcam`](../../../vcam/README.md).
- **Linux**: `V4l2VirtualCamera`, writing 1280x720 at 30 fps into the first
  free v4l2loopback device, listed under the label the module was loaded
  with. Install your distribution's v4l2loopback package and load it first:

  ```sh
  sudo modprobe v4l2loopback exclusive_caps=1 card_label=media-pp
  ```

  `exclusive_caps=1` is what browsers need to list the device.

Without its camera this says what is missing and exits.

```sh
cargo run -p virtual_camera -- [video.mp4]
```
