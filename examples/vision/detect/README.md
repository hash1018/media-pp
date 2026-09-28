# detect

`FileDemuxer -> SwDecoder -> Queue -> SwScaler (640x640 RGB24) ->
OrtDetector`, drawing every detection onto the same 640x640 frame the
detector saw and showing it in a plain window. The boxes are drawn on the CPU
and blitted into a `winit` window through `softbuffer`; no GPU renderer is
involved.

The model is an Ultralytics YOLOv8 or YOLOv11 ONNX export with a 640x640
input; labels are read as COCO's. There is no `Pacer`, so frames show as fast
as decoding and inference allow, not at playback speed.

```sh
cargo run -p detect -- path/to/model.onnx path/to/video.mp4
```
