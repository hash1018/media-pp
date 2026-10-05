# detect

`FileDemuxer -> SwDecoder -> Queue -> SwOrtDetector -> SwScaler (960x540
RGB24) -> AppSink`, drawing what the detector found onto each picture and
showing it in a plain window. The boxes are drawn on the CPU and blitted into
a `winit` window through `softbuffer`; no GPU renderer is involved.

The detector hands each decoded picture on as it came, carrying the
`Detections` it found as metadata; the scaler after it carries them on, and
the sink reads them off the picture it draws. Their boxes are fractions of
the picture, so they fit it at whatever size it is shown.

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26; labels are read as COCO's. There is no `Pacer`, so frames show as
fast as decoding and inference allow, not at playback speed.

```sh
cargo run -p detect -- path/to/model.onnx path/to/video.mp4
```
