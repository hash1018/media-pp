# detect

`FileDemuxer -> SwDecoder -> Queue -> SwOrtDetector -> SwScaler (960x540
RGB24) -> SwDetectionOverlay -> AppSink`, drawing what the detector found
onto each picture and showing it in a plain window. Everything is on the CPU,
blitted into a `winit` window through `softbuffer`; no GPU renderer is
involved.

The detector hands each decoded picture on as it came, carrying the
`Detections` it found as metadata; the scaler after it carries them on, and
the overlay draws them onto the scaled picture. Their boxes are fractions of
the picture, so they fit it at whatever size it is. Labels are drawn where a
system font (DejaVu Sans, or Arial) is found.

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26; what it found is printed with COCO's names where the model has
none of its own. There is no `Pacer`, so frames show as
fast as decoding and inference allow, not at playback speed.

```sh
cargo run -p detect -- path/to/model.onnx path/to/video.mp4
```
