# metal_detect

`FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector -> AppSink`:
finds objects in a file's pictures with the pictures fitted on the GPU —
VideoToolbox decodes, a Metal kernel fits each picture into the model's input
where it is, Core ML runs the model on the GPU or the Neural Engine — and
prints what each picture carries on, then how fast it went. The macOS
counterpart of [`cuda_detect`](../cuda_detect).

With `--out boxes.mp4`, a Tee after the detector also records each picture
with what was found drawn on it, still on the GPU: `MetalDetectionOverlay ->
Queue -> VideoToolboxEncoder -> FileMuxer`. The overlay draws on copies, so
the printing branch beside it is handed the pictures as they were. Labels are
drawn in Arial, which macOS ships.

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26 — of the stock weights: the boxes are named with COCO's 80 classes,
whatever the model says, so an export that lost its class names still has
them. It is built with `ort-coreml`, on an Apple silicon Mac.

```sh
cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 \
  [--out boxes.mp4] [--pictures N]
```

The file's video has to be one VideoToolbox decodes to NV12 — 8-bit H.264 or
HEVC, say: a 10-bit one is refused as the pipeline is wired, the decoder's
P010 being a layout the detector does not take. `--pictures` stops it after
about that many.

On an M5, YOLOv10n over a 720p H.264 file runs at about 220 pictures a
second, decoding included, and at 150 to 200 when it is also drawn on and
encoded.
