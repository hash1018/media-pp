# metal_detect

`FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector -> AppSink`:
finds objects in a file's pictures with the pictures fitted on the GPU —
VideoToolbox decodes, a Metal kernel fits each picture into the model's input
where it is, Core ML runs the model on the GPU or the Neural Engine — and
prints what each picture carries on, then how fast it went. The macOS
counterpart of [`cuda_detect`](../../cuda/cuda_detect).

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26. It is built with `ort-coreml`, on an Apple silicon Mac.

```sh
cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 [pictures]
```

The file's video has to be one VideoToolbox decodes to NV12 — 8-bit H.264 or
HEVC, say: a 10-bit one is refused as the pipeline is wired, the decoder's
P010 being a layout the detector does not take. `pictures` stops it after
about that many.

On an M5, YOLOv10n over a 720p H.264 file runs at about 220 pictures a
second, decoding included.
