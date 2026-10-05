# cuda_detect

`FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> AppSink`: finds
objects in a file's pictures without them leaving the GPU — NVDEC decodes, a
kernel fits each picture into the model's input in device memory, TensorRT
runs the model — and prints what each picture carries on, then how fast it
went.

With `--out boxes.mp4`, a Tee after the detector also records each picture
with what was found drawn on it, still on the GPU: `CudaDetectionOverlay ->
Queue -> CudaEncoder -> FileMuxer`. The overlay draws on copies, so the
printing branch beside it is handed the pictures as they were. Labels are
drawn where DejaVu Sans is found.

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26. The first run builds a TensorRT engine for the model and this GPU,
which takes minutes; later runs load it from `~/.cache/media-pp/tensorrt` in
under a second.

It is built with `ort-tensorrt`, which links CUDA 13.2, cuDNN 9.23 and
TensorRT 10.15 or newer into it: building needs them where the linker finds
them, and running where the loader does — here both through
`LD_LIBRARY_PATH`:

```sh
LD_LIBRARY_PATH=/path/to/cuda13-cudnn9-tensorrt10/lib \
  cargo run --release -p cuda_detect -- path/to/model.onnx path/to/video.mp4 \
    [--out boxes.mp4] [--pictures N]
```

On an RTX 3050, YOLOv10n over a 1080p H.264 file runs at about 530 pictures a
second, decoding included, and about 420 when it is also drawn on and
encoded.
