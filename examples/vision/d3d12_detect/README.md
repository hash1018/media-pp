# d3d12_detect

Windows only. `FileDemuxer -> D3d12Decoder -> Queue -> D3d12OrtDetector ->
ObjectTracker -> AppSink`: objects found and followed from picture to picture
on any vendor's GPU, through D3D12 and ONNX Runtime's DirectML, without the
pictures leaving the GPU. Each object keeps its number; the detector looks at
one picture in `--interval N` + 1 and the tracker puts where it expects each
object on the rest. `--visual` follows each object by how it looks as well as
by its motion, on the GPU. It prints what is in the picture once a second of
the video, then how many pictures went through, how fast, and how many
objects were followed.

`--out tracked.mp4` records it as well, through a `Tee` after the tracker:

```text
… -> ObjectTracker -> Tee ┬ AppSink
                          └ D3d12DetectionOverlay -> Queue -> D3d12VideoEncoder -> FileMuxer
```

each object in a colour of its own and labelled with its number, encoded to
HEVC on the GPU in the file's own colours — HEVC because it is what FFmpeg 8.0
encodes through D3D12. `--hide person=blur` hides a class there instead of
boxing it — `mosaic`, `blur` or `fill`, a mosaic where none is said, the
ellipse inside each box with `,ellipse` after it — and may be given once for
each class. Hide with the detector on every picture, `--interval 0`, so an
object just come into the picture is hidden from the first one it is found
on. The recording is of 8-bit video: a 10-bit file decodes to P010, which the
encoder does not take.

A phone's portrait recording, stored on its side, is looked at and labelled
the right way up, and recorded saying it is turned, as the file does.

```sh
cargo run --release -p d3d12_detect -- yolov10n.onnx video.mp4 --interval 2 --visual --out tracked.mp4
cargo run --release -p d3d12_detect -- yolov10n.onnx video.mp4 --out hidden.mp4 --hide person=blur,ellipse
```

## The model

Any YOLO exported to ONNX — YOLOv8 and YOLO11, or YOLOv10 and YOLO26 — at
**opset 20 or lower**, and fastest with half-precision weights. DirectML hands
the operators of a later opset to the CPU, and ONNX Runtime does not say so;
the detector warns of it in the log when it is made. With Ultralytics:

```sh
yolo export model=yolov10n.pt format=onnx opset=20
```

On a 20-second 1080p clip and an RTX 3050, with the detector on every
picture and nothing recorded, YOLOv10n exported at opset 13 ran 195 pictures
a second, and YOLO11n exported at opset 22 ran 9 a second. At
`--interval 2` YOLOv10n ran 508 a second, and 165 a second following by
look and recording too.

## Running it

The build fetches ONNX Runtime's DirectML build, a few hundred megabytes,
once, and puts `DirectML.dll` beside the program; a program of your own ships
it beside itself. Logs go to `./logs`.
