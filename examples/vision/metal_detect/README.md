# metal_detect

`FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector -> AppSink`:
finds objects in a file's pictures with the pictures fitted on the GPU —
VideoToolbox decodes, a Metal kernel fits each picture into the model's input
where it is, Core ML runs the model on the GPU or the Neural Engine — and
prints what each picture carries on, then how fast it went. The macOS
counterpart of [`cuda_detect`](../cuda_detect).

With `--track`, an `ObjectTracker` after the detector numbers each object,
the same on every picture it is followed through, and with `--interval N` the
detector lets N pictures by between two it looks at while the tracker puts
where it expects each object on them — DeepStream's `interval`, as
[`cuda_track`](../cuda_track) measures it. `--confirm 1` numbers a new object
when it is first seen rather than on its second sighting, which a long
interval needs. `--visual` follows each object by how it looks as well as by
its motion — a correlation filter on the pixels around it, read from the
VideoToolbox picture where it is. Each implies `--track`.

`--classifier imagenet.onnx` puts a `MetalOrtClassifier` after the tracker —
DeepStream's secondary inference — so that each object is also named by a
second model, once per object it follows, and labelled with it:
`car #5 87% | minivan 48%`. `--classifier-labels` names its classes, one a
line, and `--classify 2,5,7` limits it to the detector's classes listed. The
classifier is taken for an ImageNet one, as the public models are, such as the
ONNX model zoo's MobileNetV2. It implies `--track`.

`--line X1,Y1,X2,Y2`, in fractions of the picture and once per line, puts an
`ObjectAnalytics` after the tracker that counts the objects crossing each line
— forward from its left to its right as seen from its start, so a line drawn
left to right counts what moves down — and prints each crossing as it happens
and the totals at the end. It implies `--track`.

With `--out boxes.mp4`, a Tee at the end also records each picture with what
was found drawn on it, still on the GPU: `MetalDetectionOverlay -> Queue ->
VideoToolboxEncoder -> FileMuxer`, each object tracked in a colour of its own
and labelled with its number, and with `--line` each line drawn across it,
labelled with its crossings each way. The overlay draws on copies, so the
printing branch beside it is handed the pictures as they were. Labels are
drawn in Arial, which macOS ships.

A phone's portrait recording, stored on its side, is looked at and labelled
the right way up, and recorded saying it is turned, as the file does.

`--hide person=blur` hides a class in the recording instead of boxing it:
`mosaic`, `blur` or `fill` (black), a mosaic where none is said, and once
for each class to hide, by the name the model gives it. `,ellipse` after it
— `--hide face=mosaic,ellipse` — hides the ellipse inside each box rather
than the whole box. On the people clip,
YOLO11n records at the same rate hiding people as boxing them, about 235
pictures a second on an M5 MacBook Air once both are measured in turns.
Hide with the detector on every picture, as
[`cuda_track`](../cuda_track/README.md) explains: a picture the tracker
fills in may miss someone only just come in.

The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or YOLOv10
and YOLO26 — of the stock weights: the boxes are named with COCO's 80 classes,
whatever the model says, so an export that lost its class names still has
them. Or with `--retinaface` it is a RetinaFace export (biubug6's
`Pytorch_Retinaface`, MobileNet or ResNet), whose faces are labelled `face`
and carry five points each: `--retinaface --out faces.mp4 --hide
face=mosaic,ellipse` hides every face. With `--tiles` each picture is looked
at in six overlapping tiles as well as whole, the seven run as one batch:
small, distant faces are found that the whole picture shrunk to the model
loses — on the 1080p people clip RetinaFace MobileNet 0.25 found 38 faces
where it found 7 whole — at 42 pictures a second with YOLO11n on an M5
against 210. It is built with `ort-coreml`, on an Apple silicon Mac.

```sh
cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 \
  [--retinaface] [--tiles] [--track] [--interval N] [--confirm N] [--visual] [--line X1,Y1,X2,Y2]... \
  [--classifier imagenet.onnx [--classifier-labels classes.txt] [--classify 2,5,7]] \
  [--out boxes.mp4 [--hide CLASS[=mosaic|blur|fill][,ellipse]]...] [--pictures N]
```

A 10-bit file — an iPhone's HDR — is decoded to P010, looked at through an
SDR copy, and recorded as it came: HEVC Main 10, BT.2020 and HLG or PQ,
`hvc1`. Nothing is drawn on it; `--hide` still hides in its ten bits.
`--pictures` stops it after about that many.

On an M5, YOLOv10n over a 720p H.264 file runs at about 220 pictures a
second, decoding included, and at 150 to 200 when it is also drawn on and
encoded; with `--interval 4`, detecting one picture in five, at about 520.
