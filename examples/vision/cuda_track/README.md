# cuda_track

`FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> ObjectTracker`:
objects found on the GPU and followed from picture to picture, each keeping
its number, with the detector looking at only some of the pictures
(`--interval N` lets N by between two it looks at) and the tracker putting
where it expects each object on the rest — DeepStream's `interval`.

`--out tracked.mp4` records it through `CudaDetectionOverlay`, each object in
a colour of its own and labelled with its number, and encodes it on NVENC.
`--confirm 1` numbers a new object when it is first seen rather than on its
second sighting, which a long interval needs. `--visual` follows each object
by how it looks as well as by its motion (`TrackerOptions::visual`).

`--classifier imagenet.onnx` puts a `CudaOrtClassifier` after the tracker —
DeepStream's secondary inference — so that each object is also named by a
second model, once per object it follows, and labelled with it:
`car #5 0.48 | minivan`. `--classifier-labels` names its classes, one a line,
and `--classify 2,5,7` limits it to the detector's classes listed. The
classifier is taken for an ImageNet one, as the public models are, such as
the ONNX model zoo's MobileNetV2 — which, trained on pictures from the side,
names a car seen from above a minivan and a bicycle from above a shower cap:
what a classifier says is only as good as what it was trained on.

```sh
cargo run --release -p cuda_track -- model.onnx video.mp4 --interval 2 --confirm 1 \
  --classifier mobilenetv2-12.onnx --classifier-labels imagenet_classes.txt \
  --classify 1,2,3,5,7 --out classified.mp4
```

`--eval 1,2,4,9` measures how good the filled-in pictures are: it runs the
file with the detector on every picture as the reference, then once per
interval with the tracker, and on each picture let by compares the boxes the
tracker expected with the reference's confident ones — beside holding the
last detected boxes still, which is filling in without a motion model.

```sh
LD_LIBRARY_PATH=/path/to/cuda13-cudnn9-tensorrt10/lib \
  cargo run --release -p cuda_track -- model.onnx video.mp4 --interval 2 --out tracked.mp4
LD_LIBRARY_PATH=/path/to/cuda13-cudnn9-tensorrt10/lib \
  cargo run --release -p cuda_track -- model.onnx video.mp4 --eval 1,2,4,9 --confirm 1
```

Measured with YOLOv10n on an RTX 3050, `--confirm 1`, as mean IoU and the
share of reference objects overlapped by half or more, on the pictures let
by — the tracker by motion alone against holding still:

| interval | people walking, 12 fps | people, bicycles, cars, 12 fps |
|---|---|---|
| 1 | 0.837, 96.2% / 0.819, 96.2% | 0.779, 89.6% / 0.723, 88.9% |
| 2 | 0.779, 91.6% / 0.731, 89.6% | 0.728, 86.0% / 0.645, 83.8% |
| 4 | 0.707, 86.9% / 0.633, 73.8% | 0.638, 77.5% / 0.544, 62.9% |
| 9 | 0.469, 50.6% / 0.436, 40.7% | 0.309, 29.0% / 0.301, 27.8% |

And by motion alone against `--visual`, which keeps the short gaps and
holds on over the long ones:

| interval | people walking: motion / visual | people, bicycles, cars: motion / visual |
|---|---|---|
| 1 | 0.837, 96.2% / 0.836, 96.2% | 0.779, 89.6% / 0.776, 89.6% |
| 2 | 0.779, 91.6% / 0.778, 92.0% | 0.728, 86.0% / 0.742, 88.8% |
| 4 | 0.707, 86.9% / 0.726, 90.6% | 0.638, 77.5% / 0.661, 84.0% |
| 9 | 0.469, 50.6% / 0.606, 74.3% | 0.309, 29.0% / 0.542, 73.0% |

The clips are Intel's `people-detection.mp4` and
`person-bicycle-car-detection.mp4` sample videos. On a 30 fps concert video,
where people mostly stand, the two are within a thousandth of each other. A
gap of a second or more — interval 9 at 12 fps — is past what carrying the
motion on can follow, and where following by look earns its cost: on that
concert, a hundred people at 1080p, `--visual` took 24 seconds where motion
alone took 8.
