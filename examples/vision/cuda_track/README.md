# cuda_track

`FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> ObjectTracker`:
objects found on the GPU and followed from picture to picture, each keeping
its number, with the detector looking at only some of the pictures
(`--interval N` lets N by between two it looks at) and the tracker putting
where it expects each object on the rest — DeepStream's `interval`.

`--out tracked.mp4` records it through `CudaDetectionOverlay`, each object in
a colour of its own and labelled with its number, and encodes it on NVENC.
`--confirm 1` numbers a new object when it is first seen rather than on its
second sighting, which a long interval needs.

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
by — the tracker against holding still:

| interval | people walking, 12 fps | people, bicycles, cars, 12 fps |
|---|---|---|
| 1 | 0.837, 96.2% / 0.819, 96.2% | 0.779, 89.6% / 0.723, 88.9% |
| 2 | 0.779, 91.6% / 0.731, 89.6% | 0.728, 86.0% / 0.645, 83.8% |
| 4 | 0.707, 86.9% / 0.633, 73.8% | 0.638, 77.5% / 0.544, 62.9% |
| 9 | 0.469, 50.6% / 0.436, 40.7% | 0.309, 29.0% / 0.301, 27.8% |

The clips are Intel's `people-detection.mp4` and
`person-bicycle-car-detection.mp4` sample videos. On a 30 fps concert video,
where people mostly stand, the two are within a thousandth of each other. A
gap of a second or more — interval 9 at 12 fps — is past what carrying the
motion on can follow.
