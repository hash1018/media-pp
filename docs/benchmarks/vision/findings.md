# What the vision benchmarks turned up

Things learned while measuring [linux-rtx3050.md](linux-rtx3050.md), each
with the measurement that settled it. Some changed media-pp; some are
there to know before reading a number, or to fix later.

## The classifier

E7 showed `CudaOrtClassifier` holding one stream to 228 pictures a second
where the detector and tracker before it ran at 545. MobileNetV2 alone
runs at 887 pictures a second on CUDA's provider (E0), so the model's own
cost does not explain it, and the GPU was half idle (49%).

**Where the time went.** Timing each part of a classification inside the
pipeline — one stream, every object classified on every picture, about
two objects a run:

| Part | Per run |
|---|---:|
| Cutting the objects out into the model's input (kernels) | 0.009 ms |
| Waiting for those kernels (`cuCtxSynchronize`) | 0.06 ms |
| `Session::run`, MobileNetV2 on CUDA's provider | 12.7 ms |

`Session::run` is all of it, and is six times what the same model takes
alone at that batch (2.1 ms). The same model run in a second process while
a detector pipeline runs takes 5.9 ms a batch of two instead of 2.1: CUDA's
provider runs MobileNetV2 as about a hundred small kernels, each of which
waits its turn behind TensorRT's large ones on the same GPU. The rest of
the gap is the batch changing size from picture to picture, as many as
there are objects.

**What was done.** The classifier ran on CUDA alone because its batch is
however many objects a picture has, and a TensorRT engine built for one
shape is rebuilt for another. But TensorRT takes a range of shapes as
well as one, as `CudaOrtDetector` already tells it for a `StreamMux`'s
batches. Built with `ort-tensorrt`, the classifier now runs through
TensorRT in half precision, its engine built once for every batch from 1
to 32 and kept beside the detector's, and falls back to CUDA where
TensorRT cannot run.

Measured as two arms inside one run, three runs each (1080p, YOLO11n
on TensorRT fp16, MobileNetV2 after the motion tracker):

| Configuration | CUDA classifier | TensorRT classifier | Change |
|---|---:|---:|---:|
| 1 stream, detector and tracker, no classifier | 545 | 542 | 0% |
| 1 stream, each object classified every 30 pictures | 228 | 514 | +125% |
| 1 stream, each object classified every picture | 139 | 473 | +240% |
| 4 streams batched, every 30 pictures | 418 | 624 | +50% |
| 4 streams batched, every picture | 272 | 554 | +104% |
| E9, 1 stream: interval 2, tracker, classifier, overlay, NVENC | 292 | 441 | +51% |
| E9, 4 streams | 393 | 443 | +13% |
| E9, 8 streams | 421 | 443 | +5% |

One stream with a classifier now runs at 94% of the rate without one, and
the GPU is busy again (50% → 89%). Every E9 row now meets NVENC's limit
(448), which is what holds them all at 443: on this GPU the encoder, not
media-pp, is the last limit left. The classifier's first start builds its
engine, about 50 seconds for MobileNetV2; later starts load it.

**What it answers.** On the people clip, picture 150, six boxes across it:
TensorRT fp16 and CUDA fp32 give the same class for five of the six. Both
agree with the CPU's `SwOrtClassifier` on only three — see
[the next finding](#the-gpu-and-the-cpu-classify-differently).

## The GPU and the CPU classify differently

`it_says_on_the_gpu_what_the_cpu_says` asks that the GPU and the CPU give
the same class for at least four of six boxes. With MobileNetV2 it passes
on `sample.mp4`, where every box is the same thing, and fails on the
people clip, three of six, **on CUDA's provider as much as on TensorRT** —
so it predates the change above.

The two cut the boxes differently. The GPU cuts on whole 2×2 blocks of an
NV12 picture and shrinks each input pixel from the mean of the pixels it
covers. The CPU scales the RGB picture with swscale. On a soft picture —
this clip is 432p made 1080p — an ImageNet model is unsure of every box,
and unsure answers flip on small differences in the input. The test's
threshold was set on a picture where the answers were sure. Whether the
GPU's cut should match the CPU's more closely, or the test should use a
picture where the answers are sure, is open.

## Waiting on the whole context

The first suspect for the classifier was `CudaDriver::synchronize`, which
calls `cuCtxSynchronize`. That waits for everything on the context,
including the detector's TensorRT run in flight on ONNX Runtime's own
stream. Changing it to wait on the legacy default stream alone
(`cuStreamSynchronize(NULL)`), where this crate's kernels and FFmpeg's
copies run, would keep every ordering it was there for and drop that one.

Measured as two arms inside one run (the full matrix, three runs each), it
changed nothing outside the noise: E7's classifier rows were 228 and 228,
139 and 140; E9's 290 and 290, 392 and 394, 421 and 422. The timing above
shows why: the wait takes 0.06 ms. The change was not kept, since a
narrower wait is a risk to correctness that nothing measured pays for.

## Engines for two batches overwrite each other

ONNX Runtime names a TensorRT engine after the model's graph and the
precision, not the batch range it was built for. Two `CudaOrtDetector`s
of one model with different `max_batch` — one stream alone and four
batched — share a file in the engine cache, and each start finds the
other's engine and builds its own again, for minutes. `vision_bench`
gives each model, precision and batch a directory of its own with
`--engine-cache-root`. `CudaOrtDetectorOptions::engine_cache` lets an
application do the same, but the default puts every engine in one
directory, so the default could take the batch range into the path
itself.

## Two builds of one engine differ

YOLO11n at batch 1 ran at 587 pictures a second through an engine built
early in the day, and at 673 through one built later for the same model,
precision and batch: TensorRT chooses its kernels by timing them as it
builds, and the choice varies. Numbers from different engine builds
differ by more than many of the changes worth measuring. The arms of a
comparison should share engines, as `bench.py`'s do.

## The fixed-function engines are the ceiling

On this GPU, NVDEC decodes 1080p H.264 at about 780 pictures a second for
one stream and 855 for several, and NVENC encodes it at 448. With more
than one stream, or any encoding, these set the limit rather than the
model or media-pp. DeepStream drives the same NVDEC and NVENC and would
meet the same limits. The rows that reach them are E4's eight streams
(NVDEC 99%), E5's intervals (NVDEC 88–100%) and E8 and E9's encoding
(NVENC 93–100%).
