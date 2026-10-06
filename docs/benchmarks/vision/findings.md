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

**Unsure boxes.** A detector before a tracker keeps boxes down to 0.1,
for the tracker to match a partly hidden object with, and the tracker
hands on those it numbered nothing for. A classifier classified each of
them again on every picture, as an object with no number is. It now
leaves boxes under `OrtClassifierOptions::min_detection_score`, 0.25 by
default, and keeps an answer it gave a followed object whatever its box
scores later. Classifying every detection against those at 0.25 or more,
in turns, three runs each:

| Configuration | Every detection | 0.25 or more | Change |
|---|---:|---:|---:|
| 1 stream, every 30 pictures | 512 | 524 | +2% |
| 4 streams batched, every 30 pictures | 625 | 647 | +3% |
| 1 stream, every picture | 470 | 470 | 0% |
| E9, 1, 4 and 8 streams | 441–442 | 441–442 | 0%, NVENC's limit |

**The batch its engine is fastest at.** A TensorRT engine built for a
range of batches is quickest at the one it was told to optimise for. The
classifier's engine was built for 8, but a picture has one to three
objects to classify. MobileNetV2 alone, milliseconds a run (`model_only
--opt N --max 32`, the median of two runs):

| Built for | 1 | 2 | 3 | 4 | 8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.308 | 0.490 | 0.657 | 0.816 | 1.496 |
| 2 | 0.325 | 0.454 | 0.610 | 0.776 | 1.422 |
| 4 | 0.350 | 0.478 | 0.609 | 0.741 | 1.405 |
| 8 | 0.361 | 0.490 | 0.622 | 0.758 | 1.341 |
| 16 | 0.399 | 0.528 | 0.657 | 0.792 | 1.373 |
| 32 | 0.402 | 0.532 | 0.666 | 0.800 | 1.381 |

It is now built for 2. In the pipeline, classifying every object on every
picture, that is 1% (one stream, 470 → 476) and 0.6% (four batched,
568 → 571) — small, as a run is a few hundredths of a millisecond shorter,
but in every run.

**What it answers.** Half precision does not change what it says: on the
same boxes TensorRT fp16 gives CUDA fp32's class wherever the model is
sure, and on a BGRA picture the CPU's scores to within 0.02. Measuring
this turned up a difference between the GPU and the CPU that predates
TensorRT — [the next finding](#the-gpu-and-the-cpu-classified-differently).

## The GPU and the CPU classified differently

`it_says_on_the_gpu_what_the_cpu_says` asked that the GPU and the CPU give
the same class for at least four of six boxes. With MobileNetV2 it passed
on `sample.mp4`, where every box is the same thing, and failed on the
people clip, three of six, on CUDA's provider as much as on TensorRT. Two
things differed, and neither was the model.

- **How a box was shrunk.** The GPU makes each input pixel the mean of the
  source pixels it covers, as `5038cd66` made the detectors' fitting. The
  CPU's `SwOrtClassifier` took a bilinear sample at each input pixel's
  centre, which reads four of the dozen pixels a person's box gives each
  one: the same defect the GPU's fitting had before that commit. It now
  takes the mean over the same span the kernels do.
- **What the test handed each.** The CPU was given an RGB24 picture the
  test converted with swscale's default, BT.601; the GPU the NV12 one,
  which it reads as BT.709, as an untagged HD picture is. The test now
  hands both the very same picture.

With both, on a BGRA picture the GPU's scores are the CPU's to within
0.003 on CUDA's provider and 0.02 through TensorRT fp16. On NV12 they
differ by up to about 0.1: swscale interpolates the chroma the GPU takes
as it is, and the GPU averages Y'CbCr where the CPU averages RGB. Closing
that would take the CUDA and Metal kernels interpolating chroma too.

The answers that still differ are those the model is not sure of: on the
people clip, two boxes whose best class scores 0.06 and 0.15 out of a
thousand, where the answer turns on the last digit. The test now compares
the boxes the CPU is sure of (0.3 or more), asks the GPU for the same
class within 0.15 of the score, and asks that there be at least one. It
passes on both clips, on CUDA and through TensorRT.

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

## Engines for two batches overwrote each other

ONNX Runtime names a TensorRT engine after the model's graph and the
precision, not the batch range it was built for. Two `CudaOrtDetector`s
of one model with different `max_batch` — one stream alone and four
batched — shared a file in the engine cache, and each start found the
other's engine and built its own again: a four-stream start after a
one-stream run took 162 seconds.

The default cache now keeps an engine built for a range of batches in a
directory of that range's own (`batch-1-4-4`; the classifier's is
`batch-1-2-32`), and one built for none in the cache itself, where
engines from before still are. The timing cache stays shared, which made
the first build of a new range take 6 seconds rather than minutes.
Starting four batched and one alone in turn, three times, every start
after the first loaded in 0.3 to 0.4 seconds. A directory an application
gives in `CudaOrtDetectorOptions::engine_cache` is used as it is, so
detectors of one model with different batches want one each, as
`vision_bench --engine-cache-root` gives them.

## CUDA graphs: faster, and wrong

TensorRT's provider can capture a run as a CUDA graph and replay it,
launching the model whole rather than kernel by kernel. On the model alone
(`model_only --cuda-graph`) it is a real gain where batches are small:

| Model | Batch | Without | With | Change |
|---|---:|---:|---:|---:|
| YOLO11n | 1 | 669 | 776 | +16% |
| YOLO11n | 8 | 922 | 950 | +3% |
| MobileNetV2 | 1 | 3277 | 4029 | +23% |
| MobileNetV2 | 4 | 5370 | 5890 | +10% |

A graph replays the addresses it was captured with, so `CudaOrtDetector`
was made to keep one `IoBinding` for each number of pictures, captured
under its own `gpu_graph_id`. Comparing every picture's detections with
and without, over the 2982 pictures of the 1080p clip, 625 differed —
every other picture or so — and 2208 objects were found against 2222.

ONNX Runtime copies an input into memory of its own when it is bound,
here even when the tensor is in its own CUDA allocator's memory: a binding
kept from run to run saw only the first picture, and found nothing at
all, with or without the graph. Bound again on every run, as the detector
always has, the copy goes to one of two buffers in turn, and the graph
reads the one it was captured with. Until a model's input can be bound
where it is, a graph replays stale pictures, so the detector does not use
one. `vision_bench` reports `found`, the objects found over the whole run,
which is what told the two apart.

## INT8 pays only where the model is heavy

TensorRT runs INT8 at up to twice FP16's arithmetic on this GPU's tensor
cores. Measured on the model alone (`model_only`), two runs each, the
models quantized with ONNX Runtime's `quantize_static` — Q/DQ nodes,
weights per channel, both symmetric, calibrated by min and max over 64
pictures of the people clip, the DFL that makes box edges left in float:

| Model | Quantized | Batch 1 | Batch 8 |
|---|---|---:|---:|
| YOLO11n | FP16 | 670 | 921 |
| YOLO11n | INT8, Conv only | 474 (−29%) | 918 (0%) |
| YOLO11n, static | INT8, every op | 458 (−32%) | — |
| YOLO11s | FP16 | 401 | 522 |
| YOLO11s | INT8, Conv only | 364 (−9%) | 609 (+17%) |

A small model one picture at a time is bound by launching kernels and
moving between INT8 and FP16 at each quantized layer, not by arithmetic,
and INT8 adds conversions to it. Only a heavier model, eight pictures at
a time, keeps the tensor cores busy enough for INT8 to pay — 17% for
YOLO11s. Whether its answers hold up was not measured: `CudaOrtDetector`
has no way to ask for INT8 yet.

TensorRT's own calibration — a float model and a table of each tensor's
range, from which TensorRT picks INT8 or FP16 layer by layer, usually
the faster way — failed to build: ONNX Runtime's calibrator writes ranges
for the model's tensors, TensorRT's network has tensors of its own, and
the provider refuses an engine with any range missing (`failed to set
INT8 dynamic range`), on the dynamic model and the static alike.

Calibrating by percentile over 199 pictures ran the quantizer out of 32
GB of memory: it keeps every activation's histogram. Min and max over 64
needed under 10.

## A full label cache stopped the pipeline

Measuring the above, the 1-stream E9 run never ended: three hours on, the
GPU idle. Run under gdb, the CUDA overlay's thread had panicked at
`expect("made while the marks were")`. Since the overlays place every
mark of a picture before drawing any, a label made early in a picture
could be emptied out of the cache by one made later, once the cache
reached its 256 masks — and with a classifier, whose labels carry a score
that changes from picture to picture, it reached them in a minute. The CPU
and Metal overlays read the cache the same way without the `expect`, and
drew the lost labels as solid blocks. All three now empty the cache only
between pictures; a test fills it to one short of full and draws three
new labels, and fails with the old eviction.

That the panic stopped the run without a word was a second defect: an
element's thread that panicked left its pipeline without a `Finished` or
an `Error` on the bus, so whatever waited for either waited for good. A
queue's worker and the source's thread now catch a panic and post it as
`Error::Panicked`, the failure of the element the queue fed or of the
source, once; tests make an element panic behind a queue and on the
source's thread, and fail without it.

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
