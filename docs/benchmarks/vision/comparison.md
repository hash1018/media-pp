# Vision benchmarks: Linux, Windows and macOS side by side

The same matrix on three systems, read together: what each operating
system costs, where the hardware is the limit, and which way of building
the pipeline suits each. Every number is the median of the page it comes
from — [linux-rtx3050.md](linux-rtx3050.md),
[windows-rtx3050.md](windows-rtx3050.md) and
[macos-m5-air.md](macos-m5-air.md) — and the Linux column is the one the
Windows page sets beside its own, measured with the classifier through
TensorRT.

## What is being compared

| | Linux | Windows | macOS |
|---|---|---|---|
| Machine | i5-12400F, 32 GB, RTX 3050 8 GB (115 W) | the same machine | M5 MacBook Air, 24 GB, no fan |
| Model runs on | TensorRT fp16 | TensorRT fp16 | Core ML, on the GPU as it chooses |
| Decode, encode | NVDEC, NVENC | NVDEC, NVENC | VideoToolbox |
| Driver model | — | WDDM | — |

Linux and Windows are one machine under two operating systems, so their
difference is the system, its driver and its libraries. macOS is other
hardware of another class — a fanless laptop chip against a 115 W desktop
GPU — measured at the rate it holds once warm, which fell by up to 40%
over its two-hour run ([macos-m5-air.md](macos-m5-air.md#read-every-table-as-sustained-throttled-performance)).
Read the macOS column as the same pipeline on other hardware, not as a
contest of operating systems. Its clips were also lighter to decode: the
same pictures at 2.9 Mbit/s for 1080p where the others' are 8.

Pictures a second, every stream together, YOLO11n at 1080p unless said.

## The model alone (E0)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| One picture at a time | 673 | 509 | 295 |
| Batch of 8 | 924 | 909 | 345 |
| Gain from batching | +37% | +79% | +17% |
| YOLO11s, one at a time | 398 | 338 | 121 |
| YOLOv10n, one at a time | 646 | 449 | 273 |

The RTX 3050 does two to three times the M5's work. What differs between
the systems is what a batch is worth: Windows loses a quarter of the model's
rate one picture at a time and gets it all back in a batch — a cost paid on
each submission to the GPU, as under the WDDM driver — while the M5 gains
little from a batch at all. Held to its Neural Engine, the M5 runs YOLO11n
at 76.

## Decoding (E1, E4)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| Hardware, one stream | 779 | 640 | 418 |
| Hardware, eight streams | 855 | 849 | 1928 |
| Software, one stream | 255 | 248 | 563 |

NVDEC is one engine that two streams already fill, whatever the system.
VideoToolbox decodes one stream no faster than software, and eight at more
than twice NVDEC's rate. The macOS software figure is helped by its lighter
clip and the M5's cores, and is not comparable.

## Who runs the model (E2)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| CPU | 47 | 32 | 49 |
| GPU, one stream | 544 | 407 | 244 |
| GPU against CPU | ×11.6 | ×12.7 | ×5.0 |

ONNX Runtime on the CPU is a third slower on Windows. On the M5 the CPU
detector keeps up with the desktop's on fewer cores (3.3 against 4.3).

## Streams: a detector each, or one batching them (E4)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| 1 stream | 541 | 408 | 146 |
| 1 stream through a mux | 541 | 333 | 130 |
| 4, a detector each | 676 | 477 | 211 |
| 4, batched | 671 | 670 | 186 |
| 8, a detector each | 657 | 426 | 207 |
| 8, batched | 705 | 714 | 180 |
| **8: batched against a detector each** | **+7%** | **+68%** | **−13%** |

(macOS's single stream here ran after E2's and E3's, the chip already
hot: 146 against 244 in E2.)

This is the result that changes how a pipeline should be built. On
Windows, batching is what makes several streams run at all — eight
detectors lose a third to Linux, eight batched match it. On Linux it is a
small gain. On the M5 it is a loss: the model gains little from a batch,
and a batching detector fits, runs and reads one after another on one
thread where several detectors overlap theirs
([findings.md](findings.md#batching-on-apple-silicon)).

## Detecting one picture in N, a tracker between (E5)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| 1 stream, every picture | 543 | 398 | 134 |
| 1 stream, one in five | 753 | 658 | 421 |
| 4 batched, every picture | 677 | 667 | 217 |
| 4 batched, one in five | 817 | 821 | 880 |

The one setting that pays everywhere. With four streams detecting one
picture in five, all three reach 800 to 900 pictures a second — the
detector no longer the limit, decoding and the rest are — and the M5 runs
level with the RTX 3050.

## Tracking by look (E6)

| Interval 0 | Linux | Windows | macOS |
|---|---:|---:|---:|
| Motion only | 542 | 407 | 165 |
| By look on the GPU | 523 | 385 | 159 |
| By look on the CPU | 331 | 226 | — |

Following by look on the GPU costs 3–5% on every system; on the CPU it
costs 40%. macOS has no CPU row: Metal follows VideoToolbox pictures on
the GPU with nothing to turn off.

## A classifier after the tracker (E7)

| 4 streams, batched | Linux | Windows | macOS |
|---|---:|---:|---:|
| Detector and tracker | 676 | 670 | 204 |
| Classifier, each object every 30 pictures | 624 | 625 | 176 |
| Classifier, every object every picture | 554 | 514 | 150 |

A classifier costs 7–23% on the RTX 3050 — the more, the more often it asks — and 14–26% on the M5.

## Drawing, encoding, and everything at once (E8, E9)

| | Linux | Windows | macOS |
|---|---:|---:|---:|
| Hardware encoder alone, 1080p | 448 | 451 | 276 |
| Detect, overlay, encode, one stream | 432 | 305 | 153 |
| Everything, 1 stream | 441 | 436 | 234 |
| Everything, 4 streams | 443 | 428 | 272 |
| Everything, 8 streams | 443 | 418 | 272 |

Everything at once — interval 2, tracker, classifier, overlay, encoding —
runs at the hardware encoder's limit on all three: NVENC's about 450,
VideoToolbox's about 275. Above one stream the operating system no longer
shows; the encoder does.

## What it means for building a pipeline

| | Linux | Windows | macOS |
|---|---|---|---|
| Several streams | Either way; batching saves CPU (0.71 cores against 2.40 for eight) | **Batch them** through a `StreamMux` | **A detector each** |
| One stream | Straight through, or a mux, alike | **Straight through**: a mux costs 18% | Straight through |
| First setting to reach for | An interval, with the motion tracker | The same | The same |
| Following by look | On the GPU (`cuda-visual-tracking`) | On the GPU | Always on the GPU |
| The ceiling with encoding | NVENC, about 450 at 1080p | NVENC, about 450 | VideoToolbox, about 275 |

- **The ceilings are the hardware's.** With enough streams, Linux and
  Windows meet at NVDEC's and NVENC's limits, and the M5 at
  VideoToolbox's encoder.
- **The operating system shows where one thing runs at a time.** Windows
  pays 15–30% on each submission to the GPU — one stream, one picture at
  a time — and nothing where the GPU's engines are full.
- **The M5 laptop does a third to two thirds of the RTX 3050's work** —
  a third for the model alone in batches, about a half for one stream,
  six tenths with everything at once — on a fanless chip at its sustained
  rate. Everything at once, it runs
  272 pictures a second: nine 1080p streams at 30 a second, analysed,
  drawn and encoded. Power was not measured; a Mac with a fan would hold
  more of its peak.
