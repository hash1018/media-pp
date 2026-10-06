# Vision benchmarks: Windows, RTX 3050

The same experiments as [the Linux page](linux-rtx3050.md), on the same
hardware — an i5-12400F with 32 GB and an RTX 3050 — under Windows 11, so
the difference is the operating system, its NVIDIA driver and the libraries
it runs on. Measured on 2026-10-06 with media-pp at `2ff5320`, as
[method.md](method.md) describes: `bench.py`, three runs of each
configuration taking turns, the median with the lowest and highest beside
it. Each table sets the Linux median beside it, with the classifier through
TensorRT on both.

## The machine

| | |
|---|---|
| CPU, memory, GPU | As on Linux: i5-12400F, 32 GB, RTX 3050 8 GB (sm_86), 115 W limit, PCIe 4.0 ×8 |
| OS | Windows 11 Pro, build 26200, the Balanced power plan; the GPU drives the desktop |
| NVIDIA | Driver 610.88 under WDDM, CUDA runtime 13.2, cuBLAS 13.3, cuDNN 9.23.2, TensorRT 10.15.1 |
| ONNX Runtime | Through `ort` 2.0.0-rc.13, as on Linux |
| FFmpeg | 8.0.1 |
| Rust | 1.99.0, release profile |

`vision_bench` read the process's CPU time only through Unix's
`getrusage` when this was measured, so the CPU column is left out here;
it reads Windows's `GetProcessTimes` now.
## Summary

- **Where one thing runs at a time, Windows is 15–30% slower.** The model
  alone, one picture at a time, runs at 509 a second against 673; one
  stream, decode and detect, at 407 against 544; NVDEC on one 1080p
  stream at 640 against 779. A batch of eight closes the model's gap to
  2% (909 against 924), so what is lost is a cost paid on each submission
  to the GPU rather than in the GPU's work — consistent with the WDDM
  driver's, which Linux does not have, though nothing here separates it
  from the different driver and library versions.
- **Where the GPU's engines are full, Windows matches Linux.** Eight
  streams batched run at 714 against 705, four with a detector looking at
  one picture in five at 821 against 817, everything at once — interval 2,
  tracker, classifier, overlay, NVENC — at 418–436 against 441–443. NVDEC
  (849 against 855) and NVENC (451 against 448) are the same engines.
- **So batching matters far more on Windows.** Eight streams with a
  detector each run at 426; batched, at 714 — 68% more, where Linux gains
  7% (657 to 705). One stream through a mux, on the other hand, costs 18%
  (408 to 333) where Linux loses nothing.
- **ONNX Runtime on the CPU is a third slower** (32 against 47 pictures a
  second for YOLO11n); FFmpeg's software decoder is not (248 against 255).

## E0: the model alone

| Model | Provider | Precision | Batch | Windows | Linux | Change |
|---|---|---|---:|---:|---:|---:|
| YOLO11n | TensorRT | fp16 | 1 | 509 | 673 | −24% |
| YOLO11n | TensorRT | fp16 | 2 | 688 | 792 | −13% |
| YOLO11n | TensorRT | fp16 | 4 | 816 | 876 | −7% |
| YOLO11n | TensorRT | fp16 | 8 | 909 | 924 | −2% |
| YOLO11n | TensorRT | fp32 | 1 | 291 | 324 | −10% |
| YOLO11n | TensorRT | fp32 | 8 | 411 | 413 | 0% |
| YOLO11n | CUDA | fp32 | 1 | 161 | 217 | −26% |
| YOLO11n | CUDA | fp32 | 8 | 240 | 264 | −9% |
| YOLO11s | TensorRT | fp16 | 1 | 338 | 398 | −15% |
| YOLO11s | TensorRT | fp16 | 8 | 516 | 522 | −1% |
| YOLO11s | CUDA | fp32 | 1 | 105 | 112 | −6% |
| YOLOv10n | TensorRT | fp16 | 1 | 449 | 646 | −30% |
| YOLOv10n | CUDA | fp32 | 1 | 176 | 200 | −12% |

The shorter the run, the more is lost: YOLOv10n and YOLO11n at a batch of
one lose a quarter or more, YOLO11s — three times the work — 15%, and any
model at a batch of eight 2% or less.

## E1: decoding alone

| Configuration | Windows | Spread | Linux | Change | NVDEC % |
|---|---:|---:|---:|---:|---:|
| nvdec people-432p.mp4 | 2511 | 2467–2530 | 2957 | −15% | 61 |
| sw-decode people-432p.mp4 | 1204 | 1125–1210 | 1219 | −1% | – |
| nvdec people-1080p.mp4 | 640 | 629–678 | 779 | −18% | 76 |
| sw-decode people-1080p.mp4 | 248 | 228–249 | 255 | −3% | – |
| nvdec people-2160p.mp4 | 182 | 180–192 | 213 | −15% | 80 |
| sw-decode people-2160p.mp4 | 65 | 62–65 | 66 | −2% | – |

One stream keeps NVDEC 76% busy where it kept it 89% on Linux; with four
streams it is full on both (E4).

## E2: who runs the model

| Configuration | Windows | Spread | Linux | Change | GPU % | NVDEC % |
|---|---:|---:|---:|---:|---:|---:|
| cpu yolo11n | 32 | 31–32 | 47 | −32% | – | – |
| cuda fp32 yolo11n | 153 | 145–153 | 201 | −24% | 76 | 22 |
| tensorrt fp32 yolo11n | 259 | 247–259 | 276 | −6% | 90 | 35 |
| tensorrt fp16 yolo11n | 407 | 392–407 | 544 | −25% | 76 | 50 |
| cpu yolo11s | 13 | 13–13 | 21 | −38% | – | – |
| cuda fp32 yolo11s | 101 | 99–101 | 105 | −4% | 91 | 16 |
| tensorrt fp16 yolo11s | 295 | 287–298 | 341 | −13% | 87 | 37 |
| cuda fp32 yolov10n | 162 | 161–162 | 190 | −15% | 83 | 24 |
| tensorrt fp16 yolov10n | 399 | 398–400 | 538 | −26% | 84 | 49 |

One stream through YOLO11n on TensorRT fp16 runs at 80% of the model alone
(407 of 509), as on Linux (81%): the pipeline adds the same share; it is
the model's own rate that is lower.

## E3: the picture's size

| Configuration | Windows | Spread | Linux | Change | NVDEC % |
|---|---:|---:|---:|---:|---:|
| tensorrt fp16 people-432p.mp4 | 380 | 376–384 | 564 | −33% | 11 |
| tensorrt fp16 people-1080p.mp4 | 411 | 408–412 | 542 | −24% | 50 |
| tensorrt fp16 people-2160p.mp4 | 205 | 205–205 | 204 | 0% | 92 |

At 2160p NVDEC is the limit on both, and they meet.

## E4: streams

| Configuration | Windows | Spread | Linux | Change | GPU % | NVDEC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 stream, a detector | 408 | 407–410 | 541 | −25% | 76 | 49 | 878 |
| 1 stream, batched | 333 | 332–335 | 541 | −38% | 63 | 41 | 894 |
| 2 streams, a detector each | 470 | 470–471 | 641 | −27% | 82 | 59 | 1089 |
| 2 streams, batched | 564 | 562–565 | 631 | −11% | 88 | 72 | 1089 |
| 4 streams, a detector each | 477 | 476–477 | 676 | −29% | 77 | 61 | 1513 |
| 4 streams, batched | 670 | 668–670 | 671 | 0% | 93 | 89 | 1475 |
| 8 streams, a detector each | 426 | 424–426 | 657 | −35% | 72 | 54 | 2386 |
| 8 streams, batched | 714 | 714–714 | 705 | +1% | 91 | 99 | 2252 |
| 1 stream, decode only | 666 | 652–697 | 775 | −14% | 5 | 75 | 744 |
| 4 streams, decode only | 848 | 848–849 | 850 | 0% | 6 | 100 | 1082 |
| 8 streams, decode only | 849 | 849–850 | 855 | −1% | 6 | 100 | 1548 |

A detector for each stream never fills the GPU here — 72–82% busy, with
more streams going slower — while batching fills it and NVDEC with it.
Batched, four streams and eight run as on Linux. One stream through the
mux is the one batched row that loses more than the detector alone: the
mux's pipeline is one more thread for each picture to pass, which costs
nothing measurable on Linux and 18% here.

## E5: detecting some pictures, the tracker filling in

| Configuration | Windows | Spread | Linux | Change |
|---|---:|---:|---:|---:|
| 1 stream, interval 0 | 398 | 383–401 | 543 | −27% |
| 1 stream, interval 1 | 628 | 619–632 | 768 | −18% |
| 1 stream, interval 2 | 638 | 632–656 | 766 | −17% |
| 1 stream, interval 4 | 658 | 653–686 | 753 | −13% |
| 4 streams, interval 0 | 667 | 663–669 | 677 | −1% |
| 4 streams, interval 1 | 783 | 783–783 | 772 | +1% |
| 4 streams, interval 2 | 804 | 803–804 | 796 | +1% |
| 4 streams, interval 4 | 821 | 821–821 | 817 | 0% |

One stream skipping pictures stops at what NVDEC decodes of one stream
(E1, 640–666); four batched stop at NVDEC's whole rate, as on Linux.

## E6: tracking

| Configuration | Windows | Spread | Linux | Change |
|---|---:|---:|---:|---:|
| interval 0, no tracker | 409 | 378–411 | 544 | −25% |
| interval 0, motion | 407 | 378–408 | 542 | −25% |
| interval 0, by look on the CPU | 226 | 214–226 | 331 | −32% |
| interval 0, by look on the GPU | 385 | 365–386 | 523 | −26% |
| interval 4, no tracker | 661 | 656–672 | 736 | −10% |
| interval 4, motion | 652 | 651–658 | 747 | −13% |
| interval 4, by look on the CPU | 396 | 391–398 | 570 | −31% |
| interval 4, by look on the GPU | 646 | 643–646 | 758 | −15% |

Following by motion costs nothing measurable, by look on the GPU 6% and on
the CPU 45% — 4% and 39% on Linux.

## E7: a classifier after the tracker

| Configuration | Windows | Spread | Linux | Change |
|---|---:|---:|---:|---:|
| 1 stream, detector and tracker | 404 | 397–406 | 545 | −26% |
| 1 stream, classifier, every 30 pictures | 368 | 357–371 | 514 | −28% |
| 1 stream, classifier, every picture | 330 | 329–330 | 473 | −30% |
| 4 streams, detector and tracker | 670 | 669–670 | 676 | −1% |
| 4 streams, classifier, every 30 pictures | 625 | 605–625 | 624 | 0% |
| 4 streams, classifier, every picture | 514 | 507–521 | 554 | −7% |

The classifier through TensorRT costs one stream 9% here and 6% on Linux.

## E8: drawing and encoding

| Configuration | Windows | Spread | Linux | Change | NVENC % |
|---|---:|---:|---:|---:|---:|
| detect | 408 | 408–409 | 544 | −25% | 0 |
| detect, overlay | 388 | 387–391 | 524 | −26% | 0 |
| detect, overlay, NVENC | 305 | 304–308 | 432 | −29% | 69 |
| decode, NVENC | 451 | 451–451 | 448 | +1% | 100 |
| cpu detect 432p | 33 | 33–33 | 49 | −33% | – |
| cpu detect, overlay 432p | 33 | 32–33 | 49 | −33% | – |

NVENC encodes 1080p H.264 at 451 a second, as on Linux. One stream
detecting, drawing and encoding does not reach it here (69% busy): the
detector is the limit before the encoder is.

## E9: everything at once

Interval 2, the motion tracker, the classifier, the overlay and NVENC.

| Streams | Windows | Spread | Linux | Change | NVENC % | GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 436 | 436–437 | 441 | −1% | 96 | 1236 |
| 4 | 428 | 427–428 | 443 | −3% | 95 | 2353 |
| 8 | 418 | 417–418 | 443 | −6% | 93 | 3796 |

Every row stops at the one NVENC, as on Linux: detecting one picture in
three leaves the detector enough room that the encoder sets the pace even
for one stream.
