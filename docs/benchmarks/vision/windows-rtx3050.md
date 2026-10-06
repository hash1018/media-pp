# Vision benchmarks: Windows, RTX 3050

The same experiments as [the Linux page](linux-rtx3050.md), on the same
hardware — an i5-12400F with 32 GB and an RTX 3050 — under Windows 11, so
the difference is the operating system, its NVIDIA driver and the libraries
it runs on. Measured on 2026-10-06 with media-pp at `3746bab`, as
[method.md](method.md) describes: `bench.py`, three runs of each
configuration taking turns, the median with the lowest and highest beside
it. Each table sets the Linux median beside it, with the classifier through
TensorRT on both; CPU cores are the process's CPU time over the run, 1.00
being one core kept busy, read through `GetProcessTimes`.

The whole matrix was run twice, the first time before `vision_bench` read
the CPU time on Windows. The tables are the second run. The two agree
within 5% on almost every row; the widest apart are detecting, drawing and
encoding one stream (305 then 350) and following by look on the GPU (385
then 355).

## The machine

| | |
|---|---|
| CPU, memory, GPU | As on Linux: i5-12400F, 32 GB, RTX 3050 8 GB (sm_86), 115 W limit, PCIe 4.0 ×8 |
| OS | Windows 11 Pro, build 26200, the Balanced power plan; the GPU drives the desktop |
| NVIDIA | Driver 610.88 under WDDM, CUDA runtime 13.2, cuBLAS 13.3, cuDNN 9.23.2, TensorRT 10.15.1 |
| ONNX Runtime | Through `ort` 2.0.0-rc.13, as on Linux |
| FFmpeg | 8.0.1 |
| Rust | 1.99.0, release profile |

## Summary

- **Where one thing runs at a time, Windows is 15–30% slower.** The model
  alone, one picture at a time, runs at 498 a second against 673; one
  stream, decode and detect, at 403 against 544; NVDEC on one 1080p
  stream at 673 against 779. A batch of eight closes the model's gap to
  5% (881 against 924), so what is lost is a cost paid on each submission
  to the GPU rather than in the GPU's work — consistent with the WDDM
  driver's, which Linux does not have, though nothing here separates it
  from the different driver and library versions. One stream also takes
  more of the CPU for it: 0.87 cores against 0.56.
- **Where the GPU's engines are full, Windows matches Linux.** Eight
  streams batched run at 712 against 705, four with a detector looking at
  one picture in five at 821 against 817, everything at once — interval 2,
  tracker, classifier, overlay, NVENC — at 417–440 against 441–443. NVDEC
  (849 against 850) and NVENC (451 against 448) are the same engines.
- **So batching matters far more on Windows.** Eight streams with a
  detector each run at 412 on 3.74 cores; batched, at 712 on 0.32 — 73%
  more on a twelfth of the CPU, where Linux gains 7% and goes from 2.40
  cores to 0.71. One stream through a mux, on the other hand, costs 14%
  (403 to 345) where Linux loses nothing.
- **ONNX Runtime on the CPU is a third slower** (31 against 47 pictures a
  second for YOLO11n, on 4.2 cores against 4.3); FFmpeg's software decoder
  is not (238 against 255, on one core both).

## E0: the model alone

| Model | Provider | Precision | Batch | Windows | Linux | Change |
|---|---|---|---:|---:|---:|---:|
| YOLO11n | TensorRT | fp16 | 1 | 498 | 673 | −26% |
| YOLO11n | TensorRT | fp16 | 2 | 680 | 792 | −14% |
| YOLO11n | TensorRT | fp16 | 4 | 788 | 876 | −10% |
| YOLO11n | TensorRT | fp16 | 8 | 881 | 924 | −5% |
| YOLO11n | TensorRT | fp32 | 1 | 286 | 324 | −12% |
| YOLO11n | TensorRT | fp32 | 8 | 403 | 413 | −2% |
| YOLO11n | CUDA | fp32 | 1 | 158 | 217 | −27% |
| YOLO11n | CUDA | fp32 | 8 | 237 | 264 | −10% |
| YOLO11s | TensorRT | fp16 | 1 | 332 | 398 | −17% |
| YOLO11s | TensorRT | fp16 | 8 | 507 | 522 | −3% |
| YOLO11s | CUDA | fp32 | 1 | 104 | 112 | −7% |
| YOLOv10n | TensorRT | fp16 | 1 | 439 | 646 | −32% |
| YOLOv10n | CUDA | fp32 | 1 | 172 | 200 | −14% |

The shorter the run, the more is lost: YOLOv10n and YOLO11n at a batch of
one lose a quarter or more, YOLO11s — three times the work — a sixth, and
any model at a batch of eight 5% or less.

## E1: decoding alone

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) | NVDEC % |
|---|---:|---:|---:|---:|---:|---:|
| nvdec people-432p.mp4 | 2520 | 2502–2536 | 2957 | −15% | 0.24 (0.34) | 62 |
| sw-decode people-432p.mp4 | 1200 | 1167–1223 | 1219 | −2% | 1.09 (1.02) | – |
| nvdec people-1080p.mp4 | 673 | 667–686 | 779 | −14% | 0.13 (0.15) | 78 |
| sw-decode people-1080p.mp4 | 238 | 233–238 | 255 | −7% | 1.00 (1.01) | – |
| nvdec people-2160p.mp4 | 194 | 185–198 | 213 | −9% | 0.14 (0.10) | 86 |
| sw-decode people-2160p.mp4 | 63 | 60–64 | 66 | −5% | 1.01 (1.01) | – |

One stream keeps NVDEC 78% busy where it kept it 89% on Linux; with four
streams it is full on both (E4).

## E2: who runs the model

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) | GPU % |
|---|---:|---:|---:|---:|---:|---:|
| cpu yolo11n | 31 | 28–31 | 47 | −34% | 4.23 (4.30) | – |
| cuda fp32 yolo11n | 140 | 136–144 | 201 | −30% | 0.90 (0.53) | 72 |
| tensorrt fp32 yolo11n | 250 | 250–253 | 276 | −9% | 0.42 (0.55) | 90 |
| tensorrt fp16 yolo11n | 403 | 368–403 | 544 | −26% | 0.87 (0.56) | 78 |
| cpu yolo11s | 13 | 12–13 | 21 | −38% | 4.75 (5.10) | – |
| cuda fp32 yolo11s | 100 | 98–101 | 105 | −5% | 0.40 (0.52) | 91 |
| tensorrt fp16 yolo11s | 293 | 276–297 | 341 | −14% | 0.56 (0.54) | 86 |
| cuda fp32 yolov10n | 154 | 154–160 | 190 | −19% | 0.62 (0.53) | 80 |
| tensorrt fp16 yolov10n | 392 | 392–396 | 538 | −27% | 0.69 (0.57) | 85 |

One stream through YOLO11n on TensorRT fp16 runs at 81% of the model alone
(403 of 498), as on Linux (81%): the pipeline adds the same share; it is
the model's own rate that is lower. The fastest runs take the most CPU
beside them — 0.87 cores where TensorRT fp32, at 250 a second, takes 0.42
— which Linux does not show (0.55 and 0.56).

## E3: the picture's size

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) | NVDEC % |
|---|---:|---:|---:|---:|---:|---:|
| tensorrt fp16 people-432p.mp4 | 386 | 374–388 | 564 | −32% | 0.76 (0.57) | 11 |
| tensorrt fp16 people-1080p.mp4 | 394 | 372–401 | 542 | −27% | 0.86 (0.57) | 48 |
| tensorrt fp16 people-2160p.mp4 | 205 | 204–205 | 204 | 0% | 0.46 (0.58) | 91 |

At 2160p NVDEC is the limit on both, and they meet.

## E4: streams

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) | GPU % | NVDEC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 stream, a detector | 403 | 396–405 | 541 | −26% | 0.87 (0.57) | 77 | 50 | 1219 |
| 1 stream, batched | 345 | 336–382 | 541 | −36% | 0.78 (0.58) | 67 | 41 | 1312 |
| 2 streams, a detector each | 457 | 452–465 | 641 | −29% | 1.14 (0.83) | 81 | 57 | 1440 |
| 2 streams, batched | 555 | 526–564 | 631 | −12% | 0.54 (0.59) | 86 | 70 | 1454 |
| 4 streams, a detector each | 445 | 436–473 | 676 | −34% | 2.27 (1.54) | 74 | 56 | 1861 |
| 4 streams, batched | 651 | 610–660 | 671 | −3% | 0.50 (0.67) | 90 | 87 | 1869 |
| 8 streams, a detector each | 412 | 398–412 | 657 | −37% | 3.74 (2.40) | 70 | 52 | 2743 |
| 8 streams, batched | 712 | 712–714 | 705 | +1% | 0.32 (0.71) | 94 | 98 | 2562 |
| 1 stream, decode only | 688 | 655–693 | 775 | −11% | 0.13 (0.14) | 8 | 78 | 1083 |
| 4 streams, decode only | 849 | 849–850 | 850 | 0% | 0.19 (0.25) | 8 | 100 | 1442 |
| 8 streams, decode only | 845 | 844–847 | 855 | −1% | 0.16 (0.34) | 7 | 100 | 1905 |

A detector for each stream never fills the GPU here — 70–81% busy, with
more streams going slower and taking more of the CPU, 3.74 cores for eight
— while batching fills it and NVDEC with it on a third of a core. Batched,
four streams and eight run as on Linux, on less of the CPU than Linux
takes. One stream through the mux is the one batched row that loses more
than the detector alone: the mux's pipeline is one more thread for each
picture to pass, which costs nothing measurable on Linux and 14% here.

## E5: detecting some pictures, the tracker filling in

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) |
|---|---:|---:|---:|---:|---:|
| 1 stream, interval 0 | 384 | 376–398 | 543 | −29% | 0.90 (0.58) |
| 1 stream, interval 1 | 625 | 624–631 | 768 | −19% | 0.68 (0.58) |
| 1 stream, interval 2 | 637 | 635–647 | 766 | −17% | 0.59 (0.60) |
| 1 stream, interval 4 | 662 | 659–672 | 753 | −12% | 0.40 (0.57) |
| 4 streams, interval 0 | 659 | 642–660 | 677 | −3% | 0.45 (0.65) |
| 4 streams, interval 1 | 782 | 782–783 | 772 | +1% | 0.35 (0.65) |
| 4 streams, interval 2 | 803 | 802–803 | 796 | +1% | 0.28 (0.55) |
| 4 streams, interval 4 | 821 | 820–821 | 817 | 0% | 0.22 (0.46) |

One stream skipping pictures stops at what NVDEC decodes of one stream
(E1, 673); four batched stop at NVDEC's whole rate, as on Linux. The fewer
the detector's runs, the less CPU they take: 0.90 cores detecting every
picture of one stream, 0.40 one in five.

## E6: tracking

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) |
|---|---:|---:|---:|---:|---:|
| interval 0, no tracker | 401 | 368–405 | 544 | −26% | 0.85 (0.57) |
| interval 0, motion | 395 | 374–400 | 542 | −27% | 0.87 (0.57) |
| interval 0, by look on the CPU | 222 | 203–224 | 331 | −33% | 0.90 (0.68) |
| interval 0, by look on the GPU | 355 | 354–377 | 523 | −32% | 0.85 (0.57) |
| interval 4, no tracker | 663 | 657–673 | 736 | −10% | 0.47 (0.57) |
| interval 4, motion | 661 | 659–674 | 747 | −12% | 0.33 (0.55) |
| interval 4, by look on the CPU | 393 | 385–400 | 570 | −31% | 0.75 (0.70) |
| interval 4, by look on the GPU | 660 | 654–662 | 758 | −13% | 0.44 (0.62) |

Following by motion costs nothing measurable. By look, detecting every
picture, it costs 10% on the GPU — 5% in the first run — and 44% on the
CPU, against 4% and 39% on Linux; detecting one picture in five, nothing
on the GPU and 41% on the CPU.

## E7: a classifier after the tracker

| Configuration | Windows | Spread | Linux | Change | CPU cores |
|---|---:|---:|---:|---:|---:|
| 1 stream, detector and tracker | 385 | 380–386 | 545 | −29% | 0.90 |
| 1 stream, classifier, every 30 pictures | 346 | 337–355 | 514 | −33% | 0.80 |
| 1 stream, classifier, every picture | 325 | 325–326 | 473 | −31% | 0.77 |
| 4 streams, detector and tracker | 665 | 659–669 | 676 | −2% | 0.42 |
| 4 streams, classifier, every 30 pictures | 612 | 612–622 | 624 | −2% | 0.42 |
| 4 streams, classifier, every picture | 517 | 506–518 | 554 | −7% | 0.45 |

The classifier through TensorRT costs one stream 10–16% here and 6–13% on
Linux; four streams 8–22% here and 8–18% there. Linux's page has no CPU
time for the classifier through TensorRT.

## E8: drawing and encoding

| Configuration | Windows | Spread | Linux | Change | CPU cores (Linux) | NVENC % |
|---|---:|---:|---:|---:|---:|---:|
| detect | 387 | 375–405 | 544 | −29% | 0.88 (0.56) | 0 |
| detect, overlay | 370 | 355–386 | 524 | −29% | 0.92 (0.58) | 0 |
| detect, overlay, NVENC | 350 | 308–358 | 432 | −19% | 0.89 (0.61) | 77 |
| decode, NVENC | 451 | 451–451 | 448 | +1% | 0.14 (0.19) | 100 |
| cpu detect 432p | 31 | 31–33 | 49 | −37% | 4.34 (4.22) | – |
| cpu detect, overlay 432p | 31 | 30–33 | 49 | −37% | 4.29 (4.21) | – |

NVENC encodes 1080p H.264 at 451 a second, as on Linux. One stream
detecting, drawing and encoding does not reach it here (77% busy): the
detector is the limit before the encoder is.

## E9: everything at once

Interval 2, the motion tracker, the classifier, the overlay and NVENC.

| Streams | Windows | Spread | Linux | Change | CPU cores | NVENC % | GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 440 | 439–440 | 441 | 0% | 0.59 | 96 | 1598 |
| 4 | 430 | 428–433 | 443 | −3% | 0.45 | 95 | 2715 |
| 8 | 417 | 417–419 | 443 | −6% | 0.44 | 94 | 4163 |

Every row stops at the one NVENC, as on Linux: detecting one picture in
three leaves the detector enough room that the encoder sets the pace even
for one stream, on about half a core.
