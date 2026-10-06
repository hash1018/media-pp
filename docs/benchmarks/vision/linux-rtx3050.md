# Vision benchmarks: Linux, RTX 3050

Measured on 2026-10-06 with media-pp at `d6a46fe3`, and E7 and E9 again
with the classifier running through TensorRT (see
[findings.md](findings.md#the-classifier)), as
[method.md](method.md) describes: three runs of each configuration taking
turns, the median given with the lowest and highest beside it. Pictures a
second count every stream together. CPU cores are the process's CPU time
over the run, so 1.00 is one core kept busy. GPU, NVDEC and NVENC are
`nvidia-smi`'s utilisation, and GPU MiB the most memory in use.

## The machine

| | |
|---|---|
| CPU | Intel Core i5-12400F, 6 cores and 12 threads, up to 4.4 GHz, `powersave` governor |
| Memory | 32 GB |
| GPU | NVIDIA GeForce RTX 3050, 8 GB, Ampere (sm_86), 115 W limit, PCIe 4.0 ×8 |
| | One NVDEC and one NVENC. The GPU also drives the desktop, idle while measuring (about 0.4 GB in use by other processes). |
| OS | Ubuntu 26.04, Linux 7.0.0 |
| NVIDIA | Driver 595.91.07, CUDA runtime 13.2, cuBLAS 13.4, cuDNN 9.27, TensorRT 10.16.1 |
| ONNX Runtime | 1.28.0, through `ort` 2.0.0-rc.13 |
| FFmpeg | 8.1, LGPL build |
| Rust | 1.97.1, release profile |

DeepStream was not run beside it. It needs root to install, or Docker,
and this machine has neither. [E0](#e0-the-model-alone) stands in for it.

## Summary

- **The model sets the pace of one stream, and NVDEC and NVENC set the
  pace of several.** One 1080p stream through YOLO11n on TensorRT fp16
  runs at 544 pictures a second, 81% of what the model alone runs at
  (673). Eight streams batched run at 705, 82% of what NVDEC can decode at
  all (855). Encoding them too, the one NVENC (448 at 1080p) is what
  stops them.
- **Precision and provider matter most.** TensorRT fp16 is 2.7 times CUDA
  in the pipeline, and twice TensorRT fp32.
- **Batching helps the CPU more than the rate.** Eight streams batched go
  7% faster than eight detectors, on a third of the CPU (0.71 cores
  against 2.40).
- **Skipping pictures is close to free with a tracker.** Detecting one
  picture in two takes a stream to NVDEC's limit. Following by motion
  costs nothing measurable, and following by look on the GPU costs 4%
  where the CPU's costs 39%.
- **The classifier was the one stage far from its ceiling.** On CUDA's
  provider beside TensorRT's detector it took one stream from 545 to 228
  pictures a second. Run through TensorRT, as it now is, it costs 6%
  (514), and everything at once — interval 2, tracker, classifier,
  overlay, NVENC — runs at 441 to 443 a second for one stream or eight,
  NVENC's limit. [findings.md](findings.md#the-classifier) has how it was
  found.

## E0: the model alone

`model_only`: the detector run over and over on a tensor already on the
GPU, its output left there. Nothing of the pipeline is in it, so it is the
most the model can run at on this GPU, and what DeepStream's `nvinfer`
could reach at best, since it runs the same engine.

| Model | Provider | Precision | Batch | Pictures/s | ms a batch |
|---|---|---|---:|---:|---:|
| YOLO11n | TensorRT | fp16 | 1 | 673 | 1.49 |
| YOLO11n | TensorRT | fp16 | 2 | 792 | 2.52 |
| YOLO11n | TensorRT | fp16 | 4 | 876 | 4.57 |
| YOLO11n | TensorRT | fp16 | 8 | 924 | 8.66 |
| YOLO11n | TensorRT | fp32 | 1 | 324 | 3.09 |
| YOLO11n | TensorRT | fp32 | 8 | 413 | 19.39 |
| YOLO11n | CUDA | fp32 | 1 | 217 | 4.61 |
| YOLO11n | CUDA | fp32 | 8 | 264 | 30.33 |
| YOLO11s | TensorRT | fp16 | 1 | 398 | 2.51 |
| YOLO11s | TensorRT | fp16 | 8 | 522 | 15.33 |
| YOLO11s | CUDA | fp32 | 1 | 112 | 8.93 |
| YOLOv10n | TensorRT | fp16 | 1 | 646 | 1.55 |
| YOLOv10n | CUDA | fp32 | 1 | 200 | 4.99 |
| MobileNetV2 (classifier, 224×224) | TensorRT | fp16 | 1 | 3282 | 0.30 |
| MobileNetV2 | TensorRT | fp16 | 4 | 5319 | 0.75 |
| MobileNetV2 | CUDA | fp32 | 1 | 887 | 1.13 |
| MobileNetV2 | CUDA | fp32 | 4 | 1082 | 3.70 |

TensorRT fp16 is three times CUDA's provider and twice TensorRT fp32. A
batch of eight gets 37% more out of YOLO11n than one picture at a time:
on its own, a small model leaves much of the GPU idle. The three runs of
each agreed to within 1%.

## E1: decoding alone

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| nvdec people-432p.mp4 | 2957 | 2718–2991 | 0.34 | 4 | 74 | 0 | 710 |
| sw-decode people-432p.mp4 | 1219 | 1218–1221 | 1.02 | – | – | – | – |
| nvdec people-1080p.mp4 | 779 | 777–780 | 0.15 | 5 | 89 | 0 | 808 |
| sw-decode people-1080p.mp4 | 255 | 249–256 | 1.01 | 0 | 0 | 0 | 598 |
| nvdec people-2160p.mp4 | 213 | 212–213 | 0.10 | 6 | 92 | 0 | 1043 |
| sw-decode people-2160p.mp4 | 66 | 64–67 | 1.01 | 0 | 0 | 0 | 598 |

NVDEC decodes 1080p at 780 pictures a second, 26 streams of 30 fps, on
0.15 of a core. FFmpeg's software decoder, one thread as `SwDecoder` runs
it, manages 255. Everything below that decodes on NVDEC can go no faster
than this row, and at 2160p (213) NVDEC is the limit before anything
else.

## E2: who runs the model

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu yolo11n | 47 | 46–47 | 4.30 | 0 | 0 | 0 | 598 |
| cuda fp32 yolo11n | 201 | 201–202 | 0.53 | 93 | 30 | 0 | 943 |
| tensorrt fp32 yolo11n | 276 | 276–278 | 0.55 | 91 | 38 | 0 | 948 |
| tensorrt fp16 yolo11n | 544 | 540–550 | 0.56 | 89 | 68 | 0 | 948 |
| cpu yolo11s | 21 | 20–21 | 5.10 | 0 | 0 | 0 | 598 |
| cuda fp32 yolo11s | 105 | 103–105 | 0.52 | 92 | 16 | 0 | 1136 |
| tensorrt fp16 yolo11s | 341 | 337–343 | 0.54 | 90 | 44 | 0 | 959 |
| cuda fp32 yolov10n | 190 | 190–190 | 0.53 | 93 | 28 | 0 | 944 |
| tensorrt fp16 yolov10n | 538 | 538–546 | 0.57 | 91 | 68 | 0 | 951 |

Through the pipeline, at 1080p, one stream:

| | Pipeline | Model alone (E0) | Share |
|---|---:|---:|---:|
| YOLO11n, TensorRT fp16 | 544 | 673 | 81% |
| YOLO11n, TensorRT fp32 | 276 | 324 | 85% |
| YOLO11n, CUDA fp32 | 201 | 217 | 93% |
| YOLO11s, TensorRT fp16 | 341 | 398 | 86% |
| YOLOv10n, TensorRT fp16 | 538 | 646 | 83% |

The faster the model, the larger the share the rest of the pipeline takes
of each picture: decoding, fitting the picture into the model's input,
reading the boxes. At 544 a second the GPU is 89% busy and NVDEC 68%, so
the two are sharing the GPU rather than waiting on each other. On the CPU,
YOLO11n runs at 47 pictures a second on 4.3 cores.

## E3: the picture's size

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| tensorrt fp16 people-432p.mp4 | 564 | 560–567 | 0.57 | 90 | 18 | 0 | 822 |
| tensorrt fp16 people-1080p.mp4 | 542 | 531–544 | 0.57 | 91 | 69 | 0 | 949 |
| tensorrt fp16 people-2160p.mp4 | 204 | 203–205 | 0.58 | 39 | 91 | 0 | 1263 |

The detector shrinks every picture to 640, so its cost hardly depends on
the picture's size: 432p and 1080p differ by 4%. At 2160p NVDEC is the
limit (E1: 213), and the GPU is 39% busy waiting for it.

## E4: several streams

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 streams, a detector each | 541 | 529–544 | 0.57 | 90 | 69 | 0 | 949 |
| 1 streams, batched | 541 | 538–545 | 0.58 | 91 | 68 | 0 | 965 |
| 2 streams, a detector each | 641 | 640–658 | 0.83 | 95 | 85 | 0 | 1163 |
| 2 streams, batched | 631 | 620–632 | 0.59 | 92 | 85 | 0 | 1160 |
| 4 streams, a detector each | 676 | 670–687 | 1.54 | 95 | 92 | 0 | 1596 |
| 4 streams, batched | 671 | 671–675 | 0.67 | 91 | 95 | 0 | 1572 |
| 8 streams, a detector each | 657 | 630–658 | 2.40 | 96 | 89 | 0 | 2486 |
| 8 streams, batched | 705 | 704–706 | 0.71 | 90 | 99 | 0 | 2330 |
| 1 streams decode only | 775 | 771–776 | 0.14 | 5 | 89 | 0 | 808 |
| 4 streams decode only | 850 | 849–851 | 0.25 | 6 | 100 | 0 | 1157 |
| 8 streams decode only | 855 | 854–858 | 0.34 | 6 | 100 | 0 | 1625 |

Up to four streams, one detector batching them and a detector each go
about as fast. At eight, batching is 7% faster (705 against 657). Either
way the streams together reach NVDEC's limit: decoding alone tops out at
855 a second, and NVDEC is 99% busy at eight streams batched. What
batching saves is the CPU: 0.71 cores for eight streams against 2.40,
since eight detectors are eight sessions each waiting on the GPU from a
thread of its own. One stream through a mux costs nothing against one
without (541 either way).

## E5: detecting one picture in N

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 stream(s), interval 0 | 543 | 539–546 | 0.58 | 90 | 68 | 0 | 948 |
| 1 stream(s), interval 1 | 768 | 756–768 | 0.58 | 66 | 93 | 0 | 948 |
| 1 stream(s), interval 2 | 766 | 766–767 | 0.60 | 46 | 91 | 0 | 948 |
| 1 stream(s), interval 4 | 753 | 750–754 | 0.57 | 30 | 88 | 0 | 949 |
| 4 stream(s), interval 0 | 677 | 677–678 | 0.65 | 90 | 95 | 0 | 1547 |
| 4 stream(s), interval 1 | 772 | 771–772 | 0.65 | 55 | 99 | 0 | 1547 |
| 4 stream(s), interval 2 | 796 | 796–796 | 0.55 | 40 | 99 | 0 | 1547 |
| 4 stream(s), interval 4 | 817 | 817–818 | 0.46 | 27 | 100 | 0 | 1547 |

`--interval N` lets the detector look at one picture in N+1, and the
tracker fills in the rest. One stream reaches NVDEC's limit at interval 1
(768 against 775 decoding alone); after that the GPU idles (66% → 30%)
and nothing more is gained. Four streams batched go from 677 to 817. The
fewer pictures detected, the fewer boxes come out: the objects on a
picture fall from 0.90 to 0.65 at interval 4, as the tracker hands on
only those it is following.

## E6: following

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval 0, no tracker | 544 | 541–546 | 0.57 | 90 | 69 | 0 | 948 |
| interval 0, motion | 542 | 536–542 | 0.57 | 89 | 68 | 0 | 948 |
| interval 0, visual on the CPU | 331 | 330–333 | 0.68 | 58 | 42 | 0 | 948 |
| interval 0, visual on the GPU | 523 | 522–526 | 0.57 | 86 | 64 | 0 | 956 |
| interval 4, no tracker | 736 | 719–741 | 0.57 | 28 | 83 | 0 | 948 |
| interval 4, motion | 747 | 719–749 | 0.55 | 28 | 84 | 0 | 948 |
| interval 4, visual on the CPU | 570 | 560–571 | 0.70 | 24 | 60 | 0 | 948 |
| interval 4, visual on the GPU | 758 | 754–760 | 0.62 | 31 | 89 | 0 | 956 |

Following by motion — a Kalman filter and matching — costs nothing that
can be measured. Following by look as well, with correlation filters,
costs 39% on the CPU and 4% on the GPU (`--features gpu-dcf`, the
filters sampled, transformed and matched on the device). With the
detector looking at one picture in five, by look on the GPU runs at 758,
NVDEC's limit, and on the CPU at 570.

## E7: a second model on what was found

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 stream(s), detector and tracker | 545 | 542–546 | 0.57 | 89 | 67 | 0 | 948 |
| 1 stream(s), classifier, each object every 30 pictures | 228 | 227–228 | 0.78 | 49 | 33 | 0 | 1126 |
| 1 stream(s), classifier, each object every picture | 139 | 139–141 | 0.82 | 31 | 15 | 0 | 1255 |
| 4 stream(s), detector and tracker | 676 | 667–678 | 0.67 | 90 | 94 | 0 | 1547 |
| 4 stream(s), classifier, each object every 30 pictures | 413 | 412–415 | 0.68 | 68 | 58 | 0 | 1742 |
| 4 stream(s), classifier, each object every picture | 273 | 271–273 | 0.68 | 60 | 37 | 0 | 1859 |

The table above is the classifier as it was, on CUDA's provider:
MobileNetV2 on each tracked object, its answer kept 30 pictures or asked
again on every picture. Either way it took far more than the model costs —
545 → 228 a second for one stream, with the GPU 49% busy and NVDEC 33%:
something waiting rather than working. [findings.md](findings.md#the-classifier)
follows it down to CUDA's many small kernels queueing behind TensorRT's.
Through TensorRT, as the classifier now runs, measured against it in the
same run:

| Configuration | CUDA classifier | TensorRT classifier | Change |
|---|---:|---:|---:|
| 1 stream, every 30 pictures | 228 | 514 | +125% |
| 1 stream, every picture | 139 | 473 | +240% |
| 4 streams batched, every 30 pictures | 418 | 624 | +50% |
| 4 streams batched, every picture | 272 | 554 | +104% |

## E8: drawing and encoding

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| detect | 544 | 535–548 | 0.56 | 88 | 67 | 0 | 948 |
| detect, overlay | 524 | 514–530 | 0.58 | 89 | 67 | 0 | 952 |
| detect, overlay, NVENC | 432 | 432–432 | 0.61 | 74 | 53 | 95 | 1112 |
| decode, NVENC | 448 | 448–448 | 0.19 | 5 | 51 | 100 | 907 |
| cpu detect 432p | 49 | 49–49 | 4.22 | 0 | 0 | 0 | 598 |
| cpu detect, overlay 432p | 49 | 49–49 | 4.21 | 0 | 0 | 0 | 598 |

Drawing boxes and labels costs 4%. Encoding is bounded by NVENC: the RTX
3050's one encoder takes 1080p H.264 at 448 pictures a second (decode and
encode alone, NVENC 100%), and detection, drawing and encoding together
run at 432, NVENC 95% busy. On the CPU, drawing onto 432p pictures costs
nothing beside the detector.

## E9: everything at once

| Configuration | Pictures/s | Spread | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 stream(s) 1080p: interval 2, tracker, classifier, overlay, NVENC | 290 | 288–290 | 0.72 | 24 | 35 | 66 | 1429 |
| 4 stream(s) 1080p: interval 2, tracker, classifier, overlay, NVENC | 392 | 392–396 | 0.56 | 27 | 46 | 89 | 2520 |
| 8 stream(s) 1080p: interval 2, tracker, classifier, overlay, NVENC | 421 | 419–422 | 0.54 | 30 | 51 | 93 | 3971 |

Interval 2, the motion tracker, the classifier, the overlay and NVENC:
DeepStream's reference shape. The table is the classifier on CUDA's
provider. There one stream was held by the classifier (290, the GPU 24%
busy), and eight streams by NVENC (93% busy, 421). With the classifier
through TensorRT, measured against it in the same run:

| Streams | CUDA classifier | TensorRT classifier | NVENC busy |
|---:|---:|---:|---:|
| 1 | 292 | 441 | 100% |
| 4 | 393 | 443 | 99% |
| 8 | 421 | 443 | 99% |

Every row now stops at the one NVENC, which encodes 1080p H.264 at 448 a
second (E8). A DeepStream pipeline encoding the same streams on this GPU
would meet the same encoder.
