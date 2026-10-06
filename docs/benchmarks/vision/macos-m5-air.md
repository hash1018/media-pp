# Vision benchmarks: macOS, M5 MacBook Air

Measured on 2026-10-06 with media-pp at the commit that adds the `metal`
backend to `vision_bench`, as [method.md](method.md) describes and
[macos.md](macos.md) maps onto a Mac: three runs of each configuration
taking turns, the median given with the lowest and highest beside it.
Pictures a second count every stream together; CPU cores are the
process's CPU time over the run. There is no `nvidia-smi` on a Mac, so
the GPU's utilisation is not measured.

## The machine

| | |
|---|---|
| Model | MacBook Air (Mac17,3) — **no fan** |
| Chip | Apple M5: 10 CPU cores (4 performance, 6 efficiency), 10 GPU cores, Neural Engine |
| Memory | 24 GB, shared by the CPU, the GPU and the Neural Engine |
| OS | macOS 26.5 |
| ONNX Runtime | `ort` 2.0.0-rc.13's prebuilt build with Core ML, the same release as on Linux |
| FFmpeg | 8.0.1 (Homebrew) |
| Rust | 1.98.1, release profile |

### Read every table as sustained, throttled performance

A MacBook Air has no fan: under minutes of load its chip lowers its clocks
to stay cool, and the matrix runs for two hours. The same configuration —
one 1080p stream through YOLO11n on Core ML — ran at 244 pictures a second
in E2, 165 in E3 and 146 in E4, the experiments one after another. Taking
turns spreads the heat over every configuration of an experiment alike,
so comparisons inside a table hold; comparing across tables, or with
[the Linux machine](linux-rtx3050.md), compares a fanless laptop at its
sustained rate with a desktop GPU at its full one. A wide range beside a
median is mostly this. A Mac with a fan — a mini, a Studio, a MacBook Pro
— would give higher and steadier numbers with the same harness.

The clips are those of [method.md](method.md#the-clips), made with
`h264_videotoolbox`, whose rate control gave 2.9 Mbit/s at 1080p and 6.2
at 2160p where 8 and 32 were asked for: the same pictures, a lighter file
to decode than the Linux machine's.

## Summary

- **Core ML on the GPU is the backend, five times the CPU.** YOLO11n at
  1080p: 244 pictures a second against 49 on the CPU's ten cores, on 0.4
  of a core against 3.3. Held to the Neural Engine it
  runs a quarter as fast (61) — Core ML's own choice, with every unit
  allowed, is the GPU's rate.
- **Batching does not pay here; a detector per stream does.** Eight
  streams batched run at 180 pictures a second, eight detectors at 207 —
  the opposite of the RTX 3050, where batching gained 7%. The model alone
  gains only 17% from batches of eight (295 to 345), and a batching
  detector runs its fitting, its model and its reading one after another
  on one thread, where eight detectors overlap theirs.
  [findings.md](findings.md#batching-on-apple-silicon) has the experiment
  that showed it.
- **A detection interval is the setting that matters.** Four streams
  batched, detecting one picture in five with the motion tracker filling
  in, run at 880 pictures a second against 217 detecting every one.
- **Following by look costs nothing on the GPU.** Metal's correlation
  filters run beside the motion tracker at its rate (159 against 165;
  436 against 432 at interval 4).
- **The encoder is the ceiling of everything at once.** One to eight
  streams with interval 2, tracker, classifier, overlay and encoding run
  at 234 to 272 pictures a second, and 272 is what VideoToolbox's H.264
  encoder takes at 1080p on this chip with nothing else running (276).

## E0: the model alone

`model_only --provider coreml`: the detector run over and over on a
tensor in memory, Core ML copying it in on each run as it does for the
detector. The GPU and the Neural Engine are Core ML's compute units:
`all` lets it choose layer by layer, `gpu` and `ane` hold it to one, the
CPU allowed beside each for what the other does not take.

| model | provider | precision | batch | pictures/s (median) | ms a batch |
|---|---|---|---:|---:|---:|
| yolo11n | coreml | all | 1 | 295 | 3.39 |
| yolo11n | coreml | all | 2 | 329 | 6.08 |
| yolo11n | coreml | all | 4 | 344 | 11.63 |
| yolo11n | coreml | all | 8 | 345 | 23.20 |
| yolo11n | coreml | gpu | 1 | 288 | 3.47 |
| yolo11n | coreml | gpu | 8 | 341 | 23.44 |
| yolo11n | coreml | ane | 1 | 76 | 13.07 |
| yolo11n | coreml | ane | 8 | 70 | 115.11 |
| yolo11s | coreml | all | 1 | 121 | 8.28 |
| yolo11s | coreml | all | 8 | 115 | 69.81 |
| yolo11s | coreml | ane | 1 | 31 | 31.95 |
| yolov10n | coreml | all | 1 | 273 | 3.66 |
| yolov10n | coreml | ane | 1 | 71 | 14.10 |

`all` is the GPU's rate: Core ML put these models there. The Neural
Engine is a quarter of it and gains nothing from batches. These are the
first numbers of the run, before the chip heated, and the closest to its
peak.

## E1: decoding alone

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| videotoolbox people-432p.mp4 | 1347 (557–1442) | 0.20 | 0.00 |
| sw-decode people-432p.mp4 | 1571 (1528–1583) | 1.00 | 0.00 |
| videotoolbox people-1080p.mp4 | 418 (313–474) | 0.21 | 0.00 |
| sw-decode people-1080p.mp4 | 563 (560–568) | 1.00 | 0.00 |
| videotoolbox people-2160p.mp4 | 214 (198–225) | 0.12 | 0.00 |
| sw-decode people-2160p.mp4 | 191 (187–195) | 1.00 | 0.00 |

VideoToolbox decodes one stream at a time no faster than FFmpeg's software
decoder on one core below 4K — a hardware session's latency on a stream
that is never waited for — on a fifth of the CPU. Its strength is
several at once ([E4](#e4-streams): 1928 a second for eight 1080p streams).

## E2: who runs the model

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| cpu yolo11n | 49 (48–53) | 3.28 | 0.96 |
| core ml all yolo11n | 244 (236–250) | 0.39 | 0.75 |
| core ml gpu yolo11n | 224 (169–246) | 0.45 | 0.75 |
| core ml ane yolo11n | 61 (61–64) | 1.38 | 0.75 |
| cpu yolo11s | 19 (17–23) | 3.54 | 1.17 |
| core ml all yolo11s | 72 (71–76) | 0.18 | 0.90 |
| core ml ane yolo11s | 26 (24–26) | 1.42 | 0.90 |
| core ml all yolov10n | 193 (135–199) | 0.43 | 0.77 |
| core ml ane yolov10n | 59 (54–61) | 1.40 | 0.77 |

## E3: the picture's size

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| core ml people-432p.mp4 | 191 (111–197) | 0.36 | 0.76 |
| core ml people-1080p.mp4 | 165 (160–172) | 0.46 | 0.75 |
| core ml people-2160p.mp4 | 120 (119–148) | 0.52 | 0.75 |

The model takes 640×640 whatever the picture, so what grows is decoding
and the Metal fitting that shrinks the picture.

## E4: streams

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| 1 streams, a detector each | 146 (144–187) | 0.44 | 0.75 |
| 1 streams, batched | 130 (122–148) | 0.49 | 0.75 |
| 2 streams, a detector each | 217 (214–225) | 0.64 | 0.75 |
| 2 streams, batched | 202 (199–203) | 0.25 | 0.75 |
| 4 streams, a detector each | 211 (208–221) | 0.96 | 0.75 |
| 4 streams, batched | 186 (186–193) | 0.30 | 0.75 |
| 8 streams, a detector each | 207 (207–209) | 1.19 | 0.75 |
| 8 streams, batched | 180 (172–190) | 0.26 | 0.75 |
| 1 streams decode only | 445 (308–477) | 0.21 | 0.00 |
| 4 streams decode only | 1655 (1616–1742) | 0.29 | 0.00 |
| 8 streams decode only | 1928 (1845–1934) | 0.27 | 0.00 |

Detection is the limit — decoding alone goes nine times as fast — and
past two streams it does not grow: the GPU is shared. A detector per
stream beats one batching them at every count, on more of the CPU.

## E5: detecting one picture in N, the tracker between

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| 1 stream(s), interval 0 | 134 (106–201) | 0.46 | 0.90 |
| 1 stream(s), interval 1 | 369 (203–407) | 0.56 | 0.77 |
| 1 stream(s), interval 2 | 327 (289–396) | 0.58 | 0.72 |
| 1 stream(s), interval 4 | 421 (397–450) | 0.51 | 0.65 |
| 4 stream(s), interval 0 | 217 (214–227) | 0.28 | 0.90 |
| 4 stream(s), interval 1 | 369 (362–382) | 0.40 | 0.77 |
| 4 stream(s), interval 2 | 562 (489–584) | 0.43 | 0.72 |
| 4 stream(s), interval 4 | 880 (775–952) | 0.43 | 0.65 |

## E6: following

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| interval 0, no tracker | 112 (99–148) | 0.52 | 0.75 |
| interval 0, motion | 165 (136–183) | 0.42 | 0.90 |
| interval 0, visual on the GPU | 159 (110–170) | 0.49 | 0.90 |
| interval 4, no tracker | 405 (386–510) | 0.48 | 0.15 |
| interval 4, motion | 432 (405–498) | 0.48 | 0.65 |
| interval 4, visual on the GPU | 436 (428–442) | 0.49 | 0.65 |

Metal follows VideoToolbox pictures by look on the GPU, with no build of
its own; there is no CPU row to set beside it on a Mac.

## E7: a classifier after the tracker

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| 1 stream(s), detector and tracker | 111 (105–181) | 0.53 | 0.90 |
| 1 stream(s), classifier, each object every 30 pictures | 202 (100–202) | 0.35 | 0.90 |
| 1 stream(s), classifier, each object every picture | 168 (126–168) | 0.34 | 0.90 |
| 4 stream(s), detector and tracker | 204 (203–210) | 0.21 | 0.90 |
| 4 stream(s), classifier, each object every 30 pictures | 176 (168–178) | 0.29 | 0.90 |
| 4 stream(s), classifier, each object every picture | 150 (132–161) | 0.20 | 0.90 |

The first row ran below the rows with a classifier after it, which do
strictly more: its range (105–181) is the heat. Within the four-stream
rows, a classifier costs 14% keeping each answer 30 pictures and 26%
asking every picture.

## E8: drawing and encoding

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| detect | 175 (92–184) | 0.35 | 0.75 |
| detect, overlay | 165 (132–180) | 0.37 | 0.75 |
| detect, overlay, VideoToolbox encode | 153 (100–170) | 0.46 | 0.00 |
| decode, VideoToolbox encode | 276 (276–276) | 0.24 | 0.00 |
| cpu detect 432p | 43 (42–46) | 3.17 | 0.96 |
| cpu detect, overlay 432p | 42 (39–44) | 3.20 | 0.96 |

Objects are counted on the pictures that reach the sink; with an encoder
before it, the sink counts packets, which carry none.

## E9: everything at once

| configuration | pictures/s | CPU cores | objects/picture |
|---|---:|---:|---:|
| 1 stream(s) 1080p: interval 2, tracker, classifier, overlay, VideoToolbox encode | 234 (232–244) | 0.50 | 0.00 |
| 4 stream(s) 1080p: interval 2, tracker, classifier, overlay, VideoToolbox encode | 272 (272–272) | 0.49 | 0.00 |
| 8 stream(s) 1080p: interval 2, tracker, classifier, overlay, VideoToolbox encode | 272 (272–272) | 0.41 | 0.00 |
