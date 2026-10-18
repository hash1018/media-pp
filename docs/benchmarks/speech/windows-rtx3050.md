# Speech benchmarks: Windows, RTX 3050

Every experiment of [method.md](method.md) on an i5-12400F with an RTX 3050
under Windows 11 — the machine the [vision benchmarks](../vision/windows-rtx3050.md)
ran on. Measured on 2026-10-18 with media-pp at `9ac0142`: `bench.py`, three
runs of each configuration taking turns, the median with the lowest and
highest beside it. Error rates are the same every run (decoding is
greedy), so they stand alone. CPU cores are the process's CPU time over the
run, 1.00 being one core kept busy; GPU % and MiB are `nvidia-smi`'s
utilisation and the memory the run added, sampled every 200 ms. The whole
matrix took 2 h 37 min.

Korean is scored by CER, English by WER — see [method.md](method.md#what-is-scored).

## The machine

| | |
|---|---|
| CPU, memory | Intel Core i5-12400F (6 cores, 12 threads), 32 GB |
| GPU | NVIDIA GeForce RTX 3050 8 GB (sm_86), 115 W limit; it drives the desktop |
| OS | Windows 11 Pro, build 26200, the Balanced power plan |
| NVIDIA | Driver 610.88 under WDDM, CUDA runtime and cuBLAS 13.2 from the redistributable archives |
| Vulkan | SDK 1.4.357 |
| whisper.cpp | 1.8.3, through `whisper-rs` 0.16 and `whisper-rs-sys` 0.15, `CMAKE_CUDA_ARCHITECTURES=86` |
| FFmpeg | 8.0.1 |
| Rust | 1.99.0, release profile |

## Summary

- **`large-v3-turbo` is the model to stream.** On a GPU it hears Korean
  six to seven times faster than it is spoken and misses 3.3–3.5% of the
  characters, as it does hearing the whole clip at once; English it hears
  at 8.5% WER. `large-v3` misses more and takes twice as long; `medium` is
  slower and misses more than either.
- **Quantised, it loses little and halves the memory.** `large-v3-turbo-q5_0`
  takes 1.1 GB of the GPU where the full one takes 2.1 GB, runs as fast,
  and misses 5.0% where it missed 3.5%.
- **Live captions trail the speech by four seconds** with a round every
  two seconds: 3.9 s on average, 6.6 s at worst, 29% of the GPU. Every four
  seconds, the default, it is twice that and half the GPU.
- **Vulkan is faster than CUDA here**, on half the CPU, and hears the same:
  `large-v3-turbo` streams at 6.9x through Vulkan against 5.8x through
  CUDA, on 0.42 cores against 0.80. Only the quantised turbo runs alike
  through both.
- **The CPU cannot stream any model worth hearing.** `small` streams at
  0.6x real time on four cores; `base` keeps up but misses 11%.
- **Streaming costs three to six times hearing a file at once**, since
  every round hears up to 25 seconds again — 28x against 5.8x for
  `large-v3-turbo` — and hears as well for the large models.

## S0: the model alone

The whole clip handed to whisper.cpp in one call. Korean, the long clip on
the GPU and the short one on the CPU.

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB | Load, s |
|---|---|---:|---:|---:|---:|---:|---:|
| `tiny` | CUDA | 123.4 (123.3–125.1) | 18.1 | 1.21 | 50 | 374 | 0.33 |
| `base` | CUDA | 91.7 (91.0–92.2) | 13.2 | 1.14 | 63 | 486 | 0.34 |
| `small` | CUDA | 51.4 (51.3–51.5) | 15.3 | 1.07 | 72 | 924 | 0.59 |
| `medium` | CUDA | 21.4 (21.4–21.5) | 7.5 | 0.99 | 85 | 2108 | 1.30 |
| `large-v3-turbo` | CUDA | 28.0 (27.9–28.0) | 3.3 | 0.83 | 86 | 2112 | 1.40 |
| `large-v3-turbo-q5_0` | CUDA | 31.9 (31.8–31.9) | 3.1 | 0.97 | 90 | 1118 | 0.64 |
| `large-v3` | CUDA | 8.0 (8.0–8.1) | 6.9 | 0.94 | 88 | 3794 | 2.41 |
| `tiny` | CPU | 29.4 (29.0–29.4) | 16.8 | 3.79 | — | — | 0.23 |
| `base` | CPU | 15.2 (15.2–15.3) | 12.5 | 3.87 | — | — | 0.28 |
| `small` | CPU | 4.5 (4.5–4.5) | 6.2 | 3.94 | — | — | 0.52 |

`large-v3-turbo` has `large-v3`'s encoder and an eighth of its decoder,
and here it is both faster and better: `large-v3` mishears more of the
Korean, and passes over more of it. The quantised
turbo is the fastest of the large models, at half the memory. The CPU rows
are a shorter clip with fewer of the hard sentences, which flatters
`small`.

## S1: who runs the model

Streamed, the short Korean clip (75 s).

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB |
|---|---|---:|---:|---:|---:|---:|
| `tiny` | CPU | 4.2 (4.2–4.2) | 19.8 | 3.78 | — | — |
| `tiny` | CUDA | 24.1 (23.9–24.2) | 24.5 | 1.18 | 50 | 374 |
| `tiny` | Vulkan | 28.9 (28.6–29.1) | 24.5 | 1.04 | 45 | 299 |
| `base` | CPU | 2.2 (2.2–2.2) | 11.0 | 3.90 | — | — |
| `base` | CUDA | 22.1 (22.0–22.2) | 15.0 | 1.11 | 63 | 486 |
| `base` | Vulkan | 24.7 (24.6–24.9) | 15.4 | 0.83 | 58 | 417 |
| `small` | CPU | 0.6 (0.6–0.6) | 12.5 | 3.95 | — | — |
| `small` | CUDA | 10.0 (10.0–10.1) | 12.5 | 1.06 | 78 | 925 |
| `small` | Vulkan | 11.4 (11.4–11.5) | 12.5 | 0.58 | 74 | 867 |
| `large-v3-turbo` | CUDA | 5.4 (5.4–5.4) | 4.0 | 0.78 | 87 | 2113 |
| `large-v3-turbo` | Vulkan | 6.7 (6.6–6.7) | 4.0 | 0.39 | 81 | 2071 |

The three backends hear almost the same — `small` exactly the same — and
differ in speed: Vulkan is 12–24% faster than CUDA and keeps half the CPU
busy. On the CPU only `tiny` and `base` keep up with speech, and `small`,
the smallest model that hears Korean tolerably, runs at 0.6x.

## S2: the model's size and precision

Streamed, the long Korean clip (316 s), on each GPU backend.

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB | Load, s |
|---|---|---:|---:|---:|---:|---:|---:|
| `tiny` | CUDA | 22.1 (22.1–22.2) | 21.6 | 1.17 | 52 | 378 | 0.31 |
| `tiny` | Vulkan | 26.3 (26.2–26.4) | 20.1 | 1.03 | 45 | 305 | 0.30 |
| `base` | CUDA | 20.7 (20.6–20.7) | 30.8 | 1.14 | 62 | 487 | 0.35 |
| `base` | Vulkan | 26.2 (26.2–26.3) | 26.3 | 0.89 | 55 | 417 | 0.36 |
| `small` | CUDA | 8.9 (8.8–8.9) | 10.7 | 1.07 | 79 | 926 | 0.59 |
| `small` | Vulkan | 9.7 (9.7–9.7) | 12.8 | 0.61 | 75 | 866 | 0.69 |
| `medium` | CUDA | 4.0 (4.0–4.0) | 8.1 | 0.98 | 89 | 2109 | 1.27 |
| `medium` | Vulkan | 4.2 (4.2–4.2) | 8.8 | 0.45 | 83 | 2059 | 1.61 |
| `large-v3-turbo` | CUDA | 5.8 (5.7–5.8) | 3.5 | 0.80 | 91 | 2112 | 1.38 |
| `large-v3-turbo` | Vulkan | 6.9 (6.9–6.9) | 3.3 | 0.42 | 84 | 2070 | 1.63 |
| `large-v3-turbo-q5_0` | CUDA | 6.6 (6.5–6.6) | 5.0 | 0.95 | 89 | 1121 | 0.63 |
| `large-v3-turbo-q5_0` | Vulkan | 6.5 (6.5–6.5) | 5.7 | 0.39 | 85 | 1071 | 0.79 |
| `large-v3` | CUDA | 2.7 (2.7–2.7) | 5.7 | 0.87 | 92 | 3795 | 2.43 |
| `large-v3` | Vulkan | 3.0 (3.0–3.0) | 6.5 | 0.41 | 87 | 3754 | 2.99 |

Streaming hears the large models as well as hearing at once: turbo 3.3–3.5%
against 3.3%, `large-v3` 5.7% against 6.9%. The small ones it hears worse
or better by a sentence: whisper passes a whole sentence over now and then,
whichever way it hears, and which one moves with every round — on 26
sentences one is several points (see [findings](findings.md#a-clip-of-26-sentences-is-too-few-to-judge-a-change-by)).
`base` streams worst of all, at twice its error at once; for Korean it is
no use either way.

## S3: the same in English

Streamed, the long English clip (303 s), through CUDA.

| Model | × real time | WER % | CPU cores | GPU % | GPU MiB |
|---|---:|---:|---:|---:|---:|
| `tiny` | 27.6 (27.5–27.9) | 21.3 | 1.20 | 51 | 375 |
| `base` | 22.9 (22.8–23.0) | 22.8 | 1.18 | 62 | 487 |
| `small` | 10.7 (10.7–10.7) | 18.7 | 1.07 | 77 | 925 |
| `medium` | 5.0 (5.0–5.0) | 14.0 | 0.99 | 90 | 2109 |
| `large-v3-turbo` | 6.3 (6.3–6.3) | 8.5 | 0.80 | 90 | 2113 |
| `large-v3-turbo-q5_0` | 7.4 (7.4–7.4) | 9.9 | 0.95 | 92 | 1121 |
| `large-v3` | 3.6 (3.6–3.6) | 11.9 | 0.87 | 92 | 3795 |

The order is Korean's. Of turbo's 8.5%, most is what any transcript would
be marked down for: digits where words were said ("100%", "11th"), names
spelled as heard ("Shandra Shankar Solansky" for Chandra Shekhar Solanki,
"tuba tuba tsi ai" for Tupua Tamasese Lealofi), American spelling, and a
few small words; it passes no sentence over.

## S4: the language, and how words are timed

`large-v3-turbo`, CUDA, the long Korean clip.

| Configuration | × real time | CER % | GPU % |
|---|---:|---:|---:|
| Language detected | 3.3 (3.3–3.3) | 3.5 | 93 |
| Language given | 5.7 (5.7–5.8) | 3.5 | 90 |
| Language given, words aligned | 5.2 (5.2–5.3) | 3.5 | 88 |

Detecting the language, every round, costs 42% of the speed and here
nothing else; giving it is also what keeps a stretch of music from being
heard as another language. Aligning each word against the audio
(`TokenTiming::Aligned`) costs 9% and hears the same: it times the lines,
not what is in them.

## S5: how often each round comes

`large-v3-turbo`, CUDA, the long Korean clip.

| Round every | × real time | CER % | GPU % |
|---|---:|---:|---:|
| 2 s | 3.0 (3.0–3.0) | 5.9 | 91 |
| 4 s (the default) | 5.7 (5.6–5.8) | 3.5 | 90 |
| 8 s | 9.8 (9.6–9.9) | 5.1 | 87 |

A round costs much the same however much is new in it, since the encoder
hears thirty seconds either way, so twice as many rounds is twice the work.
How well each hears is within the sentence-or-two that moves between any
two runs.

## S6: live

The short Korean clip handed over as it would arrive live, through a
`Pacer`. Lag is how long after the last word of a line was said the line
came out.

| Model | Round every | Lag, s | Lag p90, s | Lag max, s | CER % | CPU cores | GPU % | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base`, CUDA | 4 s | 8.87 | 17.69 | 21.79 | 15.0 | 0.11 | 10 | 487 |
| `small`, CUDA | 4 s | 8.67 | 17.27 | 23.24 | 12.5 | 0.16 | 13 | 925 |
| `large-v3-turbo`, CUDA | 4 s | 7.65 | 10.38 | 17.06 | 4.0 | 0.15 | 22 | 2113 |
| `large-v3`, CUDA | 4 s | 10.70 | 20.51 | 22.34 | 5.5 | 0.30 | 37 | 3795 |
| `large-v3-turbo`, CUDA | 2 s | 3.94 | 5.91 | 6.57 | 3.7 | 0.27 | 29 | 2113 |
| `large-v3-turbo`, CUDA | 8 s | 11.06 | 18.08 | 19.29 | 3.7 | 0.09 | 10 | 2113 |
| `base`, CPU | 4 s | 12.58 | 22.81 | 24.08 | 11.0 | 1.71 | — | — |

A word is reported once two rounds agree on it, so a line trails the
speech by one to two rounds and whatever holds it after — a line ends at
a sentence's end or a pause, and the small models, whose rounds disagree
more, hold theirs longer. With a round every two seconds turbo's lines
come out four seconds behind and never seven, for 29% of the GPU: the
setting for captions on a screen. Four seconds, the default, suits a
recording, where a line's place is its timestamp and the GPU is wanted
for other things.

## Before the fixes

The same clip and model through the transcriber as it was before this
benchmark — rounds cut by time (`4005792`, the first that reads the
clips) — and after each fix, on the long Korean clip:

| `large-v3-turbo`, CUDA | Streamed CER | Streamed, words aligned |
|---|---:|---:|
| Heard at once | 3.3% | |
| `4005792`, rounds cut by time | 25.0% | 9.6% |
| `8724cdb`, rounds agreed by text | 7.5% | 7.1% |
| `ef824cd`, pauses for cuts, repeats passed over | 4.6% | 5.3% |
| `9ac0142`, silence heard as well as agreed | 3.5% | 3.5% |

What each run turned up is in [findings.md](findings.md).
