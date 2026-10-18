# Speech benchmarks: Linux, RTX 3050

Every experiment of [method.md](method.md) on the machine
[windows-rtx3050.md](windows-rtx3050.md) measured, booted into Ubuntu
instead: the same i5-12400F, the same RTX 3050, the same clips (their
SHA-256s match) and models. Measured on 2026-10-19 with media-pp at
`fa0f2ec`: `bench.py`, three runs of each configuration taking turns, the
median with the lowest and highest beside it. Error rates are the same
every run (decoding is greedy), so they stand alone. CPU cores are the
process's CPU time over the run, 1.00 being one core kept busy; GPU % and
MiB are `nvidia-smi`'s utilisation and the memory the run added, sampled
every 200 ms. The whole matrix took 2 h 26 min.

Korean is scored by CER, English by WER — see [method.md](method.md#what-is-scored).

## The machine

| | |
|---|---|
| CPU, memory | Intel Core i5-12400F (6 cores, 12 threads), 32 GB |
| GPU | NVIDIA GeForce RTX 3050 8 GB (sm_86), 115 W limit; it drives the desktop (GNOME on Wayland) |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-38, the balanced power profile |
| NVIDIA | Driver 595.91.07; `nvcc`, the CUDA runtime 13.2.86 and cuBLAS 13.4.2 from NVIDIA's pip wheels (below) |
| Vulkan | The loader and headers of Ubuntu's `libvulkan-dev` 1.4.341, `glslc` from the LunarG SDK 1.4.363 |
| whisper.cpp | 1.8.3, through `whisper-rs` 0.16 and `whisper-rs-sys` 0.15, `CMAKE_CUDA_ARCHITECTURES=86` |
| FFmpeg | 8.1.2 |
| Rust | 1.97.1, release profile |

The machine has no CUDA toolkit installed and no root to install one, so
the toolkit was put together in a directory of its own from NVIDIA's
wheels on PyPI — `nvidia-cuda-nvcc`, `-cuda-runtime`, `-nvvm`, `-cuda-crt`
and `-cuda-cccl` 13.2.86, and `nvidia-cublas` 13.4.2.4 — with `CUDA_PATH`
pointing at it. Two things `whisper-rs-sys` 0.15 expects of a Linux CUDA
build had to be met by hand: it looks for the libraries under
`/usr/local/cuda` and `/opt/cuda` only, so `LIBRARY_PATH` names the
wheels' `lib/` and `lib/stubs/`; and it links `culibos`, a static library
of the full toolkit that a dynamic cuBLAS does not use, which an empty
archive stands in for. A machine with NVIDIA's own toolkit under
`/usr/local/cuda` needs neither.

## Summary

- **The same order as Windows.** `large-v3-turbo` is the model to stream —
  3.5% of Korean's characters missed, 7.1x real time through Vulkan and
  5.9x through CUDA — `large-v3` misses more at half the speed, and the
  quantised turbo halves the memory for a little more missed.
- **The CPU is 40% faster under Linux.** `small` heard at once runs at 6.4x
  against Windows's 4.5x, and streamed at 0.9x against 0.6x; `tiny` and
  `base` gain the same. The CPU still cannot stream a model that hears
  Korean well.
- **The GPU gains least where it works hardest.** Through CUDA `tiny` and
  `base` run 14–36% faster than under Windows, `small` 12–16%, `medium`
  7–9% and the large models 2–7%; through Vulkan most rows are within a
  few percent either way. The large models load in half the time or less.
- **CUDA keeps more of the CPU busy here**: 1.05 cores beside the GPU where
  Windows kept 0.80. Vulkan is the same on both, 0.43.
- **Live captions are the same**: a round every two seconds puts turbo's
  lines 3.9 s behind, 6.5 s at worst.
- **One difference in what is heard**: CUDA hearing the long clip at once
  passes over one sentence that Vulkan here, and CUDA under Windows, hear —
  turbo 8.7% where Windows measured 3.3%. Streamed, the same model through
  the same CUDA hears it, at 3.5% as under Windows. See S0.

## S0: the model alone

The whole clip handed to whisper.cpp in one call. Korean, the long clip on
the GPU and the short one on the CPU.

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB | Load, s | Windows × |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny` | CUDA | 153.7 (153.4–154.0) | 18.2 | 1.30 | 66 | 392 | 0.29 | 123.4 |
| `base` | CUDA | 111.9 (111.7–112.1) | 13.2 | 1.22 | 72 | 504 | 0.23 | 91.7 |
| `small` | CUDA | 59.8 (59.7–59.8) | 15.3 | 1.12 | 86 | 942 | 0.30 | 51.4 |
| `medium` | CUDA | 23.4 (23.3–23.4) | 7.5 | 1.05 | 93 | 2126 | 0.53 | 21.4 |
| `large-v3-turbo` | CUDA | 29.1 (28.9–29.1) | 8.7 | 1.06 | 91 | 2130 | 0.61 | 28.0 |
| `large-v3-turbo-q5_0` | CUDA | 33.6 (33.5–33.6) | 3.4 | 1.07 | 94 | 1134 | 0.33 | 31.9 |
| `large-v3` | CUDA | 8.6 (8.6–8.6) | 7.7 | 1.02 | 93 | 3813 | 0.98 | 8.0 |
| `tiny` | CPU | 41.7 (41.5–41.9) | 16.8 | 3.75 | — | — | 0.13 | 29.4 |
| `base` | CPU | 21.4 (21.3–21.4) | 12.5 | 3.87 | — | — | 0.19 | 15.2 |
| `small` | CPU | 6.4 (6.4–6.4) | 6.2 | 3.95 | — | — | 0.37 | 4.5 |

The CPU rows hear exactly what they heard under Windows, and so do `base`,
`small` and `medium` on the GPU; `tiny` and the quantised turbo differ by a
few characters. `large-v3-turbo` through CUDA passes over one sentence whole — *한 실험용 백신이 에볼라
사망률을 줄일 수 있을 것처럼 보이지만…*, two lines of the transcript — and
hears the rest word for word as Vulkan does here, which misses 3.3% as CUDA
did under Windows. `large-v3` misses 7.7% against 6.9%. Greedy decoding
takes the likeliest token every time, so a difference in the last bits of
the GPU's arithmetic can tip it onto another path, and a sentence passed
over is the path whisper takes most readily (see
[findings](findings.md#the-same-clip-heard-on-another-system-can-lose-a-sentence)).
Swapping cuBLAS 13.4.2 for 13.3.0.5 or 13.2.2.2 changes nothing, so the
difference lies in the driver or in whisper.cpp's CUDA kernels as compiled
here.

The CPU rows are 40% faster than under Windows on the same processor, with
the same four cores kept busy.

## S1: who runs the model

Streamed, the short Korean clip (75 s).

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB | Windows × |
|---|---|---:|---:|---:|---:|---:|---:|
| `tiny` | CPU | 6.0 (6.0–6.0) | 24.5 | 3.76 | — | — | 4.2 |
| `tiny` | CUDA | 29.9 (29.8–29.9) | 24.5 | 1.26 | 59 | 392 | 24.1 |
| `tiny` | Vulkan | 31.0 (30.9–31.0) | 24.5 | 1.05 | 43 | 321 | 28.9 |
| `base` | CPU | 3.2 (3.2–3.2) | 11.4 | 3.90 | — | — | 2.2 |
| `base` | CUDA | 26.6 (26.6–26.7) | 15.0 | 1.20 | 70 | 504 | 22.1 |
| `base` | Vulkan | 28.2 (28.1–28.2) | 15.0 | 0.88 | 57 | 437 | 24.7 |
| `small` | CPU | 0.9 (0.8–0.9) | 12.5 | 3.97 | — | — | 0.6 |
| `small` | CUDA | 11.4 (11.4–11.4) | 12.5 | 1.10 | 82 | 942 | 10.0 |
| `small` | Vulkan | 11.5 (11.5–11.6) | 12.5 | 0.62 | 76 | 885 | 11.4 |
| `large-v3-turbo` | CUDA | 5.5 (5.5–5.5) | 4.0 | 1.05 | 93 | 2130 | 5.4 |
| `large-v3-turbo` | Vulkan | 6.8 (6.8–6.8) | 4.0 | 0.42 | 88 | 2085 | 6.7 |

The three backends hear the same; `small` is 12.5% on all three, as under
Windows. Vulkan stays ahead of CUDA, by less on the small models than
under Windows, since CUDA gains more here. `small` on the CPU comes to
0.9x real time — closer than Windows's 0.6x, and still short of keeping up.

## S2: the model's size and precision

Streamed, the long Korean clip (316 s), on each GPU backend.

| Model | Backend | × real time | CER % | CPU cores | GPU % | GPU MiB | Load, s | Windows ×, CER % |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tiny` | CUDA | 25.1 (25.0–25.1) | 21.4 | 1.24 | 57 | 392 | 0.21 | 22.1, 21.6 |
| `tiny` | Vulkan | 26.3 (26.1–26.4) | 19.1 | 1.01 | 49 | 320 | 0.20 | 26.3, 20.1 |
| `base` | CUDA | 24.2 (24.1–24.2) | 30.2 | 1.21 | 68 | 504 | 0.22 | 20.7, 30.8 |
| `base` | Vulkan | 25.4 (25.4–25.4) | 30.3 | 0.90 | 57 | 439 | 0.22 | 26.2, 26.3 |
| `small` | CUDA | 10.0 (10.0–10.0) | 10.7 | 1.11 | 83 | 942 | 0.31 | 8.9, 10.7 |
| `small` | Vulkan | 10.1 (10.0–10.1) | 10.0 | 0.63 | 72 | 886 | 0.33 | 9.7, 12.8 |
| `medium` | CUDA | 4.3 (4.3–4.3) | 8.1 | 1.04 | 92 | 2127 | 0.53 | 4.0, 8.1 |
| `medium` | Vulkan | 4.2 (4.2–4.2) | 6.9 | 0.47 | 84 | 2078 | 0.63 | 4.2, 8.8 |
| `large-v3-turbo` | CUDA | 5.9 (5.9–5.9) | 3.5 | 1.05 | 92 | 2131 | 0.61 | 5.8, 3.5 |
| `large-v3-turbo` | Vulkan | 7.1 (7.1–7.1) | 3.5 | 0.43 | 89 | 2086 | 0.70 | 6.9, 3.3 |
| `large-v3-turbo-q5_0` | CUDA | 6.8 (6.8–6.8) | 4.0 | 1.06 | 91 | 1137 | 0.33 | 6.6, 5.0 |
| `large-v3-turbo-q5_0` | Vulkan | 6.6 (6.6–6.6) | 5.9 | 0.41 | 92 | 1087 | 0.36 | 6.5, 5.7 |
| `large-v3` | CUDA | 2.8 (2.8–2.8) | 4.5 | 1.03 | 95 | 3813 | 1.01 | 2.7, 5.7 |
| `large-v3` | Vulkan | 2.9 (2.9–2.9) | 6.5 | 0.45 | 84 | 3772 | 1.18 | 3.0, 6.5 |

`large-v3-turbo` streams Korean at 3.5% on both backends, as under
Windows — including the sentence CUDA passes over when it hears the clip
at once. The others move a point or two either way between the two
systems — `large-v3` through CUDA and `medium` through Vulkan for the
better here, `base` through Vulkan for the worse — which on 23 sentences
is what passing over or hearing one more amounts to. Nothing here says one
system hears better; it says how far one sentence moves the figure (see
[findings](findings.md#a-clip-of-26-sentences-is-too-few-to-judge-a-change-by)).

## S3: the same in English

Streamed, the long English clip (303 s), through CUDA.

| Model | × real time | WER % | CPU cores | GPU % | GPU MiB | Windows ×, WER % |
|---|---:|---:|---:|---:|---:|---|
| `tiny` | 37.5 (37.2–37.6) | 25.8 | 1.31 | 58 | 392 | 27.6, 21.3 |
| `base` | 23.3 (23.3–23.4) | 15.4 | 1.20 | 69 | 504 | 22.9, 22.8 |
| `small` | 12.3 (12.3–12.3) | 18.7 | 1.10 | 84 | 942 | 10.7, 18.7 |
| `medium` | 5.4 (5.4–5.4) | 13.7 | 1.04 | 92 | 2127 | 5.0, 14.0 |
| `large-v3-turbo` | 6.7 (6.7–6.7) | 10.4 | 1.06 | 93 | 2131 | 6.3, 8.5 |
| `large-v3-turbo-q5_0` | 7.6 (7.6–7.6) | 9.0 | 1.06 | 91 | 1136 | 7.4, 9.9 |
| `large-v3` | 3.8 (3.8–3.8) | 11.9 | 1.03 | 94 | 3813 | 3.6, 11.9 |

The same order of the large models; on 26 sentences a few words is two
points, and turbo's 10.4% against 8.5% is that. The small models swing
further: `base` hears English much better here, 15.4% against 22.8%, and
`tiny` worse, 25.8% against 21.3% — most likely a sentence or two passed
over on one system and heard on the other, which the small models do most.
The transcripts were not compared line by line.

## S4: the language, and how words are timed

`large-v3-turbo`, CUDA, the long Korean clip.

| Configuration | × real time | CER % | GPU % | Windows ×, CER % |
|---|---:|---:|---:|---|
| Language detected | 3.4 (3.4–3.4) | 3.5 | 96 | 3.3, 3.5 |
| Language given | 5.9 (5.9–5.9) | 3.5 | 93 | 5.7, 3.5 |
| Language given, words aligned | 5.5 (5.5–5.5) | 2.8 | 92 | 5.2, 3.5 |

The same costs as under Windows: detecting the language every round 42%
of the speed, aligning each word 7% (Windows 9%). Aligned, it hears a
little more here, 2.8%.

## S5: how often each round comes

`large-v3-turbo`, CUDA, the long Korean clip.

| Round every | × real time | CER % | GPU % | Windows ×, CER % |
|---|---:|---:|---:|---|
| 2 s | 3.1 (3.1–3.1) | 5.9 | 93 | 3.0, 5.9 |
| 4 s (the default) | 5.9 (5.9–5.9) | 3.5 | 92 | 5.7, 3.5 |
| 8 s | 10.1 (10.1–10.1) | 5.1 | 91 | 9.8, 5.1 |

The same on both systems, error rates to the decimal.

## S6: live

The short Korean clip handed over as it would arrive live, through a
`Pacer`. Lag is how long after the last word of a line was said the line
came out.

| Model | Round every | Lag, s | Lag p90, s | Lag max, s | CER % | CPU cores | GPU % | GPU MiB | Windows lag, p90, max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base`, CUDA | 4 s | 8.60 | 17.39 | 21.49 | 15.0 | 0.06 | 4 | 504 | 8.87, 17.69, 21.79 |
| `small`, CUDA | 4 s | 8.39 | 16.92 | 22.89 | 12.5 | 0.12 | 8 | 942 | 8.67, 17.27, 23.24 |
| `large-v3-turbo`, CUDA | 4 s | 7.43 | 10.09 | 16.77 | 4.0 | 0.21 | 17 | 2131 | 7.65, 10.38, 17.06 |
| `large-v3`, CUDA | 4 s | 10.50 | 20.38 | 22.21 | 5.5 | 0.35 | 31 | 3813 | 10.70, 20.51, 22.34 |
| `large-v3-turbo`, CUDA | 2 s | 3.93 | 5.88 | 6.53 | 3.7 | 0.36 | 31 | 2131 | 3.94, 5.91, 6.57 |
| `large-v3-turbo`, CUDA | 8 s | 10.86 | 17.82 | 19.02 | 3.7 | 0.12 | 7 | 2131 | 11.06, 18.08, 19.29 |
| `base`, CPU | 4 s | 11.30 | 19.09 | 23.19 | 11.4 | 1.22 | — | — | 12.58, 22.81, 24.08 |

Lag is set by the rounds, not the system: every GPU row is within 0.3 s
of Windows's, a little ahead throughout. The CPU row
gains more, a second and a quarter, as the CPU's own speed does.
