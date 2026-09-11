# transcribe

Transcribes a file's speech and writes a copy of it carrying the result as a
subtitle track — video, audio and text in one `.mp4`.

```text
                   ┌─ video packets ─────────────────────────────┐
                   │                                             │
FileDemuxer ───────┤              ┌─ audio packets ──────────────┤─ FileMuxer
                   │              │                              │
                   └─ audio ─ Tee ┤                              │
                                  └─ SwDecoder ─ AudioResampler ─┤
                                       ─ Queue ─ WhisperTranscriber
                                                       │         │
                                                    segments ────┘
```

The picture and the sound are copied through as packets — nothing is decoded
for them and nothing re-encoded. Only the audio is decoded, and only to be
resampled into the one shape Whisper reads: 16 kHz mono f32.

```sh
cargo run -p transcribe --release -- model.bin input.mp4 [output.mp4]
cargo run -p transcribe --release --features gpu -- --language ko model.bin input.mp4
```

`--release` matters more than usual here: a debug build of the inference is
slower than real time even on a GPU.

The language is detected unless `--language` names it. Name it where it is
known: detection runs again for every few seconds of audio, which cost 60%
more time on the measurement below, and a stretch of music can be heard as
a different language from the speech around it.

## The model

A whisper.cpp GGML file, from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).
Nothing that size belongs in a repository, so this takes a path. `ggml-base.bin`
(148 MB) is enough to see it work; `ggml-large-v3-turbo.bin` (1.6 GB) is what
gets a language other than English right. `ggml-small.bin` (488 MB) is faster
and mishears more: on Korean it wrote 알프스 삼백 for 알프스 산맥.

## CPU or GPU

`--features gpu` builds whisper.cpp's Vulkan backend, which needs the
[Vulkan SDK](https://vulkan.lunarg.com/) at build time — its shader compiler,
not its runtime, which ships with every graphics driver. Vulkan rather than
CUDA because CUDA needs a 3 GB toolkit to build and serves only NVIDIA.

The difference is not small. Measured with this example on an i5-12400F with
an RTX 3050, against a minute of dense Korean speech — nearly two words a
second — with `--language ko`:

| model | CPU | Vulkan |
|---|---|---|
| `small` | — | 9.7x real time |
| `large-v3-turbo` | slower than 0.07x — stopped after 14 minutes | 6.5x |

Transcribing a whole file in one pass is several times faster than this, and
the difference is the streaming loop rather than the model: every inference
pays for a whole 30-second encoder window to hear eight seconds of audio,
four of them context. That is the price of lines arriving while the audio is
still coming in, and on a CPU it is not affordable at all.

The first GPU run also pays for the driver to compile whisper.cpp's shaders,
which took over a minute here and under two seconds every run after — that
cache outlives the process.

### Building the GPU feature on Windows

ggml builds its Vulkan shader generator as a CMake ExternalProject, and the
paths that produces run past Windows' 260-character limit. Two things are
needed, and neither alone was enough here.

Long paths on, from an elevated prompt — without this MSBuild cannot create
the directories at all (MSB4018 / MSB6003):

```text
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

And a short target directory. MSBuild honours long paths once the setting is
on, but `FileTracker` — the native tool it logs file access with — does not,
and fails with FTK1011 from a target directory as ordinary as
`D:\Project\media-pp\target`:

```text
set CARGO_TARGET_DIR=C:\t
```

Turning `FileTracker` off with `TrackFileAccess=false` clears FTK1011 and
breaks the ExternalProject's step ordering instead, so it is not a way round
the second. The CPU build needs neither, and nor does Linux.

whisper.cpp falls back to the CPU without complaining when it finds no GPU, so
a run that seems inexplicably slow is worth checking against its own log:

```text
whisper_backend_init_gpu: using Vulkan0 backend
```

## Why the subtitles arrive late

`WhisperTranscriber` gathers a few seconds before transcribing any of them,
then holds back the newest second as too uncertain to report — a word only
half-heard is guessed at, and the guess changes once the rest of it arrives.
So a line for the fifth second is handed over while the tenth is being
written, and at the end of a file the last lines arrive after every video and
audio packet has already been muxed.

That is not something to work around. A packet's place in an MP4 is its
timestamp and not its arrival, and the index is written at the end, so a
sample landing late still lands where it belongs. The `Queue` before the
transcriber is what keeps the waiting off the demuxer's thread.

## What it does not do

Only MP4, because `mov_text` is what MP4 takes and this writes `mov_text`.
Matroska carries SRT and ASS as they are, so a caller wanting one of those
would not need `subtitle` at all.
