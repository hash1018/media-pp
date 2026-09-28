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
                                                       │
                                        (--sidecar) FileMuxer, .srt or .vtt
```

The picture and the sound are copied through as packets. Only the audio is
decoded, and only to be resampled into what Whisper reads: 16 kHz mono f32.

```sh
cargo run -p transcribe --release -- model.bin input.mp4 [output.mp4]
cargo run -p transcribe --release --features gpu -- --language ko --align model.bin input.mp4
cargo run -p transcribe --release --features gpu -- --sidecar out.srt model.bin input.mp4
```

The output defaults to `transcribed.mp4`, and is only ever an MP4, since
`mov_text` is the text track MP4 takes. `--release` matters: a debug build of
the inference is slower than real time even on a GPU.

- `--language` skips detection, which otherwise runs again for every chunk —
  60% more time on the measurement below — and can hear music as another
  language. See `WhisperTranscriber::with_language`.
- `--align` times each word against the audio instead of taking whisper.cpp's
  estimate, which is what cuts lines cleanly between chunks, for about 5%
  more time. See `TokenTiming`.
- `--sidecar out.srt|out.vtt` also writes the lines to a subtitle file of
  their own, as they arrive.

## The model

A whisper.cpp GGML file, from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).
`ggml-base.bin` (148 MB) is enough to see it work; `ggml-large-v3-turbo.bin`
(1.6 GB) is what gets a language other than English right. `ggml-small.bin`
(488 MB) is faster and mishears more: on Korean it wrote 알프스 삼백 for
알프스 산맥.

## CPU or GPU

`--features gpu` builds whisper.cpp's Vulkan backend (`whisper-vulkan`). On
an i5-12400F with an RTX 3050, against a minute of dense Korean speech with
`--language ko`:

| model | CPU | Vulkan |
|---|---|---|
| `small` | — | 9.7x real time |
| `large-v3-turbo` | slower than 0.07x — stopped after 14 minutes | 6.4x |

A whole file in one pass is several times faster; the streaming loop pays for
a 30-second encoder window to hear each few seconds (see `ChunkPolicy`). The
first GPU run also waits for the driver to compile the shaders, which is
cached after. whisper.cpp falls back to the CPU silently where it finds no
GPU; its log says `whisper_backend_init_gpu: using Vulkan0 backend` when it
does.

Building it on Windows needs long paths enabled (from an elevated prompt:
`reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f`)
and a short target directory, such as `set CARGO_TARGET_DIR=C:\t`: MSBuild's
`FileTracker` ignores long paths and fails with FTK1011. Linux needs neither.

## Why the subtitles arrive late

`WhisperTranscriber` gathers a few seconds before transcribing and holds back
the newest second as too uncertain to report, so at the end of a file the
last lines arrive after every video and audio packet has been muxed. That is
fine: a sample's place in an MP4 is its timestamp, not its arrival. The
`Queue` before the transcriber keeps the waiting off the demuxer's thread.
