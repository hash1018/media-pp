# How the speech benchmarks are measured

Everything here is reproducible from the repository and public files: the
harness is [`examples/speech/speech_bench`](../../../examples/speech/speech_bench),
the clips are made from Google's FLEURS test sets, and the models are
whisper.cpp's own conversions of OpenAI's Whisper.

## The harness

| Piece | What it does |
|---|---|
| `speech_bench` | Builds the pipeline its flags describe, plays a clip through it, writes what was heard to a file, and prints one `RESULT key=value …` line. |
| `model_only` | Hands the whole clip to whisper.cpp in one call, as its own command-line tool would: the ceiling the streaming transcribers are held against, in speed and in what is heard. |
| `make_clips.py` | Makes the clips and what was said in them from the FLEURS test sets. |
| `bench.py` | Runs the whole matrix below, each configuration several times, scores what each heard, and prints a Markdown table for each experiment. |

`speech_bench`'s pipeline, the bracketed stage there only when asked for:

```text
FileDemuxer -> SwDecoder -> AudioResampler -> [Pacer] -> Queue
    -> XxxWhisperTranscriber -> AppSink (collects the lines)
```

| Flag | Meaning |
|---|---|
| `--model m.bin` | The whisper.cpp model. |
| `--backend cpu\|cuda\|vulkan\|metal` | `SwWhisperTranscriber`, or `CudaWhisperTranscriber`, `VulkanWhisperTranscriber`, `MetalWhisperTranscriber` on the backend's first GPU. Each GPU backend is there only in a build with its feature. |
| `--language ko` | The language, which is otherwise detected for every chunk. |
| `--align` | Each word timed against the audio (`TokenTiming::Aligned`) rather than by whisper.cpp's estimate. |
| `--chunk-ms N` | `ChunkPolicy::chunk_ms`: how much new sound each round waits for, 4000 by default. |
| `--threads N` | whisper.cpp's CPU threads. |
| `--realtime` | A `Pacer` hands the sound over as it would arrive live. |
| `--transcript out.txt` | Where what was heard is written, a line each. |

What a `RESULT` line says:

| Key | Meaning |
|---|---|
| `speed` | Seconds of sound heard a second: 2.0 is twice real time. For `speech_bench` from the pipeline's start to its end; for `model_only` the inference alone, the clip already decoded. |
| `cpu` | The process's CPU time over the same span, in cores: 1.00 is one core kept busy. `getrusage` on Unix, `GetProcessTimes` on Windows. |
| `ready` | Seconds to load the model. |
| `lines` | Lines heard. |
| `lag`, `lag_p90`, `lag_max` | With `--realtime`: how long after the last word of a line was said the line came out, in seconds — the mean, the 90th percentile and the longest. What a viewer of live captions waits. |

`bench.py` adds the error rates and, from `nvidia-smi` every 200 ms after a
run's first second, the GPU's utilisation and the memory the run added to
what was in use before it started.

### What is scored

The character error rate (CER) for Korean and the word error rate (WER)
for English: the fewest insertions, deletions and substitutions that turn
what was heard into what was said, over the length of what was said.
Korean is spaced loosely enough — 할 수 있다 and 할수있다 are both seen —
that its words are a poor unit, so its CER leaves the spaces out. Before
either is counted, both sides are NFKC-normalised, lower-cased and
reduced to letters and digits, so punctuation and capitals cost nothing.
A number heard as digits where it was said in words — 100% for
"one-hundred percent" — does count, as it would for anyone comparing the
text.

Whisper decodes greedily here, as the transcribers do, so a run hears the
same every time on the same hardware: the error rate is a property of the
configuration, and only its speed is a median.

### How the numbers are made trustworthy

- **A warm pass first.** Every configuration runs once, untimed, before any
  is measured: a GPU backend's first run on a machine compiles its shaders
  or kernels for the card — over a minute for Vulkan — and the model files
  come into the page cache.
- **Turns, not blocks.** Each configuration runs three times, the
  configurations taking turns — a, b, c, a, b, c — so a GPU warming up or
  something starting on the machine falls on all of them alike. The tables
  give the median and the lowest and highest of the three.
- **Nothing else running.** No build or other GPU work while the matrix
  runs: the CPU rows are as sensitive to a compiler as the GPU rows are to
  another process's kernels.

## The clips

[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) reads
the same sentences, from Wikipedia articles, aloud in a hundred languages,
one sentence a recording, each with what was said written down. A clip is
a run of the test set's recordings back to back, in the order its
`test.tsv` lists them, half a second of silence between, each sentence
once.

Most sentences are read by two speakers, and many recordings are all but
silent — the English ones especially, about half of them 40 dB below the
rest, at −60 to −67 dBFS. Whisper passes such a sentence over whole, heard
at once or streamed, so it measures the recording rather than the
transcriber: the loudest recording of each sentence is taken, and a
sentence whose loudest is below −55 dBFS RMS is left out.

| Clip | Language | Sentences | Left out as too quiet | Length |
|---|---|---:|---:|---:|
| `ko-short.wav` | Korean | 5 | 0 | 75.1 s |
| `ko-long.wav` | Korean | 23 | 3 | 316.4 s |
| `en-short.wav` | English (US) | 5 | 5 | 68.6 s |
| `en-long.wav` | English (US) | 26 | 21 | 302.7 s |

Each is 32-bit float 16 kHz mono WAV, which is what the recordings are:
nothing is re-encoded, so every machine makes the same bytes, and the
short clip is the long one's start. Beside each is a `.txt` of what was
said in it, a sentence a line. The test sets, 214 MB and 290 MB, are the
same on every system:

```sh
mkdir -p fleurs/ko_kr fleurs/en_us
for lang in ko_kr en_us; do
  base=https://huggingface.co/datasets/google/fleurs/resolve/main/data/$lang
  curl -L -o fleurs/$lang/test.tsv "$base/test.tsv"
  curl -L -o fleurs/$lang/test.tar.gz "$base/audio/test.tar.gz"
done
python3 examples/speech/speech_bench/make_clips.py --fleurs fleurs --out clips
```

That runs as it is in a Linux or macOS shell, and in Git Bash on Windows,
where Windows' own `curl.exe` serves as well; `make_clips.py` needs only
Python 3's standard library. The clips it makes hash to:

| Clip | SHA-256 |
|---|---|
| `ko-short.wav` | `43bcd9f71940553731bd4c3e6d24150c5a837ca4616f6c44f6873839d345ac2c` |
| `ko-long.wav` | `aaa29afa1f9a61a7850afc240f7ac808d03d963dc6f0f62fee13950a4a3ac482` |
| `en-short.wav` | `1a1bc9652db9ff071f0e0ab42072129e7ad6aa483e73904a2ff16a98d7fd74a0` |
| `en-long.wav` | `92bddbe47e88d70a7d05d298e42481d9445ba3d763f8158f21a98163e9b8374e` |

## The models

whisper.cpp's GGML files, from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp),
into the same directory as the clips:

```sh
for m in tiny base small medium large-v3-turbo large-v3-turbo-q5_0 large-v3; do
  curl -L -o clips/ggml-$m.bin \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$m.bin"
done
```

| Model | File | Parameters | Weights |
|---|---:|---:|---|
| `tiny` | 78 MB | 39 M | fp16 |
| `base` | 148 MB | 74 M | fp16 |
| `small` | 488 MB | 244 M | fp16 |
| `medium` | 1.5 GB | 769 M | fp16 |
| `large-v3-turbo` | 1.6 GB | 809 M | fp16 |
| `large-v3-turbo-q5_0` | 574 MB | 809 M | 5-bit, quantised by whisper.cpp |
| `large-v3` | 3.1 GB | 1550 M | fp16 |

`large-v3-turbo` is `large-v3` with its decoder cut from 32 layers to 4:
the same encoder, which is most of the work in a streaming loop, and a
decoder an eighth the size. The multilingual models are used throughout,
English included, since one model for both is what a caller picks.

## Running it

Build both binaries with every GPU backend the machine has, and point
`bench.py` at them:

```sh
cargo build --release -p speech_bench --bins --features cuda,vulkan   # macOS: --features metal
mkdir -p bin && cp target/release/{speech_bench,model_only} bin/
python3 examples/speech/speech_bench/bench.py --dir clips --bin bin \
    [--backends cpu,cuda,vulkan] [--reps 3] [--only S2,S6]
```

`--backends` defaults to `cpu,cuda,vulkan`, or `cpu,metal` on a Mac; the
first GPU in it is the one the single-backend experiments run on. Leave out
a backend the build or the machine lacks.

What each backend needs to build is in [`docs/features.md`](../../features.md)
under `whisper-cuda`, `whisper-vulkan` and `whisper-metal`:

- **Linux.** `cuda` needs the CUDA toolkit (`nvcc`, cuBLAS) where CMake
  finds it, `vulkan` the Vulkan SDK's headers and `glslc`. Nothing else.
- **macOS.** `metal` needs only Xcode's command-line tools; there is no CUDA
  or Vulkan there, and no `nvidia-smi`, so the GPU columns stay empty.
- **Windows.** `vulkan` needs the Vulkan SDK, long paths enabled and a
  short target directory (`CARGO_TARGET_DIR=C:\t`); `cuda` needs the CUDA
  toolkit — the redistributable archives of `nvcc`, `cudart`, `cublas` and
  `libnvvm` will do, named by `CUDA_PATH` — built from a Visual Studio
  developer prompt with `CMAKE_GENERATOR=Ninja` and, for an RTX 30,
  `CMAKE_CUDA_ARCHITECTURES=86`. Running needs `cudart64_13.dll` and the
  cuBLAS DLLs on `PATH`, and `PYTHONUTF8=1` where the system's code page is
  not UTF-8, as a Korean Windows's is not.

The whole matrix takes two and a half hours on an RTX 3050, most of it the
CPU rows, `large-v3` and the live rows, which take as long as their clip.
`--report results.csv` prints the tables of a finished run again.

## The experiments

| | Compares | Fixed |
|---|---|---|
| S0 | The model alone, the whole clip in one call: every model on the GPU, the small ones on the CPU | Korean |
| S1 | Who runs the model: CPU, CUDA, Vulkan | Korean, short clip; `tiny`, `base`, `small`, and `large-v3-turbo` on the GPUs |
| S2 | The model's size and precision, on each GPU | Korean, long clip |
| S3 | The same in English | English, long clip, the first GPU |
| S4 | The language detected or given; words timed by estimate or against the audio | `large-v3-turbo`, Korean, long clip |
| S5 | How much new sound each round hears: 2, 4 and 8 seconds | `large-v3-turbo`, Korean, long clip |
| S6 | Live: how long after a line was said it comes out | Korean, short clip, paced; several models, two chunk sizes, and `base` on the CPU |

Every experiment but S4 gives the language, as a caller who knows it
would.
