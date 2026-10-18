# Speech benchmarks

How fast and how well media-pp's transcribers hear, what each model and
backend costs, and how far behind the speech live captions run. The
comparisons are the ones a caller has to make:

- which model, and at what precision;
- who runs it — the CPU, CUDA, Vulkan or Metal;
- the whole file heard at once against a stream heard as it arrives;
- how often each round comes, against how late its lines are.

| Page | What is in it |
|---|---|
| [method.md](method.md) | The harness, the clips and models and how to fetch them on Linux, macOS and Windows, what is scored, and the experiments S0–S6 |
| [windows-rtx3050.md](windows-rtx3050.md) | The machine, and every experiment's results on it, each read |
| [findings.md](findings.md) | What measuring turned up: words lost at every seam, a model repeating itself, cuts inside words, silence that was speech — each fixed — and what was tried and dropped |

## In short

On an RTX 3050, streaming FLEURS's Korean, scored by the characters it
misses:

| | × real time | CER | GPU memory |
|---|---:|---:|---:|
| `large-v3-turbo`, Vulkan | 6.9 | 3.3% | 2.1 GB |
| `large-v3-turbo`, CUDA | 5.8 | 3.5% | 2.1 GB |
| `large-v3-turbo-q5_0`, CUDA | 6.6 | 5.0% | 1.1 GB |
| `large-v3`, CUDA | 2.7 | 5.7% | 3.8 GB |
| `medium`, CUDA | 4.0 | 8.1% | 2.1 GB |
| `small`, CUDA | 8.9 | 10.7% | 0.9 GB |
| The whole clip at once, `large-v3-turbo` | 28.0 | 3.3% | 2.1 GB |

`large-v3-turbo` streams as well as it hears a file whole, and English at
8.5% WER. With a round every two seconds its lines come out 3.9 seconds
after they were said, 6.6 at worst, on 29% of the GPU. The CPU cannot
stream a model that hears Korean well: `small` runs at 0.6x real time.

Before the fixes [findings.md](findings.md) describes, the same stream
missed 25% of the characters.

Linux and macOS have not been measured yet; [method.md](method.md#running-it)
says how.

## Running it

```sh
cargo build --release -p speech_bench --bins --features cuda,vulkan   # macOS: --features metal
python3 examples/speech/speech_bench/make_clips.py --fleurs fleurs --out clips
python3 examples/speech/speech_bench/bench.py --dir clips --bin target/release
```

[method.md](method.md#the-clips) has the downloads for the clips and
models, and [the running of it](method.md#running-it) what each system
needs first.
