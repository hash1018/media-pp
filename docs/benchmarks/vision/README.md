# Vision benchmarks

How fast media-pp's video analysis runs, what each part of it costs, and
how close it comes to what the GPU can do. The comparisons are the ones
DeepStream is judged by:

- who runs the model, and at what precision;
- one stream against several batched;
- detecting every picture against a tracker filling in;
- a second model on what was found;
- the boxes drawn and encoded.

| Page | What is in it |
|---|---|
| [method.md](method.md) | The harness, the clips and models and how to make them, how runs are made comparable, and the experiments E0–E9 |
| [linux-rtx3050.md](linux-rtx3050.md) | The machine, and every experiment's results on it, each read |
| [findings.md](findings.md) | What measuring turned up: the classifier's cost and its fix, a change that did nothing, engine caching, and the hardware's ceilings |
| [macos.md](macos.md) | What it takes to run the same experiments on Apple silicon, and where its results go |

## In short

On an RTX 3050 with YOLO11n through TensorRT in half precision:

| | Pictures a second, 1080p |
|---|---:|
| The model alone, one picture at a time | 673 |
| One stream: decode, detect | 544 |
| One stream, detecting one picture in two, tracked between | 768 — NVDEC's limit |
| Eight streams batched through one detector | 705 — 82% of what NVDEC decodes |
| One to eight streams: interval 2, tracker, classifier, overlay, NVENC | 441–443 — NVENC's limit (448) |

The model sets the pace of one stream; the GPU's decoder and encoder set
the pace of several, as they would for DeepStream on the same GPU.
DeepStream itself was not run: installing it needs root or Docker, which
the machine did not have. The model alone through the same TensorRT
([E0](linux-rtx3050.md#e0-the-model-alone)) is the bound it could reach,
and every table is read against it.

## Running it

```sh
cargo build --release -p vision_bench --bins
python3 examples/vision/vision_bench/bench.py --dir path/to/clips-and-models \
    --arm this=path/to/builds
```

[method.md](method.md#running-it) has the whole of it, and
[macos.md](macos.md) what a Mac needs first.
