# Running the vision benchmarks on macOS

What it takes to measure the same experiments on Apple silicon, so the
tables can be set beside [the Linux ones](linux-rtx3050.md). The clips,
the models and `bench.py`'s method are the same — see
[method.md](method.md); what differs is the backend, which has yet to be
added to the harness, and the GPU's counters.

## What the harness needs

`vision_bench` builds on macOS today with the `cpu` backend alone. A
`metal` backend goes beside `cuda` and `tensorrt` in
[`src/main.rs`](../../../examples/vision/vision_bench/src/main.rs), each
of its `match args.backend` arms taking the macOS element where the CUDA
one is:

| Stage | Linux (`tensorrt`) | macOS (`metal`) |
|---|---|---|
| Decoder | `CudaDecoder` | `VideoToolboxDecoder` |
| Detector | `CudaOrtDetector` | `MetalOrtDetector`, `max_batch` from `--batch` |
| Tracker | `ObjectTracker` | `ObjectTracker` — following by look on the GPU with `metal` |
| Classifier | `CudaOrtClassifier` | `MetalOrtClassifier` |
| Overlay | `CudaDetectionOverlay` | `MetalDetectionOverlay` |
| Encoder | `CudaEncoder`, H.264 8 Mbit/s | `VideoToolboxEncoder`, the same |

with `media-pp`'s `ort-coreml` feature in a
`[target.'cfg(target_os = "macos")'.dependencies]` table of its
`Cargo.toml`. The `RESULT` line stays as it is — `decode=videotoolbox`,
`backend=metal`, `precision` as Core ML is asked for — so `bench.py` reads
it unchanged. `metal_multi_detect` and `metal_detect` already build each of
these stages and are the reference for how.

Three things in the matrix change meaning:

- **`--fp32` and the engine cache** are TensorRT's. Core ML has its own
  choice instead: which compute units it may use. A `--compute-units
  all|gpu|ane|cpu` flag (`MLComputeUnits`) is the comparison that matters
  on a Mac — the Neural Engine against the GPU — and needs
  `core_ml_session` in `infer/ort/metal/mod.rs` to take it; it currently
  leaves the choice to Core ML.
- **`--track visual` on the CPU.** `metal` follows VideoToolbox pictures by
  look on the GPU with no feature to turn that off, so E6's CPU row needs a
  way to ask for the CPU's — an option on `TrackerOptions`, or the pictures
  downloaded to system memory before the tracker.
- **E0's ceiling.** `model_only` runs ONNX Runtime's TensorRT and CUDA
  providers. A `--provider coreml` beside them, the session made as
  `core_ml_session` makes it and the input a tensor in memory, gives the
  Mac's ceiling — though Core ML copies its input in on every run, so it is
  the model's rate plus that copy, which `MetalOrtDetector` pays as well.

## The GPU's counters

`nvidia-smi` has no Mac counterpart that needs no root. `bench.py` leaves
the GPU columns empty where it finds no `nvidia-smi`; for the GPU's and the
Neural Engine's activity, `sudo powermetrics --samplers gpu_power,ane_power
-i 200` alongside a run is the closest, and is worth sampling as
`GpuSampler` samples `nvidia-smi`, keeping the mean after the first second.

## The matrix on a Mac

| | Linux | macOS |
|---|---|---|
| E0 | TensorRT fp16/fp32, CUDA; batch 1–8 | Core ML: all units, GPU only, Neural Engine only; batch 1–8 |
| E1 | NVDEC against software | VideoToolbox against software |
| E2 | CPU, CUDA, TensorRT fp32/fp16 | CPU, Core ML by compute units |
| E3–E5 | as written | as written, `--backend metal` |
| E6 | motion; by look on the CPU and the GPU | motion; by look on the CPU and with Metal |
| E7–E9 | NVENC | VideoToolbox's encoder |

The clips are made with `h264_videotoolbox` in place of `h264_nvenc`, as
[method.md](method.md#the-clips) shows; Homebrew's FFmpeg has it.

## Results

To be filled in from a run, as [linux-rtx3050.md](linux-rtx3050.md) is,
with the machine described first: the chip and its GPU cores, memory,
macOS version, ONNX Runtime and FFmpeg versions, and the media-pp commit.
