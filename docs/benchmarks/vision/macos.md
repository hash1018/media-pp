# Running the vision benchmarks on macOS

The same experiments on Apple silicon, so the tables can be set beside
[the Linux ones](linux-rtx3050.md). The clips, the models and `bench.py`'s
method are the same — see [method.md](method.md); the backend differs, and
the GPU's counters. The results of one Mac are in
[macos-m5-air.md](macos-m5-air.md).

## The harness on a Mac

`vision_bench` builds on macOS with a `metal` backend, its default there,
beside `cpu`; each stage takes the macOS element where Linux takes the
CUDA one:

| Stage | Linux (`tensorrt`) | macOS (`metal`) |
|---|---|---|
| Decoder | `CudaDecoder` | `VideoToolboxDecoder` |
| Detector | `CudaOrtDetector` | `MetalOrtDetector`, `max_batch` from `--batch` |
| Tracker | `ObjectTracker` | `ObjectTracker` — following by look on the GPU with Metal |
| Classifier | `CudaOrtClassifier` | `MetalOrtClassifier` |
| Overlay | `CudaDetectionOverlay` | `MetalDetectionOverlay` |
| Encoder | `CudaEncoder`, H.264 8 Mbit/s | `VideoToolboxEncoder`, the same |

Its `RESULT` line reads `decode=videotoolbox`, `backend=metal` and, for
the precision, the compute units Core ML was allowed, so `bench.py` reads
it unchanged; on a Mac `bench.py` runs the matrix below in place of the
Linux one.

Three things in the matrix change meaning:

- **`--fp32` and the engine cache** are TensorRT's. Core ML picks its own
  precision; what it is told is where it may run, `--compute-units
  all|gpu|ane` — `MetalOrtDetectorOptions::compute_units`, a
  `CoreMlComputeUnits`. The Neural Engine against the GPU is the
  comparison that matters on a Mac.
- **`--track visual` on the CPU.** Metal follows VideoToolbox pictures by
  look on the GPU with no feature to turn that off, so E6 has no CPU row
  on a Mac.
- **E0's ceiling.** `model_only --provider coreml [--compute-units …]`
  runs the model as `MetalOrtDetector` sets it up, its input a tensor in
  memory: the model's rate plus the copy Core ML makes of its input on
  every run, which the detector pays as well.

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

[macos-m5-air.md](macos-m5-air.md): an M5 MacBook Air, which has no fan —
every table there is the chip's sustained, throttled rate.
