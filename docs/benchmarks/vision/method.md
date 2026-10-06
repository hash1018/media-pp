# How the vision benchmarks are measured

Everything here is reproducible from the repository and public files: the
harness is [`examples/vision/vision_bench`](../../../examples/vision/vision_bench),
the clips are made from one of Intel's public sample videos, and the models
are Ultralytics' and the ONNX model zoo's.

## The harness

| Piece | What it does |
|---|---|
| `vision_bench` | Builds the pipeline its flags describe, plays a file through it as fast as it goes, and prints one `RESULT key=value …` line. |
| `model_only` | Runs the detector alone through ONNX Runtime on a tensor already on the GPU, its output left there: the ceiling the pipelines are held against. `--cuda-graph` replays it as a CUDA graph; `--int8` runs a Q/DQ model in INT8, `--int8-table` a float one from a calibration table; `--opt`, `--max` build for a range of batches. Linux and Windows. |
| `bench.py` | Runs the whole matrix below, each configuration several times, and prints a Markdown table for each experiment. |

`vision_bench`'s pipeline, each bracketed stage there only when asked for:

```text
FileDemuxer -> decoder -> Queue -> [detector] -> [ObjectTracker] -> [classifier]
            -> [overlay] -> [Queue -> encoder] -> AppSink (counts)
```

With `--streams N --batch B` the streams meet in a `StreamMux`, one detector
runs up to B pictures at once, and the tracker and classifier follow it
before a demux hands each stream to its own overlay, encoder and sink —
DeepStream's `nvstreammux -> nvinfer -> nvtracker -> nvinfer(secondary) ->
nvstreamdemux`. With `--batch 0` each stream is a pipeline of its own with
a detector of its own.

| Flag | Meaning |
|---|---|
| `--model m.onnx` | The detector; without it, decoding alone is measured. |
| `--backend cpu\|cuda\|tensorrt` | `cpu`: `SwDecoder`, `SwOrtDetector`, `SwOrtClassifier`, `SwDetectionOverlay`. `cuda`/`tensorrt`: NVDEC, `CudaOrtDetector` on ONNX Runtime's CUDA or TensorRT provider, `CudaOrtClassifier`, `CudaDetectionOverlay`, NVENC. |
| `--fp32` | TensorRT in single precision; half precision otherwise. |
| `--streams N`, `--batch B` | N copies of the file; batched B at a time through a `StreamMux`, or `0` for a pipeline each. |
| `--interval N` | The detector looks at one picture in N+1 (DeepStream's `interval`). |
| `--track motion\|visual` | `ObjectTracker`: Kalman and matching, or that and correlation filters. `visual` follows on the GPU in a build with `--features gpu-dcf`, on the CPU without. |
| `--classifier m.onnx`, `--reclassify N` | A second model on each tracked object, its answer kept N pictures (0: for good, 1: asked again on every picture). |
| `--overlay`, `--encode` | Boxes and labels drawn; H.264 encoded at 8 Mbit/s. |
| `--pictures N`, `--warmup N` | Stop after N pictures a stream; leave the first N (all streams together) out of the rate. |
| `--engine-cache-root DIR` | Where TensorRT engines are kept, one directory per model, precision and batch. |

What a `RESULT` line says:

| Key | Meaning |
|---|---|
| `fps` | Pictures a second through the sinks, all streams together, after the warm-up. |
| `cpu` | The process's CPU time over the same span, in cores: 1.00 is one core kept busy. |
| `objects` | Objects on each picture that reached the sink, on average. |
| `found` | Objects over the whole run: two builds that should find the same find exactly the same number. |
| `ready` | Seconds to build the pipeline, a TensorRT engine's loading included. |

`bench.py` adds the GPU's utilisation, NVDEC's, NVENC's and the GPU memory
in use, from `nvidia-smi` every 200 ms after a run's first second.

### How the numbers are made trustworthy

- **A warm pass first.** Every configuration runs once, untimed, before
  any is measured: TensorRT builds an engine for each new model, precision
  and batch, which takes minutes. The engines are kept a directory each,
  because ONNX Runtime names an engine after the model's graph alone and
  two batches of one model would otherwise rebuild each other's on every
  run.
- **A warm-up inside each run.** The first 60 pictures (40 on the CPU) are
  not counted, so a provider's first runs and the decoder's start do not
  weigh on the rate.
- **Turns, not blocks.** Each configuration runs three times, the
  configurations taking turns — a, b, c, a, b, c — so a GPU warming up or
  something starting on the machine falls on all of them alike. The tables
  give the median and the lowest and highest of the three.
- **Arms inside one run.** Comparing two builds of media-pp, `bench.py`
  runs each configuration on both, one after the other, swapping which
  goes first each time round. Two separate runs on this kind of machine
  can differ by more than the change being measured; two arms that meet
  the same machine minute by minute cannot.

## The clips

Intel's [`people-detection.mp4`](https://github.com/intel-iot-devkit/sample-videos)
— people walking through a hall, 768×432 at 12 fps, 596 pictures — made
into three clips of the same content at three sizes, retimed to 30 fps and
looped to be long enough to measure:

```sh
url=https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
curl -L -o people-detection.mp4 "$url"
enc="-c:v h264_nvenc -profile:v high"      # macOS: -c:v h264_videotoolbox -profile:v high
ffmpeg -stream_loop 4 -i people-detection.mp4 -vf "setpts=N/(30*TB),format=nv12" \
    -r 30 -an $enc -b:v 2M -g 60 -bf 0 people-432p.mp4
ffmpeg -stream_loop 4 -i people-detection.mp4 \
    -vf "setpts=N/(30*TB),scale=1920:1080:flags=bicubic,format=nv12" \
    -r 30 -an $enc -b:v 8M -maxrate 8M -bufsize 16M -g 60 -bf 0 people-1080p.mp4
ffmpeg -stream_loop 1 -i people-detection.mp4 \
    -vf "setpts=N/(30*TB),scale=3840:2160:flags=bicubic,format=nv12" \
    -r 30 -an $enc -b:v 32M -maxrate 32M -bufsize 64M -g 60 -bf 0 people-2160p.mp4
```

| Clip | Size | Pictures | Bit rate |
|---|---|---:|---:|
| `people-432p.mp4` | 768×432 | 2982 | 2 Mbit/s |
| `people-1080p.mp4` | 1920×1080 | 2982 | 8 Mbit/s |
| `people-2160p.mp4` | 3840×2160 | 1194 | 30 Mbit/s |

H.264 High, a key picture every 60, no B-pictures. Another machine's
encoder makes a slightly different file of the same size, rate and
structure, which decodes at much the same speed.

## The models

```sh
python3 -m venv venv && venv/bin/pip install ultralytics onnx onnxslim
venv/bin/yolo export model=yolo11n.pt format=onnx dynamic=True imgsz=640
venv/bin/yolo export model=yolo11s.pt format=onnx dynamic=True imgsz=640
curl -L -o yolov10n.onnx https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx
curl -L -o mobilenetv2-12.onnx \
    https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
```

| Model | Role | Input | Batch |
|---|---|---|---|
| YOLO11n | Detector, the default | 640×640 | open |
| YOLO11s | Detector, about three times the work | 640×640 | open |
| YOLOv10n | Detector, no NMS | 640×640 | 1 |
| MobileNetV2 | Classifier on what was found | 224×224 | open |

The classifier is an ImageNet model, which knows no people: what is
measured is the classifying, not what it answers, and every answer is kept
however unsure, as an answer not kept is asked for again on the next
picture.

## Running it

```sh
cargo build --release -p vision_bench --bins
mkdir -p bin/arm && cp target/release/{vision_bench,model_only} bin/arm/
cargo build --release -p vision_bench --bin vision_bench --features gpu-dcf
cp target/release/vision_bench bin/arm/vision_bench-gpu-dcf
cp target/release/libonnxruntime_providers_*.so bin/arm/    # Linux; DLLs on Windows

python3 examples/vision/vision_bench/bench.py --dir path/to/clips-and-models \
    --arm this=bin/arm [--arm other=bin/other] [--reps 3] [--only E2,E4]
```

On Linux, building and running need CUDA 13, cuDNN 9 and TensorRT 10 where
the linker and the loader find them — see
[`docs/features.md`](../../features.md) for `ort-tensorrt`. The whole matrix
takes one to two hours on an RTX 3050, most of it the warm pass building
engines the first time.

On Windows the builds are `.exe`s, which `bench.py` looks for, and the
providers `onnxruntime_providers_{shared,cuda,tensorrt}.dll` go beside
them. Running needs the NVIDIA DLLs' directories on `PATH` — the
`bin\x64` of CUDA's and cuDNN's archives and TensorRT's `tensorrt_libs`, as
for [`cuda_detect`](../../../examples/vision/cuda_detect/README.md) — and
`PYTHONUTF8=1` where the system's code page is not UTF-8, as a Korean or
Japanese Windows's is not. A build started without those DLLs on `PATH`
stops at a system dialog that waits for a click rather than failing.

## The experiments

| | Compares | Fixed |
|---|---|---|
| E0 | The model alone: provider, precision, model, batch | — |
| E1 | Decoding alone: NVDEC against FFmpeg's software decoder, three sizes | — |
| E2 | Who runs the model: CPU, CUDA, TensorRT fp32 and fp16; three models | 1080p, one stream |
| E3 | The picture's size | YOLO11n, TensorRT fp16, one stream |
| E4 | Streams: 1, 2, 4, 8, a detector each against one batching them; decoding alone | 1080p, YOLO11n, TensorRT fp16 |
| E5 | Detecting one picture in 1, 2, 3, 5, the tracker filling in | 1080p, one stream and four batched |
| E6 | Tracking: none, motion, by look on the CPU, by look on the GPU | 1080p, every picture and one in five |
| E7 | A classifier after the tracker: answers kept 30 pictures, or asked every picture | 1080p, one stream and four batched |
| E8 | Boxes drawn and the result encoded | 1080p, one stream; 432p on the CPU |
| E9 | Everything at once — interval 2, tracker, classifier, overlay, NVENC | 1080p, 1, 4 and 8 streams |
