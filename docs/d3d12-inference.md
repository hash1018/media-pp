# Detection on D3D12: DirectML

Status: stages 0 to 2 done, 2026-10-15 — this document and its
measurements, `D3d12OrtDetector` and `D3d12OrtEmbedder`, and beside them
`D3d12OrtClassifier`. Stage 3, gamchum on a GPU without NVIDIA, is set
aside: gamchum stays on CUDA for now.

Built as designed but for one thing found on the way: `ort`'s
`TensorRefMut::from_raw` locates every tensor it makes in host memory
(`MemoryInfo::to_owned` in `ort` 2.0.0-rc.13 makes a CPU one whatever it is
called on), and ONNX Runtime copied the DirectML allocation "up" as though
it were host memory — an access violation. The detector makes the input
tensor through ONNX Runtime's own `CreateTensorWithDataAsOrtValue` instead.
Through the element, RetinaFace R50 in fp16 ran a 1080p picture whole and
in six tiles in 47 ms on the RTX 3050 — 6.7 ms an input — and found the
same faces as the CPU's fp32 on the pictures compared. The embedder's
warp (`shaders/d3d12/warp.hlsl`) reads what the CPU works out for the same
maps, and on DirectML gamchum's SFace and OSNet made vectors 0.9975 and
0.9999 alike the CPU's of the same objects.

## Why

Every GPU detector this crate has runs on one vendor's stack:
`CudaOrtDetector` on NVIDIA (CUDA, TensorRT), `MetalOrtDetector` on Apple
silicon (Core ML). A Windows PC without an NVIDIA GPU — most laptops, every
AMD or Intel desktop GPU — can decode on the GPU (`D3d12Decoder`,
`D3d11Decoder`) but has to bring every picture down and detect on the CPU
(`SwOrtDetector`): about 33 ms a 640×640 input for YOLO11n and 250 ms for
RetinaFace R50 on an i5-12400F. An application that analyses whole videos —
gamchum — cannot be offered to such a machine at all.

DirectML runs ONNX models on any D3D12 GPU, through each vendor's own
driver. ONNX Runtime has an execution provider for it, the `ort` crate
exposes it (`ort/directml`), and the prebuilt ONNX Runtime this crate
already fetches for `ort-tensorrt` (`cuda13,tensorrt,nvrtx,directml`)
contains it. A DirectML-only build exists for machines without CUDA.

## Measured (stage 0)

The model alone, on a host tensor of 640×640 inputs, RTX 3050 (the only GPU
here), ONNX Runtime 1.28, a debug build of the probe. Milliseconds an input:

| Model | TensorRT fp16 | DirectML fp32 | DirectML fp16 | CPU |
|---|---:|---:|---:|---:|
| RetinaFace R50, batch 7 | 5.5 | 20.3 | **7.9** | 246 (batch 1) |
| YOLO11n, batch 7 | 2.2 | 4.6 | **3.9** | 33 |

- **DirectML is 1.4–1.8× TensorRT on NVIDIA**, and 9–30× the CPU. On AMD
  and Intel it cannot be measured here; vendors supply their own DirectML
  metacommands, so the ratio there is unknown.
- **The model has to be fp16 itself.** TensorRT turns fp32 into fp16 when
  told; DirectML runs what the graph says. R50 went from 20.3 to 7.9 ms
  converted (`onnxruntime.transformers.float16`, inputs and outputs kept
  fp32).
- **Opset 22 runs on the CPU.** The YOLO11n export gamchum uses is opset 22;
  DirectML had no kernels for it and took 99–114 ms an input, slower than
  the CPU, until the model was taken to opset 20 (`onnx.version_converter`,
  after `onnxslim` folded its shape arithmetic for a fixed input). Opset 20
  is what this ONNX Runtime's DirectML provider covers.
- **Shapes should be fixed.** DirectML is fastest with every dimension known
  when the session is made; ONNX Runtime's dimension overrides do that for
  an open batch, height and width without editing the model.

## Choice: DirectML, not Windows ML

Microsoft has put DirectML in maintenance; its successor, Windows ML, needs
Windows 11 24H2 and provisions vendor execution providers itself. DirectML
is chosen now because it runs on Windows 10 and every Windows 11, ships as
one DLL beside the program, and is already in the ONNX Runtime build this
crate uses. The input side below — a D3D12 buffer filled on the GPU — is the
same for either, so moving to Windows ML later changes the session, not the
element.

## The element: `D3d12OrtDetector`

Backend-prefixed per the naming rule, the D3D12 sibling of
`CudaOrtDetector`, built on `D3d12Gpu`:

```text
D3d12Decoder ─▶ D3d12OrtDetector ─▶ … (the picture passed on, Detections attached)
                 │ fit: compute shader, picture or tile → letterboxed NCHW fp32 in a D3D12 buffer
                 │ run: ONNX Runtime, DirectML, on the same ID3D12Device and queue
                 └ decode: the crate's model decoders (YOLO, RetinaFace, DETR, custom), tiles merged
```

- **One device.** The DirectML provider is appended with
  `OrtDmlApi::SessionOptionsAppendExecutionProvider_DML1`, given an
  `IDMLDevice` made from `D3d12Gpu`'s device (`DMLCreateDevice`) and
  `D3d12Gpu`'s queue. ONNX Runtime then runs on the queue the decoder's
  pictures are on, and a buffer this element fills can be handed to it as a
  tensor (`ort::ep::directml::resource_from_d3d`) through an I/O binding —
  nothing goes through host memory on the way in. `ort`'s own `DirectML`
  builder takes only an adapter index, which would give ONNX Runtime a
  device of its own.
- **Fit on the GPU.** A compute shader samples the decoder's NV12 or P010
  texture (and its array slice), letterboxes the whole picture or a tile
  into the model's input with grey, converts to the model's channel order,
  scale and bias (`ModelInput`), and writes fp32 NCHW — what
  `CudaOrtDetector`'s fit kernel does, ported. The picture is decoded once
  and looked at whole and in tiles in one batch, `Tiles` and `TileChooser`
  as on CUDA.
- **Outputs to the host** at first: detections are decoded and merged on the
  CPU by the code the other detectors share. A YOLO batch of 7 is 20 MB,
  RetinaFace's 7.5 MB; if copying them shows, decoding moves to a shader.
- **Ordering.** The fit is recorded on the shared queue ahead of the
  inference, so the queue orders them; a fence the element waits on says
  when the outputs may be read.
- **The model.** Opset 20 or lower, fp16 inside for speed. The element
  overrides the open batch, height and width with what it runs; it does
  not convert a model, and a model with ops DirectML lacks is logged as
  running partly on the CPU rather than refused.
- **Contract:** `PortContract::Frames` in `MemoryDomain::D3d12`, layouts
  NV12 and P010 (what the fit shader reads); the output passes the picture
  on unchanged (`OutputContract::SameLayout`).
- **Feature:** `ort-directml = ["ort", "ort/directml"]`, Windows only, beside
  `ort-tensorrt`. Both together select the one prebuilt that has both.

## Stages

1. **`D3d12OrtDetector`**: device sharing, the fit shader, I/O binding,
   decode and tiles; tested against `SwOrtDetector` on the same pictures
   (boxes within a tolerance) and with an `ort-directml` CI job.
2. **`D3d12OrtEmbedder`**: the same input path with the embedders' crop and
   alignment (five points or box), for gamchum's face and clothes vectors.
   `D3d12OrtClassifier` came after, the classifiers' box cut through the
   detector's fit shader.
3. **gamchum on a GPU without NVIDIA**: analysis chooses D3D12 where CUDA is
   absent. Its other CUDA-only parts need an answer each: scene cuts
   (`CudaCutDetector`; a D3D12 one, or the software one on small copies),
   the tracker's visual following (it has a CPU path), the export's
   redaction overlay (software after a download, or a D3D12 one) and its
   encoder (Media Foundation's, through D3D11). Planned in gamchum, after
   stage 2.

## Testing without a GPU

GitHub's Windows runners have none, but D3D12 has a software adapter (WARP)
and DirectML runs on it. CI can check correctness there, slowly; speed is
measured on the RTX 3050 here, and on AMD or Intel only by someone who has
one.

## Open questions

- Whether `DMLCreateDevice` on the decoder's device works on every vendor's
  driver as on NVIDIA's, and whether sharing the queue with video decoding
  costs either side; a second queue with a fence is the fallback.
- Whether DirectML's fp16 changes the detections enough to move gamchum's
  missed-face figure (10.0% on its eleven scored clips with TensorRT); to be
  measured with its `plan_trial.py` once stage 2 is in.
