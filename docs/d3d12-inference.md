# D3D12에서 검출하기: DirectML

상태: 제안, 2026-10-15. 0단계(이 문서와 아래 측정)는 끝났다. 아래 내용은 아직
만들지 않았다.

## 왜

지금 GPU 검출기는 모두 한 회사의 스택에서만 돈다. `CudaOrtDetector`는 NVIDIA
(CUDA, TensorRT), `MetalOrtDetector`는 Apple 실리콘(Core ML)이다. NVIDIA GPU가 없는
Windows PC — 대부분의 노트북, AMD나 Intel GPU 데스크톱 — 는 디코딩은 GPU로 할 수
있지만(`D3d12Decoder`, `D3d11Decoder`) 모든 프레임을 내려받아 CPU에서 검출해야
한다(`SwOrtDetector`). i5-12400F에서 640×640 입력 하나에 YOLO11n이 약 33 ms,
RetinaFace R50이 250 ms다. 영상 전체를 분석하는 앱(감춤)은 이런 PC에서는 쓸 수
없다.

DirectML은 어떤 D3D12 GPU에서든 각 회사의 드라이버로 ONNX 모델을 돌린다. ONNX
Runtime에 실행기(execution provider)가 있고, `ort` 크레이트가 이를 연다
(`ort/directml`). 이 크레이트가 `ort-tensorrt`로 이미 받는 ONNX Runtime 빌드
(`cuda13,tensorrt,nvrtx,directml`)에 DirectML이 들어 있다. CUDA가 없는 PC용으로는
DirectML만 든 빌드가 따로 있다.

## 측정 (0단계)

모델만, 호스트 텐서로, 640×640 입력, RTX 3050(이 PC의 유일한 GPU), ONNX Runtime
1.28, 측정 프로그램은 디버그 빌드. 입력 하나당 밀리초:

| 모델 | TensorRT fp16 | DirectML fp32 | DirectML fp16 | CPU |
|---|---:|---:|---:|---:|
| RetinaFace R50, 배치 7 | 5.5 | 20.3 | **7.9** | 246 (배치 1) |
| YOLO11n, 배치 7 | 2.2 | 4.6 | **3.9** | 33 |

- **DirectML은 NVIDIA에서 TensorRT의 1.4–1.8배**, CPU보다 9–30배 빠르다. AMD와
  Intel에서는 여기서 잴 수 없다. 회사마다 DirectML 최적화(metacommand)를 따로
  넣으므로 그쪽 비율은 모른다.
- **모델 자체가 fp16이어야 한다.** TensorRT는 시키면 fp32를 fp16으로 돌리지만
  DirectML은 그래프에 적힌 대로 돌린다. R50은 변환하자 20.3 ms에서 7.9 ms가 됐다
  (`onnxruntime.transformers.float16`, 입출력은 fp32로 둠).
- **opset 22는 CPU에서 돈다.** 감춤이 쓰는 YOLO11n 내보내기는 opset 22인데, DirectML에
  해당 커널이 없어 입력 하나에 99–114 ms로 CPU보다 느렸다. `onnxslim`으로 고정 입력의
  모양 계산을 접고 `onnx.version_converter`로 opset 20으로 내리자 정상이 됐다. opset
  20이 이 ONNX Runtime의 DirectML 실행기가 다루는 범위다.
- **모양은 고정하는 편이 좋다.** DirectML은 세션을 만들 때 모든 차원을 알면 가장
  빠르다. ONNX Runtime의 차원 지정(dimension override)으로 모델을 고치지 않고 열린
  배치, 높이, 너비를 정할 수 있다.

## 선택: Windows ML이 아니라 DirectML

Microsoft는 DirectML을 유지보수 상태로 두었고, 후속인 Windows ML은 Windows 11 24H2가
필요하며 회사별 실행기를 스스로 설치한다. 지금 DirectML을 고르는 까닭은 Windows 10과
모든 Windows 11에서 돌고, 프로그램 옆에 DLL 하나로 배포되며, 이 크레이트가 쓰는 ONNX
Runtime 빌드에 이미 있기 때문이다. 아래의 입력 쪽 — GPU에서 채운 D3D12 버퍼 — 은 둘
다 같으므로, 나중에 Windows ML로 옮기면 엘리먼트가 아니라 세션만 바뀐다.

## 엘리먼트: `D3d12OrtDetector`

백엔드 접두사 규칙에 따른 `CudaOrtDetector`의 D3D12 형제로, `D3d12Gpu` 위에 만든다:

```text
D3d12Decoder ─▶ D3d12OrtDetector ─▶ … (프레임은 그대로, Detections가 붙어서)
                 │ 맞추기: 컴퓨트 셰이더, 전체나 타일 → 레터박스한 NCHW fp32, D3D12 버퍼에
                 │ 실행: ONNX Runtime, DirectML, 같은 ID3D12Device와 큐에서
                 └ 해석: 크레이트의 모델 디코더(YOLO, RetinaFace, DETR, 사용자 정의), 타일 합치기
```

- **장치는 하나.** DirectML 실행기를
  `OrtDmlApi::SessionOptionsAppendExecutionProvider_DML1`로 붙이고, `D3d12Gpu`의
  장치로 만든 `IDMLDevice`(`DMLCreateDevice`)와 `D3d12Gpu`의 큐를 넘긴다. 그러면 ONNX
  Runtime이 디코더의 프레임과 같은 큐에서 돌고, 이 엘리먼트가 채운 버퍼를 I/O
  바인딩으로 텐서처럼 넘길 수 있다(`ort::ep::directml::resource_from_d3d`). 입력은
  호스트 메모리를 거치지 않는다. `ort`의 `DirectML` 빌더는 어댑터 번호만 받아서,
  쓰면 ONNX Runtime이 장치를 따로 만든다.
- **맞추기는 GPU에서.** 컴퓨트 셰이더가 디코더의 NV12나 P010 텍스처(와 그 배열
  슬라이스)를 읽어, 전체나 타일을 회색 여백으로 레터박스해 모델 입력에 넣는다. 채널
  순서, 배율, 편향(`ModelInput`)을 맞추고 fp32 NCHW로 쓴다. `CudaOrtDetector`의 맞추기
  커널을 옮긴 것이다. 프레임은 한 번 디코딩하고 전체와 타일을 한 배치로 본다.
  `Tiles`와 `TileChooser`는 CUDA와 같다.
- **출력은 처음엔 호스트로.** 검출 결과는 다른 검출기들이 함께 쓰는 코드로 CPU에서
  해석하고 합친다. YOLO 배치 7의 출력은 20 MB, RetinaFace는 7.5 MB다. 복사가
  문제되면 해석을 셰이더로 옮긴다.
- **순서.** 맞추기를 공유 큐에 추론보다 먼저 기록하므로 큐가 순서를 지킨다.
  엘리먼트가 기다리는 펜스로 출력을 읽어도 되는 때를 안다.
- **모델.** opset 20 이하, 속도를 위해 내부는 fp16. 엘리먼트가 열린 배치, 높이,
  너비를 실제로 돌리는 값으로 지정한다. 모델을 변환하지는 않는다. DirectML에 없는
  연산이 있는 모델은 거절하지 않고, 일부가 CPU에서 돈다고 로그에 남긴다.
- **계약:** 입력은 `MemoryDomain::D3d12`의 `PortContract::Frames`, 레이아웃은 NV12와
  P010(맞추기 셰이더가 읽는 것). 출력은 프레임을 그대로 넘긴다
  (`OutputContract::SameLayout`).
- **기능 플래그:** `ort-directml = ["ort", "ort/directml"]`, Windows 전용,
  `ort-tensorrt`와 나란히. 둘을 함께 켜면 둘 다 든 빌드 하나를 고른다.

## 단계

1. **`D3d12OrtDetector`:** 장치 공유, 맞추기 셰이더, I/O 바인딩, 해석과 타일. 같은
   프레임에서 `SwOrtDetector`와 비교해(박스가 허용 오차 안) 시험하고, `ort-directml`
   CI 작업을 둔다.
2. **`D3d12OrtEmbedder`:** 같은 입력 경로에 임베더의 자르기와 정렬(5점이나 박스).
   감춤의 얼굴과 옷차림 벡터용.
3. **NVIDIA 없는 GPU에서 감춤:** CUDA가 없으면 분석이 D3D12를 고른다. 감춤의 다른
   CUDA 전용 부분도 하나씩 답이 필요하다. 장면 전환(`CudaCutDetector`: D3D12판을
   만들거나 작은 사본에 소프트웨어판), 추적기의 시각 추적(CPU 경로가 있다),
   내보내기의 모자이크(내려받은 뒤 소프트웨어로, 또는 D3D12판), 인코더(D3D11을 거친
   Media Foundation). 2단계 뒤에 감춤에서 계획한다.

## GPU 없이 시험하기

GitHub의 Windows 러너에는 GPU가 없지만, D3D12에는 소프트웨어 어댑터(WARP)가 있고
DirectML이 그 위에서 돈다. CI에서는 느리지만 정확도를 확인할 수 있다. 속도는 이 PC의
RTX 3050에서 재고, AMD나 Intel은 그 GPU를 가진 사람만 잴 수 있다.

## 남은 질문

- 디코더 장치 위의 `DMLCreateDevice`가 NVIDIA처럼 모든 회사의 드라이버에서 되는지,
  큐를 비디오 디코딩과 함께 쓰는 것이 어느 쪽에 비용이 되는지. 안 되면 큐를 따로
  두고 펜스로 맞춘다.
- DirectML의 fp16이 감춤의 놓친 얼굴 비율(TensorRT로 11개 채점 클립에서 10.0%)을
  움직일 만큼 검출을 바꾸는지. 2단계 뒤에 감춤의 `plan_trial.py`로 잰다.
