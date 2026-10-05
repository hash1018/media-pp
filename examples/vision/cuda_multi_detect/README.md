# cuda_multi_detect

Several files through one detector, a batch at a time — DeepStream's
`nvstreammux -> nvinfer -> nvstreamdemux`. Each file is a pipeline of its own
that decodes on NVDEC and ends in an input of a `StreamMux`:

```
FileDemuxer -> CudaDecoder -> Queue -> StreamMux input      (one per stream)
```

and the mux's pipeline gathers a picture of each stream into a batch, runs
the batch through TensorRT at once, and splits the streams out again, each to
a branch of its own that counts its pictures and what was found in them:

```
StreamMux -> CudaOrtDetector -> demux ─┬→ AppSink   (stream 0)
                                       └→ AppSink   (stream 1) …
```

The mux is offline, so nothing is dropped: a batch goes out once every stream
that has not ended has a picture, and the files are decoded as fast as the
detector takes them. At the end it prints each stream's count and how many
pictures a second went through altogether; `--batch 1` runs the same streams
one picture at a time, to compare.

```sh
cargo run --release -p cuda_multi_detect -- model.onnx video.mp4 [video.mp4 ...] \
    [--streams N] [--batch B] [--pictures N]
```

Given one file, `--streams N` runs N copies of it. `--batch` defaults to the
number of streams, at most 8; `--pictures` stops after that many pictures a
stream.

The model is an Ultralytics YOLO ONNX export whose batch is left open —
exported with `dynamic=True`; one made for a single picture, such as the
YOLOv10n release, runs one picture at a time, with a warning. The first run
with a new `--batch` builds a TensorRT engine for the model, this GPU and
every batch up to it, which takes minutes; later runs load it from the cache.
It is built with `ort-tensorrt` and needs that feature's libraries at build
and run time as [`cuda_detect`](../cuda_detect/README.md) does: on Linux
where the linker and the loader find them, on Windows their DLLs on `PATH`.

On an RTX 3050 under Windows, YOLO11n through TensorRT in half precision over
copies of a 1080p H.264 file, decoding included, in pictures a second:

| streams | a detector for each | one detector, batched |
|---|---|---|
| 4 | 403 | 476 (`--batch 4`) |
| 8 | 363 | 523 (`--batch 8`) |

One stream on its own goes faster without a mux — about 300 against 245 — as
there is nothing to batch and the mux's pipeline is one more thread to pass.
