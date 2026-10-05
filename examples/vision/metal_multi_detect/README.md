# metal_multi_detect

Several files through one detector, a batch at a time — DeepStream's
`nvstreammux -> nvinfer -> nvstreamdemux`, on macOS. Each file is a pipeline
of its own that decodes on VideoToolbox and ends in an input of a
`StreamMux`:

```
FileDemuxer -> VideoToolboxDecoder -> Queue -> StreamMux input      (one per stream)
```

and the mux's pipeline gathers a picture of each stream into a batch, fits the
batch into the model's input in one Metal pass, runs it through Core ML at
once, and splits the streams out again, each to a branch of its own that
counts its pictures and what was found in them:

```
StreamMux -> MetalOrtDetector -> demux ─┬→ AppSink   (stream 0)
                                        └→ AppSink   (stream 1) …
```

The mux is offline, so nothing is dropped: a batch goes out once every stream
that has not ended has a picture, and the files are decoded as fast as the
detector takes them. At the end it prints each stream's count and how many
pictures a second went through altogether; `--batch 1` runs the same streams
one picture at a time, to compare.

```sh
cargo run --release -p metal_multi_detect -- model.onnx video.mp4 [video.mp4 ...] \
    [--streams N] [--batch B] [--pictures N]
```

Given one file, `--streams N` runs N copies of it. `--batch` defaults to the
number of streams, at most 8; `--pictures` stops after that many pictures a
stream.

The model is an Ultralytics YOLO ONNX export whose batch is left open —
exported with `dynamic=True` — which Core ML compiles once, fixed at
`--batch`; one made for a single picture, such as the YOLOv10n release, runs
one picture at a time, with a warning. It is built with `ort-coreml` and runs
on Apple silicon, as [`metal_detect`](../metal_detect/README.md) does.
