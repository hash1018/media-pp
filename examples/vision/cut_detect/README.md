# cut_detect

Finds where one shot of an edited video ends and the next begins, and
prints each cut — the picture the new shot begins on, its time and how far
it was from the picture before — then how many there were and how fast it
went:

```text
FileDemuxer -> SwDecoder -> SwCutDetector -> AppSink
```

With `--cuda`, on Linux and Windows, the pictures are decoded on an NVIDIA
GPU and the cuts found there, the pictures never leaving it:

```text
FileDemuxer -> CudaDecoder -> Queue -> CudaCutDetector -> AppSink
```

Both find the same cuts in the same file, since both make the same
thumbnail of each picture. How well they find them is in the
[cut detection benchmark](../../../docs/benchmarks/vision/cuts.md), which
scores this example's output.

```sh
cargo run --release -p cut_detect -- path/to/video.mp4 [--cuda]
```
