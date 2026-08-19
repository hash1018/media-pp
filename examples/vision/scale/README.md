# scale

Demux -> SwDecoder -> SwScaler -> (prints the first scaled frame's actual
format/size, then counts the rest). Proves `SwScaler` really converts pixel
format (whatever the decoder produces -> RGB24) and resizes (source
resolution -> a fixed 640x640, the kind of input an ONNX object-detection
model would want), not just that it compiles.

```sh
cargo run -p scale -- path/to/video.mp4
```
