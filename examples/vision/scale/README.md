# scale

`FileDemuxer -> SwDecoder -> Queue -> SwScaler -> (prints the first scaled
frame's format and size, then counts the rest)`. Proves `SwScaler` converts
the pixel format (whatever the decoder produces to RGB24) and resizes (to a
fixed 640x640, the kind of input an object-detection model wants).

```sh
cargo run -p scale -- path/to/video.mp4
```
