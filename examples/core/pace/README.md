# pace

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> FrameCounter`: proves `Pacer`
hands decoded frames on at real playback speed, by their timestamps against
the pipeline's playback clock, instead of as fast as decode can make them.
Compare `decode`, which runs the same chain without a `Pacer` and finishes as
fast as possible. It stops on `BusEvent::Finished`.

```sh
cargo run -p pace -- path/to/video.mp4
```
