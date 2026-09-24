# transcode

A file's picture re-encoded to H.264 into a new `.mp4`, with its sound copied
across as it is:

```text
FileDemuxer ┬ video -> VideoDecodeBin -> Queue -> VideoEncodeBin -> FileMuxer
            └ audio ------------------------------------------------> FileMuxer
```

The decode bin decodes onto a device — D3D11 on Windows, CUDA on Linux where
an NVIDIA GPU is there, system memory otherwise — and
`EncodeInput::for_decoded` turns where it put the pictures into what the
encode bin takes, so the two meet with nothing in between and the pictures
stay on the GPU where it takes both. The encode bin opens NVENC where it can,
Media Foundation next on Windows, and software otherwise; both say which
they chose, and the example prints it, as `decoding Hardware, encoding
Nvenc`. The encoder is opened at the picture's own size and rate and told
its colour, all read off the input's `StreamInfo`, and the muxer takes its
track from the encode bin itself.

The file's source waits at its end rather than ending, so the example stops
the pipeline on `Finished` — every packet has reached the file — or on an
error.

```sh
cargo run -p transcode -- input.mp4 [output.mp4]
```

The output defaults to `transcoded.mp4`.
