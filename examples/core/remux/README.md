# remux

FileDemuxer -> Mp4Muxer: remuxes every video/audio stream in a file straight
into a new `.mp4` container — no decode/re-encode, just repackaging. Packets
pass through byte-for-byte; only their timestamps get rescaled to whatever
time_base the output container actually assigns each stream (see
`Mp4Muxer::open`'s own docs).

`FileDemuxer` is a single source with one `src_pad` per container stream, so
— unlike combining two independent *live* sources (see `screen_audio_record`,
which needs `PipelineBuilder` for exactly that) — this only ever needs one
`Pipeline`: `Eos` reaches every kept stream's `Mp4Muxer` sink from that same
source thread, no multi-source coordination needed.

```sh
cargo run -p remux -- [input.mp4] [output.mp4]
```
