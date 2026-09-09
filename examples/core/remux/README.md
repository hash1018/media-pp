# remux

FileDemuxer -> FileMuxer: remuxes every stream this crate has a kind for —
video, audio and subtitles — straight into a new `.mp4` container, with no
decode/re-encode, just repackaging. Packets pass through byte-for-byte; only
their timestamps get rescaled to whatever time_base the output container
actually assigns each stream (see `FileMuxer::open`'s own docs).

Which streams those are is asked of `MediaKind::packet_for` rather than
listed in the example, so it does not go stale as the crate learns to carry
more: subtitles started travelling when `MediaKind::SubtitlePacket` was
added, and this example needed no edit to start keeping them. A stream with
no kind — a data stream, an attachment — is skipped with a line saying so,
rather than failing the whole remux over one track nothing here can
describe.

`FileDemuxer` is a single source with one `src_pad` per container stream, so
— unlike combining two independent *live* sources (see `screen_record_av`,
which needs `PipelineBuilder` for exactly that) — this only ever needs one
`Pipeline`: `Eos` reaches every kept stream's `FileMuxer` sink from that same
source thread, no multi-source coordination needed.

```sh
cargo run -p remux -- [input.mp4] [output.mp4]
```
