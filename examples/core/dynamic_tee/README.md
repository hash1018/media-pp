# dynamic_tee

`TestVideoSource -> Tee -> FrameCounter`: keeps that one fixed branch from
initial wiring, then adds a second `-> FrameCounter` branch through
`TeeHandle::attach` while synthetic video is flowing, and removes it again
with `TeeHandle::detach` — proving branches can join and leave a running
`Tee` without disturbing the fixed one.

`detach` is the right call here because a `FrameCounter` has nothing to
finalize. A branch that ends in an encoder or a muxer needs
`TeeHandle::finish_branch` instead: `detach` abandons the branch, so no EOS
reaches it, delayed frames are never flushed, and an MP4 comes out without a
trailer — unplayable. `finish_branch` sends that EOS first and detaches once
it is on its way.

```sh
cargo run -p dynamic_tee
```
