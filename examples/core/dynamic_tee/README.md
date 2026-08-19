# dynamic_tee

`TestVideoSource -> Tee -> FrameCounter`: keeps that one fixed branch from
initial wiring, then adds a second `-> FrameCounter` branch through
`TeeHandle::attach` while synthetic video is flowing, and removes it again
with `TeeHandle::detach` — proving branches can join and leave a running
`Tee` without disturbing the fixed one.

```sh
cargo run -p dynamic_tee
```
