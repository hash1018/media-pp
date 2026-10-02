# custom_element

Elements of your own, one of each kind, in a pipeline with nothing else in
it: `Ramp -> Invert -> Brightness`.

- `Ramp` is a `Source`: asked for the next thing, it makes a grey picture
  whose brightness climbs from one to the next, thirty a second, waiting
  only through the `Wait` it is handed — so a pause or a stop never waits on
  it — and ends its stream after the count.
- `Invert` is a `Filter`: each picture becomes its negative. It makes a new
  frame rather than changing the one it was handed, which may still be read
  elsewhere, and carries its timing and colour across.
- `Brightness` is a `Sink`: it reads each picture's average luma and keeps
  it where `main` can read it once the pipeline is done.

Each declares what it takes and hands on, so a wrong link is refused before
the pipeline starts. None of them sees a control message: the framework runs
the source's loop, hands the end of the stream on after the filter's
`drain`, and holds the terminal while paused. `main` checks that every
picture arrived, in order, inverted, and exits non-zero if one did not.

No media is needed; the argument is how many pictures to make (90 by
default).

```sh
cargo run -p custom_element -- [pictures]
```
