# Contributing to media-pp

[`AGENTS.md`](AGENTS.md) holds the design and error-handling conventions every
change is held to; this file is how to build and test one.

## Tests

```sh
cargo test -p media-pp
```

Tests need no media: they synthesize their fixture from the crate's own
sources and encoders, so every machine tests the same file. A backend's tests
run under its feature (`--features d3d11,d3d12,cuda`, or the Linux ones) and
skip, saying why, on a machine without the hardware.

## Control sequences

`core::pipeline::tests::conformance` runs random orders of pause, resume,
seek, finish and stop against the shapes of pipeline this crate is used in,
and checks what every terminal was handed: nothing new while paused, nothing
from before a seek after it, an `Eos` after a finish, and no call that fails
to return. An ordinary test run plays a few fixed sequences. The races these
are for show when threads are short of cores, so CI also runs them pinned to
two, seeded from the clock:

```sh
MEDIA_PP_CONTROL_ITERS=10 MEDIA_PP_CONTROL_RANDOM=1 taskset -c 0,1 cargo test -p media-pp --lib -- conformance
```

On Windows, set the shell's own affinity first —
`[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = 3` —
and cargo and the tests inherit it. A failure prints its seed and the steps
it took; `MEDIA_PP_CONTROL_SEED=<seed>` replays exactly that sequence, and
`MEDIA_PP_CONTROL_TRACE=<directory>` writes the crate's log there at `Trace`,
every control message at every element, for reading what it did. A change to
how control travels — a new message, a new element that waits, a new source
loop — should pass a few hundred of these before it lands.

## Stress and leak scenarios

The scenarios in `lib/tests/soak.rs` run for tens of seconds and are
`#[ignore]`d. They read a real recording from `MEDIA_PP_TEST_VIDEO`:

```sh
cargo test -p media-pp --features d3d11,d3d12,cuda --test soak -- --ignored --nocapture
```

On Linux, `pipewire-screen-capture` takes the place of `d3d11`, and the
capture scenarios also need `MEDIA_PP_SOAK_RESTORE_TOKEN`, since the portal
would otherwise show its picker; any run of `screen_record_software` prints a
token to reuse.

## Documentation

[docs.rs] builds for Linux and so omits the Windows-only API. To build the
complete documentation locally, labelled by feature:

```powershell
$env:RUSTDOCFLAGS = "--cfg docsrs"
cargo +nightly doc -p media-pp --open --features d3d11,d3d12,dxgi-capture,wgc-capture,mf-capture,wasapi-capture,wasapi-renderer,webrtc
```

CI builds the documentation with `-D warnings` for every feature set, so a
public item's documentation may not link to a private one, or to one its
feature set does not have.

[docs.rs]: https://docs.rs/media-pp
