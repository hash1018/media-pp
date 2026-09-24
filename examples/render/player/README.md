# player

A video file played in a window with its sound, through `Player` — the whole
of a player in one type: `FileDemuxer`, a software decode of each stream, the
picture synchronized to the sound played, a `VideoWindow` and the platform's
audio output, built by `Player::open`:

```text
FileDemuxer -> SwDecoder -> Queue -> VideoSynchronizer -> VideoWindow
            -> SwDecoder -> AudioResampler -> Queue -> audio renderer
```

The audio renderer is `WasapiRenderer` on Windows and `PipeWireAudioRenderer`
on Linux, on the default output device. Space pauses and plays, the arrows
move five seconds, F or a double click fills the screen, Escape or closing
the window stops; the title shows where playback is. One program for Windows
and Linux, with no `#[cfg]` but the one choosing its `main`.

```sh
cargo run -p player -- path/to/video.mp4
```
