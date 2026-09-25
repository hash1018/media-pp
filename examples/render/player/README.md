# player

A video file played in a window with its sound, through `Player` — the whole
of a player in one type: `FileDemuxer`, the picture decoded on the window's
GPU where it can be and the sound in software, the picture synchronized to
the sound played, a `VideoWindow` and the platform's audio output, built by
`Player::open`:

```text
FileDemuxer -> VideoDecodeBin -> Queue -> VideoSynchronizer -> VideoWindow
            -> SwDecoder -> AudioResampler -> Queue -> audio renderer
```

`VideoDecodeBin` decodes onto the window's GPU: D3D11VA on Windows, NVDEC on
Linux in a build with `cuda` on a machine with an NVIDIA GPU, in software and
uploaded where the GPU does not take the stream. Where there is no such GPU —
Linux without `cuda` or NVIDIA — it is a software decode straight into the
window. What it chose is printed when playback starts, as `decoding:
Hardware` or `decoding: Software(...)` with the reason — or `sound only` for a
file with no picture, which plays with its waveform in the window
(`AudioWaveform`, shown in time with the sound).

The audio renderer is `WasapiRenderer` on Windows and `PipeWireAudioRenderer`
on Linux, on the default output device. Space pauses and plays, the left and
right arrows move five seconds, the full stop and the comma step a picture
on and back, the minus and plus play slower and faster, either way round,
Backspace at the file's own speed and R turns round at the same speed, the
up and down arrows turn the volume, M mutes, F or a double click fills the
screen, Escape or closing the window stops; the title shows where playback
is. One program for Windows and Linux, with no `#[cfg]` but the one choosing
its `main`.

```sh
cargo run -p player -- path/to/video.mp4
```
