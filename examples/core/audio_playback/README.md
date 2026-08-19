# audio_playback

`TestAudioSource -> AudioResampler -> AudioVolume -> Queue -> renderer`: plays
a 440Hz tone for three seconds and demonstrates click-free runtime gain/mute
changes. `AudioResampler` deliberately targets a rate/channel count that
differs from the device's own, proving it owns the format conversion rather
than the renderer doing it implicitly.

Both platforms run the identical graph and CLI; only the renderer and its
device type differ — `WasapiRenderer` on Windows, `PipeWireAudioRenderer` on
Linux.

```sh
cargo run -p audio_playback
cargo run -p audio_playback -- list
cargo run -p audio_playback -- <device-name-substring>
```
