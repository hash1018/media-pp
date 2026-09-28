# audio_playback

`TestAudioSource -> AudioResampler -> AudioVolume -> Queue -> renderer`: plays
a 440 Hz tone for three seconds and changes its gain and mute while it plays,
without clicks. The tone is made at 48 kHz stereo whatever the device uses,
and `AudioResampler` converts it to the device's format, so the renderer never
converts implicitly.

The renderer is `WasapiRenderer` on Windows and `PipeWireAudioRenderer` on
Linux. A device name that matches nothing is an error on Windows; on Linux it
falls back to the first sink, as does a session with no default.

```sh
cargo run -p audio_playback
cargo run -p audio_playback -- list
cargo run -p audio_playback -- <device-name-substring>
```
