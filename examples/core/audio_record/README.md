# audio_record

TestAudioSource -> SwAudioEncoder -> FileMuxer: encodes a generated sine tone
straight into a playable `.mp4` file — the audio-only counterpart to
`screen_record_software`'s video path, and `FileMuxer`'s single-track path (see
`screen_record_av` for a video+audio track combined into one file
instead).

```sh
cargo run -p audio_record -- [output.mp4] [seconds]
```
