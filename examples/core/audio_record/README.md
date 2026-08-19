# audio_record

TestAudioSource -> SwAudioEncoder -> Mp4Muxer: encodes a generated sine tone
straight into a playable `.mp4` file — the audio-only counterpart to
`screen_record`'s video path, and `Mp4Muxer`'s single-track path (see
`screen_audio_record` for a video+audio track combined into one file
instead).

```sh
cargo run -p audio_record -- [output.mp4] [seconds]
```
