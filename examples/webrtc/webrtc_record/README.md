# webrtc_record

Sends the first video and audio streams from a media file over one WebRTC
connection, then records both received tracks into one MP4. The sender
transcodes to WebRTC-compatible H.264 and Opus:

```text
FileDemuxer(video) -> SwDecoder -> Queue -> Pacer -> SwScaler
                   -> SwEncoder(H.264) -> WebRtcTrackSink
FileDemuxer(audio) -> SwDecoder -> Queue -> Pacer
                   -> SwAudioEncoder(Opus) -> WebRtcTrackSink
```

The receiver does not reuse either sender encoder's parameters and does not
decode or re-encode. `WebRtcTrackSource::wait_stream_info` waits for received
H.264 SPS/PPS and derives the muxer parameters from the actual incoming
bitstream; Opus parameters come from its negotiated stream definition:

```text
WebRtcTrackSource(H.264) -\
                           -> FileMuxer
WebRtcTrackSource(Opus)  --/
```

Both input and output paths are required, and the input must contain at least
one video and one audio stream.

```sh
cargo run -p webrtc_record -- input.mp4 output.mp4
```
