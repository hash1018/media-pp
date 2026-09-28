# webrtc_record

Sends a media file's video and audio — the streams `FileDemuxer::best` picks
— over one WebRTC connection, then records both received tracks into one MP4.
The sender transcodes to WebRTC's H.264 and Opus:

```text
FileDemuxer(video) -> SwDecoder -> Queue -> Pacer -> SwScaler
                   -> SwEncoder(H.264) -> WebRtcTrackSink
FileDemuxer(audio) -> SwDecoder -> Queue -> Pacer
                   -> SwAudioEncoder(Opus) -> WebRtcTrackSink
```

The receiver neither decodes nor re-encodes, and takes nothing from the
sender's encoders: `WebRtcTrackSource::wait_stream_info` waits for the
received H.264's SPS and PPS and derives the muxer's parameters from the
bitstream itself, and Opus's come from its negotiated stream:

```text
WebRtcTrackSource(H.264) -\
                           -> FileMuxer
WebRtcTrackSource(Opus)  --/
```

Both paths are required, and the input needs a video and an audio stream.

```sh
cargo run -p webrtc_record -- input.mp4 output.mp4
```
