# webrtc_record

Sends a media file's video and audio — the streams `FileDemuxer::best` picks
— over one WebRTC connection, as two tracks of one `WebRtcMuxer`, then records
both received tracks into one MP4. The sender transcodes to WebRTC's H.264 and
Opus:

```text
FileDemuxer(video) -> SwDecoder -> Queue -> Pacer -> SwScaler
                   -> SwEncoder(H.264) -> WebRtcMuxer
FileDemuxer(audio) -> SwDecoder -> Queue -> Pacer
                   -> SwAudioEncoder(Opus) -> WebRtcMuxer
```

The receiver neither decodes nor re-encodes, and takes nothing from the
sender's encoders: the two tracks come out of `WebRtcHandle::next_source` as
one `WebRtcSource` with an output each, once each has said what it carries —
H.264 by the SPS and PPS it receives, from which the muxer's parameters are
derived, and Opus by its negotiated stream:

```text
WebRtcSource(H.264) -\
                      -> FileMuxer
WebRtcSource(Opus)  --/
```

Both paths are required, and the input needs a video and an audio stream.

```sh
cargo run -p webrtc_record -- input.mp4 output.mp4
```
