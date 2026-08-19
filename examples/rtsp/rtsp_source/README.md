# rtsp_source

`RtspSource -> Queue -> PacketCounter`: connects to a live RTSP stream and
counts video packets on the queue's worker thread for a few seconds, then
stops — proves `RtspSource` actually connects, negotiates a transport, and
demuxes real packets from a live camera, not just that it compiles.

```sh
cargo run -p rtsp_source -- rtsp://host:port/path
cargo run -p rtsp_source                            # falls back to a hardcoded test URL in the source
```
