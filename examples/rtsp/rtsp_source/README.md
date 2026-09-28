# rtsp_source

`RtspSource -> Queue -> PacketCounter`: connects to a live RTSP stream,
counts video packets on the queue's worker thread for 10 seconds, then stops —
proves `RtspSource` connects, negotiates a transport and demuxes real packets
from a live camera or server. A dropped connection ends it early with the
error.

```sh
cargo run -p rtsp_source -- rtsp://host:port/path
```

Any RTSP server or camera will do — `rtsp_serve` publishing a file to a
local [MediaMTX](https://github.com/bluenviron/mediamtx) gives it one at
`rtsp://127.0.0.1:8554/stream`.
