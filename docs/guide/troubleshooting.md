# Troubleshooting

Start with the doctor: it checks FFmpeg, the encoders and the GPUs, and says
what to do about each problem.

```sh
cargo run -p doctor
```

## Building

**`ffmpeg-sys-next` fails with errors like `no field pix_fmts` or
`AV_CODEC_ID_V410 not found`.**
The build found FFmpeg 8.1 or newer. media-pp needs FFmpeg **8.0** exactly:
the binding it uses has not caught up with 8.1. Build 8.0 with
[`scripts/setup-ffmpeg`](../../scripts) and make sure `FFMPEG_DIR` points at
it, so no other FFmpeg on the machine is found first.

**`Could not find FFmpeg` or a pkg-config error.**
Set `FFMPEG_DIR` to an FFmpeg 8.0 development tree — the folder holding
`include` and `lib`. The setup script does it.

**`Unable to find libclang`.**
The FFmpeg binding is generated at build time with libclang. Install LLVM;
on Windows, set `LIBCLANG_PATH` to its `bin` folder if it is not found.

**media-pp fails to compile with `this function takes 1 argument but 0
arguments were supplied` at `Session::builder()`.**
Your crate resolved a newer `ort` release candidate than the one media-pp
is written against; 0.3.1 and earlier allow it. Hold it back:

```sh
cargo update -p ort --precise 2.0.0-rc.13
```

**Building with `ort` sits for minutes in `ort-sys`'s build script.**
It is downloading ONNX Runtime, a few hundred megabytes, once for each set
of `ort-*` features — see [features](../features.md).

**The `ort` feature fails on Windows creating symlinks.**
Building ONNX Runtime links its DLLs into the target folder as symlinks,
which Windows allows only in Developer Mode: *Settings → System → For
developers*.

## Starting

**`STATUS_DLL_NOT_FOUND` (Windows) or `error while loading shared
libraries: libavcodec.so.62` (Linux).**
The program was built, but the FFmpeg libraries are not where the loader
looks. Add FFmpeg's `bin` to `PATH` (Windows), or its `lib` to
`LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS). The setup script
does it: on Windows for good with `-Persist`, elsewhere for each shell that
sources the `env.sh` it writes.

**A program built with `ort-cuda` or `ort-tensorrt` does not start.**
Those link NVIDIA's CUDA runtime, cuDNN and TensorRT, which must be found at
start: add the folders holding their DLLs to `PATH` (Windows) or their
libraries to `LD_LIBRARY_PATH` (Linux). The plain `cuda` feature needs only
the driver and starts on any machine. See [features](../features.md).

**On macOS, a window never appears.**
Windows belong to the main thread there. Run your code inside
`media_pp::elements::run_with_windows`, as
[getting started](../getting-started.md#7-playing-a-file) shows.

## Wiring

**Building a branch fails with `… produces X, which Y cannot accept`.**
The two elements do not fit, and nothing is converted for you. The message
says what goes between — a decoder, an encoder, an `…Upload` or
`…Download`. [How media-pp works](concepts.md#building-a-branch) explains
why.

**`the picture was made on another … device`.**
Every GPU element of a pipeline must share one device. Make one
`D3d12Gpu`, `CudaDevice` or the like, and pass the same one to each —
[choosing a backend](choosing-a-backend.md#keep-a-pipeline-on-one-device).

## Running

**A file's pipeline never ends.**
On purpose: a file's source waits at the end, since a seek could still
take it back. Wait for `BusEvent::Finished` on the bus, then call
`pipeline.stop()` — see
[ending a pipeline](concepts.md#ending-a-pipeline).

**The recording is empty or cannot be played.**
End a recording with `pipeline.finish()`, not `stop()`: `finish` lets the
encoder hand over what it holds and the muxer write the file's index.
`stop` abandons both. A fragmented MP4 (`Mp4Layout::Fragmented`) still opens
after a crash.

**A file plays far faster than real time.**
Nothing waits for the clock unless you ask: put a `Pacer` after the decoder
for playback. Without one a file is read as fast as it decodes, which is
what you want for converting.

**`seek` is refused.**
A pipeline writing a file refuses to seek — a recording is a record of what
happened — and so does one reading a live source. `check_seek()` asks
before trying.

**A browser on another network gets an answer from `WebRtcServer` but never
plays.**
The answer tells the browser where to send its media: the address the
server is bound to, which behind a router is not one the browser can reach.
Put the router's public address in `WebRtcServerOptions::candidates` — a
port of 0 keeps the bound one — and forward the server's UDP ports to the
machine: the bound port and one more for each further thread. The server
does not look its public address up itself, which a closed network could
not answer.

**Hardware encoding is not used.**
Run the doctor. An FFmpeg built without NVIDIA's `nvcodec` has no NVENC;
the setup script includes it. `VideoEncodeBin::path()` says which
encoder it opened.

**The process dies with a segmentation fault inside `libnvcuvid.so`, on
Linux, soon after a `CudaDecoder` is dropped.**
NVIDIA's NVDEC library can read a block it has just freed when one decoder
is closed while another decodes. It faults where that block was given back
to the system, which a low glibc mmap threshold makes likely. See
[NVDEC reads a block it has freed](../issues/nvdec-use-after-free/).

**Detection on D3D12 is slow — tens of milliseconds a picture.**
DirectML hands the operators of a model exported past ONNX opset 20 to the
CPU, and the log says so when the detector is made. Export it again at
opset 20, in fp16 — see [D3D12 inference](../d3d12-inference.md).

## Still stuck

Turn on media-pp's own log at `Trace` — `media_pp::log::init` in
[how media-pp works](concepts.md#logging) — and look at the lines around
the failure: each names the pipeline, element and thread. Then open an
issue with that log and the doctor's output.
