# NVDEC reads a block it has freed

NVIDIA's `libnvcuvid` — the library FFmpeg decodes through on NVDEC, and so
[`CudaDecoder`](../../elements.md) — can read memory it has already freed.
When one decoder is closed while another in the same process goes on
decoding, a thread of `libnvcuvid`'s own sometimes reads a block of about
2.3 MB that the close has just freed. Where that block was given back to the
system the read faults, and the process dies with a segmentation fault
inside `libnvcuvid.so`. Nothing media-pp or FFmpeg does is involved:
the block is allocated, freed and read by `libnvcuvid` alone, and the
program in this folder, which uses nothing but FFmpeg's public API,
reproduces it.

## Seen on

| | |
|---|---|
| Driver | 595.91.07 (Ubuntu's `nvidia-driver-595-open`) |
| GPU | GeForce RTX 3050 |
| System | Ubuntu 26.04, Linux 7.0.0, glibc |
| FFmpeg | 8.1.2 (`n8.1.2-44-g7c533d0f86`) |
| First seen | 2026-10-27, in relay-pp's soak: its HEVC channels converted to H.264 for WebRTC players, one channel's conversion stopping while the other's went on |

## What happens

The faulting thread is one `libnvcuvid` starts for itself; its stack holds
nothing else:

```text
#0  libnvcuvid.so.1 +0x19fc1     mov 0xfc438(%rsi),%edx
#1  libnvcuvid.so.1 +0x1a0e2
#2  libnvcuvid.so.1 +0x9e58e
#3  libnvcuvid.so.1 +0x9ecfd
#4  start_thread
```

Run under [`fence.so`](fence.c), which gives every block of 1 MB or more a
mapping of its own and makes it inaccessible when freed, the faulting
address falls in this block:

| | |
|---|---|
| Size | 2,344,120 bytes |
| Allocated by | `libnvcuvid.so.1` +0x49896, as decoder B is opened |
| Freed by | `libnvcuvid.so.1` +0x2a515, as decoder B is closed (`avcodec_free_context`) |
| Read by | `libnvcuvid.so.1` +0x19fc1, on its own thread, seconds after the free — 0.6 s in this core, 6 to 7 s after the decoder's element dropped it in relay-pp's two |

Offsets are from the start of `libnvcuvid.so.595.91.07`. Each decoder has
two such blocks.

## When it faults

The read always happens after the free, and faults only where the block
was handed back to the system when it was freed. glibc serves an
allocation that large with a mapping of its own while its mmap threshold
is under it, and unmaps it on free:

- **glibc as it is** starts its threshold at 128 KiB and raises it, up to
  32 MB, each time it unmaps a block. Early in a process, before a block
  of more than 2.3 MB has been freed, the read faults; afterwards the block
  comes from a heap that keeps freed memory, and the read takes whatever
  is there now, silently.
- **The threshold held at 128 KiB** — `mallopt(M_MMAP_THRESHOLD, …)` or
  `GLIBC_TUNABLES=glibc.malloc.mmap_threshold` — keeps every such block
  mapped on its own, so the read faults whenever the race is lost. relay-pp
  tried exactly that to keep its heap small, and crashed within minutes.

Other allocators, and Windows' heap, which also maps blocks of about a
megabyte and more on their own, have not been tried.

## How often

It is a race: the read does not follow every close.

| Run | Closes while another decoder decoded | Faults |
|---|---|---|
| relay-pp, threshold held at 128 KiB | 49 | 2, both within 4 minutes of the start |
| relay-pp under `fence.so` | 56 | 0 |
| `nvdec_uaf` and its first versions, threshold held or under `fence.so`, processes of 20 to 60 closes | 210 | 0 |
| `nvdec_uaf` under `fence.so`, a fresh process every 6 closes | 127 | 1, in the 22nd process |
| relay-pp with one decoder only, opened and closed again and again | 36 | 0 |

Roughly one close in a hundred or two loses the race; with one decoder
none did.

## Who is exposed

Any process with two or more NVDEC decoders that closes one while another
decodes — through `CudaDecoder`, or any other way to `libnvcuvid`:

- a server converting several streams, one of which stops (relay-pp);
- an editor with several RTSP or file Sources on the CUDA backend, one of
  which is removed (obs-rs on Linux);
- several cameras analysed at once, one of which is closed
  (`cuda_multi_detect`, `vision_bench`, gamchum).

Whether Windows' `nvcuvid.dll` has the same fault has not been tried. On
Windows media-pp decodes through D3D11VA unless `CudaDecoder` is chosen,
and D3D11VA does not go through `nvcuvid`.

[`CudaDevice`](../../../lib/src/platform/cuda/device.rs)'s documentation
records an earlier sighting of what may be the same fault: a segmentation
fault inside `libnvcuvid`, on a thread of the driver's own, while NVDEC or
NVENC work was in flight.

## What to do

- **Do not hold glibc's mmap threshold low** in a process that decodes on
  NVDEC, nor use an allocator that gives large blocks back on free, unless
  one decoder at most is ever open at a time.
- **Decode on Vulkan Video where CUDA frames are not needed.** It reaches
  the same NVDEC hardware through NVIDIA's Vulkan driver, without
  `libnvcuvid`.
- **Try another driver**, with the program below, and record it here.

## Drivers tried

| Driver | Result |
|---|---|
| 595.91.07 | faults (above) |

## Reproducing it

On Linux with an NVIDIA GPU, FFmpeg's development files and an `.mp4` of
HEVC pictures a few minutes long (`bench/soak/hevcify` in relay-pp makes
one):

```sh
cd docs/issues/nvdec-use-after-free
FFMPEG=$HOME/ffmpeg   # the folder holding FFmpeg's include and lib
gcc -O1 -g -pthread -o nvdec_uaf nvdec_uaf.c -I$FFMPEG/include -L$FFMPEG/lib \
    -Wl,-rpath,$FFMPEG/lib -lavformat -lavcodec -lavutil
gcc -O1 -g -shared -fPIC -o fence.so fence.c
./run.sh ~/media/stream-hevc.mp4 100
```

[`run.sh`](run.sh) runs [`nvdec_uaf`](nvdec_uaf.c) under `fence.so` in a
fresh process each time, six closes a process, and stops at the first
fault; on 595.91.07 that took 22 processes, about 45 minutes. Calling a
driver fixed takes several times that with no fault. To see which block
was read, give the core to [`fence_log.py`](fence_log.py):

```sh
gdb -batch -x fence_log.py ./nvdec_uaf CORE
```

On Ubuntu, apport keeps the core in `/var/lib/apport/coredump/` once
`ulimit -c` allows one, which `run.sh` does.
