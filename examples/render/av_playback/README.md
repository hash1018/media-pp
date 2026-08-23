# av_playback

Starts with video only, then lets the terminal attach/detach a decoded audio
branch at runtime. `VideoSynchronizer` uses wall time while audio is absent
and automatically hands scheduling to the audio renderer's played-sample
position while the branch is attached.

Both platforms run the identical graph —
`FileDemuxer -> decoder -> Queue -> VideoSynchronizer -> (upload) -> renderer`,
with a dynamic `Tee` holding the audio pad open. Only the backend types differ:
`SwDecoder`/`D3d12Renderer`/`WasapiRenderer` on Windows,
`CudaDecoder`/`CudaRenderer`/`PipeWireAudioRenderer` on Linux. The Linux
branch is the one that never brings decoded pixels to the CPU: NVDEC keeps
every frame in CUDA memory and the renderer copies it straight into
Vulkan-owned memory.

```sh
cargo run -p av_playback -- path/to/video-with-audio.mp4
audio on
audio off
pause
resume
seek 30
seek 1:15
q
```
