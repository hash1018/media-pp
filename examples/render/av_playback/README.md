# av_playback

Starts with video only, then lets the terminal attach/detach a decoded audio
branch at runtime. `VideoSynchronizer` uses wall time while audio is absent
and automatically hands scheduling to the audio renderer's played-sample
position while the branch is attached.

Both platforms hold the audio pad open with a dynamic `Tee` and run the same
audio branch — `SwDecoder -> AudioResampler -> Queue -> renderer`
(`WasapiRenderer` on Windows, `PipeWireAudioRenderer` on Linux). The video
branches differ in more than backend types, because only one of them decodes on
the GPU:

```text
Windows: FileDemuxer -> SwDecoder -> Queue -> VideoSynchronizer
         -> SwScaler(NV12) -> D3d12Upload -> D3d12Renderer
Linux:   FileDemuxer -> CudaDecoder -> Queue -> VideoSynchronizer
         -> CudaRenderer
```

The Linux branch is the one that never brings decoded pixels to the CPU: NVDEC
keeps every frame in CUDA memory and the renderer copies it straight into
Vulkan-owned memory. Windows decodes in system memory, so it has to convert and
upload before the renderer can take it.

```sh
cargo run -p av_playback -- path/to/video-with-audio.mp4
audio on
audio off
pause
resume
seek 30
seek 1:15
keyseek 30
q
```

`seek` decodes forward to the frame covering the requested instant.
`keyseek` previews the first decodable frame at the preceding keyframe.
