# render_common

Shared library crate, not a runnable example. Every windowed example draws
through one of the library's window renderers in a window the renderer opens
itself — `D3d11WindowRenderer` or `D3d12WindowRenderer` on Windows,
`VulkanWindowRenderer` on Linux — and all of them report that window through
the library's one `WindowEvents`. What is left to share is turning a close of
that window into a stop of the pipelines drawing into it:

- `stop_on_close(windows)` watches each window on a thread of its own and,
  on a close or Escape, stops whatever has been published by then. The stop
  happens on that thread, never the window's, so a renderer waiting for its
  window never waits on the stop.
- `Shutdown` is the handshake: the worker `publish`es the pipelines it built,
  and learns from `publish` itself when a close arrived before it had
  anything to publish. It holds the pipelines weakly — dropping them is the
  worker's, and that is what joins their threads.

Fitting a software decode to a renderer's input is the library's own,
`SwScaler::if_needed`.

Depended on by `av_playback`, `d3d11_scale_render`, `d3d11_upload`,
`gpu_video_compositor`, `hw_decode_render`, `screen_preview_cpu`,
`screen_preview_gpu`, `seek_render`, `sw_decode_render`, `test_video` and
`transcode_render`.
