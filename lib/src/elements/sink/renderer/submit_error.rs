/// Errors a `D3d12FrameRenderer`/`D3d11FrameRenderer` implementation can
/// report. GPU-vendor-agnostic (no D3D11/D3D12-specific type in here), so
/// it's shared by both instead of each defining its own copy — and left
/// ungated (not behind either renderer feature) so it's a stable type to
/// reference regardless of which one a caller actually enables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    /// The submitted frame buffer pointer is null.
    #[error("the submitted frame buffer pointer is null")]
    NullBuffer,
    /// The submitted frame metadata or resource is not valid for the renderer.
    #[error("the submitted frame is not valid for this renderer")]
    InvalidFrame,
    /// Every in-flight submission slot is occupied; retry after rendering progresses.
    #[error("every in-flight submission slot is occupied")]
    NoFreeSlot,
    /// The renderer has shut down and accepts no more frames.
    #[error("the renderer has shut down")]
    RendererStopped,
    /// Rendering failed without indicating permanent device removal.
    #[error("rendering failed")]
    RenderFailed,
    /// The GPU device is no longer valid (driver reset/removal). Recovery
    /// requires recreating the whole rendering setup, not just retrying.
    #[error("the GPU device was removed or reset")]
    DeviceRemoved,
}
