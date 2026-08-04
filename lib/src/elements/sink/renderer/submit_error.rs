/// Errors a `D3d12FrameRenderer`/`D3d11FrameRenderer` implementation can
/// report. GPU-vendor-agnostic (no D3D11/D3D12-specific type in here), so
/// it's shared by both instead of each defining its own copy — and left
/// ungated (not behind either renderer feature) so it's a stable type to
/// reference regardless of which one a caller actually enables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    NullBuffer,
    InvalidFrame,
    NoFreeSlot,
    RendererStopped,
    RenderFailed,
    /// The GPU device is no longer valid (driver reset/removal). Recovery
    /// requires recreating the whole rendering setup, not just retrying.
    DeviceRemoved,
}
