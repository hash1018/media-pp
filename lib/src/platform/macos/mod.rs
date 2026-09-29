/// AVFoundation, for `AvFoundationCaptureSource`: the cameras there are,
/// their modes, and whether this program may use them.
#[cfg(feature = "avfoundation-capture")]
pub(crate) mod avfoundation;
/// Core Audio, for `CoreAudioRenderer` and `CoreAudioCaptureSource`: the
/// devices there are, what each takes, and the AUHAL unit both play and
/// record through.
#[cfg(any(feature = "coreaudio-renderer", feature = "coreaudio-capture"))]
pub(crate) mod coreaudio;
/// Metal, for `MetalVideoCompositor` and `MetalWindowRenderer`: the device,
/// its kernels, and textures over the pixel buffers VideoToolbox frames
/// hold.
#[cfg(feature = "metal")]
pub(crate) mod metal;
/// One Metal kernel from a VideoToolbox frame into a BGRA one — the
/// per-pixel Metal filters.
#[cfg(feature = "metal")]
pub(crate) mod metal_pass;
/// Core Video pixel buffers as frames — what a camera and a screen hand
/// over — and the pixel buffers VideoToolbox frames hold, for Metal.
#[cfg(any(
    feature = "avfoundation-capture",
    feature = "screencapturekit-capture",
    feature = "metal"
))]
pub(crate) mod pixel_buffer;
/// ScreenCaptureKit, for `ScreenCaptureKitSource`: the displays and windows
/// there are, and whether this program may record them.
#[cfg(feature = "screencapturekit-capture")]
pub(crate) mod screencapturekit;
/// VideoToolbox through FFmpeg: the context its decoders, encoders and
/// frames are on.
#[cfg(feature = "videotoolbox")]
pub(crate) mod videotoolbox;
/// A window of `MetalWindowRenderer`'s own, on the main thread's event
/// loop.
#[cfg(feature = "metal")]
pub(crate) mod window;
