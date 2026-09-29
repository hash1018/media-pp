/// AVFoundation, for `AvFoundationCaptureSource`: the cameras there are,
/// their modes, and whether this program may use them.
#[cfg(feature = "avfoundation-capture")]
pub(crate) mod avfoundation;
/// Core Audio, for `CoreAudioRenderer` and `CoreAudioCaptureSource`: the
/// devices there are, what each takes, and the AUHAL unit both play and
/// record through.
#[cfg(any(feature = "coreaudio-renderer", feature = "coreaudio-capture"))]
pub(crate) mod coreaudio;
/// Core Video pixel buffers as frames — what a camera and a screen hand
/// over.
#[cfg(any(feature = "avfoundation-capture", feature = "screencapturekit-capture"))]
pub(crate) mod pixel_buffer;
/// ScreenCaptureKit, for `ScreenCaptureKitSource`: the displays and windows
/// there are, and whether this program may record them.
#[cfg(feature = "screencapturekit-capture")]
pub(crate) mod screencapturekit;
/// VideoToolbox through FFmpeg: the context its decoders, encoders and
/// frames are on.
#[cfg(feature = "videotoolbox")]
pub(crate) mod videotoolbox;
