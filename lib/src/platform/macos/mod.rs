/// Core Audio, for `CoreAudioRenderer` and `CoreAudioCaptureSource`: the
/// devices there are, what each takes, and the AUHAL unit both play and
/// record through.
#[cfg(any(feature = "coreaudio-renderer", feature = "coreaudio-capture"))]
pub(crate) mod coreaudio;
/// VideoToolbox through FFmpeg: the context its decoders, encoders and
/// frames are on.
#[cfg(feature = "videotoolbox")]
pub(crate) mod videotoolbox;
