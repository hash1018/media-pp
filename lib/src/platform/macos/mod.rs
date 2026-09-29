/// Core Audio, for `CoreAudioRenderer` and `CoreAudioCaptureSource`: the
/// devices there are, what each takes, and the AUHAL unit both play and
/// record through.
#[cfg(any(feature = "coreaudio-renderer", feature = "coreaudio-capture"))]
pub(crate) mod coreaudio;
