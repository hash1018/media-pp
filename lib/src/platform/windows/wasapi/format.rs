//! Conversion from native WASAPI mix formats to media-pp audio formats.

use ffmpeg_next as ffmpeg;
use windows::Win32::Media::{
    Audio::{WAVE_FORMAT_PCM, WAVEFORMATEX, WAVEFORMATEXTENSIBLE},
    KernelStreaming::{KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE},
    Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT},
};

use crate::elements::AudioFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WasapiMixFormatError {
    pub(crate) format_tag: u32,
    pub(crate) bits: u16,
}

pub(crate) fn resolve_mix_format(
    mix_format: *const WAVEFORMATEX,
) -> std::result::Result<AudioFormat, WasapiMixFormatError> {
    let wf = unsafe { &*mix_format };
    let bits = wf.wBitsPerSample;
    let format_tag = if wf.wFormatTag as u32 == WAVE_FORMAT_EXTENSIBLE {
        let ext = mix_format as *const WAVEFORMATEXTENSIBLE;
        let sub_format = unsafe { (*ext).SubFormat };
        if sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
            WAVE_FORMAT_IEEE_FLOAT
        } else if sub_format == KSDATAFORMAT_SUBTYPE_PCM {
            WAVE_FORMAT_PCM
        } else {
            0
        }
    } else {
        wf.wFormatTag as u32
    };

    let sample_format = match (format_tag, bits) {
        (tag, 32) if tag == WAVE_FORMAT_IEEE_FLOAT => {
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
        }
        (tag, 16) if tag == WAVE_FORMAT_PCM => {
            ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed)
        }
        (tag, 32) if tag == WAVE_FORMAT_PCM => {
            ffmpeg::format::Sample::I32(ffmpeg::format::sample::Type::Packed)
        }
        _ => {
            return Err(WasapiMixFormatError { format_tag, bits });
        }
    };

    Ok(AudioFormat::new(
        sample_format,
        wf.nSamplesPerSec,
        wf.nChannels,
    ))
}
