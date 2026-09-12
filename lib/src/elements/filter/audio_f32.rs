//! Decoded `f32` audio, a channel at a time, whether FFmpeg packed it or not.
//!
//! What an audio filter that works on the signal itself — a gate, a
//! denoiser — needs to read and write, and the one place that knows where
//! FFmpeg keeps each channel. `f32` only: it is what every capture source
//! and the mixer produce and what a decoder of any common codec hands back,
//! and a filter that also took integers would be converting on both sides
//! of the one thing it does. An [`AudioResampler`](super::AudioResampler)
//! in front converts anything else.

use std::sync::Arc;

use ffmpeg_next as ffmpeg;

/// Why a frame's samples could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unreadable {
    /// Not `f32`, packed or planar.
    Format(ffmpeg::format::Sample),
    /// No channel layout.
    NoChannels,
    /// The one plane of a packed frame is shorter than its samples need.
    TooShort { required: usize, actual: usize },
}

/// Each channel of an `f32` frame, copied out.
pub(crate) fn read(frame: &ffmpeg::frame::Audio) -> Result<Vec<Vec<f32>>, Unreadable> {
    let format = frame.format();
    let ffmpeg::format::Sample::F32(packing) = format else {
        return Err(Unreadable::Format(format));
    };
    let channels = usize::from(frame.channels());
    if channels == 0 {
        return Err(Unreadable::NoChannels);
    }
    let samples = frame.samples();
    match packing {
        // `plane::<f32>(c)` rather than `data(c)`: FFmpeg records a planar
        // frame's size in `linesize[0]` alone, so `data(c)` is empty for
        // every channel after the first — see `AudioVolume`'s `apply_gain`.
        ffmpeg::format::sample::Type::Planar => Ok((0..channels)
            .map(|channel| frame.plane::<f32>(channel)[..samples].to_vec())
            .collect()),
        ffmpeg::format::sample::Type::Packed => {
            let interleaved = packed(frame.data(0), samples * channels)?;
            Ok((0..channels)
                .map(|channel| {
                    interleaved
                        .iter()
                        .skip(channel)
                        .step_by(channels)
                        .copied()
                        .collect()
                })
                .collect())
        }
    }
}

/// Writes `channels` over the samples of a frame [`read`] accepted.
///
/// Each channel must hold exactly the frame's sample count.
pub(crate) fn write(frame: &mut ffmpeg::frame::Audio, channels: &[Vec<f32>]) {
    let samples = frame.samples();
    debug_assert!(channels.iter().all(|channel| channel.len() == samples));
    match frame.format() {
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar) => {
            for (index, channel) in channels.iter().enumerate() {
                frame.plane_mut::<f32>(index)[..samples].copy_from_slice(channel);
            }
        }
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed) => {
            let count = channels.len();
            let data = &mut frame.data_mut(0)[..samples * count * 4];
            for (index, bytes) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *bytes = channels[index % count][index / count].to_ne_bytes();
            }
        }
        _ => unreachable!("only a frame `read` accepted is written back"),
    }
}

/// Scales every channel of an `f32` frame by one gain per sample, asked of
/// `gain` with the loudest channel's level at that sample.
///
/// One gain for every channel is what the dynamics filters share — a gate, a
/// compressor, a limiter — and why they are linked: a gain worked out per
/// channel would pull a stereo image toward whichever side was quieter.
///
/// Answers the frame itself, uncopied, when every gain comes out exactly
/// one; otherwise a copy only when another branch still holds it, as
/// `AudioVolume` does.
pub(crate) fn scale_linked(
    frame: Arc<ffmpeg::frame::Audio>,
    mut gain: impl FnMut(f32) -> f32,
) -> Result<Arc<ffmpeg::frame::Audio>, Unreadable> {
    let mut channels = read(&frame)?;
    let gains: Vec<f32> = (0..frame.samples())
        .map(|index| {
            gain(
                channels
                    .iter()
                    .map(|channel| channel[index].abs())
                    .fold(0.0, f32::max),
            )
        })
        .collect();
    if gains.iter().all(|gain| *gain == 1.0) {
        return Ok(frame);
    }
    for channel in &mut channels {
        for (sample, gain) in channel.iter_mut().zip(&gains) {
            *sample *= gain;
        }
    }
    let mut frame = Arc::try_unwrap(frame).unwrap_or_else(|shared| shared.as_ref().clone());
    write(&mut frame, &channels);
    Ok(Arc::new(frame))
}

/// A level in dBFS as an amplitude, where full scale is one.
pub(crate) fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// An amplitude as a level in dBFS; silence is far below anything a
/// threshold is set to rather than negative infinity, so arithmetic on it
/// stays finite.
pub(crate) fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.max(1e-10).log10()
}

/// The first `count` samples of a packed plane.
fn packed(data: &[u8], count: usize) -> Result<Vec<f32>, Unreadable> {
    let required = count * 4;
    if data.len() < required {
        return Err(Unreadable::TooShort {
            required,
            actual: data.len(),
        });
    }
    Ok(data[..required]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use ffmpeg::format::sample::Type;

    use super::*;

    /// A frame of `channels`, each given sample for sample, packed or not.
    pub(crate) fn frame(
        channels: &[Vec<f32>],
        rate: u32,
        packing: Type,
    ) -> Arc<ffmpeg::frame::Audio> {
        let samples = channels[0].len();
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(packing),
            samples,
            ffmpeg::ChannelLayout::default(channels.len() as i32),
        );
        frame.set_rate(rate);
        frame.set_pts(Some(123));
        write(&mut frame, channels);
        Arc::new(frame)
    }

    /// Both packings carry each channel through unchanged, and in its own
    /// place — a packed frame read back with its channels swapped would pass
    /// a test that only ever used one channel.
    #[test]
    fn every_channel_reads_back_as_it_was_written_in_either_packing() {
        let channels = vec![vec![0.1, 0.2, 0.3], vec![-0.4, -0.5, -0.6]];
        for packing in [Type::Packed, Type::Planar] {
            let frame = frame(&channels, 48_000, packing);
            assert_eq!(read(&frame).unwrap(), channels, "{packing:?}");
        }
    }

    #[test]
    fn a_frame_that_is_not_f32_is_refused_rather_than_misread() {
        let frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            4,
            ffmpeg::ChannelLayout::MONO,
        );
        assert_eq!(
            read(&frame),
            Err(Unreadable::Format(ffmpeg::format::Sample::I16(
                Type::Packed
            )))
        );
    }
}
