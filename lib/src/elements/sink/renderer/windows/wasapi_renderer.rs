use std::{ffi::c_void, ptr, sync::Arc, thread, time::Duration};

use crate::clog::{CLog, cerror, cinfo};
use ffmpeg_next::{self as ffmpeg, Rescale, Rounding};
use thiserror::Error as ThisError;
use windows::Win32::{
    Media::Audio::{
        AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED, IAudioClient, IAudioClock,
        IAudioRenderClient,
    },
    System::Com::{CLSCTX_ALL, CoTaskMemFree},
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_clog},
    elements::{AudioFormat, WasapiDevice, WasapiDeviceKind},
    error::Result,
    platform::windows::wasapi::{
        ComApartment, list_devices as enumerate_wasapi_devices, open_device, resolve_mix_format,
    },
    playback_clock::{AudioMasterRegistration, PlaybackClock, PlaybackClockError},
    time::{MediaTimestamp, TimeBase},
};

const BUFFER_DURATION_100NS: i64 = 100 * 10_000;
const POLL_INTERVAL: Duration = Duration::from_millis(2);

#[derive(Debug, Clone)]
pub struct WasapiRendererOptions {
    pub device: WasapiDevice,
}

#[derive(Debug, ThisError)]
pub enum WasapiRendererError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error("AUDCLNT_E_DEVICE_INVALIDATED — audio device needs to be reopened")]
    DeviceInvalidated,

    #[error("WasapiRenderer requires a Render endpoint, got {0:?}")]
    NotRenderDevice(WasapiDeviceKind),

    #[error("unsupported WASAPI mix format: format_tag={format_tag}, bits_per_sample={bits}")]
    UnsupportedMixFormat { format_tag: u32, bits: u16 },

    #[error(
        "audio format mismatch: expected {expected:?}, got {actual:?}; insert AudioResampler before WasapiRenderer"
    )]
    FormatMismatch {
        expected: AudioFormat,
        actual: AudioFormat,
    },

    #[error("audio frame buffer is shorter than its declared sample count")]
    TruncatedFrame,

    #[error("WasapiRenderer only renders decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),

    #[error(transparent)]
    PlaybackClock(#[from] PlaybackClockError),

    #[error("cannot bind a playback clock after this audio endpoint has started")]
    PlaybackClockBoundAfterStart,

    #[error("this WasapiRenderer is already bound to a playback clock")]
    PlaybackClockAlreadyBound,

    #[error("audio frames need a PTS when WasapiRenderer is the playback-clock master")]
    MissingPts,

    #[error("WASAPI reported an invalid audio-clock frequency of {0}")]
    InvalidClockFrequency(u64),
}

/// Terminal audio sink backed by a WASAPI shared-mode render endpoint.
/// The endpoint's mix format is returned by [`WasapiRenderer::open`] so a
/// caller can place an [`crate::elements::AudioResampler`] immediately
/// before this sink. This element intentionally performs no hidden format
/// conversion. Call [`WasapiRenderer::bind_playback_clock`] while wiring a
/// fixed A/V pipeline to publish this endpoint's actual played-sample position
/// as that pipeline's audio master. A branch attached to a running dynamic Tee
/// uses [`WasapiRenderer::bind_playback_clock_deferred`] instead, so it cannot
/// stall video before the first audio frame reaches the renderer.
///
/// Device-buffer backpressure is the playback clock: `consume` waits for
/// enough WASAPI ring-buffer space to submit the whole input frame. Put a
/// [`crate::queue::Queue`] immediately before this sink when its blocking
/// must not hold up another branch.
pub struct WasapiRenderer {
    clog: CLog,
    name: Arc<str>,
    audio_client: IAudioClient,
    audio_clock: IAudioClock,
    audio_clock_frequency: u64,
    render_client: IAudioRenderClient,
    format: AudioFormat,
    buffer_frames: u32,
    running: bool,
    paused: bool,
    clock_binding: PlaybackClockBinding,
    timeline: Option<DeviceTimeline>,
}

enum PlaybackClockBinding {
    Unbound,
    Deferred(Arc<PlaybackClock>),
    Registered(AudioMasterRegistration),
}

impl PlaybackClockBinding {
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Unbound)
    }

    fn registration(&self) -> Option<&AudioMasterRegistration> {
        match self {
            Self::Registered(master) => Some(master),
            Self::Unbound | Self::Deferred(_) => None,
        }
    }

    fn ensure_registered(&mut self) -> std::result::Result<(), PlaybackClockError> {
        let registration = match self {
            Self::Deferred(playback_clock) => Some(playback_clock.register_audio_master()?),
            Self::Unbound | Self::Registered(_) => None,
        };
        if let Some(registration) = registration {
            *self = Self::Registered(registration);
        }
        Ok(())
    }
}

struct DeviceTimeline {
    device_origin: u64,
    media_origin_ns: i64,
    submitted_until_ns: i64,
}

// SAFETY: WASAPI client interfaces are free-threaded. Every method that
// touches them requires `&mut self`, and each calling thread joins a COM
// apartment for the duration of the call via `ComApartment`.
unsafe impl Send for WasapiRenderer {}

impl WasapiRenderer {
    pub fn list_devices() -> std::result::Result<Vec<WasapiDevice>, WasapiRendererError> {
        Ok(enumerate_wasapi_devices(Some(WasapiDeviceKind::Render))?)
    }

    pub fn open(
        name: impl Into<String>,
        options: WasapiRendererOptions,
    ) -> std::result::Result<(Self, AudioFormat), WasapiRendererError> {
        if options.device.kind != WasapiDeviceKind::Render {
            return Err(WasapiRendererError::NotRenderDevice(options.device.kind));
        }

        let _apartment = ComApartment::new()?;
        let name: Arc<str> = name.into().into();
        let clog = element_clog(ElementType::WasapiRenderer, &name, None);
        let device = open_device(&options.device.id)?;
        let audio_client: IAudioClient = unsafe { device.Activate(CLSCTX_ALL, None)? };
        let mix_format = unsafe { audio_client.GetMixFormat()? };
        let format = resolve_mix_format(mix_format).map_err(|error| {
            WasapiRendererError::UnsupportedMixFormat {
                format_tag: error.format_tag,
                bits: error.bits,
            }
        })?;
        let initialize_result = unsafe {
            audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                BUFFER_DURATION_100NS,
                0,
                mix_format,
                None,
            )
        };
        unsafe { CoTaskMemFree(Some(mix_format as *const c_void)) };
        initialize_result?;

        let render_client: IAudioRenderClient = unsafe { audio_client.GetService()? };
        let audio_clock: IAudioClock = unsafe { audio_client.GetService()? };
        let audio_clock_frequency = unsafe { audio_clock.GetFrequency()? };
        if audio_clock_frequency == 0 {
            return Err(WasapiRendererError::InvalidClockFrequency(
                audio_clock_frequency,
            ));
        }
        let buffer_frames = unsafe { audio_client.GetBufferSize()? };
        cinfo!(
            clog: &clog,
            "opened: device={:?}, {}Hz, {} channel(s), format={:?}, buffer_frames={buffer_frames}",
            options.device.name,
            format.sample_rate,
            format.channels,
            format.sample_format
        );

        Ok((
            Self {
                name,
                clog,
                audio_client,
                audio_clock,
                audio_clock_frequency,
                render_client,
                format,
                buffer_frames,
                running: false,
                paused: false,
                clock_binding: PlaybackClockBinding::Unbound,
                timeline: None,
            },
            format,
        ))
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }

    /// Makes this endpoint the pipeline's exclusive audio playback master.
    /// Call during the wiring closure, before boxing the renderer into its
    /// terminal branch.
    pub fn bind_playback_clock(
        &mut self,
        playback_clock: Arc<PlaybackClock>,
    ) -> std::result::Result<(), WasapiRendererError> {
        if self.clock_binding.is_bound() {
            return Err(WasapiRendererError::PlaybackClockAlreadyBound);
        }
        if self.running || self.timeline.is_some() {
            return Err(WasapiRendererError::PlaybackClockBoundAfterStart);
        }
        let master = playback_clock.register_audio_master()?;
        self.clock_binding = PlaybackClockBinding::Registered(master);
        Ok(())
    }

    /// Binds a dynamically attached endpoint without claiming the audio-master
    /// slot until its first non-empty audio frame arrives.
    ///
    /// This avoids a priming deadlock when an upstream demuxer can block on a
    /// full video queue before reaching the first packet for the newly attached
    /// audio branch. Unlike [`Self::bind_playback_clock`], an exclusive-master
    /// conflict is therefore returned from that first [`Sink::consume`] call.
    pub fn bind_playback_clock_deferred(
        &mut self,
        playback_clock: Arc<PlaybackClock>,
    ) -> std::result::Result<(), WasapiRendererError> {
        if self.clock_binding.is_bound() {
            return Err(WasapiRendererError::PlaybackClockAlreadyBound);
        }
        if self.running || self.timeline.is_some() {
            return Err(WasapiRendererError::PlaybackClockBoundAfterStart);
        }
        self.clock_binding = PlaybackClockBinding::Deferred(playback_clock);
        Ok(())
    }

    fn ensure_playback_master(&mut self) -> Result<()> {
        self.clock_binding
            .ensure_registered()
            .map_err(WasapiRendererError::from)?;
        Ok(())
    }

    fn classify_error(&self, error: windows::core::Error) -> WasapiRendererError {
        if error.code() == AUDCLNT_E_DEVICE_INVALIDATED {
            WasapiRendererError::DeviceInvalidated
        } else {
            WasapiRendererError::Windows(error)
        }
    }

    fn start(&mut self) -> Result<()> {
        if !self.running {
            unsafe { self.audio_client.Start() }.map_err(|error| self.classify_error(error))?;
            self.running = true;
        }
        Ok(())
    }

    fn stop_and_reset(&mut self) -> Result<()> {
        if self.running {
            unsafe { self.audio_client.Stop() }.map_err(|error| self.classify_error(error))?;
        }
        self.running = false;
        self.publish_device_position(false)?;
        unsafe { self.audio_client.Reset() }.map_err(|error| self.classify_error(error))?;
        self.timeline = None;
        Ok(())
    }

    fn device_position(&self) -> std::result::Result<u64, WasapiRendererError> {
        let mut position = 0;
        unsafe { self.audio_clock.GetPosition(&mut position, None) }
            .map_err(|error| self.classify_error(error))?;
        Ok(position)
    }

    fn publish_device_position(&self, running: bool) -> Result<()> {
        let (Some(master), Some(timeline)) = (self.clock_binding.registration(), &self.timeline)
        else {
            return Ok(());
        };
        let position = self.device_position()?;
        let device_delta = position.saturating_sub(timeline.device_origin);
        let elapsed_ns = ((u128::from(device_delta) * 1_000_000_000u128)
            / u128::from(self.audio_clock_frequency))
        .min(i64::MAX as u128) as i64;
        master
            .publish(
                timeline.media_origin_ns.saturating_add(elapsed_ns),
                timeline.submitted_until_ns,
                running,
            )
            .map_err(WasapiRendererError::from)?;
        Ok(())
    }

    fn audio_pts_ns(&self, frame: &ffmpeg::frame::Audio) -> Result<i64> {
        let pts = frame.pts().ok_or(WasapiRendererError::MissingPts)?;
        let source =
            TimeBase::new_unchecked(ffmpeg::Rational::new(1, self.format.sample_rate as i32));
        let nanos = TimeBase::new_unchecked(ffmpeg::Rational::new(1, 1_000_000_000));
        Ok(MediaTimestamp::new_unchecked(pts, source).rescale(nanos))
    }

    fn sample_offset_ns(&self, samples: usize) -> i64 {
        (samples as i64).rescale(
            ffmpeg::Rational::new(1, self.format.sample_rate as i32),
            ffmpeg::Rational::new(1, 1_000_000_000),
        )
    }

    fn render(&mut self, frame: &ffmpeg::frame::Audio) -> Result<()> {
        let bytes = validate_frame(self.format, frame)?;
        if frame.samples() == 0 || self.paused {
            return Ok(());
        }
        self.ensure_playback_master()?;

        let bytes_per_frame = self.format.sample_format.bytes() * self.format.channels as usize;
        let frame_pts_ns = if self.clock_binding.registration().is_some() {
            Some(self.audio_pts_ns(frame)?)
        } else {
            None
        };
        let mut frame_offset = 0usize;
        if let (Some(master), Some(frame_pts_ns)) =
            (self.clock_binding.registration(), frame_pts_ns)
            && let Some(target_ns) = master
                .priming_target_ns()
                .map_err(WasapiRendererError::from)?
        {
            let delta_ns = target_ns.saturating_sub(frame_pts_ns);
            if delta_ns > 0 {
                frame_offset =
                    priming_trim_samples(frame_pts_ns, target_ns, self.format.sample_rate);
                if frame_offset >= frame.samples() {
                    return Ok(());
                }
            }
        }

        while frame_offset < frame.samples() {
            let padding = unsafe { self.audio_client.GetCurrentPadding() }
                .map_err(|error| self.classify_error(error))?;
            // IAudioClock keeps advancing through an endpoint underrun.
            // If the previous submitted range has fully drained, map the
            // next real sample to the current device position instead of
            // counting the intervening silence as media time.
            let rebase_timeline = padding == 0 && self.running && self.timeline.is_some();
            let available = self.buffer_frames.saturating_sub(padding) as usize;
            if available == 0 {
                self.start()?;
                self.publish_device_position(true)?;
                thread::sleep(POLL_INTERVAL);
                continue;
            }

            let take = available.min(frame.samples() - frame_offset);
            let destination = unsafe { self.render_client.GetBuffer(take as u32) }
                .map_err(|error| self.classify_error(error))?;
            let byte_offset = frame_offset * bytes_per_frame;
            let byte_count = take * bytes_per_frame;
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes[byte_offset..byte_offset + byte_count].as_ptr(),
                    destination,
                    byte_count,
                );
            }
            unsafe { self.render_client.ReleaseBuffer(take as u32, 0) }
                .map_err(|error| self.classify_error(error))?;
            if let Some(frame_pts_ns) = frame_pts_ns {
                let submitted_until_ns = frame_pts_ns
                    .saturating_add(self.sample_offset_ns(frame_offset.saturating_add(take)));
                if !rebase_timeline && let Some(timeline) = &mut self.timeline {
                    timeline.submitted_until_ns = submitted_until_ns;
                } else {
                    self.timeline = Some(DeviceTimeline {
                        device_origin: self.device_position()?,
                        media_origin_ns: frame_pts_ns
                            .saturating_add(self.sample_offset_ns(frame_offset)),
                        submitted_until_ns,
                    });
                }
            }
            frame_offset += take;
            self.start()?;
            self.publish_device_position(true)?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        let padding = unsafe { self.audio_client.GetCurrentPadding() }
            .map_err(|error| self.classify_error(error))?;
        if padding > 0 {
            self.start()?;
        }
        loop {
            let padding = unsafe { self.audio_client.GetCurrentPadding() }
                .map_err(|error| self.classify_error(error))?;
            if padding == 0 {
                break;
            }
            self.publish_device_position(true)?;
            thread::sleep(POLL_INTERVAL);
        }
        self.publish_device_position(false)?;
        let final_position = self
            .timeline
            .as_ref()
            .map(|timeline| timeline.submitted_until_ns);
        self.stop_and_reset()?;
        if let (Some(master), Some(final_position)) =
            (self.clock_binding.registration(), final_position)
        {
            master
                .finish(final_position)
                .map_err(WasapiRendererError::from)?;
        }
        Ok(())
    }
}

fn validate_frame(
    expected: AudioFormat,
    frame: &ffmpeg::frame::Audio,
) -> std::result::Result<&[u8], WasapiRendererError> {
    let actual = AudioFormat::new(frame.format(), frame.rate(), frame.channels());
    if actual != expected {
        return Err(WasapiRendererError::FormatMismatch { expected, actual });
    }
    let tight_bytes = frame
        .samples()
        .saturating_mul(expected.channels as usize)
        .saturating_mul(expected.sample_format.bytes());
    frame
        .data(0)
        .get(..tight_bytes)
        .ok_or(WasapiRendererError::TruncatedFrame)
}

fn priming_trim_samples(frame_pts_ns: i64, target_ns: i64, sample_rate: u32) -> usize {
    target_ns
        .saturating_sub(frame_pts_ns)
        .max(0)
        .rescale_with(
            ffmpeg::Rational::new(1, 1_000_000_000),
            ffmpeg::Rational::new(1, sample_rate as i32),
            Rounding::Up,
        )
        .max(0) as usize
}

impl Element for WasapiRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WasapiRenderer
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Sink for WasapiRenderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let _apartment = ComApartment::new().map_err(WasapiRendererError::from)?;
        match buf {
            MediaBuffer::Audio(frame) => self
                .render(&frame)
                .inspect_err(|error| cerror!(self, "render failed: {error}")),
            MediaBuffer::Eos => self
                .drain()
                .inspect_err(|error| cerror!(self, "drain failed: {error}")),
            MediaBuffer::Packet(_) => Err(WasapiRendererError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(WasapiRendererError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        let _apartment = ComApartment::new().map_err(WasapiRendererError::from)?;
        match msg {
            ControlMsg::Pause => {
                if self.running {
                    unsafe { self.audio_client.Stop() }
                        .map_err(|error| self.classify_error(error))?;
                    self.running = false;
                }
                self.publish_device_position(false)?;
                self.paused = true;
            }
            ControlMsg::Resume => {
                self.paused = false;
                let padding = unsafe { self.audio_client.GetCurrentPadding() }
                    .map_err(|error| self.classify_error(error))?;
                if padding > 0 {
                    self.start()?;
                    self.publish_device_position(true)?;
                }
            }
            ControlMsg::Stop => {
                self.paused = false;
                self.stop_and_reset()?;
            }
            ControlMsg::Seek(_) => {
                if self.running {
                    unsafe { self.audio_client.Stop() }
                        .map_err(|error| self.classify_error(error))?;
                }
                self.running = false;
                self.paused = false;
                unsafe { self.audio_client.Reset() }.map_err(|error| self.classify_error(error))?;
                self.timeline = None;
                if let Some(master) = self.clock_binding.registration() {
                    master.reset_for_seek().map_err(WasapiRendererError::from)?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for WasapiRenderer {
    fn drop(&mut self) {
        let Ok(_apartment) = ComApartment::new() else {
            return;
        };
        if self.running {
            let _ = unsafe { self.audio_client.Stop() };
            self.running = false;
        }
        // A dynamically detached renderer must hand the last actually played
        // position back to PlaybackClock before its master registration drops.
        // Otherwise video can resume wall-clock pacing from the last periodic
        // update, a few milliseconds behind the audible handoff point.
        let _ = self.publish_device_position(false);
        let _ = unsafe { self.audio_client.Reset() };
        self.timeline = None;
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg::format::sample::Type;

    use super::*;
    use crate::{clock::Clock, playback_clock::PlaybackMaster};

    fn frame(format: AudioFormat, samples: usize) -> ffmpeg::frame::Audio {
        let mut frame =
            ffmpeg::frame::Audio::new(format.sample_format, samples, format.channel_layout());
        frame.set_rate(format.sample_rate);
        frame.data_mut(0).fill(0);
        frame
    }

    #[test]
    fn binding_does_not_claim_the_clock_until_audio_can_prime_it() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        playback.ensure_wall_origin(1_000);
        let mut binding = PlaybackClockBinding::Deferred(playback.clone());

        assert_eq!(playback.master(), PlaybackMaster::Wall);
        binding.ensure_registered().unwrap();
        assert!(matches!(binding, PlaybackClockBinding::Registered(_)));
        assert_eq!(playback.master(), PlaybackMaster::AudioPriming);
    }

    #[test]
    fn failed_deferred_registration_keeps_the_deferred_state() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        let _existing_master = playback.register_audio_master().unwrap();
        let mut binding = PlaybackClockBinding::Deferred(playback);

        assert!(matches!(
            binding.ensure_registered(),
            Err(PlaybackClockError::AudioMasterAlreadyRegistered)
        ));
        assert!(matches!(binding, PlaybackClockBinding::Deferred(_)));
    }

    #[test]
    fn validates_the_exact_device_mix_format() {
        let expected = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let frame = frame(expected, 480);
        assert_eq!(validate_frame(expected, &frame).unwrap().len(), 480 * 2 * 4);
    }

    #[test]
    fn rejects_audio_that_skipped_the_required_resampler() {
        let expected = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let actual = AudioFormat::new(ffmpeg::format::Sample::I16(Type::Packed), 44_100, 1);
        let error = validate_frame(expected, &frame(actual, 441)).unwrap_err();
        assert!(matches!(
            error,
            WasapiRendererError::FormatMismatch {
                expected: error_expected,
                actual: error_actual,
            } if error_expected == expected && error_actual == actual
        ));
    }

    #[test]
    fn priming_trim_rounds_forward_to_the_first_sample_not_before_wall_position() {
        assert_eq!(priming_trim_samples(0, 10_000_001, 48_000), 481);
        assert_eq!(priming_trim_samples(20_000_000, 10_000_000, 48_000), 0);
    }
}
