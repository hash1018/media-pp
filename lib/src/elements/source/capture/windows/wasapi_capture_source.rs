use std::{
    ffi::c_void,
    ptr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::Win32::{
    Media::{
        Audio::{
            AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK, IAudioCaptureClient, IAudioClient, WAVEFORMATEX,
        },
        Multimedia::WAVE_FORMAT_IEEE_FLOAT,
    },
    System::Com::{CLSCTX_ALL, CoTaskMemFree},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlReceiver,
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::AudioFormat,
    error::Result,
    pad::SrcPad,
    platform::windows::wasapi::{
        ComApartment, WasapiDevice, WasapiDeviceKind, WasapiProcess, activate_process_loopback,
        list_devices as enumerate_wasapi_devices, list_processes as enumerate_wasapi_processes,
        open_device, resolve_mix_format,
    },
    schedule::ActiveTimeline,
};

/// How long [`WasapiCaptureSource::run`] sleeps between checks of
/// `GetNextPacketSize` — also bounds `Stop` latency, same reasoning as
/// `DxgiCaptureSource`'s own `POLL_GRANULARITY`. Plain
/// polling rather than `IAudioClient::SetEventHandle` + `WaitForSingleObject`:
/// event-driven signaling is well documented as unreliable specifically
/// for loopback capture (Microsoft's own WASAPILoopbackCapture sample
/// polls for exactly this reason, rather than using
/// `AUDCLNT_STREAMFLAGS_EVENTCALLBACK`). Using the same poll loop for
/// `AudioCaptureMode::Microphone` too keeps one code path instead of
/// branching between event-driven and polled just for a latency
/// difference that doesn't matter at these timescales.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// WASAPI shared-mode buffer size, in 100ns units (200ms) — comfortably
/// larger than `POLL_INTERVAL` so this element's own wakeup cadence, not
/// the device's ring buffer, is what bounds latency.
const BUFFER_DURATION_100NS: i64 = 200 * 10_000;

/// What a process loopback capture asks Windows to hand it.
///
/// A device capture inherits its endpoint's mix format; there is no endpoint
/// behind a process, so this is chosen instead and Windows converts whatever
/// the process is playing into it. 48 kHz float stereo is what a Windows
/// mixer works in, so in the ordinary case the conversion is nothing at all
/// — and a caller that wants another rate has a resampler for it, as it does
/// for every device whose endpoint disagrees with its project.
const PROCESS_LOOPBACK_FORMAT: WAVEFORMATEX = WAVEFORMATEX {
    wFormatTag: WAVE_FORMAT_IEEE_FLOAT as u16,
    nChannels: 2,
    nSamplesPerSec: 48_000,
    nAvgBytesPerSec: 48_000 * 2 * 4,
    nBlockAlign: 2 * 4,
    wBitsPerSample: 32,
    cbSize: 0,
};

/// Errors specific to `WasapiCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WasapiCaptureSourceError {
    /// A COM or WASAPI operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// `AUDCLNT_E_DEVICE_INVALIDATED` specifically, broken out of the
    /// generic [`WasapiCaptureSourceError::Windows`] variant for the same
    /// reason `DxgiCaptureSourceError::AccessLost` is:
    /// the single most common *recoverable* failure (default device
    /// changed, device unplugged, format changed) surfaces this way. Same
    /// "fail fast, caller rebuilds a fresh one" contract
    /// [`crate::elements::RtspSource`]/`DxgiCaptureSource`
    /// already document: this element doesn't retry internally.
    #[error("AUDCLNT_E_DEVICE_INVALIDATED — audio device needs to be reopened")]
    DeviceInvalidated,

    /// Seeking was requested on a live audio capture.
    #[error("WasapiCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,

    /// The endpoint mix format cannot be represented by [`AudioFormat`](crate::elements::AudioFormat).
    #[error("unsupported WASAPI mix format: format_tag={format_tag}, bits_per_sample={bits}")]
    UnsupportedMixFormat {
        /// WAVE format tag reported by WASAPI.
        format_tag: u32,
        /// Bits per sample reported by WASAPI.
        bits: u16,
    },
}

/// Construction-time options for [`WasapiCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct WasapiCaptureOptions {
    /// Which endpoint to capture from — one entry out of
    /// [`WasapiCaptureSource::list_devices`] (or hand-built, if the caller
    /// already knows a device's id/kind some other way).
    pub device: WasapiDevice,
}

/// Captures audio via WASAPI (`IAudioClient`/`IAudioCaptureClient`) —
/// GStreamer's `wasapi2src` equivalent. One src pad, pushing
/// `MediaBuffer::Audio` frames in the captured device's own native mix
/// format/rate/channel count — no resampling. Same division of labor as
/// `DxgiCaptureSource` emitting raw `Pixel::BGRA` and
/// leaving conversion to a downstream [`crate::elements::SwScaler`]: if
/// something downstream needs a fixed sample rate/format, use
/// [`crate::elements::AudioResampler`] rather than hiding conversion in
/// this element.
///
/// Polls `IAudioCaptureClient::GetNextPacketSize` on a short fixed
/// interval (`POLL_INTERVAL`) rather than waiting on a WASAPI-signaled
/// event — see that constant's own docs on why event-driven mode isn't
/// used here.
///
/// Emits continuously from the moment `run` starts, `pts` always in
/// lockstep with wall-clock time — backed by real WASAPI data when it's
/// available and synthesized silence otherwise (see
/// `WasapiCaptureSource::fill_silence_gap`), since WASAPI itself
/// delivers literally nothing whenever the render engine has no active
/// session at all (e.g. nothing currently playing, for
/// [`WasapiDeviceKind::Render`]). Without this, a quiet period would be a
/// real gap in the audio timeline rather than silence, which would leave
/// a downstream muxer/encoder with no way to keep audio and video in
/// sync across it.
///
/// Every WASAPI object here is created by [`WasapiCaptureSource::open`] on
/// its caller's thread, then actually driven by [`SourceElement::run`] on
/// whichever thread [`crate::pipeline::Pipeline`] spawns for this source
/// — a different thread in the normal case. COM requires every thread
/// that touches an interface to have joined an apartment itself (even
/// though the interfaces here are free-threaded/agile and can be handed
/// across threads freely), so `run` makes its own `CoInitializeEx` call
/// before touching anything, paired with `CoUninitialize` when it returns
/// — the same two-`CoInitializeEx`-calls-per-object-lifetime pattern
/// `cpal`'s own WASAPI backend uses. `open` also joins its caller's COM
/// apartment while it creates the agile WASAPI interfaces, then balances
/// that call before returning; the pipeline worker joins its own apartment
/// independently in `run`.
///
/// Deliberately does **not** retry internally on
/// `AUDCLNT_E_DEVICE_INVALIDATED` (default device changed, unplugged,
/// format changed) — same "fail fast, caller rebuilds" contract as
/// `DxgiCaptureSource`/`RtspSource`; watch for
/// [`WasapiCaptureSourceError::DeviceInvalidated`] and call
/// [`WasapiCaptureSource::open`] again.
///
/// Runs until `Stop` — never reaches `Eos` on its own, same as every other
/// live source in this crate.
pub struct WasapiCaptureSource {
    pp_log: PpLog,
    name: Arc<str>,
    audio_client: IAudioClient,
    capture_client: IAudioCaptureClient,
    sample_rate: u32,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    /// Cumulative sample count across every emitted frame — this
    /// element's `pts` unit (see [`WasapiCaptureSource::time_base`]), same
    /// "integer tick counter" convention every other source in this crate
    /// uses.
    samples_emitted: i64,
    pad: SrcPad,
}

// SAFETY: every WASAPI/COM handle here is a `windows-rs` COM interface
// wrapper — thread-safe to hand off (refcounting is interlocked, and
// these specific interfaces are documented free-threaded/agile).
// `&mut self` on every method that touches them already rules out
// concurrent access from multiple threads — same reasoning
// `DxgiCaptureSource` documents for its own `unsafe impl Send`.
unsafe impl Send for WasapiCaptureSource {}

impl WasapiCaptureSource {
    /// Enumerates every currently-active audio endpoint — both `Render`
    /// (playback) and `Capture` (recording) — as an [`WasapiDevice`] list a
    /// caller can show in a picker UI and index/search into, then hand the
    /// chosen entry straight to [`WasapiCaptureOptions::device`]. No
    /// concept of "mode" to reason about beforehand: the picked device's
    /// own [`WasapiDeviceKind`] is what tells `open` whether to use
    /// loopback.
    pub fn list_devices() -> std::result::Result<Vec<WasapiDevice>, WasapiCaptureSourceError> {
        Ok(enumerate_wasapi_devices(None)?)
    }

    /// Opens `options.device` and starts a shared-mode WASAPI capture
    /// session. Returns the element alongside the captured stream's actual
    /// [`AudioFormat`] — what a caller needs to build a matching downstream
    /// encoder/muxer, same pattern as
    /// `DxgiCaptureSource::open` returning a
    /// [`crate::elements::VideoFormat`]. Doesn't carry a `time_base`
    /// (unlike `VideoFormat`) — every audio element in this crate derives
    /// it as `1 / sample_rate` (see [`WasapiCaptureSource::time_base`]),
    /// so it's never an independent value a caller could get wrong by
    /// threading it separately.
    pub fn open(
        name: impl Into<String>,
        options: WasapiCaptureOptions,
    ) -> std::result::Result<(Self, AudioFormat), WasapiCaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WasapiCaptureSource, &name, None);
        let _apartment = ComApartment::new()?;

        // SAFETY: COM is initialized on this thread. Every activated interface
        // is live, `mix_format` is the non-null allocation returned by WASAPI
        // and remains valid through format parsing/initialization, then is
        // freed exactly once after `Initialize` has finished reading it.
        unsafe {
            let device = open_device(&options.device.id)?;
            let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

            let mix_format = audio_client.GetMixFormat()?;
            let audio_format = resolve_mix_format(mix_format).map_err(|error| {
                WasapiCaptureSourceError::UnsupportedMixFormat {
                    format_tag: error.format_tag,
                    bits: error.bits,
                }
            })?;

            let stream_flags = match options.device.kind {
                WasapiDeviceKind::Render => AUDCLNT_STREAMFLAGS_LOOPBACK,
                WasapiDeviceKind::Capture => 0,
            };
            let init_result = audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                stream_flags,
                BUFFER_DURATION_100NS,
                0,
                mix_format,
                None,
            );
            CoTaskMemFree(Some(mix_format as *const c_void));
            init_result?;

            let capture_client: IAudioCaptureClient = audio_client.GetService()?;

            pp_info!(
                pp_log: &pp_log,
                "opened: device={:?} ({:?}), {}Hz, {} channel(s), format={:?}",
                options.device.name,
                options.device.kind,
                audio_format.sample_rate,
                audio_format.channels,
                audio_format.sample_format
            );
            Ok((
                Self::assemble(name, pp_log, audio_client, capture_client, audio_format),
                audio_format,
            ))
        }
    }

    /// Every process with an audio session of its own — the applications a
    /// per-process capture can be pointed at, as [`WasapiProcess`] entries
    /// whose `id` goes straight to [`WasapiCaptureSource::open_process`].
    ///
    /// What a picker shows. Not a process list: a text editor that has never
    /// made a sound is not something to capture the audio of, and one that
    /// played something an hour ago still has the session that says it can.
    pub fn list_processes() -> std::result::Result<Vec<WasapiProcess>, WasapiCaptureSourceError> {
        Ok(enumerate_wasapi_processes()?)
    }

    /// Opens a capture of what one process tree is playing, rather than of
    /// an endpoint — Windows' own process loopback, which is what lets a
    /// caller record a game without the chat program beside it.
    ///
    /// `process_id` is a live process; the capture covers it and everything
    /// it started, since a game that plays through a child process and a
    /// browser that gives each tab one would otherwise capture as silence.
    /// It is bound to *that* process: when it exits, this goes quiet rather
    /// than following the next instance of the same application, and it is
    /// the caller — which is the half that knows what the user picked — that
    /// looks the new one up and opens again.
    ///
    /// There is no endpoint here and so no mix format to inherit, which is
    /// the one thing that differs from [`WasapiCaptureSource::open`]: the
    /// format below is what this asks the virtual device to hand over, and
    /// Windows converts whatever the process is actually playing into it.
    ///
    /// Needs Windows 10 2004 or newer, where process loopback was added. An
    /// older build fails the activation, which arrives as
    /// [`WasapiCaptureSourceError::Windows`] — this does not check a version
    /// of its own.
    pub fn open_process(
        name: impl Into<String>,
        process_id: u32,
    ) -> std::result::Result<(Self, AudioFormat), WasapiCaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WasapiCaptureSource, &name, None);
        let _apartment = ComApartment::new()?;

        let audio_client = activate_process_loopback(process_id)?;
        let format = PROCESS_LOOPBACK_FORMAT;
        // SAFETY: COM is initialized on this thread, the client was just
        // activated, and `format` is a fully initialized `WAVEFORMATEX` that
        // outlives the call reading it.
        unsafe {
            audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                // Loopback, as an endpoint's playback side is captured — a
                // process loopback client is a capture of what is played and
                // takes the same flag. Polled rather than event-driven, for
                // the reason `POLL_INTERVAL` gives.
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                BUFFER_DURATION_100NS,
                0,
                &raw const format,
                None,
            )?;
        }
        let audio_format = resolve_mix_format(&raw const format).map_err(|error| {
            WasapiCaptureSourceError::UnsupportedMixFormat {
                format_tag: error.format_tag,
                bits: error.bits,
            }
        })?;
        // SAFETY: the client is initialized, which is what makes its capture
        // service available.
        let capture_client: IAudioCaptureClient = unsafe { audio_client.GetService()? };

        pp_info!(
            pp_log: &pp_log,
            "opened: process={}, {}Hz, {} channel(s), format={:?}",
            process_id,
            audio_format.sample_rate,
            audio_format.channels,
            audio_format.sample_format
        );
        Ok((
            Self::assemble(name, pp_log, audio_client, capture_client, audio_format),
            audio_format,
        ))
    }

    /// The parts every open ends with, whichever kind of capture it was.
    fn assemble(
        name: Arc<str>,
        pp_log: PpLog,
        audio_client: IAudioClient,
        capture_client: IAudioCaptureClient,
        audio_format: AudioFormat,
    ) -> Self {
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        );
        Self {
            name,
            pp_log,
            audio_client,
            capture_client,
            sample_rate: audio_format.sample_rate,
            format: audio_format.sample_format,
            channel_layout: audio_format.channel_layout(),
            samples_emitted: 0,
            pad,
        }
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.sample_rate as i32)
    }

    fn classify_error(&self, error: windows::core::Error) -> WasapiCaptureSourceError {
        if error.code() == AUDCLNT_E_DEVICE_INVALIDATED {
            WasapiCaptureSourceError::DeviceInvalidated
        } else {
            WasapiCaptureSourceError::Windows(error)
        }
    }

    /// Wraps one WASAPI packet (`data`/`frames`/`flags` straight out of
    /// [`IAudioCaptureClient::GetBuffer`]) into a fresh `ffmpeg::frame::Audio`
    /// and stamps its `pts`. `AUDCLNT_BUFFERFLAGS_SILENT` (the device has
    /// nothing real to report this tick, e.g. right after `Start`) or a
    /// null `data` pointer both mean "emit silence" rather than reading
    /// past the end of nothing.
    fn build_frame(&mut self, data: *mut u8, frames: u32, flags: u32) -> ffmpeg::frame::Audio {
        let mut frame =
            ffmpeg::frame::Audio::new(self.format, frames as usize, self.channel_layout);
        frame.set_rate(self.sample_rate);
        // `frame.data_mut(0)`'s length is FFmpeg's own padded linesize,
        // not necessarily `frames * channels * format.bytes()` exactly —
        // only ever touch that tight amount (the same bound
        // `frame.plane::<T>()` itself reads via `samples()`), never the
        // destination's full length, or a WASAPI buffer exactly `frames`
        // frames long could get read past its end.
        let tight_bytes =
            frames as usize * self.channel_layout.channels() as usize * self.format.bytes();
        if data.is_null() || flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0 {
            frame.data_mut(0)[..tight_bytes].fill(0);
        } else {
            // SAFETY: a successful `GetBuffer` makes `data` readable for
            // `frames * channels * sample_bytes` until `ReleaseBuffer`; the
            // destination slice was allocated for at least that tight size.
            unsafe {
                ptr::copy_nonoverlapping(data, frame.data_mut(0).as_mut_ptr(), tight_bytes);
            }
        }
        frame.set_pts(Some(self.samples_emitted));
        crate::buffer::set_time_base(&mut frame, self.time_base());
        self.samples_emitted += frames as i64;
        frame
    }

    /// Pushes `frame` downstream, reporting (rather than dying on) a
    /// failing `Sink` — same "drop this one buffer, keep going" contract
    /// [`crate::elements::DxgiCaptureSource::run`]/[`crate::elements::TestVideoSource::run`]
    /// give their own pushes.
    fn push_frame(&mut self, frame: ffmpeg::frame::Audio, bus: &Bus) {
        if let Err(error) = self.pad.push(MediaBuffer::Audio(Arc::new(frame))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::WasapiCaptureSource,
                    name: self.name.clone(),
                    error,
                },
            );
        }
    }

    /// Synthesizes and pushes one silence frame covering however many
    /// samples real WASAPI delivery has fallen behind `elapsed` — active
    /// wall-clock time since `run_captured` started, already excluding
    /// time spent frozen inside `Pause` (see
    /// [`crate::schedule::ActiveTimeline`]; a raw `Instant::elapsed()`
    /// would count a pause as a deficit to fill, making `Resume`
    /// synthesize one giant silence frame covering the whole pause) — a
    /// no-op (`deficit <= 0`) whenever real packets have kept up. WASAPI
    /// delivers **zero** packets
    /// whenever the render engine has no active session at all (as
    /// opposed to an active-but-quiet session, which still delivers
    /// `AUDCLNT_BUFFERFLAGS_SILENT`-flagged packets `build_frame` already
    /// turns into silence) — without this, nothing plays on the system
    /// would mean nothing at all comes out of this source, leaving a real
    /// gap in the audio timeline exactly when a downstream muxer/encoder
    /// needs `pts` to keep advancing to stay in sync with video. Backing
    /// every gap with synthesized silence (rather than, say, stretching
    /// the next real frame's `pts`) keeps `pts` a plain, always-accurate
    /// sample count no matter which samples were real.
    fn fill_silence_gap(&mut self, elapsed: Duration, bus: &Bus) {
        let expected = (elapsed.as_secs_f64() * self.sample_rate as f64) as i64;
        let deficit = expected - self.samples_emitted;
        if deficit <= 0 {
            return;
        }
        let mut frame =
            ffmpeg::frame::Audio::new(self.format, deficit as usize, self.channel_layout);
        frame.set_rate(self.sample_rate);
        frame.data_mut(0).fill(0);
        frame.set_pts(Some(self.samples_emitted));
        crate::buffer::set_time_base(&mut frame, self.time_base());
        self.samples_emitted += deficit;
        self.push_frame(frame, bus);
    }

    /// The main capture loop, run once COM has joined this thread's
    /// apartment (see [`SourceElement::run`]) and the audio client has been
    /// started. Drains every buffer WASAPI has ready on each
    /// `POLL_INTERVAL` tick (`GetNextPacketSize` returning `0` means
    /// caught up), pushing one `MediaBuffer::Audio` per packet, then tops
    /// up with synthesized silence (see [`WasapiCaptureSource::fill_silence_gap`])
    /// so `pts` keeps advancing with wall-clock time even across a tick
    /// where WASAPI delivered nothing at all.
    fn run_captured(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        let mut timeline = ActiveTimeline::new(Instant::now());
        loop {
            let outcome = crate::control::drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            timeline.account_pause(outcome.paused_for);

            thread::sleep(POLL_INTERVAL);

            loop {
                // SAFETY: `capture_client` is live while its parent audio
                // client is started; no packet is currently held here.
                let packet_size = match unsafe { self.capture_client.GetNextPacketSize() } {
                    Ok(size) => size,
                    Err(error) => return Err(self.classify_error(error).into()),
                };
                if packet_size == 0 {
                    break;
                }

                let mut data: *mut u8 = ptr::null_mut();
                let mut frames_available = 0u32;
                let mut flags = 0u32;
                // SAFETY: all three outputs are live locals, optional position
                // outputs are intentionally omitted, and no earlier packet is
                // outstanding on this serialized capture client.
                if let Err(error) = unsafe {
                    self.capture_client.GetBuffer(
                        &mut data,
                        &mut frames_available,
                        &mut flags,
                        None,
                        None,
                    )
                } {
                    return Err(self.classify_error(error).into());
                }

                let frame = self.build_frame(data, frames_available, flags);
                // SAFETY: balances the successful `GetBuffer` above with the
                // exact frame count it returned; `build_frame` copied all data
                // before the buffer becomes invalid.
                if let Err(error) = unsafe { self.capture_client.ReleaseBuffer(frames_available) } {
                    return Err(self.classify_error(error).into());
                }

                self.push_frame(frame, bus);
            }

            self.fill_silence_gap(timeline.elapsed(Instant::now()), bus);
        }
    }
}

impl Element for WasapiCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WasapiCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for WasapiCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WasapiCaptureSource {
    fn is_live(&self) -> bool {
        true
    }

    /// Stops the capture session for the pause: left running, nothing
    /// drains `IAudioCaptureClient::GetBuffer` while paused — harmless, as
    /// WASAPI's shared-mode ring buffer just overwrites what it cannot hold
    /// (see `BUFFER_DURATION_100NS`), but there is no reason to keep the
    /// device and the driver work behind it capturing audio nobody reads.
    /// `Stop` freezes the stream without discarding what was already
    /// pending; `Reset` discards it, so `Resume` cannot emit stale pre-pause
    /// audio as a short burst. A failure is fatal: the capture's state can no
    /// longer be vouched for.
    fn pausing(&mut self) -> Result<()> {
        // SAFETY: `audio_client` is initialized and currently started;
        // this thread serializes its lifecycle calls.
        unsafe { self.audio_client.Stop() }.map_err(|error| self.classify_error(error))?;
        // SAFETY: the client was successfully stopped immediately above,
        // which is the required state for `Reset`.
        unsafe { self.audio_client.Reset() }.map_err(|error| self.classify_error(error))?;
        Ok(())
    }

    /// Starts the capture again once everything downstream has resumed,
    /// so no audio accumulates during a slow cascade.
    fn resuming(&mut self) -> Result<()> {
        // SAFETY: this initialized client is stopped/reset and the source
        // thread exclusively sequences its lifecycle.
        unsafe { self.audio_client.Start() }.map_err(|error| self.classify_error(error))?;
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");

        let _apartment = ComApartment::new().map_err(WasapiCaptureSourceError::from)?;

        // SAFETY: this client was initialized in `open` and the source thread
        // exclusively owns its start/stop lifecycle.
        if let Err(error) = unsafe { self.audio_client.Start() } {
            return Err(self.classify_error(error).into());
        }

        let result = self.run_captured(control, bus);

        // SAFETY: balances the successful start for this run; calling Stop on
        // the live client is also the required cleanup after a loop error.
        if let Err(error) = unsafe { self.audio_client.Stop() } {
            pp_error!(self, "Stop failed: {error}");
        }
        result
    }

    fn seek(&mut self, _target: std::time::Duration) -> Result<std::time::Duration> {
        Err(WasapiCaptureSourceError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The format a process loopback asks for has to be one this crate can
    /// describe, since nothing else ever tells it what is arriving — a
    /// device capture reads its endpoint's, and there is no endpoint here.
    #[test]
    fn the_process_loopback_format_is_one_the_crate_can_name() {
        let asked = PROCESS_LOOPBACK_FORMAT;
        let format = resolve_mix_format(&raw const asked)
            .expect("the format this asks Windows for must be one this can read back");
        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 2);
        assert_eq!(
            format.sample_format,
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
        );
        // What `build_frame` copies per frame, which the format has to agree
        // with or every packet would be read short or long.
        assert_eq!(
            asked.nBlockAlign as usize,
            format.channels as usize * format.sample_format.bytes()
        );
    }

    /// What a process loopback hands over on a machine that has one to open:
    /// frames in the format it asked for, and a timeline that keeps up with
    /// the clock whether or not the process is making a noise — silence is
    /// filled, exactly as it is for a device (see
    /// [`WasapiCaptureSource::fill_silence_gap`]).
    ///
    /// Skips where there is nothing to capture, or where Windows refuses the
    /// activation — a build older than 10 2004 has no process loopback at
    /// all, and a runner with no audio stack has no session to point at.
    #[test]
    fn a_process_capture_keeps_a_timeline_of_its_own() {
        use crate::elements::AppSink;
        use std::sync::{Arc as StdArc, Mutex};

        let Ok(processes) = WasapiCaptureSource::list_processes() else {
            eprintln!("skipped: WASAPI would not enumerate sessions on this machine");
            return;
        };
        let Some(process) = processes.first() else {
            eprintln!("skipped: nothing on this machine holds an audio session");
            return;
        };
        let opened = WasapiCaptureSource::open_process("test-process-audio", process.id);
        let (source, format) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipped: this machine would not activate a process loopback: {error}");
                return;
            }
        };
        assert_eq!(format.sample_rate, 48_000, "the format it asked for");

        let samples = StdArc::new(Mutex::new(0i64));
        let sink = AppSink::new("test-process-audio-sink", {
            let samples = StdArc::clone(&samples);
            move |buffer| {
                if let MediaBuffer::Audio(frame) = &buffer {
                    *samples.lock().expect("sample count poisoned") += frame.samples() as i64;
                }
                Ok(())
            }
        });
        let (pipeline, ()) =
            crate::pipeline::Pipeline::new("test-process-audio", source, |source, ctx| {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("test pipeline wiring must succeed");

        pipeline.run().expect("the capture must start");
        let measured = Duration::from_millis(400);
        thread::sleep(measured);
        pipeline.stop();
        pipeline.bus().log_events();

        let samples = *samples.lock().expect("sample count poisoned");
        let expected = (measured.as_secs_f64() * format.sample_rate as f64) as i64;
        // Half, not all: the window includes starting the client and the
        // last poll interval's worth is still in flight when this stops.
        assert!(
            samples > expected / 2,
            "a capture of {} delivered {samples} samples where {expected} were due",
            process.executable
        );
    }

    /// Whatever this machine is running, a listed process is one a capture
    /// could be opened against: a live id, and a name for the picker to show
    /// where Windows would give one.
    #[test]
    fn every_listed_process_is_one_that_could_be_captured() {
        let Ok(processes) = WasapiCaptureSource::list_processes() else {
            eprintln!("skipped: WASAPI would not enumerate sessions on this machine");
            return;
        };
        if processes.is_empty() {
            eprintln!("skipped: nothing on this machine holds an audio session");
            return;
        }
        for process in &processes {
            assert_ne!(process.id, 0, "a session with no process is not listed");
        }
        let mut ids: Vec<u32> = processes.iter().map(|process| process.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            processes.len(),
            "a process playing to two endpoints is one entry"
        );
    }
}
