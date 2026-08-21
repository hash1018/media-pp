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
    Media::Audio::{
        AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK, IAudioCaptureClient, IAudioClient,
    },
    System::Com::{CLSCTX_ALL, CoTaskMemFree},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{self, ControlMsg, ControlOutcome, ControlReceiver, RequestKind},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::AudioFormat,
    error::Result,
    pad::SrcPad,
    platform::windows::wasapi::{
        ComApartment, WasapiDevice, WasapiDeviceKind, list_devices as enumerate_wasapi_devices,
        open_device, resolve_mix_format,
    },
    schedule::ActiveTimeline,
};

/// How long [`WasapiCaptureSource::run`] sleeps between checks of
/// `GetNextPacketSize` — also bounds `Stop` latency, same reasoning as
/// [`crate::elements::DxgiCaptureSource`]'s own `POLL_GRANULARITY`. Plain
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

/// Errors specific to `WasapiCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WasapiCaptureSourceError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// `AUDCLNT_E_DEVICE_INVALIDATED` specifically, broken out of the
    /// generic [`WasapiCaptureSourceError::Windows`] variant for the same
    /// reason [`crate::elements::DxgiCaptureSourceError::AccessLost`] is:
    /// the single most common *recoverable* failure (default device
    /// changed, device unplugged, format changed) surfaces this way. Same
    /// "fail fast, caller rebuilds a fresh one" contract
    /// [`crate::elements::RtspSource`]/[`crate::elements::DxgiCaptureSource`]
    /// already document: this element doesn't retry internally.
    #[error("AUDCLNT_E_DEVICE_INVALIDATED — audio device needs to be reopened")]
    DeviceInvalidated,

    #[error("WasapiCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,

    #[error("unsupported WASAPI mix format: format_tag={format_tag}, bits_per_sample={bits}")]
    UnsupportedMixFormat { format_tag: u32, bits: u16 },
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
/// [`crate::elements::DxgiCaptureSource`] emitting raw `Pixel::BGRA` and
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
    /// [`crate::elements::DxgiCaptureSource::open`] returning a
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
            let pad = SrcPad::new(format!("{name}_src"));

            Ok((
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
                },
                audio_format,
            ))
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
        self.samples_emitted += deficit;
        self.push_frame(frame, bus);
    }

    /// Like [`crate::control::drain_control`], but drives the raw control
    /// receiver and [`crate::control::apply_one`] directly (same reason
    /// [`crate::elements::AppSource`] does — see `drain_control`'s own
    /// docs) so it can bracket the blocking `Pause` wait with
    /// `IAudioClient::Stop`/`Reset`/`Start`: without this, the capture
    /// session keeps running the whole time this source is paused, with
    /// nothing ever draining `IAudioCaptureClient::GetBuffer` to read it —
    /// harmless in that WASAPI's shared-mode ring buffer just silently
    /// overwrites whatever it can't hold (see `BUFFER_DURATION_100NS`), but
    /// there's no reason to keep the device (and whatever driver-side work
    /// backs it) actively capturing audio nobody is ever going to read.
    /// `Reset` also flushes data already pending at the pause boundary, so
    /// Resume cannot emit stale pre-pause packets as a short burst. These
    /// device transitions happen before their synchronous control request
    /// is acknowledged; failure is fatal because correct capture state can
    /// no longer be guaranteed.
    fn handle_control(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<ControlOutcome> {
        let mut paused_for = Duration::ZERO;
        while let Some((request, ack)) = control.try_recv() {
            let RequestKind::Control(msg) = request else {
                control::apply_finish(self, bus, &ack);
                return Ok(ControlOutcome {
                    stopped: true,
                    paused_for,
                });
            };
            if msg != ControlMsg::Pause {
                if control::apply_one(self, bus, msg, &ack)? {
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                }
                continue;
            }

            // Include device shutdown and both synchronous downstream
            // cascades in the frozen interval: this source produces no
            // media during any of them.
            let pause_start = Instant::now();
            // SAFETY: `audio_client` is initialized and currently started;
            // this thread serializes its lifecycle calls.
            unsafe { self.audio_client.Stop() }.map_err(|error| self.classify_error(error))?;
            // Stop freezes the stream but does not discard packets that
            // were already pending. Reset flushes those packets so Resume
            // cannot dump stale pre-pause audio in a short burst.
            // SAFETY: the client was successfully stopped immediately above,
            // which is the required state for `Reset`.
            unsafe { self.audio_client.Reset() }.map_err(|error| self.classify_error(error))?;
            control::apply_one(self, bus, msg, &ack)?;

            loop {
                let Some((paused_msg, paused_ack)) = control.recv() else {
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                };

                let RequestKind::Control(paused_msg) = paused_msg else {
                    control::apply_finish(self, bus, &paused_ack);
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                };

                if paused_msg == ControlMsg::Resume {
                    // Keep capture stopped until every downstream element
                    // has resumed. Start the device and only then ack, so
                    // the synchronous caller cannot observe a half-resumed
                    // source and no audio accumulates during a slow cascade.
                    control::apply_one_unacked(self, bus, paused_msg)?;
                    // SAFETY: this initialized client is stopped/reset and the
                    // source thread exclusively sequences its lifecycle.
                    unsafe { self.audio_client.Start() }
                        .map_err(|error| self.classify_error(error))?;
                    let _ = paused_ack.send(());
                    paused_for += pause_start.elapsed();
                    break;
                }

                if control::apply_one(self, bus, paused_msg, &paused_ack)? {
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                }
                // A redundant Pause (or another one-shot control) was
                // forwarded and acknowledged; remain frozen until Resume.
            }
        }
        Ok(ControlOutcome {
            stopped: false,
            paused_for,
        })
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
            let outcome = self.handle_control(control, bus)?;
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
