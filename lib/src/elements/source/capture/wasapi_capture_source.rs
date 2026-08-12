use std::{
    ffi::c_void,
    ptr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Devices::FunctionDiscovery::PKEY_Device_FriendlyName,
        Media::{
            Audio::{
                AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK, DEVICE_STATE_ACTIVE, IAudioCaptureClient,
                IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, WAVE_FORMAT_PCM,
                WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eCapture, eConsole, eRender,
            },
            KernelStreaming::{KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE},
            Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT},
        },
        System::{
            Com::{
                CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
                CoUninitialize, STGM_READ,
                StructuredStorage::{PROPVARIANT, PropVariantClear},
            },
            Variant::VT_LPWSTR,
        },
        UI::Shell::PropertiesSystem::IPropertyStore,
    },
    core::HSTRING,
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
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

/// Which direction an [`AudioDevice`] flows. Determines how
/// [`WasapiCaptureSource::open`] has to `Initialize` the stream — a
/// `Render` endpoint (speakers/headphones) only has an outgoing signal to
/// tap via WASAPI loopback (`AUDCLNT_STREAMFLAGS_LOOPBACK`); a `Capture`
/// endpoint (microphone) is already an input, captured directly. Callers
/// picking a device out of [`WasapiCaptureSource::list_devices`] don't need
/// to reason about this themselves — it travels with the `AudioDevice`
/// they chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDeviceKind {
    /// A playback endpoint — its *outgoing* mix is what gets captured
    /// (loopback). The system audio a screen recording should carry
    /// alongside [`crate::elements::DxgiCaptureSource`].
    Render,
    /// A microphone or other recording endpoint — captured directly.
    Capture,
}

/// One WASAPI endpoint enumerated by [`WasapiCaptureSource::list_devices`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDevice {
    /// Opaque device identifier (`IMMDevice::GetId`) — pass straight into
    /// [`WasapiCaptureOptions::device`], never parsed. Stable for as long
    /// as the device stays plugged in, not guaranteed stable across a
    /// reboot or a driver reinstall.
    pub id: String,
    /// Human-readable name for a picker UI (e.g. "Speakers (Realtek High
    /// Definition Audio)"), falling back to `id` if a friendly name
    /// couldn't be read.
    pub name: String,
    pub kind: AudioDeviceKind,
    /// Whether this was the system default endpoint for `kind`'s role at
    /// enumeration time — a `Render` device with `is_default: true` is
    /// what most "record my screen" callers want.
    pub is_default: bool,
}

/// Construction-time options for [`WasapiCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct WasapiCaptureOptions {
    /// Which endpoint to capture from — one entry out of
    /// [`WasapiCaptureSource::list_devices`] (or hand-built, if the caller
    /// already knows a device's id/kind some other way).
    pub device: AudioDevice,
}

/// Captures audio via WASAPI (`IAudioClient`/`IAudioCaptureClient`) —
/// GStreamer's `wasapi2src` equivalent. One src pad, pushing
/// `MediaBuffer::Audio` frames in the captured device's own native mix
/// format/rate/channel count — no resampling. Same division of labor as
/// [`crate::elements::DxgiCaptureSource`] emitting raw `Pixel::BGRA` and
/// leaving conversion to a downstream [`crate::elements::Scaler`]: if
/// something downstream needs a fixed sample rate/format, that's a future
/// audio resampler filter's job, not this element's.
///
/// Polls `IAudioCaptureClient::GetNextPacketSize` on a short fixed
/// interval (`POLL_INTERVAL`) rather than waiting on a WASAPI-signaled
/// event — see that constant's own docs on why event-driven mode isn't
/// used here.
///
/// Emits continuously from the moment `run` starts, `pts` always in
/// lockstep with wall-clock time — backed by real WASAPI data when it's
/// available and synthesized silence otherwise (see
/// [`WasapiCaptureSource::fill_silence_gap`]), since WASAPI itself
/// delivers literally nothing whenever the render engine has no active
/// session at all (e.g. nothing currently playing, for
/// [`AudioDeviceKind::Render`]). Without this, a quiet period would be a
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
/// `cpal`'s own WASAPI backend uses. `open`'s own `CoInitializeEx` (on the
/// *caller's* thread) is deliberately never paired with a matching
/// `CoUninitialize`: this struct can be dropped on a different thread than
/// `open` was called from (typically it is — the pipeline's worker
/// thread), so there is no single correct thread to uninitialize from.
/// One extra, never-undone `CoInitializeEx` on the calling thread for the
/// life of the process is harmless — the same thing most GUI apps already
/// do on their main thread.
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
#[rust_hlog::hlog]
pub struct WasapiCaptureSource {
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
    /// (playback) and `Capture` (recording) — as an [`AudioDevice`] list a
    /// caller can show in a picker UI and index/search into, then hand the
    /// chosen entry straight to [`WasapiCaptureOptions::device`]. No
    /// concept of "mode" to reason about beforehand: the picked device's
    /// own [`AudioDeviceKind`] is what tells `open` whether to use
    /// loopback.
    pub fn list_devices() -> std::result::Result<Vec<AudioDevice>, WasapiCaptureSourceError> {
        unsafe {
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
            if hr.is_err() {
                return Err(windows::core::Error::from(hr).into());
            }

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;

            let mut devices = Vec::new();
            for (dataflow, kind) in [
                (eRender, AudioDeviceKind::Render),
                (eCapture, AudioDeviceKind::Capture),
            ] {
                let default_id = enumerator
                    .GetDefaultAudioEndpoint(dataflow, eConsole)
                    .ok()
                    .and_then(|device| device.GetId().ok())
                    .and_then(|id| id.to_string().ok());

                let collection = enumerator.EnumAudioEndpoints(dataflow, DEVICE_STATE_ACTIVE)?;
                let count = collection.GetCount()?;
                for index in 0..count {
                    let device = collection.Item(index)?;
                    let Some(id) = device.GetId().ok().and_then(|id| id.to_string().ok()) else {
                        continue; // couldn't read this one's id — skip rather than fail the whole list
                    };
                    let name = device_friendly_name(&device).unwrap_or_else(|| id.clone());
                    let is_default = default_id.as_deref() == Some(id.as_str());
                    devices.push(AudioDevice {
                        id,
                        name,
                        kind,
                        is_default,
                    });
                }
            }
            Ok(devices)
        }
    }

    /// Opens `options.device` and starts a shared-mode WASAPI capture
    /// session. Returns the element alongside the captured stream's
    /// actual `(sample_rate, channels)` — what a caller needs to build a
    /// matching downstream encoder/muxer, same pattern as
    /// [`crate::elements::DxgiCaptureSource::open`] returning
    /// `(width, height)`.
    pub fn open(
        name: impl Into<String>,
        options: WasapiCaptureOptions,
    ) -> std::result::Result<(Self, u32, u16), WasapiCaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::WasapiCaptureSource, &name, None);

        unsafe {
            // See this struct's own docs on why this call is intentionally
            // never paired with a `CoUninitialize`.
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
            if hr.is_err() {
                return Err(windows::core::Error::from(hr).into());
            }

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let device_id = HSTRING::from(options.device.id.as_str());
            let device: IMMDevice = enumerator.GetDevice(&device_id)?;
            let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

            let mix_format = audio_client.GetMixFormat()?;
            let (format, channel_layout) = resolve_sample_format(mix_format)?;
            let sample_rate = (*mix_format).nSamplesPerSec;
            let channels = (*mix_format).nChannels;

            let stream_flags = match options.device.kind {
                AudioDeviceKind::Render => AUDCLNT_STREAMFLAGS_LOOPBACK,
                AudioDeviceKind::Capture => 0,
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

            hinfo!(
                hlog: &hlog,
                "opened: device={:?} ({:?}), {}Hz, {} channel(s), format={:?}",
                options.device.name,
                options.device.kind,
                sample_rate,
                channels,
                format
            );
            let pad = SrcPad::new(format!("{name}_src"));

            Ok((
                Self {
                    name,
                    hlog,
                    audio_client,
                    capture_client,
                    sample_rate,
                    format,
                    channel_layout,
                    samples_emitted: 0,
                    pad,
                },
                sample_rate,
                channels,
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
                &self.hlog,
                BusEvent::Error {
                    element_type: ElementType::WasapiCaptureSource,
                    name: self.name.clone(),
                    error,
                },
            );
        }
    }

    /// Synthesizes and pushes one silence frame covering however many
    /// samples real WASAPI delivery has fallen behind wall-clock time
    /// since `run_captured` started — a no-op (`deficit <= 0`) whenever
    /// real packets have kept up. WASAPI delivers **zero** packets
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
    fn fill_silence_gap(&mut self, start: Instant, bus: &Bus) {
        let expected = (start.elapsed().as_secs_f64() * self.sample_rate as f64) as i64;
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

    /// The main capture loop, run once COM has joined this thread's
    /// apartment (see [`SourceElement::run`]) and the audio client has been
    /// started. Drains every buffer WASAPI has ready on each
    /// `POLL_INTERVAL` tick (`GetNextPacketSize` returning `0` means
    /// caught up), pushing one `MediaBuffer::Audio` per packet, then tops
    /// up with synthesized silence (see [`WasapiCaptureSource::fill_silence_gap`])
    /// so `pts` keeps advancing with wall-clock time even across a tick
    /// where WASAPI delivered nothing at all.
    fn run_captured(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        let start = Instant::now();
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }

            thread::sleep(POLL_INTERVAL);

            loop {
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
                if let Err(error) = unsafe { self.capture_client.ReleaseBuffer(frames_available) } {
                    return Err(self.classify_error(error).into());
                }

                self.push_frame(frame, bus);
            }

            self.fill_silence_gap(start, bus);
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

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for WasapiCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WasapiCaptureSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");

        // See this struct's own docs on why every thread that touches
        // `audio_client`/`capture_client` needs its own `CoInitializeEx`
        // call, this one paired with `CoUninitialize` below since both run
        // on this same thread start to finish.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_err() {
            return Err(WasapiCaptureSourceError::from(windows::core::Error::from(hr)).into());
        }

        if let Err(error) = unsafe { self.audio_client.Start() } {
            unsafe { CoUninitialize() };
            return Err(self.classify_error(error).into());
        }

        let result = self.run_captured(control, bus);

        if let Err(error) = unsafe { self.audio_client.Stop() } {
            herror!(self, "Stop failed: {error}");
        }
        unsafe { CoUninitialize() };
        result
    }

    fn seek(&mut self, _target: std::time::Duration) -> Result<std::time::Duration> {
        Err(WasapiCaptureSourceError::SeekUnsupported.into())
    }
}

/// Reads `device`'s `PKEY_Device_FriendlyName` property (e.g. "Speakers
/// (Realtek High Definition Audio)") for [`WasapiCaptureSource::list_devices`].
/// `None` if the property store can't be opened or the value isn't a
/// string — every real endpoint has this property, so that's effectively
/// "shouldn't happen", not something worth a hard error over; the caller
/// falls back to the device's id instead.
fn device_friendly_name(device: &IMMDevice) -> Option<String> {
    unsafe {
        let store: IPropertyStore = device.OpenPropertyStore(STGM_READ).ok()?;
        let mut variant: PROPVARIANT = store.GetValue(&PKEY_Device_FriendlyName).ok()?;
        let name = property_variant_to_string(&variant);
        let _ = PropVariantClear(&mut variant);
        name
    }
}

/// Extracts a `VT_LPWSTR` string out of a `PROPVARIANT` — the only variant
/// type `PKEY_Device_FriendlyName` is ever actually stored as. `None` for
/// any other `vt`, rather than guessing at a union field that isn't
/// active.
fn property_variant_to_string(variant: &PROPVARIANT) -> Option<String> {
    unsafe {
        if variant.Anonymous.Anonymous.vt != VT_LPWSTR {
            return None;
        }
        variant
            .Anonymous
            .Anonymous
            .Anonymous
            .pwszVal
            .to_string()
            .ok()
    }
}

/// Resolves a WASAPI `WAVEFORMATEX` (possibly a `WAVEFORMATEXTENSIBLE` in
/// disguise — `wFormatTag == WAVE_FORMAT_EXTENSIBLE`, common on modern
/// multi-channel devices) into the `ffmpeg::format::Sample` it corresponds
/// to plus a matching [`ffmpeg::ChannelLayout`]. Only PCM and IEEE float,
/// 16/32-bit, are handled — every WASAPI shared-mode mix format actually
/// seen on real hardware is one of these two; anything else is a hard
/// `open`-time error rather than a silent, likely-wrong guess.
fn resolve_sample_format(
    mix_format: *const WAVEFORMATEX,
) -> std::result::Result<(ffmpeg::format::Sample, ffmpeg::ChannelLayout), WasapiCaptureSourceError>
{
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

    let format = match (format_tag, bits) {
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
            return Err(WasapiCaptureSourceError::UnsupportedMixFormat { format_tag, bits });
        }
    };
    let channel_layout = ffmpeg::ChannelLayout::default(wf.nChannels as i32);
    Ok((format, channel_layout))
}
