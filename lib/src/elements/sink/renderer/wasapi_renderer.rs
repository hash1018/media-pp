use std::{ffi::c_void, ptr, sync::Arc, thread, time::Duration};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;
use windows::Win32::{
    Media::Audio::{
        AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED, IAudioClient, IAudioRenderClient,
    },
    System::Com::{CLSCTX_ALL, CoTaskMemFree},
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    elements::{AudioFormat, WasapiDevice, WasapiDeviceKind},
    error::Result,
    platform::windows::wasapi::{
        ComApartment, list_devices as enumerate_wasapi_devices, open_device, resolve_mix_format,
    },
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
}

/// Terminal audio sink backed by a WASAPI shared-mode render endpoint.
/// The endpoint's mix format is returned by [`WasapiRenderer::open`] so a
/// caller can place an [`crate::elements::AudioResampler`] immediately
/// before this sink. This element intentionally performs no hidden format
/// conversion.
///
/// Device-buffer backpressure is the playback clock: `consume` waits for
/// enough WASAPI ring-buffer space to submit the whole input frame. Put a
/// [`crate::queue::Queue`] immediately before this sink when its blocking
/// must not hold up another branch.
#[rust_hlog::hlog]
pub struct WasapiRenderer {
    name: Arc<str>,
    audio_client: IAudioClient,
    render_client: IAudioRenderClient,
    format: AudioFormat,
    buffer_frames: u32,
    running: bool,
    paused: bool,
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
        let hlog = element_hlog(ElementType::WasapiRenderer, &name, None);
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
        let buffer_frames = unsafe { audio_client.GetBufferSize()? };
        hinfo!(
            hlog: &hlog,
            "opened: device={:?}, {}Hz, {} channel(s), format={:?}, buffer_frames={buffer_frames}",
            options.device.name,
            format.sample_rate,
            format.channels,
            format.sample_format
        );

        Ok((
            Self {
                name,
                hlog,
                audio_client,
                render_client,
                format,
                buffer_frames,
                running: false,
                paused: false,
            },
            format,
        ))
    }

    pub fn format(&self) -> AudioFormat {
        self.format
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
        unsafe { self.audio_client.Reset() }.map_err(|error| self.classify_error(error))?;
        Ok(())
    }

    fn render(&mut self, frame: &ffmpeg::frame::Audio) -> Result<()> {
        let bytes = validate_frame(self.format, frame)?;
        if frame.samples() == 0 || self.paused {
            return Ok(());
        }

        let bytes_per_frame = self.format.sample_format.bytes() * self.format.channels as usize;
        let mut frame_offset = 0usize;
        while frame_offset < frame.samples() {
            let padding = unsafe { self.audio_client.GetCurrentPadding() }
                .map_err(|error| self.classify_error(error))?;
            let available = self.buffer_frames.saturating_sub(padding) as usize;
            if available == 0 {
                self.start()?;
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
            frame_offset += take;
            self.start()?;
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
            thread::sleep(POLL_INTERVAL);
        }
        self.stop_and_reset()
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

impl Element for WasapiRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WasapiRenderer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for WasapiRenderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let _apartment = ComApartment::new().map_err(WasapiRendererError::from)?;
        match buf {
            MediaBuffer::Audio(frame) => self
                .render(&frame)
                .inspect_err(|error| herror!(self, "render failed: {error}")),
            MediaBuffer::Eos => self
                .drain()
                .inspect_err(|error| herror!(self, "drain failed: {error}")),
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
                self.paused = true;
            }
            ControlMsg::Resume => {
                self.paused = false;
                let padding = unsafe { self.audio_client.GetCurrentPadding() }
                    .map_err(|error| self.classify_error(error))?;
                if padding > 0 {
                    self.start()?;
                }
            }
            ControlMsg::Stop | ControlMsg::Seek(_) => {
                self.paused = false;
                self.stop_and_reset()?;
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
        }
        let _ = unsafe { self.audio_client.Reset() };
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg::format::sample::Type;

    use super::*;

    fn frame(format: AudioFormat, samples: usize) -> ffmpeg::frame::Audio {
        let mut frame =
            ffmpeg::frame::Audio::new(format.sample_format, samples, format.channel_layout());
        frame.set_rate(format.sample_rate);
        frame.data_mut(0).fill(0);
        frame
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
}
