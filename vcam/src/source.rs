//! The camera's media source: one live video stream, started and stopped
//! as Frame Server asks.

use std::sync::{Arc, Mutex};

use windows::Win32::{
    Foundation::{E_INVALIDARG, E_POINTER, ERROR_SET_NOT_FOUND, S_OK},
    Media::{
        KernelStreaming::{
            IKsControl, IKsControl_Impl, KSCAMERAPROFILE_HighFrameRate, KSCAMERAPROFILE_Legacy,
            KSIDENTIFIER,
        },
        MediaFoundation::*,
    },
    System::Com::StructuredStorage::PROPVARIANT,
};
use windows_core::{BOOL, ComObject, GUID, HRESULT, IUnknown, Interface, Ref, Result, implement};

use crate::{
    attributes::{delegate_attributes, store},
    stream::Stream,
};

/// The source, as Media Foundation sees it.
#[implement(
    IMFMediaSource2,
    IMFGetService,
    IKsControl,
    IMFSampleAllocatorControl,
    IMFAttributes
)]
pub(crate) struct Source {
    attributes: IMFAttributes,
    queue: IMFMediaEventQueue,
    stream: ComObject<Stream>,
    descriptor: Mutex<Option<IMFPresentationDescriptor>>,
}

impl Source {
    /// A source whose stream shares pictures through the section `section_name`.
    pub(crate) fn create(section_name: Arc<str>) -> Result<IMFMediaSource> {
        let stream = ComObject::new(Stream::new(0, section_name)?);
        // SAFETY: a descriptor over the stream's own descriptor.
        let descriptor =
            unsafe { MFCreatePresentationDescriptor(Some(&[Some(stream.descriptor().clone())]))? };
        let attributes = store()?;
        sensor_profiles(&attributes)?;
        // SAFETY: creates a new, empty event queue.
        let queue = unsafe { MFCreateEventQueue()? };
        let source = ComObject::new(Source {
            attributes,
            queue,
            stream: stream.clone(),
            descriptor: Mutex::new(Some(descriptor)),
        });
        let source: IMFMediaSource = source.to_interface::<IMFMediaSource2>().cast()?;
        stream.set_source(source.clone());
        Ok(source)
    }

    fn descriptor(&self) -> Result<IMFPresentationDescriptor> {
        self.descriptor
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| MF_E_SHUTDOWN.into())
    }

    fn queue_event(&self, event: MF_EVENT_TYPE) -> Result<()> {
        // SAFETY: an event this source's own queue carries.
        unsafe {
            self.queue
                .QueueEventParamVar(event.0 as u32, &GUID::zeroed(), S_OK, std::ptr::null())
        }
    }
}

/// The two profiles Windows' camera pipeline expects a camera to describe —
/// ordinary, and high frame rate — each allowing any size and subtype.
fn sensor_profiles(attributes: &IMFAttributes) -> Result<()> {
    // SAFETY: builds a collection this function owns, then hands it over.
    unsafe {
        let collection = MFCreateSensorProfileCollection()?;
        for (kind, filter) in [
            (&KSCAMERAPROFILE_Legacy, "((RES==;FRT<=30,1;SUT==))"),
            (&KSCAMERAPROFILE_HighFrameRate, "((RES==;FRT>=60,1;SUT==))"),
        ] {
            let profile = MFCreateSensorProfile(kind, 0, None)?;
            profile.AddProfileFilter(0, &windows_core::HSTRING::from(filter))?;
            collection.AddProfile(&profile)?;
        }
        attributes.SetUnknown(&MF_DEVICEMFT_SENSORPROFILE_COLLECTION, &collection)
    }
}

impl IMFMediaEventGenerator_Impl for Source_Impl {
    fn GetEvent(&self, flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS) -> Result<IMFMediaEvent> {
        // SAFETY: forwarded to the source's own queue.
        unsafe { self.queue.GetEvent(flags.0) }
    }

    fn BeginGetEvent(&self, callback: Ref<IMFAsyncCallback>, state: Ref<IUnknown>) -> Result<()> {
        // SAFETY: as above.
        unsafe { self.queue.BeginGetEvent(callback.as_ref(), state.as_ref()) }
    }

    fn EndGetEvent(&self, result: Ref<IMFAsyncResult>) -> Result<IMFMediaEvent> {
        // SAFETY: as above.
        unsafe { self.queue.EndGetEvent(result.as_ref()) }
    }

    fn QueueEvent(
        &self,
        event: u32,
        extended: *const GUID,
        status: HRESULT,
        value: *const PROPVARIANT,
    ) -> Result<()> {
        // SAFETY: as above.
        unsafe {
            self.queue
                .QueueEventParamVar(event, extended, status, value)
        }
    }
}

impl IMFMediaSource_Impl for Source_Impl {
    fn GetCharacteristics(&self) -> Result<u32> {
        self.descriptor()?;
        Ok(MFMEDIASOURCE_IS_LIVE.0 as u32)
    }

    fn CreatePresentationDescriptor(&self) -> Result<IMFPresentationDescriptor> {
        // SAFETY: hands out a copy, so the caller's selections are its own.
        unsafe { self.descriptor()?.Clone() }
    }

    fn Start(
        &self,
        descriptor: Ref<IMFPresentationDescriptor>,
        time_format: *const GUID,
        _position: *const PROPVARIANT,
    ) -> Result<()> {
        let ours = self.descriptor()?;
        let theirs = descriptor.ok()?;
        // SAFETY: `time_format` is null or points at a GUID, per the call.
        if !time_format.is_null() && unsafe { *time_format } != GUID::zeroed() {
            return Err(MF_E_UNSUPPORTED_TIME_FORMAT.into());
        }
        // SAFETY: the stream descriptors of both presentation descriptors,
        // read and selected through their own calls.
        unsafe {
            let count = theirs.GetStreamDescriptorCount()?;
            for index in 0..count {
                let mut selected = BOOL::default();
                let mut stream_descriptor = None;
                theirs.GetStreamDescriptorByIndex(index, &mut selected, &mut stream_descriptor)?;
                let stream_descriptor =
                    stream_descriptor.ok_or_else(|| windows_core::Error::from(E_POINTER))?;
                if stream_descriptor.GetStreamIdentifier()? != 0 {
                    return Err(E_INVALIDARG.into());
                }
                if selected.as_bool() {
                    ours.SelectStream(index)?;
                    let media_type = stream_descriptor
                        .GetMediaTypeHandler()?
                        .GetCurrentMediaType()?;
                    let stream: IUnknown = self.stream.to_interface();
                    self.queue.QueueEventParamUnk(
                        MENewStream.0 as u32,
                        &GUID::zeroed(),
                        S_OK,
                        &stream,
                    )?;
                    self.stream.start(Some(media_type))?;
                } else {
                    ours.DeselectStream(index)?;
                    self.stream.stop()?;
                }
            }
        }
        self.queue_event(MESourceStarted)
    }

    fn Stop(&self) -> Result<()> {
        let ours = self.descriptor()?;
        self.stream.stop()?;
        // SAFETY: deselects the one stream the descriptor holds.
        unsafe { ours.DeselectStream(0)? };
        self.queue_event(MESourceStopped)
    }

    fn Pause(&self) -> Result<()> {
        Err(MF_E_INVALID_STATE_TRANSITION.into())
    }

    fn Shutdown(&self) -> Result<()> {
        let taken = self
            .descriptor
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if taken.is_none() {
            return Err(MF_E_SHUTDOWN.into());
        }
        self.stream.shutdown();
        // SAFETY: shuts the queue down once; later calls fail with
        // `MF_E_SHUTDOWN`, as they should.
        unsafe { self.queue.Shutdown() }
    }
}

impl IMFMediaSourceEx_Impl for Source_Impl {
    fn GetSourceAttributes(&self) -> Result<IMFAttributes> {
        Ok(self.attributes.clone())
    }

    fn GetStreamAttributes(&self, stream: u32) -> Result<IMFAttributes> {
        if stream != 0 {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        Ok(self.stream.to_interface())
    }

    fn SetD3DManager(&self, _manager: Ref<IUnknown>) -> Result<()> {
        // Pictures are written in system memory, which every reader takes;
        // a device would only matter to a source that drew on the GPU.
        Ok(())
    }
}

impl IMFMediaSource2_Impl for Source_Impl {
    fn SetMediaType(&self, stream: u32, _media_type: Ref<IMFMediaType>) -> Result<()> {
        if stream != 0 {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        Ok(())
    }
}

impl IMFGetService_Impl for Source_Impl {
    fn GetService(
        &self,
        _: *const GUID,
        _: *const GUID,
        _: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        Err(MF_E_UNSUPPORTED_SERVICE.into())
    }
}

impl IMFSampleAllocatorControl_Impl for Source_Impl {
    fn SetDefaultAllocator(&self, stream: u32, allocator: Ref<IUnknown>) -> Result<()> {
        if stream != 0 {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        let allocator = match allocator.as_ref() {
            Some(allocator) => Some(allocator.cast::<IMFVideoSampleAllocatorEx>()?),
            None => None,
        };
        self.stream.set_allocator(allocator);
        Ok(())
    }

    fn GetAllocatorUsage(
        &self,
        stream: u32,
        input: *mut u32,
        usage: *mut MFSampleAllocatorUsage,
    ) -> Result<()> {
        if stream != 0 {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        if input.is_null() || usage.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: both checked non-null above, and are the caller's to write.
        unsafe {
            *input = stream;
            *usage = MFSampleAllocatorUsage_UsesProvidedAllocator;
        }
        Ok(())
    }
}

impl IKsControl_Impl for Source_Impl {
    fn KsProperty(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }

    fn KsMethod(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }

    fn KsEvent(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }
}

delegate_attributes!(Source_Impl);
