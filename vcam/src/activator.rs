//! What the class factory hands out: Frame Server activates the camera
//! through an `IMFActivate`, which makes the source on first use and keeps
//! it until told to let go.

use std::sync::{Arc, Mutex};

use windows::Win32::Media::MediaFoundation::*;
use windows_core::{GUID, Interface, Result, implement};

use crate::{
    attributes::{delegate_attributes, store},
    source::Source,
};

#[implement(IMFActivate, IMFAttributes)]
pub(crate) struct Activator {
    attributes: IMFAttributes,
    section_name: Arc<str>,
    source: Mutex<Option<IMFMediaSource>>,
}

impl Activator {
    /// An activator whose source shares pictures through `section_name`.
    pub(crate) fn new(section_name: &str) -> Result<Self> {
        let attributes = store()?;
        // SAFETY: a plain value on a store this activator owns.
        unsafe { attributes.SetUINT32(&MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES, 1)? };
        Ok(Self {
            attributes,
            section_name: section_name.into(),
            source: Mutex::new(None),
        })
    }

    fn source(&self) -> std::sync::MutexGuard<'_, Option<IMFMediaSource>> {
        self.source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl IMFActivate_Impl for Activator_Impl {
    fn ActivateObject(&self, iid: *const GUID, object: *mut *mut core::ffi::c_void) -> Result<()> {
        let mut source = self.source();
        if source.is_none() {
            *source = Some(Source::create(self.section_name.clone())?);
        }
        let source = source.as_ref().expect("made above");
        // SAFETY: `iid` and `object` are the caller's, passed straight to
        // the source's own `QueryInterface`.
        unsafe { source.query(iid, object).ok() }
    }

    fn ShutdownObject(&self) -> Result<()> {
        if let Some(source) = self.source().take() {
            // SAFETY: shuts down the source this activator made.
            let _ = unsafe { source.Shutdown() };
        }
        Ok(())
    }

    fn DetachObject(&self) -> Result<()> {
        self.source().take();
        Ok(())
    }
}

delegate_attributes!(Activator_Impl);
