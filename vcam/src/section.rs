//! The camera's side of the shared section: it creates it, so that it can
//! name it globally and say who may open it.

use windows::Win32::{
    Foundation::{CloseHandle, HANDLE, HLOCAL, INVALID_HANDLE_VALUE, LocalFree},
    Security::{
        Authorization::{ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1},
        PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES,
    },
    System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        PAGE_READWRITE, UnmapViewOfFile,
    },
};
use windows_core::HSTRING;

use crate::protocol::{self, Header};

/// Who may open the section: the system, the service the camera runs in,
/// administrators, and any signed-in user for reading and writing — the
/// producing application runs as one, in another session.
const SECURITY: &str = "D:P(A;;GA;;;SY)(A;;GA;;;LS)(A;;GA;;;BA)(A;;GRGW;;;AU)";

/// A mapped section, created (or found) under one name.
pub(crate) struct Section {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
}

// SAFETY: a section handle and a mapped view may be used from any thread;
// what is in the view is only touched through the atomics of `Header` and
// the sequence-guarded copy in `protocol`.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    /// Creates the section `name`, or opens it where it is already there,
    /// and stamps it as this protocol's.
    pub(crate) fn create(name: &str) -> windows_core::Result<Self> {
        let mut descriptor = PSECURITY_DESCRIPTOR::default();
        // SAFETY: `SECURITY` is a valid SDDL string; the descriptor it makes
        // is freed below with `LocalFree`, as the call requires.
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                &HSTRING::from(SECURITY),
                SDDL_REVISION_1,
                &mut descriptor,
                None,
            )?
        };
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor.0,
            bInheritHandle: false.into(),
        };
        let size = protocol::SECTION_BYTES as u64;
        // SAFETY: a pagefile-backed mapping of a fixed size, with the
        // attributes built above, which outlive the call.
        let created = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                Some(&attributes),
                PAGE_READWRITE,
                (size >> 32) as u32,
                size as u32,
                &HSTRING::from(name),
            )
        };
        // SAFETY: allocated by the conversion above, and no longer used.
        unsafe { LocalFree(Some(HLOCAL(descriptor.0))) };
        let handle = created?;
        // SAFETY: `handle` is the mapping just created or opened.
        let view =
            unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, protocol::SECTION_BYTES) };
        if view.Value.is_null() {
            let error = windows_core::Error::from_thread();
            // SAFETY: the handle is ours and unused from here.
            let _ = unsafe { CloseHandle(handle) };
            return Err(error);
        }
        let section = Self { handle, view };
        let header = section.header();
        header
            .magic
            .store(protocol::MAGIC, std::sync::atomic::Ordering::Release);
        header
            .version
            .store(protocol::VERSION, std::sync::atomic::Ordering::Release);
        Ok(section)
    }

    /// The start of the mapping.
    pub(crate) fn view(&self) -> *mut u8 {
        self.view.Value.cast()
    }

    pub(crate) fn header(&self) -> &Header {
        // SAFETY: the view is the whole section, mapped for as long as
        // `self` lives.
        unsafe { Header::at(self.view()) }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: the view and the handle are this section's own, and
        // nothing borrows them past `self`.
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.handle);
        }
    }
}
