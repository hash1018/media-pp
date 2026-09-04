//! COM apartment management, shared by every Windows backend that needs one.

use windows::Win32::{
    Foundation::RPC_E_CHANGED_MODE,
    System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize},
};

/// Balances one successful `CoInitializeEx` on the current thread.
pub(crate) struct ComApartment {
    uninitialize: bool,
}

impl ComApartment {
    pub(crate) fn new() -> windows::core::Result<Self> {
        // SAFETY: initializes COM for the current thread with no reserved
        // pointer; the successful initialization is balanced in `Drop`.
        let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if result == RPC_E_CHANGED_MODE {
            // The caller already initialized this thread as STA. COM is
            // available and these backends work there; only the apartment
            // model cannot be changed, and this call must not be balanced.
            return Ok(Self {
                uninitialize: false,
            });
        }
        result.ok()?;
        Ok(Self { uninitialize: true })
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.uninitialize {
            // SAFETY: this instance records a successful `CoInitializeEx` on
            // this same thread, so this call balances it exactly once.
            unsafe { CoUninitialize() };
        }
    }
}
