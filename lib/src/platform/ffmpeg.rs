//! Small RAII owners for FFmpeg objects not wrapped by `ffmpeg-next`.

use std::ptr::NonNull;

use ffmpeg_next::ffi;

/// Owns one non-null FFmpeg `AVBufferRef` reference.
///
/// `av_buffer_ref` can fail, so cloning is deliberately exposed as
/// [`try_clone`](Self::try_clone) instead of implementing infallible `Clone`.
pub(crate) struct AvBufferRef(NonNull<ffi::AVBufferRef>);

impl AvBufferRef {
    /// Takes ownership of one raw reference, returning `None` for null.
    ///
    /// # Safety
    /// A non-null `ptr` must be an owned `AVBufferRef` reference that the
    /// caller will no longer unref or transfer elsewhere.
    pub(crate) unsafe fn from_raw(ptr: *mut ffi::AVBufferRef) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::AVBufferRef {
        self.0.as_ptr()
    }

    pub(crate) fn try_clone(&self) -> Option<Self> {
        unsafe { Self::from_raw(ffi::av_buffer_ref(self.as_ptr())) }
    }

    /// Transfers this owned reference to an FFmpeg object.
    pub(crate) fn into_raw(self) -> *mut ffi::AVBufferRef {
        let ptr = self.as_ptr();
        std::mem::forget(self);
        ptr
    }
}

// SAFETY: AVBuffer's reference count is atomic and FFmpeg hardware contexts
// are designed to be retained/released across the worker threads that use
// them. This owner only refs/unrefs the buffer; access to mutable payloads is
// still synchronized by each element or the backend API.
unsafe impl Send for AvBufferRef {}
unsafe impl Sync for AvBufferRef {}

impl Drop for AvBufferRef {
    fn drop(&mut self) {
        let mut ptr = self.as_ptr();
        unsafe { ffi::av_buffer_unref(&mut ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_ref_clone_and_drop_balance_references() {
        unsafe {
            let owner = AvBufferRef::from_raw(ffi::av_buffer_alloc(1)).unwrap();
            assert_eq!(ffi::av_buffer_get_ref_count(owner.as_ptr()), 1);

            let clone = owner.try_clone().unwrap();
            assert_eq!(ffi::av_buffer_get_ref_count(owner.as_ptr()), 2);

            drop(clone);
            assert_eq!(ffi::av_buffer_get_ref_count(owner.as_ptr()), 1);
        }
    }

    #[test]
    fn into_raw_transfers_the_owned_reference() {
        unsafe {
            let owner = AvBufferRef::from_raw(ffi::av_buffer_alloc(1)).unwrap();
            let mut raw = owner.into_raw();
            assert_eq!(ffi::av_buffer_get_ref_count(raw), 1);

            ffi::av_buffer_unref(&mut raw);
            assert!(raw.is_null());
        }
    }
}
