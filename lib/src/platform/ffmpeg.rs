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
        // SAFETY: `self` owns a live reference, so taking another is valid;
        // `av_buffer_ref` returns null only on allocation failure, which `from_raw`
        // turns into `None` rather than an owner of nothing.
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

// SAFETY: as above — `&self` only ever reads the pointer or takes another
// atomic reference through it, so sharing one owner across threads adds no
// unsynchronized access of its own.
unsafe impl Sync for AvBufferRef {}

impl Drop for AvBufferRef {
    fn drop(&mut self) {
        let mut ptr = self.as_ptr();
        // SAFETY: this owner's reference has not been given away — `into_raw`
        // forgets the owner rather than leaving one behind — so this drops exactly
        // the one reference it holds. `av_buffer_unref` takes the pointer by
        // address, hence the local.
        unsafe { ffi::av_buffer_unref(&mut ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_ref_clone_and_drop_balance_references() {
        // SAFETY: `av_buffer_alloc` hands back one owned reference, which `from_raw`
        // takes; every count read below is of a buffer still owned by a live
        // `AvBufferRef`.
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
        // SAFETY: as above for the allocation. After `into_raw` this test owns the
        // reference itself, which is why it unrefs by hand — that is the transfer
        // being asserted.
        unsafe {
            let owner = AvBufferRef::from_raw(ffi::av_buffer_alloc(1)).unwrap();
            let mut raw = owner.into_raw();
            assert_eq!(ffi::av_buffer_get_ref_count(raw), 1);

            ffi::av_buffer_unref(&mut raw);
            assert!(raw.is_null());
        }
    }
}
