use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crossbeam_queue::SegQueue;

/// A growable pool of reusable `T`s — lets a frame-producing element
/// (see [`crate::elements::SwDecoder`], [`crate::elements::Scaler`]) hand
/// out the *same* buffers over and over instead of allocating a fresh one
/// (and freeing the old one) every single frame. Reusing an
/// already-allocated `ffmpeg_next::frame::Video` also lets ffmpeg's own
/// `avcodec_receive_frame`/`sws_scale` skip *their* internal buffer
/// allocation when the reused frame already matches — not just savings
/// on this crate's side.
///
/// Never blocks and never fails: [`UnboundObjectPool::get`] pops a
/// previously-returned item if one's available, or calls `init` to build
/// a fresh one on the spot otherwise — the pool just grows to whatever
/// depth turns out to be needed (e.g. however many frames a downstream
/// `Queue` lets pile up at once) instead of capping it and having to
/// handle "pool exhausted" as a distinct case.
///
/// Deliberately a private implementation detail owned by whichever
/// element produces the frames (a struct field, initialized once in that
/// element's own constructor) — not something the `Pipeline` holds or
/// passes around. Nothing outside that one element needs to know a pool
/// is involved at all; sharing the *frames themselves* downstream still
/// goes through the ordinary `Arc<UnboundObjectPoolRef<T>>` in
/// [`crate::buffer::MediaBuffer::Video`].
pub struct UnboundObjectPool<T: Send> {
    share: Arc<Share<T>>,
    init: Box<dyn Fn() -> T + Send + Sync>,
}

struct Share<T: Send> {
    pool: SegQueue<Box<T>>,
    release: Box<dyn Fn(&mut T) + Send + Sync>,
}

impl<T: Send + Sync> UnboundObjectPool<T> {
    /// Pre-fills with `size` items built via `init` (`0` is fine — the
    /// pool still grows on demand, it just starts out empty and pays for
    /// the first `size`-ish `get()` calls up front instead of amortized
    /// over the stream). `release` runs on an item right before it goes
    /// back into the pool (e.g. to reset state) — a no-op closure is
    /// fine if there's nothing to reset, which is the common case for a
    /// video frame: the next `consume` overwrites every pixel (and every
    /// piece of metadata it cares about, like `pts`) before anyone
    /// downstream sees it again.
    pub fn new(
        size: usize,
        init: impl Fn() -> T + Send + Sync + 'static,
        release: impl Fn(&mut T) + Send + Sync + 'static,
    ) -> UnboundObjectPool<T> {
        let pool = SegQueue::new();
        for _ in 0..size {
            pool.push(Box::new(init()));
        }

        UnboundObjectPool {
            share: Arc::new(Share {
                pool,
                release: Box::new(release),
            }),
            init: Box::new(init),
        }
    }

    /// Never blocks, never fails — see the type docs.
    pub fn get(&self) -> UnboundObjectPoolRef<T> {
        let item = self
            .share
            .pool
            .pop()
            .unwrap_or_else(|| Box::new((self.init)()));
        UnboundObjectPoolRef {
            share: self.share.clone(),
            item: Some(item),
        }
    }

    /// How many items are currently sitting in the pool, unused. Mainly
    /// for tests/diagnostics — nothing in this crate depends on this
    /// number for correctness.
    pub fn size(&self) -> usize {
        self.share.pool.len()
    }
}

/// One item borrowed from an [`UnboundObjectPool`] — `Deref`/`DerefMut`
/// to `T` for normal use, and returns itself to the pool (after running
/// `release` on it) when dropped.
///
/// Meant to be wrapped in an `Arc` wherever it needs to be shared/cloned
/// downstream (see [`crate::buffer::MediaBuffer::Video`]) — cloning the
/// `Arc` is what lets e.g. [`crate::elements::Tee`] fan the same frame
/// out to multiple branches cheaply, and the item only actually goes
/// back to the pool once every one of those clones has been dropped.
/// This type itself is deliberately *not* `Clone`: only one thing can
/// hold the actual boxed value at a time, or "return it once the last
/// reference drops" wouldn't mean anything.
pub struct UnboundObjectPoolRef<T: Send> {
    share: Arc<Share<T>>,
    item: Option<Box<T>>,
}

impl<T: Send> Deref for UnboundObjectPoolRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item.as_deref().expect("item only taken in Drop")
    }
}

impl<T: Send> DerefMut for UnboundObjectPoolRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_deref_mut().expect("item only taken in Drop")
    }
}

impl<T: Send> Drop for UnboundObjectPoolRef<T> {
    fn drop(&mut self) {
        let mut item = self.item.take().expect("item only taken once, here");
        (self.share.release)(&mut item);
        self.share.pool.push(item);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[test]
    fn returned_item_is_reused_by_the_next_get() {
        let pool = UnboundObjectPool::new(0, || 0i32, |_| {});
        assert_eq!(pool.size(), 0);

        let item = pool.get();
        assert_eq!(pool.size(), 0, "checked out, not sitting in the pool");

        drop(item);
        assert_eq!(pool.size(), 1, "returned automatically on drop");

        let _item2 = pool.get();
        assert_eq!(pool.size(), 0, "reused, not left behind");
    }

    #[test]
    fn get_never_fails_even_when_empty() {
        let pool = UnboundObjectPool::new(0, || 5i32, |_| {});
        // Nothing's ever been returned, so both of these fall back to
        // `init` — proves `get` doesn't block/panic/return `Option` when
        // the pool has nothing to give out.
        assert_eq!(*pool.get(), 5);
        assert_eq!(*pool.get(), 5);
    }

    #[test]
    fn release_runs_before_the_item_goes_back_into_the_pool() {
        let release_calls = Arc::new(AtomicUsize::new(0));
        let counted = release_calls.clone();
        let pool = UnboundObjectPool::new(
            1,
            || 0i32,
            move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
            },
        );

        drop(pool.get());
        assert_eq!(release_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_the_last_arc_clone_returns_the_item() {
        // Mirrors how `MediaBuffer::Video` actually uses this: wrapped in
        // an `Arc` so it can be cheaply cloned downstream (e.g. by
        // `Tee`), only going back to the pool once every clone is gone —
        // not on the first `Arc` that happens to drop.
        let pool = UnboundObjectPool::new(0, || 0i32, |_| {});
        let shared = Arc::new(pool.get());
        let clone = shared.clone();

        drop(shared);
        assert_eq!(pool.size(), 0, "one clone still alive — not returned yet");

        drop(clone);
        assert_eq!(pool.size(), 1, "last clone dropped — now it's returned");
    }
}
