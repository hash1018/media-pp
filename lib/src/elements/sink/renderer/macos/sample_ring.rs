//! The bytes between `CoreAudioRenderer`, which writes them on the pipeline's
//! thread, and its output unit's render callback, which reads them on Core
//! Audio's real-time IO thread.
//!
//! That thread must never wait: not on a lock, not on an allocation. So this
//! is a fixed ring with one writer and one reader, each moving only its own
//! cursor, and nothing allocated after [`SampleRing::new`].

use std::{
    cell::UnsafeCell,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

/// A ring of `capacity` bytes, written by exactly one thread and read by
/// exactly one other.
///
/// The cursors count bytes since the ring was made and are never wrapped
/// themselves, so how much it holds is their difference; a cursor's place
/// in the storage is that count modulo `capacity`. What is between the read
/// and write cursors belongs to the reader, the rest to the writer.
pub(super) struct SampleRing {
    storage: Box<[UnsafeCell<u8>]>,
    read: AtomicUsize,
    write: AtomicUsize,
}

// SAFETY: the one writer only writes storage outside [read, write), and the
// one reader only reads inside it; each publishes what it did with a Release
// store of its own cursor that the other Acquires before touching storage.
// So no byte is ever written by one thread while the other reads it.
unsafe impl Sync for SampleRing {}

impl SampleRing {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            storage: (0..capacity).map(|_| UnsafeCell::new(0)).collect(),
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
        }
    }

    pub(super) fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// How many bytes it holds. Exact from either side about its own cursor;
    /// the other may have moved by the time this returns.
    pub(super) fn len(&self) -> usize {
        // Read cursor first: it never passes the write cursor, so one loaded
        // after it is never behind it.
        let read = self.read.load(Ordering::Acquire);
        let write = self.write.load(Ordering::Acquire);
        write.wrapping_sub(read)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writer: copies as much of `data` as there is room for, and returns
    /// how much that was.
    pub(super) fn write(&self, data: &[u8]) -> usize {
        let read = self.read.load(Ordering::Acquire);
        let write = self.write.load(Ordering::Relaxed);
        let free = self.capacity() - write.wrapping_sub(read);
        let count = free.min(data.len());
        self.copy_in(write, &data[..count]);
        self.write
            .store(write.wrapping_add(count), Ordering::Release);
        count
    }

    /// Reader: fills as much of `out` as it holds, and returns how much
    /// that was.
    pub(super) fn read_into(&self, out: &mut [u8]) -> usize {
        let write = self.write.load(Ordering::Acquire);
        let read = self.read.load(Ordering::Relaxed);
        let count = write.wrapping_sub(read).min(out.len());
        self.copy_out(read, &mut out[..count]);
        self.read.store(read.wrapping_add(count), Ordering::Release);
        count
    }

    /// Writer, and only while nothing reads: lets go of everything it holds.
    pub(super) fn clear(&self) {
        let write = self.write.load(Ordering::Relaxed);
        self.read.store(write, Ordering::Release);
    }

    fn base(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.storage.as_ptr())
    }

    /// `data` into the storage from cursor `at` on, wrapping at the end.
    fn copy_in(&self, at: usize, data: &[u8]) {
        let start = at % self.capacity().max(1);
        let first = data.len().min(self.capacity() - start);
        // SAFETY: both ranges lie inside the storage — `first` stops at its
        // end and the rest starts at zero, no longer than the room counted —
        // and the writer owns them (see the `Sync` impl).
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.base().add(start), first);
            ptr::copy_nonoverlapping(data.as_ptr().add(first), self.base(), data.len() - first);
        }
    }

    /// The storage from cursor `at` on into `out`, wrapping at the end.
    fn copy_out(&self, at: usize, out: &mut [u8]) {
        let start = at % self.capacity().max(1);
        let first = out.len().min(self.capacity() - start);
        // SAFETY: as in `copy_in`, with the reader owning what it copies.
        unsafe {
            ptr::copy_nonoverlapping(self.base().add(start), out.as_mut_ptr(), first);
            ptr::copy_nonoverlapping(self.base(), out.as_mut_ptr().add(first), out.len() - first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What goes in comes out in order, across the end of the storage, and
    /// the writer is told how much fitted.
    #[test]
    fn bytes_come_out_in_order_across_the_wrap() {
        let ring = SampleRing::new(8);
        assert_eq!(ring.write(&[1, 2, 3, 4, 5, 6]), 6);
        let mut out = [0; 4];
        assert_eq!(ring.read_into(&mut out), 4);
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(ring.write(&[7, 8, 9, 10, 11, 12, 13]), 6, "room for six");
        assert_eq!(ring.len(), 8);
        let mut out = [0; 10];
        assert_eq!(ring.read_into(&mut out), 8);
        assert_eq!(&out[..8], &[5, 6, 7, 8, 9, 10, 11, 12]);
        assert!(ring.is_empty());
    }

    /// Clearing leaves it empty and usable, from wherever its cursors were.
    #[test]
    fn a_cleared_ring_takes_new_bytes() {
        let ring = SampleRing::new(4);
        ring.write(&[1, 2, 3]);
        ring.read_into(&mut [0; 2]);
        ring.write(&[4, 5]);
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.write(&[6, 7, 8, 9]), 4);
        let mut out = [0; 4];
        ring.read_into(&mut out);
        assert_eq!(out, [6, 7, 8, 9]);
    }

    /// One thread writing and another reading see every byte once and in
    /// order.
    #[test]
    fn a_writer_and_a_reader_on_two_threads_agree() {
        let ring = std::sync::Arc::new(SampleRing::new(64));
        let total = 100_000usize;
        let reader = {
            let ring = ring.clone();
            std::thread::spawn(move || {
                let mut next = 0usize;
                let mut out = [0u8; 48];
                while next < total {
                    let count = ring.read_into(&mut out);
                    for &byte in &out[..count] {
                        assert_eq!(byte, (next % 251) as u8);
                        next += 1;
                    }
                }
            })
        };
        let data: Vec<u8> = (0..total).map(|index| (index % 251) as u8).collect();
        let mut written = 0;
        while written < total {
            written += ring.write(&data[written..(written + 40).min(total)]);
        }
        reader.join().expect("the reader saw every byte in order");
    }
}
