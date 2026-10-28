//! Keeping glibc's heap near what a process uses.
//!
//! A server built on this crate runs a thread or more for every pipeline,
//! and glibc's malloc gives each thread an arena of its own — up to eight
//! for each core. As large blocks are freed glibc raises the size past
//! which an arena gives the free memory at its end back to the system: to
//! twice its mmap threshold, which itself rises to 32 MB. An arena then
//! keeps 10 to 14 MB it no longer uses. relay-pp, with fifty cameras, 160
//! threads and 96 arenas, came to 1.8 GB of heap holding under a tenth of
//! that in use; deleting every channel left 850 MB.
//!
//! [`keep_small`] is for such a process, and only it: it changes how the
//! whole process allocates, which is the program's to decide — as choosing
//! a `#[global_allocator]` is — so nothing in this crate calls it.

/// Holds glibc's heap near what the process uses, on Linux with glibc;
/// elsewhere it does nothing.
///
/// Both of glibc's thresholds are held, since setting either stops both
/// rising: an arena's free end goes back to the system past 1 MB, and a
/// block is mapped on its own only past 32 MB, where glibc would put the
/// threshold anyway. And a thread of its own, `heap-trim`, gives back the
/// free pages in the middle of every arena every ten seconds, which the
/// thresholds cannot reach.
///
/// relay-pp's three-hour soak — fifty cameras, a hundred and more players,
/// HEVC converted on NVDEC — held 546 MB on average and stayed level, where
/// it rose to 1.8 GB without; its processor rose 2%, and in its benchmarks
/// between none and 3%.
///
/// Holding the mmap threshold low instead (`M_MMAP_THRESHOLD` at 128 KiB)
/// kept the heap as small for 3% but crashed inside NVIDIA's NVDEC library,
/// which reads a 2.3 MB block it has freed and faults once that block is
/// unmapped (`docs/issues/nvdec-use-after-free`). Here such a block stays
/// in an arena. Freed, its pages may still be trimmed; in the arena of a
/// thread other than the main one that leaves them mapped, reading as
/// zeros, unless `vm.overcommit_memory` is 2. The main thread's arena gives
/// its end back by lowering the program break, which unmaps it, so open
/// decoders off the main thread.
///
/// Call it first in `main`, before any thread or decoder: a block mapped on
/// its own before it is unmapped when freed. A second call does nothing.
///
/// # Errors
///
/// Where the trimming thread cannot be started; the thresholds are held
/// all the same, and a later call tries the thread again.
pub fn keep_small() -> std::io::Result<()> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    return glibc::keep_small();
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    Ok(())
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod glibc {
    use std::ffi::c_int;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    // malloc.h's parameters, the same in every glibc.
    const M_TRIM_THRESHOLD: c_int = -1;
    const M_MMAP_THRESHOLD: c_int = -3;

    unsafe extern "C" {
        fn mallopt(param: c_int, value: c_int) -> c_int;
        fn malloc_trim(pad: usize) -> c_int;
    }

    static TRIMMING: AtomicBool = AtomicBool::new(false);

    pub(super) fn keep_small() -> std::io::Result<()> {
        if TRIMMING.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        // SAFETY: mallopt sets one of malloc's parameters under malloc's own
        // lock, and takes and returns plain integers; both parameters are
        // ones glibc's mallopt knows.
        unsafe {
            mallopt(M_MMAP_THRESHOLD, 32 << 20);
            mallopt(M_TRIM_THRESHOLD, 1 << 20);
        }
        std::thread::Builder::new()
            .name("heap-trim".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_secs(10));
                    // SAFETY: malloc_trim takes no pointer and works under
                    // each arena's own lock.
                    unsafe { malloc_trim(0) };
                }
            })
            .map(drop)
            .inspect_err(|_| TRIMMING.store(false, Ordering::Release))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_second_call_starts_no_second_trimming_thread() {
        super::keep_small().expect("the trimming thread");
        super::keep_small().expect("nothing to do");
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            let trimming = std::fs::read_dir("/proc/self/task")
                .expect("this process's threads")
                .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
                .filter(|name| name.trim_end() == "heap-trim")
                .count();
            assert_eq!(trimming, 1);
        }
    }
}
