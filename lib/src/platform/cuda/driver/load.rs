//! The CUDA driver, opened when it is first called rather than when the
//! program starts.
//!
//! Linked by name, `libcuda.so.1` — or `nvcuda.dll` — becomes something the
//! program cannot start without, and only a machine with NVIDIA's driver has
//! it. A program built with `cuda` that looks for a GPU and uses Vulkan
//! where there is no NVIDIA one would never get to look. So the driver is
//! opened on first use instead, the way FFmpeg opens it for its own CUDA
//! code, and each entry point answers a `CUresult` whether it is there or
//! not: an error where the driver or the symbol is missing, which every
//! call site already checks for.

use std::sync::OnceLock;

use super::CUresult;

/// `CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`: the driver is there and this
/// entry point is not — one older than the call.
pub(super) const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CUresult = 302;
/// `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`: there is no driver to call.
pub(super) const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CUresult = 303;

#[cfg(windows)]
const LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libcuda.so.1";

/// The driver, opened once and kept for the life of the process: its
/// entry points are handed out as plain function pointers, which must not
/// outlive it.
pub(super) fn library() -> Option<&'static libloading::Library> {
    static LIBRARY_HANDLE: OnceLock<Option<libloading::Library>> = OnceLock::new();
    LIBRARY_HANDLE
        // SAFETY: loading the NVIDIA driver runs its initialisers, which are
        // the driver's own and what linking against it would have run at
        // startup anyway.
        .get_or_init(|| unsafe { libloading::Library::new(LIBRARY) }.ok())
        .as_ref()
}

/// Declares CUDA driver entry points as functions of the same names and
/// signatures, each resolved from [`library`] the first time any of them is
/// called — in place of an `extern "C"` block, whose call sites do not
/// change.
macro_rules! cuda_driver {
    ($(
        $(#[$meta:meta])*
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> CUresult;
    )*) => {
        /// This block's entry points, as the driver answered for each.
        #[allow(non_snake_case)]
        struct Entries {
            $($name: Option<unsafe extern "C" fn($($ty),*) -> CUresult>,)*
        }

        fn entries() -> Option<&'static Entries> {
            static ENTRIES: std::sync::OnceLock<Option<Entries>> = std::sync::OnceLock::new();
            ENTRIES
                .get_or_init(|| {
                    let library = $crate::platform::cuda::driver::load::library()?;
                    Some(Entries {
                        $(
                            // SAFETY: the type is the entry point's own, from
                            // `cuda.h`, and the library it points into is
                            // never unloaded.
                            $name: unsafe {
                                library
                                    .get::<unsafe extern "C" fn($($ty),*) -> CUresult>(
                                        concat!(stringify!($name), "\0").as_bytes(),
                                    )
                                    .ok()
                                    .map(|symbol| *symbol)
                            },
                        )*
                    })
                })
                .as_ref()
        }

        $(
            $(#[$meta])*
            #[allow(non_snake_case, clippy::too_many_arguments)]
            unsafe fn $name($($arg: $ty),*) -> CUresult {
                let Some(entries) = entries() else {
                    return $crate::platform::cuda::driver::load::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
                };
                match entries.$name {
                    // SAFETY: the caller's, as for the C function itself.
                    Some(entry) => unsafe { entry($($arg),*) },
                    None => {
                        $crate::platform::cuda::driver::load::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
                    }
                }
            }
        )*
    };
}

pub(super) use cuda_driver;
