//! What a machine has of the libraries ONNX Runtime's CUDA and TensorRT
//! providers open at run time, asked of the loader and of the libraries
//! themselves before any session is — see [`CudaOrtDetector::runtime`].
//!
//! [`CudaOrtDetector::runtime`]: super::CudaOrtDetector::runtime

use std::fmt;

use libloading::Library;

/// What this machine can run a [`CudaOrtDetector`](super::CudaOrtDetector)
/// on, asked of the loader without a model or a GPU's time — see
/// [`CudaOrtDetector::runtime`](super::CudaOrtDetector::runtime).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaRuntime {
    /// TensorRT, with CUDA for what it cannot run.
    TensorRt,
    /// CUDA alone: TensorRT is missing or too old.
    CudaOnly {
        /// What is wrong with TensorRT.
        tensorrt: RuntimeShortfall,
    },
    /// Nothing: the driver, the CUDA runtime or cuDNN is missing or too
    /// old, and a detector cannot be made.
    Unavailable {
        /// What is wrong with them.
        cuda: RuntimeShortfall,
    },
}

/// Why a part of the runtime a detector needs cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeShortfall {
    /// The loader could not open these, by the names it was asked for —
    /// `LD_LIBRARY_PATH` and the system's directories on Linux, `PATH` and
    /// the executable's directory on Windows.
    Missing {
        /// The libraries not found.
        libraries: Vec<&'static str>,
    },
    /// A library opened and is older than the ONNX Runtime build this
    /// crate takes was made against.
    Outdated {
        /// The library, by the name the loader opened.
        library: &'static str,
        /// The version it says it is.
        found: LibraryVersion,
        /// The oldest version taken.
        needed: LibraryVersion,
    },
}

impl fmt::Display for RuntimeShortfall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { libraries } => {
                write!(f, "not found where the loader looks: {libraries:?}")
            }
            Self::Outdated {
                library,
                found,
                needed,
            } => write!(f, "{library} is {found}, and {needed} or newer is needed"),
        }
    }
}

/// A library's version, as it reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LibraryVersion {
    /// The major version.
    pub major: u32,
    /// The minor version.
    pub minor: u32,
    /// The patch version; 0 where the library reports none.
    pub patch: u32,
}

impl LibraryVersion {
    const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl fmt::Display for LibraryVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// How a library reports its version.
#[derive(Debug, Clone, Copy)]
enum Report {
    /// `int f(int *version)`, answering 0 and `1000 × major + 10 × minor` —
    /// `cuDriverGetVersion`, `cudaRuntimeGetVersion`.
    Cuda,
    /// `size_t f(void)`, answering `10000 × major + 100 × minor + patch` —
    /// `cudnnGetVersion` from cuDNN 9 on.
    Cudnn,
    /// `int32_t f(void)`, answering `10000 × major + 100 × minor + patch` —
    /// `getInferLibVersion` from TensorRT 10 on.
    TensorRt,
}

impl Report {
    fn decode(self, raw: u64) -> LibraryVersion {
        let raw = u32::try_from(raw).unwrap_or(u32::MAX);
        match self {
            Self::Cuda => LibraryVersion::new(raw / 1000, raw % 1000 / 10, 0),
            Self::Cudnn | Self::TensorRt => {
                LibraryVersion::new(raw / 10000, raw % 10000 / 100, raw % 100)
            }
        }
    }

    /// The version `library` reports through `symbol`, or `None` where it
    /// has no such symbol or refuses — a library this cannot ask is left
    /// to the session, which opens it anyway.
    fn ask(self, library: &Library, symbol: &[u8]) -> Option<LibraryVersion> {
        // SAFETY: each symbol is called with the C signature its library
        // documents for it, the one `Report`'s variants name: the CUDA ones
        // write one int through a pointer to a live local and return a
        // status, the others take nothing and return the number. None of
        // them needs initialisation first — `cuDriverGetVersion` is
        // documented as callable before `cuInit`. `library` outlives the
        // call.
        let raw = unsafe {
            match self {
                Self::Cuda => {
                    let get = library
                        .get::<unsafe extern "C" fn(*mut i32) -> i32>(symbol)
                        .ok()?;
                    let mut version = 0;
                    if get(&mut version) != 0 {
                        return None;
                    }
                    u64::try_from(version).ok()?
                }
                Self::Cudnn => {
                    let get = library
                        .get::<unsafe extern "C" fn() -> usize>(symbol)
                        .ok()?;
                    get() as u64
                }
                Self::TensorRt => {
                    let get = library.get::<unsafe extern "C" fn() -> i32>(symbol).ok()?;
                    u64::try_from(get()).ok()?
                }
            }
        };
        Some(self.decode(raw))
    }
}

/// A library a provider opens, and the oldest version of it taken where it
/// can be asked.
#[derive(Debug, Clone, Copy)]
pub(super) struct Needed {
    library: &'static str,
    version: Option<(&'static [u8], Report, LibraryVersion)>,
}

impl Needed {
    const fn present(library: &'static str) -> Self {
        Self {
            library,
            version: None,
        }
    }

    const fn at_least(
        library: &'static str,
        symbol: &'static [u8],
        report: Report,
        version: LibraryVersion,
    ) -> Self {
        Self {
            library,
            version: Some((symbol, report, version)),
        }
    }
}

// The oldest versions taken are the ones ONNX Runtime 1.28's CUDA build was
// made against, read off pyke's ort-artifacts build script: CUDA 13.2,
// cuDNN 9.23.2 and TensorRT 10.15.1. Older ones may work, and nothing here
// has tried them — TensorRT's API is C++ interfaces, where a method the
// provider was compiled to call and an older library lacks is a call
// through the wrong slot rather than a missing symbol, so an older one is
// refused rather than tried. The driver is the exception: CUDA's
// minor-version compatibility runs any 13.x runtime on a driver for 13.0.
// Raise these with the `ort` this crate takes.

const CUDA_13_0: LibraryVersion = LibraryVersion::new(13, 0, 0);
const CUDA_13_2: LibraryVersion = LibraryVersion::new(13, 2, 0);
const CUDNN_9_23_2: LibraryVersion = LibraryVersion::new(9, 23, 2);
const TENSORRT_10_15_1: LibraryVersion = LibraryVersion::new(10, 15, 1);

// The library names are what the providers ask the loader for. The Linux
// ones are read off the providers' dynamic sections; the Windows ones are
// the DLL names those releases ship, not read off a Windows build, so a
// mismatch is a library said missing where it is present — which the
// session then contradicts. Only these need finding: cuDNN and TensorRT open
// their own parts from beside themselves.

/// The driver, CUDA 13's runtime, cuBLAS and cuRAND, and cuDNN 9: what the
/// CUDA provider, and so every detector, needs.
#[cfg(not(windows))]
pub(super) const CUDA: &[Needed] = &[
    Needed::at_least(
        "libcuda.so.1",
        b"cuDriverGetVersion\0",
        Report::Cuda,
        CUDA_13_0,
    ),
    Needed::at_least(
        "libcudart.so.13",
        b"cudaRuntimeGetVersion\0",
        Report::Cuda,
        CUDA_13_2,
    ),
    Needed::present("libcublas.so.13"),
    Needed::present("libcublasLt.so.13"),
    Needed::present("libcurand.so.10"),
    Needed::at_least(
        "libcudnn.so.9",
        b"cudnnGetVersion\0",
        Report::Cudnn,
        CUDNN_9_23_2,
    ),
];
#[cfg(windows)]
pub(super) const CUDA: &[Needed] = &[
    Needed::at_least(
        "nvcuda.dll",
        b"cuDriverGetVersion\0",
        Report::Cuda,
        CUDA_13_0,
    ),
    Needed::at_least(
        "cudart64_13.dll",
        b"cudaRuntimeGetVersion\0",
        Report::Cuda,
        CUDA_13_2,
    ),
    Needed::present("cublas64_13.dll"),
    Needed::present("cublasLt64_13.dll"),
    Needed::present("curand64_10.dll"),
    Needed::at_least(
        "cudnn64_9.dll",
        b"cudnnGetVersion\0",
        Report::Cudnn,
        CUDNN_9_23_2,
    ),
];

/// TensorRT 10 and its ONNX parser: what the TensorRT provider needs
/// beside the CUDA ones. The parser ships with TensorRT and is not asked
/// its own version.
#[cfg(not(windows))]
pub(super) const TENSORRT: &[Needed] = &[
    Needed::at_least(
        "libnvinfer.so.10",
        b"getInferLibVersion\0",
        Report::TensorRt,
        TENSORRT_10_15_1,
    ),
    Needed::present("libnvonnxparser.so.10"),
];
#[cfg(windows)]
pub(super) const TENSORRT: &[Needed] = &[
    Needed::at_least(
        "nvinfer_10.dll",
        b"getInferLibVersion\0",
        Report::TensorRt,
        TENSORRT_10_15_1,
    ),
    Needed::present("nvonnxparser_10.dll"),
];

/// What is wrong with `needed` on this machine, if anything: every library
/// the loader cannot open, else the first that is too old.
///
/// Asked before a session is, because ONNX Runtime's own answer to a
/// missing library names its provider bridge rather than the library, and
/// to one too old is whatever the first call into it does.
pub(super) fn shortfall(needed: &[Needed]) -> Option<RuntimeShortfall> {
    let mut opened = Vec::with_capacity(needed.len());
    let mut missing = Vec::new();
    for need in needed {
        // SAFETY: loading runs a library's initialisers. These are NVIDIA's
        // runtime libraries, which ONNX Runtime loads the same way moments
        // later; each is let go of again when this returns.
        match unsafe { Library::new(need.library) } {
            Ok(library) => opened.push((need, library)),
            Err(_) => missing.push(need.library),
        }
    }
    if !missing.is_empty() {
        return Some(RuntimeShortfall::Missing { libraries: missing });
    }
    opened.iter().find_map(|(need, library)| {
        let (symbol, report, needed) = need.version?;
        let found = report.ask(library, symbol)?;
        (found < needed).then_some(RuntimeShortfall::Outdated {
            library: need.library,
            found,
            needed,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_library_s_report_is_read_as_it_writes_it() {
        assert_eq!(Report::Cuda.decode(13020), LibraryVersion::new(13, 2, 0));
        assert_eq!(Report::Cuda.decode(12080), LibraryVersion::new(12, 8, 0));
        assert_eq!(Report::Cudnn.decode(92700), LibraryVersion::new(9, 27, 0));
        assert_eq!(Report::Cudnn.decode(92302), CUDNN_9_23_2);
        assert_eq!(
            Report::TensorRt.decode(101601),
            LibraryVersion::new(10, 16, 1)
        );
        assert_eq!(
            Report::TensorRt.decode(100300),
            LibraryVersion::new(10, 3, 0)
        );
    }

    #[test]
    fn versions_order_by_major_then_minor_then_patch() {
        assert!(LibraryVersion::new(10, 3, 0) < TENSORRT_10_15_1);
        assert!(LibraryVersion::new(10, 15, 0) < TENSORRT_10_15_1);
        assert!(LibraryVersion::new(10, 16, 0) > TENSORRT_10_15_1);
        assert!(LibraryVersion::new(11, 0, 0) > TENSORRT_10_15_1);
    }

    #[test]
    fn a_library_the_loader_cannot_open_is_named() {
        assert_eq!(
            shortfall(&[Needed::present("libmedia-pp-no-such-library.so.1")]),
            Some(RuntimeShortfall::Missing {
                libraries: vec!["libmedia-pp-no-such-library.so.1"]
            })
        );
        assert_eq!(shortfall(&[]), None);
    }

    /// Asks this machine's CUDA runtime for a version past any there is,
    /// and for one there is; skipped, saying so, where it has none.
    #[test]
    fn a_library_older_than_needed_is_named_with_both_versions() {
        let library = CUDA[1].library;
        let ask = |needed| {
            shortfall(&[Needed::at_least(
                library,
                b"cudaRuntimeGetVersion\0",
                Report::Cuda,
                needed,
            )])
        };
        if let Some(RuntimeShortfall::Missing { .. }) = ask(CUDA_13_0) {
            eprintln!("skipping: no {library} where the loader looks");
            return;
        }
        assert_eq!(ask(LibraryVersion::new(13, 0, 0)), None);
        match ask(LibraryVersion::new(99, 0, 0)) {
            Some(RuntimeShortfall::Outdated {
                library: named,
                found,
                needed,
            }) => {
                assert_eq!(named, library);
                assert_eq!(found.major, 13);
                assert_eq!(needed, LibraryVersion::new(99, 0, 0));
            }
            other => panic!("expected {library} to be too old, got {other:?}"),
        }
    }
}
