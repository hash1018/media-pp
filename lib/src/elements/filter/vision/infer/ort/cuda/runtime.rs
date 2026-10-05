//! The libraries ONNX Runtime's CUDA and TensorRT providers need, and what
//! this machine has of them — see [`CudaOrtDetector::runtime`].
//!
//! CUDA's runtime, cuBLAS, cuRAND and cuDNN — and with `ort-tensorrt`,
//! TensorRT and its ONNX parser — are linked into the executable: a program
//! built with the feature does not start without them, as it does not
//! without FFmpeg, and the providers find them already open under the names
//! they ask for. What linking leaves to run time is their versions, which
//! only they can say, and the driver, which is the system's and is opened
//! here rather than linked, so that a build machine needs no driver.
//!
//! On Linux build.rs links each by its versioned file name, which is how
//! NVIDIA's pip wheels ship them. On Windows each block below names its DLL
//! itself (`raw-dylib`): the wheels there carry the DLLs and no import
//! libraries, and these few calls are all this crate makes of them, so
//! nothing has to be there to build — only, as on Linux, to run.
//!
//! The link keeps a library only while something in the program calls it —
//! rustc passes `--as-needed` — so each is called here, for its version.
//!
//! [`CudaOrtDetector::runtime`]: super::CudaOrtDetector::runtime

use std::fmt;

use libloading::Library;

#[cfg_attr(windows, link(name = "cudart64_13", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `cudaError_t cudaRuntimeGetVersion(int *)`, in the CUDA runtime.
    fn cudaRuntimeGetVersion(version: *mut i32) -> i32;
}

#[cfg_attr(windows, link(name = "cublas64_13", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `cublasStatus_t cublasGetProperty(libraryPropertyType, int *)`, in
    /// cuBLAS: the major, minor and patch level for 0, 1 and 2.
    fn cublasGetProperty(property: i32, value: *mut i32) -> i32;
}

#[cfg_attr(windows, link(name = "cublasLt64_13", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `size_t cublasLtGetVersion(void)`, in cuBLASLt.
    fn cublasLtGetVersion() -> usize;
}

#[cfg_attr(windows, link(name = "curand64_10", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `curandStatus_t curandGetVersion(int *)`, in cuRAND.
    fn curandGetVersion(version: *mut i32) -> i32;
}

#[cfg_attr(windows, link(name = "cudnn64_9", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `size_t cudnnGetVersion(void)`, in cuDNN.
    fn cudnnGetVersion() -> usize;
}

#[cfg(feature = "ort-tensorrt")]
#[cfg_attr(windows, link(name = "nvinfer_10", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `int32_t getInferLibVersion(void)`, in TensorRT.
    fn getInferLibVersion() -> i32;
}

#[cfg(feature = "ort-tensorrt")]
#[cfg_attr(windows, link(name = "nvonnxparser_10", kind = "raw-dylib"))]
unsafe extern "C" {
    /// `int getNvOnnxParserVersion(void)`, in TensorRT's ONNX parser.
    fn getNvOnnxParserVersion() -> i32;
}

/// What this machine can run a [`CudaOrtDetector`](super::CudaOrtDetector)
/// on, without a model or a GPU's time — see
/// [`CudaOrtDetector::runtime`](super::CudaOrtDetector::runtime).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaRuntime {
    /// TensorRT, with CUDA for what it cannot run.
    TensorRt,
    /// CUDA alone: this build has no TensorRT, or TensorRT is too old.
    CudaOnly {
        /// Why not TensorRT.
        tensorrt: RuntimeShortfall,
    },
    /// Nothing: the driver is missing, or it, the CUDA runtime or cuDNN is
    /// too old, and a detector cannot be made.
    Unavailable {
        /// What is wrong with them.
        cuda: RuntimeShortfall,
    },
}

/// Why a part of the runtime a detector needs cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeShortfall {
    /// The loader could not open these, by the names it was asked for: the
    /// GPU's driver, which is not linked.
    Missing {
        /// The libraries not found.
        libraries: Vec<&'static str>,
    },
    /// A library is older than the ONNX Runtime build this crate takes was
    /// made against.
    Outdated {
        /// The library.
        library: &'static str,
        /// The version it says it is.
        found: LibraryVersion,
        /// The oldest version taken.
        needed: LibraryVersion,
    },
    /// Not in this build: TensorRT comes with the `ort-tensorrt` feature.
    NotBuilt,
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
            Self::NotBuilt => f.write_str("not in this build, which is without `ort-tensorrt`"),
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

    /// CUDA's own encoding: `1000 × major + 10 × minor`.
    fn of_cuda(raw: i32) -> Self {
        let raw = u32::try_from(raw).unwrap_or(0);
        Self::new(raw / 1000, raw % 1000 / 10, 0)
    }

    /// cuDNN 9's, cuBLASLt's and TensorRT 10's: `10000 × major + 100 ×
    /// minor + patch`.
    fn of_hundreds(raw: u64) -> Self {
        let raw = u32::try_from(raw).unwrap_or(u32::MAX);
        Self::new(raw / 10000, raw % 10000 / 100, raw % 100)
    }

    /// cuRAND's: `1000 × major + 100 × minor + patch`.
    fn of_curand(raw: i32) -> Self {
        let raw = u32::try_from(raw).unwrap_or(0);
        Self::new(raw / 1000, raw % 1000 / 100, raw % 100)
    }
}

impl fmt::Display for LibraryVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
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
#[cfg(feature = "ort-tensorrt")]
const TENSORRT_10_15_1: LibraryVersion = LibraryVersion::new(10, 15, 1);

/// The GPU's driver library, opened rather than linked.
#[cfg(not(windows))]
const DRIVER: &str = "libcuda.so.1";
#[cfg(windows)]
const DRIVER: &str = "nvcuda.dll";

/// The version of CUDA the installed driver supports, or why there is
/// none.
fn driver() -> Result<LibraryVersion, RuntimeShortfall> {
    let missing = || RuntimeShortfall::Missing {
        libraries: vec![DRIVER],
    };
    // SAFETY: loading runs the driver library's initialisers, as the CUDA
    // runtime does when it first needs the driver; it is let go of again
    // when this returns. `cuDriverGetVersion` is `CUresult (int *)`,
    // documented as callable before `cuInit`, and writes one int through a
    // pointer to a live local while `library` is open.
    unsafe {
        let library = Library::new(DRIVER).map_err(|_| missing())?;
        let get = library
            .get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cuDriverGetVersion\0")
            .map_err(|_| missing())?;
        let mut version = 0;
        if get(&mut version) != 0 {
            return Err(missing());
        }
        Ok(LibraryVersion::of_cuda(version))
    }
}

/// The linked libraries' versions, as each reports its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Linked {
    cudart: LibraryVersion,
    cublas: LibraryVersion,
    cublas_lt: LibraryVersion,
    curand: LibraryVersion,
    cudnn: LibraryVersion,
    #[cfg(feature = "ort-tensorrt")]
    tensorrt: LibraryVersion,
}

impl Linked {
    pub(super) fn ask() -> Self {
        let (mut cudart, mut curand) = (0, 0);
        let mut cublas = [0; 3];
        // SAFETY: each is called with the C signature its library documents,
        // as declared above; the ones taking a pointer are handed live locals
        // and write one int through it. None needs a device, a context or a
        // handle first, and a failure leaves the local at 0, which reads as a
        // version older than any taken.
        let (cublas_lt, cudnn) = unsafe {
            cudaRuntimeGetVersion(&mut cudart);
            for (property, value) in (0..).zip(cublas.iter_mut()) {
                cublasGetProperty(property, value);
            }
            curandGetVersion(&mut curand);
            (cublasLtGetVersion(), cudnnGetVersion())
        };
        let [major, minor, patch] = cublas.map(|n| u32::try_from(n).unwrap_or(0));
        Self {
            cudart: LibraryVersion::of_cuda(cudart),
            cublas: LibraryVersion::new(major, minor, patch),
            cublas_lt: LibraryVersion::of_hundreds(cublas_lt as u64),
            curand: LibraryVersion::of_curand(curand),
            cudnn: LibraryVersion::of_hundreds(cudnn as u64),
            #[cfg(feature = "ort-tensorrt")]
            tensorrt: {
                // SAFETY: as above; both take nothing and return a number.
                // The parser is called so that the link keeps it, and its
                // number is not a version this checks.
                let raw = unsafe {
                    getNvOnnxParserVersion();
                    getInferLibVersion()
                };
                LibraryVersion::of_hundreds(u64::try_from(raw).unwrap_or(0))
            },
        }
    }
}

impl fmt::Display for Linked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA runtime {}, cuBLAS {}, cuBLASLt {}, cuRAND {}, cuDNN {}",
            self.cudart, self.cublas, self.cublas_lt, self.curand, self.cudnn
        )?;
        #[cfg(feature = "ort-tensorrt")]
        write!(f, ", TensorRT {}", self.tensorrt)?;
        Ok(())
    }
}

/// `library` at `found`, if it is older than `needed`.
fn older(
    library: &'static str,
    found: LibraryVersion,
    needed: LibraryVersion,
) -> Option<RuntimeShortfall> {
    (found < needed).then_some(RuntimeShortfall::Outdated {
        library,
        found,
        needed,
    })
}

/// What this machine can run a detector on, from the driver and from what
/// the linked libraries say of themselves.
pub(super) fn runtime(linked: &Linked) -> CudaRuntime {
    let driver = match driver() {
        Ok(version) => version,
        Err(cuda) => return CudaRuntime::Unavailable { cuda },
    };
    let cuda = older("the driver", driver, CUDA_13_0)
        .or_else(|| older("the CUDA runtime", linked.cudart, CUDA_13_2))
        .or_else(|| older("cuDNN", linked.cudnn, CUDNN_9_23_2));
    if let Some(cuda) = cuda {
        return CudaRuntime::Unavailable { cuda };
    }
    #[cfg(feature = "ort-tensorrt")]
    let tensorrt = older("TensorRT", linked.tensorrt, TENSORRT_10_15_1);
    #[cfg(not(feature = "ort-tensorrt"))]
    let tensorrt = Some(RuntimeShortfall::NotBuilt);
    match tensorrt {
        None => CudaRuntime::TensorRt,
        Some(tensorrt) => CudaRuntime::CudaOnly { tensorrt },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_library_s_report_is_read_as_it_writes_it() {
        assert_eq!(
            LibraryVersion::of_cuda(13020),
            LibraryVersion::new(13, 2, 0)
        );
        assert_eq!(
            LibraryVersion::of_cuda(12080),
            LibraryVersion::new(12, 8, 0)
        );
        assert_eq!(
            LibraryVersion::of_hundreds(92700),
            LibraryVersion::new(9, 27, 0)
        );
        assert_eq!(LibraryVersion::of_hundreds(92302), CUDNN_9_23_2);
        assert_eq!(
            LibraryVersion::of_hundreds(130402),
            LibraryVersion::new(13, 4, 2)
        );
        assert_eq!(
            LibraryVersion::of_hundreds(101601),
            LibraryVersion::new(10, 16, 1)
        );
        assert_eq!(
            LibraryVersion::of_curand(10402),
            LibraryVersion::new(10, 4, 2)
        );
    }

    #[test]
    fn versions_order_by_major_then_minor_then_patch() {
        let needed = LibraryVersion::new(10, 15, 1);
        assert!(LibraryVersion::new(10, 3, 0) < needed);
        assert!(LibraryVersion::new(10, 15, 0) < needed);
        assert!(LibraryVersion::new(10, 16, 0) > needed);
        assert!(LibraryVersion::new(11, 0, 0) > needed);
    }

    #[test]
    fn an_older_library_is_named_with_both_versions() {
        let found = LibraryVersion::new(9, 3, 0);
        assert_eq!(
            older("cuDNN", found, CUDNN_9_23_2),
            Some(RuntimeShortfall::Outdated {
                library: "cuDNN",
                found,
                needed: CUDNN_9_23_2
            })
        );
        assert_eq!(older("cuDNN", CUDNN_9_23_2, CUDNN_9_23_2), None);
    }

    /// The linked libraries answer, each with a version of its own major —
    /// which is also what proves they were linked and opened.
    #[test]
    fn the_linked_libraries_say_their_versions() {
        let linked = Linked::ask();
        eprintln!("{linked}");
        assert_eq!(linked.cudart.major, 13);
        assert_eq!(linked.cublas.major, 13);
        assert_eq!(linked.cublas_lt.major, 13);
        assert_eq!(linked.curand.major, 10);
        assert_eq!(linked.cudnn.major, 9);
        #[cfg(feature = "ort-tensorrt")]
        assert_eq!(linked.tensorrt.major, 10);
    }
}
