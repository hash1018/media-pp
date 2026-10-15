//! Inference on D3D12 pictures through ONNX Runtime's DirectML provider:
//! the picture fitted to the model by a compute shader into a buffer on the
//! GPU, which DirectML reads where it is — on the [`D3d12Gpu`]'s own device
//! and queue, so the fit and the model are ordered by the queue alone and
//! nothing goes through host memory on the way in. See
//! `docs/d3d12-inference.md` for the measurements this rests on.

mod d3d12_ort_classifier;
mod d3d12_ort_detector;
mod d3d12_ort_embedder;
mod fitting;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use ort::{
    AsPointer,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::ValueType,
};
use windows::{
    Win32::AI::MachineLearning::DirectML::{
        DML_CREATE_DEVICE_FLAG_NONE, DMLCreateDevice, IDMLDevice,
    },
    core::Interface,
};

use super::OrtError;
use super::classify::{Input, image_input};
use crate::platform::windows::d3d12_gpu::D3d12Gpu;
use crate::pp_log::{PpLog, pp_warn};

pub use d3d12_ort_classifier::D3d12OrtClassifier;
pub use d3d12_ort_detector::{D3d12OrtDetector, D3d12OrtDetectorOptions};
pub use d3d12_ort_embedder::D3d12OrtEmbedder;

/// What DirectML runs on: a DirectML device made on a [`D3d12Gpu`]'s
/// device, and the GPU's queue it is handed beside it.
struct DirectMl {
    device: IDMLDevice,
    gpu: D3d12Gpu,
}

// SAFETY: a DirectML device is free-threaded, as the D3D12 objects are.
unsafe impl Send for DirectMl {}

impl DirectMl {
    /// DirectML on `gpu`.
    fn new(gpu: &D3d12Gpu) -> Result<Self, OrtError> {
        if ort::ep::directml::api().is_none() {
            return Err(OrtError::DirectMlUnavailable);
        }
        let mut device: Option<IDMLDevice> = None;
        // SAFETY: a live D3D12 device and a correctly typed out-parameter.
        unsafe { DMLCreateDevice(gpu.device(), DML_CREATE_DEVICE_FLAG_NONE, &mut device) }?;
        Ok(Self {
            device: device.expect("DMLCreateDevice succeeded without a device"),
            gpu: gpu.clone(),
        })
    }

    /// A session on the model at `model_path`, run by DirectML on this
    /// device and queue, the dimensions `fixed` names fixed at their sizes.
    ///
    /// DirectML wants every dimension known when the session is made, and
    /// neither ONNX Runtime's memory patterns nor its parallel execution.
    fn session(&self, model_path: &Path, fixed: &[(&str, i64)]) -> Result<Session, OrtError> {
        let api = ort::ep::directml::api().ok_or(OrtError::DirectMlUnavailable)?;
        let mut builder = Session::builder()?
            .with_memory_pattern(false)
            .map_err(|error| OrtError::Ort(error.into()))?
            .with_parallel_execution(false)
            .map_err(|error| OrtError::Ort(error.into()))?;
        for &(name, size) in fixed {
            builder = builder
                .with_dimension_override(name, size)
                .map_err(|error| OrtError::Ort(error.into()))?;
        }
        // `ort`'s own DirectML provider takes only an adapter's index and
        // makes a device of its own; this hands it the GPU's, so a buffer
        // of that device is a tensor it can read.
        // SAFETY: the options are the builder's own and live; the DirectML
        // device and queue are live COM objects, which ONNX Runtime takes
        // references of.
        unsafe {
            ort::Error::result_from_status((api.SessionOptionsAppendExecutionProvider_DML1)(
                builder.ptr_mut().cast(),
                self.device.as_raw(),
                self.gpu.queue().as_raw(),
            ))?;
        }
        Ok(builder.commit_from_file(model_path)?)
    }
}

/// How many objects a model that takes any number at once is fixed to take
/// for DirectML, which is fastest with every dimension fixed: a picture's
/// objects go in eights, the inputs past them left as they were.
const DIRECTML_BATCH: usize = 8;

/// A session on the object model at `model_path` — an embedder's or a
/// classifier's, of one input `[batch, 3, height, width]`, `open_side` for a side
/// it leaves open — run by DirectML on `gpu`, with its input as fixed.
///
/// The model is read on the CPU first for what it calls the dimensions it
/// leaves open, which are fixed to those sizes, its batch to
/// [`DIRECTML_BATCH`].
fn object_model_session(
    gpu: &D3d12Gpu,
    model_path: &Path,
    open_side: u32,
    pp_log: &PpLog,
) -> Result<(Session, Input), OrtError> {
    warn_past_directml_opset(model_path, pp_log);
    let probe = Session::builder()
        .map_err(OrtError::from)?
        .commit_from_file(model_path)
        .map_err(OrtError::from)?;
    let open = image_input(&probe, open_side)?;
    let mut fixed: Vec<(String, i64)> = Vec::new();
    if let Some(ValueType::Tensor {
        shape,
        dimension_symbols,
        ..
    }) = probe.inputs().first().map(|input| input.dtype())
    {
        let named = |axis: usize| {
            dimension_symbols
                .get(axis)
                .filter(|symbol| !symbol.is_empty() && shape[axis] <= 0)
                .cloned()
        };
        for (axis, size) in [(2, open.size.1), (3, open.size.0)] {
            if let Some(symbol) = named(axis) {
                fixed.push((symbol, i64::from(size)));
            }
        }
        match (open.batch, named(0)) {
            (None, Some(symbol)) => fixed.push((symbol, DIRECTML_BATCH as i64)),
            (None, None) => pp_warn!(
                pp_log: pp_log,
                "the model leaves its batch open without a name to fix it by: \
                 DirectML runs it at each number of objects as it comes"
            ),
            (Some(_), _) => {}
        }
    }
    drop(probe);
    let fixed: Vec<(&str, i64)> = fixed
        .iter()
        .map(|(symbol, size)| (symbol.as_str(), *size))
        .collect();
    let session = DirectMl::new(gpu)?.session(model_path, &fixed)?;
    let input = image_input(&session, open_side)?;
    Ok((session, input))
}

/// The newest ONNX operator set DirectML runs whole: an operator of a later
/// one goes to the CPU, a node at a time, and the model then runs many times
/// slower than on the GPU — see `docs/d3d12-inference.md`.
const DIRECTML_OPSET: u64 = 20;

/// Warns, through `pp_log`, where the model at `model_path` is written
/// against a later operator set than [`DIRECTML_OPSET`]. ONNX Runtime says
/// nothing when it hands DirectML's missing operators to the CPU, so this is
/// the one place a user hears why the model is slow.
fn warn_past_directml_opset(model_path: &Path, pp_log: &PpLog) {
    if let Some(opset) = onnx_opset(model_path).filter(|&opset| opset > DIRECTML_OPSET) {
        pp_warn!(
            pp_log: pp_log,
            "the model is exported at ONNX opset {opset}: DirectML runs opset \
             {DIRECTML_OPSET} and lower, and hands the operators of later ones to \
             the CPU, which makes it many times slower; export it at opset \
             {DIRECTML_OPSET} or lower"
        );
    }
}

/// The ONNX operator set the model file at `path` is written against — its
/// `ai.onnx` import — read from the file's top level without loading the
/// graph, which is skipped over. `None` where the file cannot be read or
/// does not say.
fn onnx_opset(path: &Path) -> Option<u64> {
    // `ModelProto` field 8, `opset_import`: repeated `OperatorSetIdProto`.
    const OPSET_IMPORT: u64 = 8;
    let mut file = BufReader::new(File::open(path).ok()?);
    let mut opset = None;
    // A clean end of the file is where the key would be.
    while let Some(key) = varint(&mut file) {
        match (key >> 3, key & 7) {
            (_, 0) => {
                varint(&mut file)?;
            }
            (_, 1) => file.seek_relative(8).ok()?,
            (_, 5) => file.seek_relative(4).ok()?,
            (OPSET_IMPORT, 2) => {
                let mut entry = vec![0; usize::try_from(varint(&mut file)?).ok()?];
                file.read_exact(&mut entry).ok()?;
                if let Some((domain, version)) = operator_set(&entry)
                    && (domain.is_empty() || domain == b"ai.onnx")
                {
                    opset = Some(version);
                }
            }
            (_, 2) => {
                let len = i64::try_from(varint(&mut file)?).ok()?;
                file.seek_relative(len).ok()?;
            }
            _ => return None,
        }
    }
    opset
}

/// An `OperatorSetIdProto`'s domain (field 1) and version (field 2).
fn operator_set(mut entry: &[u8]) -> Option<(&[u8], u64)> {
    let (mut domain, mut version): (&[u8], _) = (&[], None);
    while !entry.is_empty() {
        let key = varint(&mut entry)?;
        match (key >> 3, key & 7) {
            (2, 0) => version = Some(varint(&mut entry)?),
            (_, 0) => {
                varint(&mut entry)?;
            }
            (field, 2) => {
                let len = usize::try_from(varint(&mut entry)?).ok()?;
                let bytes = entry.get(..len)?;
                if field == 1 {
                    domain = bytes;
                }
                entry = &entry[len..];
            }
            _ => return None,
        }
    }
    Some((domain, version?))
}

/// A protobuf base-128 varint, `None` at the end of `reader` or past ten
/// bytes.
fn varint(reader: &mut impl Read) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let mut byte = [0u8];
        reader.read_exact(&mut byte).ok()?;
        value |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// Where a tensor DirectML reads is: a D3D12 buffer on its device.
fn directml_memory() -> Result<MemoryInfo<'static>, OrtError> {
    Ok(MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

/// Where outputs are copied for the CPU to read.
fn host_memory() -> Result<MemoryInfo<'static>, OrtError> {
    Ok(MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A protobuf key: field number and wire type.
    fn key(field: u64, wire: u64) -> Vec<u8> {
        encode(field << 3 | wire)
    }

    fn encode(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn bytes_field(field: u64, bytes: &[u8]) -> Vec<u8> {
        [key(field, 2), encode(bytes.len() as u64), bytes.to_vec()].concat()
    }

    fn operator_set_id(domain: &str, version: u64) -> Vec<u8> {
        let mut entry = Vec::new();
        if !domain.is_empty() {
            entry.extend(bytes_field(1, domain.as_bytes()));
        }
        entry.extend([key(2, 0), encode(version)].concat());
        bytes_field(8, &entry)
    }

    fn model_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("media-pp-opset-{}-{name}.onnx", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn the_opset_is_read_past_a_graph_it_skips() {
        // ir_version, producer_name, a graph of a megabyte, then the
        // imports: a custom domain's first, the default one's after it.
        let contents = [
            [key(1, 0), encode(10)].concat(),
            bytes_field(2, b"pytorch"),
            bytes_field(7, &vec![0xff; 1 << 20]),
            operator_set_id("com.microsoft", 1),
            operator_set_id("", 22),
            [key(5, 0), encode(1)].concat(),
        ]
        .concat();
        let path = model_file("default", &contents);
        assert_eq!(onnx_opset(&path), Some(22));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_ai_onnx_domain_counts_as_the_default_one() {
        let contents = [
            operator_set_id("ai.onnx", 17),
            operator_set_id("ai.onnx.ml", 3),
        ]
        .concat();
        let path = model_file("named", &contents);
        assert_eq!(onnx_opset(&path), Some(17));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_file_that_is_not_a_model_says_no_opset() {
        let path = model_file("garbage", b"\x0f\x0f\x0f not protobuf");
        assert_eq!(onnx_opset(&path), None);
        std::fs::remove_file(path).unwrap();
        assert_eq!(onnx_opset(Path::new("no such model.onnx")), None);
    }
}
