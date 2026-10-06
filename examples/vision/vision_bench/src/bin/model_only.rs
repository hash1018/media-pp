//! The model alone: ONNX Runtime running the detector on a tensor already
//! on the GPU, over and over, its output left there — no decoding, no
//! fitting, no reading of boxes. It is the most pictures a second the model
//! can be run at on this GPU, the ceiling `vision_bench` is measured
//! against, and what DeepStream's `nvinfer` could reach at best: it runs the
//! same model through the same TensorRT.
//!
//! The TensorRT provider is set up as `CudaOrtDetector` sets it up, with
//! the same engine cache — or, with `--engine-cache-root`, the directory
//! `vision_bench` gives the same model, precision and batch — so an engine
//! one has built loads in the other in a second.
//!
//!     cargo run --release -p vision_bench --bin model_only -- yolo11n.onnx \
//!         [--provider tensorrt|cuda] [--fp32] [--batch 1,2,4,8] [--runs N] \
//!         [--engine-cache-root DIR] [--cuda-graph] [--opt N] [--max N] [--int8] \
//!         [--int8-table FILE]

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn main() {
    eprintln!("model_only runs ONNX Runtime's TensorRT and CUDA providers: Linux and Windows");
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn main() -> ort::Result<()> {
    use std::path::PathBuf;
    use std::time::Instant;

    use ort::{
        ep::{CUDA, TensorRT},
        memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
        session::Session,
        value::{Tensor, ValueType},
    };

    let usage = || -> ! {
        eprintln!(
            "usage: model_only <model.onnx> [--provider tensorrt|cuda] [--fp32] \
             [--batch 1,2,4,8] [--runs N] [--engine-cache-root DIR] [--cuda-graph] [--opt N] [--max N] [--int8] [--int8-table FILE]"
        );
        std::process::exit(2);
    };
    let mut model = None;
    let mut engine_root: Option<PathBuf> = None;
    let mut cuda_graph = false;
    // A profile of its own — 1 to `max`, fastest at `opt` — rather than the
    // batch alone, as a classifier's engine is built.
    let (mut opt, mut max): (Option<usize>, Option<usize>) = (None, None);
    let (mut tensorrt, mut fp16, mut batches, mut runs) = (true, true, vec![1], 500);
    // A model quantized to INT8 Q/DQ is run in INT8 where TensorRT is let.
    let mut int8 = false;
    // TensorRT's own INT8 for a float model: a calibration table ONNX
    // Runtime's calibrator wrote, from which it picks INT8 or FP16 per layer.
    let mut int8_table: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--provider" => {
                tensorrt = match args.next().as_deref() {
                    Some("tensorrt") => true,
                    Some("cuda") => false,
                    _ => usage(),
                }
            }
            "--fp32" => fp16 = false,
            "--int8" => int8 = true,
            "--int8-table" => {
                int8 = true;
                int8_table = Some(args.next().unwrap_or_else(|| usage()));
            }
            "--cuda-graph" => cuda_graph = true,
            "--opt" => {
                opt = Some(
                    args.next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage()),
                )
            }
            "--max" => {
                max = Some(
                    args.next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage()),
                )
            }
            "--batch" => {
                batches = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .split(',')
                    .map(|n| n.parse().unwrap_or_else(|_| usage()))
                    .collect()
            }
            "--engine-cache-root" => {
                engine_root = Some(args.next().unwrap_or_else(|| usage()).into())
            }
            "--runs" => {
                runs = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            _ if model.is_none() => model = Some(arg),
            _ => usage(),
        }
    }
    let model = model.unwrap_or_else(|| usage());

    // CudaOrtDetector's own engine cache.
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("media-pp")
        .join("tensorrt");

    let probe = Session::builder()?.commit_from_file(&model)?;
    let input = &probe.inputs()[0];
    let name = input.name().to_owned();
    let ValueType::Tensor { shape, .. } = input.dtype() else {
        usage()
    };
    let open_batch = shape[0] < 0;
    // An open size is a detector's 640.
    let (h, w) = (
        if shape[2] > 0 { shape[2] } else { 640 },
        if shape[3] > 0 { shape[3] } else { 640 },
    );
    let stem = std::path::Path::new(&model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&model)
        .to_owned();
    drop(probe);

    for batch in batches {
        if batch > 1 && !open_batch {
            eprintln!("{model} takes one picture at a time: batch {batch} skipped");
            continue;
        }
        let mut builder = Session::builder()?;
        let mut providers = Vec::new();
        if tensorrt {
            // ONNX Runtime names an engine after the model's graph alone:
            // each batch keeps its own in a directory of its own, as
            // vision_bench names them.
            let cache = match &engine_root {
                Some(root) => root.join(format!(
                    "{stem}-{}-{}",
                    match (int8, fp16) {
                        (true, _) => "int8",
                        (false, true) => "fp16",
                        (false, false) => "fp32",
                    },
                    match (opt, max) {
                        (None, None) => format!("b{batch}"),
                        (opt, max) => format!(
                            "batch-1-{}-{}",
                            opt.unwrap_or(batch),
                            max.unwrap_or(batch).max(batch)
                        ),
                    }
                )),
                None => cache.clone(),
            };
            // ONNX Runtime makes the directory, but not the ones above it.
            if let Err(error) = std::fs::create_dir_all(&cache) {
                eprintln!("no engine cache at {}: {error}", cache.display());
            }
            let mut provider = TensorRT::default()
                .with_device_id(0)
                .with_fp16(fp16)
                .with_int8(int8)
                .with_engine_cache(true)
                .with_engine_cache_path(cache.display())
                .with_timing_cache(true)
                .with_timing_cache_path(cache.display())
                .with_cuda_graph(cuda_graph);
            let (opt, max) = (opt.unwrap_or(batch), max.unwrap_or(batch).max(batch));
            if max > 1 {
                let shape = |b: usize| format!("{name}:{b}x3x{h}x{w}");
                provider = provider
                    .with_profile_min_shapes(shape(1))
                    .with_profile_opt_shapes(shape(opt))
                    .with_profile_max_shapes(shape(max));
            }
            if let Some(table) = &int8_table {
                provider = provider
                    .with_int8_calibration_table_name(table)
                    .with_int8_use_native_calibration_table(false);
            }
            providers.push(provider.build().error_on_failure());
        }
        providers.push(CUDA::default().with_device_id(0).build().error_on_failure());
        builder = builder.with_execution_providers(providers)?;
        let mut session = builder.commit_from_file(&model)?;

        let device = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let allocator = Allocator::new(&session, device.clone())?;
        let tensor = Tensor::<f32>::new(&allocator, [batch, 3, h as usize, w as usize])?;
        let mut binding = session.create_binding()?;
        binding.bind_input(&name, &tensor)?;
        for output in session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect::<Vec<_>>()
        {
            binding.bind_output_to_device(output, &device)?;
        }

        // The first runs build or load the engine and settle the clocks.
        for _ in 0..50 {
            session.run_binding(&binding)?;
        }
        binding.synchronize()?;
        let started = Instant::now();
        for _ in 0..runs {
            session.run_binding(&binding)?;
        }
        binding.synchronize()?;
        let seconds = started.elapsed().as_secs_f64();
        let per_batch = seconds / runs as f64;
        println!(
            "MODEL model={stem} provider={}{} precision={} profile=1-{}-{} batch={batch} \
             runs={runs} ms_per_batch={:.3} fps={:.1}",
            if tensorrt { "tensorrt" } else { "cuda" },
            if tensorrt && cuda_graph { "+graph" } else { "" },
            match (tensorrt, int8, fp16) {
                (true, true, _) => "int8",
                (true, false, true) => "fp16",
                _ => "fp32",
            },
            opt.unwrap_or(batch),
            max.unwrap_or(batch).max(batch),
            per_batch * 1e3,
            batch as f64 / per_batch,
        );
    }
    Ok(())
}
