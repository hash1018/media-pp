//! Shared measurement helpers for `soak.rs`.
//!
//! **Why process private bytes and not a `#[global_allocator]` counter.**
//! FFmpeg allocates through `av_malloc`, i.e. the C heap, so a Rust-side
//! byte counter cannot see decoder contexts, frame pools, or packets — the
//! largest allocations this library makes. Private bytes covers both heaps
//! at once, which is what a leak in this crate actually looks like from
//! outside the process.
//!
//! **Why a slope and not a before/after delta.** Neither heap returns freed
//! memory to the OS eagerly, and FFmpeg initializes tables lazily, so the
//! first cycles always grow. A soak test comparing one endpoint pair is
//! therefore either flaky or blind. Every scenario here runs a few warm-up
//! cycles it does not measure, then fits a least-squares line through the
//! remaining samples: a leak keeps a positive slope, while allocator noise
//! averages out.

#![allow(dead_code)]

pub mod gpu;

use std::time::Duration;

pub const MIB: f64 = 1024.0 * 1024.0;

/// This process's private (committed, non-shared) bytes — the number that
/// grows when anything in the process leaks, Rust heap and C heap alike.
#[cfg(windows)]
pub fn private_bytes() -> u64 {
    use windows::Win32::System::{
        ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        },
        Threading::GetCurrentProcess,
    };

    let mut counters = PROCESS_MEMORY_COUNTERS_EX::default();
    unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            std::ptr::from_mut(&mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
            size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
        .expect("GetProcessMemoryInfo failed");
    }
    counters.PrivateUsage as u64
}

/// Resident set size from `/proc/self/statm`, the closest Linux equivalent
/// of the Windows counter above. The page size is the 4 KiB every target
/// this crate builds for uses; nothing measured here is precise enough for
/// that to matter beyond scaling the reported numbers.
#[cfg(not(windows))]
pub fn private_bytes() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").expect("read /proc/self/statm");
    let resident: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|pages| pages.parse().ok())
        .expect("/proc/self/statm has a resident-pages field");
    resident * 4096
}

/// What a gauge's samples are counted in — reporting only, but a live
/// object count printed as megabytes is worse than no report at all.
#[derive(Clone, Copy)]
pub enum Unit {
    Bytes,
    Objects,
}

impl Unit {
    fn scale(self) -> f64 {
        match self {
            Self::Bytes => MIB,
            Self::Objects => 1.0,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::Bytes => "MiB",
            Self::Objects => "objects",
        }
    }
}

/// A series of samples of one gauge, taken once per soak iteration.
pub struct Trend {
    label: String,
    unit: Unit,
    gauge: Box<dyn FnMut() -> u64>,
    samples: Vec<u64>,
}

impl Trend {
    pub fn new(label: impl Into<String>, unit: Unit, gauge: impl FnMut() -> u64 + 'static) -> Self {
        Self {
            label: label.into(),
            unit,
            gauge: Box::new(gauge),
            samples: Vec::new(),
        }
    }

    /// The default gauge: this process's private bytes.
    pub fn private_bytes(label: impl Into<String>) -> Self {
        Self::new(label, Unit::Bytes, private_bytes)
    }

    pub fn sample(&mut self) {
        let value = (self.gauge)();
        self.samples.push(value);
    }

    pub fn samples(&self) -> &[u64] {
        &self.samples
    }

    /// Least-squares growth per sample, in bytes. Negative means the gauge
    /// fell over the measured window.
    pub fn slope(&self) -> f64 {
        let n = self.samples.len();
        assert!(
            n >= 8,
            "{}: {n} samples is too few to fit a trend — raise the iteration \
             count (MEDIA_PP_SOAK_ITERS / MEDIA_PP_SOAK_SECS)",
            self.label
        );
        let mean_x = (n - 1) as f64 / 2.0;
        let mean_y = self.samples.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
        let mut covariance = 0.0;
        let mut variance = 0.0;
        for (i, &y) in self.samples.iter().enumerate() {
            let dx = i as f64 - mean_x;
            covariance += dx * (y as f64 - mean_y);
            variance += dx * dx;
        }
        covariance / variance
    }

    /// The standard error of [`Trend::slope`]: how far the fitted slope
    /// would move from run to run on this much noise alone.
    ///
    /// `sigma / sqrt(sum((x - mean_x)^2))`, with `sigma` the residual
    /// spread around the fitted line. The denominator is
    /// `sqrt(n * (n^2 - 1) / 12)`, so this shrinks with `n^1.5` — doubling
    /// a scenario's iteration count buys roughly 2.8x the sensitivity,
    /// which is the knob `MEDIA_PP_SOAK_ITERS` exists for.
    pub fn slope_standard_error(&self) -> f64 {
        let n = self.samples.len();
        let slope = self.slope();
        let mean_x = (n - 1) as f64 / 2.0;
        let mean_y = self.samples.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
        let mut residual_squares = 0.0;
        let mut variance = 0.0;
        for (i, &y) in self.samples.iter().enumerate() {
            let dx = i as f64 - mean_x;
            let residual = y as f64 - (mean_y + slope * dx);
            residual_squares += residual * residual;
            variance += dx * dx;
        }
        // Two degrees of freedom go to the fitted line itself.
        let sigma = (residual_squares / (n - 2) as f64).sqrt();
        sigma / variance.sqrt()
    }

    /// The smallest per-iteration growth this measurement could actually
    /// tell apart from its own noise, given `max_slope`: anything at or
    /// above this trips the assertion with ~95% probability, anything well
    /// below it is invisible no matter how many times the scenario runs at
    /// this iteration count.
    pub fn resolution(&self, max_slope: f64) -> f64 {
        max_slope + 2.0 * self.slope_standard_error()
    }

    /// Fails when the gauge grows faster than `max_slope` per iteration, in
    /// this trend's own unit. The threshold is per scenario because the
    /// scenarios allocate at wildly different scales; each one documents
    /// how its own number was picked.
    ///
    /// Also fails when the window is too noisy for `max_slope` to mean
    /// anything — a scenario that cannot resolve its own threshold is not
    /// passing, it is only failing to look, and that has to be visible
    /// rather than silently reported as a pass.
    ///
    /// The series and what it resolves are printed here rather than at each
    /// call site, so that no scenario can report a pass without also
    /// reporting how sensitive that pass was.
    pub fn assert_flat(&self, max_slope: f64) {
        let slope = self.slope();
        let scale = self.unit.scale();
        let standard_error = self.slope_standard_error();
        eprintln!(
            "{}\n  +- {:.3} {}/iter (1 sigma); resolves growth of {:.3}/iter and above",
            self.report(),
            standard_error / scale,
            self.unit.suffix(),
            self.resolution(max_slope) / scale,
        );
        assert!(
            2.0 * standard_error <= max_slope.max(f64::MIN_POSITIVE),
            "{} is too noisy to judge: slope {:+.3} +- {:.3} {} per iteration against a \
             {:.3} limit. Raise MEDIA_PP_SOAK_ITERS (sensitivity improves with n^1.5) or \
             the scenario's own threshold.\n{}",
            self.label,
            slope / scale,
            standard_error / scale,
            self.unit.suffix(),
            max_slope / scale,
            self.report()
        );
        assert!(
            slope <= max_slope,
            "{} grew {:.3} {} per iteration (limit {:.3})\n{}",
            self.label,
            slope / scale,
            self.unit.suffix(),
            max_slope / scale,
            self.report()
        );
    }

    /// A human-readable summary, printed by every scenario so a run with
    /// `--nocapture` shows the actual numbers whether it passed or failed.
    pub fn report(&self) -> String {
        let first = *self.samples.first().expect("at least one sample");
        let last = *self.samples.last().expect("at least one sample");
        let scale = self.unit.scale();
        let series = self
            .samples
            .iter()
            .map(|&value| format!("{:.1}", value as f64 / scale))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "{}: {:.1} -> {:.1} {} over {} samples, slope {:+.3}/iter\n  {series}",
            self.label,
            first as f64 / scale,
            last as f64 / scale,
            self.unit.suffix(),
            self.samples.len(),
            self.slope() / scale,
        )
    }

    pub fn print(&self) {
        eprintln!("{}", self.report());
    }
}

/// Re-runs one scenario in its own child process, and reports whether the
/// caller is that parent (`true`, nothing left to do) or the child that
/// should go on to run the scenario body (`false`).
///
/// Every gauge here measures a whole process, and a process that has
/// already run other scenarios is not quiet: the graphics driver in
/// particular trims its allocations for tens of seconds afterwards, which
/// lands inside the next scenario's window as drift far larger than
/// anything being measured. Serializing them is not enough, and waiting it
/// out ([`settle`]) only shortens it. A scenario that starts from a fresh
/// process has no such history — which is what makes its numbers mean what
/// they say.
///
/// `name` is the test's own path (`d3d11::decode_cycles_...`), passed to
/// the child as `--exact`. A name that matches nothing would make the child
/// exit successfully having run zero tests, so that case is checked for
/// explicitly rather than passing silently.
pub fn spawn_isolated(name: &str) -> bool {
    use std::io::Write;

    const MARKER: &str = "MEDIA_PP_SOAK_ISOLATED";
    if std::env::var_os(MARKER).is_some() {
        return false;
    }

    let executable = std::env::current_exe().expect("this test binary's own path");
    let output = std::process::Command::new(executable)
        .args([
            name,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(MARKER, "1")
        .output()
        .expect("run this scenario in its own process");

    // Forwarded so that `--nocapture` still shows the child's measurements
    // and, on failure, its assertion.
    std::io::stdout()
        .write_all(&output.stdout)
        .expect("forward the child's output");
    std::io::stderr()
        .write_all(&output.stderr)
        .expect("forward the child's output");

    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(
        summary.contains("1 passed") || !output.status.success(),
        "{name} matched no test in its own process — the name passed to spawn_isolated has \
         drifted from the test's actual path"
    );
    assert!(
        output.status.success(),
        "{name} failed in its own process (its output is above)"
    );
    true
}

/// Blocks until this process's private bytes stop moving, or gives up
/// after a few seconds and says so.
///
/// Scenarios share one process, and a heap does not hand memory back the
/// instant a scenario ends: a window that starts too soon after the
/// previous one measures that release instead of its own subject, which
/// shows up as a large slope in *either* direction with residuals to
/// match. That is not a leak and not noise the iteration count can average
/// away — it is a baseline still in motion, so the fix is to wait for it
/// rather than to measure through it.
pub fn settle() {
    // A GPU-heavy scenario's release drains slowly — a few MiB per second
    // for tens of seconds — so this bar has to sit below that rate, not
    // merely below a cycle's worth. An earlier, looser version (1 MiB per
    // 250 ms, i.e. 4 MiB/s) declared a 3.5 MiB/s decay "settled" and let it
    // straight into the measurement window.
    const QUIET_BYTES: u64 = 256 * 1024;
    const QUIET_SAMPLES: usize = 8;

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let mut previous = private_bytes();
    let mut quiet = 0;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(250));
        let current = private_bytes();
        quiet = if current.abs_diff(previous) < QUIET_BYTES {
            quiet + 1
        } else {
            0
        };
        previous = current;
        if quiet >= QUIET_SAMPLES {
            return;
        }
    }
    eprintln!(
        "note: private bytes never settled; this scenario's window starts on a moving baseline"
    );
}

/// Serializes the scenarios against each other.
///
/// Every gauge here measures the whole process, and `cargo test` runs test
/// functions on parallel threads by default — two scenarios at once would
/// each be sampling the other's allocations. Binding this guard for the
/// body of a scenario makes the file correct on its own, without anyone
/// having to remember `--test-threads=1`.
pub fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A scenario that panics while holding this must not turn every later
    // one into a poison error instead of its own real result.
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// How many measured iterations a scenario runs, from
/// `MEDIA_PP_SOAK_ITERS`. The defaults keep the whole file inside a few
/// minutes; a real overnight soak raises this.
pub fn iterations(default: usize) -> usize {
    env_parsed("MEDIA_PP_SOAK_ITERS", default)
}

/// How long each duration-driven scenario runs, from `MEDIA_PP_SOAK_SECS`.
pub fn soak_duration(default_secs: u64) -> Duration {
    Duration::from_secs(env_parsed("MEDIA_PP_SOAK_SECS", default_secs))
}

fn env_parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Path to a real video file for the scenarios that need one, from
/// `MEDIA_PP_TEST_VIDEO` — the same variable, and the same skip-with-a-
/// reason contract, as the library's own `test_support::try_test_video`
/// (which is `pub(crate)`, so an integration test cannot call it).
pub fn try_test_video() -> Option<String> {
    let Ok(path) = std::env::var("MEDIA_PP_TEST_VIDEO") else {
        eprintln!(
            "skipping: set MEDIA_PP_TEST_VIDEO to a video file to run this test \
             (no media is checked into this repository)"
        );
        return None;
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("skipping: MEDIA_PP_TEST_VIDEO={path} is not a readable file");
        return None;
    }
    Some(path)
}

/// A scratch directory that deletes itself on drop, for the scenarios that
/// record to disk. Deleting is also an assertion: Windows refuses to remove
/// a file something still holds open, so a clean teardown proves no muxer
/// kept a handle past its own finalization.
pub struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    pub fn new(prefix: &str) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("media-pp-soak-{prefix}-{unique}"));
        std::fs::create_dir_all(&path).expect("create the soak scratch directory");
        Self { path }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn join(&self, name: &str) -> std::path::PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.path)
            .expect("every recording must be closed by teardown, so its directory can be removed");
    }
}
