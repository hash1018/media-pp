//! GPU-side gauges for the soak scenarios.
//!
//! Neither heap counter in this module's parent can see a leaked texture or
//! a leaked CUDA surface: GPU memory is not part of the process's private
//! bytes at all. These are the two ways to observe it from inside the
//! process (D3D11) or from the driver (CUDA).

#![allow(dead_code, unused_imports)]

/// Per-process GPU memory, in bytes, as the NVIDIA driver reports it for
/// this PID. `None` when `nvidia-smi` is missing, fails, or — as on a
/// GeForce card in WDDM mode — declines to break usage down per process;
/// the caller then has to fall back to what the CPU-side gauges can see and
/// say so.
pub fn nvidia_process_bytes() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pid = std::process::id();
    let listing = String::from_utf8_lossy(&output.stdout);
    for line in listing.lines() {
        let mut fields = line.split(',').map(str::trim);
        let (Some(entry_pid), Some(used_mib)) = (fields.next(), fields.next()) else {
            continue;
        };
        if entry_pid.parse::<u32>() != Ok(pid) {
            continue;
        }
        // "[N/A]" / "[Not Supported]" on drivers that do not break usage
        // down per process — not zero, unknown.
        let used_mib: u64 = used_mib.parse().ok()?;
        return Some(used_mib * 1024 * 1024);
    }
    // The driver lists a process only once it holds a CUDA context, so
    // "absent" legitimately means "nothing allocated yet".
    Some(0)
}

#[cfg(windows)]
pub use windows_gpu::*;

#[cfg(windows)]
mod windows_gpu {
    use std::sync::{Arc, Mutex};

    use windows::{
        Win32::Graphics::{
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11_CREATE_DEVICE_DEBUG, D3D11_MESSAGE, D3D11_RLDO_DETAIL, D3D11_SDK_VERSION,
                D3D11CreateDevice, ID3D11Debug, ID3D11Device, ID3D11DeviceContext, ID3D11InfoQueue,
            },
            Dxgi::{
                DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
                IDXGIDevice,
            },
        },
        core::Interface,
    };

    #[cfg(feature = "d3d12")]
    use windows::Win32::Graphics::{
        Direct3D12::ID3D12Device,
        Dxgi::{CreateDXGIFactory1, IDXGIFactory4},
    };

    /// A D3D11 device, its shared immediate context, and — when the SDK
    /// debug layer is installed — the debug interfaces that can enumerate
    /// live objects. `None`, after printing why, on a machine without a
    /// usable device, the same way every other hardware test here skips.
    ///
    /// The debug layer is requested first and dropped if unavailable, since
    /// only it can answer "did this cycle leave an object behind"; a
    /// machine without it still runs the VRAM half of the scenario.
    pub fn try_d3d11_device() -> Option<(
        ID3D11Device,
        Arc<Mutex<ID3D11DeviceContext>>,
        Option<D3d11LiveObjects>,
    )> {
        if let Some(created) = create_device(true) {
            let debug = D3d11LiveObjects::new(&created.0);
            if debug.is_none() {
                eprintln!(
                    "note: D3D11 debug layer created but exposes no ID3D11Debug/ID3D11InfoQueue; \
                     measuring VRAM only"
                );
            }
            return Some((created.0, created.1, debug));
        }
        eprintln!(
            "note: no D3D11 debug layer on this machine (install the Graphics Tools optional \
             feature to enumerate live objects); measuring VRAM only"
        );
        let created = create_device(false).or_else(|| {
            eprintln!("skipping: D3D11CreateDevice failed on this machine");
            None
        })?;
        Some((created.0, created.1, None))
    }

    fn create_device(debug: bool) -> Option<(ID3D11Device, Arc<Mutex<ID3D11DeviceContext>>)> {
        let flags = if debug {
            D3D11_CREATE_DEVICE_DEBUG
        } else {
            Default::default()
        };
        let mut device = None;
        let mut context = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                flags,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if result.is_err() {
            return None;
        }
        Some((
            device.expect("D3D11CreateDevice succeeded without producing a device"),
            Arc::new(Mutex::new(context.expect(
                "D3D11CreateDevice succeeded without producing a context",
            ))),
        ))
    }

    /// This process's current usage of the adapter's own video memory, in
    /// bytes. Unlike the CPU gauges this one is reported by DXGI itself, so
    /// it counts every texture the pipeline's D3D11 elements allocate —
    /// including the decoder's fixed D3D11VA pool and the scaler's output
    /// pool — and nothing that belongs to another process.
    pub fn vram_bytes(device: &ID3D11Device) -> u64 {
        let dxgi_device: IDXGIDevice = device.cast().expect("a D3D11 device is an IDXGIDevice");
        let adapter: IDXGIAdapter3 = unsafe { dxgi_device.GetAdapter() }
            .expect("IDXGIDevice::GetAdapter")
            .cast()
            .expect("IDXGIAdapter3 needs Windows 10 or newer");
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .expect("QueryVideoMemoryInfo");
        info.CurrentUsage
    }

    /// This process's current local-video-memory usage on the adapter that
    /// owns `device`. D3D12 devices do not implement `IDXGIDevice`, so find
    /// the same adapter by its LUID before asking DXGI for the process gauge.
    #[cfg(feature = "d3d12")]
    pub fn d3d12_vram_bytes(device: &ID3D12Device) -> u64 {
        let luid = unsafe { device.GetAdapterLuid() };
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }.expect("CreateDXGIFactory1");
        let adapter: IDXGIAdapter3 =
            unsafe { factory.EnumAdapterByLuid(luid) }.expect("EnumAdapterByLuid for D3D12 device");
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .expect("QueryVideoMemoryInfo");
        info.CurrentUsage
    }

    /// The debug layer's live-object enumeration, as a gauge.
    ///
    /// `ID3D11Debug::ReportLiveDeviceObjects` writes its findings to the
    /// debug output, which a test process cannot read back — but the same
    /// findings also land in `ID3D11InfoQueue` as ordinary messages, one
    /// per live object plus a summary. Clearing the queue first therefore
    /// turns the report into a number this process can actually compare
    /// cycle over cycle, and the messages themselves into text a failure
    /// can print.
    pub struct D3d11LiveObjects {
        debug: ID3D11Debug,
        info: ID3D11InfoQueue,
    }

    impl D3d11LiveObjects {
        fn new(device: &ID3D11Device) -> Option<Self> {
            let debug: ID3D11Debug = device.cast().ok()?;
            let info: ID3D11InfoQueue = device.cast().ok()?;
            // The queue drops messages past its default limit, which a
            // detailed live-object report can exceed on its own.
            unsafe { info.SetMessageCountLimit(u64::MAX) }.ok()?;
            Some(Self { debug, info })
        }

        /// How many objects the device still owns right now. The device,
        /// its context, and the debug interfaces themselves are always
        /// among them, so the absolute value carries no meaning — only its
        /// trend across identical cycles does.
        pub fn count(&self) -> u64 {
            self.report_into_queue();
            unsafe { self.info.GetNumStoredMessages() }
        }

        /// One line per live object, for a failure message.
        pub fn describe(&self) -> Vec<String> {
            self.report_into_queue();
            let stored = unsafe { self.info.GetNumStoredMessages() };
            let mut lines = Vec::with_capacity(stored as usize);
            for index in 0..stored {
                let mut length = 0usize;
                if unsafe { self.info.GetMessage(index, None, &mut length) }.is_err() {
                    continue;
                }
                let mut buffer = vec![0u8; length];
                let message = buffer.as_mut_ptr().cast::<D3D11_MESSAGE>();
                if unsafe { self.info.GetMessage(index, Some(message), &mut length) }.is_err() {
                    continue;
                }
                let message = unsafe { &*message };
                let description = unsafe {
                    std::slice::from_raw_parts(
                        message.pDescription.cast::<u8>(),
                        message.DescriptionByteLength.saturating_sub(1),
                    )
                };
                lines.push(String::from_utf8_lossy(description).into_owned());
            }
            lines
        }

        fn report_into_queue(&self) {
            unsafe {
                self.info.ClearStoredMessages();
                self.debug
                    .ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)
                    .expect("ReportLiveDeviceObjects");
            }
        }
    }
}
