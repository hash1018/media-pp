//! What a DXGI flip-model swap chain takes to put a picture on the screen,
//! learned from its frame statistics without holding the renderer — shared
//! by the D3D11 and D3D12 window renderers.

use std::{collections::VecDeque, time::Duration};

use windows::Win32::{
    Foundation::HWND,
    Graphics::{
        Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo},
        Dxgi::{DXGI_FRAME_STATISTICS, IDXGISwapChain1},
    },
    System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency},
};

use crate::elements::sink::renderer::presentation_delay::PresentationDelay;

/// How many measured presents may wait for their statistics at once: far
/// more than a flip chain holds in flight.
const PENDING: usize = 8;

/// A swap chain's presentation delay, measured from its frame statistics.
///
/// A present picked by [`PresentationDelay::due`] is noted with its present
/// count and the moment its frame was handed over. DXGI reports, for the
/// last present that reached the screen, its count and the vertical blank
/// it was shown at; when that is a noted one, the time between is one
/// measurement. Reading the statistics does not wait for anything, so
/// unlike a present waited for, a measurement costs the renderer nothing.
///
/// Where the statistics say nothing — before the first present is shown,
/// or on a desktop that does not give them to a window — two refreshes of
/// the desktop's compositor are published instead: the wait for the next
/// refresh, and the composition after it, as on X11.
pub(super) struct PresentTiming {
    pub(super) delay: PresentationDelay,
    /// Ticks per second of the performance counter frame statistics are in.
    frequency: i64,
    /// Measured presents not yet seen on the screen: count, handed over at.
    pending: VecDeque<(u32, i64)>,
    /// Whether this swap chain has been estimated for, or measured, since
    /// it was made: an estimate is for the time before a measurement.
    estimated: bool,
    measured: bool,
}

impl PresentTiming {
    pub(super) fn new() -> Self {
        let mut frequency = 0;
        // SAFETY: an out-pointer to a local.
        let _ = unsafe { QueryPerformanceFrequency(&mut frequency) };
        Self {
            delay: PresentationDelay::default(),
            frequency,
            pending: VecDeque::new(),
            estimated: false,
            measured: false,
        }
    }

    /// The moment a frame is handed over, in the counter's ticks — what a
    /// measurement counts from, as a synchronizer schedules by it.
    pub(super) fn now() -> i64 {
        let mut now = 0;
        // SAFETY: an out-pointer to a local.
        let _ = unsafe { QueryPerformanceCounter(&mut now) };
        now
    }

    /// After a frame handed over at `handed` was presented on `swap_chain`:
    /// takes what the statistics say of presents measured before, and notes
    /// this one where it is due.
    pub(super) fn presented(&mut self, swap_chain: &IDXGISwapChain1, handed: i64) {
        if self.frequency <= 0 {
            return;
        }
        let mut statistics = DXGI_FRAME_STATISTICS::default();
        // SAFETY: a live swap chain, used only under its renderer's lock,
        // and an out-pointer to a local.
        let shown = unsafe { swap_chain.GetFrameStatistics(&mut statistics) }.is_ok()
            && statistics.PresentCount != 0;
        if shown {
            while let Some(&(count, at)) = self.pending.front() {
                if count > statistics.PresentCount {
                    break;
                }
                self.pending.pop_front();
                if count == statistics.PresentCount && statistics.SyncQPCTime >= at {
                    let ticks = (statistics.SyncQPCTime - at) as f64;
                    self.delay
                        .record(Duration::from_secs_f64(ticks / self.frequency as f64));
                    self.measured = true;
                }
            }
        } else if !self.estimated && !self.measured {
            self.estimate();
        }
        if self.delay.due() {
            // SAFETY: as above.
            if let Ok(count) = unsafe { swap_chain.GetLastPresentCount() } {
                if self.pending.len() == PENDING {
                    self.pending.pop_front();
                }
                self.pending.push_back((count, handed));
            }
        }
    }

    /// Forgets what was measured, for buffers remade at another size — full
    /// screen may skip the compositor.
    pub(super) fn restart(&mut self) {
        self.delay.restart();
        self.pending.clear();
        self.estimated = false;
        self.measured = false;
    }

    fn estimate(&mut self) {
        self.estimated = true;
        if let Some(interval) = desktop_refresh() {
            self.delay.estimate(interval * 2, interval);
        }
    }
}

/// How often the desktop's compositor refreshes — `None` where there is no
/// compositor to ask, as in a session with no desktop.
pub(super) fn desktop_refresh() -> Option<Duration> {
    let mut frequency = 0;
    let mut timing = DWM_TIMING_INFO {
        cbSize: size_of::<DWM_TIMING_INFO>() as u32,
        ..Default::default()
    };
    // SAFETY: out-pointers to locals, the timing's size set; a null window
    // asks for the desktop's own timing, the only kind Windows 8 and later
    // give.
    unsafe {
        QueryPerformanceFrequency(&mut frequency).ok()?;
        DwmGetCompositionTimingInfo(HWND::default(), &mut timing).ok()?;
    }
    (frequency > 0 && timing.qpcRefreshPeriod > 0)
        .then(|| Duration::from_secs_f64(timing.qpcRefreshPeriod as f64 / frequency as f64))
}
