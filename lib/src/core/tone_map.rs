//! HDR video brought into SDR BT.709 RGB — the one definition
//! `D3d11ToneMap`'s shader and `CudaConverter`'s `p010_to_bgra` kernel both
//! evaluate, and [`ToneMap::apply`] evaluates in Rust for a test to state
//! what a colour should come out as.
//!
//! # Why this crate does it at all
//!
//! A D3D11 video processor would be the obvious place, and does not: the RTX
//! 3050's reports no conversion from a PQ or HLG colour space to SDR RGB,
//! PQ only as far as linear FP16, and HLG not at all. `scale_cuda` has no
//! notion of transfer functions. So both backends do it in their own code,
//! from these numbers.
//!
//! # The steps
//!
//! 1. Y'CbCr to R'G'B' by BT.2020's matrix and the frame's range — the rows
//!    [`crate::color::yuv_to_rgb_rows`] makes — clipped to 0..1.
//! 2. R'G'B' to light, in nits. PQ by SMPTE ST 2084's EOTF, which is
//!    absolute. HLG by BT.2100's inverse OETF and its OOTF for a
//!    1000-nit display (system gamma 1.2), which puts HLG's reference white,
//!    75% signal, at 203 nits as BT.2408 has it.
//! 3. BT.2020's primaries into BT.709's, in that linear light, negatives
//!    clipped.
//! 4. BT.2390's EETF on the largest of the three channels, in PQ's own
//!    domain, from the content's peak down to 203 nits, the three scaled by
//!    what it did to that one — so a colour keeps its hue as it is brought
//!    down, where a curve per channel would pull it towards white.
//! 5. 203 nits is SDR's 1.0 (BT.2408's HDR reference white), clipped, and
//!    encoded with gamma 2.2 — the transfer the rest of this crate's colour
//!    conversions assume.
//!
//! The content's peak is taken as 1000 nits, what HDR10 is most often
//! mastered to, and HLG's is 1000 by its own definition. A stream's
//! MaxCLL and mastering display metadata are not read: FFmpeg's Rust
//! bindings do not carry their structs, and mirroring an FFmpeg struct by
//! hand is what this crate learned not to do (see `AGENTS.md`).

/// SDR's 1.0, in nits — BT.2408's reference white for HDR.
pub(crate) const SDR_WHITE_NITS: f32 = 203.0;

/// The content peak assumed where a PQ stream says nothing, and the display
/// peak HLG's OOTF is evaluated for.
pub(crate) const DEFAULT_PEAK_NITS: f32 = 1000.0;

/// Which HDR transfer a stream's R'G'B' is encoded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HdrTransfer {
    /// SMPTE ST 2084, absolute up to 10000 nits.
    Pq,
    /// ARIB STD-B67 / BT.2100 hybrid log-gamma, relative to its display.
    Hlg,
}

impl HdrTransfer {
    /// The transfer a frame or stream tagged `transfer` is in, where it is
    /// an HDR one.
    pub(crate) fn of(transfer: ffmpeg_next::color::TransferCharacteristic) -> Option<Self> {
        match transfer {
            ffmpeg_next::color::TransferCharacteristic::SMPTE2084 => Some(Self::Pq),
            ffmpeg_next::color::TransferCharacteristic::ARIB_STD_B67 => Some(Self::Hlg),
            _ => None,
        }
    }

    /// The number a shader or kernel is told this by.
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Pq => 1,
            Self::Hlg => 2,
        }
    }
}

const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;

#[cfg(test)]
const HLG_A: f32 = 0.178_832_77;
#[cfg(test)]
const HLG_B: f32 = 1.0 - 4.0 * HLG_A;
// 0.5 - a * ln(4a)
#[cfg(test)]
const HLG_C: f32 = 0.559_910_7;
#[cfg(test)]
const HLG_GAMMA: f32 = 1.2;

#[cfg(test)]
/// ST 2084's EOTF: a PQ signal 0..1 to nits.
pub(crate) fn pq_to_nits(signal: f32) -> f32 {
    let power = signal.max(0.0).powf(1.0 / PQ_M2);
    let numerator = (power - PQ_C1).max(0.0);
    let denominator = PQ_C2 - PQ_C3 * power;
    10000.0 * (numerator / denominator).powf(1.0 / PQ_M1)
}

/// ST 2084's inverse EOTF: nits to a PQ signal 0..1.
pub(crate) fn nits_to_pq(nits: f32) -> f32 {
    let y = (nits / 10000.0).max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)).powf(PQ_M2)
}

#[cfg(test)]
/// BT.2100's HLG inverse OETF: a signal 0..1 to scene light 0..1.
fn hlg_to_scene(signal: f32) -> f32 {
    if signal <= 0.5 {
        signal * signal / 3.0
    } else {
        (((signal - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// What a shader or kernel is handed to bring one stream down: the rows,
/// the transfer, and the EETF's constants, all worked out once here.
///
/// Laid out as a D3D11 constant buffer reads it — each group of four
/// floats one register — and read field by field by the CUDA kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ToneMap {
    /// R'G'B' from normalised `(Y', Cb, Cr, 1)`.
    pub(crate) rows: [[f32; 4]; 3],
    /// [`HdrTransfer::code`].
    pub(crate) transfer: u32,
    /// The content's peak in PQ's domain, the EETF's 1.0.
    pub(crate) source_peak_pq: f32,
    /// SDR white in the EETF's normalised domain — BT.2390's `maxLum`.
    pub(crate) target_peak: f32,
    /// Where the EETF's knee starts — BT.2390's `KS`. At or above 1 there
    /// is nothing to bring down.
    pub(crate) knee: f32,
    /// BT.2020's primaries into BT.709's, one row to a register.
    pub(crate) gamut: [[f32; 4]; 3],
}

impl ToneMap {
    /// For a stream in `transfer`, limited or full `range`, whose content
    /// peaks at `peak_nits` — or, where `None`, at [`DEFAULT_PEAK_NITS`].
    /// HLG ignores `peak_nits`: its display is always the 1000-nit one.
    pub(crate) fn new(
        transfer: HdrTransfer,
        range: ffmpeg_next::color::Range,
        peak_nits: Option<f32>,
    ) -> Self {
        let peak = match transfer {
            HdrTransfer::Pq => peak_nits
                .filter(|peak| peak.is_finite() && *peak > 0.0)
                .unwrap_or(DEFAULT_PEAK_NITS),
            HdrTransfer::Hlg => DEFAULT_PEAK_NITS,
        };
        let source_peak_pq = nits_to_pq(peak);
        let target_peak = nits_to_pq(SDR_WHITE_NITS) / source_peak_pq;
        let gamut = super::color::BT2020_TO_BT709.map(|[r, g, b]| [r, g, b, 0.0]);
        Self {
            rows: super::color::yuv_to_rgb_rows(ffmpeg_next::color::Space::BT2020NCL, range, 1080),
            transfer: transfer.code(),
            source_peak_pq,
            target_peak,
            knee: 1.5 * target_peak - 0.5,
            gamut,
        }
    }

    /// For `frame`, by its own tags — `None` where it is not HDR.
    pub(crate) fn of_frame(frame: &ffmpeg_next::frame::Video) -> Option<Self> {
        let transfer = HdrTransfer::of(frame.color_transfer_characteristic())?;
        Some(Self::new(transfer, frame.color_range(), None))
    }

    #[cfg(test)]
    /// BT.2390's EETF on a PQ signal normalised to the content's peak.
    fn eetf(&self, normalised: f32) -> f32 {
        if normalised < self.knee {
            return normalised;
        }
        let t = (normalised - self.knee) / (1.0 - self.knee);
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.knee
            + (t3 - 2.0 * t2 + t) * (1.0 - self.knee)
            + (-2.0 * t3 + 3.0 * t2) * self.target_peak
    }

    #[cfg(test)]
    /// This module's steps, for one sample: Y', Cb and Cr normalised 0..1
    /// as a shader samples them, to 8-bit SDR BT.709 R, G and B.
    pub(crate) fn apply(&self, luma: f32, cb: f32, cr: f32) -> [u8; 3] {
        let signal = self.rows.map(|[from_y, from_cb, from_cr, offset]| {
            (from_y * luma + from_cb * cb + from_cr * cr + offset).clamp(0.0, 1.0)
        });
        let nits = if self.transfer == HdrTransfer::Pq.code() {
            signal.map(pq_to_nits)
        } else {
            let scene = signal.map(hlg_to_scene);
            let luminance = 0.2627 * scene[0] + 0.6780 * scene[1] + 0.0593 * scene[2];
            let gain = DEFAULT_PEAK_NITS * luminance.max(1e-6).powf(HLG_GAMMA - 1.0);
            scene.map(|channel| gain * channel)
        };
        let bt709 = self
            .gamut
            .map(|[r, g, b, _]| (r * nits[0] + g * nits[1] + b * nits[2]).max(0.0));
        let largest = bt709[0].max(bt709[1]).max(bt709[2]);
        let scale = if largest > 0.0 && self.knee < 1.0 {
            let brought = pq_to_nits(
                self.eetf((nits_to_pq(largest) / self.source_peak_pq).min(1.0))
                    * self.source_peak_pq,
            );
            brought / largest
        } else {
            1.0
        };
        bt709.map(|channel| {
            let sdr = (channel * scale / SDR_WHITE_NITS).clamp(0.0, 1.0);
            (sdr.powf(1.0 / 2.2) * 255.0 + 0.5) as u8
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey at `nits` in PQ, as limited-range Y'CbCr normalised 0..1.
    fn pq_grey(nits: f32) -> (f32, f32, f32) {
        let signal = nits_to_pq(nits);
        (
            (16.0 + 219.0 * signal) / 255.0,
            128.0 / 255.0,
            128.0 / 255.0,
        )
    }

    #[test]
    fn pq_round_trips() {
        for nits in [0.0, 0.5, 100.0, 203.0, 1000.0, 4000.0, 10000.0] {
            let back = pq_to_nits(nits_to_pq(nits));
            assert!(
                (back - nits).abs() <= nits * 1e-3 + 1e-3,
                "{nits} came back {back}"
            );
        }
    }

    /// SDR white, black and a grey below the knee come through as they are;
    /// the content's own peak comes out white; and nothing brighter than
    /// SDR white is left to clip on its own.
    #[test]
    fn pq_greys_land_where_the_definition_says() {
        let map = ToneMap::new(HdrTransfer::Pq, ffmpeg_next::color::Range::MPEG, None);
        let at = |nits| {
            let (y, cb, cr) = pq_grey(nits);
            map.apply(y, cb, cr)
        };
        assert_eq!(at(0.0), [0, 0, 0]);
        let dim = at(20.0)[0];
        let want = ((20.0f32 / SDR_WHITE_NITS).powf(1.0 / 2.2) * 255.0).round() as u8;
        assert!(
            dim.abs_diff(want) <= 2,
            "20 nits came out {dim}, want {want}"
        );
        assert!(
            at(1000.0).iter().all(|channel| *channel >= 253),
            "the peak is white"
        );
        let (mid, top) = (at(400.0)[0], at(1000.0)[0]);
        assert!(
            mid < top,
            "400 nits ({mid}) is not brought all the way to white"
        );
        assert!(at(203.0)[0] < mid, "the knee compresses rather than clips");
    }

    /// A brighter peak compresses harder, so the same light comes out
    /// darker; a peak at or under SDR white leaves everything as it is.
    #[test]
    fn the_peak_decides_how_hard_it_compresses() {
        let (y, cb, cr) = pq_grey(500.0);
        let bright = ToneMap::new(
            HdrTransfer::Pq,
            ffmpeg_next::color::Range::MPEG,
            Some(4000.0),
        );
        let dim = ToneMap::new(
            HdrTransfer::Pq,
            ffmpeg_next::color::Range::MPEG,
            Some(600.0),
        );
        assert!(bright.apply(y, cb, cr)[0] < dim.apply(y, cb, cr)[0]);
        let sdr = ToneMap::new(
            HdrTransfer::Pq,
            ffmpeg_next::color::Range::MPEG,
            Some(200.0),
        );
        assert!(sdr.knee >= 1.0, "nothing to bring down");
    }

    /// HLG's reference white — 75% signal — is 203 nits, which lands a
    /// little under SDR white: BT.2390's knee, bringing 1000 nits down to
    /// 203, starts compressing at about 94.
    #[test]
    fn hlg_reference_white_is_near_sdr_white() {
        let map = ToneMap::new(HdrTransfer::Hlg, ffmpeg_next::color::Range::MPEG, None);
        let signal = 0.75f32;
        let white = map.apply(
            (16.0 + 219.0 * signal) / 255.0,
            128.0 / 255.0,
            128.0 / 255.0,
        );
        assert!(
            white.iter().all(|channel| (220..=245).contains(channel)),
            "{white:?}"
        );
    }
}
