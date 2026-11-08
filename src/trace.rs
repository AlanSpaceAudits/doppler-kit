//! Received frequency against time from an IQ or audio recording (port of iq_to_trace.py).
//!
//! 1. Decimate to about 48 kS/s (the scipy `resample_poly` FIR: Kaiser beta 5, 20*dec+1 taps), remove the DC
//!    line (10 s running mean).
//! 2. Pass A: 0.25 s spectra, median-normalised, fixed receiver lines masked, 300 Hz boxcar peak per frame,
//!    a robust S-curve fit through the detections (steers the search only).
//! 3. Dechirp with the current reference, 1.0 s Hann windows every 0.25 s on a 1 Hz grid, search +-400,
//!    +-100, then +-150 Hz with a better reference each time. Final value: noise-subtracted power centroid.
//! 4. Keep a point if its peak is > 14 dB over the frame median, it is off the fixed receiver lines and
//!    within 4 sigma of a smoothing spline (sigma from second differences).
//!
//! The smoothing spline is the natural cubic smoothing spline (Reinsch form, Green & Silverman 1994),
//! the same minimiser as scipy's `make_smoothing_spline` with the same lambda scale; lambda = None is chosen
//! by GCV with the same bounded Brent search (0, n), xatol 1e-5. The trace of the hat matrix comes from the
//! Hutchinson & de Hoog band inverse.

use crate::input::{self, BlockReader, RawFormat};
use rustfft::num_complex::{Complex, Complex32};
use rustfft::FftPlanner;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const TARGET: f64 = 48000.0;
const EDGE_S: f64 = 1.0;
const COARSE_THR_DB: f64 = 3.0;
const A_THR_DB: f64 = 15.0;
const C_THR_DB: f64 = 14.0;
const SEARCH_B: f64 = 400.0;
const SEARCH_C: f64 = 100.0;
const SEARCH_D: f64 = 150.0;
const SPUR_HALF: f64 = 40.0;
const CLIP_K: f64 = 4.0;
const BURST_FRAME_S: f64 = 0.5;
const BURST_SMOOTH_HZ: f64 = 100.0;
const BURST_MEDIAN_S: f64 = 20.0;
const BURST_BAND_HZ: f64 = 2000.0;
const BURST_THR_DB: f64 = 6.0;
const BURST_FSCALE_HZ: f64 = 300.0;
const BURST_GATE_HZ: f64 = 800.0;
pub const NOT_FOUND: &str = "signal not found: check --center, --span and --swap-iq";
pub const WINDOW_NOTE: &str = "1.0 s Hann, 0.25 s hop (only every 4th point is independent)";
pub const BURST_WINDOW_NOTE: &str = "0.5 s Hann frames, no overlap";
pub const NO_BURSTS: &str = "no bursts found: check --center, --span and --band";

#[derive(Clone, Debug)]
pub struct TraceOpts {
    pub center: Option<f64>,
    pub start: Option<String>,
    pub span: f64,
    pub rate: Option<f64>,
    pub format: RawFormat,
    pub swap_iq: bool,
    pub out: Option<PathBuf>,
    /// Tone band (lo, hi) in Hz: audio frequencies for audio, offsets from the centre for IQ.
    pub band: Option<(f64, f64)>,
    /// Short-burst mode (FSK/packet satellites).
    pub burst: bool,
}

impl Default for TraceOpts {
    fn default() -> Self {
        Self { center: None, start: None, span: 15000.0, rate: None, format: RawFormat::Cs16, swap_iq: false, out: None, band: None, burst: false }
    }
}

#[derive(Clone, Debug)]
pub struct TracePoint {
    pub t_s: f64,
    pub utc: Option<String>,
    pub f_offset: f64,
    pub f: f64,
    pub snr: f64,
}

/// 0.1 s spectra every 0.5 s, dB over the per-bin and per-row medians. `db[i * f.len() + j]` is time i, offset j.
#[derive(Clone, Debug, Default)]
pub struct Waterfall {
    pub t: Vec<f64>,
    pub f: Vec<f64>,
    pub db: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TraceResult {
    pub name: String,
    pub file: String,
    pub points: Vec<TracePoint>,
    pub centre_hz: f64,
    pub audio_tone_hz: Option<f64>,
    pub band_hz: Option<(f64, f64)>,
    pub burst: bool,
    pub sample_rate: u32,
    pub decimated_rate: f64,
    pub duration_s: f64,
    pub start_utc: Option<String>,
    pub point_noise_hz: f64,
    pub point_noise_pairs: usize,
    pub swing_hz: f64,
    pub fixed_receiver_lines_hz: Vec<f64>,
    pub window: String,
    pub note: Option<String>,
    /// Suggests --burst when an IQ trace covers little of the recording or hardly moves (not in the JSON file).
    pub burst_hint: Option<String>,
    pub waterfall: Waterfall,
    pub csv_path: PathBuf,
    pub summary_path: PathBuf,
}

// ---------- small numerics ----------
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let (_, &mut hi, _) = v.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

fn median_of(it: impl Iterator<Item = f64>) -> f64 {
    median(&mut it.collect::<Vec<_>>())
}

fn mad(r: &[f64]) -> f64 {
    let m = median_of(r.iter().copied());
    1.4826 * median_of(r.iter().map(|x| (x - m).abs()))
}

fn argmax<T: PartialOrd + Copy>(v: &[T]) -> usize {
    let mut j = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[j] {
            j = i;
        }
    }
    j
}

fn searchsorted(f: &[f64], x: f64) -> usize {
    f.partition_point(|&v| v < x)
}

fn nthreads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
}

/// Run `f(i)` for i in 0..n on worker threads, results in order.
fn par_map<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let nt = nthreads().min(n.max(1));
    let per = n.div_ceil(nt);
    let f = &f;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt).map(|k| s.spawn(move || (k * per..((k + 1) * per).min(n)).map(f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn bessel_i0(x: f64) -> f64 {
    let (mut s, mut t, mut k) = (1.0, 1.0, 1.0);
    while t > 1e-17 * s {
        t *= (x / (2.0 * k)).powi(2);
        s += t;
        k += 1.0;
    }
    s
}

/// scipy firwin(2*half+1, 1/max(up,down), window=('kaiser', 5.0)) * up, as used by resample_poly.
fn resample_taps(up: usize, down: usize) -> (Vec<f64>, usize) {
    let mx = up.max(down);
    let half = 10 * mx;
    let l = 2 * half + 1;
    let fc = 1.0 / mx as f64;
    let i0b = bessel_i0(5.0);
    let mut h: Vec<f64> = (0..l)
        .map(|m| {
            let x = m as f64 - half as f64;
            let arg = std::f64::consts::PI * fc * x;
            let sinc = if x == 0.0 { 1.0 } else { arg.sin() / arg };
            let r = 2.0 * m as f64 / (l - 1) as f64 - 1.0;
            fc * sinc * bessel_i0(5.0 * (1.0 - r * r).max(0.0).sqrt()) / i0b
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v *= up as f64 / s);
    (h, half)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Polyphase resampling by up/down with zero padding outside the record, streamed from `rd`:
/// y[k] = sum_m h[m] x_up[k*down + half - m]. Returns `n_out` samples.
pub fn resample(rd: &mut dyn BlockReader, n_in: u64, up: usize, down: usize, n_out: usize) -> Result<Vec<Complex32>, String> {
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        let mut v = Vec::with_capacity(n_in as usize);
        while rd.read(&mut v, 1 << 20)? > 0 {}
        v.truncate(n_out);
        return Ok(v);
    }
    let (h, half) = resample_taps(up, down);
    let l = h.len() as i64;
    let (up_i, down_i, half_i, n_in_i) = (up as i64, down as i64, half as i64, n_in as i64);
    let mut out = Vec::with_capacity(n_out);
    let mut buf: Vec<Complex32> = Vec::new();
    let mut base: i64 = 0; // input index of buf[0]
    let mut eof = false;
    // about 4 M input samples per block, so a high-rate recording does not pull gigabytes into one block
    let block = ((1usize << 22) / down).clamp(256, 1 << 16);
    let mut k0 = 0usize;
    while k0 < n_out {
        let k1 = (k0 + block).min(n_out);
        let lo = ((k0 as i64 * down_i + half_i - l + 1).max(0) + up_i - 1) / up_i;
        let hi = ((k1 as i64 - 1) * down_i + half_i).div_euclid(up_i).min(n_in_i - 1);
        let drop = (lo - base).clamp(0, buf.len() as i64) as usize;
        buf.drain(..drop);
        base += drop as i64;
        while !eof && base + (buf.len() as i64) <= hi {
            let want = (hi + 1 - base - buf.len() as i64).max(1 << 16) as usize;
            eof = rd.read(&mut buf, want)? == 0;
        }
        let (bufr, hr) = (&buf, &h);
        let ys = par_map(k1 - k0, |q| {
            let jc = (k0 + q) as i64 * down_i + half_i;
            let i_lo = ((jc - l + 1).max(0) + up_i - 1) / up_i;
            let i_hi = (jc / up_i).min(base + bufr.len() as i64 - 1);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            let mut i = i_lo.max(base);
            while i <= i_hi {
                let w = hr[(jc - i * up_i) as usize];
                let x = bufr[(i - base) as usize];
                re += w * x.re as f64;
                im += w * x.im as f64;
                i += 1;
            }
            Complex32::new(re as f32, im as f32)
        });
        out.extend(ys);
        k0 = k1;
    }
    Ok(out)
}

/// In place. The two running sums are the cumsum values c[b] and c[a] bit for bit, without an n-long array;
/// the ring keeps the last h+1 originals that the trailing sum still needs.
fn remove_dc(y: &mut [Complex32], h: usize) {
    let n = y.len();
    let c64 = |v: Complex32| Complex::<f64>::new(v.re as f64, v.im as f64);
    let (mut lead, mut trail) = (Complex::<f64>::new(0.0, 0.0), Complex::<f64>::new(0.0, 0.0));
    let (mut nb, mut na) = (0usize, 0usize);
    let mut ring = vec![Complex32::new(0.0, 0.0); h + 1];
    for i in 0..n {
        let (a, b) = (i.saturating_sub(h), (i + h + 1).min(n));
        while nb < b {
            lead += c64(y[nb]);
            nb += 1;
        }
        while na < a {
            trail += c64(ring[na % (h + 1)]);
            na += 1;
        }
        ring[i % (h + 1)] = y[i];
        let m = (lead - trail) / (b - a) as f64;
        y[i] = Complex32::new((y[i].re as f64 - m.re) as f32, (y[i].im as f64 - m.im) as f32);
    }
}

// ---------- spectra ----------
struct Grid {
    t: Vec<f64>,
    f: Vec<f64>,
}

/// Hann-windowed spectra of frames i*hop..i*hop+win (nfft = win), bins with |f| <= span, power as f32.
/// `per` maps (frame, power row) to the stored result.
fn stft<R: Send>(y: &[Complex32], fsd: f64, win: usize, hop: usize, span: f64, per: impl Fn(usize, &[f32]) -> R + Sync) -> (Grid, Vec<R>) {
    let nfr = if win > 0 && hop > 0 && y.len() >= win { (y.len() - win) / hop + 1 } else { 0 };
    let n = win;
    let val = 1.0 / (n as f64 * (1.0 / fsd));
    let fall: Vec<f64> = (0..n).map(|k| (k as i64 - (n / 2) as i64) as f64 * val).collect();
    let keep: Vec<usize> = (0..n).filter(|&k| fall[k].abs() <= span).collect();
    let w: Vec<f32> = (0..win)
        .map(|k| if win == 1 { 1.0 } else { (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (win - 1) as f64).cos()) as f32 })
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
    let nt = nthreads();
    let per_t = nfr.div_ceil(nt.max(1)).max(1);
    let (fft, w, keep, per) = (&fft, &w, &keep, &per);
    let rows: Vec<R> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nfr.div_ceil(per_t))
            .map(|c| {
                s.spawn(move || {
                    let mut buf = vec![Complex::<f64>::new(0.0, 0.0); n];
                    let mut scratch = vec![Complex::<f64>::new(0.0, 0.0); fft.get_inplace_scratch_len()];
                    let mut row = vec![0f32; keep.len()];
                    let mut res = Vec::new();
                    for i in c * per_t..((c + 1) * per_t).min(nfr) {
                        let seg = &y[i * hop..i * hop + win];
                        for k in 0..n {
                            let v = seg[k] * w[k];
                            buf[k] = Complex::new(v.re as f64, v.im as f64);
                        }
                        fft.process_with_scratch(&mut buf, &mut scratch);
                        for (o, &k) in row.iter_mut().zip(keep.iter()) {
                            let x = buf[(k + n - n / 2) % n];
                            *o = (x.re * x.re + x.im * x.im) as f32;
                        }
                        res.push(per(i, &row));
                    }
                    res
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let t = (0..nfr).map(|i| (i as f64 * hop as f64 + win as f64 / 2.0) / fsd).collect();
    (Grid { t, f: keep.iter().map(|&k| fall[k]).collect() }, rows)
}

/// R = P / time-median per bin / frame-median per row.
fn normalise(rows: &mut [Vec<f32>]) {
    if rows.is_empty() {
        return;
    }
    let nf = rows[0].len();
    let colmed: Vec<f32> = (0..nf).map(|j| median_of(rows.iter().map(|r| r[j] as f64)) as f32).collect();
    for r in rows.iter_mut() {
        for (v, m) in r.iter_mut().zip(&colmed) {
            *v /= m;
        }
        let rm = median_of(r.iter().map(|&v| v as f64)) as f32;
        r.iter_mut().for_each(|v| *v /= rm);
    }
}

/// Receiver spurs: bins whose time-median power stands > 6 dB over the local floor.
fn fixed_lines(f: &[f64], rows: &[Vec<f32>]) -> Vec<f64> {
    let nf = f.len();
    // numpy keeps float32 P in float32 through median, log10 and the 6 dB test
    let m: Vec<f32> = (0..nf).map(|j| 10.0 * (median_of(rows.iter().map(|r| r[j] as f64)) as f32 + 1e-30).log10()).collect();
    let size = 5usize.max((2000.0 / (f[1] - f[0])) as usize);
    let (lo, rank) = (size / 2, size / 2); // scipy median_filter: offsets -size/2.., rank size/2 (upper median)
    let mut win = vec![0f32; size];
    let mut lines = Vec::new();
    let mut grp: Vec<usize> = Vec::new();
    for j in 0..nf {
        for (q, w) in win.iter_mut().enumerate() {
            *w = m[(j as i64 + q as i64 - lo as i64).clamp(0, nf as i64 - 1) as usize];
        }
        let (_, &mut base, _) = win.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
        if m[j] - base > 6.0 {
            if let Some(&last) = grp.last() {
                if j - last > 2 {
                    lines.push(grp.iter().map(|&g| f[g]).sum::<f64>() / grp.len() as f64);
                    grp.clear();
                }
            }
            grp.push(j);
        }
    }
    if !grp.is_empty() {
        lines.push(grp.iter().map(|&g| f[g]).sum::<f64>() / grp.len() as f64);
    }
    lines
}

fn near(x: f64, lines: &[f64]) -> bool {
    lines.iter().map(|l| (x - l).abs()).fold(f64::INFINITY, f64::min) < SPUR_HALF
}

fn peak_interp(row: &[f32], j: usize) -> f64 {
    if j == 0 || j + 1 >= row.len() {
        return 0.0;
    }
    let l = |v: f32| (v as f64 + 1e-30).ln();
    let (a, b, c) = (l(row[j - 1]), l(row[j]), l(row[j + 1]));
    let den = a - 2.0 * b + c;
    if den != 0.0 {
        0.5 * (a - c) / den
    } else {
        0.0
    }
}

fn uniform_filter_reflect(x: &[f32], size: usize) -> Vec<f32> {
    let n = x.len() as i64;
    let refl = |mut i: i64| -> usize {
        let p = 2 * n;
        i = i.rem_euclid(p);
        (if i >= n { p - 1 - i } else { i }) as usize
    };
    let (s1, s2) = ((size / 2) as i64, (size - size / 2 - 1) as i64);
    let mut acc: f64 = (-s1..=s2).map(|o| x[refl(o)] as f64).sum();
    let mut out = Vec::with_capacity(x.len());
    for i in 0..n {
        out.push((acc / size as f64) as f32);
        acc += x[refl(i + s2 + 1)] as f64 - x[refl(i - s1)] as f64;
    }
    out
}

// ---------- smoothing spline ----------
#[derive(Clone, Debug)]
pub struct Spline {
    x: Vec<f64>,
    g: Vec<f64>,
    gam: Vec<f64>,
}

impl Spline {
    /// Natural cubic smoothing spline minimising sum (y - f)^2 + lam * int f''^2; lam None = GCV.
    pub fn fit(x: &[f64], y: &[f64], lam: Option<f64>) -> Result<Spline, String> {
        let n = x.len();
        if n < 5 {
            return Err("too few points for the smoothing spline".into());
        }
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        if h.iter().any(|&v| v <= 0.0) {
            return Err("spline abscissas must be ascending".into());
        }
        let m = n - 2;
        // Q columns a = 0..m: rows a, a+1, a+2
        let q: Vec<[f64; 3]> = (0..m).map(|a| [1.0 / h[a], -1.0 / h[a] - 1.0 / h[a + 1], 1.0 / h[a + 1]]).collect();
        // QtQ bands
        let qq0: Vec<f64> = (0..m).map(|a| q[a].iter().map(|v| v * v).sum()).collect();
        let qq1: Vec<f64> = (0..m.saturating_sub(1)).map(|a| q[a][1] * q[a + 1][0] + q[a][2] * q[a + 1][1]).collect();
        let qq2: Vec<f64> = (0..m.saturating_sub(2)).map(|a| q[a][2] * q[a + 2][0]).collect();
        let r0: Vec<f64> = (0..m).map(|a| (h[a] + h[a + 1]) / 3.0).collect();
        let r1: Vec<f64> = (0..m.saturating_sub(1)).map(|a| h[a + 1] / 6.0).collect();
        let qty: Vec<f64> = (0..m).map(|a| q[a][0] * y[a] + q[a][1] * y[a + 1] + q[a][2] * y[a + 2]).collect();
        let solve = |lam: f64, want_tr: bool| -> Result<(Vec<f64>, Vec<f64>, f64), String> {
            // LDL^T of M = R + lam QtQ (pentadiagonal)
            let mut d = vec![0.0; m];
            let mut l1 = vec![0.0; m];
            let mut l2 = vec![0.0; m];
            for i in 0..m {
                let mut di = r0[i] + lam * qq0[i];
                if i >= 1 {
                    di -= l1[i - 1] * l1[i - 1] * d[i - 1];
                }
                if i >= 2 {
                    di -= l2[i - 2] * l2[i - 2] * d[i - 2];
                }
                if di <= 0.0 || !di.is_finite() {
                    return Err("Seems like the problem is ill-posed".into());
                }
                d[i] = di;
                if i + 1 < m {
                    let mut v = r1[i] + lam * qq1[i];
                    if i >= 1 {
                        v -= l2[i - 1] * l1[i - 1] * d[i - 1];
                    }
                    l1[i] = v / di;
                }
                if i + 2 < m {
                    l2[i] = lam * qq2[i] / di;
                }
            }
            let mut z = qty.clone();
            for i in 0..m {
                if i >= 1 {
                    z[i] -= l1[i - 1] * z[i - 1];
                }
                if i >= 2 {
                    z[i] -= l2[i - 2] * z[i - 2];
                }
            }
            for i in 0..m {
                z[i] /= d[i];
            }
            for i in (0..m).rev() {
                if i + 1 < m {
                    z[i] -= l1[i] * z[i + 1];
                }
                if i + 2 < m {
                    z[i] -= l2[i] * z[i + 2];
                }
            }
            let mut g = y.to_vec();
            for a in 0..m {
                for k in 0..3 {
                    g[a + k] -= lam * q[a][k] * z[a];
                }
            }
            let mut tr = 0.0;
            if want_tr {
                // central bands of M^-1 (Hutchinson & de Hoog), then tr(I - A) = lam * sum S_ab (QtQ)_ab
                let (mut s0, mut s1, mut s2) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
                for i in (0..m).rev() {
                    let a1 = if i + 1 < m { l1[i] } else { 0.0 };
                    let a2 = if i + 2 < m { l2[i] } else { 0.0 };
                    let g11 = if i + 1 < m { s0[i + 1] } else { 0.0 };
                    let g22 = if i + 2 < m { s0[i + 2] } else { 0.0 };
                    let g12 = if i + 2 < m { s1[i + 1] } else { 0.0 };
                    if i + 2 < m {
                        s2[i] = -a1 * g12 - a2 * g22;
                    }
                    if i + 1 < m {
                        s1[i] = -a1 * g11 - a2 * g12;
                    }
                    s0[i] = 1.0 / d[i] - a1 * s1[i] - a2 * s2[i];
                }
                let mut acc = 0.0;
                for a in 0..m {
                    acc += s0[a] * qq0[a];
                    if a + 1 < m {
                        acc += 2.0 * s1[a] * qq1[a];
                    }
                    if a + 2 < m {
                        acc += 2.0 * s2[a] * qq2[a];
                    }
                }
                tr = lam * acc;
            }
            let mut gam = vec![0.0; n];
            gam[1..n - 1].copy_from_slice(&z);
            Ok((g, gam, tr))
        };
        let lam = match lam {
            Some(l) => l,
            None => {
                let nf = n as f64;
                let gcv = |l: f64| -> f64 {
                    match solve(l, true) {
                        Ok((g, _, tr_ia)) => {
                            let rss: f64 = g.iter().zip(y).map(|(a, b)| (b - a) * (b - a)).sum();
                            (rss / nf) / (tr_ia / nf).powi(2)
                        }
                        Err(_) => f64::NAN,
                    }
                };
                fminbound(gcv, 0.0, nf, 1e-5, 500)
            }
        };
        let (g, gam, _) = solve(lam, false)?;
        Ok(Spline { x: x.to_vec(), g, gam })
    }

    /// Value at t; outside the data the end cubic pieces continue (as scipy's BSpline extrapolates).
    pub fn eval(&self, t: f64) -> f64 {
        let n = self.x.len();
        let i = self.x.partition_point(|&v| v <= t).saturating_sub(1).min(n - 2);
        let (x0, x1) = (self.x[i], self.x[i + 1]);
        let h = x1 - x0;
        let (a, b) = (t - x0, x1 - t);
        (a * self.g[i + 1] + b * self.g[i]) / h - a * b / 6.0 * ((1.0 + a / h) * self.gam[i + 1] + (1.0 + b / h) * self.gam[i])
    }
}

/// scipy.optimize.minimize_scalar(method='bounded') (Brent's fminbound), same steps and stopping rule.
fn fminbound(f: impl Fn(f64) -> f64, x1: f64, x2: f64, xatol: f64, maxfun: usize) -> f64 {
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden = 0.5 * (3.0 - 5f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut gold = true;
        if e.abs() > tol1 {
            gold = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                gold = true;
            }
        }
        if gold {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            break;
        }
    }
    xf
}

/// Iterated k-sigma clipping around a smoothing spline. sigma None: GCV spline, clip at k * max(MAD, 2 Hz).
/// sigma given: the smoothest spline (lambda 1e-8..1e6, 57 steps) whose residual MAD stays <= sigma.
fn robust_spline(t: &[f64], f: &[f64], k: f64, sigma: Option<f64>) -> Result<(Spline, Vec<bool>), String> {
    let fit = |a: &[f64], b: &[f64]| -> Result<Spline, String> {
        let Some(sig) = sigma else { return Spline::fit(a, b, None) };
        let mut best = Spline::fit(a, b, Some(1e-8))?;
        for i in 0..57 {
            let lam = if i == 56 { 1e6 } else { 10f64.powf(-8.0 + i as f64 * 0.25) };
            let sp = Spline::fit(a, b, Some(lam))?;
            let r: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - sp.eval(*x)).collect();
            if mad(&r) > sig {
                break;
            }
            best = sp;
        }
        Ok(best)
    };
    let sub = |ok: &[bool]| -> (Vec<f64>, Vec<f64>) {
        (0..t.len()).filter(|&i| ok[i]).map(|i| (t[i], f[i])).unzip()
    };
    let mut ok = vec![true; t.len()];
    for _ in 0..10 {
        let (a, b) = sub(&ok);
        let sp = fit(&a, &b)?;
        let r: Vec<f64> = t.iter().zip(f).map(|(x, y)| y - sp.eval(*x)).collect();
        let rk: Vec<f64> = (0..t.len()).filter(|&i| ok[i]).map(|i| r[i]).collect();
        let scale = match sigma {
            None => mad(&rk).max(2.0),
            Some(s) if s.is_nan() => f64::NAN,
            Some(s) => s.max(2.0),
        };
        let new: Vec<bool> = r.iter().map(|v| v.abs() < k * scale).collect();
        if new == ok {
            break;
        }
        ok = new;
    }
    let (a, b) = sub(&ok);
    Ok((fit(&a, &b)?, ok))
}

/// Point noise from second differences at 1 s spacing: 1.4826 MAD / sqrt(6), and the number of differences.
fn second_diff_sigma(t: &[f64], f: &[f64]) -> (f64, usize) {
    let key = |x: f64| (x * 8.0).round_ties_even() as i64;
    let lut: std::collections::HashMap<i64, f64> = t.iter().zip(f).map(|(&x, &v)| (key(x), v)).collect();
    let d2: Vec<f64> = t
        .iter()
        .zip(f)
        .filter_map(|(&x, &v)| Some(v - 2.0 * lut.get(&key(x + 1.0))? + lut.get(&key(x + 2.0))?))
        .collect();
    let s = if d2.len() > 5 { mad(&d2) / 6f64.sqrt() } else { f64::NAN };
    (s, d2.len())
}

// ---------- S-curve model ----------
pub fn pass_model(t: f64, p: &[f64; 4]) -> f64 {
    let u = t - p[2];
    p[0] - p[1] * u / (p[3] * p[3] + u * u).sqrt()
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for c in 0..4 {
        let p = (c..4).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..4 {
            let m = a[r][c] / a[c][c];
            for k in c..4 {
                a[r][k] -= m * a[c][k];
            }
            b[r] -= m * b[c];
        }
    }
    let mut x = [0.0; 4];
    for c in (0..4).rev() {
        x[c] = (b[c] - (c + 1..4).map(|k| a[c][k] * x[k]).sum::<f64>()) / a[c][c];
    }
    Some(x)
}

/// Robust (soft_l1, f_scale 200 Hz) bounded fit of the S-curve, like scipy least_squares(loss="soft_l1").
/// Levenberg-Marquardt on the soft_l1 cost with IRLS weights and projection onto the bounds.
fn fit_model(t: &[f64], f: &[f64], span: f64) -> [f64; 4] {
    let med = median_of(f.iter().copied());
    let tm = t[argmax(&f.iter().map(|v| -(v - med).abs()).collect::<Vec<_>>())];
    let (fmin, fmax) = f.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let (tmin, tmax) = (t[0].min(t[t.len() - 1]), t[0].max(t[t.len() - 1]));
    let lo = [-span, 10.0, tmin - 300.0, 2.0];
    let hi = [span, 2.0 * span, tmax + 300.0, 2000.0];
    soft_l1_fit(t, f, [med.clamp(-span * 0.9, span * 0.9), ((fmax - fmin) / 2.0).max(100.0), tm, 60.0], lo, hi, 200.0).0
}

/// Bounded soft_l1 fit of the S-curve (scale `C` Hz) from `p0`. Returns the parameters and the summed soft_l1 cost.
fn soft_l1_fit(t: &[f64], f: &[f64], p0: [f64; 4], lo: [f64; 4], hi: [f64; 4], #[allow(non_snake_case)] C: f64) -> ([f64; 4], f64) {
    let clampp = |p: [f64; 4]| -> [f64; 4] { std::array::from_fn(|i| p[i].clamp(lo[i], hi[i])) };
    let mut p = clampp(p0);
    let cost = |p: &[f64; 4]| -> f64 {
        t.iter().zip(f).map(|(&x, &y)| {
            let z = ((pass_model(x, p) - y) / C).powi(2);
            C * C * ((1.0 + z).sqrt() - 1.0)
        }).sum()
    };
    let mut c0 = cost(&p);
    let mut mu = 1e-3;
    for _ in 0..500 {
        let mut hm = [[0.0; 4]; 4];
        let mut g = [0.0; 4];
        for (&x, &y) in t.iter().zip(f) {
            let u = x - p[2];
            let s = (p[3] * p[3] + u * u).sqrt();
            let r = p[0] - p[1] * u / s - y;
            let w = 1.0 / (1.0 + (r / C).powi(2)).sqrt();
            let s3 = s * s * s;
            let j = [1.0, -u / s, p[1] * p[3] * p[3] / s3, p[1] * u * p[3] / s3];
            for a in 0..4 {
                g[a] += w * r * j[a];
                for b in 0..4 {
                    hm[a][b] += w * j[a] * j[b];
                }
            }
        }
        // Active set: a parameter on a bound with the gradient pointing outward stays out of the step,
        // otherwise the clamped step does not lower the cost and the loop stops early.
        let pin: [bool; 4] = std::array::from_fn(|i| (p[i] <= lo[i] && g[i] > 0.0) || (p[i] >= hi[i] && g[i] < 0.0));
        let mut improved = false;
        for _ in 0..30 {
            let mut a = hm;
            for i in 0..4 {
                a[i][i] += mu * hm[i][i].max(1e-12);
            }
            for i in (0..4).filter(|&i| pin[i]) {
                for k in 0..4 {
                    a[i][k] = 0.0;
                    a[k][i] = 0.0;
                }
                a[i][i] = 1.0;
            }
            let rhs: [f64; 4] = std::array::from_fn(|i| if pin[i] { 0.0 } else { -g[i] });
            let Some(dx) = solve4(a, rhs) else { mu *= 10.0; continue };
            let pn = clampp(std::array::from_fn(|i| p[i] + dx[i]));
            let cn = cost(&pn);
            if cn < c0 {
                let small = (0..4).all(|i| (pn[i] - p[i]).abs() <= 1e-10 * (1.0 + p[i].abs()));
                let rel = (c0 - cn) / c0.max(1e-300);
                p = pn;
                c0 = cn;
                mu = (mu / 3.0).max(1e-12);
                improved = !(small || rel < 1e-15);
                break;
            }
            mu *= 4.0;
        }
        if !improved {
            break;
        }
    }
    (p, c0)
}

/// Soft-L1 fit of the pass shape from 15 x 3 starts (t0 across the trace, width T/30, T/10, T/3); the lowest cost wins.
fn robust_pass(t: &[f64], f: &[f64], tt: f64, span: f64) -> [f64; 4] {
    let (tmin, tmax) = t.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let (fmin, fmax) = f.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let med = median_of(f.iter().copied());
    let lo = [-span, 10.0, tmin - tt, 0.5];
    let hi = [span, 2.0 * span, tmax + tt, 10.0 * tt];
    let step = (tmax - tmin) / 14.0;
    let mut best: Option<([f64; 4], f64)> = None;
    for k in 0..15 {
        let t0 = if k == 14 { tmax } else { k as f64 * step + tmin }; // numpy linspace
        for w in [tt / 30.0, tt / 10.0, tt / 3.0] {
            let r = soft_l1_fit(t, f, [med, ((fmax - fmin) / 2.0).max(100.0), t0, w.max(1.0)], lo, hi, BURST_FSCALE_HZ);
            if best.is_none_or(|b| r.1 < b.1) {
                best = Some(r);
            }
        }
    }
    best.map(|b| b.0).unwrap_or([0.0; 4])
}

// ---------- tracking ----------
/// dc: offset of the receiver DC line (blanked with the fixed lines), or None (audio with a band: no lines at all).
fn pass_a(y: &[Complex32], fsd: f64, tt: f64, span: f64, dc: Option<f64>) -> Result<([f64; 4], Vec<f64>), String> {
    let win = (fsd / 4.0).round_ties_even() as usize;
    let (gr, mut rows) = stft(y, fsd, win, win / 2, span, |_, r| r.to_vec()); // 4 Hz grid
    if rows.len() < 5 || gr.f.len() < 3 {
        return Err(NOT_FOUND.into());
    }
    let (t, f) = (&gr.t, &gr.f);
    let lines = match dc {
        Some(d) => {
            let mut l = fixed_lines(f, &rows);
            l.push(d);
            l
        }
        None => Vec::new(),
    };
    normalise(&mut rows);
    let masked: Vec<bool> = f.iter().map(|&v| near(v, &lines)).collect();
    for r in rows.iter_mut() {
        for (v, &m) in r.iter_mut().zip(&masked) {
            if m {
                *v = 1.0;
            }
        }
    }
    let df = f[1] - f[0];
    let inside: Vec<bool> = t.iter().map(|&x| x > EDGE_S && x < tt - EDGE_S).collect();
    let size = 3usize.max((300f64.min(span / 2.0) / df) as usize); // narrower in a narrow band
    let coarse: Vec<(f64, f64)> = par_map(rows.len(), |i| {
        let rs = uniform_filter_reflect(&rows[i], size);
        let ci = argmax(&rs);
        let m = median_of(rs.iter().map(|&v| v as f64));
        (f[ci], (10.0 * (rs[ci] / m as f32).log10()) as f64)
    });
    let (ct, cf): (Vec<f64>, Vec<f64>) =
        (0..t.len()).filter(|&i| coarse[i].1 > COARSE_THR_DB && inside[i]).map(|i| (t[i], coarse[i].0)).unzip();
    let (sp, _) = robust_spline(&ct, &cf, 3.0, None).map_err(|_| NOT_FOUND.to_string())?;
    let (mut dt, mut dpk) = (Vec::new(), Vec::new());
    for i in 0..t.len() {
        if !inside[i] {
            continue;
        }
        let r = sp.eval(t[i]);
        let (j0, j1) = (searchsorted(f, r - 600.0), searchsorted(f, r + 600.0));
        if j1 < j0 + 3 {
            continue;
        }
        let j = argmax(&rows[i][j0..j1]) + j0;
        if j > 0 && j + 1 < f.len() && (10.0 * rows[i][j].log10()) as f64 > A_THR_DB {
            dt.push(t[i]);
            dpk.push(f[j] + peak_interp(&rows[i], j) * df);
        }
    }
    if dt.len() <= 20 {
        return Err(NOT_FOUND.into());
    }
    Ok((fit_model(&dt, &dpk, span), lines))
}

struct Pass {
    t: Vec<f64>,
    pk: Vec<f64>,
    snr: Vec<f64>,
    lc: Vec<f64>,
    inside: Vec<bool>,
}

/// cut: the signal was cut to +-span (--band). Bins whose source frequency (bin + reference) lies outside the band
/// are set to 0, the frame floor is the median of the other bins, and there is no per-bin floor.
fn dechirp_pass(y: &[Complex32], fsd: f64, tt: f64, rf: &(dyn Fn(f64) -> f64 + Sync), search: f64, span: f64, cut: bool) -> Pass {
    let mut acc = 0.0f64;
    let tau = 2.0 * std::f64::consts::PI;
    let z: Vec<Complex32> = y
        .iter()
        .enumerate()
        .map(|(i, v)| {
            acc += rf(i as f64 / fsd);
            let (s, c) = (-tau * acc / fsd).sin_cos();
            let (re, im) = (v.re as f64, v.im as f64);
            Complex32::new((re * c - im * s) as f32, (re * s + im * c) as f32)
        })
        .collect();
    let win = fsd.round_ties_even() as usize;
    let val = 1.0 / (win as f64 * (1.0 / fsd));
    let fall = |k: usize| (k as i64 - (win / 2) as i64) as f64 * val;
    let k_lo = (0..win).find(|&k| fall(k).abs() <= span).unwrap_or(0);
    let nk = (0..win).filter(|&k| fall(k).abs() <= span).count();
    let fk: Vec<f64> = (k_lo..k_lo + nk).map(fall).collect();
    let (j0, j1) = (searchsorted(&fk, -search), searchsorted(&fk, search));
    let (b0, b1) = (j0.saturating_sub(1), (j1 + 1).min(nk)); // stored band, with one guard bin each side
    let hop = win / 4;
    let (gr, rows) = stft(&z, fsd, win, hop, span, |i, r| {
        if !cut {
            let m = median_of(r.iter().map(|&v| v as f64)) as f32;
            return (m, r[b0..b1].to_vec());
        }
        let fref = rf((i as f64 * hop as f64 + win as f64 / 2.0) / fsd);
        let out = |k: usize| (fk[k] + fref).abs() > span;
        let mut kept: Vec<f64> = (0..r.len()).filter(|&k| !out(k)).map(|k| r[k] as f64).collect();
        let m = if kept.is_empty() { f32::INFINITY } else { median(&mut kept) as f32 };
        (m, (b0..b1).map(|k| if out(k) { 0.0 } else { r[k] }).collect())
    });
    drop(z);
    let f = &gr.f;
    let df = f[1] - f[0];
    let nbin: Vec<f32> = if cut {
        vec![1.0; j1 - j0]
    } else {
        (j0..j1).map(|j| median_of(rows.iter().map(|(m, r)| (r[j - b0] / m) as f64)) as f32).collect()
    };
    // numpy: one infinite frame floor turns the floor array into float64
    let med64 = rows.iter().any(|(m, _)| m.is_infinite());
    let n = gr.t.len();
    let (mut pk, mut snr, mut lc) = (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    for (i, (m, r)) in rows.iter().enumerate() {
        let fref = rf(gr.t[i]);
        let band = &r[j0 - b0..j1 - b0];
        let j = argmax(band) + j0;
        let jl = j - b0;
        let pi = if jl >= 1 && jl + 1 < r.len() { peak_interp(&r[jl - 1..jl + 2], 1) } else { 0.0 };
        pk[i] = fref + f[j] + pi * df;
        if cut {
            // float64 here: med * nbin (float64 ones) promotes
            snr[i] = if med64 { 10.0 * (r[jl] as f64 / *m as f64).log10() } else { (10.0 * (r[jl] / m).log10()) as f64 };
            let e: Vec<f64> = band.iter().map(|&p| (p as f64 - *m as f64).max(0.0)).collect();
            let se = crate::lm::np_sum(&e);
            if se > 0.0 {
                let ef: Vec<f64> = e.iter().zip(&f[j0..j1]).map(|(&v, &x)| v * x).collect();
                lc[i] = fref + crate::lm::np_sum(&ef) / se;
            }
            continue;
        }
        snr[i] = (10.0 * (r[jl] / m).log10()) as f64;
        let e: Vec<f32> = band.iter().zip(&nbin).map(|(&p, &nb)| (p - m * nb).max(0.0)).collect();
        let se = crate::lm::np_sum(&e);
        if se > 0.0 {
            let ef: Vec<f64> = e.iter().zip(&f[j0..j1]).map(|(&v, &x)| v as f64 * x).collect();
            lc[i] = fref + crate::lm::np_sum(&ef) / se as f64;
        }
    }
    let inside = gr.t.iter().map(|&x| x - 0.5 > EDGE_S && x + 0.5 < tt - EDGE_S).collect();
    Pass { t: gr.t, pk, snr, lc, inside }
}

fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let i = xp.partition_point(|&v| v <= x) - 1;
    fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])
}

fn select(v: &[f64], m: &[bool]) -> Vec<f64> {
    v.iter().zip(m).filter(|(_, &k)| k).map(|(x, _)| *x).collect()
}

/// Output of the tracker on a decimated complex signal.
pub struct Track {
    pub t: Vec<f64>,
    pub lc: Vec<f64>,
    pub snr: Vec<f64>,
    pub acc: Vec<bool>,
    pub sigma: f64,
    pub nsigma: usize,
    pub lines: Vec<f64>,
    /// Frames of the last measuring pass clear of the recording edges.
    pub ninside: usize,
}

/// Track the carrier in a complex baseband signal at `fsd` S/s (offsets from 0 Hz within +-span).
/// dc: offset of the receiver DC line (blanked), or None for audio with a band. cut: the signal was cut to +-span.
pub fn track(y: &[Complex32], fsd: f64, span: f64, dc: Option<f64>, cut: bool) -> Result<Track, String> {
    let tt = y.len() as f64 / fsd;
    let (pa, lines) = pass_a(y, fsd, tt, span, dc)?;
    let det_of = |d: &Pass, lc: bool| -> Vec<bool> {
        (0..d.t.len())
            .map(|i| d.snr[i] > C_THR_DB && d.inside[i] && !near(d.pk[i], &lines) && (!lc || d.lc[i].is_finite()))
            .collect()
    };
    let err = |_| NOT_FOUND.to_string();
    let d = dechirp_pass(y, fsd, tt, &|x| pass_model(x, &pa), SEARCH_B, span, cut);
    let det = det_of(&d, false);
    let (td, pd) = (select(&d.t, &det), select(&d.pk, &det));
    let (_, ok) = robust_spline(&td, &pd, CLIP_K, None).map_err(err)?;
    let (tb, fb) = (select(&td, &ok), select(&pd, &ok));
    if tb.len() < 4 {
        return Err(NOT_FOUND.into());
    }
    let pb = fit_model(&tb, &fb, span);
    let resb: Vec<f64> = tb.iter().zip(&fb).map(|(&x, &v)| v - pass_model(x, &pb)).collect();
    let (mut cx, mut cy) = (Vec::new(), Vec::new());
    let mut e0 = 0.0;
    while e0 < tt + 10.0 {
        let idx: Vec<usize> = (0..tb.len()).filter(|&i| tb[i] >= e0 && tb[i] < e0 + 10.0).collect();
        if idx.len() >= 3 {
            cx.push(idx.iter().map(|&i| tb[i]).sum::<f64>() / idx.len() as f64);
            cy.push(median_of(idx.iter().map(|&i| resb[i])));
        }
        e0 += 10.0;
    }
    if cx.is_empty() {
        return Err(NOT_FOUND.into());
    }
    let ref_c = |x: f64| pass_model(x, &pb) + interp(x, &cx, &cy);
    let d = dechirp_pass(y, fsd, tt, &ref_c, SEARCH_C, span, cut);
    let det = det_of(&d, false);
    let (td, pd) = (select(&d.t, &det), select(&d.pk, &det));
    let (_, ok0) = robust_spline(&td, &pd, CLIP_K, None).map_err(err)?;
    let (sig, _) = second_diff_sigma(&select(&td, &ok0), &select(&pd, &ok0));
    let (sp, ok) = robust_spline(&td, &pd, CLIP_K, Some(sig)).map_err(err)?;
    let tc = select(&td, &ok);
    let (c0, c1) = (tc[0], tc[tc.len() - 1]);
    let ref_d = |x: f64| {
        let xc = x.clamp(c0, c1);
        ref_c(x) + sp.eval(xc) - ref_c(xc)
    };
    let d = dechirp_pass(y, fsd, tt, &ref_d, SEARCH_D, span, cut);
    let det = det_of(&d, true);
    let (td, ld) = (select(&d.t, &det), select(&d.lc, &det));
    let (_, ok0) = robust_spline(&td, &ld, CLIP_K, None).map_err(err)?;
    let (sig, nsig) = second_diff_sigma(&select(&td, &ok0), &select(&ld, &ok0));
    let (_, ok) = robust_spline(&td, &ld, CLIP_K, Some(sig)).map_err(err)?;
    let mut acc = vec![false; d.t.len()];
    for (q, i) in (0..d.t.len()).filter(|&i| det[i]).enumerate() {
        acc[i] = ok[q];
    }
    let ninside = d.inside.iter().filter(|&&v| v).count();
    Ok(Track { t: d.t, lc: d.lc, snr: d.snr, acc, sigma: sig, nsigma: nsig, lines, ninside })
}

/// Running median over `2 * half + 1` samples, edges repeated (scipy median_filter, mode "nearest").
fn running_median(x: &[f32], half: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let at = |i: i64| x[i.clamp(0, n as i64 - 1) as usize];
    let h = half as i64;
    let mut win: Vec<f32> = (-h..=h).map(at).collect();
    win.sort_by(|a, b| a.total_cmp(b));
    let mut out = Vec::with_capacity(n);
    for i in 0..n as i64 {
        out.push(win[half]);
        let old = at(i - h);
        let p = win.partition_point(|v| v.total_cmp(&old).is_lt());
        win.remove(p);
        let new = at(i + h + 1);
        let q = win.partition_point(|v| v.total_cmp(&new).is_lt());
        win.insert(q, new);
    }
    out
}

/// Burst mode: 0.5 s Hann frames, 100 Hz boxcar across frequency, each bin over its running median of +-20 s,
/// excess E = ratio - 1. Strongest 2000 Hz band of E per frame, kept at >= 6 dB; value = centroid of max(E, 0)
/// in that band. Outliers: soft-L1 pass fit from a grid of starts, points more than 800 Hz off it dropped.
pub fn burst_track(y: &[Complex32], fsd: f64, span: f64) -> Result<Track, String> {
    let tt = y.len() as f64 / fsd;
    let win = (BURST_FRAME_S * fsd) as usize;
    let (gr, mut s) = stft(y, fsd, win, win, span, |_, r| r.to_vec());
    let nf = gr.f.len();
    if s.len() < 2 || nf < 2 {
        return Err(NO_BURSTS.into());
    }
    let (t, f) = (&gr.t, &gr.f);
    let df = f[1] - f[0];
    let smooth = 1usize.max((BURST_SMOOTH_HZ / df) as usize);
    s = par_map(s.len(), |i| uniform_filter_reflect(&s[i], smooth));
    let half = (BURST_MEDIAN_S / BURST_FRAME_S) as usize;
    let cols: Vec<Vec<f32>> = par_map(nf, |j| running_median(&s.iter().map(|r| r[j]).collect::<Vec<_>>(), half));
    let e: Vec<Vec<f32>> = (0..s.len()).map(|i| (0..nf).map(|j| s[i][j] / (cols[j][i] + 1e-30f32) - 1.0).collect()).collect();
    drop(cols);
    drop(s);
    let nb = 1usize.max((BURST_BAND_HZ / df) as usize);
    let rows: Vec<(f64, f64)> = par_map(e.len(), |i| {
        let b = uniform_filter_reflect(&e[i], nb);
        let j = argmax(&b);
        // numpy: max(B, 1e-6) keeps the float32 value when it is the larger, else a Python float
        let snr = if b[j] as f64 >= 1e-6 { (10.0 * (b[j] + 1.0).log10()) as f64 } else { 10.0 * (1e-6f64 + 1.0).log10() };
        let (lo, hi) = (j.saturating_sub(nb / 2), nf.min(j + nb / 2 + 1));
        let ev: Vec<f32> = e[i][lo..hi].iter().map(|&v| v.max(0.0)).collect();
        let se = crate::lm::np_sum(&ev);
        let mut lc = f64::NAN;
        if snr >= BURST_THR_DB && se > 0.0 {
            let ef: Vec<f64> = ev.iter().zip(&f[lo..hi]).map(|(&v, &x)| v as f64 * x).collect();
            lc = crate::lm::np_sum(&ef) / se as f64;
        }
        (snr, lc)
    });
    let (snr, lc): (Vec<f64>, Vec<f64>) = rows.into_iter().unzip();
    let det: Vec<bool> = lc.iter().map(|v| v.is_finite()).collect();
    if det.iter().filter(|&&v| v).count() <= 10 {
        return Err(NO_BURSTS.into());
    }
    let (td, ld) = (select(t, &det), select(&lc, &det));
    let p = robust_pass(&td, &ld, tt, span);
    let acc: Vec<bool> = (0..t.len()).map(|i| det[i] && (lc[i] - pass_model(t[i], &p)).abs() < BURST_GATE_HZ).collect();
    let r: Vec<f64> = (0..t.len()).filter(|&i| acc[i]).map(|i| lc[i] - pass_model(t[i], &p)).collect();
    let nsig = r.len();
    Ok(Track { t: gr.t.clone(), lc, snr, acc, sigma: mad(&r), nsigma: nsig, lines: Vec::new(), ninside: 0 })
}

/// Zero everything outside lo..hi in one FFT of the whole signal (audio: also the negative frequencies, x2 keeps
/// the amplitude) and move the band centre to 0 Hz. Returns the signal and the centre (Hz). The cut comes first:
/// the dechirp passes shift the spectrum by up to the swing and would pull outside sound into the search window.
pub fn band_shift(y: &[Complex32], fsd: f64, band: (f64, f64), audio: bool) -> (Vec<Complex32>, f64) {
    let n = y.len();
    let mut a: Vec<Complex<f64>> = y.iter().map(|v| Complex::new(v.re as f64, v.im as f64)).collect();
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(n).process(&mut a);
    let val = 1.0 / (n as f64 * (1.0 / fsd)); // numpy fftfreq
    let gain = if audio { 2.0 } else { 1.0 };
    for (k, v) in a.iter_mut().enumerate() {
        let fk = if k < n.div_ceil(2) { k as f64 } else { k as f64 - n as f64 } * val;
        *v *= if fk >= band.0 && fk <= band.1 { gain } else { 0.0 };
    }
    planner.plan_fft_inverse(n).process(&mut a);
    let c = (band.0 + band.1) / 2.0;
    let (inv, w, rf) = (1.0 / n as f64, -2.0 * std::f64::consts::PI * c, 1.0 / fsd);
    let z = a
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // numpy: (-2j pi c k) / fsd is a complex division, i.e. times 1/fsd
            let (s, co) = ((w * i as f64) * rf).sin_cos();
            let (re, im) = (v.re * inv, v.im * inv);
            Complex32::new((re * co - im * s) as f32, (re * s + im * co) as f32)
        })
        .collect();
    (z, c)
}

fn waterfall(y: &[Complex32], fsd: f64, span: f64) -> Waterfall {
    let win = (fsd / 10.0).round_ties_even() as usize;
    let (gr, mut rows) = stft(y, fsd, win, win * 5, span, |_, r| r.to_vec()); // 0.1 s FFT, 10 Hz bins, a row every 0.5 s
    normalise(&mut rows);
    let db = rows.iter().flat_map(|r| r.iter().map(|&v| 10.0 * v.log10())).collect();
    Waterfall { t: gr.t, f: gr.f, db }
}

/// Audio: keep the positive frequencies only and move the strongest tone (above 50 Hz) to 0 Hz.
fn to_analytic(x: &[f32], fsd: f64) -> (Vec<Complex32>, f64) {
    let n = x.len();
    let nps = 8192.min(n);
    let step = nps - nps / 2;
    let w: Vec<f64> = (0..nps).map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / nps as f64).cos()).collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nps);
    let nb = nps / 2 + 1;
    let mut p = vec![0.0f64; nb];
    let nseg = if n >= nps { (n - nps) / step + 1 } else { 0 };
    let mut buf = vec![Complex::<f64>::new(0.0, 0.0); nps];
    for s in 0..nseg {
        let seg = &x[s * step..s * step + nps];
        let mean = seg.iter().map(|&v| v as f64).sum::<f64>() / nps as f64;
        for k in 0..nps {
            buf[k] = Complex::new((seg[k] as f64 - mean) * w[k], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..nb {
            let dbl = if k == 0 || (nps % 2 == 0 && k == nps / 2) { 1.0 } else { 2.0 };
            p[k] += dbl * buf[k].norm_sqr();
        }
    }
    let fk = |k: usize| k as f64 * fsd / nps as f64;
    let best = (0..nb).filter(|&k| fk(k) >= 50.0).fold(None::<usize>, |b, k| match b {
        Some(j) if p[j] >= p[k] => Some(j),
        _ => Some(k),
    });
    let tone = best.map(fk).unwrap_or(0.0);
    // scipy.signal.hilbert keeps float32 input in complex64; its rounding floor in the empty half feeds fixed_lines()
    let mut a: Vec<Complex32> = x.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    let mut p32 = FftPlanner::<f32>::new();
    p32.plan_fft_forward(n).process(&mut a);
    for (k, v) in a.iter_mut().enumerate() {
        if k > n / 2 {
            *v = Complex32::new(0.0, 0.0);
        } else if k > 0 && k < n.div_ceil(2) {
            *v *= 2.0;
        }
    }
    p32.plan_fft_inverse(n).process(&mut a);
    let inv = 1.0 / n as f32;
    let y = a
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (s, c) = (-2.0 * std::f64::consts::PI * tone * i as f64 / fsd).sin_cos();
            let (re, im) = ((v.re * inv) as f64, (v.im * inv) as f64);
            Complex32::new((re * c - im * s) as f32, (re * s + im * c) as f32)
        })
        .collect();
    (y, tone)
}

// ---------- time and text ----------
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

/// ISO time as (microseconds since 1970 in its own clock, UTC offset in minutes if given).
pub fn parse_iso(s: &str) -> Result<(i64, Option<i64>), String> {
    let bad = || format!("cannot read the start time '{s}' (use e.g. 2026-06-14T18:10:09Z)");
    let s = s.trim();
    if !s.is_ascii() {
        return Err(bad());
    }
    let (mut body, mut off) = (s, None);
    if let Some(b) = s.strip_suffix('Z').or_else(|| s.strip_suffix('z')) {
        body = b;
        off = Some(0);
    } else if s.len() > 10 {
        if let Some(p) = s[10..].rfind(['+', '-']).map(|p| p + 10) {
            let o = s[p + 1..].replace(':', "");
            if o.len() != 4 || !o.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            let mins = o[..2].parse::<i64>().unwrap() * 60 + o[2..].parse::<i64>().unwrap();
            off = Some(if &s[p..p + 1] == "-" { -mins } else { mins });
            body = &s[..p];
        }
    }
    let num = |t: &str| t.parse::<i64>().map_err(|_| bad());
    if body.len() < 10 || &body[4..5] != "-" || &body[7..8] != "-" {
        return Err(bad());
    }
    let (y, mo, d) = (num(&body[..4])?, num(&body[5..7])?, num(&body[8..10])?);
    let mut us = 0i64;
    if body.len() > 10 {
        let tpart = &body[11..];
        let (hms, frac) = tpart.split_once(['.', ',']).unwrap_or((tpart, ""));
        let f: Vec<&str> = hms.split(':').collect();
        let h = num(f[0])?;
        let mi = if f.len() > 1 { num(f[1])? } else { 0 };
        let se = if f.len() > 2 { num(f[2])? } else { 0 };
        let fr = if frac.is_empty() { 0 } else { num(&format!("{frac:0<6}")[..6])? };
        us = ((h * 60 + mi) * 60 + se) * 1_000_000 + fr;
    }
    Ok((days_from_civil(y, mo, d) * 86_400_000_000 + us, off))
}

/// Python datetime.isoformat(timespec="milliseconds") of start + t seconds.
pub fn iso_ms(start: (i64, Option<i64>), t: f64) -> String {
    let us = start.0 + (t * 1e6).round_ties_even() as i64;
    let (days, r) = (us.div_euclid(86_400_000_000), us.rem_euclid(86_400_000_000));
    let (y, m, d) = civil_from_days(days);
    let ms = r / 1000;
    let mut s = format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    );
    if let Some(o) = start.1 {
        let _ = write!(s, "{}{:02}:{:02}", if o < 0 { '-' } else { '+' }, o.abs() / 60, o.abs() % 60);
    }
    s
}

/// Python repr(float).
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let s = format!("{x}");
        if s.contains('.') {
            s
        } else {
            s + ".0"
        }
    } else {
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// Python round(x, nd) for the values written here.
pub fn py_round(x: f64, nd: usize) -> f64 {
    if x.is_finite() {
        format!("{x:.nd$}").parse().unwrap_or(x)
    } else {
        x
    }
}

/// Python f"{x:.{nd}f}" (nan prints "nan").
pub fn fmt_f(x: f64, nd: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        format!("{x:.nd$}")
    }
}

/// JSON string with Python json.dumps escaping (ensure_ascii).
pub fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    let _ = write!(o, "\\u{:04x}", u);
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn sdrpp_centre(name: &str) -> Option<f64> {
    let b = name.as_bytes();
    let mut i = 0;
    while let Some(p) = name[i..].find("Hz").map(|p| p + i) {
        let mut s = p;
        while s > 0 && b[s - 1].is_ascii_digit() {
            s -= 1;
        }
        if p - s >= 6 {
            return name[s..p].parse().ok();
        }
        i = p + 2;
    }
    None
}

impl TraceResult {
    pub fn csv_text(&self) -> String {
        let mut s = String::from("t_s,utc,f_offset_Hz,f_Hz,snr_dB\n");
        for p in &self.points {
            let _ = writeln!(
                s,
                "{},{},{},{},{}",
                fmt_f(p.t_s, 3),
                p.utc.as_deref().unwrap_or(""),
                fmt_f(p.f_offset, 2),
                fmt_f(p.f, 2),
                fmt_f(p.snr, 1)
            );
        }
        s
    }

    /// The Python summary JSON (json.dump(..., indent=1)); like Python, the note is not in the file.
    pub fn summary_json(&self) -> String {
        let opt_f = |v: Option<f64>| v.map(py_float).unwrap_or_else(|| "null".into());
        let lines = if self.fixed_receiver_lines_hz.is_empty() {
            "[]".to_string()
        } else {
            format!("[\n{}\n ]", self.fixed_receiver_lines_hz.iter().map(|v| format!("  {}", py_float(*v))).collect::<Vec<_>>().join(",\n"))
        };
        let kv: Vec<(&str, String)> = vec![
            ("file", json_str(&self.file)),
            ("centre_Hz", py_float(self.centre_hz)),
            ("audio_tone_Hz", opt_f(self.audio_tone_hz)),
            ("band_Hz", self.band_hz.map_or("null".into(), |(a, b)| format!("[\n  {},\n  {}\n ]", py_float(a), py_float(b)))),
            ("burst", (if self.burst { "true" } else { "false" }).into()),
            ("sample_rate", self.sample_rate.to_string()),
            ("decimated_rate", py_float(self.decimated_rate)),
            ("duration_s", py_float(self.duration_s)),
            ("start_utc", self.start_utc.as_deref().map(json_str).unwrap_or_else(|| "null".into())),
            ("points", self.points.len().to_string()),
            ("point_noise_Hz", py_float(self.point_noise_hz)),
            ("point_noise_pairs", self.point_noise_pairs.to_string()),
            ("swing_Hz", py_float(self.swing_hz)),
            ("fixed_receiver_lines_Hz", lines),
            ("window", json_str(&self.window)),
        ];
        format!("{{\n{}\n}}", kv.iter().map(|(k, v)| format!(" \"{k}\": {v}")).collect::<Vec<_>>().join(",\n"))
    }

    pub fn write_csv(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.csv_text()).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn write_summary_json(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.summary_json()).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Trace one recording, write <name>_trace.csv and <name>_summary.json to the output folder
/// (default: next to the recording) and return everything.
pub fn run(path: &Path, o: &TraceOpts, log: &mut dyn FnMut(&str)) -> Result<TraceResult, String> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("recording").to_string();
    let orig = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let out = match &o.out {
        Some(d) => d.clone(),
        None => std::path::absolute(path).map_err(|e| e.to_string())?.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let fc0 = o.center.or_else(|| sdrpp_centre(&name));
    let t_start = o.start.as_deref().filter(|s| !s.trim().is_empty()).map(parse_iso).transpose()?;
    let inp = input::open(path, o.rate, o.format, log)?;
    if inp.rate < 100 {
        return Err(format!("sample rate {} S/s is too low", inp.rate));
    }
    if inp.n == 0 {
        return Err("the recording holds no samples".into());
    }
    let audio = inp.audio();
    if fc0.is_none() && !audio {
        return Err("give the centre frequency (Hz)".into());
    }
    let mut fc = fc0.unwrap_or(0.0);
    let (rate, n) = if inp.compressed { (48000u32, (inp.n as f64 * 48000.0 / inp.rate as f64).ceil() as u64) } else { (inp.rate, inp.n) };
    log(&format!(
        "{name}: {rate} S/s, {}-bit, {}, {:.1} s, centre {:.6} MHz",
        inp.bits,
        if audio { "audio" } else { "IQ" },
        n as f64 / rate as f64,
        fc / 1e6
    ));
    let mut rd = inp.reader(o.swap_iq)?;
    let (y, fsd) = if inp.compressed {
        (resample(rd.as_mut(), inp.n, 48000, inp.rate as usize, n as usize)?, 48000.0)
    } else {
        let dec = ((inp.rate as f64 / TARGET).round_ties_even() as usize).max(1);
        (resample(rd.as_mut(), inp.n, 1, dec, (inp.n / dec as u64) as usize)?, inp.rate as f64 / dec as f64)
    };
    drop(rd);
    drop(inp);
    let mut span = o.span;
    let (mut tone, mut dc) = (0.0, None);
    let band = match o.band {
        Some((a, b)) => {
            let (lo, hi) = (a.min(b), a.max(b));
            if hi - lo < 50.0 {
                return Err("the band must be at least 50 Hz wide".into());
            }
            if !((if audio { 0.0 <= lo } else { -fsd / 2.0 < lo }) && hi < fsd / 2.0) {
                return Err(format!("the band must lie inside the recording (+-{} Hz)", fmt_f(fsd / 2.0, 0)));
            }
            Some((lo, hi))
        }
        None => None,
    };
    let y = if let (true, Some(b)) = (audio, band) {
        let (z, c) = band_shift(&y, fsd, b, true);
        tone = c;
        span = (b.1 - b.0) / 2.0;
        log(&format!("  audio band {} to {} Hz", fmt_f(b.0, 0), fmt_f(b.1, 0)));
        z
    } else if audio {
        let re: Vec<f32> = y.iter().map(|v| v.re).collect();
        drop(y);
        let (mut a, t) = to_analytic(&re, fsd);
        tone = t;
        remove_dc(&mut a, (5.0 * fsd) as usize);
        dc = Some(0.0);
        span = span.min(fsd / 2.0 - 100.0);
        log(&format!("  strongest audio tone {} Hz", fmt_f(tone, 0)));
        a
    } else {
        let mut y = y;
        remove_dc(&mut y, (5.0 * fsd) as usize);
        dc = Some(0.0);
        if let Some(b) = band {
            let (z, c) = band_shift(&y, fsd, b, false);
            tone = c;
            span = (b.1 - b.0) / 2.0;
            dc = Some(-c);
            log(&format!("  offsets {} to {} Hz from the centre", fmt_f(b.0, 0), fmt_f(b.1, 0)));
            z
        } else {
            y
        }
    };
    fc += tone;
    let tr = if o.burst {
        log("  burst mode");
        burst_track(&y, fsd, span)?
    } else {
        track(&y, fsd, span, dc, band.is_some())?
    };
    let points: Vec<TracePoint> = (0..tr.t.len())
        .filter(|&i| tr.acc[i])
        .map(|i| TracePoint {
            t_s: tr.t[i],
            utc: t_start.map(|s| iso_ms(s, tr.t[i])),
            f_offset: tr.lc[i],
            f: fc + tr.lc[i],
            snr: tr.snr[i],
        })
        .collect();
    let wf = waterfall(&y, fsd, span);
    drop(y);
    let swing = if points.is_empty() {
        0.0
    } else {
        let (a, b) = points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.f_offset), b.max(p.f_offset)));
        b - a
    };
    let mut res = TraceResult {
        csv_path: out.join(format!("{name}_trace.csv")),
        summary_path: out.join(format!("{name}_summary.json")),
        name,
        file: orig,
        points,
        centre_hz: fc,
        audio_tone_hz: audio.then_some(tone),
        band_hz: band,
        burst: o.burst,
        sample_rate: rate,
        decimated_rate: fsd,
        duration_s: n as f64 / rate as f64,
        start_utc: o.start.clone().filter(|s| !s.trim().is_empty()),
        point_noise_hz: py_round(tr.sigma, 2),
        point_noise_pairs: tr.nsigma,
        swing_hz: py_round(swing, 1),
        fixed_receiver_lines_hz: tr.lines.iter().filter(|&&v| dc != Some(v)).map(|&v| py_round(v, 1)).collect(),
        window: if o.burst { BURST_WINDOW_NOTE } else { WINDOW_NOTE }.into(),
        note: None,
        burst_hint: None,
        waterfall: wf,
    };
    res.write_csv(&res.csv_path)?;
    res.write_summary_json(&res.summary_path)?;
    log(&format!(
        "  {} points, point noise {} Hz, swing {} Hz -> {}",
        res.points.len(),
        fmt_f(tr.sigma, 1),
        fmt_f(swing, 0),
        out.display()
    ));
    if !audio && swing < 0.2 * span {
        let note = format!(
            "The signal moves only {} Hz over the recording. A satellite pass at a fixed tuning moves several kHz. \
             This recording looks Doppler-corrected (the receiver followed a predicted orbit), so it holds only the leftover. \
             The fit needs a recording made at a fixed frequency with Doppler correction off.",
            fmt_f(swing, 0)
        );
        log(&format!("  NOTE: {note}"));
        res.note = Some(note);
    }
    if !audio && !o.burst {
        let cover = res.points.len() as f64 / tr.ninside.max(1) as f64;
        if swing < 0.2 * span || cover < 0.3 {
            let hint = format!(
                "The trace covers {}% of the recording and moves {} Hz. If the satellite sends short bursts \
                 (FSK or packets) under steady lines, run again with --burst.",
                fmt_f(100.0 * cover, 0),
                fmt_f(swing, 0)
            );
            log(&format!("  NOTE: {hint}"));
            res.burst_hint = Some(hint);
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
        }
    }

    const P: [f64; 4] = [1200.0, 6000.0, 60.0, 15.0];

    fn synth(fs: f64, secs: f64) -> Vec<Complex32> {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let n = (fs * secs) as usize;
        let mut ph = 0.0f64;
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                ph += pass_model(t, &P) / fs;
                let w = 2.0 * std::f64::consts::PI;
                let s = Complex::<f64>::from_polar(0.05, w * ph) + Complex::from_polar(0.02, w * 5306.0 * t);
                let v = s + Complex::new(0.1 * rng.gauss(), 0.1 * rng.gauss());
                Complex32::new(v.re as f32, v.im as f32)
            })
            .collect()
    }

    #[test]
    fn spline_interpolates_and_smooths() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v * 0.5 + 3.0).collect();
        let sp = Spline::fit(&x, &y, Some(1e-8)).unwrap();
        assert!(x.iter().zip(&y).all(|(a, b)| (sp.eval(*a) - b).abs() < 1e-4));
        let lin: Vec<f64> = x.iter().map(|v| 2.0 * v - 1.0).collect();
        let sp = Spline::fit(&x, &lin, None).unwrap();
        assert!((sp.eval(20.0) - 39.0).abs() < 1e-6);
    }

    #[test]
    fn resample_passes_dc_and_keeps_length() {
        let v = vec![1.0f32; 9600];
        let inp = input::Input::from_memory_for_test(v);
        let mut rd = inp.reader(false).unwrap();
        let y = resample(rd.as_mut(), 9600, 1, 2, 4800).unwrap();
        assert_eq!(y.len(), 4800);
        assert!((y[2400].re - 1.0).abs() < 1e-4 && y[2400].im.abs() < 1e-6);
    }

    #[test]
    fn iso_time_round_trip() {
        let s = parse_iso("2026-06-14T18:10:09Z").unwrap();
        assert_eq!(iso_ms(s, 12.25), "2026-06-14T18:10:21.250+00:00");
        assert_eq!(iso_ms(parse_iso("2026-12-31T23:59:59.9").unwrap(), 0.2), "2027-01-01T00:00:00.100");
        assert_eq!(py_float(437800000.0), "437800000.0");
        assert_eq!(py_float(1e-5), "1e-05");
    }

    #[test]
    fn bad_inputs_do_not_panic() {
        assert!(parse_iso("2026-06-1\u{e9}T00:00:00Z").is_err());
        assert!(parse_iso("2026-06-14T00\u{e9}00Z").is_err());
        assert!(track(&[Complex32::new(0.1, 0.0); 3], 2.0, 15000.0, Some(0.0), false).is_err());
        assert!(track(&[], 48000.0, 15000.0, Some(0.0), false).is_err());
        assert!(burst_track(&[Complex32::new(0.1, 0.0); 3], 2.0, 15000.0).is_err());
        assert!(burst_track(&[], 48000.0, 15000.0).is_err());
        let mut y: Vec<Complex32> = (0..1000).map(|i| Complex32::new((i % 7) as f32, 1.0 - (i % 3) as f32)).collect();
        let orig = y.clone();
        remove_dc(&mut y, 10);
        let mut c = vec![Complex::<f64>::new(0.0, 0.0)];
        for v in &orig { let l = *c.last().unwrap(); c.push(l + Complex::new(v.re as f64, v.im as f64)); }
        for i in 0..orig.len() {
            let (a, b) = (i.saturating_sub(10), (i + 11).min(orig.len()));
            let m = (c[b] - c[a]) / (b - a) as f64;
            assert_eq!(y[i], Complex32::new((orig[i].re as f64 - m.re) as f32, (orig[i].im as f64 - m.im) as f32));
        }
    }

    #[test]
    fn trace_follows_injected_doppler() {
        let fs = 48000.0;
        let mut y = synth(fs, 120.0);
        remove_dc(&mut y, (5.0 * fs) as usize);
        let tr = track(&y, fs, 15000.0, Some(0.0), false).expect("track");
        let acc: Vec<usize> = (0..tr.t.len()).filter(|&i| tr.acc[i]).collect();
        assert!(acc.len() > 100, "only {} points", acc.len());
        let worst = acc.iter().map(|&i| (tr.lc[i] - pass_model(tr.t[i], &P)).abs()).fold(0.0, f64::max);
        let rms = (acc.iter().map(|&i| (tr.lc[i] - pass_model(tr.t[i], &P)).powi(2)).sum::<f64>() / acc.len() as f64).sqrt();
        assert!(rms < 3.0 && worst < 15.0, "rms {rms} worst {worst}");
        assert!(tr.lines.iter().any(|l| (l - 5306.0).abs() < 5.0), "lines {:?}", tr.lines);
        assert!(tr.sigma.is_finite() && tr.sigma < 5.0);
    }

    #[test]
    fn band_cut_keeps_only_the_band() {
        // fftfreq grid of n = 1000 at 1000 S/s is 1 Hz; tones at 100 Hz (inside 80..120) and 300 Hz (outside)
        let (fs, n) = (1000.0, 1000usize);
        let w = 2.0 * std::f64::consts::PI;
        let y: Vec<Complex32> =
            (0..n).map(|i| { let t = i as f64 / fs; Complex32::new(((w * 100.0 * t).cos() + 2.0 * (w * 300.0 * t).cos()) as f32, 0.0) }).collect();
        let (z, c) = band_shift(&y, fs, (80.0, 120.0), true);
        assert_eq!(c, 100.0);
        // audio x2 on the positive half: the 100 Hz cosine becomes exp(i 2 pi 100 t), then shifted to 0 Hz
        assert!(z.iter().all(|v| (v.re - 1.0).abs() < 1e-5 && v.im.abs() < 1e-5), "{:?}", &z[..4]);
        // IQ: the edges LO and HI are inside (>=, <=); -20 Hz from the centre lands on 0 Hz
        let y: Vec<Complex32> = (0..n).map(|i| { let p = w * -20.0 * i as f64 / fs; Complex32::new(p.cos() as f32, p.sin() as f32) }).collect();
        let (z, c) = band_shift(&y, fs, (-30.0, -20.0), false);
        assert_eq!(c, -25.0);
        assert!(z.iter().all(|v| (v.norm() - 1.0).abs() < 1e-5));
        assert!(z.windows(2).all(|p| ((p[1] * p[0].conj()).arg() as f64 - w * 5.0 / fs).abs() < 1e-4));
        let (z, _) = band_shift(&y, fs, (-19.5, 50.0), false);
        assert!(z.iter().all(|v| v.norm() < 1e-6));
    }

    #[test]
    fn running_median_matches_brute_force() {
        let mut rng = Rng(12345);
        let x: Vec<f32> = (0..300).map(|_| (rng.u() * 100.0).floor() as f32).collect();
        for half in [0usize, 1, 5, 40, 200] {
            let m = running_median(&x, half);
            for i in 0..x.len() {
                let mut w: Vec<f32> = (i as i64 - half as i64..=i as i64 + half as i64).map(|k| x[k.clamp(0, 299) as usize]).collect();
                w.sort_by(|a, b| a.total_cmp(b));
                assert_eq!(m[i], w[half], "half {half} i {i}");
            }
        }
    }

    #[test]
    fn burst_pass_found_under_a_steady_line() {
        // 2-FSK bursts (+-500 Hz) 1 s in every 10 s under a steady line 2.5 times stronger, 12000 S/s, 200 s
        let (fs, secs) = (12000.0, 200.0);
        let p = [500.0, 2500.0, 100.0, 12.0];
        let mut rng = Rng(7);
        let n = (fs * secs) as usize;
        let mut ph = 0.0f64;
        let w = 2.0 * std::f64::consts::PI;
        let y: Vec<Complex32> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let bit = if ((t * 1200.0) as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 63 == 1 { 500.0 } else { -500.0 };
                ph += (pass_model(t, &p) + bit) / fs;
                let on = if t % 10.0 < 1.0 { 0.2 } else { 0.0 };
                let v = Complex::<f64>::from_polar(on, w * ph) + Complex::from_polar(0.5, w * 1700.0 * t)
                    + Complex::new(0.1 * rng.gauss(), 0.1 * rng.gauss());
                Complex32::new(v.re as f32, v.im as f32)
            })
            .collect();
        let tr = burst_track(&y, fs, 5500.0).expect("burst");
        let acc: Vec<usize> = (0..tr.t.len()).filter(|&i| tr.acc[i]).collect();
        assert!(acc.len() >= 30, "only {} points", acc.len());
        let worst = acc.iter().map(|&i| (tr.lc[i] - pass_model(tr.t[i], &p)).abs()).fold(0.0, f64::max);
        let rms = (acc.iter().map(|&i| (tr.lc[i] - pass_model(tr.t[i], &p)).powi(2)).sum::<f64>() / acc.len() as f64).sqrt();
        assert!(rms < 80.0 && worst < 400.0, "rms {rms} worst {worst} n {}", acc.len());
        assert!(tr.lines.is_empty() && tr.sigma.is_finite());
    }

    /// Parity run on a real file: DOPPLER_PARITY_FILE=... DOPPLER_PARITY_OUT=... cargo test --release -- --ignored parity
    #[test]
    #[ignore]
    fn parity() {
        let f = std::env::var("DOPPLER_PARITY_FILE").unwrap();
        let o = TraceOpts {
            out: std::env::var("DOPPLER_PARITY_OUT").ok().map(PathBuf::from),
            start: std::env::var("DOPPLER_PARITY_START").ok(),
            center: std::env::var("DOPPLER_PARITY_CENTER").ok().map(|v| v.parse().unwrap()),
            rate: std::env::var("DOPPLER_PARITY_RATE").ok().map(|v| v.parse().unwrap()),
            format: std::env::var("DOPPLER_PARITY_FORMAT").map(|v| v.parse().unwrap()).unwrap_or_default(),
            swap_iq: std::env::var("DOPPLER_PARITY_SWAP").is_ok(),
            band: std::env::var("DOPPLER_PARITY_BAND").ok().map(|v| {
                let (a, b) = v.split_once(',').unwrap();
                (a.parse().unwrap(), b.parse().unwrap())
            }),
            burst: std::env::var("DOPPLER_PARITY_BURST").is_ok(),
            ..Default::default()
        };
        let r = run(Path::new(&f), &o, &mut |s| eprintln!("{s}")).unwrap();
        eprintln!("{}", r.summary_json());
        let w = &r.waterfall;
        eprintln!("waterfall {} x {} t0 {} f0 {} db[7,100] {}", w.t.len(), w.f.len(), w.t[0], w.f[0], w.db[7 * w.f.len() + 100]);
    }
}
