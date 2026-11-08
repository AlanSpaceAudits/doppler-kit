//! Straight-pass Doppler fit: speed, closest distance, closest-approach time and transmitted frequency
//! from a received-frequency trace alone (port of fit_pass.py).

use crate::lm;
use std::path::{Path, PathBuf};

pub const C_LIGHT: f64 = 299792458.0;

/// Received frequency at time `t` for params `p = [t0, v, d, f0]`, travel time included.
pub fn model_at(t: f64, p: &[f64; 4], c: f64) -> f64 {
    let [t0, v, d, f0] = *p;
    let tp = t - t0;
    let mut te = tp;
    for _ in 0..6 {
        te = tp - ((d * d + v * v * te * te).sqrt() - d) / c;
    }
    f0 / (1.0 + v * v * te / (d * d + v * v * te * te).sqrt() / c)
}

pub fn model(t: &[f64], p: &[f64; 4], c: f64) -> Vec<f64> {
    t.iter().map(|&x| model_at(x, p, c)).collect()
}

#[derive(Clone, Debug)]
pub struct MethodA { pub v: f64, pub f0: f64, pub t0: f64, pub d: f64, pub dt: f64 }

#[derive(Clone, Debug)]
pub struct MethodD {
    pub t0_s: f64, pub v_m_s: f64, pub d_m: f64, pub f0_hz: f64, pub rms_hz: f64, pub n: usize,
    /// (key, value) in Python order: t0_s, v_m_s, d_m, f0_Hz (f0 absent when fixed).
    pub err: Vec<(String, f64)>,
    pub range_first_m: f64, pub range_last_m: f64, pub t0_utc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PathTest { pub bending: bool, pub p_value: f64, pub pattern_rms_hz: f64, pub scatter_rms_hz: f64, pub detectable_rms_hz: f64 }

#[derive(Clone, Debug)]
pub struct Marks {
    pub f_plus_hz: f64, pub f_minus_hz: f64, pub half_plateau_hz: [f64; 2], pub half_plateau_t_s: [f64; 2],
    pub dt_s: f64, pub slope_hz_s: f64, pub v_plateaus_m_s: f64, pub d_width_m: f64, pub d_slope_m: f64,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub c: f64,
    pub a: MethodA,
    pub d: MethodD,
    pub path: PathTest,
    pub marks: Marks,
    /// Fitted [t0, v, d, f0].
    pub p: [f64; 4],
    /// Sorted input and what the fit leaves over.
    pub t: Vec<f64>,
    pub f: Vec<f64>,
    pub left: Vec<f64>,
    pub indep: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Trace { pub t: Vec<f64>, pub f: Vec<f64>, pub utc: Option<Vec<String>>, pub f_offset: Option<Vec<f64>> }

impl Trace {
    /// Receiver centre frequency from the first row (f_Hz - f_offset_Hz), when the trace has offsets.
    pub fn centre(&self) -> Option<f64> {
        self.f_offset.as_ref().and_then(|o| Some(self.f.first()? - o.first()?))
    }
}

/// Reads a trace CSV with columns t_s and f_Hz (utc and f_offset_Hz optional).
pub fn read_trace_csv(path: impl AsRef<Path>) -> Result<Trace, String> {
    let txt = std::fs::read_to_string(path.as_ref()).map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
    let mut lines = txt.lines();
    let head: Vec<&str> = lines.next().ok_or("empty trace file")?.split(',').map(|s| s.trim().trim_start_matches('\u{feff}')).collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let (it, jf) = (col("t_s").ok_or("trace has no t_s column")?, col("f_Hz").ok_or("trace has no f_Hz column")?);
    let (iu, io) = (col("utc"), col("f_offset_Hz"));
    let mut tr = Trace::default();
    let (mut utc, mut off) = (Vec::new(), Vec::new());
    for (k, l) in lines.enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let w: Vec<&str> = l.split(',').map(str::trim).collect();
        let num = |i: usize| w.get(i).ok_or(format!("row {}: missing column", k + 2))?.parse::<f64>().map_err(|e| format!("row {}: {e}", k + 2));
        tr.t.push(num(it)?);
        tr.f.push(num(jf)?);
        if let Some(i) = iu { utc.push(w.get(i).unwrap_or(&"").to_string()); }
        if let Some(i) = io { off.push(w.get(i).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN)); }
    }
    if utc.first().is_some_and(|s| !s.is_empty()) { tr.utc = Some(utc); }
    if off.first().is_some_and(|x| x.is_finite()) { tr.f_offset = Some(off); }
    Ok(tr)
}

fn mean(x: &[f64]) -> f64 { lm::np_sum(x) / x.len() as f64 }

fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(|a, b| a.total_cmp(b));
    let n = x.len();
    if n == 0 { f64::NAN } else if n % 2 == 1 { x[n / 2] } else { 0.5 * (x[n / 2 - 1] + x[n / 2]) }
}

fn sign(x: f64) -> f64 { if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else if x == 0.0 { 0.0 } else { f64::NAN } }

fn argmin_by(x: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, f64::INFINITY);
    for (i, v) in x.enumerate() {
        if v < best.1 { best = (i, v); }
    }
    best.0
}

/// Plateaus from the raw ends and the half-plateau width.
pub fn method_a(t: &[f64], f: &[f64], c: f64, f0: Option<f64>) -> MethodA {
    let n = t.len();
    let k = ((n as f64 * 0.02).round_ties_even() as usize).max(3).min(n);
    let (fp, fm) = (mean(&f[..k]), mean(&f[n - k..]));
    let v = c * (fp - fm) / (fp + fm);
    let f0 = f0.filter(|x| *x != 0.0).unwrap_or(2.0 * fp * fm / (fp + fm));
    let t0 = t[argmin_by(f.iter().map(|x| (x - f0).abs()))];
    let cross = |ff: f64| {
        let s: Vec<usize> = (0..n.saturating_sub(1)).filter(|&i| sign(f[i + 1] - ff) - sign(f[i] - ff) != 0.0).collect();
        if s.is_empty() { return f64::NAN; }
        let j = s[argmin_by(s.iter().map(|&i| (t[i] - t0).abs()))];
        t[j] + (t[j + 1] - t[j]) * (ff - f[j]) / (f[j + 1] - f[j])
    };
    let dt = cross(f0 / (1.0 + v / (2.0 * c))) - cross(f0 / (1.0 - v / (2.0 * c)));
    MethodA { v, f0, t0, d: dt.abs() * v * 3f64.sqrt() / 2.0, dt: dt.abs() }
}

/// Points sharing one analysis window: iq_to_trace uses 1 s windows, so a 0.25 s step gives 4.
pub fn auto_indep(t_sorted: &[f64]) -> f64 {
    let step = median(t_sorted.windows(2).map(|w| w[1] - w[0]).collect());
    if step < 1.0 { (1.0 / step).max(1.0) } else { 1.0 }
}

/// Least-squares polynomial of degree k evaluated at x (numpy polyfit + polyval).
pub fn polyfit_eval(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let a: Vec<Vec<f64>> = x.iter().map(|&xi| (0..=k).map(|j| xi.powi((k - j) as i32)).collect()).collect();
    match lm::lstsq(a, y.to_vec()) {
        Some(cf) => x.iter().map(|&xi| cf.iter().fold(0.0, |acc, c| acc * xi + c)).collect(),
        None => vec![0.0; x.len()],
    }
}

/// Bending path or not: F-test of a degree-k polynomial through the leftover,
/// with n_eff = n / indep independent points; p < 0.001 = bending, otherwise no bending detected.
pub fn straightness(t: &[f64], left: &[f64], indep: f64, k: usize) -> PathTest {
    let (mt, ptp) = (mean(t), t.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t.iter().cloned().fold(f64::INFINITY, f64::min));
    let x: Vec<f64> = t.iter().map(|v| (v - mt) / ptp).collect();
    let smooth = polyfit_eval(&x, left, k);
    let n = t.len() as f64 / indep;
    let sq = |v: Vec<f64>| lm::np_sum(&v.iter().map(|x| x * x).collect::<Vec<_>>());
    let rss0 = sq(left.to_vec());
    let rss1 = sq(left.iter().zip(&smooth).map(|(a, b)| a - b).collect());
    let d2 = (n - k as f64 - 1.0).max(1.0);
    let ff = ((rss0 - rss1) / (k as f64 + 1.0)) / (rss1 / d2);
    let p = f_sf(ff, k as f64 + 1.0, d2);
    let scatter = (rss1 / t.len() as f64).sqrt();
    // Detection limit. N = number of points, A = pattern rms, s = scatter rms, d1 = k + 1.
    //   the polynomial is a least-squares projection, so rss0 - rss1 = sum(smooth^2) = N A^2
    //   rss1 = N s^2
    //   F = (N A^2 / d1) / (N s^2 / d2)
    //   F = (A^2 / s^2) (d2 / d1)
    //   bending when F >= F_crit, the 0.999 quantile of F(d1, d2)
    //   A^2 = s^2 F_crit d1 / d2
    //   A_min = s sqrt(F_crit d1 / d2)
    let d1 = k as f64 + 1.0;
    let detectable = scatter * (f_isf(0.001, d1, d2) * d1 / d2).sqrt();
    PathTest {
        bending: p < 0.001, p_value: p,
        pattern_rms_hz: (sq(smooth) / t.len() as f64).sqrt(),
        scatter_rms_hz: scatter,
        detectable_rms_hz: detectable,
    }
}

/// F with f_sf(F, d1, d2) = q (the 1 - q quantile), by bisection on the survival function.
pub fn f_isf(q: f64, d1: f64, d2: f64) -> f64 {
    let mut hi = 1.0;
    while f_sf(hi, d1, d2) > q && hi < 1e300 { hi *= 2.0; }
    let mut lo = 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi { break; }
        if f_sf(mid, d1, d2) > q { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// ln Gamma(x), Lanczos g=7, n=9.
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN { d = FPMIN; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..100000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c; if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c; if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 { break; }
    }
    h
}

/// Regularized incomplete beta I_x(a, b).
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() { return f64::NAN; }
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let lbt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        (lbt + betacf(a, b, x).ln() - a.ln()).exp()
    } else {
        1.0 - (lbt + betacf(b, a, 1.0 - x).ln() - b.ln()).exp()
    }
}

/// Survival function of the F distribution with (d1, d2) degrees of freedom.
pub fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if f.is_nan() { return f64::NAN; }
    if f <= 0.0 { return 1.0; }
    if f.is_infinite() { return 0.0; }
    betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))
}

/// Plateaus, middle, half-plateau points and the width, all from the fitted curve.
pub fn marks(p: &[f64; 4], c: f64) -> Marks {
    let [t0, v, d, f0] = *p;
    let (fp, fm) = (f0 / (1.0 - v / c), f0 / (1.0 + v / c));
    let fh = [f0 / (1.0 - v / (2.0 * c)), f0 / (1.0 + v / (2.0 * c))];
    const N: usize = 200001;
    let (lo, hi) = (t0 - 50.0 * d / v, t0 + 50.0 * d / v);
    let step = (hi - lo) / (N - 1) as f64;
    let tt: Vec<f64> = (0..N).map(|i| if i == N - 1 { hi } else { lo + i as f64 * step }).collect();
    let xp: Vec<f64> = tt.iter().map(|&x| -model_at(x, p, c)).collect();
    let interp = |x: f64| -> f64 {
        if x.is_nan() { return f64::NAN; }
        if x <= xp[0] { return tt[0]; }
        if x >= xp[N - 1] { return tt[N - 1]; }
        let i = xp.partition_point(|&v| v <= x).saturating_sub(1).min(N - 2);
        if xp[i] == x { return tt[i]; }
        (tt[i + 1] - tt[i]) / (xp[i + 1] - xp[i]) * (x - xp[i]) + tt[i]
    };
    let th = [interp(-fh[0]), interp(-fh[1])];
    let slope = (model_at(t0 + 0.5, p, c) - model_at(t0 - 0.5, p, c)) / (2.0 * 0.5);
    Marks {
        f_plus_hz: fp, f_minus_hz: fm, half_plateau_hz: fh, half_plateau_t_s: th, dt_s: th[1] - th[0], slope_hz_s: slope,
        v_plateaus_m_s: c * (fp - fm) / (fp + fm), d_width_m: (th[1] - th[0]) * v * 3f64.sqrt() / 2.0,
        d_slope_m: f0 * v * v / (c * slope.abs()),
    }
}

/// Fits one trace. `indep <= 0` picks the Python rule (`auto_indep`). `utc`, when given, is the utc column in file order.
pub fn run(t: &[f64], f: &[f64], utc: Option<&[String]>, c: f64, f0: Option<f64>, indep: f64) -> Result<FitResult, String> {
    if t.len() != f.len() { return Err("t and f differ in length".into()); }
    if t.len() < 10 { return Err("too few points on the trace to fit".into()); }
    let f0 = f0.filter(|x| *x != 0.0);
    let mut o: Vec<usize> = (0..t.len()).collect();
    o.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let ts: Vec<f64> = o.iter().map(|&i| t[i]).collect();
    let fs: Vec<f64> = o.iter().map(|&i| f[i]).collect();
    let a = method_a(&ts, &fs, c, f0);
    let p0 = [a.t0, a.v, if a.d.is_finite() && a.d > 0.0 { a.d } else { a.v * 30.0 }, a.f0];
    let free: Vec<usize> = if f0.is_some() { vec![0, 1, 2] } else { vec![0, 1, 2, 3] };
    let full = |q: &[f64]| {
        let mut p = p0;
        for (i, &k) in free.iter().enumerate() { p[k] = q[i]; }
        p
    };
    let res = |q: &[f64]| {
        let mut p = full(q);
        p[1] = p[1].abs();
        p[2] = p[2].abs();
        ts.iter().zip(&fs).map(|(&x, &y)| model_at(x, &p, c) - y).collect::<Vec<f64>>()
    };
    let x0: Vec<f64> = free.iter().map(|&k| p0[k]).collect();
    let r = lm::solve(res, &x0);
    let mut p = full(&r.x);
    p[1] = p[1].abs();
    p[2] = p[2].abs();
    let left: Vec<f64> = ts.iter().zip(&fs).map(|(&x, &y)| y - model_at(x, &p, c)).collect();
    let rms = (lm::np_sum(&left.iter().map(|v| v * v).collect::<Vec<_>>()) / left.len() as f64).sqrt();
    let indep = if indep > 0.0 { indep } else { auto_indep(&ts) };
    let cov = r.covariance(rms * rms * indep);
    const KEYS: [&str; 4] = ["t0_s", "v_m_s", "d_m", "f0_Hz"];
    let err = free.iter().enumerate().map(|(i, &k)| (KEYS[k].to_string(), cov[i][i].sqrt())).collect();
    let path = straightness(&ts, &left, indep, 6);
    let rng = |x: f64| p[2].hypot(p[1] * (x - p[0]));
    let t0_utc = utc.and_then(|u| u.first()).filter(|s| !s.is_empty()).and_then(|s| iso_add(s, p[0] - t[0]));
    let d = MethodD {
        t0_s: p[0], v_m_s: p[1], d_m: p[2], f0_hz: p[3], rms_hz: rms, n: ts.len(), err,
        range_first_m: rng(ts[0]), range_last_m: rng(ts[ts.len() - 1]), t0_utc,
    };
    Ok(FitResult { c, a, d, path, marks: marks(&p, c), p, t: ts, f: fs, left, indep })
}

/// Reads a trace, fits it and writes `<name>_fit.json` into `out` (default: next to the trace).
pub fn run_file(trace: impl AsRef<Path>, c: f64, f0: Option<f64>, out: Option<&Path>) -> Result<(FitResult, PathBuf), String> {
    let trace = trace.as_ref();
    let tr = read_trace_csv(trace)?;
    let r = run(&tr.t, &tr.f, tr.utc.as_deref(), c, f0, 0.0)?;
    let dir = match out { Some(o) => o.to_path_buf(), None => std::fs::canonicalize(trace).ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default() };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let fp = dir.join(format!("{}_fit.json", out_name(trace)));
    std::fs::write(&fp, r.to_json()).map_err(|e| format!("{}: {e}", fp.display()))?;
    Ok((r, fp))
}

/// Output stem: the file stem without "_trace".
pub fn out_name(trace: &Path) -> String {
    trace.file_stem().map(|s| s.to_string_lossy().replace("_trace", "")).unwrap_or_default()
}

impl FitResult {
    pub fn err(&self, key: &str) -> f64 {
        self.d.err.iter().find(|(k, _)| k == key).map_or(0.0, |e| e.1)
    }

    /// Fitted curve at the given times.
    pub fn curve(&self, t: &[f64]) -> Vec<f64> { model(t, &self.p, self.c) }

    /// Same keys, order and layout as fit_pass.py's json.dump(..., indent=1).
    pub fn to_json(&self) -> String {
        let (a, d, s, m) = (&self.a, &self.d, &self.path, &self.marks);
        let mut dd: Vec<(&str, J)> = vec![
            ("t0_s", J::F(d.t0_s)), ("v_m_s", J::F(d.v_m_s)), ("d_m", J::F(d.d_m)), ("f0_Hz", J::F(d.f0_hz)), ("rms_Hz", J::F(d.rms_hz)),
            ("n", J::I(d.n as i64)), ("err", J::O(d.err.iter().map(|(k, v)| (k.as_str(), J::F(*v))).collect())),
            ("range_first_m", J::F(d.range_first_m)), ("range_last_m", J::F(d.range_last_m)),
        ];
        if let Some(u) = &d.t0_utc { dd.push(("t0_utc", J::S(u.clone()))); }
        let root = J::O(vec![
            ("c_m_s", J::F(self.c)),
            ("method_A", J::O(vec![("v", J::F(a.v)), ("f0", J::F(a.f0)), ("t0", J::F(a.t0)), ("d", J::F(a.d)), ("dt", J::F(a.dt))])),
            ("method_D", J::O(dd)),
            ("path", J::O(vec![("path", J::S(self.path_word().into())), ("p_value", J::F(s.p_value)),
                ("pattern_rms_Hz", J::F(s.pattern_rms_hz)), ("scatter_rms_Hz", J::F(s.scatter_rms_hz)),
                ("detectable_rms_Hz", J::F(s.detectable_rms_hz))])),
            ("marks", J::O(vec![
                ("f_plus_Hz", J::F(m.f_plus_hz)), ("f_minus_Hz", J::F(m.f_minus_hz)),
                ("half_plateau_Hz", J::A(m.half_plateau_hz.to_vec())), ("half_plateau_t_s", J::A(m.half_plateau_t_s.to_vec())),
                ("dt_s", J::F(m.dt_s)), ("slope_Hz_s", J::F(m.slope_hz_s)), ("v_plateaus_m_s", J::F(m.v_plateaus_m_s)),
                ("d_width_m", J::F(m.d_width_m)), ("d_slope_m", J::F(m.d_slope_m)),
            ])),
        ]);
        let mut s = String::new();
        root.write(&mut s, 0);
        s
    }

    pub fn path_word(&self) -> &'static str { if self.path.bending { "bending" } else { "no bending detected" } }

    /// The three lines fit_pass.py prints.
    pub fn log_text(&self) -> String {
        let (a, d, s) = (&self.a, &self.d, &self.path);
        format!(
            "A (plateaus): v {} km/s, d {} km, f0 {} Hz\nD (full fit): v {} ± {} km/s, d {} ± {} km, t0 {} s, f0 {} Hz, left over {} Hz rms over {} points\nPath: {} (smooth pattern {} Hz rms against scatter {} Hz, p = {}; detectable from {} Hz rms)\n",
            fx(a.v / 1e3, 3), fx(a.d / 1e3, 1), fx(a.f0, 1),
            fx(d.v_m_s / 1e3, 3), fx(self.err("v_m_s") / 1e3, 3), fx(d.d_m / 1e3, 1), fx(self.err("d_m") / 1e3, 1),
            fx(d.t0_s, 2), fx(d.f0_hz, 1), fx(d.rms_hz, 1), d.n,
            self.path_word(), fx(s.pattern_rms_hz, 1), fx(s.scatter_rms_hz, 1), fg(s.p_value, 2), fx(s.detectable_rms_hz, 1),
        )
    }

    /// The FULL FIT / READ OFF THE CURVE / PATH blocks of the GUI Results tab.
    pub fn report_lines(&self) -> Vec<String> {
        let (d, a, m, s) = (&self.d, &self.a, &self.marks, &self.path);
        let km = self.c > 1e5;
        let l = |x: f64| if km { format!("{} km", fcomma(x / 1e3, 2)) } else { format!("{} m", fcomma(x, 2)) };
        let v = |x: f64| if km { format!("{} km/s", fx(x / 1e3, 4)) } else { format!("{} m/s", fx(x, 3)) };
        vec![
            "FULL FIT (straight pass, constant speed, signal travel time included)".into(),
            format!("  speed v                 {}  ± {}", v(d.v_m_s), v(self.err("v_m_s"))),
            format!("  closest distance d      {}  ± {}", l(d.d_m), l(self.err("d_m"))),
            format!("  closest approach t₀     {} s from start{}", fx(d.t0_s, 2), d.t0_utc.as_ref().map(|u| format!("  ({u} UTC)")).unwrap_or_default()),
            format!("  transmitted f₀          {} Hz", fcomma(d.f0_hz, 1)),
            format!("  distance at first point {}", l(d.range_first_m)),
            format!("  distance at last point  {}", l(d.range_last_m)),
            format!("  fit leaves over         {} Hz rms", fx(d.rms_hz, 1)),
            String::new(),
            "READ OFF THE CURVE".into(),
            format!("  plateau f₊ (coming)     {} Hz", fcomma(m.f_plus_hz, 1)),
            format!("  plateau f₋ (going)      {} Hz", fcomma(m.f_minus_hz, 1)),
            format!("  half-plateau 1          {} Hz at {} s", fcomma(m.half_plateau_hz[0], 1), fx(m.half_plateau_t_s[0], 2)),
            format!("  half-plateau 2          {} Hz at {} s", fcomma(m.half_plateau_hz[1], 1), fx(m.half_plateau_t_s[1], 2)),
            format!("  width Δt                {} s", fx(m.dt_s, 2)),
            format!("  slope at the middle     {} Hz/s", fx(m.slope_hz_s, 2)),
            format!("  v = c (f₊ − f₋)/(f₊ + f₋)      {}", v(m.v_plateaus_m_s)),
            format!("  d = Δt · v · √3/2              {}", l(m.d_width_m)),
            format!("  d = f₀ v² / (c · |slope|)       {}", l(m.d_slope_m)),
            format!("  plateaus from the raw ends:    v {}, d {}   (low when the pass is cut short)", v(a.v), l(a.d)),
            String::new(),
            "PATH".into(),
            format!("  {}: smooth pattern left by the fit {} Hz rms against scatter {} Hz rms (p = {})",
                self.path_word().to_uppercase(), fx(s.pattern_rms_hz, 1), fx(s.scatter_rms_hz, 1), fg(s.p_value, 2)),
            format!("  This recording can detect a smooth pattern of {} Hz rms or more (p below 0.001).", fx(s.detectable_rms_hz, 1)),
            "  A straight path leaves only scatter. A bending path leaves a smooth wave.".into(),
        ]
    }
}

enum J<'a> { F(f64), I(i64), S(String), A(Vec<f64>), O(Vec<(&'a str, J<'a>)>) }

impl J<'_> {
    fn write(&self, s: &mut String, ind: usize) {
        let pad = |n: usize| " ".repeat(n);
        match self {
            J::F(x) => s.push_str(&py_float(*x)),
            J::I(i) => s.push_str(&i.to_string()),
            J::S(t) => { s.push('"'); s.push_str(&t.replace('\\', "\\\\").replace('"', "\\\"")); s.push('"'); }
            J::A(v) => {
                s.push_str("[\n");
                for (i, x) in v.iter().enumerate() {
                    s.push_str(&pad(ind + 1));
                    s.push_str(&py_float(*x));
                    s.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                s.push_str(&pad(ind));
                s.push(']');
            }
            J::O(kv) => {
                s.push_str("{\n");
                for (i, (k, v)) in kv.iter().enumerate() {
                    s.push_str(&format!("{}\"{k}\": ", pad(ind + 1)));
                    v.write(s, ind + 1);
                    s.push_str(if i + 1 < kv.len() { ",\n" } else { "\n" });
                }
                s.push_str(&pad(ind));
                s.push('}');
            }
        }
    }
}

/// Python repr(float) as json.dump writes it (NaN, Infinity included).
pub fn py_float(x: f64) -> String {
    if x.is_nan() { return "NaN".into(); }
    if x.is_infinite() { return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }; }
    if x == 0.0 { return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() }; }
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sgn = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp >= 0 {
            if n <= exp + 1 { format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize)) }
            else { format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..]) }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sgn}{body}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sgn}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

/// Python f"{x:.{n}f}".
pub fn fx(x: f64, n: usize) -> String {
    if x.is_nan() { "nan".into() } else if x.is_infinite() { if x > 0.0 { "inf".into() } else { "-inf".into() } } else { format!("{x:.n$}") }
}

/// Python f"{x:,.{n}f}".
pub fn fcomma(x: f64, n: usize) -> String {
    let s = fx(x, n);
    if !x.is_finite() { return s; }
    let (sgn, rest) = s.strip_prefix('-').map_or(("", s.as_str()), |r| ("-", r));
    let (int, frac) = rest.split_once('.').map_or((rest, None), |(a, b)| (a, Some(b)));
    let mut g = String::new();
    for (i, ch) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 { g.push(','); }
        g.push(ch);
    }
    match frac { Some(f) => format!("{sgn}{g}.{f}"), None => format!("{sgn}{g}") }
}

/// Python f"{x:.{p}g}".
pub fn fg(x: f64, p: usize) -> String {
    if !x.is_finite() { return fx(x, 0); }
    if x == 0.0 { return if x.is_sign_negative() { "-0".into() } else { "0".into() }; }
    let p = p.max(1);
    let e = format!("{:.*e}", p - 1, x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: String| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if exp >= -4 && exp < p as i32 {
        strip(format!("{:.*}", (p as i32 - 1 - exp).max(0) as usize, x))
    } else {
        format!("{}e{}{:02}", strip(mant.to_string()), if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

/// Days since 1970-01-01 for a civil date.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
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

/// Python `(datetime.fromisoformat(base) + timedelta(seconds=secs)).isoformat(timespec="milliseconds")`.
pub fn iso_add(base: &str, secs: f64) -> Option<String> {
    let b = base.trim();
    let date = b.get(..10)?;
    let mut it = date.split('-');
    let (y, mo, d): (i64, i64, i64) = (it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?);
    let rest = b.get(11..).unwrap_or("");
    let tz_at = rest.find(['+', '-', 'Z']).unwrap_or(rest.len());
    let (time, tz) = rest.split_at(tz_at);
    let tz = if tz == "Z" { "+00:00".to_string() } else { tz.to_string() };
    let mut us: i64 = 0;
    if !time.is_empty() {
        let (hms, frac) = time.split_once('.').unwrap_or((time, ""));
        let p: Vec<i64> = hms.split(':').map(|x| x.parse().ok()).collect::<Option<_>>()?;
        us = (p.first()? * 3600 + p.get(1).unwrap_or(&0) * 60 + p.get(2).unwrap_or(&0)) * 1_000_000;
        if !frac.is_empty() {
            let f6: String = format!("{frac:0<6}").chars().take(6).collect();
            us += f6.parse::<i64>().ok()?;
        }
    }
    Some(format_us(days_from_civil(y, mo, d) * 86_400_000_000 + us + (secs * 1e6).round_ties_even() as i64) + &tz)
}

/// "YYYY-MM-DDTHH:MM:SS.mmm" for microseconds since 1970 (milliseconds truncated, as Python).
pub fn format_us(us: i64) -> String {
    let (day, r) = (us.div_euclid(86_400_000_000), us.rem_euclid(86_400_000_000));
    let (y, m, d) = civil_from_days(day);
    let s = r / 1_000_000;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}", s / 3600, s / 60 % 60, s % 60, r % 1_000_000 / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(v: f64, d: f64, f0: f64, noise: f64, curved: bool) -> (Vec<f64>, Vec<f64>) {
        let mut seed = 12345u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5 };
        let p = [300.0, v, d, f0];
        let t: Vec<f64> = (0..2400).map(|i| i as f64 * 0.25).collect();
        let f = t.iter().map(|&x| {
            if curved {
                // Station at height d above a circle of radius R: range rate from the circular geometry.
                let (r, om) = (6.8e6, v / 6.8e6);
                let a = om * (x - 300.0);
                let (sx, sy) = (r * a.sin(), r * a.cos() - (r - d));
                let rr = (sx * sx + sy * sy).sqrt();
                let rdot = (sx * v * a.cos() - sy * v * a.sin()) / rr;
                f0 * (1.0 - rdot / C_LIGHT) + noise * rnd()
            } else {
                model_at(x, &p, C_LIGHT) + noise * rnd()
            }
        }).collect();
        (t, f)
    }

    #[test]
    fn straight_pass_recovered() {
        let (t, f) = synth(7660.0, 420e3, 137.5e6, 20.0, false);
        let r = run(&t, &f, None, C_LIGHT, None, 0.0).unwrap();
        assert!((r.d.v_m_s - 7660.0).abs() < 5.0, "v {}", r.d.v_m_s);
        assert!((r.d.d_m - 420e3).abs() < 1e3, "d {}", r.d.d_m);
        assert!((r.d.t0_s - 300.0).abs() < 0.1 && (r.d.f0_hz - 137.5e6).abs() < 2.0);
        assert!(!r.path.bending, "p {}", r.path.p_value);
        assert_eq!(r.path_word(), "no bending detected");
        assert!(r.err("v_m_s") > 0.0 && r.marks.dt_s > 0.0);
        assert!(r.to_json().contains("\"method_D\""));
    }

    #[test]
    fn curved_pass_flagged() {
        let (t, f) = synth(7400.0, 500e3, 137.5e6, 20.0, true);
        let r = run(&t, &f, None, C_LIGHT, None, 0.0).unwrap();
        assert!(r.path.bending, "p {}", r.path.p_value);
        assert_eq!(r.path_word(), "bending");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let _ = marks(&[0.0, 7000.0, 0.0, 1e8], C_LIGHT);
        let _ = marks(&[0.0, 0.0, 1e5, 1e8], C_LIGHT);
        let t: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
        let _ = run(&t, &vec![1e8; 40], None, C_LIGHT, None, 0.0);
        let _ = run(&t, &vec![f64::NAN; 40], None, C_LIGHT, None, 0.0);
    }

    #[test]
    fn f_survival() {
        for (f, a, b, want) in [(3.0, 7.0, 100.0, 0.006685915904656252), (1.0, 7.0, 50.5, 0.4423373963454105),
            (25.0, 7.0, 3593.0, 1.5522852429577195e-33), (0.5, 3.0, 10.0, 0.6906222455335574), (10.0, 7.0, 890.25, 4.565844331740496e-12)] {
            let got = f_sf(f, a, b);
            assert!(((got - want) / want).abs() < 1e-9, "{f} {a} {b}: {got} vs {want}");
        }
    }

    #[test]
    fn f_quantile_inverts_survival() {
        // scipy.stats.f.ppf(0.999, d1, d2)
        for (d1, d2, want) in [(7.0, 100.0, 3.8286209774499786), (7.0, 50.5, 4.214139772585483), (7.0, 3593.0, 3.483911393601435)] {
            let got = f_isf(0.001, d1, d2);
            assert!(((got - want) / want).abs() < 1e-6, "{d1} {d2}: {got} vs {want}");
        }
    }

    #[test]
    fn formatting() {
        assert_eq!(py_float(299792458.0), "299792458.0");
        assert_eq!(py_float(1.5e-7), "1.5e-07");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(fg(1.234e-5, 2), "1.2e-05");
        assert_eq!(fg(0.5, 2), "0.5");
        assert_eq!(fg(123.0, 2), "1.2e+02");
        assert_eq!(fcomma(-1234567.891, 1), "-1,234,567.9");
        assert_eq!(iso_add("2026-05-01T23:59:59.900+00:00", 0.2).unwrap(), "2026-05-02T00:00:00.100+00:00");
        assert_eq!(iso_add("2026-05-01T12:00:00", 1.0004).unwrap(), "2026-05-01T12:00:01.000");
    }
}
