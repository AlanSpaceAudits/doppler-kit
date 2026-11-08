//! Speed and closest distance of a satellite from laser ranging alone (port of slr_fit.py).
//! Straight path past the station: r(t)^2 = d^2 + (v (t - t0))^2, fitted within +-3 min of the closest shot.

use crate::fit::{days_from_civil, format_us};
use crate::lm;
use std::path::{Path, PathBuf};

pub const C_KM_S: f64 = 299792.458;
pub const WINDOW_S: f64 = 180.0;

#[derive(Clone, Debug)]
pub struct Pass {
    pub station: String,
    pub sat: String,
    /// Days since 1970-01-01 of the H4 start date (UTC).
    pub day: i64,
    /// Seconds of day (past midnight rolled over), sorted.
    pub t: Vec<f64>,
    /// Range in km: round-trip time x c / 2.
    pub r: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SlrFit { pub d_km: f64, pub v_km_s: f64, pub t0: f64, pub rms_m: f64, pub n: usize, pub used: Vec<bool> }

pub struct SlrRun { pub name: String, pub passes: Vec<(Pass, SlrFit)>, pub rows: Vec<String> }

fn strip_html(txt: &str) -> String {
    let txt = txt.replace("&nbsp;", " ");
    let mut out = String::with_capacity(txt.len());
    let mut rest = txt.as_str();
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        match rest[i + 1..].find('>') {
            Some(0) => { out.push('<'); rest = &rest[i + 1..]; }
            Some(j) => { out.push('\n'); rest = &rest[i + j + 2..]; }
            None => { out.push_str(&rest[i..]); rest = ""; }
        }
    }
    out.push_str(rest);
    out
}

/// Passes from CRD text or an EDC page (HTML stripped); passes with fewer than 5 shots are dropped.
pub fn read_crd(text: &str) -> Vec<Pass> {
    let txt = strip_html(text);
    let (mut passes, mut st, mut sat): (Vec<Pass>, String, String) = (Vec::new(), String::new(), String::new());
    let mut cur = false;
    for line in txt.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let Some(&h) = w.first() else { continue };
        match h {
            "h2" | "H2" if w.len() > 1 => st = w[1].to_string(),
            "h3" | "H3" if w.len() > 1 => sat = w[1].to_string(),
            "h4" | "H4" if w.len() > 4 => {
                if let (Ok(y), Ok(m), Ok(d)) = (w[2].parse(), w[3].parse(), w[4].parse()) {
                    passes.push(Pass { station: st.clone(), sat: sat.clone(), day: days_from_civil(y, m, d), t: vec![], r: vec![] });
                    cur = true;
                }
            }
            "11" if cur && w.len() > 2 => {
                if let (Ok(t), Ok(rt)) = (w[1].parse::<f64>(), w[2].parse::<f64>()) {
                    let p = passes.last_mut().unwrap();
                    p.t.push(t);
                    p.r.push(rt * C_KM_S / 2.0);
                }
            }
            _ => {}
        }
    }
    passes.into_iter().filter(|p| p.t.len() >= 5).map(|mut p| {
        let t0 = p.t[0];
        for t in p.t.iter_mut() { if *t < t0 - 43200.0 { *t += 86400.0; } }
        let mut o: Vec<usize> = (0..p.t.len()).collect();
        o.sort_by(|&a, &b| p.t[a].total_cmp(&p.t[b]));
        p.t = o.iter().map(|&i| p.t[i]).collect();
        p.r = o.iter().map(|&i| p.r[i]).collect();
        p
    }).collect()
}

fn argmin(x: &[f64]) -> usize {
    (0..x.len()).fold(0, |b, i| if x[i] < x[b] { i } else { b })
}

pub fn fit(t: &[f64], r: &[f64]) -> SlrFit {
    let tc = t[argmin(r)];
    let used: Vec<bool> = t.iter().map(|x| (x - tc).abs() <= WINDOW_S).collect();
    let tt: Vec<f64> = t.iter().zip(&used).filter(|u| *u.1).map(|u| *u.0).collect();
    let rr: Vec<f64> = r.iter().zip(&used).filter(|u| *u.1).map(|u| *u.0).collect();
    let k = argmin(&rr);
    let q = lm::solve(|p: &[f64]| tt.iter().zip(&rr).map(|(t, r)| p[0].hypot(p[1] * (t - p[2])) - r).collect(), &[rr[k], 7.0, tt[k]]);
    SlrFit { d_km: q.x[0].abs(), v_km_s: q.x[1].abs(), t0: q.x[2], rms_m: q.rms() * 1e3, n: tt.len(), used }
}

/// Per-shot speed v = sqrt(r^2 - d^2) / |t - t0| for shots more than 20 s from closest approach: (t - t0, v).
pub fn speeds(p: &Pass, f: &SlrFit) -> Vec<(f64, f64)> {
    p.t.iter().zip(&p.r).map(|(t, r)| (t - f.t0, *r)).filter(|(t, _)| t.abs() > 20.0)
        .map(|(t, r)| (t, (r * r - f.d_km * f.d_km).max(0.0).sqrt() / t.abs())).collect()
}

/// Range of the fitted straight path at time `t` (s of day).
pub fn model(f: &SlrFit, t: f64) -> f64 { f.d_km.hypot(f.v_km_s * (t - f.t0)) }

/// Closest-approach time as Python isoformat(timespec="milliseconds") with +00:00.
pub fn closest_utc(p: &Pass, f: &SlrFit) -> String {
    format_us(p.day * 86_400_000_000 + (f.t0 * 1e6).round_ties_even() as i64) + "+00:00"
}

fn is_url(s: &str) -> bool { s.starts_with("http://") || s.starts_with("https://") }

pub fn fetch(src: &str) -> Result<String, String> {
    if is_url(src) {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(60))).build().into();
        let mut resp = agent.get(src).header("User-Agent", "doppler-kit").call().map_err(|e| format!("{src}: {e}"))?;
        let bytes = resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        std::fs::read(src).map(|b| String::from_utf8_lossy(&b).into_owned()).map_err(|e| format!("{src}: {e}"))
    }
}

/// Output stem: "edc_<digits of last URL segment>" for a URL, else the file stem.
pub fn out_name(src: &str) -> String {
    if is_url(src) {
        let last = src.trim_end_matches('/').rsplit('/').next().unwrap_or("");
        format!("edc_{}", last.chars().filter(|c| c.is_ascii_digit()).collect::<String>())
    } else {
        Path::new(src).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    }
}

/// Reads and fits every pass; `rows` are the CSV data lines.
pub fn run(src: &str) -> Result<SlrRun, String> {
    let passes = read_crd(&fetch(src)?);
    if passes.is_empty() { return Err("no passes with 5 or more '11' records found".into()); }
    let passes: Vec<(Pass, SlrFit)> = passes.into_iter().map(|p| { let f = fit(&p.t, &p.r); (p, f) }).collect();
    let rows = passes.iter().enumerate().map(|(i, (p, f))| format!("{i},{},{},{},{:.4},{:.1},{},{:.0}",
        p.sat, p.station, closest_utc(p, f), f.v_km_s, f.d_km, f.n, f.rms_m)).collect();
    Ok(SlrRun { name: out_name(src), passes, rows })
}

impl SlrRun {
    /// The per-pass lines slr_fit.py prints.
    pub fn log_lines(&self) -> Vec<String> {
        self.passes.iter().enumerate().map(|(i, (p, f))| format!("pass {i}: {} past {}, closest {}, v {:.4} km/s, d {:.1} km, {} shots, miss {:.0} m rms",
            p.sat, p.station, closest_utc(p, f), f.v_km_s, f.d_km, f.n, f.rms_m)).collect()
    }
    /// GUI laser_report body (without the Files block).
    pub fn report_lines(&self) -> Vec<String> {
        let mut l = vec!["LASER RANGING: distance of each shot = round-trip time × c/2; straight path r² = d² + (v·t)²".to_string(), String::new()];
        l.extend(self.passes.iter().enumerate().map(|(i, (p, f))| format!("pass {i}: {} past {}: v {:.4} km/s, d {:.1} km, {} shots, fit miss {:.0} m rms",
            p.sat, p.station, f.v_km_s, f.d_km, f.n, f.rms_m)));
        l
    }
    pub fn csv(&self) -> String {
        format!("pass,satellite,station,closest_utc,v_km_s,d_km,shots,fit_miss_m_rms\n{}\n", self.rows.join("\n"))
    }
    /// Writes `<name>_slr.csv` into `out` (default: next to the file, or the current folder for a URL).
    pub fn write_csv(&self, src: &str, out: Option<&Path>) -> Result<PathBuf, String> {
        let dir = match out {
            Some(o) => o.to_path_buf(),
            None if is_url(src) => std::env::current_dir().map_err(|e| e.to_string())?,
            None => std::fs::canonicalize(src).ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default(),
        };
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let p = dir.join(format!("{}_slr.csv", self.name));
        std::fs::write(&p, self.csv()).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_crd_recovered() {
        let (d, v, tc) = (500.0, 7.4, 86400.0 - 60.0);
        let mut s = String::from("<html><body>H1 CRD 2 2024 01 05 12<br>\nH2 TEST 7000 01 01 4<br/>\nH3 sat1 1234 5678 9 0 1<br>\nH4 1 2024 01 05 23 55 00 2024 01 06 00 05 00 0 0 0 0 1 0 2 0<br>\n");
        let mut t = -150.0;
        while t <= 150.0 {
            let r = (d * d + (v * t) * (v * t) as f64).sqrt();
            s.push_str(&format!("11&nbsp;{:.7} {:.12} 1 2 3<br>\n", (tc + t) % 86400.0, 2.0 * r / C_KM_S));
            t += 5.0;
        }
        s.push_str("H8<br></body></html>");
        let p = read_crd(&s);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].sat, "sat1");
        let f = fit(&p[0].t, &p[0].r);
        assert!((f.v_km_s - v).abs() < 1e-3 && (f.d_km - d).abs() < 0.1, "{f:?}");
        assert!((f.t0 - tc).abs() < 0.01);
        assert_eq!(closest_utc(&p[0], &f), "2024-01-05T23:59:00.000+00:00");
        let sp = speeds(&p[0], &f);
        assert!(sp.iter().all(|(_, s)| (s - v).abs() < 1e-3));
    }

    #[test]
    fn html_strip() {
        assert_eq!(strip_html("a<b>c&nbsp;d<>e<"), "a\nc d<>e<");
    }
}
