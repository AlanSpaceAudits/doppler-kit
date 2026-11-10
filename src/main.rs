#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
mod app;

use doppler_kit::{fit, input::RawFormat, slr, trace};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const USAGE: &str = "Doppler Kit: speed and closest distance of a passing source from a Doppler or laser-ranging recording

Usage:
  doppler-kit                      open the window
  doppler-kit trace FILE [--center HZ] [--start ISO_UTC] [--span HZ] [--rate SPS] [--format cs16|cf32|cu8] [--swap-iq] [--out DIR]
                         [--band LO HI] [--burst] [--fm]
  doppler-kit fit TRACE.csv [--c M_S] [--f0 HZ] [--out DIR]
  doppler-kit run FILE [trace options] [--c M_S] [--f0 HZ]
  doppler-kit laser SRC [--pass N] [--out DIR]      SRC: CRD file (.np2/.npt/.crd) or EDC page URL

  --band LO HI: audio: track the tone between LO and HI Hz; IQ: search offsets LO to HI Hz from the centre.
  --fm: the signal is an FM carrier (ISS SSTV, voice): trace the centre of its energy.
  --burst: the signal comes in short bursts (FSK/packet satellites); without it, tried when the trace looks like bursts.
  Blank options are worked out from the file (SigMF meta, WAV header, SDR++ file name, SatNOGS observation,
  file time) and logged with their source; --span blank: a scan of the whole band; --band blank (audio): the moving tone.
  --c defaults to 299792458 (radio / light); use 343 for sound in air.";

pub const LASER_EXT: [&str; 5] = ["np2", "npt", "crd", "html", "htm"];

pub fn is_laser(src: &str) -> bool {
    src.starts_with("http://") || src.starts_with("https://")
        || Path::new(src).extension().and_then(|e| e.to_str()).is_some_and(|e| LASER_EXT.contains(&e.to_ascii_lowercase().as_str()))
}

/// Number as typed in a field: thousands separators allowed, blank means none.
pub fn num(s: &str) -> Result<Option<f64>, String> {
    let s = s.trim().replace([',', '_'], "");
    if s.is_empty() { return Ok(None); }
    s.parse::<f64>().ok().filter(|v| v.is_finite()).map(Some).ok_or(format!("not a number: {s}"))
}

/// Text of the Results tab after a trace and fit (doppler_gui.report).
pub fn report_text(s: &trace::TraceResult, f: &fit::FitResult, fit_json: &Path) -> String {
    let mut l = vec![
        format!("File            {}", s.file),
        format!("Points          {} (point noise {} Hz, swing {} Hz)", s.points.len(), trace::py_float(s.point_noise_hz), trace::py_float(s.swing_hz)),
        String::new(),
    ];
    l.extend(f.report_lines());
    if !s.auto.is_empty() {
        l.extend([String::new(), "AUTO VALUES (from the file; a value you type always wins)".into()]);
        l.extend(s.auto.iter().map(|(_, v)| format!("  {v}")));
    }
    if let Some(n) = &s.note { l.extend([String::new(), format!("NOTE: {n}")]); }
    l.extend([String::new(), "Files:".into()]);
    l.extend([&s.csv_path, &s.summary_path, &fit_json.to_path_buf()].iter().map(|p| format!("  {}", p.display())));
    l.join("\n") + "\n"
}

/// Text of the Results tab after a laser-ranging run (doppler_gui.laser_report).
pub fn laser_text(r: &slr::SlrRun, csv: &Path) -> String {
    let mut l = r.report_lines();
    l.extend([String::new(), "Files:".into(), format!("  {}", csv.display())]);
    l.join("\n") + "\n"
}

/// The one-shot triangle of a pass: (t from closest approach, range, along-track distance) for the used shot farthest in time.
pub fn triangle(p: &slr::Pass, f: &slr::SlrFit) -> (f64, f64, f64) {
    let k = (0..p.t.len()).fold(0, |b, i| {
        let w = |j: usize| if f.used[j] { (p.t[j] - f.t0).abs() } else { 0.0 };
        if w(i) > w(b) { i } else { b }
    });
    let (tk, rk) = (p.t[k] - f.t0, p.r[k]);
    (tk, rk, (rk * rk - f.d_km * f.d_km).max(0.0).sqrt())
}

struct Args { pos: Vec<String>, kv: HashMap<String, String>, swap: bool, band: Option<(f64, f64)>, burst: bool, fm: bool }

impl Args {
    fn parse(a: &[String], valued: &[&str]) -> Result<Args, String> {
        let (mut pos, mut kv, mut swap, mut band, mut burst, mut fm) = (vec![], HashMap::new(), false, None, false, false);
        let traces = valued.contains(&"span");
        let mut it = a.iter();
        while let Some(x) = it.next() {
            if x == "--swap-iq" {
                swap = true;
            } else if traces && x == "--burst" {
                burst = true;
            } else if traces && x == "--fm" {
                fm = true;
            } else if traces && x == "--band" {
                let mut v = || -> Result<f64, String> {
                    it.next().and_then(|s| num(s).ok().flatten()).ok_or("--band needs two numbers: LO HI".into())
                };
                band = Some((v()?, v()?));
            } else if let Some(k) = x.strip_prefix("--") {
                let (k, v) = match k.split_once('=') {
                    Some((k, v)) => (k.to_string(), v.to_string()),
                    None => (k.to_string(), it.next().ok_or(format!("--{k} needs a value"))?.clone()),
                };
                if !valued.contains(&k.as_str()) { return Err(format!("unknown option --{k}")); }
                kv.insert(k, v);
            } else {
                pos.push(x.clone());
            }
        }
        if pos.len() != 1 { return Err("give exactly one input file".into()); }
        Ok(Args { pos, kv, swap, band, burst, fm })
    }
    fn s(&self, k: &str) -> Option<String> { self.kv.get(k).cloned() }
    fn n(&self, k: &str) -> Result<Option<f64>, String> { self.kv.get(k).map_or(Ok(None), |v| num(v)) }
    fn out(&self) -> Option<PathBuf> { self.s("out").map(PathBuf::from) }
    fn trace_opts(&self) -> Result<trace::TraceOpts, String> {
        Ok(trace::TraceOpts {
            center: self.n("center")?,
            start: self.s("start"),
            span: self.n("span")?,
            rate: self.n("rate")?,
            format: self.s("format").map(|f| f.parse::<RawFormat>()).transpose()?,
            swap_iq: self.swap,
            out: self.out(),
            band: self.band,
            burst: self.burst,
            fm: self.fm,
        })
    }
    fn c(&self) -> Result<f64, String> {
        let c = self.n("c")?.unwrap_or(fit::C_LIGHT);
        if c > 0.0 { Ok(c) } else { Err("the wave speed must be above 0".into()) }
    }
}

const TRACE_OPTS: [&str; 6] = ["center", "start", "span", "rate", "format", "out"];

fn laser(src: &str, which: usize, out: Option<&Path>, report: bool) -> Result<(), String> {
    let r = slr::run(src)?;
    let csv = r.write_csv(src, out)?;
    if report {
        print!("{}", laser_text(&r, &csv));
        return Ok(());
    }
    r.log_lines().iter().for_each(|l| println!("{l}"));
    let (p, f) = r.passes.get(which).ok_or(format!("--pass {which}: the file has {} passes", r.passes.len()))?;
    let (tk, rk, along) = triangle(p, f);
    println!("pass {which} one shot: t = {tk:+.1} s, r = {rk:.1} km, √(r² − d²) = {along:.1} km, v = {along:.1} / {:.1} = {:.4} km/s", tk.abs(), along / tk.abs());
    println!("wrote {}", csv.display());
    Ok(())
}

fn cli(a: &[String]) -> Result<(), String> {
    let mut log = |s: &str| println!("{s}");
    match a[0].as_str() {
        "trace" => {
            let x = Args::parse(&a[1..], &TRACE_OPTS)?;
            trace::run(Path::new(&x.pos[0]), &x.trace_opts()?, &mut log)?;
        }
        "fit" => {
            let x = Args::parse(&a[1..], &["c", "f0", "out"])?;
            let (r, p) = fit::run_file(&x.pos[0], x.c()?, x.n("f0")?, x.out().as_deref())?;
            print!("{}", r.log_text());
            println!("wrote {}", p.display());
        }
        "run" => {
            let x = Args::parse(&a[1..], &[&TRACE_OPTS[..], &["c", "f0"]].concat())?;
            if is_laser(&x.pos[0]) {
                return laser(&x.pos[0], 0, x.out().as_deref(), true);
            }
            let mut elog = |s: &str| eprintln!("{s}");
            let s = trace::run(Path::new(&x.pos[0]), &x.trace_opts()?, &mut elog)?;
            if s.points.len() < 20 { return Err("too few points on the trace to fit".into()); }
            let (f, fp) = fit::run_file(&s.csv_path, x.c()?, x.n("f0")?, x.out().as_deref())?;
            eprint!("{}", f.log_text());
            print!("{}", report_text(&s, &f, &fp));
        }
        "laser" => {
            let x = Args::parse(&a[1..], &["pass", "out"])?;
            let which = x.s("pass").map_or(Ok(0), |v| v.trim().parse::<usize>().map_err(|_| format!("--pass {v}: not a pass number")))?;
            laser(&x.pos[0], which, x.out().as_deref(), false)?;
        }
        "-h" | "--help" | "help" => println!("{USAGE}"),
        o => return Err(format!("unknown command '{o}'\n\n{USAGE}")),
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        if let Err(e) = app::run() {
            eprintln!("window: {e}");
            std::process::exit(1);
        }
        return;
    }
    // The Windows release build has no console of its own; borrow the one it was started from for the CLI.
    #[cfg(all(windows, not(debug_assertions)))]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn AttachConsole(pid: u32) -> i32;
        }
        unsafe { AttachConsole(u32::MAX) };
    }
    if let Err(e) = cli(&a) {
        eprintln!("Stopped: {e}");
        std::process::exit(1);
    }
}
