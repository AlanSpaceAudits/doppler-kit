//! The Doppler Kit window (port of doppler_gui.py).

use doppler_kit::{fit, input::RawFormat, slr, trace};
use eframe::egui::{self, Align2, Color32, ColorImage, Id, RichText, TextStyle, TextureHandle, TextureOptions};
use egui_plot::{Arrows, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, PlotPoint, PlotUi, Points, Text, VLine};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

const BLUE: Color32 = Color32::from_rgb(0x63, 0xb3, 0xed);
const RED: Color32 = Color32::from_rgb(0xfc, 0x81, 0x81);
const GOLD: Color32 = Color32::from_rgb(0xf6, 0xc3, 0x4a);
const YELLOW: Color32 = Color32::from_rgb(0xff, 0xd2, 0x3f);
const GREY: Color32 = Color32::from_gray(150);

pub fn run() -> Result<(), String> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]).with_min_inner_size([900.0, 600.0]).with_title("Doppler Kit"),
        ..Default::default()
    };
    eframe::run_native("Doppler Kit", opts, Box::new(|cc| {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        Ok(Box::new(App::default()))
    }))
    .map_err(|e| e.to_string())
}

#[derive(PartialEq, Clone, Copy)]
enum Wave { Radio, Sound, Other }

#[derive(PartialEq, Clone, Copy)]
enum Tab { Results, Waterfall, SCurve, Marked, Laser }

enum Msg {
    Log(String),
    Trace(Box<trace::TraceResult>),
    Fit(Box<fit::FitResult>, PathBuf),
    Laser(Box<slr::SlrRun>, String, PathBuf),
    End,
}

struct Job { path: String, opts: trace::TraceOpts, c: f64, f0: Option<f64> }

struct App {
    file: String, centre: String, start: String, wave: Wave, c_other: String, f0: String, span: String,
    rate: String, fmt: RawFormat, swap: bool, which: String, out: String,
    log: String, results: String, tab: Tab, rx: Option<Receiver<Msg>>,
    trace: Option<trace::TraceResult>, fit: Option<(fit::FitResult, PathBuf)>, laser: Option<(slr::SlrRun, String)>,
    wf_tex: Option<TextureHandle>,
}

impl Default for App {
    fn default() -> Self {
        App {
            file: String::new(), centre: String::new(), start: String::new(), wave: Wave::Radio, c_other: String::new(),
            f0: String::new(), span: "15000".into(), rate: String::new(), fmt: RawFormat::Cs16, swap: false, which: "0".into(),
            out: String::new(), log: String::new(), results: String::new(), tab: Tab::Results, rx: None,
            trace: None, fit: None, laser: None, wf_tex: None,
        }
    }
}

/// First (\d{6,})Hz in a file name, as in doppler_gui.pick.
fn centre_from_name(name: &str) -> Option<String> {
    let b = name.as_bytes();
    let mut i = 0;
    while let Some(p) = name[i..].find("Hz").map(|p| p + i) {
        let mut s = p;
        while s > 0 && b[s - 1].is_ascii_digit() { s -= 1; }
        if p - s >= 6 { return Some(name[s..p].to_string()); }
        i = p + 2;
    }
    None
}

fn work(job: Job, tx: &Sender<Msg>, ctx: &egui::Context) -> Result<(), String> {
    let send = |m: Msg| { let _ = tx.send(m); ctx.request_repaint(); };
    let out = job.opts.out.clone();
    if crate::is_laser(&job.path) {
        let r = slr::run(&job.path)?;
        let csv = r.write_csv(&job.path, out.as_deref())?;
        r.log_lines().into_iter().for_each(|l| send(Msg::Log(l + "\n")));
        send(Msg::Laser(Box::new(r), job.path, csv));
        return Ok(());
    }
    let s = trace::run(Path::new(&job.path), &job.opts, &mut |l: &str| send(Msg::Log(format!("{l}\n"))))?;
    let (csv, n) = (s.csv_path.clone(), s.points.len());
    send(Msg::Trace(Box::new(s)));
    if n < 20 { return Err("too few points on the trace to fit".into()); }
    let (f, fp) = fit::run_file(&csv, job.c, job.f0, out.as_deref())?;
    send(Msg::Log(f.log_text()));
    send(Msg::Fit(Box::new(f), fp));
    Ok(())
}

fn viridis(x: f32) -> Color32 {
    const S: [[u8; 3]; 9] = [[68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142], [38, 130, 142],
        [31, 158, 137], [53, 183, 121], [110, 206, 88], [253, 231, 37]];
    let x = if x.is_finite() { x.clamp(0.0, 1.0) * 8.0 } else { 0.0 };
    let (i, u) = ((x as usize).min(7), x - (x as usize).min(7) as f32);
    let m = |k: usize| (S[i][k] as f32 + (S[i + 1][k] as f32 - S[i][k] as f32) * u) as u8;
    Color32::from_rgb(m(0), m(1), m(2))
}

/// Waterfall as an image on viridis 0..15 dB, time upwards; blocks are max-pooled to fit the GPU texture limit.
fn waterfall_image(w: &trace::Waterfall, max_side: usize) -> ColorImage {
    let (nt, nf) = (w.t.len(), w.f.len());
    let (by, bx) = (nt.div_ceil(max_side).max(1), nf.div_ceil(max_side).max(1));
    let (h, wd) = (nt.div_ceil(by), nf.div_ceil(bx));
    let mut px = Vec::with_capacity(h * wd);
    for r in (0..h).rev() {
        for c in 0..wd {
            let mut m = f32::NEG_INFINITY;
            for i in r * by..((r + 1) * by).min(nt) {
                for j in c * bx..((c + 1) * bx).min(nf) { m = m.max(w.db[i * nf + j]); }
            }
            px.push(viridis(m / 15.0));
        }
    }
    ColorImage::new([wd, h], px)
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

fn field(ui: &mut egui::Ui, l: &str, s: &mut String) {
    ui.label(l);
    ui.add(egui::TextEdit::singleline(s).desired_width(250.0));
    ui.end_row();
}

fn label(p: &mut PlotUi, x: f64, y: f64, s: impl Into<String>, c: Color32, a: Align2) {
    p.add(Text::new("", PlotPoint::new(x, y), RichText::new(s.into()).color(c)).anchor(a));
}

fn seg(p: &mut PlotUi, a: [f64; 2], b: [f64; 2], c: Color32, w: f32, dotted: bool) {
    let l = Line::new("", vec![a, b]).color(c).width(w);
    p.add(if dotted { l.style(LineStyle::dotted_dense()) } else { l });
}

/// Double-headed arrow between two points.
fn span_arrow(p: &mut PlotUi, a: [f64; 2], b: [f64; 2], c: Color32) {
    let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    p.add(Arrows::new("", vec![m, m], vec![a, b]).color(c).tip_length(10.0));
}

impl App {
    fn busy(&self) -> bool { self.rx.is_some() }

    fn num(&mut self, s: &str, what: &str) -> Option<Option<f64>> {
        match crate::num(s) {
            Ok(v) => Some(v),
            Err(e) => { self.log += &format!("Stopped: {what}: {e}\n"); None }
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        self.log.clear();
        self.results.clear();
        let path = self.file.trim().to_string();
        if path.is_empty() { self.log += "Stopped: pick a recording first\n"; return; }
        let (centre, centre_s) = (self.centre.clone(), self.c_other.clone());
        let (f0s, spans, rates) = (self.f0.clone(), self.span.clone(), self.rate.clone());
        let Some(center) = self.num(&centre, "centre") else { return };
        let Some(f0) = self.num(&f0s, "f0") else { return };
        let Some(span) = self.num(&spans, "search span") else { return };
        let Some(rate) = self.num(&rates, "sample rate") else { return };
        let c = match self.wave {
            Wave::Radio => fit::C_LIGHT,
            Wave::Sound => 343.0,
            Wave::Other => match self.num(&centre_s, "wave speed") {
                Some(Some(c)) if c > 0.0 => c,
                Some(_) => { self.log += "Stopped: give the wave speed\n"; return; }
                None => return,
            },
        };
        let out = Some(self.out.trim()).filter(|s| !s.is_empty()).map(PathBuf::from);
        let start = Some(self.start.trim().to_string()).filter(|s| !s.is_empty());
        let opts = trace::TraceOpts { center, start, span: span.unwrap_or(15000.0), rate, format: self.fmt, swap_iq: self.swap, out };
        self.trace = None;
        self.fit = None;
        self.laser = None;
        self.wf_tex = None;
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();
        let job = Job { path, opts, c, f0 };
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(job, &tx, &ctx)));
            let msg = match r {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(format!("Stopped: {e}\n")),
                Err(p) => Some(format!("Stopped: internal error: {}\n",
                    p.downcast_ref::<String>().map(String::as_str).or(p.downcast_ref::<&str>().copied()).unwrap_or("panic"))),
            };
            if let Some(m) = msg { let _ = tx.send(Msg::Log(m)); }
            let _ = tx.send(Msg::End);
            ctx.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        let msgs: Vec<Msg> = rx.try_iter().collect();
        for m in msgs {
            match m {
                Msg::Log(s) => self.log += &s,
                Msg::Trace(t) => { self.trace = Some(*t); self.tab = Tab::Waterfall; }
                Msg::Fit(f, p) => {
                    if let Some(t) = &self.trace { self.results = crate::report_text(t, &f, &p); }
                    self.fit = Some((*f, p));
                    self.tab = Tab::Results;
                }
                Msg::Laser(r, src, csv) => {
                    self.results = crate::laser_text(&r, &csv);
                    self.laser = Some((*r, src));
                    self.tab = Tab::Results;
                }
                Msg::End => self.rx = None,
            }
        }
    }

    fn save(&mut self) {
        let dir = Some(self.out.trim()).filter(|s| !s.is_empty()).map(PathBuf::from);
        let r: Result<Vec<PathBuf>, String> = (|| {
            if let Some((run, src)) = &self.laser { return Ok(vec![run.write_csv(src, dir.as_deref())?]); }
            let t = self.trace.as_ref().ok_or("nothing to save yet")?;
            let d = dir.clone().or_else(|| t.csv_path.parent().map(Path::to_path_buf)).unwrap_or_default();
            std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
            let (csv, js) = (d.join(t.csv_path.file_name().unwrap_or_default()), d.join(t.summary_path.file_name().unwrap_or_default()));
            t.write_csv(&csv)?;
            t.write_summary_json(&js)?;
            let mut v = vec![csv, js];
            if let Some((f, _)) = &self.fit {
                let p = d.join(format!("{}_fit.json", fit::out_name(&t.csv_path)));
                std::fs::write(&p, f.to_json()).map_err(|e| format!("{}: {e}", p.display()))?;
                v.push(p);
            }
            Ok(v)
        })();
        match r {
            Ok(v) => v.iter().for_each(|p| self.log += &format!("saved {}\n", p.display())),
            Err(e) => self.log += &format!("Stopped: {e}\n"),
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("controls").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Recording");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.file).desired_width(250.0));
                if ui.button("Browse…").clicked() {
                    let p = rfd::FileDialog::new()
                        .add_filter("Recordings", &["wav", "raw", "cs16", "cf32", "cu8", "sigmf-data", "ogg", "oga", "mp3", "flac", "m4a", "opus", "aac"])
                        .add_filter("Laser ranging", &["np2", "npt", "crd", "html", "htm"])
                        .add_filter("All files", &["*"])
                        .pick_file();
                    if let Some(p) = p {
                        if let Some(c) = p.file_name().and_then(|n| centre_from_name(&n.to_string_lossy())) { self.centre = c; }
                        self.file = p.display().to_string();
                    }
                }
            });
            ui.end_row();
            field(ui, "Centre / dial frequency (Hz)", &mut self.centre);
            field(ui, "Start time UTC (optional)", &mut self.start);
            ui.label("Wave");
            ui.vertical(|ui| {
                ui.radio_value(&mut self.wave, Wave::Radio, "radio / light (299,792,458 m/s)");
                ui.radio_value(&mut self.wave, Wave::Sound, "sound in air (343 m/s)");
                ui.radio_value(&mut self.wave, Wave::Other, "other (m/s below)");
            });
            ui.end_row();
            field(ui, "Wave speed m/s (other)", &mut self.c_other);
            field(ui, "Fix transmitted f₀ (Hz, optional)", &mut self.f0);
            field(ui, "Search ± Hz", &mut self.span);
            field(ui, "Raw IQ sample rate (S/s)", &mut self.rate);
            ui.label("Raw IQ format");
            egui::ComboBox::from_id_salt("fmt").selected_text(format!("{:?}", self.fmt).to_lowercase()).show_ui(ui, |ui| {
                for f in [RawFormat::Cs16, RawFormat::Cf32, RawFormat::Cu8] {
                    ui.selectable_value(&mut self.fmt, f, format!("{f:?}").to_lowercase());
                }
            });
            ui.end_row();
            ui.label("Q stored before I");
            ui.checkbox(&mut self.swap, "swap IQ");
            ui.end_row();
            field(ui, "Laser file: pass to plot", &mut self.which);
            ui.label("Output folder (blank: next to file)");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.out).desired_width(250.0));
                if ui.button("…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() { self.out = p.display().to_string(); }
                }
            });
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let w = ui.available_width();
            if ui.add_enabled(!self.busy(), egui::Button::new(if self.busy() { "Running…" } else { "Run" }).min_size([w * 0.7, 28.0].into())).clicked() {
                self.start(ui.ctx());
            }
            let can = !self.busy() && (self.trace.is_some() || self.laser.is_some());
            if ui.add_enabled(can, egui::Button::new("Save").min_size([ui.available_width(), 28.0].into())).clicked() { self.save(); }
        });
        ui.add_space(6.0);
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            ui.add(egui::TextEdit::multiline(&mut self.log.as_str()).font(TextStyle::Monospace).desired_width(f32::INFINITY));
        });
    }

    fn wf_texture(&mut self, ctx: &egui::Context) -> Option<TextureHandle> {
        if self.wf_tex.is_none() {
            let t = self.trace.as_ref()?;
            if t.waterfall.t.len() < 2 || t.waterfall.f.len() < 2 { return None; }
            let side = ctx.input(|i| i.max_texture_side).min(8192);
            self.wf_tex = Some(ctx.load_texture("waterfall", waterfall_image(&t.waterfall, side), TextureOptions::NEAREST));
        }
        self.wf_tex.clone()
    }

    fn wf_image(&self, tex: &TextureHandle) -> PlotImage {
        let w = &self.trace.as_ref().unwrap().waterfall;
        let (f0, f1, t0, t1) = (w.f[0] / 1e3, w.f[w.f.len() - 1] / 1e3, w.t[0], w.t[w.t.len() - 1]);
        PlotImage::new("waterfall", tex.id(), PlotPoint::new((f0 + f1) / 2.0, (t0 + t1) / 2.0), [(f1 - f0) as f32, (t1 - t0) as f32])
    }

    fn waterfall_tab(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = self.wf_texture(ui.ctx()) else { ui.label("Run a recording to see its waterfall."); return };
        let t = self.trace.as_ref().unwrap();
        ui.label(format!("{}   centre {:.6} MHz   (scroll to zoom, drag to pan, double-click to reset)", t.name, t.centre_hz / 1e6));
        let img = self.wf_image(&tex);
        let tr: Vec<[f64; 2]> = t.points.iter().map(|p| [p.f_offset / 1e3, p.t_s]).collect();
        Plot::new("wf").x_axis_label("offset from centre (kHz)").y_axis_label("time from start of recording (s)")
            .legend(Legend::default()).show(ui, |p| {
                p.add(img);
                p.add(Points::new("measured trace", tr).color(RED).radius(1.2));
            });
    }

    fn scurve_tab(&mut self, ui: &mut egui::Ui) {
        let Some((r, _)) = &self.fit else { ui.label("Run a recording to see the S-curve."); return };
        let (m, [t0, v, d, f0]) = (&r.marks, r.p);
        let (th, fh) = (m.half_plateau_t_s, m.half_plateau_hz);
        let (lo, hi) = (r.t[0].min(th[0]) - 20.0, r.t[r.t.len() - 1].max(th[1]) + 20.0);
        let k = |x: f64| (x - f0) / 1e3;
        let yb = k(m.f_plus_hz) * 1.12 + 0.1;
        ui.label(RichText::new("Doppler S-curve: plateaus, middle, half-plateau points and width").strong());
        ui.label(RichText::new(format!(
            "v = c (f₊ − f₋)/(f₊ + f₋) = {:.3} km/s\nd = Δt · v · √3/2 = {:.1} km\nd = f₀ v² / (c · |slope at middle|) = {:.1} km   (slope {:.1} Hz/s)\nfull fit: v {:.3} km/s, d {:.1} km, f₀ {} Hz",
            m.v_plateaus_m_s / 1e3, m.d_width_m / 1e3, m.d_slope_m / 1e3, m.slope_hz_s, v / 1e3, d / 1e3, fit::fcomma(f0, 1))).monospace());
        let h = ui.available_height() - 40.0;
        let tt = linspace(lo, hi, 1500);
        let curve: Vec<[f64; 2]> = tt.iter().zip(r.curve(&tt)).map(|(t, f)| [t - t0, k(f)]).collect();
        let meas: Vec<[f64; 2]> = r.t.iter().zip(&r.f).map(|(t, f)| [t - t0, k(*f)]).collect();
        let link = Id::unique("scurve_x");
        Plot::new("scurve").height(h * 0.72).link_axis(link, [true, false]).legend(Legend::default())
            .y_axis_label("received − f₀ (kHz)")
            .default_x_bounds(lo - t0, hi - t0 + (hi - lo) * 0.18)
            .default_y_bounds(k(m.f_minus_hz) * 1.25 - 0.1, yb + k(m.f_plus_hz).abs() * 0.15 + 0.3)
            .show(ui, |p| {
                p.add(Points::new("measured", meas).color(BLUE).radius(1.5));
                p.add(Line::new("straight-pass fit", curve).color(RED).width(1.5));
                for (f, l) in [(m.f_plus_hz, "plateau f₊ (coming)"), (m.f_minus_hz, "plateau f₋ (going)"), (f0, "middle f₀")] {
                    p.add(HLine::new("", k(f)).color(GREY).style(LineStyle::dashed_dense()));
                    label(p, hi - t0, k(f), format!(" {l}"), Color32::LIGHT_GRAY, Align2::LEFT_CENTER);
                }
                p.add(VLine::new("", 0.0).color(GREY).style(LineStyle::dotted_dense()));
                for i in 0..2 {
                    let (x, y) = (th[i] - t0, k(fh[i]));
                    p.add(Points::new("", vec![[x, y]]).shape(MarkerShape::Circle).filled(false).radius(9.0).color(GOLD));
                    let txt = format!("half-plateau {}\nrange rate {}v/2", i + 1, if i == 0 { '−' } else { '+' });
                    label(p, x, y, txt, GOLD, if i == 0 { Align2::RIGHT_TOP } else { Align2::LEFT_BOTTOM });
                    seg(p, [x, y], [x, yb], GOLD, 1.0, true);
                }
                span_arrow(p, [th[0] - t0, yb], [th[1] - t0, yb], GOLD);
                label(p, (th[0] + th[1]) / 2.0 - t0, yb, format!("Δt = {:.2} s", m.dt_s), GOLD, Align2::CENTER_BOTTOM);
            });
        ui.label(format!("left over: {} path (smooth pattern {:.1} Hz rms, scatter {:.1} Hz rms, p = {})",
            r.path_word(), r.path.pattern_rms_hz, r.path.scatter_rms_hz, fit::fg(r.path.p_value, 2)));
        let left: Vec<[f64; 2]> = r.t.iter().zip(&r.left).map(|(t, l)| [t - t0, *l]).collect();
        Plot::new("left").height(ui.available_height()).link_axis(link, [true, false])
            .x_axis_label("time from closest approach (s)").y_axis_label("left over (Hz)")
            .show(ui, |p| {
                p.add(Points::new("left over", left).color(BLUE).radius(1.5));
                p.add(HLine::new("", 0.0).color(Color32::LIGHT_GRAY));
            });
    }

    fn marked_tab(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = self.wf_texture(ui.ctx()) else { ui.label("Run a recording to see the marked waterfall."); return };
        let Some((r, _)) = &self.fit else { ui.label("The fit did not finish."); return };
        let t = self.trace.as_ref().unwrap();
        let (w, m, [_, v, d, f0], centre) = (&t.waterfall, &r.marks, r.p, t.centre_hz);
        let x = |f: f64| (f - centre) / 1e3;
        let (wt0, wt1, wf0, wf1) = (w.t[0], w.t[w.t.len() - 1], w.f[0], w.f[w.f.len() - 1]);
        ui.label(RichText::new(format!("Waterfall with the half-plateau points: v {:.3} km/s, d {:.1} km", v / 1e3, d / 1e3)).strong());
        let ty = linspace(wt0, wt1, 1500);
        let curve: Vec<[f64; 2]> = ty.iter().zip(r.curve(&ty)).map(|(t, f)| [x(f), *t]).collect();
        let img = self.wf_image(&tex);
        let xb = wf1 / 1e3 * 0.85;
        Plot::new("marked").x_axis_label("offset from centre (kHz)").y_axis_label("time from start of recording (s)")
            .legend(Legend::default()).show(ui, |p| {
                p.add(img);
                p.add(Line::new("straight-pass fit", curve).color(Color32::from_rgba_unmultiplied(255, 90, 90, 204)).width(1.0));
                for (f, l) in [(m.f_plus_hz, "f₊"), (m.f_minus_hz, "f₋"), (f0, "f₀")] {
                    if wf0 < f - centre && f - centre < wf1 {
                        p.add(VLine::new("", x(f)).color(Color32::WHITE).style(LineStyle::dashed_dense()));
                        label(p, x(f) + 0.15, wt0 + 0.02 * (wt1 - wt0), l, Color32::WHITE, Align2::LEFT_BOTTOM);
                    }
                }
                for i in 0..2 {
                    let (fx, ty) = (x(m.half_plateau_hz[i]), m.half_plateau_t_s[i]);
                    p.add(Points::new("", vec![[fx, ty]]).shape(MarkerShape::Circle).filled(false).radius(10.0).color(YELLOW));
                    label(p, fx + if i == 1 { 0.6 } else { -0.6 }, ty, format!("half-plateau {}", i + 1), YELLOW,
                        if i == 1 { Align2::LEFT_CENTER } else { Align2::RIGHT_CENTER });
                    seg(p, [fx, ty], [xb, ty], YELLOW, 1.0, true);
                }
                let th = m.half_plateau_t_s;
                span_arrow(p, [xb, th[0]], [xb, th[1]], YELLOW);
                label(p, xb - 0.3, (th[0] + th[1]) / 2.0, format!("Δt = {:.2} s", m.dt_s), YELLOW, Align2::RIGHT_CENTER);
            });
    }

    fn laser_tab(&mut self, ui: &mut egui::Ui) {
        let Some((run, _)) = &self.laser else { ui.label("Run a laser-ranging file (.np2 / .npt / .crd or an EDC page URL) to see this."); return };
        let which = self.which.trim().parse::<usize>().unwrap_or(0);
        let Some((p, f)) = run.passes.get(which) else {
            ui.label(format!("Pass {which} does not exist: the file has {} passes (0 to {}).", run.passes.len(), run.passes.len().saturating_sub(1)));
            return;
        };
        let (d, v) = (f.d_km, f.v_km_s);
        let when: String = slr::closest_utc(p, f).replace('T', " ").chars().take(21).collect();
        ui.label(RichText::new(format!("pass {which}: {} past {}: v {v:.4} km/s, d {d:.1} km, closest {when} UTC, {} shots, fit miss {:.0} m rms",
            p.sat, p.station, f.n, f.rms_m)).strong());
        let t: Vec<f64> = p.t.iter().map(|x| x - f.t0).collect();
        let (tmin, tmax) = t.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x), a.1.max(x)));
        let tt = linspace(tmin, tmax, 600);
        let (tk, rk, along) = crate::triangle(p, f);
        let h = ui.available_height() - 20.0;
        let link = Id::unique("laser_x");
        Plot::new("range").height(h * 0.6).link_axis(link, [true, false]).legend(Legend::default())
            .y_axis_label("distance to the satellite (km)").show(ui, |pl| {
                pl.add(Points::new("measured: round-trip time × c/2", t.iter().zip(&p.r).map(|(t, r)| [*t, *r]).collect::<Vec<_>>()).color(BLUE).radius(2.5));
                pl.add(Line::new("straight path: r² = d² + (v·t)²", tt.iter().map(|&x| [x, d.hypot(v * x)]).collect::<Vec<_>>()).color(RED).width(1.5));
                pl.add(HLine::new("", d).color(GREY).style(LineStyle::dashed_dense()));
                label(pl, tmin, d, format!(" d = {d:.1} km (closest)"), Color32::LIGHT_GRAY, Align2::LEFT_BOTTOM);
                seg(pl, [0.0, d], [tk, d], GOLD, 1.0, true);
                seg(pl, [tk, d], [tk, rk], GOLD, 2.0, false);
                seg(pl, [0.0, d], [tk, rk], GOLD, 1.0, true);
                label(pl, tk, (d + rk) / 2.0, format!("shot at t = {tk:+.1} s\nr = {rk:.1} km\n√(r² − d²) = {along:.1} km\nv = {along:.1} / {:.1} = {:.4} km/s ",
                    tk.abs(), along / tk.abs()), GOLD, if tk > 0.0 { Align2::RIGHT_CENTER } else { Align2::LEFT_CENTER });
            });
        let sp: Vec<[f64; 2]> = slr::speeds(p, f).into_iter().map(|(t, s)| [t, s]).collect();
        Plot::new("speeds").height(ui.available_height()).link_axis(link, [true, false]).legend(Legend::default())
            .x_axis_label("time from closest approach (s)").y_axis_label("speed (km/s)").show(ui, |pl| {
                pl.add(Points::new("v = √(r² − d²) / |t|, each shot", sp).color(BLUE).radius(2.5));
                pl.add(HLine::new(format!("fit {v:.4} km/s"), v).color(RED).width(1.5));
            });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        egui::Panel::left("controls").resizable(true).default_size(470.0).show(ui, |ui| {
            ui.add_space(6.0);
            self.controls(ui);
        });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (t, l) in [(Tab::Results, "Results"), (Tab::Waterfall, "Waterfall"), (Tab::SCurve, "S-curve"), (Tab::Marked, "Marked waterfall"), (Tab::Laser, "Laser")] {
                    ui.selectable_value(&mut self.tab, t, l);
                }
            });
            ui.separator();
            match self.tab {
                Tab::Results => {
                    egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                        ui.add(egui::TextEdit::multiline(&mut self.results.as_str()).font(egui::FontId::monospace(14.0)).desired_width(f32::INFINITY));
                    });
                }
                Tab::Waterfall => self.waterfall_tab(ui),
                Tab::SCurve => self.scurve_tab(ui),
                Tab::Marked => self.marked_tab(ui),
                Tab::Laser => self.laser_tab(ui),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_centre_and_waterfall_image() {
        assert_eq!(centre_from_name("x_12Hz_437800000Hz_IQ").as_deref(), Some("437800000"));
        assert_eq!(centre_from_name("rec_1000Hz"), None);
        let w = trace::Waterfall { t: vec![0.0, 0.5, 1.0], f: (0..5).map(|j| j as f64).collect(), db: (0..15).map(|v| v as f32).collect() };
        let im = waterfall_image(&w, 2);
        assert_eq!(im.size, [2, 2]);
        assert_eq!(im.pixels[1], viridis(14.0 / 15.0));
        assert_eq!(im.pixels[2], viridis(7.0 / 15.0));
    }
}
