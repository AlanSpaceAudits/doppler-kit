//! The Doppler Kit window (port of doppler_gui.py).

use doppler_kit::{fit, input::RawFormat, slr, trace};
use eframe::egui::{self, Align2, Color32, ColorImage, Id, RichText, TextStyle, TextureHandle, TextureOptions};
use egui_plot::{uniform_grid_spacer, Arrows, Corner, GridInput, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, PlotPoint, PlotTransform, PlotUi, Points, Text, VLine};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

const BLUE: Color32 = Color32::from_rgb(0x63, 0xb3, 0xed);
const RED: Color32 = Color32::from_rgb(0xfc, 0x81, 0x81);
const GOLD: Color32 = Color32::from_rgb(0xf6, 0xc3, 0x4a);
const YELLOW: Color32 = Color32::from_rgb(0xff, 0xd2, 0x3f);
const GREY: Color32 = Color32::from_gray(150);
const GREEN: Color32 = Color32::from_rgb(0x68, 0xd3, 0x91);
const PURPLE: Color32 = Color32::from_rgb(0xc0, 0x9a, 0xff);
/// Waterfall brightness: dB above the row median shown from black (first) to full (second).
const WF_DB: [f32; 2] = [5.0, 20.0];
/// Largest waterfall image side, about one screen pixel per image pixel.
const WF_PX: usize = 1024;

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
enum Tab { Results, Waterfall, SCurve, Marked, Laser, Math }

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
    rate: String, fmt: Option<RawFormat>, swap: bool, band_lo: String, band_hi: String, burst: bool, fm: bool, which: String, out: String,
    log: String, results: String, tab: Tab, rx: Option<Receiver<Msg>>,
    trace: Option<trace::TraceResult>, fit: Option<(fit::FitResult, PathBuf)>, laser: Option<(slr::SlrRun, String)>,
    wf_tex: Option<TextureHandle>,
    /// Counts runs; part of every plot id so a new run starts from its default view.
    run: u64,
    show_trace: bool,
    picker: Option<Receiver<(bool, PathBuf)>>,
}

impl Default for App {
    fn default() -> Self {
        App {
            file: String::new(), centre: String::new(), start: String::new(), wave: Wave::Radio, c_other: String::new(),
            f0: String::new(), span: String::new(), rate: String::new(), fmt: None, swap: false, band_lo: String::new(), band_hi: String::new(), burst: false, fm: false, which: "0".into(),
            out: String::new(), log: String::new(), results: String::new(), tab: Tab::Results, rx: None,
            trace: None, fit: None, laser: None, wf_tex: None, run: 0, show_trace: true, picker: None,
        }
    }
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

/// Inferno colour map (perceptual, black to pale yellow), x in 0..1.
fn inferno(x: f32) -> Color32 {
    const S: [[u8; 3]; 9] = [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
        [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]];
    let x = if x.is_finite() { x.clamp(0.0, 1.0) * 8.0 } else { 0.0 };
    let (i, u) = ((x as usize).min(7), x - (x as usize).min(7) as f32);
    let m = |k: usize| (S[i][k] as f32 + (S[i + 1][k] as f32 - S[i][k] as f32) * u) as u8;
    Color32::from_rgb(m(0), m(1), m(2))
}

/// Waterfall as an image, time upwards. Each row is taken relative to its own median and shown on inferno over WF_DB.
/// Blocks are pooled down to WF_PX: power averaged over time (less speckle), then the strongest bin (thin lines survive).
fn waterfall_image(w: &trace::Waterfall, max_side: usize) -> ColorImage {
    let (nt, nf) = (w.t.len(), w.f.len());
    let med: Vec<f32> = (0..nt).map(|i| {
        let mut r = w.db[i * nf..(i + 1) * nf].to_vec();
        r.sort_by(f32::total_cmp);
        r[nf / 2]
    }).collect();
    let side = max_side.min(WF_PX);
    let (by, bx) = (nt.div_ceil(side).max(1), nf.div_ceil(side).max(1));
    let (h, wd) = (nt.div_ceil(by), nf.div_ceil(bx));
    let mut px = Vec::with_capacity(h * wd);
    for r in (0..h).rev() {
        for c in 0..wd {
            let rows = r * by..((r + 1) * by).min(nt);
            let m = (c * bx..((c + 1) * bx).min(nf)).map(|j| {
                rows.clone().map(|i| 10f32.powf((w.db[i * nf + j] - med[i]) / 10.0)).sum::<f32>() / rows.len() as f32
            }).fold(0.0f32, f32::max);
            px.push(inferno((10.0 * m.log10() - WF_DB[0]) / (WF_DB[1] - WF_DB[0])));
        }
    }
    ColorImage::new([wd, h], px)
}

/// Default waterfall view: the measured trace's frequency range plus a margin (kHz) and the whole recording (s).
fn signal_view(t: &trace::TraceResult) -> ([f64; 2], [f64; 2]) {
    let w = &t.waterfall;
    let (f0, f1) = (w.f[0] / 1e3, w.f[w.f.len() - 1] / 1e3);
    let (lo, hi) = t.points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.f_offset / 1e3), a.1.max(p.f_offset / 1e3)));
    let m = (0.15 * (hi - lo)).max(1.0);
    let x = [(lo - m).max(f0), (hi + m).min(f1)];
    (if x[0] < x[1] { x } else { [f0, f1] }, [w.t[0], w.t[w.t.len() - 1]])
}

/// Range of finite values, or None.
fn extent<'a>(v: impl IntoIterator<Item = &'a f64>) -> Option<[f64; 2]> {
    let r = v.into_iter().filter(|x| x.is_finite()).fold([f64::INFINITY, f64::NEG_INFINITY], |a, &x| [a[0].min(x), a[1].max(x)]);
    (r[0] < r[1]).then_some(r)
}

/// Strip under the waterfall title: what brightness means.
fn colour_bar(ui: &mut egui::Ui) {
    ui.label(format!("brightness = power above the row median:  {:.0} dB (noise)", WF_DB[0]));
    let (rect, _) = ui.allocate_exact_size(egui::vec2(160.0, 14.0), egui::Sense::hover());
    const N: usize = 64;
    for i in 0..N {
        let x0 = rect.left() + rect.width() * i as f32 / N as f32;
        let r = egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x0 + rect.width() / N as f32 + 0.5, rect.bottom()));
        ui.painter().rect_filled(r, 0.0, inferno(i as f32 / (N - 1) as f32));
    }
    ui.label(format!("{:.0} dB+", WF_DB[1]));
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

fn field(ui: &mut egui::Ui, l: &str, s: &mut String) {
    field_auto(ui, l, s, None);
}

/// A text field; when it is empty and the last run worked the value out, the value shows as the hint text.
fn field_auto(ui: &mut egui::Ui, l: &str, s: &mut String, auto: Option<&str>) {
    ui.label(l);
    let mut te = egui::TextEdit::singleline(s).desired_width(250.0);
    if let Some(a) = auto {
        te = te.hint_text(format!("auto: {a}"));
    }
    let r = ui.add(te);
    if let Some(a) = auto {
        r.on_hover_text(format!("auto: {a}"));
    }
    ui.end_row();
}

/// The half-plateau triangle: path, observer, closest distance d, both half-plateau points with their lines of sight,
/// the 60° and 30° angles, the v/2 range rate and v · Δt = 2d/√3, with the numbers of the current run when there is one.
fn half_plateau_diagram(ui: &mut egui::Ui, fit: Option<&fit::FitResult>) {
    use egui::{pos2, vec2, FontId, Pos2, Shape, Stroke};
    const LOS: Color32 = Color32::from_rgb(0xf6, 0xc3, 0x4a);
    const DBLUE: Color32 = Color32::from_rgb(0x7f, 0xb2, 0xf0);
    const ORANGE: Color32 = Color32::from_rgb(0xe9, 0x7b, 0x5a);
    const VGREEN: Color32 = Color32::from_rgb(0x8f, 0xd8, 0x84);
    let txt = Color32::from_gray(225);
    let km = fit.is_none_or(|r| r.c > 1e5);
    let len = |x: f64| if km { format!("{} km", fit::fcomma(x / 1e3, 1)) } else { format!("{} m", fit::fcomma(x, 1)) };
    let spd = |x: f64| if km { format!("{} km/s", fit::fx(x / 1e3, 4)) } else { format!("{} m/s", fit::fx(x, 3)) };
    let num = |sym: &str, v: Option<f64>| v.map_or(sym.to_string(), |x| format!("{sym} = {}", len(x)));
    let dd = fit.map(|r| r.marks.d_width_m);
    let w = ui.available_width().min(980.0);
    let dp = (0.35 * w).min(420.0);
    let hx = dp / 3f32.sqrt();
    let (rect, _) = ui.allocate_exact_size(vec2(w, dp + 160.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    let cx = rect.left() + 0.42 * w;
    let py = rect.top() + 110.0;
    let o = pos2(cx, py + dp);
    let (p1, p2, c) = (pos2(cx - hx, py), pos2(cx + hx, py), pos2(cx, py));
    let font = FontId::proportional(14.0);
    let t = |at: Pos2, a: Align2, s: &str, col: Color32| { p.text(at, a, s, font.clone(), col); };
    // arrow with a fixed head (egui's head grows with the length)
    let arrow = |a: Pos2, v: egui::Vec2, st: Stroke| {
        let b = a + v;
        p.line_segment([a, b], st);
        let u = v.normalized();
        for s in [-1.0f32, 1.0] {
            let (sn, cs) = (s * 25f32.to_radians()).sin_cos();
            let back = -vec2(u.x * cs - u.y * sn, u.x * sn + u.y * cs) * 10.0;
            p.line_segment([b, b + back], st);
        }
    };
    let arc = |ctr: Pos2, r: f32, a0: f32, a1: f32| {
        let pts: Vec<Pos2> = (0..=24).map(|i| { let a = (a0 + (a1 - a0) * i as f32 / 24.0).to_radians(); ctr + vec2(a.cos(), a.sin()) * r }).collect();
        p.add(Shape::line(pts, Stroke::new(1.2, txt)));
    };
    // the path, with its direction
    let x0 = rect.left() + 10.0;
    arrow(pos2(x0, py), vec2(cx + 2.1 * hx - x0, 0.0), Stroke::new(1.5, GREY));
    t(pos2(x0, py - 10.0), Align2::LEFT_BOTTOM, "straight path of the source, speed v", GREY);
    // d and the right angle
    p.line_segment([c, o], Stroke::new(2.0, DBLUE));
    p.add(Shape::line(vec![pos2(cx, py + 12.0), pos2(cx + 12.0, py + 12.0), pos2(cx + 12.0, py)], Stroke::new(1.2, DBLUE)));
    t(pos2(cx - 8.0, py + 0.12 * dp), Align2::RIGHT_TOP, &format!("d (closest distance)\n{}", dd.map_or(String::new(), |x| format!("= {}", len(x)))), DBLUE);
    // lines of sight
    for q in [p1, p2] {
        p.line_segment([q, o], Stroke::new(2.5, LOS));
    }
    t(pos2(cx + 0.4 * hx + 14.0, py + 0.6 * dp), Align2::LEFT_TOP, &format!("line of sight\n{}", num("r = 2d/√3", dd.map(|x| 2.0 * x / 3f64.sqrt()))), LOS);
    // d/√3 along the path
    p.line_segment([c, p2], Stroke::new(4.0, ORANGE));
    t(pos2(cx + 0.5 * hx, py - 10.0), Align2::CENTER_BOTTOM, &num("d/√3", dd.map(|x| x / 3f64.sqrt())), ORANGE);
    // v at half-plateau 2 and its v/2 part along the line of sight
    let vl = 0.6 * hx;
    arrow(p2, vec2(vl, 0.0), Stroke::new(3.0, VGREEN));
    t(p2 + vec2(vl + 8.0, 0.0), Align2::LEFT_CENTER, "v", VGREEN);
    let u = (p2 - o).normalized();
    let tip = p2 + u * (vl / 2.0);
    p.extend(Shape::dashed_line(&[p2, tip], Stroke::new(2.5, VGREEN), 6.0, 4.0));
    arrow(tip - u * 6.0, u * 6.0, Stroke::new(2.5, VGREEN));
    t(tip + vec2(8.0, -4.0), Align2::LEFT_BOTTOM, "v/2 along the line of sight\n(range rate +v/2)", VGREEN);
    // angles: 60° at half-plateau 2 between the path and the line of sight, 30° at the observer
    arc(p2, 0.18 * hx + 18.0, 120.0, 180.0);
    t(p2 + vec2(-(0.18 * hx + 30.0) * 0.866, (0.18 * hx + 30.0) * 0.5), Align2::RIGHT_TOP, "60°", txt);
    arc(o, 0.3 * dp, -90.0, -60.0);
    t(o + vec2(0.3 * dp * 0.26 + 6.0, -0.3 * dp - 6.0), Align2::LEFT_BOTTOM, "30°", txt);
    // the points and the observer
    for (q, l, off, a) in [(p1, "half-plateau 1", vec2(0.0, -30.0), Align2::CENTER_BOTTOM), (p2, "half-plateau 2", vec2(-8.0, -30.0), Align2::RIGHT_BOTTOM)] {
        p.circle_filled(q, 6.5, LOS);
        t(q + off, a, l, LOS);
    }
    p.circle_filled(o, 5.0, Color32::from_gray(235));
    t(o + vec2(0.0, 10.0), Align2::CENTER_TOP, "observer", txt);
    // v · Δt between the points
    let yb = py - 78.0;
    arrow(pos2(cx, yb), vec2(hx, 0.0), Stroke::new(1.3, txt));
    arrow(pos2(cx, yb), vec2(-hx, 0.0), Stroke::new(1.3, txt));
    t(pos2(cx, yb - 4.0), Align2::CENTER_BOTTOM, &num("v · Δt = 2d/√3", dd.map(|x| 2.0 * x / 3f64.sqrt())), txt);
    // the reasoning, below
    let mut lines = vec![
        "At a half-plateau point the line of sight meets the path at 60°, so the speed along it is v·cos 60° = v/2.".to_string(),
        "tan 60° = d / (distance along the path), so distance along the path = d/√3.".into(),
        "The two points are 2d/√3 apart and the source covers that in Δt: v·Δt = 2d/√3, so d = Δt · v · √3/2.".into(),
        "Half-plateau 1 is the mirror image before the pass (range rate −v/2).".into(),
    ];
    if let Some(r) = fit {
        lines.push(format!("This run: Δt = {:.2} s, v = {}, so d = Δt × v × 0.8660 = {:.2} × {} × 0.8660 = {}",
            r.marks.dt_s, spd(r.p[1]), r.marks.dt_s, spd(r.p[1]), len(r.marks.d_width_m)));
    }
    for l in lines {
        ui.label(RichText::new(l).monospace().size(13.5).color(Color32::from_gray(215)));
    }
}

fn label(p: &mut PlotUi, x: f64, y: f64, s: impl Into<String>, c: Color32, a: Align2) {
    p.add(Text::new("", PlotPoint::new(x, y), RichText::new(s.into()).color(c)).anchor(a));
}

/// Plot label painted after the plot on a dark box, so it stays readable over the waterfall and the lines.
struct Tag { at: PlotPoint, text: String, color: Color32, anchor: Align2 }

fn tag(v: &mut Vec<Tag>, x: f64, y: f64, s: impl Into<String>, c: Color32, a: Align2) {
    v.push(Tag { at: PlotPoint::new(x, y), text: s.into(), color: c, anchor: a });
}

fn paint_tags(ui: &egui::Ui, tr: &PlotTransform, tags: &[Tag]) {
    let painter = ui.painter_at(*tr.frame());
    for t in tags {
        let g = painter.layout_no_wrap(t.text.clone(), egui::FontId::monospace(14.0), t.color);
        let r = t.anchor.anchor_size(tr.position_from_point(&t.at), g.size());
        painter.rect_filled(r.expand(3.0), 3.0, Color32::from_black_alpha(210));
        painter.galley(r.min, g, t.color);
    }
}

/// Grid steps 1, 5 and 10 times a power of ten, so a kHz axis gets a label every few hundred pixels at most.
fn steps_1_5_10(g: GridInput) -> [f64; 3] {
    let a = 10f64.powf(g.base_step_size.abs().max(1e-12).log10().ceil());
    [a, 5.0 * a, 10.0 * a]
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
        let (los, his) = (self.band_lo.clone(), self.band_hi.clone());
        let Some(lo) = self.num(&los, "tone band low") else { return };
        let Some(hi) = self.num(&his, "tone band high") else { return };
        let band = match (lo, hi) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => { self.log += "Stopped: give both tone band limits, or neither\n"; return; }
        };
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
        let opts = trace::TraceOpts { center, start, span, rate, format: self.fmt, swap_iq: self.swap, out, band, burst: self.burst, fm: self.fm };
        self.trace = None;
        self.fit = None;
        self.laser = None;
        self.wf_tex = None;
        self.run += 1;
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
                    // ponytail: a failed fit is dropped so no plot or Math line shows its numbers
                    self.fit = if f.no_pass().is_empty() { Some((*f, p)) } else { None };
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

    /// File dialogs run on their own thread so the window keeps drawing while one is open.
    fn pick(&mut self, folder: bool) {
        if self.picker.is_some() { return; }
        let (tx, rx) = channel();
        self.picker = Some(rx);
        std::thread::spawn(move || {
            let d = rfd::FileDialog::new();
            let p = if folder { d.pick_folder() } else {
                d.add_filter("Recordings", &["wav", "raw", "cs16", "cf32", "cu8", "sigmf-data", "ogg", "oga", "mp3", "flac", "m4a", "opus", "aac"])
                    .add_filter("Laser ranging", &["np2", "npt", "crd", "html", "htm"])
                    .add_filter("All files", &["*"])
                    .pick_file()
            };
            if let Some(p) = p { let _ = tx.send((folder, p)); }
        });
    }

    /// Auto value of the last run for a field key (centre, start, rate, format, span, band, burst, fm).
    fn auto_of(&self, k: &str) -> Option<String> {
        self.trace.as_ref()?.auto.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone())
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let au: Vec<Option<String>> = ["centre", "start", "span", "rate", "format", "band", "burst", "fm"].iter().map(|k| self.auto_of(k)).collect();
        let small = |ui: &mut egui::Ui, a: &Option<String>| if let Some(a) = a { ui.label(RichText::new(format!("auto: {a}")).small().color(GREY)); };
        egui::Grid::new("controls").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Recording");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.file).desired_width(250.0));
                if ui.button("Browse…").clicked() {
                    self.pick(false);
                }
            });
            ui.end_row();
            field_auto(ui, "Centre / dial frequency (Hz)", &mut self.centre, au[0].as_deref());
            field_auto(ui, "Start time UTC (optional)", &mut self.start, au[1].as_deref());
            ui.label("Wave");
            ui.vertical(|ui| {
                ui.radio_value(&mut self.wave, Wave::Radio, "radio / light (299,792,458 m/s)");
                ui.radio_value(&mut self.wave, Wave::Sound, "sound in air (343 m/s)");
                ui.radio_value(&mut self.wave, Wave::Other, "other (m/s below)");
            });
            ui.end_row();
            field(ui, "Wave speed m/s (other)", &mut self.c_other);
            field(ui, "Fix transmitted f₀ (Hz, optional)", &mut self.f0);
            field_auto(ui, "Search ± Hz (blank: scan)", &mut self.span, au[2].as_deref());
            field_auto(ui, "Raw IQ sample rate (S/s)", &mut self.rate, au[3].as_deref());
            ui.label("Raw IQ format");
            ui.horizontal(|ui| {
                let name = |f: Option<RawFormat>| f.map_or("auto".to_string(), |f| format!("{f:?}").to_lowercase());
                egui::ComboBox::from_id_salt("fmt").selected_text(name(self.fmt)).show_ui(ui, |ui| {
                    for f in [None, Some(RawFormat::Cs16), Some(RawFormat::Cf32), Some(RawFormat::Cu8)] {
                        ui.selectable_value(&mut self.fmt, f, name(f));
                    }
                });
                if self.fmt.is_none() { small(ui, &au[4]); }
            });
            ui.end_row();
            ui.label("Q stored before I");
            ui.checkbox(&mut self.swap, "swap IQ");
            ui.end_row();
            let band_auto = if self.band_lo.trim().is_empty() && self.band_hi.trim().is_empty() { au[5].as_deref() } else { None };
            field_auto(ui, "Tone band low (Hz, optional)", &mut self.band_lo, band_auto);
            field_auto(ui, "Tone band high (Hz, optional)", &mut self.band_hi, band_auto);
            ui.label("Burst signal");
            ui.vertical(|ui| {
                ui.checkbox(&mut self.burst, "short FSK bursts (satellites)");
                if !self.burst { small(ui, &au[6]); }
            });
            ui.end_row();
            ui.label("FM carrier");
            ui.vertical(|ui| {
                ui.checkbox(&mut self.fm, "ISS SSTV, voice (FM)");
                if !self.fm { small(ui, &au[7]); }
            });
            ui.end_row();
            field(ui, "Laser file: pass to plot", &mut self.which);
            ui.label("Output folder (blank: next to file)");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.out).desired_width(250.0));
                if ui.button("…").clicked() {
                    self.pick(true);
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
        PlotImage::new("", tex.id(), PlotPoint::new((f0 + f1) / 2.0, (t0 + t1) / 2.0), [(f1 - f0) as f32, (t1 - t0) as f32])
    }

    fn waterfall_tab(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = self.wf_texture(ui.ctx()) else { ui.label("Run a recording to see its waterfall."); return };
        let img = self.wf_image(&tex);
        let t = self.trace.as_ref().unwrap();
        ui.label(RichText::new("Waterfall: each row is one spectrum, time runs up; the bright S-shaped line is the satellite's signal").strong());
        ui.label(format!("{}   centre {:.6} MHz   (scroll to zoom, drag to pan, double-click to reset)", t.name, t.centre_hz / 1e6));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_trace, "show the measured trace (red)");
            ui.add_space(20.0);
            colour_bar(ui);
        });
        let ([x0, x1], [y0, y1]) = signal_view(t);
        let tr: Vec<[f64; 2]> = t.points.iter().map(|p| [p.f_offset / 1e3, p.t_s]).collect();
        let show = self.show_trace;
        Plot::new(("wf", self.run)).x_axis_label("frequency offset from centre (kHz)").y_axis_label("time from start of recording (s)")
            .default_x_bounds(x0, x1).default_y_bounds(y0, y1).x_grid_spacer(uniform_grid_spacer(steps_1_5_10))
            .legend(Legend::default().position(Corner::LeftTop)).show(ui, |p| {
                p.add(img);
                if show { p.add(Points::new("measured trace", tr).color(RED).radius(1.2)); }
            });
    }

    fn scurve_tab(&mut self, ui: &mut egui::Ui) {
        let Some((r, _)) = &self.fit else { ui.label("Run a recording to see the S-curve."); return };
        let (m, [t0, v, d, f0]) = (&r.marks, r.p);
        let (th, fh) = (m.half_plateau_t_s, m.half_plateau_hz);
        let k = |x: f64| (x - f0) / 1e3;
        let (ta, tb) = (r.t[0], r.t[r.t.len() - 1]);
        let xpad = 0.03 * (tb - ta);
        let (xlo, xhi) = (ta - t0 - xpad, tb - t0 + xpad);
        let [fa, fb] = extent(r.f.iter().chain([&m.f_plus_hz, &m.f_minus_hz])).unwrap_or([f0 - 1.0, f0 + 1.0]);
        let (ya, yb) = (k(fa), k(fb));
        let yr = yb - ya;
        let ybr = ya - 0.07 * yr;
        let (ylo, yhi) = (ya - 0.17 * yr, yb + 0.06 * yr);
        ui.label(RichText::new("Doppler S-curve: plateaus, middle, half-plateau points and width").strong());
        ui.label("Half-plateau points: where the curve is halfway between the middle (f₀) and a plateau. \
            There the distance changes at half the speed. Δt between them gives d = Δt · v · √3/2.");
        ui.label(RichText::new(format!(
            "v = c (f₊ − f₋)/(f₊ + f₋) = {:.3} km/s\nd = Δt · v · √3/2 = {:.1} km\nd = f₀ v² / (c · |slope at middle|) = {:.1} km   (slope {:.1} Hz/s)\nfull fit: v {:.3} km/s, d {:.1} km, f₀ {} Hz",
            m.v_plateaus_m_s / 1e3, m.d_width_m / 1e3, m.d_slope_m / 1e3, m.slope_hz_s, v / 1e3, d / 1e3, fit::fcomma(f0, 1))).monospace());
        let h = ui.available_height() - 40.0;
        let tt = linspace(ta, tb, 1500);
        let curve: Vec<[f64; 2]> = tt.iter().zip(r.curve(&tt)).map(|(t, f)| [t - t0, k(f)]).collect();
        let meas: Vec<[f64; 2]> = r.t.iter().zip(&r.f).map(|(t, f)| [t - t0, k(*f)]).collect();
        let link = Id::unique(("scurve_x", self.run));
        let lx = xhi - 0.01 * (xhi - xlo);
        let resp = Plot::new(("scurve", self.run)).height(h * 0.72).link_axis(link, [true, false])
            .legend(Legend::default().position(Corner::LeftBottom).text_style(TextStyle::Monospace))
            .y_axis_label("received − f₀ (kHz)")
            .default_x_bounds(xlo, xhi).default_y_bounds(ylo, yhi)
            .show(ui, |p| {
                let mut tags = vec![];
                p.add(Points::new("measured", meas).color(BLUE).radius(1.5));
                p.add(Line::new("full fit (straight pass)", curve).color(RED).width(1.5));
                // All line labels sit at the right edge: there the curve is near f₋, below them or just above f₋'s label.
                // The lines carry their own labels, so the legend stays short.
                for (f, l, c, a) in [
                    (m.f_plus_hz, "plateau f₊ = f₀/(1 − v/c)  (coming)", GREEN, Align2::RIGHT_TOP),
                    (f0, "middle f₀", Color32::LIGHT_GRAY, Align2::RIGHT_BOTTOM),
                    (m.f_minus_hz, "plateau f₋ = f₀/(1 + v/c)  (going)", PURPLE, Align2::RIGHT_TOP),
                ] {
                    p.add(HLine::new("", k(f)).color(c).style(LineStyle::dashed_dense()));
                    tag(&mut tags, lx, k(f), l, c, a);
                }
                for (f, sign) in fh.into_iter().zip(['−', '+']) {
                    p.add(HLine::new("", k(f)).color(GOLD).style(LineStyle::dotted_dense()));
                    tag(&mut tags, lx, k(f), format!("halfway level f₀/(1 {sign} v/2c)"), GOLD, Align2::RIGHT_BOTTOM);
                }
                let pts: Vec<[f64; 2]> = (0..2).map(|i| [th[i] - t0, k(fh[i])]).collect();
                p.add(Points::new("half-plateau points", pts.clone()).shape(MarkerShape::Circle).filled(false).radius(8.0).color(YELLOW));
                let dx = 0.01 * (xhi - xlo);
                for (i, &[x, y]) in pts.iter().enumerate() {
                    seg(p, [x, y], [x, ybr], YELLOW, 1.2, true);
                    let txt = format!("half-plateau {}\nrange rate {}v/2\nt = {:.1} s", i + 1, if i == 0 { '−' } else { '+' }, th[i]);
                    // Below-left of each point is clear: the curve runs from upper left to lower right.
                    tag(&mut tags, x - dx, y, txt, YELLOW, Align2::RIGHT_TOP);
                }
                span_arrow(p, [pts[0][0], ybr], [pts[1][0], ybr], YELLOW);
                tag(&mut tags, (pts[0][0] + pts[1][0]) / 2.0, ybr, format!("Δt = {:.2} s", m.dt_s), YELLOW, Align2::CENTER_TOP);
                tags
            });
        paint_tags(ui, &resp.transform, &resp.inner);
        ui.label(format!("left over: {} (smooth pattern {:.1} Hz rms, scatter {:.1} Hz rms, p = {}, detectable from {:.1} Hz rms)",
            r.path_word(), r.path.pattern_rms_hz, r.path.scatter_rms_hz, fit::fg(r.path.p_value, 2), r.path.detectable_rms_hz));
        let left: Vec<[f64; 2]> = r.t.iter().zip(&r.left).map(|(t, l)| [t - t0, *l]).collect();
        let lm = extent(r.left.iter()).map_or(1.0, |[a, b]| a.abs().max(b.abs()) * 1.1);
        Plot::new(("left", self.run)).height(ui.available_height()).link_axis(link, [true, false])
            .x_axis_label("time from closest approach (s)").y_axis_label("left over (Hz)")
            .default_x_bounds(xlo, xhi).default_y_bounds(-lm, lm)
            .show(ui, |p| {
                p.add(Points::new("left over", left).color(BLUE).radius(1.5));
                p.add(HLine::new("", 0.0).color(Color32::LIGHT_GRAY));
            });
    }

    fn marked_tab(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = self.wf_texture(ui.ctx()) else { ui.label("Run a recording to see the marked waterfall."); return };
        let Some((r, _)) = &self.fit else { ui.label("The fit did not finish."); return };
        let t = self.trace.as_ref().unwrap();
        let (m, [_, v, d, f0], centre) = (&r.marks, r.p, t.centre_hz);
        let x = |f: f64| (f - centre) / 1e3;
        let ([vx0, vx1], [wt0, wt1]) = signal_view(t);
        ui.label(RichText::new(format!("Waterfall with the half-plateau points: v {:.3} km/s, d {:.1} km", v / 1e3, d / 1e3)).strong());
        ui.label("The fitted curve over the waterfall. The circles are the half-plateau points; Δt between them gives d = Δt · v · √3/2.");
        let ty = linspace(wt0, wt1, 1500);
        let curve: Vec<[f64; 2]> = ty.iter().zip(r.curve(&ty)).map(|(t, f)| [x(f), *t]).collect();
        let img = self.wf_image(&tex);
        let w = vx1 - vx0;
        let xb = vx1 - 0.06 * w;
        let resp = Plot::new(("marked", self.run)).x_axis_label("frequency offset from centre (kHz)").y_axis_label("time from start of recording (s)")
            .default_x_bounds(vx0, vx1).default_y_bounds(wt0, wt1).x_grid_spacer(uniform_grid_spacer(steps_1_5_10))
            .legend(Legend::default().position(Corner::LeftTop)).show(ui, |p| {
                let mut tags = vec![];
                p.add(img);
                p.add(Line::new("full fit (straight pass)", curve).color(Color32::from_rgba_unmultiplied(120, 200, 255, 220)).width(1.5));
                for (f, l) in [(m.f_plus_hz, "f₊"), (m.f_minus_hz, "f₋"), (f0, "f₀")] {
                    if vx0 < x(f) && x(f) < vx1 {
                        p.add(VLine::new("", x(f)).color(Color32::WHITE).style(LineStyle::dashed_dense()));
                        tag(&mut tags, x(f), wt0 + 0.01 * (wt1 - wt0), l, Color32::WHITE, Align2::CENTER_BOTTOM);
                    }
                }
                for i in 0..2 {
                    let (fx, ty) = (x(m.half_plateau_hz[i]), m.half_plateau_t_s[i]);
                    p.add(Points::new("half-plateau points", vec![[fx, ty]]).shape(MarkerShape::Circle).filled(false).radius(10.0).color(YELLOW));
                    // Time runs up and frequency falls, so the curve never passes below-left or above-right of a point.
                    let txt = format!("half-plateau {}  ({:.1} s)", i + 1, ty);
                    if i == 0 { tag(&mut tags, fx - 0.02 * w, ty, txt, YELLOW, Align2::RIGHT_TOP) } else { tag(&mut tags, fx + 0.02 * w, ty, txt, YELLOW, Align2::LEFT_BOTTOM) }
                    seg(p, [fx, ty], [xb, ty], YELLOW, 1.0, true);
                }
                let th = m.half_plateau_t_s;
                span_arrow(p, [xb, th[0]], [xb, th[1]], YELLOW);
                tag(&mut tags, xb - 0.01 * w, (th[0] + th[1]) / 2.0, format!("Δt = {:.2} s", m.dt_s), YELLOW, Align2::RIGHT_CENTER);
                tags
            });
        paint_tags(ui, &resp.transform, &resp.inner);
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
        Plot::new(("range", self.run)).height(h * 0.6).link_axis(link, [true, false]).legend(Legend::default())
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
        Plot::new(("speeds", self.run)).height(ui.available_height()).link_axis(link, [true, false]).legend(Legend::default())
            .x_axis_label("time from closest approach (s)").y_axis_label("speed (km/s)").show(ui, |pl| {
                pl.add(Points::new("v = √(r² − d²) / |t|, each shot", sp).color(BLUE).radius(2.5));
                pl.add(HLine::new(format!("fit {v:.4} km/s"), v).color(RED).width(1.5));
            });
    }

    fn math_tab(&self, ui: &mut egui::Ui) {
        let h = |ui: &mut egui::Ui, s: &str| { ui.add_space(14.0); ui.label(RichText::new(s).size(19.0).strong()); ui.add_space(2.0); };
        let p = |ui: &mut egui::Ui, s: &str| { ui.label(RichText::new(s).size(15.0).color(Color32::from_gray(215))); };
        let eq = |ui: &mut egui::Ui, s: &str| { ui.label(RichText::new(format!("    {s}")).monospace().size(15.0).color(Color32::from_rgb(0xe8, 0xec, 0xf2))); };
        let here = |ui: &mut egui::Ui, s: Option<String>| if let Some(s) = s { ui.label(RichText::new(format!("        here: {s}")).monospace().size(14.0).color(GOLD)); };
        let fit = self.fit.as_ref().map(|f| &f.0);
        let km = fit.is_none_or(|r| r.c > 1e5);
        let len = |x: f64| if km { format!("{} km", fit::fcomma(x / 1e3, 1)) } else { format!("{} m", fit::fcomma(x, 1)) };
        let spd = |x: f64| if km { format!("{} km/s", fit::fx(x / 1e3, 4)) } else { format!("{} m/s", fit::fx(x, 3)) };
        let hz = |x: f64| format!("{} Hz", fit::fcomma(x, 1));
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_max_width(ui.available_width().min(980.0));
            ui.label(RichText::new("How the kit gets speed and distance").size(22.0).strong());
            p(ui, "Gold lines show the numbers of the current run. Each formula comes first, then the numbers filled in.");

            h(ui, "The picture");
            p(ui, "A source moves in a straight line at a constant speed v. At time t₀ it is closest to you, at distance d. \
                It sends a wave at frequency f₀. The wave moves at speed c. The kit finds v, d, t₀ and f₀ from the received frequency alone.");
            here(ui, fit.map(|r| format!("c = {} m/s", fit::fcomma(r.c, 0))));

            h(ui, "The received frequency");
            p(ui, "Let τ be the time when the wave left the source, counted from closest approach.");
            eq(ui, "distance:    r(τ) = √(d² + v² τ²)");
            eq(ui, "range rate:  r' = v² τ / √(d² + v² τ²)");
            p(ui, "The wave arrives later than it left, by the travel time r/c. The kit gets τ from the receive time t by putting the answer back in, 6 times:");
            eq(ui, "τ = (t − t₀) − (√(d² + v² τ²) − d) / c");
            p(ui, "The received frequency is:");
            eq(ui, "f(t) = f₀ / (1 + r'/c)");
            eq(ui, "f(t) = f₀ / (1 + v² τ / (c √(d² + v² τ²)))");
            here(ui, fit.map(|r| format!("f₀ = {}, v = {}, d = {}, t₀ = {:.2} s from the start", hz(r.p[3]), spd(r.p[1]), len(r.p[2]), r.p[0])));

            h(ui, "The plateaus give the speed");
            p(ui, "Long before closest approach, τ is large and negative, so r' is close to −v. Long after, r' is close to +v.");
            eq(ui, "coming:  f₊ = f₀ / (1 − v/c)");
            eq(ui, "going:   f₋ = f₀ / (1 + v/c)");
            here(ui, fit.map(|r| format!("f₊ = {}, f₋ = {}", hz(r.marks.f_plus_hz), hz(r.marks.f_minus_hz))));
            p(ui, "Solve the two lines for v and f₀:");
            eq(ui, "v = c (f₊ − f₋) / (f₊ + f₋)");
            here(ui, fit.map(|r| format!("v = {}", spd(r.marks.v_plateaus_m_s))));
            eq(ui, "f₀ = 2 f₊ f₋ / (f₊ + f₋)");
            here(ui, fit.map(|r| format!("f₀ = {}", hz(2.0 * r.marks.f_plus_hz * r.marks.f_minus_hz / (r.marks.f_plus_hz + r.marks.f_minus_hz)))));
            p(ui, "These use the plateaus of the fitted curve. The kit also reads the plateaus from the raw ends of the trace \
                (the first and the last 2 % of the points). A pass cut short by the horizon never reaches its plateaus, \
                so this raw estimate alone reads v and d low.");
            here(ui, fit.map(|r| format!("raw ends: v = {}, f₀ = {}, d = {}", spd(r.a.v), hz(r.a.f0), len(r.a.d))));

            h(ui, "The half-plateau points give the distance");
            p(ui, "A half-plateau point is where the distance changes at half the speed. Step by step:");
            p(ui, "Step 1. The range rate (from above):");
            eq(ui, "r' = v² τ / √(d² + v² τ²)");
            p(ui, "Step 2. Set it to half the speed: r' = −v/2 (coming) or r' = +v/2 (going). The frequency there is:");
            eq(ui, "f = f₀ / (1 ∓ v/2c)");
            here(ui, fit.map(|r| format!("f = {} and {}", hz(r.marks.half_plateau_hz[0]), hz(r.marks.half_plateau_hz[1]))));
            p(ui, "Step 3. Put r' = ±v/2 into the range rate:");
            eq(ui, "v² τ / √(d² + v² τ²) = ±v/2");
            p(ui, "Step 4. Multiply both sides by 2 √(d² + v² τ²) and divide by v:");
            eq(ui, "2 · v τ = ±√(d² + v² τ²)");
            p(ui, "Step 5. Square both sides (2² = 4):");
            eq(ui, "4 · v² τ² = d² + v² τ²");
            p(ui, "Step 6. Take v² τ² from both sides (4 − 1 = 3):");
            eq(ui, "3 · v² τ² = d²");
            p(ui, "Step 7. Divide by 3 v² and take the square root:");
            eq(ui, "τ = ±d / (v · √3)");
            p(ui, "Step 8. The two points are this far apart in time (from −d/(v√3) to +d/(v√3)):");
            eq(ui, "Δt = 2 · d / (v · √3)");
            here(ui, fit.map(|r| format!("points at {:.2} s and {:.2} s, Δt = {:.2} s", r.marks.half_plateau_t_s[0], r.marks.half_plateau_t_s[1], r.marks.dt_s)));
            p(ui, "Step 9. Solve for d:");
            eq(ui, "d = Δt · v · √3/2");
            here(ui, fit.map(|r| format!("d = {:.2} s × {} × √3/2 = {}", r.marks.dt_s, spd(r.p[1]), len(r.marks.d_width_m))));
            p(ui, "d is the closest distance along the line of sight, not the height. The Doppler curve depends only on the distance \
                to the path, so it is the same for every direction to the path: overhead, or low in the sky to one side. \
                d equals the height only for a pass straight overhead. To get the height you also need the direction to the source \
                at closest approach (its elevation angle).");
            ui.add_space(8.0);
            half_plateau_diagram(ui, fit);

            h(ui, "The slope at the middle gives the distance again");
            p(ui, "At closest approach τ = 0. Near there r' is close to v² τ / d, so the curve falls in a straight line:");
            eq(ui, "df/dt = −f₀ v² / (c d)");
            eq(ui, "d = f₀ v² / (c |df/dt|)");
            here(ui, fit.map(|r| format!("slope = {:.2} Hz/s, d = {}", r.marks.slope_hz_s, len(r.marks.d_slope_m))));

            h(ui, "The full fit");
            p(ui, "The kit changes t₀, v, d and f₀ until the curve f(t) passes as close as it can to all the measured points. \
                Close means the smallest sum of squares. It starts from the plateau and half-plateau values. \
                When you give f₀, f₀ stays fixed and only t₀, v and d change. This is the result to use.");
            eq(ui, "S = Σ (f_i − f(t_i))²     (as small as possible; i runs over the points)");
            eq(ui, "rms = √(S / n)");
            here(ui, fit.map(|r| format!("n = {} points, rms = {:.1} Hz", r.d.n, r.d.rms_hz)));
            here(ui, fit.map(|r| format!("v = {} ± {}, d = {} ± {}", spd(r.d.v_m_s), spd(r.err("v_m_s")), len(r.d.d_m), len(r.err("d_m")))));
            p(ui, "Errors: points can lie closer together than the window each one comes from (normal trace: 1 s windows every 0.25 s; \
                FM: 1 s windows every 0.5 s; burst: 0.5 s frames, no overlap). So only every k-th point is independent. \
                The kit allows for this: each error is √k times larger than with all points counted.");
            eq(ui, "error² = k · rms² · (the matching diagonal entry of (J^T J)⁻¹)");
            p(ui, "J is the table of how each point's f(t) changes when one value changes.");
            here(ui, fit.map(|r| format!("k = {}", fit::fg(r.indep, 3))));

            h(ui, "Bending path or not");
            p(ui, "What the fit leaves over at each point:");
            eq(ui, "e_i = f_i − f(t_i)");
            p(ui, "A straight path leaves scatter around zero. A bending path leaves a smooth wave. \
                The kit puts a polynomial of degree 6 through the leftover. Then an F-test compares the smooth part with the scatter, \
                counting only the independent points.");
            eq(ui, "n_eff = n / k");
            eq(ui, "F = ((RSS₀ − RSS₁) / 7) / (RSS₁ / (n_eff − 7))");
            p(ui, "RSS₀ is the sum of squares of the leftover. RSS₁ is the same after the polynomial is taken away. \
                p is the chance of an F this large from scatter alone.");
            eq(ui, "p below 0.001  →  bending.   Otherwise no bending detected.");
            here(ui, fit.map(|r| format!("smooth pattern {:.1} Hz rms, scatter {:.1} Hz rms, p = {}: {}",
                r.path.pattern_rms_hz, r.path.scatter_rms_hz, fit::fg(r.path.p_value, 2), r.path_word())));
            p(ui, "The smallest smooth pattern the test can see. N is the number of points, A the pattern rms, s the scatter rms. \
                The polynomial is a least-squares fit, so its part and the scatter add up:");
            eq(ui, "RSS₀ − RSS₁ = N A²");
            eq(ui, "RSS₁ = N s²");
            eq(ui, "F = (N A² / 7) / (N s² / (n_eff − 7))");
            eq(ui, "F = (A² / s²) · (n_eff − 7) / 7");
            p(ui, "Bending when F reaches F_crit, the value that F from scatter alone passes with a chance of 0.001:");
            eq(ui, "A² = s² · F_crit · 7 / (n_eff − 7)");
            eq(ui, "A_min = s · √(F_crit · 7 / (n_eff − 7))");
            here(ui, fit.map(|r| format!("A_min = {:.1} Hz rms (this recording)", r.path.detectable_rms_hz)));

            h(ui, "How the trace is measured");
            let mode = self.trace.as_ref().map(|t| if t.fm { "fm" } else if t.burst { "burst" } else { "normal" }).unwrap_or("normal");
            if mode == "fm" {
            p(ui, "FM mode (this run). An FM signal is spread over about 15 kHz, and its strongest part jumps around the carrier \
                with the picture or voice. The mean frequency of an FM signal is its carrier, so the kit measures the centre of the energy.");
            p(ui, "1.  Lower the sample rate to about 48,000 samples a second. Remove the receiver's DC line.");
            p(ui, "2.  1.0 s Hann windows every 0.5 s. Noise of a frame = its median bin power.");
            p(ui, "3.  Keep a frame when its strongest 2000 Hz band is at least 15 dB over the noise.");
            p(ui, "4.  The value is the centroid of the power above 5 × the noise, within ±10 kHz of that band. \
                The window moves to the centroid and the centroid is taken again, 3 times.");
            p(ui, "5.  A soft-L1 fit of the pass shape from a grid of starts; points more than 800 Hz off it are dropped. \
                The point noise is the spread of the rest around that shape.");
            } else if mode == "burst" {
            p(ui, "Burst mode (this run), for short FSK packets under steady lines.");
            p(ui, "1.  Lower the sample rate to about 48,000 samples a second. Remove the receiver's DC line.");
            p(ui, "2.  0.5 s Hann frames, no overlap; 100 Hz smoothing across frequency; each bin divided by its running median over ±20 s, \
                so steady lines go to 1. Excess = ratio − 1.");
            p(ui, "3.  Keep a frame when its strongest 2000 Hz band of excess is at least 6 dB. The value is the centroid of the excess in that band.");
            p(ui, "4.  A soft-L1 fit of the pass shape from a grid of starts; points more than 800 Hz off it are dropped.");
            } else {
            p(ui, "1.  Lower the sample rate to about 48,000 samples a second. Remove the receiver's DC line.");
            p(ui, "2.  Find the signal: 0.25 s spectra, every bin and every frame divided by its median, fixed receiver lines masked, \
                the strongest 300 Hz band in each frame. A smooth S-shaped curve through it steers the next steps only.");
            p(ui, "3.  Measure: multiply the recording by the opposite phase of that curve (dechirp), so the carrier sits near 0 Hz. \
                Then 1.0 s Hann windows every 0.25 s on a 1 Hz grid, searching ±400 Hz, then ±100 Hz, then ±150 Hz, \
                each time with a better reference. The value is the noise-subtracted power centroid in the band.");
            p(ui, "4.  Keep a point when its peak is more than 14 dB above the frame median, it is not on a fixed receiver line, \
                and it lies within 4 sigma of a smoothing spline. Sigma (the point noise) comes from second differences.");
            }
            here(ui, self.trace.as_ref().map(|t| format!("{} points, point noise {:.1} Hz, swing {}", t.points.len(), t.point_noise_hz, hz(t.swing_hz))));
            p(ui, "The waterfall picture: 0.1 s spectra every 0.5 s with 10 Hz bins, each bin in dB above the medians of its column and row.");

            h(ui, "Laser ranging");
            p(ui, "A laser station fires a short pulse at the satellite and times the return. Each shot gives a distance:");
            eq(ui, "r = round-trip time × c / 2");
            p(ui, "A satellite that flies straight past the station at speed v, closest distance d at time t₀, is at \
                (Pythagoras, t counted from closest approach):");
            eq(ui, "r² = d² + (v · t)²");
            p(ui, "So every shot gives the speed on its own:");
            eq(ui, "v = √(r² − d²) / |t|");
            p(ui, "The kit fits d, v and t₀ to all shots within ±3 minutes of the closest one.");
            let which = self.which.trim().parse::<usize>().unwrap_or(0);
            if let Some((ps, f)) = self.laser.as_ref().and_then(|(run, _)| run.passes.get(which)) {
                let (tk, rk, along) = crate::triangle(ps, f);
                here(ui, Some(format!("pass {which}: d = {:.1} km, v = {:.4} km/s", f.d_km, f.v_km_s)));
                here(ui, Some(format!("one shot at t = {tk:+.1} s: r = {rk:.1} km")));
                here(ui, Some(format!("√(r² − d²) = √({rk:.1}² − {:.1}²) = {along:.1} km", f.d_km)));
                here(ui, Some(format!("v = {along:.1} / {:.1} = {:.4} km/s", tk.abs(), along / tk.abs())));
            }
            ui.add_space(20.0);
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) { self.draw(ui); }
}

impl App {
    fn draw(&mut self, ui: &mut egui::Ui) {
        self.poll();
        if let Some(rx) = &self.picker {
            match rx.try_recv() {
                Ok((true, p)) => { self.out = p.display().to_string(); self.picker = None; }
                Ok((false, p)) => {
                    self.file = p.display().to_string();
                    self.start.clear();
                    self.span.clear();
                    self.picker = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.picker = None,
                Err(_) => ui.ctx().request_repaint_after(std::time::Duration::from_millis(100)),
            }
        }
        egui::Panel::left("controls").resizable(true).default_size(470.0).show(ui, |ui| {
            ui.add_space(6.0);
            self.controls(ui);
        });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (t, l) in [(Tab::Results, "Results"), (Tab::Waterfall, "Waterfall"), (Tab::SCurve, "S-curve"), (Tab::Marked, "Marked waterfall"), (Tab::Laser, "Laser"), (Tab::Math, "Math")] {
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
                Tab::Math => self.math_tab(ui),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waterfall_image_colours() {
        let mut db = vec![0.0f32; 15];
        db[2 * 5 + 4] = 12.5;
        let w = trace::Waterfall { t: vec![0.0, 0.5, 1.0], f: (0..5).map(|j| j as f64).collect(), db };
        let im = waterfall_image(&w, 2);
        assert_eq!(im.size, [2, 2]);
        assert_eq!(im.pixels[1], inferno((10.0 * 10f32.powf(1.25).log10() - WF_DB[0]) / (WF_DB[1] - WF_DB[0])));
        assert_eq!(im.pixels[1], Color32::from_rgb(186, 54, 85));
        assert_eq!(im.pixels[0], inferno(0.0));
        assert_eq!(inferno(0.0), Color32::from_rgb(0, 0, 4));
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) { return v; }
            std::thread::yield_now();
        }
    }

    /// Draws every tab off screen (no window) and writes one PPM per tab.
    /// DK_SHOT_INPUT=file [DK_SHOT_CENTER=Hz] DK_SHOT_OUT=dir/prefix cargo test --release render_tabs -- --ignored
    #[test]
    #[ignore]
    fn render_tabs_offscreen() {
        use eframe::egui_wgpu::{wgpu, Renderer, RendererOptions, ScreenDescriptor};
        let input = std::env::var("DK_SHOT_INPUT").expect("DK_SHOT_INPUT");
        let out = std::env::var("DK_SHOT_OUT").expect("DK_SHOT_OUT");
        let env_u = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (w, h) = (env_u("DK_SHOT_W", 1400), env_u("DK_SHOT_H", 900));
        let ctx = egui::Context::default();
        ctx.set_theme(egui::Theme::Dark);
        let mut app = App { file: input, centre: std::env::var("DK_SHOT_CENTER").unwrap_or_default(), fm: std::env::var("DK_SHOT_FM").is_ok(),
            start: std::env::var("DK_SHOT_START").unwrap_or_default(), ..App::default() };
        app.start(&ctx);
        while app.rx.is_some() { app.poll(); std::thread::sleep(std::time::Duration::from_millis(50)); }
        print!("{}", app.log);
        app.show_trace = std::env::var("DK_SHOT_NOTRACE").is_err();
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(inst.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (dev, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut rend = Renderer::new(&dev, fmt, RendererOptions::default());
        let tex = dev.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: fmt, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let row = (w * 4).div_ceil(256) * 256;
        let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let sd = ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: 1.0 };
        let tabs = [(Tab::Results, "results"), (Tab::Waterfall, "waterfall"), (Tab::SCurve, "scurve"), (Tab::Marked, "marked"), (Tab::Laser, "laser"), (Tab::Math, "math")];
        for (tab, name) in tabs {
            app.tab = tab;
            let mut prims = vec![];
            for _ in 0..4 {
                let raw = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
                    max_texture_side: Some(8192), ..Default::default()
                };
                let o = ctx.run_ui(raw, |ui| app.draw(ui));
                for (id, ds) in &o.textures_delta.set { ds.iter().for_each(|d| rend.update_texture(&dev, &queue, *id, d)); }
                prims = ctx.tessellate(o.shapes, o.pixels_per_point);
                for id in &o.textures_delta.free { rend.free_texture(id); }
            }
            let mut enc = dev.create_command_encoder(&Default::default());
            let cmds = rend.update_buffers(&dev, &queue, &mut enc, &prims, &sd);
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                }).forget_lifetime();
                rend.render(&mut rp, &prims, &sd);
            }
            enc.copy_texture_to_buffer(tex.as_image_copy(), wgpu::TexelCopyBufferInfo {
                buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
            }, wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            queue.submit(cmds.into_iter().chain([enc.finish()]));
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            {
                let data = buf.slice(..).get_mapped_range().unwrap();
                for y in 0..h as usize {
                    for x in 0..w as usize { ppm.extend_from_slice(&data[y * row as usize + x * 4..][..3]); }
                }
            }
            buf.unmap();
            std::fs::write(format!("{out}_{name}.ppm"), ppm).unwrap();
        }
    }
}
