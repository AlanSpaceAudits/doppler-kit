//! Recording readers: IQ/audio WAV, headerless raw IQ, compressed audio (symphonia, ffmpeg fallback).
//! Samples come out as complex f32 blocks (audio: imaginary part 0), streamed so large files are never loaded whole.

use rustfft::num_complex::Complex32;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const COMPRESSED: &[&str] = &["ogg", "oga", "mp3", "flac", "m4a", "opus", "aac"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RawFormat {
    #[default]
    Cs16,
    Cf32,
    Cu8,
}

impl std::str::FromStr for RawFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cs16" => Ok(Self::Cs16),
            "cf32" => Ok(Self::Cf32),
            "cu8" => Ok(Self::Cu8),
            _ => Err(format!("unknown raw IQ format '{s}' (cs16, cf32 or cu8)")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sample {
    U8,
    I16,
    I32,
    F32,
}

impl Sample {
    fn size(self) -> usize {
        match self {
            Sample::U8 => 1,
            Sample::I16 => 2,
            Sample::I32 | Sample::F32 => 4,
        }
    }
}

enum Src {
    File { path: PathBuf, offset: u64, sample: Sample },
    Memory(Vec<Complex32>),
}

/// An opened recording. `rate` is the integer sample rate (as reported in the summary), `n` the number of
/// sample frames, `ch` 1 (audio) or 2 (I and Q). For compressed audio `rate` is the decoded rate; the tracer
/// resamples it to 48,000 S/s (`compressed` is set).
pub struct Input {
    pub rate: u32,
    pub n: u64,
    pub bits: u32,
    pub ch: usize,
    pub compressed: bool,
    src: Src,
}

pub trait BlockReader {
    /// Append up to `max` samples to `out`. Returns the number appended (0 at the end).
    fn read(&mut self, out: &mut Vec<Complex32>, max: usize) -> Result<usize, String>;
}

impl Input {
    pub fn audio(&self) -> bool {
        self.ch == 1
    }

    /// `len` sample frames starting at frame `start` (random access, for the signal scan).
    pub fn read_at(&self, start: u64, len: usize, swap: bool) -> Result<Vec<Complex32>, String> {
        let mut out = Vec::with_capacity(len);
        match &self.src {
            Src::Memory(v) => {
                let a = (start as usize).min(v.len());
                let b = (a + len).min(v.len());
                out.extend_from_slice(&v[a..b]);
            }
            Src::File { path, offset, sample } => {
                let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
                f.seek(SeekFrom::Start(offset + start * (self.ch * sample.size()) as u64)).map_err(|e| e.to_string())?;
                let mut rd = FileReader { f: BufReader::with_capacity(1 << 20, f), sample: *sample, ch: self.ch, left: (len as u64).min(self.n.saturating_sub(start)), swap, buf: Vec::new() };
                rd.read(&mut out, len)?;
            }
        }
        Ok(out)
    }

    pub fn reader(&self, swap: bool) -> Result<Box<dyn BlockReader + '_>, String> {
        match &self.src {
            Src::Memory(v) => Ok(Box::new(MemReader { v, pos: 0 })),
            Src::File { path, offset, sample } => {
                let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
                f.seek(SeekFrom::Start(*offset)).map_err(|e| e.to_string())?;
                Ok(Box::new(FileReader {
                    f: BufReader::with_capacity(1 << 20, f),
                    sample: *sample,
                    ch: self.ch,
                    left: self.n,
                    swap,
                    buf: Vec::new(),
                }))
            }
        }
    }
}

struct MemReader<'a> {
    v: &'a [Complex32],
    pos: usize,
}

impl BlockReader for MemReader<'_> {
    fn read(&mut self, out: &mut Vec<Complex32>, max: usize) -> Result<usize, String> {
        let b = (self.pos + max).min(self.v.len());
        out.extend_from_slice(&self.v[self.pos..b]);
        let k = b - self.pos;
        self.pos = b;
        Ok(k)
    }
}

struct FileReader {
    f: BufReader<File>,
    sample: Sample,
    ch: usize,
    left: u64,
    swap: bool,
    buf: Vec<u8>,
}

impl BlockReader for FileReader {
    fn read(&mut self, out: &mut Vec<Complex32>, max: usize) -> Result<usize, String> {
        let k = (max as u64).min(self.left) as usize;
        if k == 0 {
            return Ok(0);
        }
        let sz = self.sample.size();
        self.buf.resize(k * self.ch * sz, 0);
        self.f.read_exact(&mut self.buf).map_err(|e| format!("read error: {e}"))?;
        self.left -= k as u64;
        let conv = |b: &[u8]| -> f32 {
            match self.sample {
                Sample::U8 => (b[0] as f32 - 127.5) / 128.0,
                Sample::I16 => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                Sample::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0,
                Sample::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            }
        };
        out.reserve(k);
        for fr in self.buf.chunks_exact(self.ch * sz) {
            let a = conv(&fr[..sz]);
            if self.ch == 1 {
                out.push(Complex32::new(a, 0.0));
            } else {
                let b = conv(&fr[sz..2 * sz]);
                out.push(if self.swap { Complex32::new(b, a) } else { Complex32::new(a, b) });
            }
        }
        Ok(k)
    }
}

fn file_size(path: &Path) -> Result<u64, String> {
    std::fs::metadata(path).map(|m| m.len()).map_err(|e| format!("{}: {e}", path.display()))
}

/// Channels, rate, sample format and data offset. Length comes from the file size (SDR++ size fields overflow past 4 GB).
pub fn open_wav(path: &Path) -> Result<Input, String> {
    let mut head = Vec::new();
    File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(1 << 16)
        .read_to_end(&mut head)
        .map_err(|e| e.to_string())?;
    if head.len() < 12 || &head[..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([head[i], head[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([head[i], head[i + 1], head[i + 2], head[i + 3]]);
    let (mut i, mut fmt, mut data) = (12usize, None, None);
    while i + 8 <= head.len() {
        let size = u32_at(i + 4) as usize;
        let cid = &head[i..i + 4];
        if cid == b"fmt " && i + 24 <= head.len() {
            let mut tag = u16_at(i + 8);
            if tag == 0xFFFE && i + 34 <= head.len() {
                tag = u16_at(i + 32); // WAVE_FORMAT_EXTENSIBLE: sub-format GUID starts with the real tag
            }
            fmt = Some((tag, u16_at(i + 10), u32_at(i + 12), u16_at(i + 22)));
        }
        if cid == b"data" {
            data = Some(i as u64 + 8);
            break;
        }
        i += 8 + size + (size & 1);
    }
    let ((tag, ch, rate, bits), data) = match (fmt, data) {
        (Some(f), Some(d)) => (f, d),
        _ => return Err("no fmt/data chunk".into()),
    };
    if ch != 1 && ch != 2 {
        return Err("need 2 channels (I and Q) or 1 channel (audio)".into());
    }
    let sample = match (bits, tag == 3) {
        (8, _) => Sample::U8,
        (16, _) => Sample::I16,
        (32, true) => Sample::F32,
        (32, false) => Sample::I32,
        _ => return Err(format!("unsupported WAV sample size: {bits} bits (8, 16 or 32 only)")),
    };
    let n = file_size(path)?.saturating_sub(data) / (ch as u64 * sample.size() as u64);
    Ok(Input { rate, n, bits: bits as u32, ch: ch as usize, compressed: false, src: Src::File { path: path.into(), offset: data, sample } })
}

pub fn open_raw(path: &Path, rate: f64, fmt: RawFormat) -> Result<Input, String> {
    let (sample, bits) = match fmt {
        RawFormat::Cs16 => (Sample::I16, 16),
        RawFormat::Cf32 => (Sample::F32, 32),
        RawFormat::Cu8 => (Sample::U8, 8),
    };
    let n = file_size(path)? / (2 * sample.size() as u64);
    Ok(Input { rate: rate as u32, n, bits, ch: 2, compressed: false, src: Src::File { path: path.into(), offset: 0, sample } })
}

/// Compressed audio as 48 kHz 16-bit mono through ffmpeg, like the Python kit (decoders differ by 1 LSB on
/// some samples); without ffmpeg, symphonia at the file's own rate, quantised to 16 bit.
pub fn open_compressed(path: &Path) -> Result<Input, String> {
    let (v, rate) = match decode_ffmpeg(path) {
        Ok(r) => r,
        Err(e1) => decode_symphonia(path).map_err(|e2| format!("cannot decode {}: {e1}; {e2}", path.display()))?,
    };
    Ok(Input { rate, n: v.len() as u64, bits: 16, ch: 1, compressed: true, src: Src::Memory(v.into_iter().map(|x| Complex32::new(x, 0.0)).collect()) })
}

fn decode_symphonia(path: &Path) -> Result<(Vec<f32>, u32), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("symphonia: {e}"))?;
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).ok_or("symphonia: no audio track")?;
    let id = track.id;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut dec = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("symphonia: {e}"))?;
    let mut out = Vec::new();
    let mut sb: Option<SampleBuffer<f32>> = None;
    loop {
        let pkt = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(format!("symphonia: {e}")),
        };
        if pkt.track_id() != id {
            continue;
        }
        let buf = match dec.decode(&pkt) {
            Ok(b) => b,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("symphonia: {e}")),
        };
        let spec = *buf.spec();
        rate = spec.rate;
        let nch = spec.channels.count().max(1);
        if sb.as_ref().is_none_or(|s| s.capacity() < buf.capacity() * nch) {
            sb = Some(SampleBuffer::new(buf.capacity() as u64, spec));
        }
        let s = sb.as_mut().unwrap();
        s.copy_interleaved_ref(buf);
        out.extend(s.samples().chunks_exact(nch).map(|fr| fr.iter().sum::<f32>() / nch as f32));
    }
    if out.is_empty() || rate == 0 {
        return Err("symphonia: no samples decoded".into());
    }
    // The reference decodes through ffmpeg to pcm_s16le, so quantise to int16 the same way.
    for v in out.iter_mut() {
        *v = (*v * 32768.0).round().clamp(-32768.0, 32767.0) / 32768.0;
    }
    Ok((out, rate))
}

fn decode_ffmpeg(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let o = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "s16le", "-"])
        .output()
        .map_err(|_| "ffmpeg not found on PATH (needed for this audio format)".to_string())?;
    if !o.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&o.stderr).trim()));
    }
    let v: Vec<f32> = o.stdout.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
    if v.is_empty() {
        return Err("ffmpeg decoded no samples".into());
    }
    Ok((v, 48000))
}

/// Open any supported recording. `rate` / `fmt` apply to headerless raw IQ.
pub fn open(path: &Path, rate: Option<f64>, fmt: RawFormat) -> Result<Input, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if COMPRESSED.contains(&ext.as_str()) {
        return open_compressed(path);
    }
    if ext == "wav" {
        return open_wav(path);
    }
    let rate = rate.filter(|&r| r > 0.0).ok_or("give the sample rate for a raw IQ file")?;
    open_raw(path, rate, fmt)
}

/// SatNOGS observation id of an `iq_<digits>` raw file name (stem).
pub fn satnogs_id(stem: &str) -> Option<&str> {
    let d = stem.strip_prefix("iq_")?;
    (!d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).then_some(d)
}

/// Last-modified time of a file, seconds since 1970 (UTC).
pub fn mtime(path: &Path) -> Result<f64, String> {
    let m = std::fs::metadata(path).and_then(|m| m.modified()).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(match m.duration_since(std::time::UNIX_EPOCH) { Ok(d) => d.as_secs_f64(), Err(e) => -e.duration().as_secs_f64() })
}

// ---------- metadata: a small JSON reader (SigMF files, SatNOGS API) ----------
#[derive(Clone, Debug, PartialEq)]
pub enum Json { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let b = s.as_bytes();
        let mut i = 0;
        let v = parse_value(b, &mut i)?;
        skip_ws(b, &mut i);
        if i != b.len() { return Err("JSON: text after the value".into()); }
        Ok(v)
    }
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self { Json::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, x)| x), _ => None }
    }
    pub fn at(&self, i: usize) -> Option<&Json> {
        match self { Json::Arr(v) => v.get(i), _ => None }
    }
    pub fn num(&self) -> Option<f64> { if let Json::Num(x) = self { Some(*x) } else { None } }
    pub fn str(&self) -> Option<&str> { if let Json::Str(x) = self { Some(x) } else { None } }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() { *i += 1; }
}

fn parse_value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    let bad = |i: usize| format!("JSON: unexpected text at byte {i}");
    match b.get(*i) {
        Some(b'{') => {
            *i += 1;
            let mut v = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') { *i += 1; return Ok(Json::Obj(v)); }
            loop {
                skip_ws(b, i);
                let Json::Str(k) = parse_value(b, i)? else { return Err(bad(*i)) };
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') { return Err(bad(*i)); }
                *i += 1;
                v.push((k, parse_value(b, i)?));
                skip_ws(b, i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b'}') => { *i += 1; return Ok(Json::Obj(v)); } _ => return Err(bad(*i)) }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut v = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') { *i += 1; return Ok(Json::Arr(v)); }
            loop {
                v.push(parse_value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b']') => { *i += 1; return Ok(Json::Arr(v)); } _ => return Err(bad(*i)) }
            }
        }
        Some(b'"') => {
            *i += 1;
            let mut out: Vec<u16> = Vec::new();
            let mut s = String::new();
            let flush = |out: &mut Vec<u16>, s: &mut String| { s.push_str(&String::from_utf16_lossy(out)); out.clear(); };
            loop {
                let c = *b.get(*i).ok_or("JSON: string not closed")?;
                *i += 1;
                match c {
                    b'"' => { flush(&mut out, &mut s); return Ok(Json::Str(s)); }
                    b'\\' => {
                        let e = *b.get(*i).ok_or("JSON: string not closed")?;
                        *i += 1;
                        let ch = match e { b'n' => '\n', b't' => '\t', b'r' => '\r', b'b' => '\u{8}', b'f' => '\u{c}', b'u' => {
                            let h = std::str::from_utf8(b.get(*i..*i + 4).ok_or("JSON: bad escape")?).map_err(|_| "JSON: bad escape")?;
                            out.push(u16::from_str_radix(h, 16).map_err(|_| "JSON: bad escape")?);
                            *i += 4;
                            continue;
                        } o => o as char };
                        flush(&mut out, &mut s);
                        s.push(ch);
                    }
                    _ => {
                        flush(&mut out, &mut s);
                        // copy one UTF-8 character
                        let st = *i - 1;
                        let w = match c { 0..=0x7f => 1, 0xc0..=0xdf => 2, 0xe0..=0xef => 3, _ => 4 };
                        let e = (st + w).min(b.len());
                        s.push_str(&String::from_utf8_lossy(&b[st..e]));
                        *i = e;
                    }
                }
            }
        }
        Some(b't') if b[*i..].starts_with(b"true") => { *i += 4; Ok(Json::Bool(true)) }
        Some(b'f') if b[*i..].starts_with(b"false") => { *i += 5; Ok(Json::Bool(false)) }
        Some(b'n') if b[*i..].starts_with(b"null") => { *i += 4; Ok(Json::Null) }
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let st = *i;
            while *i < b.len() && (b[*i].is_ascii_digit() || matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E')) { *i += 1; }
            std::str::from_utf8(&b[st..*i]).ok().and_then(|t| t.parse::<f64>().ok()).map(Json::Num).ok_or(bad(st))
        }
        _ => Err(bad(*i)),
    }
}

/// (data file, meta file) of a SigMF pair: FILE.sigmf-meta picked, or FILE.sigmf-data with FILE.sigmf-meta beside it.
pub fn sigmf_files(path: &Path) -> (PathBuf, Option<PathBuf>) {
    let s = path.to_string_lossy();
    if let Some(stem) = s.strip_suffix(".sigmf-meta") {
        return (PathBuf::from(format!("{stem}.sigmf-data")), Some(path.to_path_buf()));
    }
    if let Some(stem) = s.strip_suffix(".sigmf-data") {
        let m = PathBuf::from(format!("{stem}.sigmf-meta"));
        if m.exists() { return (path.to_path_buf(), Some(m)); }
    }
    (path.to_path_buf(), None)
}

/// An ISO time from a metadata file as a start time the kit reads: fraction cut to 6 digits, no zone = UTC.
pub fn iso_utc(s: &str) -> Option<String> {
    let s = s.trim();
    let ok_body = |t: &str| !t.is_empty() && t.bytes().all(|c| c.is_ascii_digit() || c == b'-' || c == b'T' || c == b':') && t.contains('T');
    let (rest, zone) = if let Some(r) = s.strip_suffix(['Z', 'z']) { (r, "Z".to_string()) } else {
        match s.rfind(['+', '-']).filter(|&p| s[..p].contains('T')) {
            Some(p) if s[p + 1..].bytes().all(|c| c.is_ascii_digit() || c == b':') && !s[p + 1..].is_empty() => (&s[..p], s[p..].to_string()),
            _ => (s, "Z".to_string()),
        }
    };
    let (body, frac) = match rest.split_once(['.', ',']) { Some((a, f)) => (a, Some(f)), None => (rest, None) };
    if !ok_body(body) { return None; }
    let frac = match frac {
        Some(f) if !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit()) => format!(".{}", &f[..f.len().min(6)]),
        Some(_) => return None,
        None => String::new(),
    };
    Some(format!("{body}{frac}{zone}"))
}

/// Values a SigMF meta file gives: sample rate, format, centre (Hz) and start time (only those present).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meta { pub rate: Option<f64>, pub fmt: Option<RawFormat>, pub centre: Option<f64>, pub start: Option<String> }

pub fn sigmf_values(meta: &Path) -> Result<Meta, String> {
    let txt = std::fs::read_to_string(meta).map_err(|e| format!("{}: {e}", meta.display()))?;
    sigmf_parse(&txt)
}

pub fn sigmf_parse(txt: &str) -> Result<Meta, String> {
    let d = Json::parse(txt)?;
    let g = d.get("global");
    let c = d.get("captures").and_then(|c| c.at(0));
    let mut m = Meta { rate: g.and_then(|g| g.get("core:sample_rate")).and_then(Json::num), ..Default::default() };
    if let Some(dt) = g.and_then(|g| g.get("core:datatype")).and_then(Json::str) {
        m.fmt = Some(match dt.to_ascii_lowercase().as_str() {
            "ci16_le" | "ci16" => RawFormat::Cs16,
            "cf32_le" | "cf32" => RawFormat::Cf32,
            "cu8" | "cu8_le" => RawFormat::Cu8,
            _ => return Err(format!("SigMF datatype {dt} is not supported (ci16_le, cf32_le or cu8)")),
        });
    }
    m.centre = c.and_then(|c| c.get("core:frequency")).and_then(Json::num);
    m.start = c.and_then(|c| c.get("core:datetime")).and_then(Json::str).and_then(iso_utc);
    Ok(m)
}

/// Centre (Hz, with its source) and start of a SatNOGS observation from network.satnogs.org.
pub fn satnogs_values(obs: &str) -> Result<(Option<(f64, &'static str)>, Option<String>), String> {
    let url = format!("https://network.satnogs.org/api/observations/{obs}/");
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(10))).build().into();
    let mut resp = agent.get(&url).header("User-Agent", "doppler-kit").call().map_err(|e| e.to_string())?;
    let body = resp.body_mut().with_config().limit(4 << 20).read_to_string().map_err(|e| e.to_string())?;
    satnogs_parse(&body)
}

pub fn satnogs_parse(body: &str) -> Result<(Option<(f64, &'static str)>, Option<String>), String> {
    let d = Json::parse(body)?;
    let centre = match d.get("center_frequency").and_then(Json::num) {
        Some(f) => Some((f, "centre frequency")),
        None => d.get("transmitter_downlink_low").and_then(Json::num).map(|f| (f, "transmitter downlink")),
    };
    Ok((centre, d.get("start").and_then(Json::str).and_then(iso_utc)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmf_meta_is_read() {
        let txt = r#"{"global": {"core:author": "A \u00e9 \"q\"", "core:datatype": "ci16_le", "core:sample_rate": 3840000,
            "core:geolocation": {"coordinates": [-3.5, 40.5, 750], "type": "Point"}},
            "captures": [{"core:datetime": "2023-12-09T11:20:03.4641939375000000Z", "core:frequency": 436500000, "core:sample_start": 0}],
            "annotations": []}"#;
        let m = sigmf_parse(txt).unwrap();
        assert_eq!(m, Meta { rate: Some(3840000.0), fmt: Some(RawFormat::Cs16), centre: Some(436500000.0), start: Some("2023-12-09T11:20:03.464193Z".into()) });
        let m = sigmf_parse(r#"{"global": {"core:datatype": "cf32_le"}, "captures": [{"core:datetime": "2022-01-16T17:52:30.155110"}]}"#).unwrap();
        assert_eq!((m.fmt, m.start.as_deref(), m.rate), (Some(RawFormat::Cf32), Some("2022-01-16T17:52:30.155110Z"), None));
        assert!(sigmf_parse(r#"{"global": {"core:datatype": "ri16_le"}}"#).is_err());
        assert!(Json::parse("{\"a\": [1, 2,]}").is_err());
        let (c, st) = satnogs_parse(r#"{"start": "2026-08-23T10:23:10Z", "center_frequency": null, "transmitter_downlink_low": 435860000}"#).unwrap();
        assert_eq!((c, st.as_deref()), (Some((435860000.0, "transmitter downlink")), Some("2026-08-23T10:23:10Z")));
        assert_eq!(satnogs_id("iq_14860326"), Some("14860326"));
        assert_eq!(satnogs_id("iq_x1"), None);
        assert_eq!(sigmf_files(Path::new("/x/a.sigmf-meta")).0, PathBuf::from("/x/a.sigmf-data"));
    }
}

#[cfg(test)]
impl Input {
    pub fn from_memory_for_test(v: Vec<f32>) -> Input {
        Input { rate: 48000, n: v.len() as u64, bits: 32, ch: 1, compressed: false, src: Src::Memory(v.into_iter().map(|x| Complex32::new(x, 0.0)).collect()) }
    }
    pub fn iq_from_memory_for_test(v: Vec<Complex32>, rate: u32) -> Input {
        Input { rate, n: v.len() as u64, bits: 32, ch: 2, compressed: false, src: Src::Memory(v) }
    }
}
