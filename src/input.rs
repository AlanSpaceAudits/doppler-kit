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
    Memory(Vec<f32>),
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
    v: &'a [f32],
    pos: usize,
}

impl BlockReader for MemReader<'_> {
    fn read(&mut self, out: &mut Vec<Complex32>, max: usize) -> Result<usize, String> {
        let b = (self.pos + max).min(self.v.len());
        out.extend(self.v[self.pos..b].iter().map(|&x| Complex32::new(x, 0.0)));
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
    Ok(Input { rate, n: v.len() as u64, bits: 16, ch: 1, compressed: true, src: Src::Memory(v) })
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

/// Open any supported recording. `rate` / `fmt` apply to headerless raw IQ; SatNOGS `iq_<digits>.*` files
/// default to 48,000 S/s 16-bit.
pub fn open(path: &Path, rate: Option<f64>, fmt: RawFormat, log: &mut dyn FnMut(&str)) -> Result<Input, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if COMPRESSED.contains(&ext.as_str()) {
        return open_compressed(path);
    }
    if ext == "wav" {
        return open_wav(path);
    }
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut rate = rate.filter(|&r| r > 0.0);
    if rate.is_none() && name.len() > 3 && name.starts_with("iq_") && name[3..].bytes().all(|b| b.is_ascii_digit()) {
        rate = Some(48000.0);
        log("SatNOGS IQ file: 48,000 S/s, 16-bit assumed");
    }
    let rate = rate.ok_or("give the sample rate for a raw IQ file")?;
    open_raw(path, rate, fmt)
}

#[cfg(test)]
impl Input {
    pub fn from_memory_for_test(v: Vec<f32>) -> Input {
        Input { rate: 48000, n: v.len() as u64, bits: 32, ch: 1, compressed: false, src: Src::Memory(v) }
    }
}
