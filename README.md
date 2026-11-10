# Doppler Kit

Measure the speed and the closest distance of a passing source from your own recording.
Only the frequency shift you receive is used. No station position, no orbit and no map are needed.
The kit also reads satellite laser ranging, where the distances themselves are measured.

It reads:

- a radio IQ recording,
- the audio output of a receiver,
- a plain sound recording of a passing source (a horn, an engine, a siren),
- a laser-ranging file or an EDC page.

From one recording you get:

- **v**: the speed of the source along its path.
- **d**: the closest distance between you and the source. When the pass goes straight over you, d is its height.
  From several passes, the smallest d comes closest to the height.
- **t₀**: the time of closest approach.
- **f₀**: the frequency the source actually transmits.
- **Path**: whether the source went past in a straight line or on a bending path.
- The distance at the first and last point of the trace.
- Plots of the waterfall and the S-curve, with the plateaus, the middle, the two half-plateau points and the width Δt marked.

This is a Rust port of the Python reference kit:
https://github.com/AlanSpaceAudits/theorema-egregium/tree/main/doppler_kit
It gives the same numbers.

## Download and run

Download the program for your system from the Releases page:

| System | File |
|---|---|
| Windows | `doppler-kit-windows-x86_64.exe` |
| Linux | `doppler-kit-linux-x86_64` |
| macOS (Apple silicon) | `doppler-kit-macos-arm64` |
| macOS (Intel) | `doppler-kit-macos-x86_64` |

On Linux and macOS, make the file executable first: `chmod +x doppler-kit-*`.
On macOS, the first start needs a right-click on the file and then Open.

Or build it yourself with Rust (https://rustup.rs):

```
cargo build --release
```

The program is then `target/release/doppler-kit`.

On Linux the build needs only Rust and a C compiler (`gcc` or `clang`). No other system packages are needed.
The window needs a desktop with X11 or Wayland, `libxkbcommon` and an OpenGL driver.
The Browse buttons use `xdg-desktop-portal`.

For compressed audio (`.ogg`, `.mp3`, `.flac`, `.m4a`, `.opus`, `.aac`) the kit uses `ffmpeg` when it is on the PATH.
Without ffmpeg it decodes the file itself.

## The window

Start the program with no arguments:

```
doppler-kit
```

1. Press Browse and pick the recording.
2. Leave a field blank and the kit works the value out from the file (see "Blank fields" below).
   For receiver audio, type the dial frequency (USB). The kit cannot find a dial frequency. For a sound recording, leave the centre blank.
3. For a sound recording choose "sound in air (343 m/s)".
4. Press Run.

After a run, each worked-out value shows in its empty field as "auto: ...". The values you type stay as you typed them.

The tabs show the results, the waterfall, the S-curve and the marked waterfall.
The Results tab lists every number: v, d, t₀, f₀, the plateaus, the half-plateau points and Δt,
the three ways to v and d, the path test and the distances.
For a laser-ranging file, the Laser tab shows the distance of each shot and the speed from each shot.
Save writes the files again into the output folder.

## The command line

```
doppler-kit trace FILE [--center HZ] [--start ISO_UTC] [--span HZ] [--rate SPS] [--format cs16|cf32|cu8] [--swap-iq] [--out DIR] [--band LO HI] [--burst] [--fm]
doppler-kit fit TRACE.csv [--c M_S] [--f0 HZ] [--out DIR]
doppler-kit run FILE [trace options] [--c M_S] [--f0 HZ]
doppler-kit laser SRC [--pass N] [--out DIR]
```

- `trace` measures the received frequency against time.
- `fit` fits a trace.
- `run` does both and prints the full results.
- `laser` fits a laser-ranging file (`.np2`, `.npt`, `.crd`) or an EDC page URL.

Examples:

```
doppler-kit run my_recording.wav
doppler-kit run my_recording.wav --center 145800000 --start 2026-06-01T12:00:00Z
doppler-kit run iq_5408034.raw --rate 48000 --center 437800000            # raw 16-bit IQ
doppler-kit run capture.cf32 --rate 250000 --center 437800000 --format cf32
doppler-kit run beacon_usb.ogg --center 145796000                         # receiver audio, dial frequency
doppler-kit run horn.wav --c 343                                          # sound
doppler-kit run aircraft.wav --band 1300 2700 --c 343.8                   # sound: engine tone between 1300 and 2700 Hz
doppler-kit run pocketqube.cf32 --rate 48000 --center 436650000 --format cf32 --burst   # short FSK bursts
doppler-kit laser https://edc.dgfi.tum.de/en/data/npt2/1166240/           # one pass, from its EDC page
doppler-kit laser icesat2_20261025.np2 --pass 3                           # a daily CRD file, pass 3
```

Options:

- `--center`: the centre frequency of the recording in Hz.
- `--start`: the UTC start time of the recording. The trace and t₀ then also carry UTC times.
- `--span`: search ± Hz around the centre. The default comes from the scan (see "Blank fields").
- `--rate`: the sample rate of a raw IQ file.
- `--format`: the sample type of a raw IQ file. The default comes from SigMF, else `cs16`.
- `--swap-iq`: Q is stored before I.
- `--band LO HI`: search only this band. See "Tone band" below.
- `--burst`: the signal comes in short bursts. See "Burst signals" below.
- `--fm`: the signal is an FM carrier (ISS SSTV, voice). See "FM carriers" below.
- `--c`: the wave speed in m/s. The default is 299792458 (radio and light). Use 343 for sound in air.
- `--f0`: fix the transmitted frequency.
- `--out`: the output folder. The default is the folder of the input file.

`fit` also takes a trace from anywhere else: a CSV with the columns `t_s` and `f_Hz`.

### Blank fields

A value you type always wins. For each blank field the kit looks in the file, in this order, and logs the value with its source,
for example `auto: centre 437,800,000 Hz (from file name)`.

1. SigMF: a `FILE.sigmf-meta` beside `FILE.sigmf-data` (or pick the `.sigmf-meta` file). It gives the sample rate,
   the sample format (`ci16_le`, `cf32_le`, `cu8`), the centre frequency and the start time.
2. WAV header: the sample rate, the channels and the sample size.
3. File name: an SDR++ name holds the centre frequency (`..._437800000Hz_...`).
4. SatNOGS: a file `iq_<observation id>.raw` is 48,000 S/s, 16-bit. The kit asks network.satnogs.org for the start time
   and the centre frequency of the observation. It sends only the observation id. When the computer is offline, the kit says so and goes on.
5. Start time: when no other source has it, the kit takes the time the file was last changed and subtracts the length of the recording.
   The log marks this value "check it". Copying a file can change this time.

Radio needs the real centre frequency for v. It must come from the file name, SigMF, SatNOGS or you.

**Search span (scan).** When the span is blank, the kit first looks over the full recorded band. It takes short spectra
spread over the whole file, divides every frequency bin by its median over time (a steady line goes to 1) and follows the strongest
line that falls in frequency. It looks near the centre first (±21.6 kHz), then over the whole band. The span then covers the line
plus a margin, and stays at 15,000 Hz when the line fits inside. A line outside the default window moves the centre to it.
When no moving line is found, the span is 15,000 Hz and the log says so.

**Tone band (audio).** When no band is given, the same scan picks the strongest tone that moves (falls in frequency) and sets the band
around its swing. A steady tone, such as low engine rumble, does not count. When the strongest tone in the recording is the moving one,
no band is needed and the kit tracks it as before.

**Burst mode.** When "Burst" is not ticked and the trace looks like bursts (it covers less than 30% of the recording, moves less than
a fifth of the span, or does not fit a pass cleanly), the kit also runs burst mode. It keeps the clean pass that covers more of the
recording and logs which one it used and why. A tick (`--burst`) forces burst mode.

On Windows the program does not open a console window of its own.
The command line writes to the console it was started from.
In `cmd` the prompt can come back before the output ends. Use `start /wait doppler-kit run ...` to wait for it.

### Tone band (`--band LO HI`)

Use it when a different sound or signal is stronger than the one you want.
For example, the low rumble of an aircraft engine (50 to 135 Hz) is often stronger than the engine tone.
Without a band, the trace jumps between the two.

- Audio: LO and HI are audio frequencies in Hz. The kit uses only the sound between LO and HI.
- IQ: LO and HI are offsets in Hz from the centre frequency, for example `--band -5000 3000`.

Look at the waterfall or a spectrum of the recording first. Choose a band that holds the whole S-curve of the tone
and leaves out the other strong sounds. The search span is then half the band width.

### Burst signals (`--burst`)

Some satellites do not send a steady carrier. They send short packets (FSK) every few seconds.
The recording can also hold steady lines from the receiver or from other transmitters, and these lines can be stronger than the packets.
The normal tracer then follows a steady line or only part of a packet.
`--burst` finds the packets under the steady lines and gives one point for each half second that holds a packet.
The trace file has the same columns, so `fit` works the same way.

When you do not give `--burst`, the kit tries burst mode by itself when the trace looks like bursts (see "Blank fields").

### FM carriers (`--fm`)

Some satellites send FM: the ISS sends SSTV pictures and voice on 145.800 MHz. FM spreads the signal over about 15 kHz,
and the strongest part of the spectrum jumps around the carrier, so the normal tracer and burst mode follow the picture
content, not the carrier. `--fm` (in the window: tick "FM carrier") gives one point every half second: the centre of the
signal's energy in each 1 s frame. The mean frequency of an FM signal is its carrier.

Each frame: noise = the median bin power; the strongest 2000 Hz band must be at least 15 dB over it; the point is the
centroid of the power above 5 times the noise, within ±10 kHz of that band, with the window moved to the centroid 3 times.
Outliers are dropped as in burst mode.

## Output

| File | Content |
|---|---|
| `<name>_trace.csv` | `t_s, utc, f_offset_Hz, f_Hz, snr_dB`: received frequency against time |
| `<name>_summary.json` | settings (with `band_Hz` and `burst`), number of points, point noise, the frequency swing, fixed receiver lines found, and under `auto` every value the kit worked out, with its source |
| `<name>_fit.json` | v, d, t₀, f₀ with errors (full fit), the plateau estimate, the marks, the path test with `detectable_rms_Hz`, the distances at the first and last point |
| `<name>_slr.csv` | laser ranging: v, d, closest time, shots and fit miss for every pass |

## What to record

Record at a **fixed frequency with Doppler correction off**, through the whole pass if you can:

- a WAV file with 2 channels (I and Q), 8-bit, 16-bit or 32-bit, such as an SDR++ "baseband" recording,
- a headerless raw IQ file (`.raw`, `.cs16`, `.cf32`, `.cu8`, `.sigmf-data`) with its sample rate, or
- audio: a 1-channel WAV, `.ogg`, `.mp3`, `.flac`, `.m4a`, `.opus` or `.aac`. This can be a receiver's audio output
  in USB or CW mode, or a plain sound recording of a passing source. The strongest tone is tracked,
  or the tone inside a band you give with `--band`.

A cheap RTL-SDR and a simple antenna are enough for the ISS (437.800 MHz FM repeater, 145.800 MHz voice).

### Doppler-corrected recordings do not work

Many receivers, SatNOGS stations among them, retune along the Doppler predicted from the published orbit while they record.
Their recordings hold only what is left over after that prediction, a nearly straight line instead of an S.
The kit prints a note when the signal moves less than a fifth of the search span.

Example: SatNOGS observation 5408034, the ISS at 437.800 MHz on 3 February 2022, recorded by the Dwingeloo radio telescope.
The ISS carrier stays within about 270 Hz for the whole pass. A pass recorded at a fixed tuning moves about ±10 kHz.

## How it works

**The trace**

1. Decimate to about 48,000 samples a second with a polyphase filter, and remove the receiver's DC line.
2. Find the signal: 0.25 s spectra, every bin and every frame divided by its median, fixed receiver lines masked,
   strongest 300 Hz band per frame. A smooth S-shaped curve through it steers the next steps only.
3. Measure: multiply the recording by the conjugate phase of that curve (dechirp), so the carrier sits near 0 Hz.
   Then 1.0 s Hann windows every 0.25 s on a 1 Hz grid, searching ±400 Hz, then ±100 Hz, then ±150 Hz,
   each time with a better reference. The value is the noise-subtracted power centroid in the band.
4. Keep a point when its peak is more than 14 dB above the frame median, it is not on a fixed receiver line,
   and it lies within 4 sigma of a smoothing spline. Sigma (the point noise) comes from second differences.

Points are 0.25 s apart but each uses a 1 s window, so only every 4th point is independent. The fit allows for this in its errors.

**With `--band LO HI`**

1. One FFT of the whole recording. All bins outside LO to HI are set to zero (for audio also all negative frequencies).
   Then the inverse FFT, and the band centre is moved to 0 Hz. The search span is (HI − LO)/2.
2. The steps above follow. In each measuring pass, the bins that came from outside the band are left out.
   The noise floor of a spectrum is the median of the bins inside the band.
3. Audio has no receiver DC line and no fixed receiver lines, so nothing is blanked. A tone that stays on one plateau for most of the clip stays in the trace.

**With `--burst`**

1. Spectra of 0.5 s (Hann window, no overlap), smoothed over 100 Hz.
2. Every bin is divided by its running median over ±20 s. A steady line, present most of the time, goes to 1. A short packet stands above 1.
3. In each spectrum, take the 2000 Hz band with the most excess power. Keep the spectrum when that band is 6 dB or more above the floor.
4. The point is the power centroid of the excess in that band. For a packet with two FSK tones, this is the middle between the two tones.
5. Fit the S-curve shape to all points with a robust fit, from 45 start values. Drop the points more than 800 Hz from that curve.

**The fit**

A source moving in a straight line at constant speed v, closest distance d at time t₀, transmitting f₀.
With the travel time of the signal included, the received frequency is

    f(t) = f₀ / (1 + v² τ / (c √(d² + v² τ²)))

where τ is the emission time from closest approach. Read off the curve:

- **Plateaus** f₊ = f₀/(1 − v/c) coming and f₋ = f₀/(1 + v/c) going, so v = c (f₊ − f₋)/(f₊ + f₋) and f₀ = 2 f₊ f₋/(f₊ + f₋).
- **Half-plateau points**: where the distance changes at half the speed (±v/2), f = f₀/(1 ∓ v/2c).
  The time between them is Δt, and d = Δt · v · √3/2.
- **Slope at the middle**: d = f₀ v² / (c · |slope|).
- **Full fit**: least squares of the whole curve, free t₀, v, d and f₀. This is the result to use. A pass cut short by
  the horizon never reaches its plateaus, so the plateau estimate alone reads v and d low.

What the fit leaves over shows the path: a straight path leaves scatter around zero, a bending path leaves a smooth wave.
The kit fits a degree-6 polynomial through the leftover and compares it with the scatter (F-test, counting only independent points).
When the smooth part stands out (p below 0.001) the path is reported as bending, otherwise as "no bending detected".

The report also gives the smallest smooth pattern this recording can detect. With N points, pattern rms A, scatter rms s
and the F-test degrees of freedom d₁ = 7 and d₂ = n_eff − 7:

    RSS₀ − RSS₁ = N A²
    RSS₁ = N s²
    F = (A² / s²) · d₂ / d₁
    bending when F ≥ F_crit, the 0.999 point of the F(d₁, d₂) distribution
    A_min = s · √(F_crit · d₁ / d₂)

A_min is `detectable_rms_Hz` in the fit file. A smaller pattern can be there and not be seen.
The Math tab shows these steps and the half-plateau triangle with the numbers of the current run.

## Laser ranging

Laser stations fire short pulses at a satellite and time each return. Each shot gives a distance directly:

    r = round-trip time × c / 2

A satellite flying straight past the station at speed v, closest distance d at time t₀, is at

    r² = d² + (v · t)²        (Pythagoras, t from closest approach)

so every shot gives the speed on its own:

    v = √(r² − d²) / |t|

The kit fits d, v and t₀ to all shots within ±3 minutes of the closest one.
The window takes `.np2`, `.npt` and `.crd` files and EDC page URLs.

Example: ICESat-2 past Herstmonceux (UK) on 25 October 2026, 44 shots. The last shot, 115.9 s after closest approach,
is 1,009.8 km away: √(1,009.8² − 537.3²) = 855.0 km along the path, and 855.0 / 115.9 = 7.38 km/s.

## Tested recordings

Recordings made by others, measured with the kit alone. No position, orbit or flight data went into the measurement.

**Aircraft sound** (`--band`, `--c 343.8`). Reference: a straight-line fit to the ADS-B positions over 10 s around closest approach.

| Aircraft (ICAO) | Band (Hz) | Kit: v | Kit: d | ADS-B: v | ADS-B: d |
|---|---|---|---|---|---|
| 7C6D29 | 1300 to 2700 | 82.0 ± 1.4 m/s | 229 ± 15 m | 82.9 m/s | 224 m |
| 7C6DA0 | 1300 to 2600 | 79.6 ± 1.5 m/s | 244 ± 15 m | 80.4 m/s | 223 m |
| 7C77D9 | 2000 to 4000 | 73.0 ± 2.3 m/s | 255 ± 26 m | 74.2 m/s | 246 m |
| 7C7773 | 1500 to 3000 | 107 ± 10 m/s | 735 ± 164 m | 74.8 m/s | 290 m |

In the 7C7773 clip the tone stops at 12.5 s, before the falling half of the S-curve. The fit has only one plateau to work with.

Data: Downward, B. (2023). *AeroSonicDB (YPAD-0523): Labelled audio dataset for acoustic detection and classification of aircraft* (Version 0.3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8004108 (https://zenodo.org/records/8004108, licence on the Zenodo page: CC BY 4.0).
ADS-B positions: ADS-B Exchange, historical data samples, https://www.adsbexchange.com/

**Satellites** (`--burst`). Reference: published-orbit model (TLE), straight-line fit over the same time span.

| Satellite | Kit: v | Kit: d | Kit: t₀ (UTC) | Published-orbit model (TLE): v | d |
|---|---|---|---|---|---|
| DELFI-PQ, 2022-01-16, 436.650 MHz | 7.234 ± 0.027 km/s | 890.9 ± 10.7 km | 17:58:57.11 | 7.238 km/s | 889.7 km |
| HADES-D, 2023-12-09, 436.666 MHz | 7.41 ± 0.15 km/s | 564 ± 28 km | 11:21:17.35 | 7.355 km/s | 544.8 km |

Data: Estévez, D. (2022). *Recording of DELFI-PQ/EASAT-2/HADES PocketQubes and other amateur satellites in the SpaceX Transporter 3 launch with the Allen Telescope Array on 2022-01-16* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5876090 (https://zenodo.org/records/5876090, CC BY 4.0).

Estévez, D. (2023). *Recording of HADES-D satellite* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10343047 (https://zenodo.org/records/10343047, CC BY 4.0).

The kit read narrow channels cut from these wideband recordings: 48,000 S/s around 436.650 MHz (DELFI-PQ), and 50,000 S/s around 436.666 MHz for the second part of the HADES-D recording.

**ISS** (`--fm`). The receiver position is not published, so there is no orbit reference for this pass.

| Recording | Kit: v | Kit: d | Kit: t₀ (UTC) |
|---|---|---|---|
| ISS SSTV, 2020-10-04, 145.800 MHz, `SDRSharp_20201004_025552Z_145449912Hz_IQ.wav` | 7.176 ± 0.034 km/s | 487.8 ± 5.5 km | 02:57:36.6 |

Data: SDR# IQ recordings of ISS SSTV, 2020-10-04, Internet Archive item `sdrsharp-20201004-025552-z-145449912-hz-iq`, https://archive.org/details/sdrsharp-20201004-025552-z-145449912-hz-iq (1,024,000 S/s, 16-bit IQ, fixed tuning). The start time is the UTC time in the file name.

## Related

- Python reference kit: https://github.com/AlanSpaceAudits/theorema-egregium/tree/main/doppler_kit
- Doppler simulator and calculator: https://alanspaceaudits.github.io/theorema-egregium/doppler_sim/

## Data credits

Laser ranging data: EUROLAS Data Center (EDC), DGFI-TUM, https://edc.dgfi.tum.de/

## License

MIT. See `LICENSE`.
