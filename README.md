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

1. Press Browse and pick the recording. An SDR++ file name holds the centre frequency (`baseband_437800000Hz_...`), and the field fills in.
2. Fill in the centre frequency if the file name does not hold it.
   For receiver audio, give the dial frequency (USB). For a sound recording, leave it blank.
3. For a sound recording choose "sound in air (343 m/s)".
4. Press Run.

The tabs show the results, the waterfall, the S-curve and the marked waterfall.
The Results tab lists every number: v, d, t₀, f₀, the plateaus, the half-plateau points and Δt,
the three ways to v and d, the path test and the distances.
For a laser-ranging file, the Laser tab shows the distance of each shot and the speed from each shot.
Save writes the files again into the output folder.

## The command line

```
doppler-kit trace FILE [--center HZ] [--start ISO_UTC] [--span HZ] [--rate SPS] [--format cs16|cf32|cu8] [--swap-iq] [--out DIR]
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
doppler-kit run my_recording.wav --center 145800000 --start 2026-06-14T18:10:09Z
doppler-kit run iq_5408034.raw --rate 48000 --center 437800000            # raw 16-bit IQ
doppler-kit run capture.cf32 --rate 250000 --center 437800000 --format cf32
doppler-kit run beacon_usb.ogg --center 145796000                         # receiver audio, dial frequency
doppler-kit run horn.wav --c 343                                          # sound
doppler-kit laser https://edc.dgfi.tum.de/en/data/npt2/1166240/           # one pass, from its EDC page
doppler-kit laser icesat2_20261025.np2 --pass 3                           # a daily CRD file, pass 3
```

Options:

- `--center`: the centre frequency of the recording in Hz.
- `--start`: the UTC start time of the recording. The trace and t₀ then also carry UTC times.
- `--span`: search ± Hz around the centre. The default is 15000.
- `--rate`: the sample rate of a raw IQ file.
- `--format`: the sample type of a raw IQ file. The default is `cs16`.
- `--swap-iq`: Q is stored before I.
- `--c`: the wave speed in m/s. The default is 299792458 (radio and light). Use 343 for sound in air.
- `--f0`: fix the transmitted frequency.
- `--out`: the output folder. The default is the folder of the input file.

`fit` also takes a trace from anywhere else: a CSV with the columns `t_s` and `f_Hz`.

On Windows the program does not open a console window of its own.
The command line writes to the console it was started from.
In `cmd` the prompt can come back before the output ends. Use `start /wait doppler-kit run ...` to wait for it.

## Output

| File | Content |
|---|---|
| `<name>_trace.csv` | `t_s, utc, f_offset_Hz, f_Hz, snr_dB`: received frequency against time |
| `<name>_summary.json` | settings, number of points, point noise, the frequency swing, fixed receiver lines found |
| `<name>_fit.json` | v, d, t₀, f₀ with errors (full fit), the plateau estimate, the marks, the path test, the distances at the first and last point |
| `<name>_slr.csv` | laser ranging: v, d, closest time, shots and fit miss for every pass |

## What to record

Record at a **fixed frequency with Doppler correction off**, through the whole pass if you can:

- a WAV file with 2 channels (I and Q), 8-bit, 16-bit or 32-bit, such as an SDR++ "baseband" recording,
- a headerless raw IQ file (`.raw`, `.cs16`, `.cf32`, `.cu8`, `.sigmf-data`) with its sample rate, or
- audio: a 1-channel WAV, `.ogg`, `.mp3`, `.flac`, `.m4a`, `.opus` or `.aac`. This can be a receiver's audio output
  in USB or CW mode, or a plain sound recording of a passing source. The strongest tone is tracked.

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
When the smooth part stands out (p below 0.001) the path is reported as bending, otherwise as straight.

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

## Related

- Python reference kit: https://github.com/AlanSpaceAudits/theorema-egregium/tree/main/doppler_kit
- Doppler simulator and calculator: https://alanspaceaudits.github.io/theorema-egregium/doppler_sim/

## Data credits

Laser ranging data: EUROLAS Data Center (EDC), DGFI-TUM, https://edc.dgfi.tum.de/

## License

MIT. See `LICENSE`.
