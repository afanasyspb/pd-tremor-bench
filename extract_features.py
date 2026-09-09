"""
Window-level feature extraction for the PhysioNet TremorDB recordings.

Reads every record listed in RECORDS, converts m/s recordings to mm/s, cuts the
velocity signal into 2.0 s windows with a 1.0 s hop and computes 10
spectro-temporal features per window. The output table is the input of all
other scripts.

    python extract_features.py --data-dir data/tremordb --out data/tremordb_window_features.csv
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
DEFAULT_FS = 100

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def read_signal_txt(path):
    """Single-column ASCII signal file -> float array (non-numeric lines skipped)."""
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


def parse_subject_groups(path):
    """subject_description.txt -> (high-amplitude subjects, low-amplitude subjects)."""
    high, low, mode = set(), set(), None
    if not Path(path).exists():
        return high, low
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        u = line.upper()
        if "SUBJECTS WITH HIGH AMPLITUDE" in u:
            mode = "high"
            continue
        if "SUBJECTS WITH LOW AMPLITUDE" in u:
            mode = "low"
            continue
        if u.strip().startswith("SUBJECT"):
            ids = [s.lower() for s in u.split()[1:]]
            (high if mode == "high" else low if mode == "low" else set()).update(ids)
    return high, low


_FILE_LINE = re.compile(
    r"^(?P<subj>[gsv]\d+)\s+(?P<file>\S+)\s+(?P<range>\d+(?:\.\d+)?)\s+"
    r"(?P<unit>mm/s|m/s)\s+(?P<laser>\S+)\s+(?P<rate>\d+)\s+(?P<samples>\d+)")


def parse_file_description(path):
    """file_description.txt -> {filename: {subj, file, range, unit, laser, rate, samples}}."""
    meta = {}
    if not Path(path).exists():
        return meta
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _FILE_LINE.match(line.strip())
        if m:
            d = m.groupdict()
            d["range"], d["rate"], d["samples"] = float(d["range"]), int(d["rate"]), int(d["samples"])
            meta[d["file"]] = d
    return meta


def parse_record_name(filename):
    """'g10ren.let' -> ('g10', 'ren', 'L'); extension .let = left hand, .rit = right hand."""
    base, ext = os.path.splitext(filename)
    hand = {"let": "L", "rit": "R"}.get(ext.lower().lstrip("."))
    m = re.match(r"^(?P<subj>[gsv]\d+)(?P<cond>.+)$", base)
    if not m:
        return None, None, hand
    return m.group("subj"), m.group("cond"), hand


def cond_to_states(cond):
    """Condition code -> (dbs_state, med_state, minutes after DBS stop)."""
    if cond in ("ren", "en"):
        return "on", "on", None
    if cond in ("ref", "ef"):
        return "on", "off", None
    if cond in ("ron", "on"):
        return "off", "on", None
    if cond in ("rof", "of"):
        return "off", "off", None
    m = re.match(r"^r(?P<mins>\d+)of$", cond)
    if m:
        return "off", "off", int(m.group("mins"))
    return "unknown", "unknown", None


def bandpower(psd, freqs, f_low, f_high):
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return np.nan
    return float(_trapezoid(psd[mask], freqs[mask]))


def extract_features(signal, fs):
    """10 features of one window: rms, std, p2p, mad, peak_freq_hz, peak_psd,
    bp_3_12, bp_4_6, bp_6_12, rel_bp_3_12 (Hann-windowed periodogram)."""
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = len(x)

    rms = np.sqrt(np.mean(x ** 2))
    std = np.std(x)
    p2p = np.ptp(x)
    mad = np.median(np.abs(x - np.median(x)))

    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(window ** 2))

    if len(freqs) > 1:
        idx = np.argmax(psd[1:]) + 1
        peak_freq, peak_psd = freqs[idx], psd[idx]
    else:
        peak_freq, peak_psd = np.nan, np.nan

    bp_3_12 = bandpower(psd, freqs, 3, 12)
    bp_4_6 = bandpower(psd, freqs, 4, 6)
    bp_6_12 = bandpower(psd, freqs, 6, 12)
    bp_total = bandpower(psd, freqs, 0, fs / 2)
    rel_bp_3_12 = bp_3_12 / bp_total if bp_total and bp_total > 0 else np.nan

    return {
        "rms": float(rms), "std": float(std), "p2p": float(p2p), "mad": float(mad),
        "peak_freq_hz": float(peak_freq), "peak_psd": float(peak_psd),
        "bp_3_12": float(bp_3_12), "bp_4_6": float(bp_4_6), "bp_6_12": float(bp_6_12),
        "rel_bp_3_12": float(rel_bp_3_12),
    }


def load_record(data_dir, rec, meta):
    """Signal of one record in mm/s plus its sampling rate."""
    path = next(Path(data_dir).rglob(rec), None)
    if path is None:
        return None, None
    md = meta.get(rec)
    fs = int(md["rate"]) if md else DEFAULT_FS
    signal = read_signal_txt(path)
    if md and md["unit"] == "m/s":
        signal = signal * 1000.0
    return signal, fs


def build_table(data_dir):
    data_dir = Path(data_dir)
    records = (data_dir / "RECORDS").read_text().split()
    meta = parse_file_description(data_dir / "file_description.txt")
    high, low = parse_subject_groups(data_dir / "subject_description.txt")

    rows = []
    for rec in records:
        subj, cond, hand = parse_record_name(rec)
        dbs, med, mins = cond_to_states(cond)
        group = "high" if subj in high else "low" if subj in low else "unknown"
        label = {"off": 1, "on": 0}.get(dbs, np.nan)

        signal, fs = load_record(data_dir, rec, meta)
        if signal is None:
            print(f"record {rec} not found, skipped")
            continue
        win, hop = int(WINDOW_SECONDS * fs), int(HOP_SECONDS * fs)
        for start in range(0, len(signal) - win + 1, hop):
            row = {
                "record": rec, "subject": subj, "hand": hand, "condition": cond,
                "dbs_state": dbs, "med_state": med, "mins_after_dbs_stop": mins,
                "tremor_group": group, "fs_hz": fs, "unit": "mm/s",
                "window_start_s": start / fs, "window_size_s": WINDOW_SECONDS,
                "target_label": label,
            }
            row.update(extract_features(signal[start:start + win], fs))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/tremordb")
    ap.add_argument("--out", default="data/tremordb_window_features.csv")
    args = ap.parse_args()
    df = build_table(args.data_dir)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} windows from {df.record.nunique()} records -> {args.out}")
