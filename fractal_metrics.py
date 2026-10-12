#!/usr/bin/env python
"""Fractal / multifractal metrics for induction-motor fault diagnosis.
everything imports this module.
    python fractal_metrics.py --jobs 1
    python fractal_metrics.py --only 10_faulty_phase-removal_startup
    python fractal_metrics.py --only <stem> --out results/tables/subset.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fathon
    HAS_FATHON = True
except Exception:
    fathon = None
    HAS_FATHON = False

try:
    import antropy
    HAS_ANTROPY = True
except Exception:
    antropy = None
    HAS_ANTROPY = False

# 1 paths, acquisition constants and the dataset registry

ROOT = Path(__file__).resolve().parent      # the ten CSVs
DATA_DIR = ROOT
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (CACHE_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FS_NOMINAL = 50_000.0     # nominal freq
ZOH = 5                   # sample repeat constant
FS = FS_NOMINAL / ZOH     # real freq
F0_NOMINAL = 50.0         # supply freq
DURATION_S = 20.0         # record length

CHANNELS = ("x", "y", "Z", "I1", "I2", "I3", "V1", "V2", "V3")
ACCEL = ("x", "y", "Z")           # acceleration
CURRENT = ("I1", "I2", "I3")      # A
VOLTAGE = ("V1", "V2", "V3")      # V

UNITS = {**{c: "g" for c in ACCEL},
         **{c: "A" for c in CURRENT},
         **{c: "V" for c in VOLTAGE}}

# analysis defaults
WIN_S = 2.0               # window length
STEP_S = 0.5              # window hop
NHARM = 15                # harmonics of f0 removed by least squares
Q_LIST = tuple(round(-5 + 0.5 * k, 1) for k in range(21))   # -5 .. +5, step 0.5
MFDFA_POLORD = 2          # detrending polynomial order in MFDFA/MFDCCA
MFDCCA_POLORD = 2


@dataclass(frozen=True)
class Record:
    """One 20 s recording from the public dataset."""
    idx: int
    stem: str
    motor: str
    condition: str
    load_nm: float
    phase_loss: bool
    description: str


_FILES = [
    ("01_healthy_operation_no-load", "healthy", "normal", 0.0, False,
     "Healthy motor, normal three-phase operation, no load"),
    ("02_healthy_phase-removal_operation", "healthy", "phase_removal", 0.0, True,
     "Healthy motor, phase removed during operation, no load"),
    ("03_healthy_operation_04Nm-load", "healthy", "load", 0.4, False,
     "Healthy motor, 0.4 Nm mechanical load"),
    ("04_healthy_operation_08Nm-load", "healthy", "load", 0.8, False,
     "Healthy motor, 0.8 Nm mechanical load"),
    ("05_healthy_phase-removal_startup", "healthy", "startup_phase_loss", 0.0, True,
     "Healthy motor started with one phase disconnected"),
    ("06_faulty_operation_no-load", "faulty", "normal", 0.0, False,
     "Faulty motor (outer-race bearing defect), normal operation, no load"),
    ("07_faulty_phase-removal_operation", "faulty", "phase_removal", 0.0, True,
     "Faulty motor, phase removed during operation, no load"),
    ("08_faulty_operation_04Nm-load", "faulty", "load", 0.4, False,
     "Faulty motor, 0.4 Nm mechanical load"),
    ("09_faulty_operation_08Nm-load", "faulty", "load", 0.8, False,
     "Faulty motor, 0.8 Nm mechanical load"),
    ("10_faulty_phase-removal_startup", "faulty", "startup_phase_loss", 0.0, True,
     "Faulty motor started with one phase disconnected"),
]

RECORDS = [Record(i + 1, stem, motor, cond, load, pl, desc)
           for i, (stem, motor, cond, load, pl, desc) in enumerate(_FILES)]
BY_STEM = {r.stem: r for r in RECORDS}
MOTORS = ("healthy", "faulty")
CONDITIONS = ("normal", "load", "phase_removal", "startup_phase_loss")

# 2 loading and caching (zero-order hold remove)

def detect_zoh(v: np.ndarray, n: int = 200_000) -> int:
    """Dominant run length of identical consecutive samples."""
    v = np.asarray(v, dtype=np.float64)[:n]
    d = np.flatnonzero(np.diff(v) != 0)
    if len(d) < 10:
        return 1
    runs = np.diff(d)
    runs = runs[runs > 0]
    if runs.size == 0:
        return 1
    vals, cnt = np.unique(runs, return_counts=True)
    base = int(vals[int(np.argmax(cnt))])
    while base > 1:
        rem = runs % base
        if np.count_nonzero(rem) > 1e-4 * runs.size:
            base -= 1
        else:
            break
    return max(base, 1)


def load_record(rec: Record, force: bool = False) -> dict[str, np.ndarray]:
    """Return {channel: float64 array} sampled at the effective rate FS."""
    cache = CACHE_DIR / f"{rec.stem}.npz"
    if cache.exists() and not force:
        with np.load(cache) as z:
            return {c: z[c] for c in CHANNELS}

    df = pd.read_csv(DATA_DIR / f"{rec.stem}.csv", dtype=np.float64, engine="c")
    zoh = detect_zoh(df["I1"].to_numpy())
    out = {c: np.ascontiguousarray(df[c].to_numpy()[::zoh]) for c in CHANNELS}
    np.savez_compressed(cache, **out)
    return out

# 3 regime segmentation

OFF_V = 50.0           # supply RMS below - dgs
OPEN_PHASE_A = 0.10    # rms below - open phase
TRANSIENT_REL = 0.08
STATIONARY_TOL = 0.05


@dataclass
class Segment:
    """A maximal run of one supply regime inside one record."""
    stem: str
    idx: int
    motor: str
    condition: str
    load_nm: float
    phase_loss: bool
    regime: str          # off | balanced | open_phase
    t0: float
    t1: float
    supply_v: float
    stats: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _block_rms(v: np.ndarray, fs: float, block_s: float = 1.0) -> np.ndarray:
    """RMS of consecutive non-overlapping blocks of `block_s` seconds."""
    n = int(round(block_s * fs))
    m = v.size // n
    return np.sqrt((v[:m * n].reshape(m, n) ** 2).mean(axis=1))


def supply_rms(sig: dict[str, np.ndarray], fs: float = FS,
               block_s: float = 1.0) -> np.ndarray:
    """Mean RMS of the three phase voltages, per block."""
    return np.mean([_block_rms(sig[c], fs, block_s) for c in VOLTAGE], axis=0)


def base_regime(sig: dict[str, np.ndarray], fs: float = FS,
                block_s: float = 1.0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-block physical state: off | balanced | open_phase."""
    supply = supply_rms(sig, fs, block_s)
    i2 = _block_rms(sig["I2"], fs, block_s)
    labels = []
    for k in range(supply.size):
        if supply[k] < OFF_V:
            labels.append("off")
        elif i2[k] < OPEN_PHASE_A:
            labels.append("open_phase")
        else:
            labels.append("balanced")
    return labels, supply, i2


def regime_series(sig: dict[str, np.ndarray], fs: float = FS,
                  block_s: float = 1.0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """`base_regime` with an extra `transient` label, for the audit plot."""
    labels, supply, i2 = base_regime(sig, fs, block_s)
    out = list(labels)
    for k in range(1, len(labels)):
        if labels[k] == labels[k - 1] != "off" and \
                abs(supply[k] - supply[k - 1]) > TRANSIENT_REL * max(supply[k], supply[k - 1]):
            out[k] = "transient"
    return out, supply, i2


def _runs(labels: list[str]) -> list[tuple[str, int, int]]:
    runs, start = [], 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start]:
            runs.append((labels[start], start, k))
            start = k
    return runs


def segments_for(rec: Record, sig: dict[str, np.ndarray],
                 block_s: float = 1.0) -> list[Segment]:
    """Split a record into maximal same-regime segments."""
    labels, supply, i2 = base_regime(sig, FS, block_s)
    out = []
    for lab, a, b in _runs(labels):
        out.append(Segment(
            stem=rec.stem, idx=rec.idx, motor=rec.motor,
            condition=rec.condition, load_nm=rec.load_nm,
            phase_loss=rec.phase_loss, regime=lab,
            t0=a * block_s, t1=b * block_s,
            supply_v=float(np.mean(supply[a:b])),
            stats={"i2_rms": float(np.mean(i2[a:b]))},
        ))
    return out


def is_stationary(sig: dict[str, np.ndarray], i0: int, i1: int,
                  tol: float = STATIONARY_TOL) -> bool:
    """Supply RMS must not drift by more than `tol` between half-windows."""
    h = (i0 + i1) // 2
    if h <= i0:
        return False
    a = float(np.mean([np.sqrt(np.mean(sig[c][i0:h] ** 2)) for c in VOLTAGE]))
    b = float(np.mean([np.sqrt(np.mean(sig[c][h:i1] ** 2)) for c in VOLTAGE]))
    m = max(a, b)
    return m > OFF_V and abs(a - b) <= tol * m


def analysis_windows(rec: Record, sig: dict[str, np.ndarray],
                     win_s: float = WIN_S, step_s: float = STEP_S,
                     usable: tuple[str, ...] = ("balanced", "open_phase"),
                     ) -> list[tuple[Segment, int, int]]:
    """(segment, i0, i1) triples for the stationary windows of a record.

    Windows from the `off` regime are never returned: with the machine
    de-energised, every channel is pure quantisation noise.
    """
    fs = FS
    nw, ns = int(round(win_s * fs)), max(1, int(round(step_s * fs)))
    out = []
    for seg in segments_for(rec, sig):
        if seg.regime not in usable or seg.duration < win_s:
            continue
        i0 = int(round(seg.t0 * fs))
        i1 = int(round(seg.t1 * fs))
        for s in range(i0, i1 - nw + 1, ns):
            if is_stationary(sig, s, s + nw):
                out.append((seg, s, s + nw))
    return out


def window_context(sig: dict[str, np.ndarray], i0: int, i1: int,
                   fs: float = FS) -> dict:
    """Physical context measured on a window, independently of any label."""
    cur = sig["I1"][i0:i1]
    f0 = estimate_f0(cur, fs)
    return {
        "f0": f0,
        "supply_v": float(np.mean([np.sqrt(np.mean(sig[c][i0:i1] ** 2))
                                   for c in VOLTAGE])),
        "i1_rms": float(np.sqrt(np.mean(cur ** 2))),
        "i2_rms": float(np.sqrt(np.mean(sig["I2"][i0:i1] ** 2))),
        "i3_rms": float(np.sqrt(np.mean(sig["I3"][i0:i1] ** 2))),
        "acc_rms": float(np.mean([np.sqrt(np.mean(sig[c][i0:i1] ** 2))
                                  for c in ACCEL])),
    }

# 4 signal conditioning

def estimate_f0(x: np.ndarray, fs: float, fmin: float = 40.0,
                fmax: float = 60.0) -> float:
    """Dominant supply frequency [Hz] by Hann-windowed FFT + parabolic refine."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 16:
        return float("nan")
    nfft = 1 << int(np.ceil(np.log2(n * 16)))
    sp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    lo = int(np.searchsorted(f, fmin))
    hi = max(int(np.searchsorted(f, fmax)), lo + 1)
    k = lo + int(np.argmax(sp[lo:hi]))
    if 0 < k < sp.size - 1:
        y0, y1, y2 = sp[k - 1], sp[k], sp[k + 1]
        den = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        delta = 0.0
    return float(f[k] + delta * (fs / nfft))


def harmonic_basis(n: int, fs: float, f0: float, nharm: int) -> np.ndarray:
    """Design matrix of cos/sin pairs for f0 and its first `nharm` harmonics."""
    t = np.arange(n, dtype=np.float64) / fs
    cols = []
    for k in range(1, nharm + 1):
        w = 2.0 * np.pi * k * f0 * t
        cols.append(np.cos(w))
        cols.append(np.sin(w))
    return np.column_stack(cols)


def remove_harmonics(x: np.ndarray, fs: float, f0: float,
                     nharm: int = NHARM) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares removal of f0 and its first `nharm` harmonics."""
    x = np.asarray(x, dtype=np.float64)
    A = harmonic_basis(x.size, fs, f0, nharm)
    coeffs, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coeffs, coeffs


def fundamental_amplitude(coeffs: np.ndarray) -> float:
    """Amplitude of the fundamental component [channel unit]."""
    return float(np.hypot(coeffs[0], coeffs[1]))

# 5a nultifractal metrics
def _require_fathon() -> None:
    if not HAS_FATHON:
        raise RuntimeError(
            "fathon is required for MFDFA/MFDCCA: pip install fathon")


def to_aggregated(x: np.ndarray) -> np.ndarray:
    """Profile (random-walk) series: cumulative sum of the mean-removed signal."""
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x - x.mean())


def log_window_sizes(n: int, nmin: int = 32, nmax_frac: float = 8.0,
                     count: int = 24) -> np.ndarray:
    """Log-spaced, unique, integer window sizes for DFA-type estimators."""
    nmax = max(n // int(nmax_frac), nmin * 2)
    if nmax <= nmin:
        nmax = nmin * 2
    w = np.unique(np.logspace(np.log10(nmin), np.log10(nmax), count).astype(np.int64))
    return w.astype(np.int64)


def _slopes(win: np.ndarray, F: np.ndarray, i0: int, i1: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-q slope and R^2 of log F vs log n over window indices [i0, i1)."""
    ln = np.log10(win[i0:i1].astype(np.float64))
    lf = np.log10(np.maximum(F[:, i0:i1], 1e-300))
    h = np.empty(F.shape[0])
    r2 = np.empty(F.shape[0])
    for k in range(F.shape[0]):
        y = lf[k]
        ok = np.isfinite(y) & np.isfinite(ln)
        if ok.sum() < 3:
            h[k], r2[k] = np.nan, np.nan
            continue
        p = np.polyfit(ln[ok], y[ok], 1)
        fit = np.polyval(p, ln[ok])
        ss_res = float(((y[ok] - fit) ** 2).sum())
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        h[k] = p[0]
        r2[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return h, r2


def _d_tau_dq(q: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """d(tau)/dq"""
    try:
        from scipy.signal import savgol_filter
        w = 7 if len(q) >= 7 else (len(q) - 1 if len(q) % 2 == 0 else len(q))
        if w >= 5:
            return np.gradient(savgol_filter(tau, w, 3), q)
    except Exception:                                      # noqa: BLE001
        pass
    return np.gradient(tau, q)


def _spectrum(q: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    """tau(q), alpha(q) and f(alpha) from h(q)."""
    tau = q * h - 1.0
    alpha = _d_tau_dq(q, tau)
    f_alpha = q * alpha - tau
    return {"tau": tau, "alpha": alpha, "f_alpha": f_alpha}


def _shape_features(q: np.ndarray, h: np.ndarray, r2: np.ndarray,
                    spec: dict[str, np.ndarray], prefix: str) -> dict[str, float]:
    """Scalar descriptors of one multifractal spectrum."""
    alpha, f = spec["alpha"], spec["f_alpha"]
    ok = np.isfinite(alpha) & np.isfinite(f)
    if ok.sum() < 3:
        return {}
    a, fa = alpha[ok], f[ok]
    i_pk = int(np.argmax(fa))
    a0 = float(a[i_pk])
    a_min, a_max = float(a.min()), float(a.max())
    return {
        f"{prefix}_h2": float(h[int(np.argmin(np.abs(q - 2.0)))]),
        f"{prefix}_h_range": float(np.nanmax(h) - np.nanmin(h)),
        f"{prefix}_alpha0": a0,
        f"{prefix}_alpha_min": a_min,
        f"{prefix}_alpha_max": a_max,
        f"{prefix}_dalpha": a_max - a_min,
        f"{prefix}_asym": (a_max - a0) - (a0 - a_min),
        f"{prefix}_fmax": float(fa.max()),
        f"{prefix}_df": float(fa.max() - fa.min()),
        f"{prefix}_r2_min": float(np.nanmin(r2)),
        f"{prefix}_r2_mean": float(np.nanmean(r2)),
    }


def mfdfa(x: np.ndarray, q_list: tuple = Q_LIST,
          win: np.ndarray | None = None, polord: int = MFDFA_POLORD,
          fit_lo: int = 1, fit_hi: int | None = None, prefix: str = "mfdfa",
          ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Multifractal DFA of one series."""
    _require_fathon()
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if win is None:
        win = log_window_sizes(n)
    win = win[(win >= 8) & (win <= n // 2)].astype(np.int64)
    if win.size < 4:
        return {}, {}
    q = np.asarray(q_list, dtype=np.float64)
    m = fathon.MFDFA(to_aggregated(x))
    _, F = m.computeFlucVec(win, q, polOrd=polord, revSeg=False)
    F = np.asarray(F, dtype=np.float64)
    hi = fit_hi if fit_hi is not None else win.size
    h, r2 = _slopes(win, F, fit_lo, hi)
    spec = _spectrum(q, h)
    feats = _shape_features(q, h, r2, spec, prefix)
    curves = {"win": win, "q": q, "F": F, "h": h, "r2": r2,
              "alpha": spec["alpha"], "f_alpha": spec["f_alpha"],
              "tau": spec["tau"]}
    return feats, curves


def mfdcca(x: np.ndarray, y: np.ndarray, q_list: tuple = Q_LIST,
           win: np.ndarray | None = None, polord: int = MFDCCA_POLORD,
           fit_lo: int = 1, fit_hi: int | None = None, prefix: str = "dcca",
           ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Multifractal detrended cross-correlation between two series."""
    _require_fathon()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if win is None:
        win = log_window_sizes(n)
    win = win[(win >= 8) & (win <= n // 2)].astype(np.int64)
    if win.size < 4:
        return {}, {}
    q = np.asarray(q_list, dtype=np.float64)
    d = fathon.MFDCCA(to_aggregated(x), to_aggregated(y))
    _, F = d.computeFlucVec(win, q, polOrd=polord, revSeg=False)
    F = np.asarray(F, dtype=np.float64)
    hi = fit_hi if fit_hi is not None else win.size
    h, r2 = _slopes(win, F, fit_lo, hi)
    spec = _spectrum(q, h)
    feats = _shape_features(q, h, r2, spec, prefix)
    curves = {"win": win, "q": q, "F": F, "h": h, "r2": r2,
              "alpha": spec["alpha"], "f_alpha": spec["f_alpha"],
              "tau": spec["tau"]}
    return feats, curves

# 5b single-scale comparators

def higuchi_fd(x: np.ndarray, kmax: int = 64) -> float:
    """Higuchi fractal dimension."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    kmax = int(min(kmax, n // 4))
    if kmax < 2:
        return float("nan")
    ln_L, ln_k = [], []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            seg = np.abs(np.diff(x[idx])).sum()
            Lm.append(seg * (n - 1) / ((idx.size - 1) * k) / k)
        if Lm:
            L = float(np.mean(Lm))
            if L > 0:
                ln_L.append(np.log(L))
                ln_k.append(np.log(1.0 / k))
    if len(ln_L) < 3:
        return float("nan")
    return float(np.polyfit(ln_k, ln_L, 1)[0])


def spectral_entropy(x: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum (0 = tonal, 1 = flat)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.std() == 0:
        return float("nan")
    p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    p = p[1:]
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def classic_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Amplitude statistics plus the single-scale comparators."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    sd = float(x.std())
    if sd > 0:
        z = (x - x.mean()) / sd
        kurt = float((z ** 4).mean())
        skew = float((z ** 3).mean())
    else:
        kurt = skew = float("nan")
    peak = float(np.max(np.abs(x))) if x.size else float("nan")
    return {
        f"{prefix}_rms": rms,
        f"{prefix}_std": sd,
        f"{prefix}_kurt": kurt,
        f"{prefix}_skew": skew,
        f"{prefix}_crest": peak / rms if rms > 0 else float("nan"),
        f"{prefix}_higuchi": higuchi_fd(x),
        f"{prefix}_specent": spectral_entropy(x),
    }

# 5c entropy and complexity metrics

PERM_ORDER = 5
SAMP_ORDER = 2
SVD_ORDER = 3
MSE_SCALES = (2, 3, 4, 5, 6, 8, 10)
RENYI_QS = (2, 3)


def _guard(out: dict, name: str, fn, *args, **kwargs) -> None:
    """Call fn; store the first finite scalar under `name`, ignore failures."""
    try:
        v = fn(*args, **kwargs)
        if np.isscalar(v) and np.isfinite(v):
            out[name] = float(v)
        elif v is not None:
            out[name] = float(np.asarray(v).ravel()[0])
    except Exception:
        pass


def entropy_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Entropy block plus the Katz/Petrosian/Hjorth/zero-crossing comparators"""
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {}
    if x.size < 500 or x.std() == 0 or not HAS_ANTROPY:
        return out
    _guard(out, f"{prefix}_sampen", antropy.sample_entropy, x, order=SAMP_ORDER)
    _guard(out, f"{prefix}_appen", antropy.app_entropy, x, order=SAMP_ORDER)
    _guard(out, f"{prefix}_permen", antropy.perm_entropy, x,
           order=PERM_ORDER, delay=1, normalize=True)
    _guard(out, f"{prefix}_svden", antropy.svd_entropy, x,
           order=SVD_ORDER, delay=1, normalize=True)
    b = (x > np.median(x)).astype(np.uint8)
    _guard(out, f"{prefix}_lzc", antropy.lziv_complexity, b, normalize=True)
    _guard(out, f"{prefix}_petrosian", antropy.petrosian_fd, x)
    _guard(out, f"{prefix}_katz", antropy.katz_fd, x)
    try:
        mob, com = antropy.hjorth_params(x)
        out[f"{prefix}_hjorth_mobility"] = float(mob)
        out[f"{prefix}_hjorth_complexity"] = float(com)
    except Exception:                                      # noqa: BLE001
        pass
    _guard(out, f"{prefix}_zerocross", antropy.num_zerocross, x)
    return out


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    n = x.size // scale
    return x[: n * scale].reshape(n, scale).mean(axis=1)


def multiscale_entropy_feats(x: np.ndarray, prefix: str,
                             scales: tuple = MSE_SCALES) -> dict[str, float]:
    """Sample entropy of the coarse-grained series at several scales."""
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {}
    if not HAS_ANTROPY or x.std() == 0:
        return out
    vals = []
    for s in scales:
        xs = _coarse_grain(x, s)
        if xs.size < 500:
            continue
        try:
            v = float(antropy.sample_entropy(xs, order=SAMP_ORDER))
            if np.isfinite(v):
                out[f"{prefix}_s{s}"] = v
                vals.append(v)
        except Exception:
            pass
    if vals:
        out[f"{prefix}_area"] = float(np.mean(vals))
    return out


def renyi_features(x: np.ndarray, prefix: str,
                   qs: tuple = RENYI_QS) -> dict[str, float]:
    """Renyi entropy of the normalised power spectrum,"""
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {}
    n = x.size
    if n < 64 or x.std() == 0:
        return out
    p = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n))) ** 2
    p = p[1:]
    s = p.sum()
    if s <= 0:
        return out
    p = p / s
    p = p[p > 0]
    for q in qs:
        try:
            if q == 1:
                v = float(-(p * np.log(p)).sum())
            else:
                v = float(np.log((p ** q).sum()) / (1.0 - q))
            if np.isfinite(v):
                out[f"{prefix}_q{q}"] = v
        except Exception:
            pass
    return out


def _checked(x: np.ndarray, n_min: int = 500) -> np.ndarray | None:
    """Return x if it is long and non-degenerate enough for an entropy metric."""
    x = np.asarray(x, dtype=np.float64)
    if not HAS_ANTROPY or x.size < n_min or x.std() == 0:
        return None
    return x


def sampen(x: np.ndarray) -> float:
    y = _checked(x)
    return float(antropy.sample_entropy(y, order=SAMP_ORDER)) \
        if y is not None else np.nan


def appen(x: np.ndarray) -> float:
    y = _checked(x)
    return float(antropy.app_entropy(y, order=SAMP_ORDER)) \
        if y is not None else np.nan


def permen(x: np.ndarray) -> float:
    y = _checked(x)
    return float(antropy.perm_entropy(y, order=PERM_ORDER, delay=1,
                                      normalize=True)) if y is not None else np.nan


def svden(x: np.ndarray) -> float:
    y = _checked(x)
    return float(antropy.svd_entropy(y, order=SVD_ORDER, delay=1,
                                     normalize=True)) if y is not None else np.nan


def lzc(x: np.ndarray) -> float:
    y = _checked(x)
    if y is None:
        return np.nan
    b = (y > np.median(y)).astype(np.uint8)
    return float(antropy.lziv_complexity(b, normalize=True))


def renyi(x: np.ndarray, q: float = 2) -> float:
    """Renyi entropy of the normalised power spectrum at one q."""
    return renyi_features(x, "e", qs=(q,)).get(f"e_q{q}", np.nan)


ENTROPY_METRICS = {"sampen": sampen, "appen": appen, "permen": permen,
                   "svden": svden, "lzc": lzc}

#6 surrogates

def _quant_step(x: np.ndarray) -> float:
    """Median difference between consecutive distinct values of `x`."""
    u = np.unique(np.asarray(x, dtype=np.float64))
    d = np.diff(u)
    d = d[d > 0]
    return float(np.median(d)) if d.size else float("nan")


def quantize(x: np.ndarray, step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0:
        return np.asarray(x, dtype=np.float64)
    return np.round(np.asarray(x, dtype=np.float64) / step) * step


def shuffle_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(np.asarray(x, dtype=np.float64))


def phase_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase randomisation: same power spectrum, random phases."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    sp = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, sp.size)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    y = np.fft.irfft(np.abs(sp) * np.exp(1j * ph), n)
    return y + x.mean()


def iaaft_surrogate(x: np.ndarray, rng: np.random.Generator,
                    n_iter: int = 50) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    amps = np.abs(np.fft.rfft(x - x.mean()))
    sorted_x = np.sort(x)
    s = rng.permutation(x)
    for _ in range(n_iter):
        sf = np.fft.rfft(s - s.mean())
        phases = np.angle(sf)
        y = np.fft.irfft(amps * np.exp(1j * phases), n)
        ranks = np.argsort(np.argsort(y))
        s = sorted_x[ranks]
    return s


def pipeline_matched_surrogate(x_raw: np.ndarray, f0: float,
                               rng: np.random.Generator) -> np.ndarray:
    """White noise quantised like the raw channel, then harmonic-removed."""
    x_raw = np.asarray(x_raw, dtype=np.float64)
    step = _quant_step(x_raw)
    sd = x_raw.std()
    noise = quantize(rng.normal(0.0, sd if sd > 0 else 1.0, x_raw.size), step)
    res, _ = remove_harmonics(noise, FS, f0, NHARM)
    return res


def mfdfa_stat(x: np.ndarray) -> dict[str, float]:
    """MFDFA descriptors of a series with a one-letter prefix, for null runs."""
    feats, _ = mfdfa(x, prefix="s")
    if not feats:
        return {}
    return {k.split("_", 1)[1]: v for k, v in feats.items()}

SURROGATE_STATS = ("dalpha", "h2", "asym", "fmax")

def summarise(sur: pd.DataFrame, cols: tuple = SURROGATE_STATS) -> pd.DataFrame:
    """Ansemble mean/std per surrogate name, plus the empirical two-sided p."""
    orig = sur[sur["surrogate"] == "original"]
    ens = sur[sur["surrogate"] != "original"]
    pvals: dict[str, float] = {}
    for c in cols:
        if c not in sur.columns or orig.empty or len(ens) < 5:
            continue
        obs = float(orig[c].iloc[0])
        v = ens[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        ge = int((v >= obs).sum())
        le = int((v <= obs).sum())
        n = int(v.size)
        pvals[c] = min(1.0, 2.0 * min((ge + 1) / (n + 1), (le + 1) / (n + 1)))

    out = []
    for name, g in sur.groupby("surrogate", sort=False):
        rec = {"surrogate": name, "n": len(g)}
        for c in cols:
            if c in g.columns:
                rec[c + "_mean"] = float(g[c].mean())
                rec[c + "_std"] = float(g[c].std())
                if c in pvals:
                    rec[c + "_p_emp"] = pvals[c]
        out.append(rec)
    return pd.DataFrame(out)

# 7 res audit

def quantization_table() -> pd.DataFrame:
    """Distinct-value count and effective step of every channel and record."""
    rows = []
    for rec in RECORDS:
        sig = load_record(rec)
        for ch in CHANNELS:
            v = np.asarray(sig[ch], dtype=np.float64)
            u = np.unique(v)
            d = np.diff(u)
            d = d[d > 0]
            step = float(np.median(d)) if d.size else np.nan
            span = float(u.max() - u.min())
            rows.append({
                "stem": rec.stem, "record": rec.idx, "motor": rec.motor,
                "condition": rec.condition, "channel": ch, "unit": UNITS[ch],
                "n_unique": int(u.size), "step": step, "span": span,
                "n_levels_in_span": span / step if step > 0 else np.nan,
                "std": float(v.std()),
                "step_to_std": step / float(v.std()) if v.std() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def zoh_demo(stem: str = "09_faulty_operation_08Nm-load",
             channel: str = "I1", t_start: float = 12.0,
             duration: float = 2.0, q_ref: float = 2.0) -> dict:
    """MFDFA fluctuation curves of a segment with and without the 5x hold."""
    _require_fathon()
    rec = BY_STEM[stem]
    n_raw = int(round(duration * FS_NOMINAL))
    n_skip = int(round(t_start * FS_NOMINAL))
    df = pd.read_csv(DATA_DIR / f"{rec.stem}.csv", dtype=np.float64,
                     engine="c", skiprows=range(1, n_skip + 1), nrows=n_raw)
    raw = df[channel].to_numpy()
    zoh = detect_zoh(raw)
    dec = raw[::zoh]

    win_raw = np.unique(np.logspace(np.log10(4), np.log10(5000), 30).astype(np.int64))
    win_dec = np.unique(np.logspace(np.log10(4), np.log10(1000), 30).astype(np.int64))
    q = np.array([q_ref], dtype=np.float64)

    out = {"stem": stem, "channel": channel, "zoh": zoh}
    for tag, x, win in (("raw", raw, win_raw), ("dec", dec, win_dec)):
        m = fathon.MFDFA(to_aggregated(x))
        _, Fq = m.computeFlucVec(win.astype(np.int64), q, polOrd=2, revSeg=False)
        out[tag] = {"win": win, "F": np.asarray(Fq)[0]}
    return out

# 8 window feature extraction

# signal pairs used for multifractal cross-correlation
DCCA_PAIRS = (("I1", "I2"), ("x", "I1"), ("V1", "I1"))

MFDFA_RAW_CHANNELS = ("I1", "V1", "x")

ENTROPY_CHANNELS = ("I1", "I2", "I3", "V1", "x", "Z")
MSE_CHANNELS = ("I1", "x")

META_COLS = ["stem", "record", "motor", "condition", "file_phase_loss",
             "load_nm", "regime", "win_index", "i0", "i1", "t0", "t1"]


def features_for_window(sig: dict[str, np.ndarray], i0: int, i1: int) -> dict:
    """Every fractal, entropy and single-scale feature of one window."""
    ctx = window_context(sig, i0, i1)
    f0 = ctx["f0"]
    row: dict = dict(ctx)

    residual: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        raw = np.ascontiguousarray(sig[ch][i0:i1])
        res, coeffs = remove_harmonics(raw, FS, f0, NHARM)
        residual[ch] = res
        row["harm1_" + ch] = fundamental_amplitude(coeffs)

        row.update(classic_features(raw, f"raw_{ch}"))
        row.update(classic_features(res, f"res_{ch}"))

        if ch in MFDFA_RAW_CHANNELS:
            fr, _ = mfdfa(raw, prefix=f"mfdfa_raw_{ch}")
            row.update(fr)
        fs_, _ = mfdfa(res, prefix=f"mfdfa_res_{ch}")
        row.update(fs_)

    for a, b in DCCA_PAIRS:
        fd, _ = mfdcca(residual[a], residual[b], prefix=f"dcca_res_{a}_{b}")
        row.update(fd)

    for ch in ENTROPY_CHANNELS:
        raw = np.ascontiguousarray(sig[ch][i0:i1])
        row.update(entropy_features(raw, prefix=f"ent_raw_{ch}"))
        row.update(entropy_features(residual[ch], prefix=f"ent_res_{ch}"))
        row.update(renyi_features(raw, prefix=f"renyi_raw_{ch}"))
        row.update(renyi_features(residual[ch], prefix=f"renyi_res_{ch}"))
    for ch in MSE_CHANNELS:
        row.update(multiscale_entropy_feats(residual[ch], prefix=f"mse_res_{ch}"))

    return row


def record_rows(rec: Record, verbose: bool = False) -> list[dict]:
    """Extract every analysis window of one record.  Runs in a worker process."""
    sig = load_record(rec)
    wins = analysis_windows(rec, sig)
    rows = []
    for wi, (seg, i0, i1) in enumerate(wins):
        t0 = time.time()
        row = features_for_window(sig, i0, i1)
        row.update({
            "stem": rec.stem, "record": rec.idx, "motor": rec.motor,
            "condition": rec.condition, "file_phase_loss": rec.phase_loss,
            "load_nm": rec.load_nm, "regime": seg.regime, "win_index": wi,
            "i0": i0, "i1": i1, "t0": i0 / FS, "t1": i1 / FS,
            "calc_s": time.time() - t0,
        })
        rows.append(row)
        if verbose:
            print(f"   {rec.stem} win {wi:3d} t={i0 / FS:5.1f}s "
                  f"{seg.regime:11s} f0={row['f0']:.3f}", flush=True)
    return rows


def extract_features(only: list[str] | None = None, jobs: int | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Feature table of all (or `only`) records, one row per analysis window."""
    recs = [r for r in RECORDS if not only or r.stem in only]
    if jobs is None:
        jobs = max(1, min(8, (os.cpu_count() or 2) - 2))

    t0 = time.time()
    rows: list[dict] = []

    def _serial():
        for rec in recs:
            if verbose:
                print(f"[{rec.stem}]", flush=True)
            rows.extend(record_rows(rec, verbose))

    if jobs == 1 or len(recs) == 1:
        _serial()
    else:
        try:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(record_rows, rec, False): rec for rec in recs}
                for fut in as_completed(futs):
                    rec = futs[fut]
                    try:
                        got = fut.result()
                    except Exception as exc:                   # noqa: BLE001
                        print(f"!! {rec.stem} FAILED: {exc!r}", file=sys.stderr)
                        continue
                    if verbose:
                        print(f"[{rec.stem}] {len(got)} windows "
                              f"({time.time() - t0:.0f}s elapsed)", flush=True)
                    rows.extend(got)
        except (OSError, PermissionError) as exc:
            print(f"process pool unavailable ({exc.__class__.__name__}); "
                  f"falling back to serial", file=sys.stderr)
            rows.clear()
            _serial()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["record", "win_index"]).reset_index(drop=True)
    front = [c for c in META_COLS if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    if verbose:
        print("\ntotal windows=" + str(len(df)) + " in "
              + str(round(time.time() - t0)) + "s", flush=True)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract fractal/entropy features for every analysis window")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these record stems")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes")
    ap.add_argument("--out", default=str(TABLES_DIR / "window_features.csv"))
    args = ap.parse_args(argv)

    df = extract_features(only=args.only, jobs=args.jobs)
    if df.empty:
        print("no windows extracted", file=sys.stderr)
        return 1
    df.to_csv(args.out, index=False)
    print(f"WROTE {args.out}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
