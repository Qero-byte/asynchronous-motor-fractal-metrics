#!/usr/bin/env python
"""Verification for review: self-tests, paper numbers, and the label audit.

    python verify_results.py --selftest   # mathematics, no data and no results
    python verify_results.py              # every number quoted in the paper
    python verify_results.py --labels     # regenerate + check results/tables/label_audit.csv
    python verify_results.py --all        # all three

Exit status is 0 only when every check passes.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fractal_metrics import (FS, NHARM, TABLES_DIR, _quant_step, detect_zoh,
                             estimate_f0, higuchi_fd, iaaft_surrogate,
                             log_window_sizes, mfdfa, phase_surrogate, quantize,
                             remove_harmonics, sampen, shuffle_surrogate,
                             summarise, spectral_entropy)
from group_statistics import COMPARISONS, feature_columns, select

W_PATH = TABLES_DIR / "window_features.csv"
C_PATH = TABLES_DIR / "comparisons.csv"
K_PATH = TABLES_DIR / "classification.csv"
F_PATH = TABLES_DIR / "record_variance_family.csv"
S_PATH = TABLES_DIR / "surrogates.csv"


class Checks:
    def __init__(self) -> None:
        self.ok = 0
        self.bad = 0

    def check(self, label: str, got, want, tol: float = 5e-4) -> bool:
        good = abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
        print(f"{'OK  ' if good else 'FAIL'} {label}: got={float(got):.4g} "
              f"want={want}")
        self.ok, self.bad = self.ok + good, self.bad + (not good)
        return good

    def true(self, label: str, cond: bool, detail: str = "") -> bool:
        print(f"{'OK  ' if cond else 'FAIL'} {label}"
              + (f": {detail}" if detail else ""))
        self.ok, self.bad = self.ok + bool(cond), self.bad + (not cond)
        return bool(cond)

    def report(self) -> int:
        print()
        print(f"{self.ok} checks passed, {self.bad} failed")
        return 1 if self.bad else 0

# 1 self-tests -- synthetic data
def selftest() -> int:
    c = Checks()
    rng = np.random.default_rng(0)
    print("== mathematics and implementation ==")

    #de-holding
    base = rng.normal(size=20_000)
    c.check("detect_zoh on a 5x-held signal", detect_zoh(np.repeat(base, 5)), 5)
    c.check("detect_zoh on an unheld signal", detect_zoh(base), 1)

    #MFDFA window sizes must be int64 (fathon on Windows)
    win = log_window_sizes(20_000)
    c.true("log_window_sizes dtype is int64", win.dtype == np.int64, str(win.dtype))
    c.true("log_window_sizes are unique and increasing",
           bool(np.all(np.diff(win) > 0)), f"n={win.size}, {win[0]}..{win[-1]}")

    #supply frequency and carrier removal
    t = np.arange(20_000) / FS
    clean = 3.0 * np.sin(2 * np.pi * 50.0 * t) + 0.4 * np.cos(2 * np.pi * 150.0 * t)
    c.check("estimate_f0 on a 50 Hz carrier", estimate_f0(clean, FS), 50.0, tol=1e-4)
    res, coeffs = remove_harmonics(clean, FS, 50.0, NHARM)
    c.check("harmonic removal recovers the fundamental amplitude",
            np.hypot(coeffs[0], coeffs[1]), 3.0, tol=1e-3)
    c.true("harmonic removal suppresses the carrier",
           float(res.std()) < 0.02, f"residual std={float(res.std()):.2e}")

    #MFDFA of white noise must give h(2) ~ 0.5
    feats, curves = mfdfa(rng.normal(size=20_000), prefix="t")
    c.check("MFDFA h(2) of white noise", feats["t_h2"], 0.5, tol=0.06)
    c.true("MFDFA fit quality is reported and high",
           float(feats["t_r2_min"]) > 0.95, f"r2_min={feats['t_r2_min']:.4f}")

    #the tau = q h - 1 identities
    q = curves["q"]
    tau_from_h = q * curves["h"] - 1.0
    c.check("tau == q h - 1 (max abs deviation)",
            float(np.max(np.abs(tau_from_h - curves["tau"]))), 0.0, tol=1e-9)
    c1, c2 = 0.8, -0.15
    h_fn = lambda qq: c1 + c2 * (qq - 1.0)
    tau_fn = lambda qq: qq * h_fn(qq) - 1.0
    eps = 1e-5
    d1 = (tau_fn(1 + eps) - tau_fn(1 - eps)) / (2 * eps)
    c.check("D1 = h(1) + h'(1), not h(1)", d1, c1 + c2, tol=1e-4)
    c.true("the synthetic case separates the two formulas",
           abs(d1 - h_fn(1.0)) > 0.1, f"D1={d1:.4f}, h(1)={h_fn(1.0):.4f}")

    #single-scale comparators
    line = np.arange(5000.0)
    c.check("Higuchi FD of a straight line", higuchi_fd(line), 1.0, tol=0.02)
    c.check("Higuchi FD of white noise", higuchi_fd(rng.normal(size=5000)), 2.0,
            tol=0.1)
    tone = np.sin(2 * np.pi * 50 * np.arange(5000) / FS)
    c.true("spectral entropy: tone << noise",
           spectral_entropy(tone) < 0.3 < spectral_entropy(rng.normal(size=5000)))

    #entropy
    c.true("SampEn of noise is finite and positive",
           np.isfinite(sampen(rng.normal(size=2000))) and sampen(rng.normal(size=2000)) > 0)

    #surrogate generators
    x = rng.normal(size=4096).cumsum()          # a correlated series
    sh = shuffle_surrogate(x, rng)
    c.true("shuffle preserves the value distribution",
           np.allclose(np.sort(sh), np.sort(x)))
    ph = phase_surrogate(x, rng)
    aa = np.abs(np.fft.rfft(x - x.mean()))
    bb = np.abs(np.fft.rfft(ph - ph.mean()))
    c.check("phase surrogate preserves the power spectrum (max rel. error)",
            float(np.max(np.abs(aa - bb)) / aa.mean()), 0.0, tol=1e-9)
    ia = iaaft_surrogate(x, rng, n_iter=100)
    c.true("IAAFT preserves the amplitude distribution exactly",
           np.allclose(np.sort(ia), np.sort(x)))
    amp_err = float(np.max(np.abs(np.abs(np.fft.rfft(ia - ia.mean())) - aa))
                    / aa.mean())
    c.true("IAAFT approximately preserves the spectrum",
           amp_err < 0.2, f"max relative amplitude error={amp_err:.3f}")

    #quantisation and the pipeline-matched null
    grid = np.round(np.linspace(-1.0, 1.0, 20_001), 2)     # dense 0.01 grid
    c.check("quantisation step recovered", _quant_step(grid), 0.01, tol=1e-6)
    c.true("a quantised signal has few distinct values",
           np.unique(grid).size == 201, f"{np.unique(grid).size} levels")
    qz = quantize(np.array([0.24, 0.26, -0.31]), 0.1)
    c.true("quantize snaps values onto the grid",
           bool(np.allclose(qz, [0.2, 0.3, -0.3])))

    #ensemble summarising and pooling
    sur = pd.DataFrame([{"surrogate": "original", "dalpha": 1.0}]
                       + [{"surrogate": "shuffle", "dalpha": v}
                          for v in np.linspace(0.0, 0.9, 100)])
    s = summarise(sur, cols=("dalpha",))
    c.true("summarise reports the observed value and a finite empirical p",
           abs(float(s.loc[s.surrogate == "original", "dalpha_mean"].iloc[0]) - 1.0) < 1e-12
           and float(s.loc[s.surrogate == "shuffle", "dalpha_p_emp"].iloc[0]) <= 0.03)

    def _w(mean, sd):
        row = {"group": "g", "stem": "s", "channel": "x", "kind": "shuffle",
               "n_windows": 1, "n": 10, "dalpha_mean": mean,
               "dalpha_std": sd}
        # pool_shards expects every summarised statistic to be present
        for s in ("h2", "asym", "fmax"):
            row[f"{s}_mean"], row[f"{s}_std"] = np.nan, np.nan
        return row

    per_window = pd.DataFrame([
        {**_w(1.0, np.nan), "surrogate": "original"},
        {**_w(0.8, 0.1), "surrogate": "shuffle"},
        {**_w(1.2, np.nan), "surrogate": "original"},
        {**_w(1.0, 0.2), "surrogate": "shuffle"},
    ])
    from group_statistics import pool_shards
    pooled = pool_shards(per_window)

    c.check("pool_shards pooled mean (mixture of Gaussians)",
            float(pooled["dalpha_mean"].iloc[0]), 0.9, tol=1e-9)
    c.check("pool_shards pooled std keeps the between-window spread",
            float(pooled["dalpha_std"].iloc[0]), 0.187083, tol=1e-6)
    c.check("pool_shards pooled observation",
            float(pooled["dalpha_obs"].iloc[0]), 1.1, tol=1e-9)

    return c.report()

# 2 the numbers quoted in the paper

def numbers() -> int:
    c = Checks()
    missing = [p for p in (W_PATH, C_PATH, K_PATH, F_PATH, S_PATH) if not p.exists()]
    if missing:
        print("MISSING result tables (run 'python run_all.py' first):")
        for p in missing:
            print(f"  {p}")
        return 1

    print("== numbers quoted in the paper ==")
    W = pd.read_csv(W_PATH)
    K = pd.read_csv(K_PATH)
    F = pd.read_csv(F_PATH, index_col=0)
    S = pd.read_csv(S_PATH)

    bal = W[W.regime == "balanced"]
    opn = W[W.regime == "open_phase"]
    c.check("mfdfa_res_I2_h2 balanced", bal.mfdfa_res_I2_h2.median(), 0.773301)
    c.check("mfdfa_res_I2_h2 open", opn.mfdfa_res_I2_h2.median(), 0.434087)
    c.check("raw_I2_higuchi balanced", bal.raw_I2_higuchi.median(), 1.056612)
    c.check("raw_I2_higuchi open", opn.raw_I2_higuchi.median(), 1.965751)
    c.check("raw_I2_std balanced", bal.raw_I2_std.median(), 0.487196)
    c.check("raw_I2_std open", opn.raw_I2_std.median(), 0.007199)

    h = W[(W.motor == "healthy") & (W.regime == "balanced")]
    f = W[(W.motor == "faulty") & (W.regime == "balanced")]
    c.check("raw_x_rms healthy", h.raw_x_rms.median(), 0.018941)
    c.check("raw_x_rms faulty", f.raw_x_rms.median(), 0.030102)

    sh = W[(W.motor == "healthy") & (W.regime == "balanced")]
    so = W[(W.motor == "healthy") & (W.regime == "open_phase")]
    c.check("healthy V1_std balanced", sh.raw_V1_std.median(), 271.274979)
    c.check("healthy V1_std open", so.raw_V1_std.median(), 217.734594)
    c.check("healthy V2_rms balanced", sh.raw_V2_rms.median(), 274.994431)
    c.check("healthy V2_rms open", so.raw_V2_rms.median(), 159.962925)

    def cls(cmp_key, kind, grouped=True):
        r = K[(K.comparison == cmp_key) & (K.feature_kind == kind)
              & (K.grouped == grouped)]
        return float(r.balanced_accuracy.iloc[0])

    c.check("CV supply_healthy classical", cls("supply_healthy", "classical"), 0.6320)
    c.check("CV supply_healthy fractal", cls("supply_healthy", "fractal"), 0.9360)
    c.check("CV supply_faulty classical", cls("supply_faulty", "classical"), 0.7812)
    c.check("CV supply_faulty fractal", cls("supply_faulty", "fractal"), 0.9973)
    c.check("CV bearing classical", cls("bearing", "classical"), 1.0000)
    c.check("CV bearing fractal", cls("bearing", "fractal"), 0.8972)

    c.check("family MFDFA raw mean", F.loc["MFDFA (raw)", "eta2_mean"], 0.6412,
            tol=3e-3)
    c.check("family single-scale raw mean",
            F.loc["single-scale (raw)", "eta2_mean"], 0.6118, tol=3e-3)
    c.check("family MFDFA res mean", F.loc["MFDFA (residual)", "eta2_mean"],
            0.2662, tol=3e-3)
    c.check("family MFDCCA mean", F.loc["MFDCCA (residual)", "eta2_mean"], 0.1983,
            tol=3e-3)
    c.check("family MFDCCA frac>=0.8", F.loc["MFDCCA (residual)", "frac_ge_0.8"], 0.0)

    #composition of the descriptor sets
    n_amp = len(feature_columns(W, "amplitude"))
    n_class = len(feature_columns(W, "classical"))
    n_frac = len(feature_columns(W, "fractal"))
    n_ent = len(feature_columns(W, "entropy"))
    c.check("amplitude subset size", n_amp, 54.0)
    c.check("single-scale set size", n_class, 126.0)
    c.check("multifractal set size", n_frac, 165.0)
    c.check("entropy set size", n_ent, 160.0)
    c.true("the single-scale family is not amplitude-only",
           any(x.endswith("_higuchi") for x in feature_columns(W, "classical"))
           and any(x.endswith("_specent") for x in feature_columns(W, "classical")))

    #the two single-phasing situations are different conditions
    c.check("CV running amplitude", cls("supply_running", "amplitude"), 0.6825)
    c.check("CV running single-scale", cls("supply_running", "classical"), 0.9760)
    c.check("CV running multifractal", cls("supply_running", "fractal"), 1.0000)
    c.check("CV running vs startup single-scale",
            cls("open_running_vs_startup", "classical"), 0.1938)
    c.check("CV running vs startup multifractal",
            cls("open_running_vs_startup", "fractal"), 0.3470)
    c.check("CV amplitude supply_healthy", cls("supply_healthy", "amplitude"), 0.6800)
    c.check("CV amplitude supply_faulty", cls("supply_faulty", "amplitude"), 0.7310)
    c.check("CV amplitude bearing", cls("bearing", "amplitude"), 0.9060)

    #surrogate table
    for ch, kind, grp, want_o, want_s in [
            ("I1", "shuffle", "healthy, balanced", 0.1403, 0.0389),
            ("I1", "pipeline", "faulty, open phase", 0.1175, 0.0368),
            ("x", "shuffle", "faulty, balanced", 0.0212, 0.0383),
            ("x", "pipeline", "healthy, open phase", 0.0382, 0.0370)]:
        d = S[(S.channel == ch) & (S.kind == kind) & (S.group == grp)]
        o = float(d[d.surrogate == "original"].dalpha_mean.iloc[0])
        ss = float(d[d.surrogate == kind].dalpha_mean.iloc[0])
        c.check(f"surr {ch}/{kind}/{grp} original", o, want_o)
        c.check(f"surr {ch}/{kind}/{grp} {kind}", ss, want_s)

    def _z(ch, kind, grp):
        d = S[(S.channel == ch) & (S.kind == kind) & (S.group == grp)]
        o = float(d[d.surrogate == "original"].dalpha_mean.iloc[0])
        v = d[d.surrogate == kind]
        return (o - float(v.dalpha_mean.iloc[0])) / float(v.dalpha_std.iloc[0])

    c.check("z I1 shuffle healthy balanced",
            _z("I1", "shuffle", "healthy, balanced"), 4.4, tol=0.06)
    c.check("z I1 phase healthy balanced",
            _z("I1", "phase", "healthy, balanced"), 0.2, tol=0.15)
    c.check("z I1 pipeline healthy balanced",
            _z("I1", "pipeline", "healthy, balanced"), 4.9, tol=0.06)
    c.check("z I1 phase faulty open", _z("I1", "phase", "faulty, open phase"),
            -2.9, tol=0.06)
    c.check("z I1 pipeline faulty open",
            _z("I1", "pipeline", "faulty, open phase"), 4.0, tol=0.06)
    c.check("z x shuffle faulty balanced",
            _z("x", "shuffle", "faulty, balanced"), -0.8, tol=0.06)
    c.check("z x pipeline faulty balanced",
            _z("x", "pipeline", "faulty, balanced"), -0.8, tol=0.06)
    c.check("z x pipeline healthy open",
            _z("x", "pipeline", "healthy, open phase"), 0.1, tol=0.2)
    c.check("z x pipeline faulty open",
            _z("x", "pipeline", "faulty, open phase"), 3.4, tol=0.06)
    c.check("z x phase faulty open", _z("x", "phase", "faulty, open phase"), 2.2,
            tol=0.06)

    h2 = S[(S.surrogate == "original")]
    c.check("h2 I1 healthy balanced",
            h2[(h2.channel == "I1") & (h2.group == "healthy, balanced")].h2_mean.iloc[0],
            0.8052)
    c.check("h2 I1 faulty open",
            h2[(h2.channel == "I1") & (h2.group == "faulty, open phase")].h2_mean.iloc[0],
            0.9610)
    c.check("h2 x healthy balanced",
            h2[(h2.channel == "x") & (h2.group == "healthy, balanced")].h2_mean.iloc[0],
            0.4758)
    c.check("h2 x faulty open",
            h2[(h2.channel == "x") & (h2.group == "faulty, open phase")].h2_mean.iloc[0],
            0.4526)

    #entropy
    hbal = W[(W.motor == "healthy") & (W.regime == "balanced")]
    hopen = W[(W.motor == "healthy") & (W.regime == "open_phase")]
    fbal = W[(W.motor == "faulty") & (W.regime == "balanced")]
    fopen = W[(W.motor == "faulty") & (W.regime == "open_phase")]
    c.check("ent_raw_I2_sampen healthy bal", hbal.ent_raw_I2_sampen.median(), 0.0912)
    c.check("ent_raw_I2_sampen healthy open", hopen.ent_raw_I2_sampen.median(), 1.9992)
    c.check("renyi_raw_I2_q2 faulty bal", fbal.renyi_raw_I2_q2.median(), 0.6962)
    c.check("renyi_raw_I2_q2 faulty open", fopen.renyi_raw_I2_q2.median(), 4.1100)
    c.check("ent_raw_I2_zerocross pooled bal", bal.ent_raw_I2_zerocross.median(), 200.0)
    c.check("ent_raw_I2_zerocross pooled open", opn.ent_raw_I2_zerocross.median(),
            2412.0)
    c.check("CV supply_healthy entropy", cls("supply_healthy", "entropy"), 0.9480)
    c.check("CV supply_faulty entropy", cls("supply_faulty", "entropy"), 1.0000)
    c.check("CV supply_all entropy", cls("supply_all", "entropy"), 1.0000)
    c.check("family entropy+complexity raw mean",
            F.loc["entropy+complexity (raw)", "eta2_mean"], 0.616, tol=3e-3)
    c.check("family entropy+complexity res mean",
            F.loc["entropy+complexity (residual)", "eta2_mean"], 0.395, tol=3e-3)
    c.check("family multiscale entropy res mean",
            F.loc["multiscale entropy (residual)", "eta2_mean"], 0.239, tol=3e-3)

    ES = pd.read_csv(TABLES_DIR / "entropy_surrogates.csv")
    z_i1 = ES[(ES.channel == "I1") & (ES.metric == "sampen")]["z_pipeline"]
    z_x = ES[(ES.channel == "x") & (ES.metric == "sampen")]["z_pipeline"]
    c.check("z_pipeline I1 sampen (mean abs)", abs(z_i1).mean(), 183.1, tol=0.02)
    c.check("z_pipeline x sampen (min abs)", abs(z_x).min(), 7.6, tol=0.15)

    #hierarchical (record-level) inference
    RL = pd.read_csv(TABLES_DIR / "record_level.csv")
    WR = pd.read_csv(TABLES_DIR / "within_record.csv")
    mb = RL[RL.comparison == "bearing"]
    c.check("record-level bearing min p", mb.min_p.min(), 0.0704, tol=2e-3)
    c.check("record-level bearing records per class", mb.n_records_a.iloc[0], 4.0)
    c.check("record-level supply_healthy min p (one paired record)",
            RL[RL.comparison == "supply_healthy"].min_p.min(), 1.0)
    c.check("record-level supply_faulty min p (one paired record)",
            RL[RL.comparison == "supply_faulty"].min_p.min(), 1.0)
    c.check("record-level supply_all min p (two paired records)",
            RL[RL.comparison == "supply_all"].min_p.min(), 0.6, tol=0.02)
    c.check("within-record shift floor (1/29)", WR.min_p_shift.min(), 1.0 / 29.0,
            tol=2e-3)
    c.check("within-record windows per record",
            WR[WR.stem.str.startswith("02")].n_windows.iloc[0], 29.0)
    hbal2 = W[(W.motor == "healthy") & (W.regime == "balanced")]
    hopen2 = W[(W.motor == "healthy") & (W.regime == "open_phase")]
    c.check("healthy open-phase windows from record 02",
            float((hopen2.stem.str.startswith("02")).sum()), 16.0)
    c.check("healthy balanced windows from record 02",
            float((hbal2.stem.str.startswith("02")).sum()), 13.0)

    #label audit
    LA_PATH = TABLES_DIR / "label_audit.csv"
    if LA_PATH.exists():
        LA = pd.read_csv(LA_PATH).set_index("check")["value"]
        c.check("audit: class order fixed by the comparison",
                float(LA["encoding_A_is_0_B_is_1"]), 1.0)
        c.check("audit: BA invariant to a label swap",
                float(LA["max_symmetry_error"]), 0.0, tol=1e-12)
        c.check("audit: random-label control (single-scale)",
                float(LA["control_BA_mean_classical"]), 0.533, tol=0.02)
        c.check("audit: random-label control (all features)",
                float(LA["control_BA_mean_all"]), 0.478, tol=0.02)
        c.check("audit: single-scale grouped BA below chance",
                float(LA["grouped_BA_classical"]), 0.194, tol=0.02)
        c.check("audit: complete univariate separations",
                float(LA["univariate_complete_separation_count"]), 21.0)
        c.check("audit: lead feature median A", float(LA["lead_median_A"]),
                0.0218612, tol=1e-3)
        c.check("audit: lead feature median B", float(LA["lead_median_B"]),
                0.000842673, tol=1e-3)
    else:
        print("SKIP label audit (results/tables/label_audit.csv absent); "
              "run 'python verify_results.py --labels'")

    return c.report()

# 3 label-encoding and calibration audit
def run_label_audit() -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(0)
    contrast = "open_running_vs_startup"
    kinds = ("amplitude", "classical", "fractal", "entropy", "all")

    df = pd.read_csv(W_PATH)
    cmp = next(c for c in COMPARISONS if c.key == contrast)
    data = pd.concat([select(df, cmp.a, cmp.filt).assign(_y=0),
                      select(df, cmp.b, cmp.filt).assign(_y=1)],
                     ignore_index=True)
    data = data.copy()          # de-fragment before the many column lookups
    y = data["_y"].to_numpy()
    groups = data["stem"].to_numpy()

    def folds(cols, n_splits=5):
        X = data[cols].to_numpy(dtype=float)
        n_splits = int(min(n_splits, data["stem"].nunique()))
        return X, list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    def run(cols, yy, seeds=5, confused=False):
        X, it = folds(cols)
        bas, aucs, sym, conf = [], [], [], None
        for seed in range(seeds):
            yt, yp, ypr = [], [], []
            for tr, te in it:
                if len(np.unique(yy[tr])) < 2:
                    continue
                clf = RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=2,
                    class_weight="balanced", random_state=seed,
                    n_jobs=1).fit(X[tr], yy[tr])
                yt.append(yy[te])
                yp.append(clf.predict(X[te]))
                ypr.append(clf.predict_proba(X[te])[:, 1])
            yt, yp, ypr = (np.concatenate(a) for a in (yt, yp, ypr))
            bas.append(balanced_accuracy_score(yt, yp))
            aucs.append(roc_auc_score(yt, ypr))
            # exact symmetry test
            sym.append(balanced_accuracy_score(1 - yt, 1 - yp) - bas[-1])
            if confused:
                conf = (int(((yt == 0) & (yp == 0)).sum()),
                        int(((yt == 1) & (yp == 1)).sum()), len(yt))
        return (float(np.mean(bas)), float(np.mean(aucs)),
                float(np.max(np.abs(sym))), conf)

    def control(cols, n_draws=10, seeds=5):
        bas, aucs = [], []
        for _ in range(n_draws):
            y_rand = (rng.random(len(data)) < 0.5).astype(int)
            ba, auc, _, _ = run(cols, y_rand, seeds=seeds)
            bas.append(ba)
            aucs.append(auc)
        return float(np.mean(bas)), float(np.std(bas)), float(np.mean(aucs))

    rows = []
    enc_ok = bool((data["_y"].to_numpy()[:1] == 0).all()
                  and set(data.loc[data._y == 0, "stem"])
                  == set(select(df, cmp.a, cmp.filt)["stem"]))
    rows.append(("encoding_A_is_0_B_is_1", float(enc_ok)))
    print(f"1. label encoding fixed by the comparison definition: {enc_ok}")
    sym_max = 0.0
    for kind in kinds:
        cols = sorted(c for c in feature_columns(df, kind)
                      if data[c].notna().all() and data[c].std() > 0)
        ba, auc, sym, conf = run(cols, y, confused=True)
        sym_max = max(sym_max, sym)
        rows += [(f"grouped_BA_{kind}", ba), (f"grouped_AUC_{kind}", auc)]
        if kind == "classical":
            rows += [("classical_true_A_pred_A", float(conf[0])),
                     ("classical_true_B_pred_B", float(conf[1])),
                     ("classical_n_test", float(conf[2]))]
        print(f"   {kind:10s} BA={ba:.3f} AUC={auc:.3f}")
    rows.append(("max_symmetry_error", sym_max))
    print(f"2. max |BA(swapped labels/preds) - BA| = {sym_max:.2e} (must be 0)")
    for kind in ("classical", "all"):
        cols = sorted(c for c in feature_columns(df, kind)
                      if data[c].notna().all() and data[c].std() > 0)
        m, sd, auc = control(cols)
        rows += [(f"control_BA_mean_{kind}", m), (f"control_BA_std_{kind}", sd),
                 (f"control_AUC_mean_{kind}", auc)]
        print(f"3. random-label control, {kind:10s} BA={m:.3f} +- {sd:.3f} "
              f"AUC={auc:.3f} (chance = 0.5)")
    sub = pd.read_csv(C_PATH)
    sub = sub[sub.comparison == contrast]
    rows.append(("univariate_complete_separation_count", float((sub.auc == 0).sum())))
    lead = sub.nsmallest(1, "auc").iloc[0]
    rows += [("lead_median_A", float(lead.median_a)),
             ("lead_median_B", float(lead.median_b)),
             ("n_features", float(len(sub)))]
    print(f"4. complete univariate separation (AUC=0): {int((sub.auc == 0).sum())} "
          f"of {len(sub)} features; leader {lead.feature!r} medians "
          f"{lead.median_a:.6g} (A) vs {lead.median_b:.6g} (B)")

    out = pd.DataFrame(rows, columns=["check", "value"])
    out.to_csv(TABLES_DIR / "label_audit.csv", index=False)
    print()
    print(f"wrote {TABLES_DIR / 'label_audit.csv'} ({len(out)} rows)")
    return 0

# entry poin

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify the self-tests, the paper numbers and the label audit")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic mathematics tests (no data, no tables)")
    ap.add_argument("--numbers", action="store_true",
                    help="re-check the values quoted in the manuscript")
    ap.add_argument("--labels", action="store_true",
                    help="regenerate and check results/tables/label_audit.csv")
    ap.add_argument("--all", action="store_true", help="all of the above")
    args = ap.parse_args(argv)

    if not (args.selftest or args.numbers or args.labels or args.all):
        args.numbers = True

    rc = 0
    if args.selftest or args.all:
        rc |= selftest()
        print()
    if args.labels or args.all:
        rc |= run_label_audit()
        print()
    if args.numbers or args.all:
        rc |= numbers()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
