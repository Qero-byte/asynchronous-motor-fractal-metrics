#!/usr/bin/env python
"""Statistics and validation on top of the window feature table.

* **bearing** -- healthy motor vs faulty motor (outer-race bearing defect),
* **supply**  -- balanced vs open-phase (phase B disconnected).
    python group_statistics.py                  # all tables
    python group_statistics.py --skip-surrogates --skip-classify
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from fractal_metrics import (BY_STEM, CHANNELS, ENTROPY_METRICS, FS, NHARM,
                             SURROGATE_STATS, TABLES_DIR, iaaft_surrogate,
                             load_record, mfdfa_stat, phase_surrogate,
                             pipeline_matched_surrogate, remove_harmonics,
                             renyi, summarise)

RANDOM_STATE = 0

#representative condition groups used by the figures and the surrogate runs
CONDITION_GROUPS = (
    ("healthy, balanced", {"motor": "healthy", "regime": "balanced",
                           "load_nm": 0.4}, "tab:green"),
    ("faulty, balanced", {"motor": "faulty", "regime": "balanced",
                          "load_nm": 0.4}, "tab:red"),
    ("healthy, open phase", {"motor": "healthy",
                             "regime": "open_phase"}, "tab:olive"),
    ("faulty, open phase", {"motor": "faulty",
                            "regime": "open_phase"}, "tab:purple"),
)

CLASSICAL_PREFIXES = ("raw_", "res_")
FRACTAL_PREFIXES = ("mfdfa_", "dcca_")
ENTROPY_PREFIXES = ("ent_", "mse_", "renyi_")
AMPLITUDE_SUFFIXES = ("_rms", "_std", "_crest")

META = set(["stem", "record", "motor", "condition", "file_phase_loss",
            "load_nm", "regime", "win_index", "i0", "i1", "t0", "t1",
            "calc_s", "f0", "supply_v", "i1_rms", "i2_rms", "i3_rms",
            "acc_rms"] + [f"harm1_{c}" for c in CHANNELS])

def select(df: pd.DataFrame, spec: dict, filt: dict | None = None) -> pd.DataFrame:
    """Rows matching `spec` (and `filt`); a value may be a predicate."""
    m = pd.Series(True, index=df.index)
    for k, v in {**(filt or {}), **spec}.items():
        m &= v(df[k]) if callable(v) else df[k].eq(v)
    return df[m]


def feature_columns(df: pd.DataFrame, kind: str = "all") -> list[str]:
    cols = [c for c in df.columns
            if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if kind == "classical":
        return [c for c in cols if c.startswith(CLASSICAL_PREFIXES)]
    if kind == "amplitude":
        return [c for c in cols if c.endswith(AMPLITUDE_SUFFIXES)]
    if kind == "fractal":
        return [c for c in cols if c.startswith(FRACTAL_PREFIXES)]
    if kind == "entropy":
        return [c for c in cols if c.startswith(ENTROPY_PREFIXES)]
    return cols


def feature_family(feature: str) -> str:
    if feature.startswith("dcca_"):
        return "MFDCCA (residual)"
    if feature.startswith("mfdfa_res_"):
        return "MFDFA (residual)"
    if feature.startswith("mfdfa_raw_"):
        return "MFDFA (raw)"
    if feature.startswith("mse_"):
        return "multiscale entropy (residual)"
    if feature.startswith("renyi_res_"):
        return "Renyi spectrum (residual)"
    if feature.startswith("renyi_raw_"):
        return "Renyi spectrum (raw)"
    if feature.startswith("ent_res_"):
        return "entropy+complexity (residual)"
    if feature.startswith("ent_raw_"):
        return "entropy+complexity (raw)"
    if feature.startswith("res_"):
        return "single-scale (residual)"
    if feature.startswith("raw_"):
        return "single-scale (raw)"
    return "other"
    
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (step-up, monotone)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    ok = np.isfinite(p)
    q = np.full(n, np.nan)
    if not ok.any():
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * n / np.arange(1, ranked.size + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def auc_score(a: np.ndarray, b: np.ndarray) -> float:
    """AUC of group B scored above group A, via the Mann-Whitney U statistic."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(u / (a.size * b.size))

@dataclass
class Comparison:
    key: str
    label: str
    a: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    filt: dict = field(default_factory=dict)


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("bearing", "healthy vs faulty motor (balanced supply)",
               {"motor": "healthy"}, {"motor": "faulty"}, {"regime": "balanced"}),
    Comparison("supply_healthy", "balanced vs open-phase (healthy motor)",
               {"regime": "balanced"}, {"regime": "open_phase"}, {"motor": "healthy"}),
    Comparison("supply_faulty", "balanced vs open-phase (faulty motor)",
               {"regime": "balanced"}, {"regime": "open_phase"}, {"motor": "faulty"}),
    Comparison("supply_all", "balanced vs open-phase (both motors)",
               {"regime": "balanced"}, {"regime": "open_phase"}, {}),
    Comparison("supply_running", "running: balanced vs phase removed on line (02, 07)",
               {"regime": "balanced", "stem": lambda s: s.str.startswith(("02_", "07_"))},
               {"regime": "open_phase", "stem": lambda s: s.str.startswith(("02_", "07_"))},
               {}),
    Comparison("supply_running_healthy", "running: balanced vs phase removed (file 02)",
               {"regime": "balanced"}, {"regime": "open_phase"},
               {"motor": "healthy",
                "stem": lambda s: s.str.startswith(("02_", "07_"))}),
    Comparison("supply_running_faulty", "running: balanced vs phase removed (file 07)",
               {"regime": "balanced"}, {"regime": "open_phase"},
               {"motor": "faulty",
                "stem": lambda s: s.str.startswith(("02_", "07_"))}),
    Comparison("supply_startup", "startup: balanced vs phase missing at start (05, 10)",
               {"regime": "balanced",
                "stem": lambda s: s.str.startswith(("01_", "03_", "04_", "06_", "08_", "09_"))},
               {"regime": "open_phase",
                "stem": lambda s: s.str.startswith(("05_", "10_"))},
               {}),
    Comparison("open_running_vs_startup",
               "phase removed on line (02,07) vs missing at start (05,10)",
               {"regime": "open_phase",
                "stem": lambda s: s.str.startswith(("02_", "07_"))},
               {"regime": "open_phase",
                "stem": lambda s: s.str.startswith(("05_", "10_"))},
               {}),
    Comparison("load_0_04", "healthy, no-load vs 0.4 Nm",
               {"load_nm": 0.0}, {"load_nm": 0.4}, {"motor": "healthy", "regime": "balanced"}),
    Comparison("load_0_08", "healthy, no-load vs 0.8 Nm",
               {"load_nm": 0.0}, {"load_nm": 0.8}, {"motor": "healthy", "regime": "balanced"}),
    Comparison("load_04_08", "healthy, 0.4 Nm vs 0.8 Nm",
               {"load_nm": 0.4}, {"load_nm": 0.8}, {"motor": "healthy", "regime": "balanced"}),
)

BY_KEY = {c.key: c for c in COMPARISONS}


def compare(df: pd.DataFrame, cmp: Comparison, kind: str = "all",
            min_n: int = 5) -> pd.DataFrame:
    """Per-feature two-sided Mann-Whitney test and AUC for one comparison."""
    A = select(df, cmp.a, cmp.filt)
    B = select(df, cmp.b, cmp.filt)
    rows = []
    for c in feature_columns(df, kind):
        a = A[c].to_numpy(dtype=float)
        b = B[c].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < min_n or b.size < min_n:
            continue
        if np.allclose(a.std(), 0) and np.allclose(b.std(), 0) and \
                np.isclose(a.mean(), b.mean()):
            continue
        try:
            p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        auc = auc_score(a, b)
        rows.append({
            "comparison": cmp.key, "label": cmp.label, "feature": c,
            "n_a": a.size, "n_b": b.size,
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "auc": auc, "auc_dist": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
            "p": p,
            "cliffs_d": 2.0 * auc - 1.0 if np.isfinite(auc) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh"] = benjamini_hochberg(out["p"].to_numpy())
    return out.sort_values("auc_dist", ascending=False).reset_index(drop=True)


def compare_all(df: pd.DataFrame, kind: str = "all") -> pd.DataFrame:
    return pd.concat([compare(df, c, kind) for c in COMPARISONS],
                     ignore_index=True)

def _rf(seed: int = RANDOM_STATE):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=seed, n_jobs=1)


def classify(df: pd.DataFrame, cmp: Comparison, kind: str = "all",
             n_splits: int = 5, grouped: bool = True,
             n_seeds: int = 5) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    A = select(df, cmp.a, cmp.filt).copy()
    B = select(df, cmp.b, cmp.filt).copy()
    A["_y"], B["_y"] = 0, 1
    data = pd.concat([A, B], ignore_index=True)

    base = {"comparison": cmp.key, "label": cmp.label, "feature_kind": kind,
            "grouped": grouped, "n_windows": int(len(data)),
            "n_records": int(data["stem"].nunique()), "n_folds_used": 0}

    cols = sorted(c for c in feature_columns(df, kind)
                  if data[c].notna().all() and data[c].std() > 0)
    base["n_features"] = len(cols)
    if not cols or data["_y"].nunique() < 2:
        return {**base, "note": "degenerate design"}

    per_class_records = data.groupby("_y")["stem"].nunique()
    if grouped and int(per_class_records.min()) < 2:
        return {**base, "note": "grouped CV impossible: <2 records per class"}

    X = data[cols].to_numpy(dtype=float)
    y = data["_y"].to_numpy()
    groups = data["stem"].to_numpy()

    if grouped:
        n_splits = int(min(n_splits, data["stem"].nunique()))
        if n_splits < 2:
            return {**base, "note": "grouped CV impossible: <2 records"}
        folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    else:
        n_splits = int(min(n_splits, np.bincount(y).min()))
        if n_splits < 2:
            return {**base, "note": "not enough samples per class"}
        folds = [list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed).split(X, y))
                 for seed in range(n_seeds)]

    bas, aucs, used = [], [], 0
    for seed in range(n_seeds):
        it = folds if grouped else folds[seed]
        y_true, y_pred, y_prob = [], [], []
        for tr, te in it:
            if len(np.unique(y[tr])) < 2:
                continue
            clf = _rf(seed).fit(X[tr], y[tr])
            y_true.append(y[te])
            y_pred.append(clf.predict(X[te]))
            y_prob.append(clf.predict_proba(X[te])[:, 1])
        if not y_true:
            continue
        used += 1
        yt = np.concatenate(y_true)
        yp = np.concatenate(y_pred)
        ypr = np.concatenate(y_prob)
        if len(np.unique(yt)) < 2:
            continue
        bas.append(float(balanced_accuracy_score(yt, yp)))
        aucs.append(float(roc_auc_score(yt, ypr)))
    if not bas:
        return {**base, "note": "no usable fold"}

    return {**base,
            "n_folds_used": used,
            "n_seeds": len(bas),
            "balanced_accuracy": float(np.mean(bas)),
            "ba_std": float(np.std(bas)),
            "auc": float(np.mean(aucs)),
            "note": ""}


def classify_grid(df: pd.DataFrame,
                  kinds: tuple[str, ...] = ("all", "amplitude", "classical",
                                            "fractal", "entropy"),
                  comparisons: tuple[Comparison, ...] = COMPARISONS) -> pd.DataFrame:
    """Grouped and window-level scores for every descriptor set and contrast."""
    rows = []
    for cmp in comparisons:
        for kind in kinds:
            for grouped in (True, False):
                r = classify(df, cmp, kind, grouped=grouped)
                if r:
                    rows.append(r)
    return pd.DataFrame(rows)

def record_icc(df: pd.DataFrame, kind: str = "all",
               min_n: int = 20) -> pd.DataFrame:
    rows = []
    for c in feature_columns(df, kind):
        v = df[c]
        ok = v.notna()
        if int(ok.sum()) < min_n:
            continue
        vv = v[ok].astype(float)
        gg = df.loc[ok, "stem"]
        if vv.std() == 0:
            continue
        grand = vv.mean()
        ss_tot = float(((vv - grand) ** 2).sum())
        if ss_tot <= 0:
            continue
        ss_between = float(sum(len(g) * (g.mean() - grand) ** 2
                               for _, g in vv.groupby(gg)))
        rows.append({"feature": c, "eta2_record": ss_between / ss_tot,
                     "n_records": int(gg.nunique())})
    out = pd.DataFrame(rows)
    return out.sort_values("eta2_record", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def family_summary(icc: pd.DataFrame) -> pd.DataFrame:
    """Mean record-confound per feature family -- which preprocessing helps?"""
    if icc.empty:
        return icc
    i = icc.copy()
    i["family"] = i["feature"].map(feature_family)
    g = i.groupby("family")["eta2_record"]
    out = pd.DataFrame({
        "n_features": g.size(),
        "eta2_mean": g.mean(),
        "eta2_median": g.median(),
        "frac_ge_0.8": g.apply(lambda s: float((s >= 0.8).mean())),
    })
    return out.sort_values("eta2_mean", ascending=False)

def _exact_or_mc_unpaired(a: np.ndarray, b: np.ndarray, stat_fn,
                          n_perm: int = 20000, rng=None) -> tuple[float, float]:
    """Two-sided permutation p-value for two independent samples."""
    pool = np.concatenate([a, b])
    na = len(a)
    obs = stat_fn(a) - stat_fn(b)
    from math import comb
    total = comb(len(pool), na) if na <= len(pool) else 0
    if total and total <= n_perm:
        from itertools import combinations
        cnt = tot = 0
        for idx in combinations(range(len(pool)), na):
            mask = np.zeros(len(pool), dtype=bool)
            mask[list(idx)] = True
            s = stat_fn(pool[mask]) - stat_fn(pool[~mask])
            cnt += abs(s) >= abs(obs) - 1e-12
            tot += 1
        return float(obs), (cnt + 1) / (tot + 1)
    rng = rng or np.random.default_rng(0)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        s = stat_fn(perm[:na]) - stat_fn(perm[na:])
        cnt += abs(s) >= abs(obs) - 1e-12
    return float(obs), (cnt + 1) / (n_perm + 1)


def _exact_or_mc_paired(d: np.ndarray, stat_fn,
                        n_perm: int = 20000, rng=None) -> tuple[float, float]:
    """Two-sided sign-flip permutation p-value for paired differences."""
    n = len(d)
    obs = stat_fn(d)
    if n == 0:
        return float("nan"), float("nan")
    if n <= 15:
        cnt = tot = 0
        for bits in range(1 << n):
            signs = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(n)])
            s = stat_fn(d * signs)
            cnt += abs(s) >= abs(obs) - 1e-12
            tot += 1
        return float(obs), (cnt + 1) / (tot + 1)
    rng = rng or np.random.default_rng(0)
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        cnt += abs(stat_fn(d * signs)) >= abs(obs) - 1e-12
    return float(obs), (cnt + 1) / (n_perm + 1)


def record_level_features(df: pd.DataFrame, cmp: Comparison, kind: str = "all",
                          use_median: bool = True) -> pd.DataFrame:
    """Per-feature record-level statistic and permutation p-value."""
    a = select(df, cmp.a, cmp.filt)
    b = select(df, cmp.b, cmp.filt)
    if a.empty or b.empty:
        return pd.DataFrame()
    stat_fn = np.median if use_median else np.mean
    shared = sorted(set(a["stem"]) & set(b["stem"]))
    rows = []
    for f in feature_columns(df, kind):
        ma = a.groupby("stem")[f].median()
        mb = b.groupby("stem")[f].median()
        if shared:
            d = (ma.reindex(shared) - mb.reindex(shared)).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            stat, p = _exact_or_mc_paired(d, stat_fn)
            unit = "record-paired"
            n_a, n_b = len(shared), len(shared)
            auc = float((d > 0).mean() + 0.5 * (d == 0).mean()) if d.size else np.nan
        else:
            va = ma.to_numpy(dtype=float)
            vb = mb.to_numpy(dtype=float)
            va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            if va.size < 2 or vb.size < 2:
                continue
            stat, p = _exact_or_mc_unpaired(va, vb, stat_fn)
            unit = "record-unpaired"
            n_a, n_b = int(va.size), int(vb.size)
            u = stats.mannwhitneyu(va, vb, alternative="two-sided").statistic
            auc = float(u / (va.size * vb.size))
        rows.append({"comparison": cmp.key, "label": cmp.label, "unit": unit,
                     "feature": f, "family": feature_family(f),
                     "n_records_a": n_a, "n_records_b": n_b,
                     "n_shared": len(shared), "statistic": stat,
                     "p_perm": p, "auc_record": auc})
    return pd.DataFrame(rows)


def record_level_table(df: pd.DataFrame,
                       comparisons: tuple[Comparison, ...] = COMPARISONS,
                       kind: str = "all") -> pd.DataFrame:
    """Aggregate the record-level test per comparison and feature family."""
    rows = []
    for cmp in comparisons:
        per = record_level_features(df, cmp, kind=kind)
        if per.empty:
            continue
        for fam, g in per.groupby("family", sort=False):
            best = g.loc[g["p_perm"].idxmin()]
            rows.append({
                "comparison": cmp.key, "label": cmp.label,
                "unit": best["unit"],
                "n_records_a": best["n_records_a"],
                "n_records_b": best["n_records_b"],
                "n_shared": best["n_shared"],
                "family": fam, "n_features": len(g),
                "n_p05": int((g["p_perm"] <= 0.05).sum()),
                "min_p": float(g["p_perm"].min()),
                "min_p_bonf": float(min(1.0, g["p_perm"].min() * len(g))),
                "auc_record_median": float(g["auc_record"].median()),
                "best_feature": best["feature"],
            })
    return pd.DataFrame(rows)


def within_record_features(df: pd.DataFrame, stem: str, kind: str = "all",
                           regimes: tuple[str, ...] = ("balanced",
                                                       "open_phase")
                           ) -> pd.DataFrame:
    sub = df[(df["stem"] == stem) & df["regime"].isin(regimes)]
    if sub["regime"].nunique() < 2 or len(sub) < 6:
        return pd.DataFrame()
    sub = sub.sort_values("win_index")
    lab = sub["regime"].eq(regimes[1]).to_numpy()
    n = len(sub)
    rows = []
    for f in feature_columns(df, kind):
        x = sub[f].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            continue
        obs = float(np.median(x[lab]) - np.median(x[~lab]))
        stats_shift = []
        for k in range(1, n):
            lk = np.roll(lab, k)
            if lk.sum() == 0 or lk.sum() == n:
                continue
            stats_shift.append(float(np.median(x[lk]) - np.median(x[~lk])))
        if not stats_shift:
            continue
        extreme = sum(1 for s in stats_shift if abs(s) >= abs(obs) - 1e-12)
        p = (1 + extreme) / (1 + len(stats_shift))
        rows.append({"stem": stem, "feature": f, "family": feature_family(f),
                     "n_windows": n, "n_open": int(lab.sum()),
                     "statistic": obs, "p_shift": p})
    return pd.DataFrame(rows)


def within_record_table(df: pd.DataFrame, kind: str = "all") -> pd.DataFrame:
    """Best cyclic-shift p-value per recording and family."""
    rows = []
    for stem in sorted(df["stem"].unique()):
        sub = df[df["stem"] == stem]
        if sub["regime"].nunique() < 2:
            continue
        per = within_record_features(df, stem, kind=kind)
        if per.empty:
            continue
        for fam, g in per.groupby("family", sort=False):
            best = g.loc[g["p_shift"].idxmin()]
            rows.append({"stem": stem, "family": fam, "n_features": len(g),
                         "n_windows": int(best["n_windows"]),
                         "n_open": int(best["n_open"]),
                         "min_p_shift": float(g["p_shift"].min()),
                         "n_p05": int((g["p_shift"] <= 0.05).sum()),
                         "best_feature": best["feature"]})
    return pd.DataFrame(rows)

def _window_rows(df: pd.DataFrame, per_group: int, groups=None):
    """(label, row) pairs: evenly spaced representative windows per group."""
    for gi, (label, spec, _) in enumerate(CONDITION_GROUPS):
        if groups is not None and gi not in groups:
            continue
        sub = select(df, spec)
        if sub.empty:
            continue
        idx = np.linspace(0, len(sub) - 1, min(per_group, len(sub))).astype(int)
        for _, row in sub.iloc[idx].iterrows():
            yield label, row


def mfdfa_surrogate_table(df: pd.DataFrame,
                          channels: tuple[str, ...] = ("I1", "x"),
                          kinds: tuple[str, ...] = ("shuffle", "phase",
                                                    "pipeline"),
                          per_group: int = 1, n_rep: int = 20,
                          seed: int = 0,
                          groups: tuple[int, ...] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, row in _window_rows(df, per_group, groups):
        sig = load_record(BY_STEM[row["stem"]])
        f0 = float(row["f0"])
        for ch in channels:
            x_raw = np.ascontiguousarray(sig[ch][int(row["i0"]):int(row["i1"])])
            x, _ = remove_harmonics(x_raw, FS, f0, NHARM)
            for kind in kinds:
                recs = [{"surrogate": "original", "rep": -1, **mfdfa_stat(x)}]
                for r in range(n_rep):
                    if kind == "shuffle":
                        s = rng.permutation(x)
                    elif kind == "phase":
                        s = phase_surrogate(x, rng)
                    elif kind == "iaaft":
                        s = iaaft_surrogate(x, rng)
                    else:
                        s = pipeline_matched_surrogate(x_raw, f0, rng)
                    st = mfdfa_stat(s)
                    if st:
                        recs.append({"surrogate": kind, "rep": r, **st})
                sur = pd.DataFrame(recs)
                if "surrogate" not in sur or len(sur) < 5:
                    continue
                summary = summarise(sur)
                summary.insert(0, "group", label)
                summary.insert(1, "stem", row["stem"])
                summary.insert(2, "channel", ch)
                summary.insert(3, "kind", kind)
                rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def entropy_surrogate_table(df: pd.DataFrame,
                            channels: tuple[str, ...] = ("I1", "x"),
                            metrics: tuple[str, ...] = ("sampen", "permen",
                                                        "svden", "lzc",
                                                        "renyi_q2"),
                            per_group: int = 1, n_rep: int = 20,
                            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, row in _window_rows(df, per_group):
        sig = load_record(BY_STEM[row["stem"]])
        f0 = float(row["f0"])
        for ch in channels:
            x_raw = np.ascontiguousarray(sig[ch][int(row["i0"]):int(row["i1"])])
            x, _ = remove_harmonics(x_raw, FS, f0, NHARM)
            for met in metrics:
                fn = renyi if met == "renyi_q2" else ENTROPY_METRICS[met]
                orig = fn(x)
                if not np.isfinite(orig):
                    continue
                sh_vals, pi_vals = [], []
                for _ in range(n_rep):
                    v = fn(rng.permutation(x))
                    if np.isfinite(v):
                        sh_vals.append(v)
                    v = fn(pipeline_matched_surrogate(x_raw, f0, rng))
                    if np.isfinite(v):
                        pi_vals.append(v)
                if len(sh_vals) < 10 or len(pi_vals) < 10:
                    continue
                sh_m, sh_s = float(np.mean(sh_vals)), float(np.std(sh_vals))
                pi_m, pi_s = float(np.mean(pi_vals)), float(np.std(pi_vals))
                rows.append({
                    "group": label, "stem": row["stem"], "channel": ch,
                    "metric": met, "original": float(orig),
                    "shuffle_mean": sh_m, "shuffle_std": sh_s,
                    "pipeline_mean": pi_m, "pipeline_std": pi_s,
                    "z_shuffle": (float(orig) - sh_m) / sh_s if sh_s > 0
                    else np.nan,
                    "z_pipeline": (float(orig) - pi_m) / pi_s if pi_s > 0
                    else np.nan,
                })
    return pd.DataFrame(rows)


def pool_shards(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["group", "stem", "channel", "kind"]
    for (grp, stem, ch, kind), g in df.groupby(keys, sort=False):
        orig = g[g["surrogate"] == "original"]
        null = g[g["surrogate"] != "original"]
        if orig.empty or null.empty:
            continue
        rec = {"group": grp, "stem": stem, "channel": ch, "kind": kind,
               "n_windows": int(len(orig)),
               "surrogate": str(null["surrogate"].iloc[0]),
               "n": int(null["n"].sum())}
        for s in SURROGATE_STATS:
            om = orig[f"{s}_mean"].to_numpy(float)
            om = om[np.isfinite(om)]
            vm = null[f"{s}_mean"].to_numpy(float)
            vsd = null[f"{s}_std"].to_numpy(float)
            ns = null["n"].to_numpy(float)
            ok = np.isfinite(vm) & np.isfinite(vsd)
            if om.size == 0 or not ok.any():
                continue
            vm, vsd, ns = vm[ok], vsd[ok], ns[ok]
            w = ns / ns.sum() if ns.sum() > 0 else np.full(vm.size, 1.0 / vm.size)
            obs = float(om.mean())
            pool_mean = float(np.sum(w * vm))
            # mixture of Gaussians: E[x^2] = mean(var + mean^2)
            second = float(np.sum(w * (vsd ** 2 + vm ** 2)))
            pool_std = float(np.sqrt(max(second - pool_mean ** 2, 0.0)))
            rec[f"{s}_mean"] = pool_mean
            rec[f"{s}_std"] = pool_std
            rec[f"{s}_obs"] = obs
            if pool_std > 0:
                ge = int(np.sum(vm >= obs))
                le = int(np.sum(vm <= obs))
                n = int(vm.size)
                rec[f"{s}_p_emp"] = min(
                    1.0, 2.0 * min((ge + 1) / (n + 1), (le + 1) / (n + 1)))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        cols = ["group", "stem", "channel", "kind", "surrogate",
                "n_windows", "n"]
        cols += [c for c in out.columns if c not in cols]
        out = out[cols].sort_values(["channel", "group", "kind"]).reset_index(drop=True)
    return out

PUBLISHED_SURROGATE_SEEDS = tuple(1000 + g for g in range(len(CONDITION_GROUPS)))


def published_surrogate_table(df: pd.DataFrame, n_rep: int = 200,
                              per_group: int = 1,
                              channels: tuple[str, ...] = ("I1", "x"),
                              kinds: tuple[str, ...] = ("shuffle", "phase",
                                                        "pipeline")
                              ) -> pd.DataFrame:
    frames = []
    for gi, seed in enumerate(PUBLISHED_SURROGATE_SEEDS):
        frames.append(mfdfa_surrogate_table(
            df, channels=channels, kinds=kinds, per_group=per_group,
            n_rep=n_rep, seed=seed, groups=(gi,)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def multi_window_surrogate_table(df: pd.DataFrame, per_group: int = 3,
                                 n_rep: int = 50, seed: int = 0,
                                 channels: tuple[str, ...] = ("I1", "x"),
                                 kinds: tuple[str, ...] = ("shuffle", "phase",
                                                           "pipeline", "iaaft"),
                                 groups: tuple[int, ...] | None = None
                                 ) -> pd.DataFrame:
    per_window = mfdfa_surrogate_table(df, channels=channels, kinds=kinds,
                                       per_group=per_group, n_rep=n_rep,
                                       seed=seed, groups=groups)
    if per_window.empty:
        return per_window
    return pool_shards(per_window)

def write_tables(df: pd.DataFrame, surrogates: bool = True,
                 do_classify: bool = True, multi: bool = False,
                 tables_dir=TABLES_DIR) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    cmp_df = compare_all(df)
    cmp_df.to_csv(tables_dir / "comparisons.csv", index=False)
    out["comparisons"] = cmp_df
    print(f"  comparisons.csv ({cmp_df.shape[0]} rows)")

    if do_classify:
        cls = classify_grid(df)
        if not cls.empty:
            cls.to_csv(tables_dir / "classification.csv", index=False)
            out["classification"] = cls
            print(f"  classification.csv ({cls.shape[0]} rows)")

    icc = record_icc(df)
    fam = family_summary(icc)
    if not icc.empty:
        icc.to_csv(tables_dir / "record_variance.csv", index=False)
        fam.to_csv(tables_dir / "record_variance_family.csv")
        out["record_variance"] = icc
        out["record_variance_family"] = fam
        print(f"  record_variance.csv - "
              f"{float((icc['eta2_record'] >= 0.8).mean()):.0%} of features "
              f"have eta^2(record) >= 0.8")

    rlev = record_level_table(df)
    if not rlev.empty:
        rlev.to_csv(tables_dir / "record_level.csv", index=False)
        out["record_level"] = rlev
        print(f"  record_level.csv ({rlev.shape[0]} rows)")
    wrec = within_record_table(df)
    if not wrec.empty:
        wrec.to_csv(tables_dir / "within_record.csv", index=False)
        out["within_record"] = wrec
        print(f"  within_record.csv ({wrec.shape[0]} rows)")

    if surrogates:
        sur = published_surrogate_table(df, n_rep=200, per_group=1)
        if not sur.empty:
            sur.to_csv(tables_dir / "surrogates.csv", index=False)
            out["surrogates"] = sur
            print(f"  surrogates.csv ({sur.shape[0]} rows)")
        if multi:
            mt = multi_window_surrogate_table(df, per_group=3, n_rep=50, seed=0)
            if not mt.empty:
                mt.to_csv(tables_dir / "surrogates_multi.csv", index=False)
                out["surrogates_multi"] = mt
                print(f"  surrogates_multi.csv ({mt.shape[0]} rows, "
                      f"{mt.n_windows.sum()} windows pooled)")
        esur = entropy_surrogate_table(df)
        if not esur.empty:
            esur.to_csv(tables_dir / "entropy_surrogates.csv", index=False)
            out["entropy_surrogates"] = esur
            print(f"  entropy_surrogates.csv ({esur.shape[0]} rows)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the statistics tables from window_features.csv")
    ap.add_argument("--features", default=str(TABLES_DIR / "window_features.csv"))
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--skip-surrogates", action="store_true")
    ap.add_argument("--multi-surrogates", action="store_true",
                    help="also run the pooled multi-window robustness check")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.features)
    print(f"loaded {args.features} ({df.shape[0]} x {df.shape[1]})")
    write_tables(df, surrogates=not args.skip_surrogates,
                 do_classify=not args.skip_classify, multi=args.multi_surrogates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
