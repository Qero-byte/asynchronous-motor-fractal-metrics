#!/usr/bin/env python
"""Paper figures.
    python figures.py                  # all figures from the shipped tables
    python figures.py --skip spectra   # skip the MFDFA spectra (they need data)
run_all.py calls draw_all() after it has rebuilt the tables.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fractal_metrics import (ACCEL, BY_STEM, CHANNELS, CURRENT, FIGURES_DIR, FS,
                             FS_NOMINAL, NHARM, RECORDS, TABLES_DIR, _block_rms,
                             load_record, mfdfa, regime_series, remove_harmonics)
from group_statistics import (COMPARISONS, CONDITION_GROUPS, feature_family,
                              select)

#: alias kept short: (label, row selector, colour) triples
SPECTRA_GROUPS = CONDITION_GROUPS
GROUP_COLORS = ("tab:green", "tab:red", "tab:olive", "tab:purple",
                "tab:blue", "tab:orange")


def boxplot(ax, data, labels, **kw):
    """ax.boxplot across the matplotlib rename of labels -> tick_labels."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


# ===========================================================================
# Dataset audit figures
# ===========================================================================

def fig_regime_timeline(path=None):
    """Supply RMS and I1 RMS per second, with regime shading, for all records."""
    path = path or (FIGURES_DIR / "fig01_regime_timeline.png")
    fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True)
    colors = {"off": "#d9d9d9", "balanced": "#cfe8cf",
              "open_phase": "#f8d5cf", "transient": "#fff2cc"}
    for ax, rec in zip(axes.ravel(), RECORDS):
        sig = load_record(rec)
        labels, supply, i2 = regime_series(sig)
        t = np.arange(len(labels))
        for lab, col in colors.items():
            for k in [k for k, L in enumerate(labels) if L == lab]:
                ax.axvspan(k, k + 1, color=col, lw=0, zorder=0)
        i1 = _block_rms(sig["I1"], FS)
        ax.plot(t, supply, "o-", color="tab:blue", ms=3, lw=1, label="supply RMS [V]")
        ax2 = ax.twinx()
        ax2.plot(t, i1, "s-", color="tab:red", ms=2.5, lw=1, label="RMS(I1) [A]")
        ax2.set_ylabel("RMS(I1) [A]", color="tab:red", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
        ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(f"{rec.idx:02d}  {rec.motor}, {rec.condition}"
                     + (f", {rec.load_nm:g} Nm" if rec.load_nm else ""), fontsize=9)
        ax.set_ylabel("supply RMS [V]", color="tab:blue", fontsize=8)
        ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.25)
        if rec is RECORDS[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors), ncol=4, loc="lower center", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Operating regimes per record (shading = detected regime)", y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_quantization(table: pd.DataFrame, path=None):
    """Distinct-value counts and step-to-std ratio per channel."""
    path = path or (FIGURES_DIR / "fig02_quantization.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    order = list(ACCEL) + list(CURRENT) + list(CHANNELS[6:])

    ax = axes[0]
    # one box per channel (10 records each), individual records overplotted;
    # a grouped-bar version drew every channel at the same x, so only the last
    # one was visible.
    data = [table[table.channel == ch]["n_unique"].to_numpy() for ch in order]
    boxplot(ax, data, order)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(vals.size, i) + rng.uniform(-.14, .14, vals.size),
                   vals, s=16, zorder=3, color="tab:blue", alpha=.85,
                   edgecolor="none", label="one record" if i == 1 else None)
    ax.set_yscale("log")
    ax.set_title("distinct values per channel (log)")
    ax.set_xlabel("channel")
    ax.set_ylabel("n distinct values")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=.3, which="both")

    ax = axes[1]
    data = [table[table.channel == ch]["step"].to_numpy() for ch in order]
    boxplot(ax, data, order)
    ax.set_yscale("log")
    ax.set_title("effective quantization step")
    ax.set_ylabel("step (channel unit)")
    ax.grid(alpha=.3, which="both")

    ax = axes[2]
    data = [table[table.channel == ch]["step_to_std"].to_numpy() for ch in order]
    boxplot(ax, data, order)
    ax.set_yscale("log")
    ax.axhline(0.1, color="r", ls="--", lw=1, label="step = 0.1 sigma")
    ax.set_title("step relative to signal std")
    ax.set_ylabel("step / std")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")

    fig.suptitle("Value resolution of the recorded channels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_zoh(demo: dict, path=None):
    """log-log fluctuation curves with and without the zero-order hold."""
    path = path or (FIGURES_DIR / "fig03_zoh_artifact.png")
    series = (("raw", f"as recorded ({FS_NOMINAL / 1000:.0f} kHz, 5x hold)",
               "o-", FS_NOMINAL),
              ("dec", f"de-held ({FS / 1000:.0f} kHz)", "s-", FS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for tag, lab, style, fs in series:
        d = demo[tag]
        axes[0].loglog(d["win"] / fs * 1e3, d["F"], style, ms=3.5, lw=1.2, label=lab)
    hold_ms = demo["zoh"] / FS_NOMINAL * 1e3
    axes[0].axvline(hold_ms, color="k", ls="--", lw=1)
    axes[0].annotate(f"n = {demo['zoh']} samples\n(one hold, {hold_ms:.2f} ms)",
                     xy=(hold_ms, np.nanmin(demo["raw"]["F"]) * 1e3),
                     xytext=(hold_ms * 6, np.nanmax(demo["raw"]["F"]) * 1e-2),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
    axes[0].set_xlabel("window duration [ms]")
    axes[0].set_ylabel("F(n, q=2)")
    axes[0].set_title("scaling curve: hold artefact at short windows")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=.3, which="both")

    for tag, lab, style, fs in series:
        d = demo[tag]
        axes[1].semilogx(d["win"] / fs * 1e3, d["F"], style, ms=3.5, lw=1.2, label=lab)
    axes[1].set_xlabel("window duration [ms]")
    axes[1].set_ylabel("F(n, q=2)")
    axes[1].set_title("linear ordinate: the held curve collapses at short windows")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3, which="both")

    fig.suptitle(f"{demo['stem']} -- channel {demo['channel']}, same physical 2 s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# MFDFA spectra

def _window_signal(row: pd.Series, channel: str):
    rec = BY_STEM[row["stem"]]
    sig = load_record(rec)
    return np.ascontiguousarray(sig[channel][int(row["i0"]):int(row["i1"])])


def compute_curves(df: pd.DataFrame, channel: str, transform: str = "res",
                   per_group: int = 4) -> dict[str, list[dict]]:
    """MFDFA curves for a few representative windows of each condition group."""
    out: dict[str, list[dict]] = {}
    for label, spec, _ in SPECTRA_GROUPS:
        sub = select(df, spec)
        if sub.empty:
            continue
        idx = np.linspace(0, len(sub) - 1, min(per_group, len(sub))).astype(int)
        curves = []
        for _, row in sub.iloc[idx].iterrows():
            x = _window_signal(row, channel)
            if transform == "res":
                x, _ = remove_harmonics(x, FS, float(row["f0"]), NHARM)
            _, cv = mfdfa(x, prefix="_")
            if cv:
                curves.append(cv)
        if curves:
            out[label] = curves
    return out


def fig_mfdfa_spectra(df: pd.DataFrame, channel: str = "I1",
                      transform: str = "res", title_extra: str = "",
                      path=None):
    """h(q), tau(q) and f(alpha) for representative windows per condition."""
    path = path or (FIGURES_DIR / f"fig04_mfdfa_spectra_{channel}.png")
    data = compute_curves(df, channel, transform)
    if not data:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = {lab: col for lab, _, col in SPECTRA_GROUPS}

    for lab, curves in data.items():
        col = cmap.get(lab, "gray")
        q = curves[0]["q"]
        h = np.vstack([c["h"] for c in curves])
        tau = np.vstack([c["tau"] for c in curves])
        al = np.concatenate([c["alpha"] for c in curves])
        fa = np.concatenate([c["f_alpha"] for c in curves])
        for c in curves:
            axes[0].plot(q, c["h"], "-", color=col, lw=.7, alpha=.45)
            axes[1].plot(q, c["tau"], "-", color=col, lw=.7, alpha=.45)
            axes[2].plot(c["alpha"], c["f_alpha"], "-", color=col, lw=.7, alpha=.45)
        axes[0].plot(q, h.mean(0), "o-", color=col, ms=3.5, lw=2, label=lab)
        axes[1].plot(q, tau.mean(0), "o-", color=col, ms=3.5, lw=2, label=lab)
        o = np.argsort(al)
        axes[2].plot(al[o], fa[o], "o-", color=col, ms=3, lw=2, label=lab)

    axes[0].axhline(0.5, color="gray", ls="--", lw=.8)
    axes[0].set_xlabel("q"); axes[0].set_ylabel("h(q)")
    axes[0].set_title("generalised Hurst exponent")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("q"); axes[1].set_ylabel(r"$\tau(q)$")
    axes[1].set_title("mass exponent")

    axes[2].set_xlabel(r"$\alpha$"); axes[2].set_ylabel(r"$f(\alpha)$")
    axes[2].set_title("singularity spectrum")

    for a in axes:
        a.grid(alpha=.3)
    name = "harmonic-removed residual" if transform == "res" else "raw signal"
    fig.suptitle(f"MFDFA of {channel} ({name}) -- {title_extra}".strip(" --"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# Feature separation and classifier figures

def fig_feature_boxes(df: pd.DataFrame, cmp_df: pd.DataFrame, comparison: str,
                      n: int = 6, path=None):
    """Box/strip plots of the most discriminative features of one comparison."""
    cmp = next(c for c in COMPARISONS if c.key == comparison)
    sub = cmp_df[cmp_df.comparison == comparison].head(n)
    if sub.empty:
        return None
    path = path or (FIGURES_DIR / f"fig05_features_{comparison}.png")
    A = select(df, cmp.a, cmp.filt)
    B = select(df, cmp.b, cmp.filt)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 7))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, sub.iterrows()):
        a, b = A[r["feature"]].dropna(), B[r["feature"]].dropna()
        boxplot(ax, [a, b], ["A", "B"], widths=.5, showfliers=False)
        for i, vals in enumerate((a, b), start=1):
            ax.scatter(np.full(len(vals), i) + np.random.uniform(-.09, .09, len(vals)),
                       vals, s=7, alpha=.55,
                       color=GROUP_COLORS[i - 1], zorder=3)
        ax.set_title(f"{r['feature']}\nAUC={r['auc']:.3f}  q={r['q_bh']:.1e}",
                     fontsize=8)
        ax.grid(alpha=.3)
    for ax in axes[len(sub):]:
        ax.axis("off")
    fig.suptitle(f"{cmp.label}\nA = {cmp.a}   B = {cmp.b}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


FAMILY_ORDER = ["single-scale (raw)", "MFDFA (raw)", "entropy+complexity (raw)",
                "Renyi spectrum (raw)", "single-scale (residual)",
                "MFDFA (residual)", "entropy+complexity (residual)",
                "Renyi spectrum (residual)", "multiscale entropy (residual)",
                "MFDCCA (residual)"]

#: |AUC - 0.5| of a perfectly separating feature (AUC = 0 or 1).
AUC_CEILING = 0.5


def fig_auc_ranking(cmp_df: pd.DataFrame, path=None):
    """Saturation of feature separation per comparison."""
    path = path or (FIGURES_DIR / "fig06_auc_ranking.png")
    if "family" not in cmp_df.columns:
        cmp_df = cmp_df.assign(family=cmp_df["feature"].map(feature_family))
    keys = [c.key for c in COMPARISONS]
    keys = [k for k in keys if k in set(cmp_df.comparison)]
    labels = [next(c.label for c in COMPARISONS if c.key == k) for k in keys]
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(FAMILY_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    y = np.arange(len(keys))[::-1]

    ax = axes[0]
    for yi, key in zip(y, keys):
        sub = cmp_df[cmp_df.comparison == key]
        tie = sub[np.round(sub["auc_dist"], 6) == AUC_CEILING]
        counts = tie["family"].value_counts()
        left = 0
        for fam in FAMILY_ORDER:
            n = int(counts.get(fam, 0))
            if n == 0:
                continue
            ax.barh(yi, n, left=left, color=colors[fam], height=.68,
                    label=fam if key == keys[0] else None)
            left += n
        ax.text(left + 1.0, yi, f"{left} / {len(sub)}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, cmp_df.groupby("comparison").size().max() * 1.18)
    ax.set_xlabel("features reaching |AUC - 0.5| = 0.5 (perfect separation)",
                  fontsize=8)
    ax.legend(fontsize=6.2, ncol=2, loc="lower right", framealpha=.9)
    ax.grid(alpha=.25, axis="x")

    ax = axes[1]
    rng = np.random.default_rng(0)
    for yi, key in zip(y, keys):
        sub = cmp_df[cmp_df.comparison == key]
        sig = (sub["q_bh"] <= 0.05).to_numpy()
        ax.scatter(sub["auc_dist"], yi + rng.uniform(-.2, .2, len(sub)),
                   s=7, alpha=.75, edgecolor="none",
                   color=np.where(sig, "tab:red", "lightgray"))
    ax.axvline(AUC_CEILING, color="k", ls="--", lw=.8)
    ax.set_xlabel("|AUC - 0.5|  (red = FDR q <= 0.05)", fontsize=8)
    ax.grid(alpha=.25, axis="x")

    fig.suptitle("Where feature separation saturates, and by which family",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_classification(cls: pd.DataFrame, path=None):
    """Balanced accuracy per descriptor set under grouped cross-validation."""
    path = path or (FIGURES_DIR / "fig07_classification.png")
    if cls is None or cls.empty:
        return None
    kind_label = {"amplitude": "amplitude (RMS/sigma/crest)",
                  "classical": "single-scale",
                  "fractal": "multifractal (MFDFA+MFDCCA)",
                  "entropy": "entropy/complexity",
                  "all": "all features"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    kinds = [k for k in ("amplitude", "classical", "fractal", "entropy")
             if k in set(cls.feature_kind)]
    keys = [c.key for c in COMPARISONS if c.key in set(cls.comparison)]
    labels = [k.replace("supply_", "sup. ").replace("open_running_vs_startup",
                                                    "single-phasing")
              .replace("_", " ") for k in keys]
    x = np.arange(len(labels))
    w = 0.8 / max(len(kinds), 1)
    for i, kind in enumerate(kinds):
        vals = []
        for key in keys:
            s = cls[(cls.comparison == key) & (cls.feature_kind == kind)
                    & (cls.grouped)]
            vals.append(float(s["balanced_accuracy"].iloc[0]) if len(s) else np.nan)
        axes[0].bar(x + i * w, vals, w, label=kind_label.get(kind, kind))
    axes[0].set_xticks(x + 0.4 - w / 2)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    axes[0].axhline(0.5, color="k", ls="--", lw=.9, label="chance")
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("record-grouped CV (honest)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.3, axis="y")

    for kind, col in zip(kinds, GROUP_COLORS):
        g = cls[(cls.feature_kind == kind) & cls.grouped]
        n = cls[(cls.feature_kind == kind) & ~cls.grouped]
        m = g.merge(n, on="comparison", suffixes=("_g", "_n"))
        axes[1].scatter(m["balanced_accuracy_g"], m["balanced_accuracy_n"],
                        s=45, color=col, label=kind_label.get(kind, kind),
                        alpha=.85)
    lo, hi = 0.4, 1.02
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=.8)
    axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("grouped CV balanced accuracy")
    axes[1].set_ylabel("window-level CV balanced accuracy")
    axes[1].set_title("optimism from window-level splitting")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_record_variance(icc: pd.DataFrame, top: int = 25, path=None):
    """Features ranked by the share of variance explained by record identity."""
    path = path or (FIGURES_DIR / "fig09_record_variance.png")
    if icc is None or icc.empty:
        return None
    sub = icc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(sub) + 2))
    cols = ["tab:red" if v >= 0.8 else ("tab:orange" if v >= 0.5 else "tab:blue")
            for v in sub["eta2_record"]]
    ax.barh(np.arange(len(sub)), sub["eta2_record"], color=cols)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels([f.replace("mfdfa_", "").replace("dcca_res_", "D:")
                        for f in sub["feature"]], fontsize=7)
    ax.axvline(0.5, color="k", ls="--", lw=.8)
    ax.set_xlabel(r"$\eta^2$ explained by record identity")
    ax.set_xlim(0, 1.02)
    ax.set_title("How much of each feature is just 'which session it came from'?\n"
                 "red: >= 0.8 (mostly a session marker),  orange: >= 0.5")
    ax.grid(alpha=.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_dcca(df: pd.DataFrame, path=None):
    """MFDCCA spectra for the cross-channel pairs."""
    path = path or (FIGURES_DIR / "fig08_mfdcca.png")
    pairs = list(dict.fromkeys(
        c[len("dcca_res_"):-len("_h2")]
        for c in df.columns if c.startswith("dcca_res_") and c.endswith("_h2")))
    if not pairs:
        return None
    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 8))
    axes = np.atleast_2d(axes)
    for j, p in enumerate(pairs):
        a, b = p.split("_")
        for i, (xcol, ycol, xlab) in enumerate((
                (f"dcca_res_{p}_alpha0", f"dcca_res_{p}_dalpha",
                 r"$\alpha_0$"),
                (f"dcca_res_{p}_h2", f"dcca_res_{p}_dalpha", "h(q=2)"))):
            ax = axes[i, j]
            if xcol not in df.columns or ycol not in df.columns:
                ax.axis("off")
                continue
            for (lab, spec, col) in SPECTRA_GROUPS:
                sub = select(df, spec)
                if sub.empty:
                    continue
                ax.scatter(sub[xcol], sub[ycol], s=22, color=col, alpha=.8,
                           label=lab if i == 0 else None)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ycol.replace(f"dcca_res_{p}_", ""))
            ax.grid(alpha=.3)
            if i == 0:
                ax.set_title(f"MFDCCA {a}-{b}")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7)
    fig.suptitle("Multifractal cross-correlation features by condition")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# Entropy figures

def _entropy_panel(df: pd.DataFrame, feats, path, suptitle):
    if not feats:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.ravel(), feats):
        data, labels = [], []
        for (lab, spec, _) in SPECTRA_GROUPS:
            sub = select(df, spec)
            if sub.empty:
                continue
            v = sub[f].dropna()
            if len(v):
                data.append(v.to_numpy())
                labels.append(lab.replace(", ", "\n"))
        if data:
            boxplot(ax, data, labels, showfliers=False)
        ax.set_title(f, fontsize=9)
        ax.grid(alpha=.3, axis="y")
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_entropy_boxes(df: pd.DataFrame, path=None):
    """Key entropy features of the residuals, by condition group."""
    path = path or (FIGURES_DIR / "fig10_entropy_boxes.png")
    feats = [c for c in ("ent_res_I1_sampen", "ent_res_I2_sampen",
                         "ent_res_x_sampen", "ent_res_I1_permen",
                         "ent_res_I2_lzc", "mse_res_I1_area")
             if c in df.columns]
    return _entropy_panel(
        df, feats, path,
        "Entropy metrics of the harmonic-removed residuals, by condition")


def fig_entropy_raw_boxes(df: pd.DataFrame, path=None):
    """Raw-signal entropy features: where the open-phase signature actually lives."""
    path = path or (FIGURES_DIR / "fig12_entropy_raw.png")
    feats = [c for c in ("ent_raw_I2_sampen", "ent_raw_I2_appen",
                         "ent_raw_I2_permen", "renyi_raw_I2_q2",
                         "ent_raw_I2_zerocross", "ent_raw_V1_permen")
             if c in df.columns]
    return _entropy_panel(
        df, feats, path,
        "Entropy features of the RAW (carrier-bearing) signals, by condition:\n"
        "the open-phase signature is a carrier-disappearance effect and lives "
        "here, not in the residual")


def fig_mse_curves(df: pd.DataFrame, path=None):
    """Multiscale sample entropy of the I1 residual by supply regime."""
    path = path or (FIGURES_DIR / "fig11_mse_curves.png")
    scales = [2, 3, 4, 5, 6, 8, 10]
    cols = [c for c in (f"mse_res_I1_s{s}" for s in scales) if c in df.columns]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (lab, spec, col) in (("balanced", {"regime": "balanced"}, "tab:green"),
                             ("open phase", {"regime": "open_phase"}, "tab:red")):
        sub = select(df, spec)
        if sub.empty:
            continue
        med = sub[cols].median()
        q1 = sub[cols].quantile(0.25)
        q3 = sub[cols].quantile(0.75)
        ax.errorbar(scales[: len(cols)], med, yerr=[med - q1, q3 - med],
                    fmt="o-", color=col, label=lab, capsize=3, lw=1.6)
    ax.set_xlabel("coarse-graining scale $\\tau$")
    ax.set_ylabel("SampEn of coarse-grained $I_1$ residual")
    ax.set_title("Multiscale sample entropy of the $I_1$ residual "
                 "(median with interquartile range over windows)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# Driver

def draw_all(df: pd.DataFrame, cmp_df: pd.DataFrame | None = None,
             cls: pd.DataFrame | None = None, icc: pd.DataFrame | None = None,
             quantization: pd.DataFrame | None = None,
             zoh: dict | None = None, skip: tuple[str, ...] = ()) -> list:
    """Draw every figure that the given tables support.  Returns the paths."""
    if cmp_df is None:
        cmp_df = pd.read_csv(TABLES_DIR / "comparisons.csv")
    made = []

    def _run(tag, fn, *args, **kw):
        if tag in skip:
            return
        try:
            p = fn(*args, **kw)
            if p:
                made.append(p)
                print(f"  {p}")
        except Exception as exc:                           # noqa: BLE001
            print(f"  !! {tag}: {exc!r}")

    if quantization is not None:
        _run("quantization", fig_quantization, quantization)
    if zoh is not None:
        _run("zoh", fig_zoh, zoh)
    _run("regime_timeline", fig_regime_timeline)
    for ch in ("I1", "x"):
        if "spectra" not in skip:
            _run(f"spectra_{ch}", fig_mfdfa_spectra, df, ch)
    for cmp in COMPARISONS:
        _run(f"boxes_{cmp.key}", fig_feature_boxes, df, cmp_df, cmp.key)
    if "tables" not in skip:
        _run("auc_ranking", fig_auc_ranking, cmp_df)
        _run("classification", fig_classification, cls)
        _run("record_variance", fig_record_variance, icc)
    _run("dcca", fig_dcca, df)
    _run("entropy_residual", fig_entropy_boxes, df)
    _run("entropy_raw", fig_entropy_raw_boxes, df)
    _run("mse", fig_mse_curves, df)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the paper figures")
    ap.add_argument("--features", default=str(TABLES_DIR / "window_features.csv"))
    ap.add_argument("--skip", nargs="*", default=["spectra"],
                    help="figures to skip (default: spectra, which need the "
                         "1.5 GB recordings)")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.features)
    cmp_df = pd.read_csv(TABLES_DIR / "comparisons.csv")

    def _maybe(path, reader=pd.read_csv):
        try:
            return reader(path)
        except FileNotFoundError:
            return None

    cls = _maybe(TABLES_DIR / "classification.csv")
    icc = _maybe(TABLES_DIR / "record_variance.csv")
    quant = _maybe(TABLES_DIR / "quantization.csv")
    demo = None
    zoh_path = TABLES_DIR / "zoh_demo.npz"
    if zoh_path.exists():
        with np.load(zoh_path) as z:
            demo = {"stem": str(z["stem"]), "channel": str(z["channel"]),
                    "zoh": int(z["zoh"])}
            for tag in ("raw", "dec"):     # rebuild what fig_zoh expects
                demo[tag] = {"win": z[f"{tag}_win"], "F": z[f"{tag}_F"]}
    made = draw_all(df, cmp_df, cls, icc, quant, demo, skip=tuple(args.skip))
    print(f"{len(made)} figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
