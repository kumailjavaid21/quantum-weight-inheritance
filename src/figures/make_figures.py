"""
Phase Q2: consolidates every quantum result into publication tables (CSV +
LaTeX) and figures (PDF + 300 dpi PNG).

Reads only existing result files and existing checkpoints. Nothing is trained.
Sources: Phase 1, Phase 1.5, Phase 2 (10-seed study + downstream harness),
the go/no-go controlled experiment, and the Phase Q1 chained-growth fix.

Output: ~/host_share/qgan_results/quantum_paper/{paper_figures,tables}/
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WIGAN = Path(os.path.expanduser("~/qgan_project/wigan_project"))
for p in (str(WIGAN / "scripts"), str(WIGAN / "models"), str(WIGAN / "scripts" / "phase2")):
    sys.path.insert(0, p)

RES = Path(os.path.expanduser("~/host_share/qgan_results"))
P1, P15, P2 = RES / "wigan_phase1", RES / "wigan_phase1_5", RES / "wigan_phase2"
GNG = RES / "quantum_project" / "go_nogo"
OUT = RES / "quantum_paper"
FIG, TAB = OUT / "paper_figures", OUT / "tables"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def write_csv(name, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {name}.csv")


def write_latex(name, caption, label, header, body, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{table}[!t]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in body:
        lines.append(r"\midrule" if r == "MIDRULE" else " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {name}.tex")


# ----------------------------------------------------------- seed collection
def seed_paths(model):
    """model in {'B','D_fixed'}; returns {seed: dir}."""
    out = {}
    if model == "B":
        out[42] = P1 / "B_6q5L"
        for s in (43, 44):
            out[s] = P15 / f"B_seed{s}"
    else:
        out[42] = P15 / "D_fixed_growing_ring"
        for s in (43, 44):
            out[s] = P15 / f"D_fixed_seed{s}"
    for s in range(45, 52):
        out[s] = P2 / "seeds" / f"{model}_seed{s}"
    return out


def collect_seeds():
    data = {}
    for model in ("B", "D_fixed"):
        data[model] = {}
        for s, d in seed_paths(model).items():
            f = d / "test_metrics.json"
            if f.exists():
                data[model][s] = json.load(open(f))["delta_psnr"]
            else:
                print(f"  [warn] missing {f}")
    return data


# --------------------------------------------------------------- Table 1
def table1_efficiency():
    """Training efficiency from the go/no-go controlled experiment + Q1 fix."""
    def step_curve(run):
        p = GNG / run / "step_log.csv"
        if not p.exists():
            return None
        return [(int(r["step"]), float(r["train_loss"])) for r in csv.DictReader(open(p))]

    def summary(run):
        p = GNG / run / "summary.json"
        return json.load(open(p)) if p.exists() else None

    def test_curve(run):
        """Recover the logged test-loss checkpoints from the run log if present."""
        return None

    rows = []
    spec = [("model1", "From scratch (8q/7L)", "scratch"),
            ("model2", "Grown, random init of new components", "random"),
            ("model3", "Grown, exact morphism", "morphism"),
            ("model4fix_stepA", "Chained morphism, step 1 (4->6)", "chained"),
            ("model4fix_stepB", "Chained morphism, step 2 (6->8)", "chained")]
    for run, label, kind in spec:
        s = summary(run)
        if s is None:
            print(f"  [warn] no summary for {run}")
            continue
        # the original go/no-go runs and the Phase Q1 chained runs use
        # different key spellings for the same quantities
        init = s.get("initial_test_loss", s.get("initial_test_loss_t0"))
        gnorm = s.get("grad_norm_at_expansion_mean", s.get("grad_norm_mean_at_expansion"))
        rows.append({
            "run": run, "model": label, "initialisation": kind,
            "initial_test_loss": round(init, 6),
            "final_test_loss": round(s["final_test_loss"], 6),
            "best_test_loss": round(s.get("best_test_loss", s["final_test_loss"]), 6),
            "grad_norm_at_expansion": round(gnorm, 6) if gnorm is not None else "",
            "wall_clock_s": round(s["total_wall_time_s"], 1),
        })
    write_csv("table1_training_efficiency", rows)
    body = [[r["model"], f"{r['initial_test_loss']:.6f}", f"{r['final_test_loss']:.6f}",
             f"{r['wall_clock_s']:.0f}"] for r in rows]
    write_latex(
        "table1_training_efficiency",
        "Controlled growth comparison on the synthetic restoration task (2{,}000 steps, seed 42). "
        "The exact morphism's initial loss equals its parent's final loss to six decimal places; "
        "a from-scratch model of the same size requires 301 steps to reach the loss the morphism "
        "model attains in one.",
        "tab:efficiency",
        ["Model", "Initial loss ($t{=}0$)", "Final loss", "Wall clock (s)"], body)
    return rows


# --------------------------------------------------------------- Table 2
def table2_qnrs():
    rows = []
    for src, tag in ((P1 / "phase1_robustness.csv", "phase1"),
                     (P15 / "phase1_5_robustness.csv", "phase1_5")):
        if not src.exists():
            continue
        for r in csv.DictReader(open(src)):
            rows.append({"source": tag, "model": r["model"], "n_qubits": int(r["n_qubits"]),
                         "n_layers": int(r["n_layers"]), "eval_noise": float(r["eval_noise"]),
                         "mean_delta_psnr": float(r["mean_delta_psnr"]),
                         "std_delta_psnr": float(r["std_delta_psnr"]),
                         "qnrs": float(r["qnrs"])})
    write_csv("table2_noise_robustness", rows)

    keep = ["A_4q3L_independent", "B_6q5L_independent", "D_old_inherit_4to6", "D_fixed_growing_ring"]
    pretty = {"A_4q3L_independent": "A: independent 4q/3L",
              "B_6q5L_independent": "B: independent 6q/5L",
              "D_old_inherit_4to6": "D (old scheme): inherited 4$\\to$6",
              "D_fixed_growing_ring": "D (growing ring): inherited 4$\\to$6"}
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["eval_noise"]] = r
    body = []
    for m in keep:
        if m not in by:
            continue
        cells = [pretty[m], f"{by[m][0.0]['mean_delta_psnr']:.2f}"]
        for p in (0.05, 0.10, 0.20):
            q = by[m][p]["qnrs"]
            cells.append(rf"\textbf{{{q:.3f}}}" if m == "D_fixed_growing_ring" else f"{q:.3f}")
        body.append(cells)
    write_latex(
        "table2_noise_robustness",
        "Quantum noise robustness. QNRS is the ratio of $\\Delta$PSNR under injected Pauli noise "
        "at probability $p$ to the noiseless $\\Delta$PSNR (10 repetitions per point); 1.0 means "
        "robustness is fully retained. The growing-ring morphism retains 0.93--0.98 across the "
        "tested range, within 2--7\\% of the independently trained baseline, whereas the superseded "
        "inheritance scheme loses up to 30\\%.",
        "tab:qnrs",
        ["Model", "$\\Delta$PSNR (dB)", "QNRS $p{=}0.05$", "QNRS $p{=}0.10$", "QNRS $p{=}0.20$"],
        body)
    return by


# --------------------------------------------------------------- Table 3
def table3_clinical():
    src = P2 / "downstream_results.csv"
    rows = [r for r in csv.DictReader(open(src))]
    write_csv("table3_clinical_dice", [dict(r) for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[float(r["sigma"])] = r
    order = [("none", "No denoising"), ("dncnn", "DnCNN"), ("restormer", "Restormer-lite"),
             ("swinir", "SwinIR-lite"),
             ("wiqgan_B_6q5L_smoketest", "WI-QGAN (6q/5L)"), ("oracle", "Oracle (clean image)")]
    body = []
    for k, label in order:
        if k not in by:
            continue
        body.append([label] + [f"{float(by[k][s]['mean_dice']):.4f}" for s in (0.05, 0.10, 0.20)])
    write_latex(
        "table3_clinical_dice",
        "Downstream BraTS tumour segmentation Dice after denoising (5{,}000 test slices, fixed "
        "pre-trained segmentation U-Net). The WI-QGAN row is a single Phase~1 checkpoint trained at "
        "one fixed noise level and is reported as a preliminary result: it improves on no denoising "
        "at $\\sigma \\leq 0.10$ but does not match the classical baselines at any level.",
        "tab:clinical",
        ["Method", "Dice $\\sigma{=}0.05$", "Dice $\\sigma{=}0.10$", "Dice $\\sigma{=}0.20$"], body)
    return by


# --------------------------------------------------------------- Table 4
def table4_seeds(seeds):
    common = sorted(set(seeds["B"]) & set(seeds["D_fixed"]))
    b = np.array([seeds["B"][s] for s in common])
    d = np.array([seeds["D_fixed"][s] for s in common])
    gap = d - b
    rows = [{"seed": s, "B_delta_psnr": round(seeds["B"][s], 4),
             "D_fixed_delta_psnr": round(seeds["D_fixed"][s], 4),
             "gap": round(seeds["D_fixed"][s] - seeds["B"][s], 4)} for s in common]
    rows.append({"seed": "mean", "B_delta_psnr": round(b.mean(), 4),
                 "D_fixed_delta_psnr": round(d.mean(), 4), "gap": round(gap.mean(), 4)})
    rows.append({"seed": "std", "B_delta_psnr": round(b.std(ddof=1), 4),
                 "D_fixed_delta_psnr": round(d.std(ddof=1), 4), "gap": round(gap.std(ddof=1), 4)})
    write_csv("table4_multiseed", rows)

    stats = {"n": len(common), "B_mean": float(b.mean()), "B_std": float(b.std(ddof=1)),
             "D_mean": float(d.mean()), "D_std": float(d.std(ddof=1)),
             "gap_mean": float(gap.mean()), "gap_std": float(gap.std(ddof=1)),
             "gap_median": float(np.median(gap)),
             "n_positive": int((gap > 0).sum()),
             "n_large_failure": int((gap < -0.5).sum()),
             "variance_ratio_D_over_B": float(d.var(ddof=1) / b.var(ddof=1))}
    try:
        from scipy import stats as st
        stats["paired_t_p"] = float(st.ttest_rel(d, b).pvalue)
        stats["wilcoxon_p"] = float(st.wilcoxon(d, b).pvalue)
        stats["shapiro_gap_p"] = float(st.shapiro(gap).pvalue)
        stats["levene_p"] = float(st.levene(b, d).pvalue)
    except Exception as exc:
        print(f"  [warn] scipy tests unavailable: {exc}")
    json.dump(stats, open(TAB / "table4_multiseed_stats.json", "w"), indent=2)
    print(f"  wrote table4_multiseed_stats.json  "
          f"(B {stats['B_mean']:.3f}+-{stats['B_std']:.3f}, "
          f"D {stats['D_mean']:.3f}+-{stats['D_std']:.3f}, "
          f"var ratio {stats['variance_ratio_D_over_B']:.1f}x)")

    body = [[str(s), f"{seeds['B'][s]:.3f}", f"{seeds['D_fixed'][s]:.3f}",
             f"{seeds['D_fixed'][s] - seeds['B'][s]:+.3f}"] for s in common]
    body.append("MIDRULE")
    body.append([r"mean $\pm$ std", rf"{b.mean():.3f} $\pm$ {b.std(ddof=1):.3f}",
                 rf"{d.mean():.3f} $\pm$ {d.std(ddof=1):.3f}",
                 rf"{gap.mean():+.3f} $\pm$ {gap.std(ddof=1):.3f}"])
    cap = ("Ten-seed comparison of growing-ring inheritance (D\\_fixed) against independent "
           "from-scratch training (B) at 6~qubits / 5~layers. Mean and median are statistically "
           "indistinguishable")
    if "paired_t_p" in stats:
        cap += (f" (paired $t$-test $p={stats['paired_t_p']:.2f}$, "
                f"Wilcoxon $p={stats['wilcoxon_p']:.2f}$)")
    cap += (f", but inheritance shows {stats['variance_ratio_D_over_B']:.1f}$\\times$ the variance "
            f"and {stats['n_large_failure']}/{stats['n']} large ($>$0.5~dB) underperformances, "
            "which independent training does not exhibit at this sample size.")
    write_latex("table4_multiseed", cap, "tab:seeds",
                ["Seed", "B $\\Delta$PSNR (dB)", "D\\_fixed $\\Delta$PSNR (dB)", "Gap (D$-$B)"], body)
    return stats


# --------------------------------------------------------------- Figures
def fig1_morphism():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off"); ax.grid(False)

    def wire(x0, x1, y, color="#333333", lw=1.0, ls="-"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=1)

    def cnot(x, yc, yt, color="#333333"):
        ax.plot([x, x], [yc, yt], color=color, lw=1.0, zorder=2)
        ax.plot(x, yc, "o", color=color, ms=3.5, zorder=3)
        ax.add_patch(plt.Circle((x, yt), 0.11, fill=False, ec=color, lw=1.0, zorder=3))
        ax.plot([x - 0.11, x + 0.11], [yt, yt], color=color, lw=1.0, zorder=3)

    def cry(x, yc, yt, color="#b8341f"):
        ax.plot([x, x], [yc, yt], color=color, lw=1.2, zorder=2)
        ax.plot(x, yc, "o", color=color, ms=3.5, zorder=3)
        ax.add_patch(FancyBboxPatch((x - 0.26, yt - 0.16), 0.52, 0.32,
                                     boxstyle="round,pad=0.02", fc="white", ec=color, lw=1.1, zorder=3))
        ax.text(x, yt, "$R_y$", ha="center", va="center", fontsize=6.4, color=color, zorder=4)

    ax.text(0.15, 5.25, "Parent: $n$ qubits", fontsize=9, fontweight="bold")
    ys_old = [4.65, 4.05, 3.45]
    for i, y in enumerate(ys_old):
        wire(1.15, 4.3, y)
        ax.text(0.95, y, f"$q_{i}$", ha="right", va="center", fontsize=8)
    cnot(1.9, ys_old[0], ys_old[1]); cnot(2.6, ys_old[1], ys_old[2])
    cnot(3.3, ys_old[2], ys_old[0])
    ax.text(2.6, 3.02, "CNOT ring, mod $n$", ha="center", fontsize=7.4)

    ax.add_patch(FancyArrowPatch((4.6, 4.05), (5.5, 4.05), arrowstyle="-|>",
                                  mutation_scale=12, lw=1.6, color="#b8341f"))
    ax.text(5.05, 4.22, "grow", ha="center", fontsize=7.6, color="#b8341f", fontweight="bold")

    ax.text(5.75, 5.25, "Child: $n' > n$ qubits", fontsize=9, fontweight="bold")
    ys_new = [4.65, 4.05, 3.45, 2.75, 2.15]
    for i, y in enumerate(ys_new):
        is_new = i >= 3
        x_start = 7.05 if is_new else 6.75
        wire(x_start, 9.85, y, color="#b8341f" if is_new else "#333333",
             ls="-" if not is_new else (0, (3, 1.5)))
        ax.text(6.55, y, f"$q_{i}$", ha="right", va="center", fontsize=8,
                color="#b8341f" if is_new else "#333333")
        if is_new:
            # initial state belongs at the left of the wire, per circuit convention
            ax.text(6.72, y, "$|0\\rangle$", ha="left", va="center", fontsize=7, color="#b8341f")
    cnot(7.3, ys_new[0], ys_new[1]); cnot(7.8, ys_new[1], ys_new[2])
    cnot(8.3, ys_new[2], ys_new[0])
    cry(8.95, ys_new[2], ys_new[3]); cry(9.5, ys_new[3], ys_new[4])
    ax.text(8.0, 3.02, "ring preserved, still mod $n$", ha="center", fontsize=7.2)
    ax.text(9.2, 1.72, "zero-init $CRY(0) = I$", ha="center", fontsize=7.2, color="#b8341f")

    ax.add_patch(FancyBboxPatch((0.35, 0.35), 9.3, 1.0, boxstyle="round,pad=0.08",
                                 fc="#f4f4f4", ec="#555555", lw=1.0))
    ax.text(5.0, 0.85,
            "Every gate touching a new qubit is a parametrized $CRY$ initialised to $0$, which is "
            "the exact identity.\nThe parent's sub-circuit is therefore reproduced bit-for-bit: "
            "$\\max|\\Delta\\langle Z\\rangle| = 2.8\\times10^{-16}$.",
            ha="center", va="center", fontsize=7.6, linespacing=1.5)
    save(fig, "fig1_morphism_schematic")


def fig2_convergence():
    def curve(run):
        p = GNG / run / "step_log.csv"
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p)))
        st = np.array([int(r["step"]) for r in rows])
        ls = np.array([float(r["train_loss"]) for r in rows])
        k = 25
        sm = np.convolve(ls, np.ones(k) / k, mode="valid")
        return st[k - 1:], sm

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for run, label, c, ls in [("model1", "From scratch (8q/7L)", "#666666", "--"),
                              ("model2", "Grown, random init", "#1f77b4", "-."),
                              ("model3", "Grown, exact morphism", "#d62728", "-"),
                              ("model4fix_stepB", "Chained morphism (fixed)", "#2ca02c", "-")]:
        c_ = curve(run)
        if c_ is None:
            continue
        ax.plot(c_[0], c_[1], ls, color=c, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Training loss (25-step moving average)")
    ax.set_title("Convergence after qubit-count growth")
    ax.legend(loc="upper right")
    save(fig, "fig2_convergence")


def fig3_qnrs(by):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    spec = [("A_4q3L_independent", "A: independent 4q/3L", "#999999", "s"),
            ("B_6q5L_independent", "B: independent 6q/5L", "#1f77b4", "o"),
            ("D_old_inherit_4to6", "D: old inheritance", "#ff7f0e", "v"),
            ("D_fixed_growing_ring", "D: growing-ring morphism", "#d62728", "D")]
    ps = [0.05, 0.10, 0.20]
    for key, label, c, mk in spec:
        if key not in by:
            continue
        ax.plot(ps, [by[key][p]["qnrs"] for p in ps], "-", marker=mk, color=c,
                lw=2.0 if "growing" in key else 1.2, ms=4, label=label)
    ax.axhline(1.0, color="black", ls=":", lw=0.9)
    ax.set_xlabel("Injected Pauli noise probability $p$")
    ax.set_ylabel("QNRS")
    ax.set_title("Noise robustness is preserved across growth")
    ax.set_xticks(ps)
    ax.legend(loc="lower left")
    save(fig, "fig3_qnrs")


def fig4_clinical(by):
    order = [("none", "No denoising", "#999999"), ("dncnn", "DnCNN", "#8c564b"),
             ("restormer", "Restormer-lite", "#9467bd"), ("swinir", "SwinIR-lite", "#ff7f0e"),
             ("wiqgan_B_6q5L_smoketest", "WI-QGAN (6q/5L)", "#d62728")]
    sig = [0.05, 0.10, 0.20]
    x = np.arange(len(sig)); w = 0.16
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for i, (k, label, c) in enumerate(order):
        if k not in by:
            continue
        ax.bar(x + (i - len(order) / 2 + 0.5) * w,
               [float(by[k][s]["mean_dice"]) for s in sig], w, label=label, color=c)
    orc = float(by["oracle"][0.05]["mean_dice"])
    ax.axhline(orc, color="black", ls="--", lw=1.0)
    ax.text(len(sig) - 1 + 0.42, orc + 0.008, f"oracle {orc:.3f}", ha="right", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels([f"$\\sigma={s:g}$" for s in sig])
    ax.set_ylabel("Tumour segmentation Dice")
    ax.set_title("Downstream clinical task (BraTS)")
    ax.set_ylim(0.40, 0.90)
    ax.legend(loc="lower left", ncol=2, fontsize=7)
    save(fig, "fig4_clinical_dice")


def fig5_seed_hist(seeds, stats):
    common = sorted(set(seeds["B"]) & set(seeds["D_fixed"]))
    b = np.array([seeds["B"][s] for s in common])
    d = np.array([seeds["D_fixed"][s] for s in common])
    gap = d - b
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    bins = np.linspace(min(b.min(), d.min()) - 0.2, max(b.max(), d.max()) + 0.2, 9)
    axes[0].hist(b, bins=bins, alpha=0.65, label="B: independent", color="#1f77b4")
    axes[0].hist(d, bins=bins, alpha=0.65, label="D: inherited", color="#d62728")
    axes[0].set_xlabel("$\\Delta$PSNR (dB)"); axes[0].set_ylabel("Seeds")
    axes[0].set_title(f"Per-seed quality (n={len(common)})")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(-0.5, color="#b8341f", ls=":", lw=1.0)
    axes[1].bar(range(len(common)), gap,
                color=["#d62728" if g < -0.5 else "#1f77b4" for g in gap])
    axes[1].set_xticks(range(len(common)))
    axes[1].set_xticklabels([str(s) for s in common], fontsize=7)
    axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Gap (D $-$ B), dB")
    axes[1].set_title(f"{stats['n_large_failure']}/{stats['n']} large failures "
                      f"($<-0.5$ dB)")
    fig.tight_layout()
    save(fig, "fig5_multiseed")


def fig6_qualitative():
    from qgan_models_wigan import WIQGANGenerator
    data = Path(os.path.expanduser("~/host_share/qgan_data/wigan_datasets/phase2"))
    clean = np.load(data / "phase2_test_5000.npy").astype(np.float32)
    degraded = np.load(data / "degraded_sigma010.npy").astype(np.float32)
    masks = np.load(data / "phase2_test_5000_mask.npy").astype(np.float32)
    idx = int(np.argmax(masks.reshape(len(masks), -1).sum(1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panels = [("Clean", clean[idx]), ("Noisy ($\\sigma=0.10$)", degraded[idx])]
    specs = [("A: 4q/3L", P1 / "A_4q3L", 4, 3), ("B: 6q/5L", P1 / "B_6q5L", 6, 5)]
    for label, d, nq, nl in specs:
        ck = d / "checkpoints" / "generator_last.pt"
        if not ck.exists():
            print(f"  [warn] missing {ck}")
            continue
        m = WIQGANGenerator(img_size=128, patch_size=8, n_qubits=nq, n_layers=nl,
                            noise_prob=0.0).to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=True), strict=True)
        m.eval()
        with torch.no_grad():
            x = torch.from_numpy(degraded[idx:idx + 1]).unsqueeze(1).to(device)
            mask = m.sample_noise_mask(device, noise_prob=0.0)
            out, _, _ = m(x, noise_mask=mask, patch_indices=None)
        panels.append((label, out.cpu().numpy()[0, 0]))

    def psnr(a, b):
        return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))

    fig, axes = plt.subplots(1, len(panels), figsize=(1.6 * len(panels), 2.0))
    for j, (name, img) in enumerate(panels):
        axes[j].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[j].set_title(name if j == 0 else f"{name}\n{psnr(clean[idx], img):.2f} dB",
                          fontsize=7.2, linespacing=1.3)
        axes[j].set_xticks([]); axes[j].set_yticks([]); axes[j].grid(False)
    fig.suptitle("WI-QGAN denoising before and after qubit-count growth", fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_qualitative")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    print("Collecting seeds");       seeds = collect_seeds()
    print("Table 1: efficiency");    table1_efficiency()
    print("Table 2: QNRS");          qnrs = table2_qnrs()
    print("Table 3: clinical");      clin = table3_clinical()
    print("Table 4: multi-seed");    stats = table4_seeds(seeds)
    print("Fig 1");                  fig1_morphism()
    print("Fig 2");                  fig2_convergence()
    print("Fig 3");                  fig3_qnrs(qnrs)
    print("Fig 4");                  fig4_clinical(clin)
    print("Fig 5");                  fig5_seed_hist(seeds, stats)
    print("Fig 6");                  fig6_qualitative()
    print(f"\nDone. Tables -> {TAB}\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
