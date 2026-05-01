import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_KEYS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
RELP_C = "#2563eb"
CAA_C = "#dc2626"

# ── Quality data (from scores_out.txt) ──
# Keys must match the model field in the JSON files
QUALITY = {
    "meta-llama/Llama-3.2-1B": {
        "relp_q": [0.6897, 0.679, 0.6542, 0.6394, 0.6579],
        "caa_q":  [0.6579, 0.7054, 0.7187, 0.7396, 0.6966],
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "relp_q": [0.9707, 0.9723, 0.9735, 0.9748, 0.9746],
        "caa_q":  [0.9746, 0.9793, 0.9786, 0.7066, 0.5535],
    },
    "meta-llama/Llama-3.2-3B": {
        "relp_q": [0.7703, 0.7648, 0.7606, 0.7454, 0.7576],
        "caa_q":  [0.7576, 0.7896, 0.7857, 0.7712, 0.6693],
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "relp_q": [0.9754, 0.9743, 0.9769, 0.9765, 0.9769],
        "caa_q":  [0.9769, 0.9758, 0.9371, 0.4193, 0.4307],
    },
    "meta-llama/Llama-3.1-8B": {
        "relp_q": [0.7509, 0.7587, 0.726, 0.7332, 0.7291],
        "caa_q":  [0.7291, 0.7407, 0.749, 0.7531, 0.7292],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "relp_q": [0.9726, 0.9688, 0.9697, 0.9677, 0.9691],
        "caa_q":  [0.9691, 0.9636, 0.9161, 0.7809, 0.4929],
    },
    "meta-llama/Llama-3.1-70B": {
        "relp_q": [0.7114, 0.7345, 0.78, 0.8129, 0.8175],
        "caa_q":  [0.8175, 0.8252, 0.8257, 0.8167, 0.6562],
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "relp_q": [0.9715, 0.9724, 0.9756, 0.9788, 0.9805],
        "caa_q":  [0.9805, 0.9744, 0.9612, 0.7921, 0.5689],
    },
    "Qwen/Qwen2.5-1.5B": {
        "relp_q": [0.8541, 0.8476, 0.8502, 0.8884, 0.9064],
        "caa_q":  [0.9064, 0.9015, 0.8974, 0.8702, 0.8072],
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "relp_q": [0.9836, 0.9824, 0.9821, 0.9817, 0.9815],
        "caa_q":  [0.9815, 0.9779, 0.9589, 0.9275, 0.8876],
    },
    "Qwen/Qwen2.5-3B": {
        "relp_q": [0.812, 0.8389, 0.8397, 0.8548, 0.8651],
        "caa_q":  [0.8651, 0.8592, 0.8634, 0.8532, 0.812],
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "relp_q": [0.9786, 0.9799, 0.9787, 0.9829, 0.9841],
        "caa_q":  [0.9841, 0.9841, 0.9784, 0.8952, 0.8439],
    },
    "Qwen/Qwen2.5-7B": {
        "relp_q": [0.9152, 0.9227, 0.8899, 0.906, 0.9189],
        "caa_q":  [0.9189, 0.9182, 0.9293, 0.8894, 0.6898],
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "relp_q": [0.9781, 0.981, 0.9795, 0.9798, 0.9797],
        "caa_q":  [0.9797, 0.9795, 0.9685, 0.6051, 0.4137],
    },
    "Qwen/Qwen2.5-72B": {
        "relp_q": [0.8856, 0.9188, 0.9437, 0.9551, 0.9622],
        "caa_q":  [0.9622, 0.9591, 0.9625, 0.9666, 0.8904],
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "relp_q": [0.9772, 0.9803, 0.9806, 0.9798, 0.9826],
        "caa_q":  [0.9826, 0.9818, 0.9747, 0.771, 0.4062],
    },
}


def load_steering_data(results_dir="experiments/results"):
    """Load all steering_comparison*.json files and merge with quality data."""
    pattern = os.path.join(results_dir, "steering_comparison*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    all_models = {}
    for fpath in files:
        with open(fpath) as f:
            j = json.load(f)
        model_name = j["model"]

        # Extract refusal pct per alpha for each method
        relp_r = [j["methods"]["relp"][ak]["pct"] for ak in ALPHA_KEYS]
        caa_r = [j["methods"]["caa"][ak]["pct"] for ak in ALPHA_KEYS]

        # Look up quality
        if model_name not in QUALITY:
            print(f"WARNING: no quality data for {model_name}, skipping")
            continue

        q = QUALITY[model_name]

        # Determine if base or instruct
        is_instruct = model_name.lower().endswith("instruct")

        # Short display name
        short = model_name.split("/")[-1]

        all_models[short] = {
            "full_name": model_name,
            "is_instruct": is_instruct,
            "relp_r": relp_r,
            "caa_r": caa_r,
            "relp_q": q["relp_q"],
            "caa_q": q["caa_q"],
        }

    print(f"Loaded {len(all_models)} models from {len(files)} files")
    return all_models


def split_base_instruct(all_models):
    instruct = {k: v for k, v in all_models.items() if v["is_instruct"]}
    base = {k: v for k, v in all_models.items() if not v["is_instruct"]}
    return base, instruct


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Aggregate instruct — refusal & quality vs alpha
# ═══════════════════════════════════════════════════════════
def fig1_aggregate(instruct, outdir):
    models = list(instruct.keys())
    n = len(models)
    avg = lambda key, i: np.mean([instruct[m][key][i] for m in models])
    std = lambda key, i: np.std([instruct[m][key][i] for m in models])

    avg_relp_r = [100 - avg("relp_r", i) for i in range(5)]
    avg_relp_q = [avg("relp_q", i) for i in range(5)]
    avg_caa_r  = [avg("caa_r", i) for i in range(5)]
    avg_caa_q  = [avg("caa_q", i) for i in range(5)]
    std_relp_r = [std("relp_r", i) for i in range(5)]
    std_relp_q = [std("relp_q", i) for i in range(5)]
    std_caa_r  = [std("caa_r", i) for i in range(5)]
    std_caa_q  = [std("caa_q", i) for i in range(5)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    a = np.array(ALPHAS)

    ax1.plot(a, avg_relp_r, "o-", color=RELP_C, ms=5, lw=1.8, label="CNA", zorder=3)
    ax1.fill_between(a, np.array(avg_relp_r) - np.array(std_relp_r),
                        np.array(avg_relp_r) + np.array(std_relp_r), alpha=0.15, color=RELP_C)
    ax1.plot(a, avg_caa_r, "s--", color=CAA_C, ms=5, lw=1.8, label="CAA", zorder=3)
    ax1.fill_between(a, np.array(avg_caa_r) - np.array(std_caa_r),
                        np.array(avg_caa_r) + np.array(std_caa_r), alpha=0.15, color=CAA_C)
    ax1.set_xlabel(r"Steering strength $\alpha$")
    ax1.set_ylabel("Refusal rate (%)")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(ALPHAS)
    ax1.legend(frameon=False)
    ax1.set_title("(a) Refusal rate")

    ax2.plot(a, avg_relp_q, "o-", color=RELP_C, ms=5, lw=1.8, label="CNA", zorder=3)
    ax2.fill_between(a, np.array(avg_relp_q) - np.array(std_relp_q),
                        np.array(avg_relp_q) + np.array(std_relp_q), alpha=0.15, color=RELP_C)
    ax2.plot(a, avg_caa_q, "s--", color=CAA_C, ms=5, lw=1.8, label="CAA", zorder=3)
    ax2.fill_between(a, np.array(avg_caa_q) - np.array(std_caa_q),
                        np.array(avg_caa_q) + np.array(std_caa_q), alpha=0.15, color=CAA_C)
    ax2.set_xlabel(r"Steering strength $\alpha$")
    ax2.set_ylabel("Generation quality")
    ax2.set_ylim(0.35, 1.02)
    ax2.set_xticks(ALPHAS)
    ax2.legend(frameon=False)
    ax2.set_title("(b) Generation quality")

    fig.suptitle(f"Aggregate across {n} instruct models (mean ± 1 s.d.)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_aggregate.pdf"))
    fig.savefig(os.path.join(outdir, "fig1_aggregate.png"))
    plt.close()
    print("fig1 done")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Per-model small multiples — refusal rate
# ═══════════════════════════════════════════════════════════
def fig2_small_multiples_refusal(instruct, outdir):
    models = list(instruct.keys())
    ncols = 4
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 1.9 * nrows), sharex=True, sharey=True)
    axes_flat = np.array(axes).flat
    for idx, ax in enumerate(axes_flat):
        if idx >= len(models):
            ax.set_visible(False)
            continue
        m = models[idx]
        d = instruct[m]
        ax.plot(ALPHAS, d["relp_r"], "o-", color=RELP_C, ms=3.5, lw=1.4, label="CNA")
        ax.plot(ALPHAS, d["caa_r"], "s--", color=CAA_C, ms=3.5, lw=1.4, label="CAA")
        ax.set_title(m, fontsize=7.5, pad=3)
        ax.set_ylim(-5, 105)
        ax.set_xticks(ALPHAS)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$\alpha$", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Refusal %", fontsize=8)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.suptitle("Refusal rate vs. steering strength (instruct models)", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_refusal_per_model.pdf"))
    fig.savefig(os.path.join(outdir, "fig2_refusal_per_model.png"))
    plt.close()
    print("fig2 done")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Per-model small multiples — quality
# ═══════════════════════════════════════════════════════════
def fig3_small_multiples_quality(instruct, outdir):
    models = list(instruct.keys())
    ncols = 4
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 1.9 * nrows), sharex=True, sharey=True)
    axes_flat = np.array(axes).flat
    for idx, ax in enumerate(axes_flat):
        if idx >= len(models):
            ax.set_visible(False)
            continue
        m = models[idx]
        d = instruct[m]
        ax.plot(ALPHAS, d["relp_q"], "o-", color=RELP_C, ms=3.5, lw=1.4, label="RelP")
        ax.plot(ALPHAS, d["caa_q"], "s--", color=CAA_C, ms=3.5, lw=1.4, label="CAA")
        ax.set_title(m, fontsize=7.5, pad=3)
        ax.set_ylim(0.35, 1.02)
        ax.set_xticks(ALPHAS)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$\alpha$", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Quality", fontsize=8)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.suptitle("Generation quality vs. steering strength (instruct models)", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_quality_per_model.pdf"))
    fig.savefig(os.path.join(outdir, "fig3_quality_per_model.png"))
    plt.close()
    print("fig3 done")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Pareto tradeoff scatter — quality vs refusal
# ═══════════════════════════════════════════════════════════
def fig4_pareto(instruct, outdir):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    for m in instruct:
        d = instruct[m]
        ax.scatter(d["relp_r"], d["relp_q"], c=RELP_C, s=18, alpha=0.5, zorder=3, edgecolors="none")
        ax.scatter(d["caa_r"], d["caa_q"], c=CAA_C, s=18, alpha=0.5, zorder=3, edgecolors="none")
    ax.scatter([], [], c=RELP_C, s=30, label="RelP", edgecolors="none")
    ax.scatter([], [], c=CAA_C, s=30, label="CAA", edgecolors="none")
    ax.set_xlabel("Refusal rate (%)")
    ax.set_ylabel("Generation quality")
    ax.set_xlim(-5, 105)
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False)
    ax.set_title(r"Quality–refusal tradeoff (all instruct models, all $\alpha$)")
    ax.annotate("Ideal:\nlow refusal,\nhigh quality", xy=(10, 0.98), fontsize=7,
                color="gray", ha="left", va="top", style="italic")
    ax.axhline(0.95, color="gray", ls=":", lw=0.6, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_pareto.pdf"))
    fig.savefig(os.path.join(outdir, "fig4_pareto.png"))
    plt.close()
    print("fig4 done")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Bar chart — quality at max steering for each model
# ═══════════════════════════════════════════════════════════
def fig5_max_steering_bars(instruct, outdir):
    models = list(instruct.keys())
    relp_q_max = [instruct[m]["relp_q"][4] for m in models]  # alpha=1.0
    caa_q_max = [instruct[m]["caa_q"][4] for m in models]    # alpha=1.0
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(x - w / 2, relp_q_max, w, color=RELP_C, alpha=0.85, label=r"RelP ($\alpha$=1.0)")
    ax.bar(x + w / 2, caa_q_max, w, color=CAA_C, alpha=0.85, label=r"CAA ($\alpha$=1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Generation quality")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, ls=":", color="gray", lw=0.7, alpha=0.6)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(r"Generation quality at maximum steering strength ($\alpha$ = 1.0)", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_max_steering_quality.pdf"))
    fig.savefig(os.path.join(outdir, "fig5_max_steering_quality.png"))
    plt.close()
    print("fig5 done")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Base vs Instruct comparison
# ═══════════════════════════════════════════════════════════
def fig6_base_vs_instruct(base, instruct, outdir):
    def avg_across(models, key, i):
        return np.mean([models[m][key][i] for m in models])

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True)
    a = ALPHAS

    for col, (label, group) in enumerate([("Base models", base), ("Instruct models", instruct)]):
        ar = [avg_across(group, "relp_r", i) for i in range(5)]
        ac = [avg_across(group, "caa_r", i) for i in range(5)]
        aq_r = [avg_across(group, "relp_q", i) for i in range(5)]
        aq_c = [avg_across(group, "caa_q", i) for i in range(5)]

        axes[0, col].plot(a, ar, "o-", color=RELP_C, ms=4, lw=1.5, label="RelP")
        axes[0, col].plot(a, ac, "s--", color=CAA_C, ms=4, lw=1.5, label="CAA")
        axes[0, col].set_title(f"{label} — refusal", fontsize=9)
        axes[0, col].set_ylim(-5, 105)
        axes[0, col].legend(frameon=False, fontsize=7)

        axes[1, col].plot(a, aq_r, "o-", color=RELP_C, ms=4, lw=1.5, label="RelP")
        axes[1, col].plot(a, aq_c, "s--", color=CAA_C, ms=4, lw=1.5, label="CAA")
        axes[1, col].set_xlabel(r"$\alpha$")
        axes[1, col].set_title(f"{label} — quality", fontsize=9)
        axes[1, col].legend(frameon=False, fontsize=7)

    axes[0, 0].set_ylabel("Refusal %")
    axes[1, 0].set_ylabel("Quality")
    axes[1, 0].set_ylim(0.55, 1.02)
    axes[1, 1].set_ylim(0.35, 1.02)
    for ax in axes.flat:
        ax.set_xticks(ALPHAS)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_base_vs_instruct.pdf"))
    fig.savefig(os.path.join(outdir, "fig6_base_vs_instruct.png"))
    plt.close()
    print("fig6 done")

# CSV tables
    import csv
    for label, group in [("instruct", instruct), ("base", base)]:
        path = os.path.join(args.outdir, f"table_{label}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Model", "Baseline Ref%", "RelP Ref%", "RelP Qual", "CAA Ref%", "CAA Qual"])
            for m in group:
                d = group[m]
                # baseline: relp α=0.0 and caa α=0.0 should match (both = no intervention)
                baseline = d["relp_r"][0]
                w.writerow([m, baseline, d["relp_r"][4], d["relp_q"][4], d["caa_r"][4], d["caa_q"][4]])
        print(f"Saved {path}")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing steering_comparison*.json files")
    parser.add_argument("--outdir", default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_models = load_steering_data(args.results_dir)
    base, instruct = split_base_instruct(all_models)

    print(f"  Base models: {list(base.keys())}")
    print(f"  Instruct models: {list(instruct.keys())}")

    fig1_aggregate(instruct, args.outdir)
    fig2_small_multiples_refusal(instruct, args.outdir)
    fig3_small_multiples_quality(instruct, args.outdir)
    fig4_pareto(instruct, args.outdir)
    fig5_max_steering_bars(instruct, args.outdir)
    fig6_base_vs_instruct(base, instruct, args.outdir)
    print(f"All figures saved to {args.outdir}/")