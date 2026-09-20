# Figures produced:
#   1. calibration_curve.png    — precision/recall/F1 vs threshold
#   2. confusion_matrix.png     — end-to-end policy engine confusion matrix
#   3. detection_paths.png      — which layer caught each attack
#   4. policy_attribution.png   — how many times each policy fired
#   5. baseline_comparison.png  — unprotected vs regex vs policy framework
#
# Usage:  python generate_charts.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Consistent, print-friendly style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
AMBER = "#e9c46a"
RUST = "#e76f51"
GREY = "#8d99ae"


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# 
# 1. CALIBRATION CURVE
# 
def calibration_curve():
    df = pd.read_csv("calibration_sweep.csv").sort_values("threshold")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["threshold"], df["precision"], marker="o", ms=3, color=NAVY, label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="s", ms=3, color=RUST, label="Recall")
    ax.plot(df["threshold"], df["f1"], marker="^", ms=3, color=TEAL, label="F1 score")

    # Mark the chosen operating point (max F1)
    best = df.loc[df["f1"].idxmax()]
    ax.axvline(best["threshold"], color=GREY, ls="--", lw=1)
    ax.annotate(f"chosen threshold = {best['threshold']:.2f}\nF1 = {best['f1']:.2f}",
                xy=(best["threshold"], best["f1"]),
                xytext=(best["threshold"] + 0.04, best["f1"] + 0.12),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color=GREY))

    ax.set_xlabel("Semantic similarity threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold calibration on deepset training split")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    save(fig, "calibration_curve.png")


# 
# 2. CONFUSION MATRIX (end-to-end policy engine)
# 
def confusion_matrix():
    df = pd.read_csv("policy_engine_eval_results.csv")
    actual = df["actual"] == 1
    pred = df["predicted"] == 1
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())

    matrix = [[tp, fn], [fp, tn]]
    labels = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], ["Predicted\nAttack (DENY)", "Predicted\nBenign (ALLOW)"])
    ax.set_yticks([0, 1], ["Actual\nAttack", "Actual\nBenign"])

    vmax = max(max(r) for r in matrix)
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            colour = "white" if val > vmax / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center",
                    color=colour, fontsize=13, fontweight="bold")

    ax.set_title("End-to-end policy enforcement: confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    save(fig, "confusion_matrix.png")


# 
# 3. DETECTION PATHS (which layer caught attacks)
# 
def detection_paths():
    df = pd.read_csv("deepset_eval_results.csv")
    caught = df[(df["actual"] == 1) & (df["predicted"] == 1)]
    counts = caught["path"].value_counts()
    order = ["keyword", "semantic", "both"]
    values = [int(counts.get(k, 0)) for k in order]
    colours = [GREY, TEAL, NAVY]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(order, values, color=colours)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attacks correctly detected")
    ax.set_xlabel("Detection path")
    ax.set_title("Which layer caught each attack (held-out test set)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Keyword\nonly", "Semantic\nonly", "Both"])
    save(fig, "detection_paths.png")


# 
# 4. POLICY ATTRIBUTION
# 
def policy_attribution():
    df = pd.read_csv("policy_engine_eval_results.csv")
    counts = {}
    for cell in df["policies_fired"].dropna():
        if cell == "none":
            continue
        for pid in str(cell).split(";"):
            pid = pid.strip()
            if pid and pid != "none":
                counts[pid] = counts.get(pid, 0) + 1
    if not counts:
        print("  (policy_attribution: no policies fired, skipping)")
        return
    items = sorted(counts.items())
    ids = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(ids, vals, color=NAVY)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Times fired across test set")
    ax.set_xlabel("Policy")
    ax.set_title("Policy attribution: enforcement distributed across the policy set")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "policy_attribution.png")


# 
# 5. BASELINE COMPARISON
# 
def baseline_comparison():
    # Prefer the rigorous deepset held-out comparison if present; fall back
    # to the older 16-prompt controlled runs otherwise.
    import os
    if os.path.exists("baseline_summary_deepset.csv"):
        s = pd.read_csv("baseline_summary_deepset.csv").set_index("system")
        systems = ["Unprotected", "Regex guardrail", "Policy framework"]
        # Plot recall (proportion of real attacks blocked) on held-out data.
        values = [float(s.loc[name, "recall"]) for name in systems]
        colours = [RUST, AMBER, TEAL]
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        bars = ax.bar(systems, values, color=colours)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Recall — proportion of real attacks blocked")
        ax.set_title("Attack detection on deepset held-out test set (116 prompts)")
        ax.set_ylim(0, 1.0)
        save(fig, "baseline_comparison.png")
        return

    # ── Fallback: original 16-prompt controlled runs ──
    def blocked_count(path):
        df = pd.read_csv(path)
        d = df["Decision"].astype(str).str.lower()
        return int(d.str.contains("block").sum() + d.str.contains("deny").sum())

    try:
        unprotected = blocked_count("unprotected_results.csv")
        regex = blocked_count("regex_results.csv")
        framework = blocked_count("experiment_results.csv")
    except Exception as e:
        print(f"  (baseline_comparison skipped: {e})")
        return

    systems = ["Unprotected", "Regex guardrail", "Policy framework"]
    values = [unprotected, regex, framework]
    colours = [RUST, AMBER, TEAL]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(systems, values, color=colours)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attacks blocked (of 16 controlled prompts)")
    ax.set_title("Baseline comparison: attacks blocked by system")
    ax.set_ylim(0, max(values) * 1.2 + 1)
    save(fig, "baseline_comparison.png")


if __name__ == "__main__":
    print("Generating figures from result CSVs ...")
    calibration_curve()
    confusion_matrix()
    detection_paths()
    policy_attribution()
    baseline_comparison()
    print(f"\nAll figures written to {OUTDIR}/")
    print("Re-run this script any time to regenerate them from the CSVs.")
