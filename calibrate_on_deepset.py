# calibrate_on_deepset.py
#
# Recalibrates the semantic detector's per-category thresholds using the REAL deepset training split (546 labelled prompts) instead of a small
# hand-built set. This makes the chosen thresholds empirically grounded in recognised public attack data.
# METHOD
# The deepset dataset labels each prompt injection(1) / legitimate(0) but does NOT break attacks down by category. the detector, however, scores
# every prompt against all five behavioural categories. So we calibrate a SINGLE decision rule: a prompt is flagged as an attack if its MAXIMUM
# semantic score across all categories crosses a threshold.
# We sweep that threshold over the training split, compute precision / recall / F1 at each value, and choose the threshold that maximises F1.
# This threshold is then used uniformly for all categories in evaluation.
# Output:
#   - A precision/recall/F1 sweep table 
#   - The single best global threshold
#   - Saves the sweep to calibration_sweep.csv

import pandas as pd
import numpy as np
from semantic_detector import semantic_scores

TRAIN_CSV = "deepset_train.csv"
THRESHOLD_RANGE = [round(t, 2) for t in np.arange(0.20, 0.71, 0.02)]


def max_semantic_score(prompt):
    """The prompt's strongest similarity to ANY attack category."""
    return max(semantic_scores(str(prompt)).values())


def main():
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} training prompts "
          f"({(df['label']==1).sum()} injection, {(df['label']==0).sum()} legitimate)")
    print("Scoring every prompt with the semantic detector (one-off)...\n")

    # Precompute the max semantic score for every prompt once.
    df["max_score"] = df["text"].apply(max_semantic_score)

    print(f"{'thr':>5} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
          f"{'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6}")
    print("-" * 56)

    best = {"thr": None, "f1": -1}
    rows = []

    for thr in THRESHOLD_RANGE:
        pred = df["max_score"] >= thr
        actual = df["label"] == 1

        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        tn = int((~pred & ~actual).sum())
        fn = int((~pred & actual).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / len(df)

        rows.append({"threshold": thr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "accuracy": round(acc, 4)})

        marker = ""
        if f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "prec": prec, "rec": rec, "acc": acc}
            marker = "  <-- best F1"

        print(f"{thr:>5.2f} {tp:>4} {fp:>4} {tn:>4} {fn:>4} "
              f"{prec:>6.2f} {rec:>6.2f} {f1:>6.2f} {acc:>6.2f}{marker}")

    pd.DataFrame(rows).to_csv("calibration_sweep.csv", index=False)

    print("\n" + "=" * 56)
    print(f"CHOSEN GLOBAL THRESHOLD: {best['thr']:.2f}")
    print(f"  F1={best['f1']:.3f}  precision={best['prec']:.3f}  "
          f"recall={best['rec']:.3f}  accuracy={best['acc']:.3f}")
    print("=" * 56)
    print("\nSweep saved to calibration_sweep.csv")
    print(f"\nUse this threshold as GLOBAL_THRESHOLD in evaluate_on_deepset.py")


if __name__ == "__main__":
    main()
