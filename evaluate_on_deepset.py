# evaluate_on_deepset.py
#
# Evaluates the Layer 1 detector on the HELD-OUT deepset test split (116 prompts the calibration never saw) and reports the standard
# classification metrics: confusion matrix, precision,recall, F1, accuracy.
# It also reports, for the attacks that were correctly caught, how many were caught by the KEYWORD path, the SEMANTIC path, or BOTH — which
# quantifies exactly what the semantic layer adds over keyword matching.
#
# IMPORTANT:
#   Thresholds were calibrated on the TRAIN split (calibrate_on_deepset.py).
#   This script uses that fixed threshold on the untouched TEST split, so
#   the reported numbers are a fair, non-optimistic estimate of real performance.
# I set GLOBAL_THRESHOLD below to the value calibrate_on_deepset.py printed.

import pandas as pd
from semantic_detector import semantic_scores
from hybrid_detector import KEYWORD_SIGNALS, keyword_hit, BEHAVIOURAL_CATEGORIES

TEST_CSV = "deepset_test.csv"


GLOBAL_THRESHOLD = 0.24  


def predict(prompt):
    """
    Hybrid Layer 1 prediction for a single prompt.
    Returns (is_attack, path) where path is 'keyword', 'semantic', 'both', or 'none'.
    """
    p_lower = str(prompt).lower()

    # Keyword path: any category keyword hit
    kw = any(keyword_hit(p_lower, c) for c in BEHAVIOURAL_CATEGORIES)

    # Semantic path: max category score crosses the threshold
    sem_scores = semantic_scores(str(prompt))
    sem = max(sem_scores.values()) >= GLOBAL_THRESHOLD

    is_attack = kw or sem
    if kw and sem:
        path = "both"
    elif kw:
        path = "keyword"
    elif sem:
        path = "semantic"
    else:
        path = "none"
    return is_attack, path


def main():
    df = pd.read_csv(TEST_CSV)
    print(f"Evaluating on held-out TEST split: {len(df)} prompts "
          f"({(df['label']==1).sum()} injection, {(df['label']==0).sum()} legitimate)")
    print(f"Using calibrated global threshold: {GLOBAL_THRESHOLD}\n")

    results = []
    for _, row in df.iterrows():
        is_attack, path = predict(row["text"])
        results.append({
            "text": str(row["text"])[:120],
            "actual": int(row["label"]),
            "predicted": 1 if is_attack else 0,
            "path": path,
        })

    res = pd.DataFrame(results)
    res.to_csv("deepset_eval_results.csv", index=False)

    # ── Confusion matrix ──
    actual = res["actual"] == 1
    pred = res["predicted"] == 1
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(res)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"                  Predicted")
    print(f"                Attack   Benign")
    print(f"Actual Attack  {tp:>6}   {fn:>6}   (TP / FN)")
    print(f"Actual Benign  {fp:>6}   {tn:>6}   (FP / TN)")
    print()
    print("METRICS")
    print("=" * 50)
    print(f"  Accuracy            : {acc:.3f}")
    print(f"  Precision           : {prec:.3f}")
    print(f"  Recall (detection)  : {rec:.3f}")
    print(f"  F1 score            : {f1:.3f}")
    print(f"  False positive rate : {fpr:.3f}")

    # ── Detection path breakdown (for caught attacks) ──
    caught = res[(res["actual"] == 1) & (res["predicted"] == 1)]
    print("\nDETECTION PATH (for correctly caught attacks)")
    print("=" * 50)
    path_counts = caught["path"].value_counts().to_dict()
    total_caught = len(caught)
    for path in ["keyword", "semantic", "both"]:
        n = path_counts.get(path, 0)
        pct = (n / total_caught * 100) if total_caught else 0
        print(f"  {path:10}: {n:>3}  ({pct:.0f}%)")

    # Semantic-only contribution: attacks caught by semantic that
    # keyword alone would have MISSED.
    semantic_only = path_counts.get("semantic", 0)
    print(f"\n  Attacks the semantic layer caught that keyword")
    print(f"  matching alone would have MISSED: {semantic_only}")

    print("\nResults saved to deepset_eval_results.csv")


if __name__ == "__main__":
    main()
