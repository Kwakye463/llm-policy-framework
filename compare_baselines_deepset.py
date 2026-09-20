# compare_baselines_deepset.py
#
# THREE-WAY COMPARISON on the deepset held-out test split (116 prompts).
#
# Evaluates three systems on the SAME real, external, labelled data so the
# comparison is fair and statistically meaningful:
#
#   1. Unprotected   - no defence at all. Allows everything. 
#   2. Regex guardrail — the naive blocklist from regex_baseline.py.
#   3. Policy framework — the full hybrid detection + OPA policy engine.
#
# For each system we compute the same confusion matrix and metrics (precision, recall, F1, accuracy, false-positive rate), replacing the
# earlier 16-prompt hand-crafted comparison with a rigorous held-out one.
#
# NOTE ON SPEED: detection decisions do not require the LLM response, so TinyLlama isnt called here. The unprotected control by definition allows
# everything (its detection recall is 0 by construction); the regex and policy systems are pure decision functions. This keeps the run fast.
# Layer 2 (the LLM classifier) is OFF by default so the framework column is reproducible and quick; pass --with-llm to include it.
# Usage:  python compare_baselines_deepset.py [--with-llm]

import re
import sys
import pandas as pd

from hybrid_detector import detect_attributes
from policy_engine import PolicyEngine

TEST_CSV = "deepset_test.csv"
WITH_LLM = "--with-llm" in sys.argv


# 
# SYSTEM 1: Unprotected (control)
# 
def unprotected_decision(prompt):
    # No defence: nothing is ever blocked.
    return 0  # 0 = allow, 1 = block


# 
# SYSTEM 2: Regex guardrail
# (identical blocklist to regex_baseline.py, for a faithful comparison)
# 
BLOCKED_PATTERNS = [
    r"ignore.{0,20}instructions",
    r"disregard.{0,20}rules",
    r"forget.{0,20}instructions",
    r"act as (dan|an? unrestricted)",
    r"pretend you (have no|are)",
    r"system prompt",
    r"api key",
    r"jailbreak",
    r"delete.{0,20}records",
    r"execute.{0,20}command",
]

def regex_decision(prompt):
    p = str(prompt).lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, p):
            return 1
    return 0


# 
# SYSTEM 3: Full policy framework (hybrid detection + OPA)
# 
_engine = PolicyEngine()

def framework_decision(prompt):
    attributes, _ = detect_attributes(str(prompt), source="user")
    if WITH_LLM:
        from policy_engine import classify_intent_with_llm
        attributes["llm_judged_unsafe"] = classify_intent_with_llm(str(prompt))
    else:
        attributes["llm_judged_unsafe"] = False
    request = {"prompt": str(prompt), "source": "user", "action": attributes}
    decision = _engine.evaluate(request)
    return 1 if decision["decision"] == "DENY" else 0


# 
# METRICS
# 
def metrics(df, pred_col):
    actual = df["label"] == 1
    pred = df[pred_col] == 1
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(df)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "accuracy": round(acc, 3),
            "fpr": round(fpr, 3)}


def main():
    df = pd.read_csv(TEST_CSV)
    n_attack = int((df["label"] == 1).sum())
    n_benign = int((df["label"] == 0).sum())

    mode = "detection only" + ("  (+ Layer 2 LLM)" if WITH_LLM else "")
    print(f"Three-way baseline comparison on deepset held-out test split")
    print(f"Prompts   : {len(df)} ({n_attack} injection, {n_benign} legitimate)")
    print(f"Framework : {mode}\n")
    print("Scoring all three systems ...")

    df["pred_unprotected"] = df["text"].apply(unprotected_decision)
    df["pred_regex"] = df["text"].apply(regex_decision)
    df["pred_framework"] = df["text"].apply(framework_decision)

    systems = [
        ("Unprotected", "pred_unprotected"),
        ("Regex guardrail", "pred_regex"),
        ("Policy framework", "pred_framework"),
    ]

    print("\n" + "=" * 78)
    print(f"{'System':<20}{'Blocked':<9}{'Prec':<7}{'Recall':<8}{'F1':<7}{'Acc':<7}{'FPR':<7}")
    print("-" * 78)

    summary_rows = []
    for name, col in systems:
        m = metrics(df, col)
        blocked = m["TP"] + m["FP"]
        print(f"{name:<20}{blocked:<9}{m['precision']:<7}{m['recall']:<8}"
              f"{m['f1']:<7}{m['accuracy']:<7}{m['fpr']:<7}")
        summary_rows.append({"system": name, "attacks_blocked": m["TP"],
                             "benign_blocked": m["FP"], **m})

    print("=" * 78)

    # Save both the per-prompt predictions and the summary table.
    df[["text", "label", "pred_unprotected", "pred_regex",
        "pred_framework"]].to_csv("baseline_comparison_deepset.csv", index=False)
    pd.DataFrame(summary_rows).to_csv("baseline_summary_deepset.csv", index=False)

    print("\nPer-prompt predictions -> baseline_comparison_deepset.csv")
    print("Summary table          -> baseline_summary_deepset.csv")
    print("\nInterpretation:")
    print("  Recall  = proportion of real attacks blocked (higher = safer)")
    print("  FPR     = proportion of legitimate prompts wrongly blocked (lower = better)")
    print("  The unprotected control blocks nothing by design (recall 0).")


if __name__ == "__main__":
    main()
