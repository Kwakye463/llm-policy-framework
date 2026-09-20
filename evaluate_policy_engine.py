# evaluate_policy_engine.py
#
# END-TO-END POLICY FRAMEWORK EVALUATION.
#
# THIS script evaluates the full policy framework the project is actually about:
#     prompt
#        │
#        ▼   extract_action()  - PIP (Policy Information Point)
#     structured attributes
#        │
#        ▼   OPA + llm_policy.rego  - PDP (Policy Decision Point)
#     ALLOW / DENY  (+ which policies fired)
# It runs the deepset held-out test split through the real PolicyEngine (the same one the app and experiments use), so the numbers describe the
# framework's enforcement behaviour — not just a classifier's output.
# For each prompt it records the OPA decision and the exact policy IDs that fired, giving a fully auditable, traceable decision per input — the
# property that distinguishes this policy-based approach from an opaque classifier.
# Layer 2 (the LLM semantic classifier) is OFF by default for speed and reproducibility. Pass --with-llm to include it.
# Usage:
#   python evaluate_policy_engine.py             # Layer 1 + OPA (fast)
#   python evaluate_policy_engine.py --with-llm  # full engine (slow: ~Ollama calls)

import sys
import re
import pandas as pd
from collections import Counter

from policy_engine import PolicyEngine
from hybrid_detector import detect_attributes

TEST_CSV = "deepset_test.csv"
WITH_LLM = "--with-llm" in sys.argv


def build_attributes(prompt):
    """
    PIP step: produce the structured attribute dict OPA evaluates.
    Uses the hybrid detector (keyword + semantic). Layer 2 is added only
    when --with-llm is set, to keep the default run fast and deterministic.
    """
    attributes, _ = detect_attributes(str(prompt), source="user")
    if WITH_LLM:
        from policy_engine import classify_intent_with_llm
        attributes["llm_judged_unsafe"] = classify_intent_with_llm(str(prompt))
    else:
        attributes["llm_judged_unsafe"] = False
    return attributes


def main():
    df = pd.read_csv(TEST_CSV)
    engine = PolicyEngine()

    mode = "Layer 1 (hybrid) + OPA" + ("  + Layer 2 LLM" if WITH_LLM else "")
    print(f"End-to-end policy framework evaluation")
    print(f"Mode      : {mode}")
    print(f"Test set  : {len(df)} held-out prompts "
          f"({(df['label']==1).sum()} injection, {(df['label']==0).sum()} legitimate)")
    print("Running through PIP -> OPA (PDP) ...\n")

    results = []
    policy_fire_counts = Counter()

    for i, row in df.iterrows():
        prompt = str(row["text"])
        attributes = build_attributes(prompt)
        request = {"prompt": prompt, "source": "user", "action": attributes}

        decision = engine.evaluate(request)
        is_deny = decision["decision"] == "DENY"

        # Record which policies fired (traceability)
        for reason in decision["reasons"]:
            pid = reason.split(":")[0].strip()  # e.g. "POL-003"
            policy_fire_counts[pid] += 1

        results.append({
            "text": prompt[:120],
            "actual": int(row["label"]),           # 1 = injection
            "decision": decision["decision"],
            "predicted": 1 if is_deny else 0,       # DENY == predicted attack
            "policies_fired": "; ".join(r.split(":")[0] for r in decision["reasons"]) or "none",
        })

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(df)} ...")

    res = pd.DataFrame(results)
    out = "policy_engine_eval_results.csv"
    res.to_csv(out, index=False)

    # ── Confusion matrix (DENY == predict attack) ──
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

    print("\nEND-TO-END POLICY ENFORCEMENT — CONFUSION MATRIX")
    print("=" * 52)
    print("                    Predicted")
    print("                 DENY     ALLOW")
    print(f"Actual Attack  {tp:>6}   {fn:>6}   (correctly blocked / missed)")
    print(f"Actual Benign  {fp:>6}   {tn:>6}   (wrongly blocked / allowed)")
    print()
    print("METRICS")
    print("=" * 52)
    print(f"  Accuracy            : {acc:.3f}")
    print(f"  Precision           : {prec:.3f}")
    print(f"  Recall (block rate) : {rec:.3f}")
    print(f"  F1 score            : {f1:.3f}")
    print(f"  False positive rate : {fpr:.3f}")

    # ── Policy attribution (which policies did the enforcing) ──
    print("\nPOLICY ATTRIBUTION  (times each policy fired across the test set)")
    print("=" * 52)
    if policy_fire_counts:
        for pid in sorted(policy_fire_counts):
            print(f"  {pid}: {policy_fire_counts[pid]}")
    else:
        print("  (no policies fired)")

    print(f"\nPer-prompt decisions saved to {out}")
    print("Every DENY is traceable to the specific policy IDs that fired,")
    print("providing an auditable decision trail for each input.")


if __name__ == "__main__":
    main()
