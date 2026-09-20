# policy_analyser.py
#
# STATIC POLICY ANALYSER — integrity and consistency checking.
#
# This fulfils the proposal's commitment to test policies "for integrity and consistency" before deployment. It is the LLM-security analogue of
# tools like AWS Access Analyzer and formal IAM policy verification: it inspects the policy SET as a whole (not individual prompts) and reports
# structural problems a human author might miss.
#
# It works ONLY because policies are declarative data (policies.json). You cannot statically analyse hand-written Rego rules this way — that is
# precisely why the policy-as-data architecture matters.
#
# Checks performed:
#   1. CONTRADICTION  — an ALLOW and a DENY policy that fire on the same
#                       condition for an overlapping subject. The engine
#                       cannot honour both; this is a critical integrity
#                       failure.
#   2. SHADOWING      — a broader policy fully covers a narrower one, so
#                       the narrower policy can never fire independently
#                       (dead / redundant policy).
#   3. DUPLICATE      — two policies with identical subject + condition +
#                       effect (redundant restatement).
#   4. COVERAGE GAP   — a declared behavioural attribute that NO policy
#                       references, meaning that attack signal is detected
#                       but never acted upon.
#


import json
import sys
from itertools import combinations


# 
# CONDITION MODEL
# 
# A policy's condition is reduced to the SET of attributes that must be TRUE for it to fire (all_of + subject-implied), plus any attributes that
# must be FALSE (not). For overlap reasoning we use the "required-true" set as the primary signature; this is sufficient for the conflict classes we
# detect and keeps the analysis explainable.

SUBJECT_ATTR = {
    "external": ("source_is_external", True),
    "user": ("source_is_external", False),
}


def required_true_set(policy):
    """Attributes that must be TRUE for this policy to fire."""
    req = set()
    cond = policy.get("condition", {})
    req.update(cond.get("all_of", []))
    # any_of is a weaker constraint; we track it separately below.
    subj = policy.get("subject", "any")
    if subj in SUBJECT_ATTR:
        attr, val = SUBJECT_ATTR[subj]
        if val:
            req.add(attr)
    return req


def any_of_set(policy):
    return set(policy.get("condition", {}).get("any_of", []))


def required_false_set(policy):
    """Attributes that must be FALSE for this policy to fire."""
    req = set()
    cond = policy.get("condition", {})
    nots = cond.get("not", [])
    nots = nots if isinstance(nots, list) else [nots]
    req.update(nots)
    subj = policy.get("subject", "any")
    if subj == "user":
        req.add("source_is_external")
    return req


def subjects_overlap(p1, p2):
    """Do the two policies' subjects share any possible request?"""
    s1 = p1.get("subject", "any")
    s2 = p2.get("subject", "any")
    if s1 == "any" or s2 == "any":
        return True
    return s1 == s2


# 
# CONFLICT CHECKS
# 

def find_contradictions(policies):
    """
    ALLOW vs DENY on overlapping conditions and subjects.
    A contradiction exists if one policy ALLOWs and another DENYs and their
    firing conditions can be satisfied simultaneously.
    """
    findings = []
    for p1, p2 in combinations(policies, 2):
        if p1["effect"] == p2["effect"]:
            continue  # same effect can't contradict
        if not subjects_overlap(p1, p2):
            continue

        t1, t2 = required_true_set(p1), required_true_set(p2)
        f1, f2 = required_false_set(p1), required_false_set(p2)

        # If either policy requires an attribute TRUE that the other
        # requires FALSE, their conditions are mutually exclusive — no clash.
        if (t1 & f2) or (t2 & f1):
            continue

        # Overlap of required-true sets means a single request can satisfy
        # both firing conditions -> ALLOW and DENY collide.
        if t1 & t2 or (not t1 and not t2):
            findings.append({
                "type": "CONTRADICTION",
                "severity": "critical",
                "policies": [p1["id"], p2["id"]],
                "detail": (f"{p1['id']} ({p1['effect']}) and {p2['id']} "
                           f"({p2['effect']}) can fire on the same request "
                           f"(shared conditions: {sorted(t1 & t2) or 'none/broad'}).")
            })
    return findings


def find_shadowing(policies):
    """
    A DENY policy whose required-true set is a SUBSET of another DENY
    policy's set is BROADER; the narrower one is shadowed (can't fire
    without the broader one also firing). Flag the narrower as redundant.
    """
    findings = []
    denies = [p for p in policies if p["effect"] == "DENY"]
    for p1, p2 in combinations(denies, 2):
        if not subjects_overlap(p1, p2):
            continue
        t1, t2 = required_true_set(p1), required_true_set(p2)
        if not t1 or not t2:
            continue
        if t1 < t2:  # p1 strictly broader -> p2 shadowed
            findings.append({
                "type": "SHADOWING",
                "severity": "low",
                "policies": [p2["id"], p1["id"]],
                "detail": (f"{p2['id']} is shadowed by broader policy "
                           f"{p1['id']}: whenever {p2['id']} fires, {p1['id']} "
                           f"already fires. {p2['id']} adds no independent effect.")
            })
        elif t2 < t1:
            findings.append({
                "type": "SHADOWING",
                "severity": "low",
                "policies": [p1["id"], p2["id"]],
                "detail": (f"{p1['id']} is shadowed by broader policy "
                           f"{p2['id']}: whenever {p1['id']} fires, {p2['id']} "
                           f"already fires. {p1['id']} adds no independent effect.")
            })
    return findings


def find_duplicates(policies):
    """Two policies with identical subject + required-true + effect."""
    findings = []
    for p1, p2 in combinations(policies, 2):
        same_subject = p1.get("subject", "any") == p2.get("subject", "any")
        same_effect = p1["effect"] == p2["effect"]
        same_cond = required_true_set(p1) == required_true_set(p2) and \
                    any_of_set(p1) == any_of_set(p2)
        if same_subject and same_effect and same_cond:
            findings.append({
                "type": "DUPLICATE",
                "severity": "medium",
                "policies": [p1["id"], p2["id"]],
                "detail": (f"{p1['id']} and {p2['id']} have identical subject, "
                           f"condition, and effect — one is redundant.")
            })
    return findings


def find_coverage_gaps(policy_set):
    """Declared attributes that no policy references."""
    declared = set(policy_set.get("attributes", []))
    referenced = set()
    for p in policy_set["policies"]:
        referenced |= required_true_set(p)
        referenced |= any_of_set(p)
        referenced |= required_false_set(p)
    gaps = declared - referenced
    findings = []
    for attr in sorted(gaps):
        findings.append({
            "type": "COVERAGE_GAP",
            "severity": "medium",
            "policies": [],
            "detail": (f"Attribute '{attr}' is declared and detected by "
                       f"Layer 1/2 but no policy acts on it — the signal is "
                       f"computed but never enforced.")
        })
    return findings


# 
# REPORT
# 

def analyse(policy_set):
    policies = policy_set["policies"]
    findings = []
    findings += find_contradictions(policies)
    findings += find_shadowing(policies)
    findings += find_duplicates(policies)
    findings += find_coverage_gaps(policy_set)
    return findings


def print_report(policy_set, findings):
    name = policy_set.get("policy_set", "unnamed")
    n = len(policy_set["policies"])
    print(f"\nPOLICY CONSISTENCY ANALYSIS")
    print("=" * 70)
    print(f"Policy set : {name}")
    print(f"Policies   : {n}")
    print(f"Attributes : {len(policy_set.get('attributes', []))}")
    print("-" * 70)

    if not findings:
        print("No conflicts, redundancies, or coverage gaps detected.")
        print("Policy set is internally consistent.")
        return 0

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 9))

    counts = {}
    for f in findings:
        counts[f["type"]] = counts.get(f["type"], 0) + 1

    for f in findings:
        pol = f" [{', '.join(f['policies'])}]" if f["policies"] else ""
        print(f"\n  {f['type']}  ({f['severity']}){pol}")
        print(f"    {f['detail']}")

    print("\n" + "-" * 70)
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Critical findings -> non-zero exit (CI/CD gate).
    has_critical = any(f["severity"] == "critical" for f in findings)
    return 2 if has_critical else 1


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "policies.json"
    with open(src) as f:
        policy_set = json.load(f)
    findings = analyse(policy_set)
    code = print_report(policy_set, findings)
    sys.exit(code)


if __name__ == "__main__":
    main()
