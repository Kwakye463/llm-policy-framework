# policy_compiler.py
#
# POLICY COMPILER — the core of the policy-as-data architecture.
#
# This is the component that realises the AWS IAM analogy properly. In IAM, a policy is a JSON DOCUMENT and the IAM engine enforces whatever
# document you give it, without being rewritten. Here, the same idea:
#
#   policies.json   — policies declared as DATA by a security administrator
#        │
#        ▼  (this compiler)
#   llm_policy.rego — Rego rules GENERATED automatically
#        │
#        ▼
#   OPA             — evaluates the generated Rego
#
# The security administrator never writes Rego or Python. They edit a declarative JSON document. This compiler validates that document against
# a schema, checks it for structural problems, and emits the Rego that OPA
# enforces. Adding, removing, or changing a policy is a DATA edit, not a code change.
#
# Supported condition operators:
#   all_of  — every listed attribute must be true   (logical AND)
#   any_of  — at least one listed attribute is true  (logical OR)
#   not     — the listed attribute must be false     (logical NOT)
# These may be combined within a single policy condition.

import json
import sys

VALID_EFFECTS = {"ALLOW", "DENY"}
VALID_SUBJECTS = {"any", "user", "external"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_OPERATORS = {"all_of", "any_of", "not"}


# 
# VALIDATION
# 

class PolicyValidationError(Exception):
    pass


def validate_policy_set(policy_set):
    """
    Validates the structure and semantics of a policy set.
    Raises PolicyValidationError with a clear message on the first problem.
    Returns the list of declared attributes on success.
    """
    if "policies" not in policy_set:
        raise PolicyValidationError("Policy set has no 'policies' array.")

    declared_attributes = set(policy_set.get("attributes", []))
    if not declared_attributes:
        raise PolicyValidationError("Policy set declares no 'attributes'.")

    seen_ids = set()

    for i, policy in enumerate(policy_set["policies"]):
        loc = f"Policy #{i + 1}"

        # id
        pid = policy.get("id")
        if not pid:
            raise PolicyValidationError(f"{loc}: missing 'id'.")
        if pid in seen_ids:
            raise PolicyValidationError(f"{loc}: duplicate policy id '{pid}'.")
        seen_ids.add(pid)
        loc = f"Policy '{pid}'"

        # effect
        effect = policy.get("effect")
        if effect not in VALID_EFFECTS:
            raise PolicyValidationError(
                f"{loc}: effect must be one of {VALID_EFFECTS}, got '{effect}'.")

        # subject
        subject = policy.get("subject", "any")
        if subject not in VALID_SUBJECTS:
            raise PolicyValidationError(
                f"{loc}: subject must be one of {VALID_SUBJECTS}, got '{subject}'.")

        # severity
        severity = policy.get("severity", "medium")
        if severity not in VALID_SEVERITIES:
            raise PolicyValidationError(
                f"{loc}: severity must be one of {VALID_SEVERITIES}, got '{severity}'.")

        # condition
        condition = policy.get("condition")
        if not condition or not isinstance(condition, dict):
            raise PolicyValidationError(f"{loc}: missing or invalid 'condition'.")

        for op, operand in condition.items():
            if op not in VALID_OPERATORS:
                raise PolicyValidationError(
                    f"{loc}: unknown operator '{op}'. Valid: {VALID_OPERATORS}.")

            # Normalise operand to a list for checking
            attrs = operand if isinstance(operand, list) else [operand]
            for attr in attrs:
                if attr not in declared_attributes:
                    raise PolicyValidationError(
                        f"{loc}: condition references undeclared attribute "
                        f"'{attr}'. Declared attributes: {sorted(declared_attributes)}.")

    return declared_attributes


# 
# COMPILATION  (JSON policy -> Rego)
# 

def _subject_clause(subject):
    """Rego clause constraining the request source, or None for 'any'."""
    if subject == "any":
        return None
    if subject == "external":
        return "input.action.source_is_external == true"
    if subject == "user":
        return "input.action.source_is_external == false"
    return None


def _condition_clauses(condition):
    """Translate a condition dict into a list of Rego expression lines."""
    clauses = []

    if "all_of" in condition:
        for attr in condition["all_of"]:
            clauses.append(f"input.action.{attr} == true")

    if "any_of" in condition:
        attrs = condition["any_of"]
        # any_of becomes an OR — expressed in Rego via a helper set membership
        # We emit it as: some flag; flag == true for any listed attribute.
        ors = " ; ".join([f"input.action.{a} == true" for a in attrs])
        # Rego OR within a rule is done by separate rule bodies, but for a
        # compact inline OR we use a boolean expression:
        expr = " + ".join([f"to_number(input.action.{a})" for a in attrs])
        clauses.append(f"({expr}) > 0")

    if "not" in condition:
        attrs = condition["not"]
        attrs = attrs if isinstance(attrs, list) else [attrs]
        for attr in attrs:
            clauses.append(f"input.action.{attr} == false")

    return clauses


def compile_to_rego(policy_set):
    """
    Compiles a validated policy set into a complete Rego policy file
    (as a string). Only DENY policies generate deny rules; the default
    effect governs the rest.
    """
    lines = []
    lines.append("# ─────────────────────────────────────────────────────────")
    lines.append("# AUTO-GENERATED by policy_compiler.py — DO NOT EDIT BY HAND.")
    lines.append("# Source of truth: policies.json")
    lines.append(f"# Policy set: {policy_set.get('policy_set', 'unnamed')}"
                 f"  version: {policy_set.get('version', 'n/a')}")
    lines.append("# ─────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("package llm.security")
    lines.append("")
    lines.append("import rego.v1")
    lines.append("")
    lines.append("default allow := true")
    lines.append("")

    for policy in policy_set["policies"]:
        if policy["effect"] != "DENY":
            continue

        pid = policy["id"]
        desc = policy["description"]
        severity = policy.get("severity", "medium")

        clauses = []
        subj = _subject_clause(policy.get("subject", "any"))
        if subj:
            clauses.append(subj)
        for c in _condition_clauses(policy["condition"]):
            if c not in clauses:          # avoid duplicating the subject clause
                clauses.append(c)

        lines.append(f"# {pid} [{severity}] — {desc}")
        lines.append("deny contains reason if {")
        for c in clauses:
            lines.append(f"    {c}")
        # The reason carries the policy id so denials are traceable to policy.
        lines.append(f'    reason := "{pid}: {desc}"')
        lines.append("}")
        lines.append("")

    lines.append("# Final decision: deny overrides the default allow.")
    lines.append("allow := false if {")
    lines.append("    count(deny) > 0")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# 
# CLI
# 

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "policies.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "llm_policy.rego"

    with open(src) as f:
        policy_set = json.load(f)

    print(f"Loading policy set from {src} ...")
    try:
        attrs = validate_policy_set(policy_set)
    except PolicyValidationError as e:
        print(f"\nVALIDATION FAILED: {e}\n")
        sys.exit(1)

    n = len(policy_set["policies"])
    print(f"Validation passed: {n} policies, {len(attrs)} declared attributes.")

    rego = compile_to_rego(policy_set)
    with open(out, "w") as f:
        f.write(rego)

    print(f"Compiled {n} policies -> {out}")
    print("\nGenerated Rego preview:")
    print("─" * 60)
    print(rego)


if __name__ == "__main__":
    main()
