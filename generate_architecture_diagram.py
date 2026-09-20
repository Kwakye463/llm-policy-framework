import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)

# palette
NAVY   = "#1f3a5f"
TEAL   = "#2a9d8f"
AMBER  = "#e9c46a"
RUST   = "#e76f51"
GREY   = "#8d99ae"
LIGHT  = "#eef1f6"
PIPCOL = "#e8f1f0"
PDPCOL = "#eaf0f7"
PEPCOL = "#fdf3e7"
INK    = "#1a1a2e"

fig, ax = plt.subplots(figsize=(13, 15))
ax.set_xlim(0, 13)
ax.set_ylim(0, 15)
ax.axis("off")


def box(x, y, w, h, text, fc, ec=NAVY, tc=INK, fs=11, bold=False, rounded=True, lw=1.6):
    style = "round,pad=0.02,rounding_size=0.12" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)


def band(x, y, w, h, label, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=fc, edgecolor=GREY, linewidth=1.2,
                                linestyle="--", zorder=1, alpha=0.55))
    ax.text(x + w / 2, y + h - 0.22, label, ha="center", va="top",
            fontsize=11, color=NAVY, fontweight="bold", style="italic", zorder=6)


def arrow(x1, y1, x2, y2, color=NAVY, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=18, color=color, linewidth=lw, zorder=4))


def label(x, y, text, color=INK, fs=10, bold=False, it=False):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", style="italic" if it else "normal", zorder=5)


# ── Title ──
ax.text(6.5, 14.6, "Policy-Based LLM Security Framework — End-to-End Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

# 
# RUNTIME FLOW  (left/centre column: x ~ 0.6 .. 8.2)
# 

# Prompt entry
box(1.9, 13.4, 4.6, 0.7, "Incoming Prompt  (source: user / external)", "#ffffff", ec=NAVY, bold=True, fs=11)

# --- PIP band ---
band(0.6, 9.5, 7.6, 3.7, "Policy Information Point (PIP) — attribute extraction", PIPCOL)

# Layer 1 hybrid
box(1.0, 11.5, 3.3, 0.95, "Layer 1 — Hybrid Detection", "#ffffff", ec=TEAL, bold=True, fs=11)
box(1.0, 10.6, 1.55, 0.7, "Keyword\nsignals", "#ffffff", ec=GREY, fs=9)
box(2.75, 10.6, 1.55, 0.7, "Semantic\nsimilarity", "#ffffff", ec=GREY, fs=9)
label(2.15, 10.35, "OR  →  behavioural attributes", NAVY, 9, it=True)

# Layer 2
box(4.7, 11.0, 3.2, 1.05, "Layer 2 — LLM Semantic\nClassifier (few-shot)", "#ffffff", ec=TEAL, bold=True, fs=10)
label(6.3, 10.6, "→  llm_judged_unsafe", NAVY, 9, it=True)

# attributes bundle
box(2.4, 9.65, 3.6, 0.6, "Structured behavioural attributes", "#ffffff", ec=NAVY, bold=True, fs=10)

# --- PDP band ---
band(0.6, 6.5, 7.6, 2.6, "Policy Decision Point (PDP)", PDPCOL)
box(2.3, 7.5, 3.8, 1.1, "Open Policy Agent (OPA)\nevaluates compiled Rego", "#ffffff", ec=NAVY, bold=True, fs=11)

# Decision diamond (as box)
box(3.4, 6.5, 1.6, 0.62, "ALLOW?", AMBER, ec=NAVY, bold=True, fs=11)
label(6.15, 7.02, "decision + policy IDs (traceable)", NAVY, 9, it=True)

# --- PEP band ---
band(0.6, 0.55, 7.6, 5.15, "Policy Enforcement Point (PEP) — enforcement boundaries", PEPCOL)

# input deny
box(0.75, 4.5, 2.4, 0.62, "DENY → block at input\n(reason logged)", "#fdeae5", ec=RUST, tc=RUST, bold=True, fs=9)

# tool call
box(2.9, 3.8, 3.0, 0.9, "Tool-Call Check\n(permission model, least privilege)", "#ffffff", ec=NAVY, bold=True, fs=10)
box(0.8, 3.65, 1.7, 0.68, "DENY tool\n(not permitted)", "#fdeae5", ec=RUST, tc=RUST, fs=8)

# LLM response
box(3.1, 2.5, 2.6, 0.8, "LLM Response Generated\n(TinyLlama)", "#ffffff", ec=NAVY, bold=True, fs=10)

# output validation
box(3.0, 1.25, 2.8, 0.8, "Output Validation\n(leakage / secrets / danger)", "#ffffff", ec=NAVY, bold=True, fs=10)
box(0.8, 1.3, 1.7, 0.7, "BLOCK output\n(violation)", "#fdeae5", ec=RUST, tc=RUST, fs=8)

# delivered
box(6.1, 1.25, 1.9, 0.8, "Response\nDelivered", "#e7f6f0", ec=TEAL, tc=TEAL, bold=True, fs=10)

# ── runtime arrows ──
arrow(4.2, 13.4, 4.2, 12.5)                        # prompt -> layer1
arrow(4.3, 11.95, 4.7, 11.7)                       # layer1 -> layer2 region
arrow(4.2, 11.5, 4.2, 10.28, color=NAVY)           # layer1 -> attributes
arrow(6.3, 11.0, 4.6, 10.28, color=GREY, lw=1.6)   # layer2 -> attributes
arrow(4.2, 9.65, 4.2, 8.65)                        # attributes -> OPA
arrow(4.2, 7.5, 4.2, 7.2)                          # OPA -> diamond
arrow(3.5, 6.5, 2.0, 5.5, color=RUST)              # ALLOW? -> DENY input (no)
label(2.45, 6.0, "no", RUST, 9, bold=True)
arrow(4.2, 6.5, 4.2, 4.72, color=TEAL)             # yes -> tool check
label(4.5, 5.05, "yes", TEAL, 9, bold=True)
arrow(2.9, 4.0, 2.5, 4.0, color=RUST)              # tool -> deny tool
arrow(4.4, 3.8, 4.4, 3.32)                         # tool -> llm
arrow(4.4, 2.5, 4.4, 2.07)                         # llm -> output val
arrow(3.0, 1.65, 2.5, 1.65, color=RUST)            # outputval -> block
arrow(5.8, 1.65, 6.1, 1.65, color=TEAL)            # outputval -> delivered

# 
# POLICY AUTHORING FLOW (right column, offline): x ~ 9.0 .. 12.6
# 
band(8.7, 6.4, 4.0, 6.4, "Policy Authoring (offline, policy-as-data)", LIGHT)

box(9.2, 11.5, 3.0, 0.85, "policies.json\n(declarative policy set)", "#ffffff", ec=NAVY, bold=True, fs=10)
box(9.2, 10.2, 3.0, 0.85, "Policy Compiler\n(validate schema)", "#ffffff", ec=NAVY, bold=True, fs=10)
box(9.2, 8.9, 3.0, 0.85, "llm_policy.rego\n(auto-generated)", "#ffffff", ec=NAVY, bold=True, fs=10)
box(9.2, 7.5, 3.0, 0.8, "Policy Analyser\n(conflicts / gaps → CI gate)", "#ffffff", ec=AMBER, bold=True, fs=9)

arrow(10.7, 11.5, 10.7, 11.05)                     # json -> compiler
arrow(10.7, 10.2, 10.7, 9.75)                      # compiler -> rego
arrow(10.7, 8.9, 10.7, 8.3, color=GREY)            # rego -> analyser
# rego feeds OPA (into PDP)
arrow(9.2, 9.3, 6.1, 8.05, color=NAVY, lw=2.0, style="-|>")
label(7.7, 8.95, "loaded into OPA", NAVY, 9, it=True)

# ── legend ──
legend_items = [
    ("Runtime data flow", NAVY),
    ("Allowed path", TEAL),
    ("Denied / blocked path", RUST),
    ("Policy authoring (offline)", GREY),
]
lx, ly = 0.7, 0.15
handles = [Line2D([0], [0], color=col, lw=3) for _, col in legend_items]
ax.legend(handles, [t for t, _ in legend_items], loc="lower left",
          bbox_to_anchor=(0.02, -0.01), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("figures/architecture_diagram.png", dpi=200, bbox_inches="tight")
plt.savefig("figures/architecture_diagram.svg", bbox_inches="tight")
plt.close()
print("Saved figures/architecture_diagram.png and .svg")
