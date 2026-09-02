"""
Export presentation-ready figures from the completed rule-validation study.

NO scientific analysis is re-run. Sources:
  * Figures 2 & 3  -> rule_validation_summary.csv (the committed study output)
  * Figure 1       -> the deterministic source-tree definition, asserted against
                      the archived thresholds (a lookup of the rule set, not analysis)
  * Figure 4       -> descriptive correlations, asserted against the notebook's
                      printed values before anything is drawn
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from sklearn.tree import DecisionTreeRegressor

OUT = os.path.expanduser("~/Desktop/Rule_Validation_Figures")
os.makedirs(OUT, exist_ok=True)
DPI = 350

# validated palette (see project data-viz reference)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
GOOD, WARN, NONE_ = "#1baf7a", "#eda100", "#9c9b96"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.28,
})

# ----------------------------------------------------------------- study results
S = pd.read_csv("rule_validation_summary.csv")
FOLDCOLS = [f"Fold {k}" for k in range(1, 6)]


def parse(v):
    """'+0.194 ✓' -> (0.194, True); 'n/a'/NaN -> (nan, None)."""
    if pd.isna(v) or str(v).strip() in ("n/a", "nan"):
        return np.nan, None
    s = str(v)
    ok = "✓" in s if ("✓" in s or "✗" in s) else None
    return float(s.replace("✓", "").replace("✗", "").strip()), ok


EFF = {r["Rule"]: [parse(r[c])[0] for c in FOLDCOLS] for _, r in S.iterrows()}
OKM = {r["Rule"]: [parse(r[c])[1] for c in FOLDCOLS] for _, r in S.iterrows()}
COV = dict(zip(S["Rule"], S["Coverage"]))
REP = dict(zip(S["Rule"], S["Replicated?"]))

COND = {"R1": "Time ≤ 0.0022", "R2": "0.0022 < Time ≤ 0.1071",
        "R3": "Particle Size ≤ 9.5050", "R4": "Particle Size > 9.5050",
        "R5": "Polymer MW ≤ 21.0000",   "R6": "Polymer MW > 21.0000",
        "R7": "10.8857 < Time ≤ 17.8903", "R8": "Time > 17.8903"}
LEAF = {"R1": 0.0001, "R2": 0.1410, "R3": 0.4383, "R4": 0.2597,
        "R5": 0.6598, "R6": 0.4549, "R7": 0.6629, "R8": 0.7749}
FORM = {"R3", "R4", "R5", "R6"}
SHORT_COND = {"R3": "Particle Size ≤ 9.505", "R4": "Particle Size > 9.505",
              "R5": "Polymer MW ≤ 21", "R6": "Polymer MW > 21"}
SHORT_DIR = {"R1": "—  no formulation condition", "R2": "—  no formulation condition",
             "R3": "3 / 3 evaluable  ·  2 n/a", "R4": "3 / 3 evaluable  ·  2 n/a",
             "R5": "5 / 5 folds", "R6": "5 / 5 folds",
             "R7": "—  no formulation condition", "R8": "—  no formulation condition"}


def verdict(rid):
    if rid not in FORM:
        return "UNSUPPORTED", NONE_, "no formulation condition — pure Time"
    n_ok = sum(1 for x in OKM[rid] if x)
    n_ev = sum(1 for x in OKM[rid] if x is not None)
    if n_ev == 5 and n_ok == 5:
        return "SUPPORTED", GOOD, f"direction held {n_ok}/{n_ev} folds"
    return "BORDERLINE / WEAK", WARN, f"direction held {n_ok}/{n_ev} evaluable ({5-n_ev} unevaluable)"


# ================================================================= FIG 1 — TREE
def fig1():
    # Re-derive the tree ONLY to assert the archived structure, then draw it by hand.
    proc = pd.read_excel("mp_dataset_processed.xlsx")
    t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(
        proc[["Time", "Polymer MW", "Particle Size", "LA/GA"]], proc["Release"])
    got = sorted(round(float(x), 4) for x in t.tree_.threshold if x != -2.0)
    assert got == [0.0022, 0.1071, 3.3365, 9.505, 10.8857, 17.8903, 21.0], got
    assert t.get_n_leaves() == 8

    # (x, y, label, is_time_split)
    nodes = {
        "n0": (0.500, 0.845, "Time ≤ 3.3365", True),
        "n1": (0.250, 0.610, "Time ≤ 0.1071", True),
        "n2": (0.750, 0.610, "Time ≤ 10.8857", True),
        "n3": (0.120, 0.375, "Time ≤ 0.0022", True),
        "n4": (0.380, 0.375, "Particle Size\n≤ 9.5050", False),
        "n5": (0.620, 0.375, "Polymer MW\n≤ 21.0000", False),
        "n6": (0.880, 0.375, "Time ≤ 17.8903", True),
    }
    leaves = {
        "R1": (0.055, 0.135), "R2": (0.185, 0.135), "R3": (0.315, 0.135), "R4": (0.445, 0.135),
        "R5": (0.555, 0.135), "R6": (0.685, 0.135), "R7": (0.815, 0.135), "R8": (0.945, 0.135),
    }
    edges = [("n0", "n1", "True"), ("n0", "n2", "False"),
             ("n1", "n3", "True"), ("n1", "n4", "False"),
             ("n2", "n5", "True"), ("n2", "n6", "False"),
             ("n3", "R1", "True"), ("n3", "R2", "False"),
             ("n4", "R3", "True"), ("n4", "R4", "False"),
             ("n5", "R5", "True"), ("n5", "R6", "False"),
             ("n6", "R7", "True"), ("n6", "R8", "False")]

    fig, ax = plt.subplots(figsize=(16.5, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.0); ax.axis("off")
    pos = {k: (v[0], v[1]) for k, v in nodes.items()}
    pos.update(leaves)

    for a, b, lab in edges:
        x1, y1 = pos[a]; x2, y2 = pos[b]
        ax.annotate("", xy=(x2, y2 + 0.058), xytext=(x1, y1 - 0.045),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=2.0))
        ax.text((x1 + x2) / 2 + (-0.028 if lab == "True" else 0.028),
                (y1 + y2) / 2 + 0.012, lab, fontsize=9.5, color=INK2,
                ha="center", va="center", style="italic")

    for key, (x, y, lab, is_time) in nodes.items():
        fc = "#eceae6" if is_time else "#dce9f9"
        ec = NONE_ if is_time else BLUE
        ax.add_patch(FancyBboxPatch((x - 0.093, y - 0.045), 0.186, 0.090,
                                    boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=2.0))
        ax.text(x, y, lab, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=INK)

    for rid, (x, y) in leaves.items():
        _, col, _ = verdict(rid)
        ax.add_patch(FancyBboxPatch((x - 0.048, y - 0.056), 0.096, 0.112,
                                    boxstyle="round,pad=0.008", fc="#ffffff", ec=col, lw=2.6))
        ax.add_patch(Rectangle((x - 0.048, y + 0.026), 0.096, 0.030, fc=col, ec="none"))
        ax.text(x, y + 0.041, rid, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color="#ffffff")
        ax.text(x, y - 0.004, "Release", ha="center", va="center", fontsize=8.5, color=INK2)
        ax.text(x, y - 0.032, f"{LEAF[rid]:.4f}", ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=INK)

    ax.text(0.5, 0.995, "The source decision tree defining the 8 predefined rules",
            ha="center", fontsize=19, fontweight="bold", color=INK)
    ax.text(0.5, 0.958,
            "DecisionTreeRegressor(max_depth=3) on [Time, Polymer MW, Particle Size, LA/GA] → Release   "
            "·   8 leaves = the 8 rules   ·   thresholds verified against the archived rule set",
            ha="center", fontsize=11, color=INK2)
    ax.text(0.5, 0.925,
            "5 of the 7 splits are on Time.  Only R3–R6 carry a formulation condition;  LA/GA is never used.",
            ha="center", fontsize=12, fontweight="bold", color=ORANGE)

    ax.legend(handles=[
        mpl.patches.Patch(fc="#eceae6", ec=NONE_, lw=2, label="split on Time (5 of 7)"),
        mpl.patches.Patch(fc="#dce9f9", ec=BLUE, lw=2, label="split on a formulation descriptor (2 of 7)"),
        mpl.patches.Patch(fc=GOOD, label="leaf → rule SUPPORTED on unseen drugs"),
        mpl.patches.Patch(fc=WARN, label="leaf → rule BORDERLINE / WEAK"),
        mpl.patches.Patch(fc=NONE_, label="leaf → rule UNSUPPORTED (pure Time)"),
    ], loc="lower center", ncol=5, fontsize=10.5, bbox_to_anchor=(0.5, -0.015))

    fig.savefig(f"{OUT}/01_original_decision_tree.png", dpi=DPI)
    plt.close(fig)
    print("01_original_decision_tree.png")


# ============================================================ FIG 2 — VERDICTS
def fig2():
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.32, 1.0], hspace=0.16)

    # ---- upper: the 8 rules
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 1.045, "Validation of the 8 predefined rules on unseen drugs",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(0.0, 0.995,
            "5-fold GroupKFold grouped on exact drug SMILES  ·  every effect is time-controlled  "
            "·  replicated = same direction in ≥ 3 of 5 held-out folds",
            fontsize=10.5, color=INK2)

    cols = [0.0, 0.058, 0.300, 0.500, 0.715]
    hdr = ["Rule", "Condition", "Coverage (held-out)", "Direction held", "Verdict"]
    yh = 0.905
    for cx, h in zip(cols, hdr):
        ax.text(cx, yh, h.upper(), fontsize=9.5, fontweight="bold", color=INK2)
    ax.plot([0, 1], [yh - 0.028, yh - 0.028], color=GRID, lw=1.4)

    y = yh - 0.075
    for rid in [f"R{i}" for i in range(1, 9)]:
        vlab, vcol, vdet = verdict(rid)
        band = "#f6f5f2" if rid in FORM else SURF
        ax.add_patch(Rectangle((-0.012, y - 0.042), 1.024, 0.088,
                               fc=band, ec="none", zorder=0))
        ax.text(cols[0], y, rid, fontsize=15, fontweight="bold", color=INK, va="center")
        ax.text(cols[1], y, COND[rid], fontsize=12.5, va="center",
                color=INK if rid in FORM else INK2)
        cov = COV[rid].replace("  (", " · ").replace(")", "").replace("  ", " ")
        ax.text(cols[2], y, cov, fontsize=11, va="center", color=INK2)
        ax.text(cols[3], y, SHORT_DIR[rid], fontsize=11, va="center", color=INK2)

        ax.add_patch(FancyBboxPatch((cols[4], y - 0.030), 0.285, 0.060,
                                    boxstyle="round,pad=0.006", fc=vcol, ec="none"))
        icon = {"SUPPORTED": "✔", "BORDERLINE / WEAK": "▲", "UNSUPPORTED": "✖"}[vlab]
        ax.text(cols[4] + 0.142, y, f"{icon}  {vlab}", fontsize=12, fontweight="bold",
                color="#ffffff", ha="center", va="center")
        y -= 0.112

    ax.plot([0, 1], [y + 0.052, y + 0.052], color=GRID, lw=1.4)
    ax.text(0.0, y - 0.005,
            "R3/R4 and R5/R6 are sibling pairs — one formulation claim each, stated from both sides. "
            "They are not independent evidence.",
            fontsize=10.5, color=INK2, style="italic")

    # ---- lower: the 2 underlying claims
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.0, 0.95, "THE TWO UNDERLYING FORMULATION CLAIMS",
             fontsize=12, fontweight="bold", color=INK2)

    cards = [
        (0.0, GOOD, "Polymer MW", "R5 / R6",
         "Lower-MW PLGA (≤ 21 kDa) releases more drug\nwithin the same time window",
         "SUPPORTED", ["direction held in 5 / 5 held-out folds",
                       "mean time-controlled effect  +0.215",
                       "95% CI excludes 0 in 2 / 5 folds",
                       "holds 5/5 in the Middle (diffusion) phase",
                       "crisp and fuzzy firing agree (5/5 both)"]),
        (0.515, WARN, "Particle Size", "R3 / R4",
         "Smaller particles (≤ 9.505 µm) release more drug\nwithin the same time window",
         "BORDERLINE / WEAK", ["direction held in 3 / 3 evaluable folds",
                               "folds 1 & 3: only 2 held-out drugs — not evaluable",
                               "mean effect +0.188, but no fold's CI excludes 0",
                               "effect disappears in the Late (erosion) phase",
                               "meets the ≥ 3/5 criterion exactly — a lead, not a finding"]),
    ]
    for x0, col, title, rules, claim, verd, bullets in cards:
        ax2.add_patch(FancyBboxPatch((x0, 0.02), 0.485, 0.855,
                                     boxstyle="round,pad=0.012", fc="#ffffff", ec=col, lw=2.8))
        ax2.add_patch(Rectangle((x0, 0.735), 0.485, 0.14, fc=col, ec="none"))
        ax2.text(x0 + 0.018, 0.805, f"{title}   ({rules})", fontsize=15,
                 fontweight="bold", color="#ffffff", va="center")
        ax2.text(x0 + 0.467, 0.805, verd, fontsize=12.5, fontweight="bold",
                 color="#ffffff", va="center", ha="right")
        ax2.text(x0 + 0.018, 0.645, claim, fontsize=12, color=INK, va="center")
        yy = 0.50
        for b in bullets:
            ax2.text(x0 + 0.018, yy, "•  " + b, fontsize=10.8, color=INK2, va="center")
            yy -= 0.093

    fig.savefig(f"{OUT}/02_rules_validation_summary.png", dpi=DPI)
    plt.close(fig)
    print("02_rules_validation_summary.png")


# ========================================================= FIG 3 — FOLD BY FOLD
def fig3():
    rules = ["R5", "R6", "R3", "R4"]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16.2, 6.6), gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.20})

    # ---- left: matrix of per-fold results
    axL.set_xlim(-1.25, 6.95); axL.set_ylim(-0.95, len(rules) - 0.25)
    axL.axis("off")
    for j in range(5):
        axL.text(j + 1.4, len(rules) - 0.42, f"Fold {j+1}", ha="center", fontsize=12,
                 fontweight="bold", color=INK2)
    axL.text(6.45, len(rules) - 0.42, "Result", ha="center", fontsize=12,
             fontweight="bold", color=INK2)

    for i, rid in enumerate(rules):
        y = len(rules) - 1 - i
        _, vcol, _ = verdict(rid)
        axL.text(-1.20, y + 0.08, rid, fontsize=16, fontweight="bold", color=INK, va="center")
        axL.text(-1.20, y - 0.19, SHORT_COND[rid], fontsize=10.5, color=INK2, va="center")
        for j in range(5):
            e, ok = EFF[rid][j], OKM[rid][j]
            cx = j + 1.4
            if ok is None:
                axL.add_patch(FancyBboxPatch((cx - 0.40, y - 0.26), 0.80, 0.52,
                                             boxstyle="round,pad=0.02", fc="#f0efec",
                                             ec=GRID, lw=1.6, hatch="//"))
                axL.text(cx, y + 0.03, "n/a", ha="center", va="center",
                         fontsize=12, fontweight="bold", color=NONE_)
                axL.text(cx, y - 0.16, "< 3 drugs", ha="center", va="center",
                         fontsize=8, color=NONE_)
            else:
                axL.add_patch(FancyBboxPatch((cx - 0.40, y - 0.26), 0.80, 0.52,
                                             boxstyle="round,pad=0.02", fc=vcol,
                                             ec="none", alpha=0.20))
                axL.text(cx, y + 0.05, f"{e:+.3f}", ha="center", va="center",
                         fontsize=13, fontweight="bold", color=INK)
                axL.text(cx, y - 0.16, "✔ direction", ha="center", va="center",
                         fontsize=8.5, color=vcol, fontweight="bold")
        n_ok = sum(1 for x in OKM[rid] if x)
        n_ev = sum(1 for x in OKM[rid] if x is not None)
        axL.add_patch(FancyBboxPatch((6.08, y - 0.26), 0.74, 0.52,
                                     boxstyle="round,pad=0.02", fc=vcol, ec="none"))
        axL.text(6.45, y + 0.04, f"{n_ok}/{n_ev}", ha="center", va="center",
                 fontsize=15, fontweight="bold", color="#ffffff")
        axL.text(6.45, y - 0.17, "folds", ha="center", va="center",
                 fontsize=8, color="#ffffff")

    axL.text(-1.25, len(rules) - 0.05,
             "Time-controlled effect on each held-out fold",
             fontsize=15, fontweight="bold", color=INK)
    axL.text(-1.25, -0.62,
             "Polymer MW (R5/R6) reproduced its direction in 5 of 5 folds.",
             fontsize=11, fontweight="bold", color=GOOD)
    axL.text(-1.25, -0.82,
             "Particle Size (R3/R4) in 3 of 3 evaluable folds — folds 1 and 3 had too few\n"
             "held-out drugs to test.",
             fontsize=10.5, color=INK2, va="top")

    # ---- right: effect sizes
    axR.axhline(0, color=INK2, lw=1.4, zorder=1)
    xs = np.arange(1, 6)
    style = {"R5": (GOOD, "o", "R5  Polymer MW ≤ 21"),
             "R6": (GOOD, "s", "R6  Polymer MW > 21"),
             "R3": (WARN, "o", "R3  Particle Size ≤ 9.505"),
             "R4": (WARN, "s", "R4  Particle Size > 9.505")}
    off = {"R5": -0.15, "R6": -0.05, "R3": 0.05, "R4": 0.15}
    for rid, (c, m, lab) in style.items():
        v = np.array(EFF[rid], float)
        axR.plot(xs + off[rid], v, marker=m, ls="none", ms=11, color=c,
                 markeredgecolor=SURF, markeredgewidth=1.6, label=lab, zorder=3)
    axR.set_xticks(xs); axR.set_xticklabels([f"Fold {i}" for i in xs], fontsize=11)
    axR.set_ylabel("time-controlled effect on Release", fontsize=11)
    axR.set_title("Effect size and direction, by held-out fold", fontsize=15, pad=14)
    axR.grid(axis="y", lw=0.8); axR.set_axisbelow(True)
    axR.set_ylim(-0.52, 0.52)
    axR.legend(loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.29))
    axR.text(5.62, 0.36, "higher\nrelease", fontsize=9.5, color=INK2, ha="center", va="center")
    axR.text(5.62, -0.36, "lower\nrelease", fontsize=9.5, color=INK2, ha="center", va="center")

    fig.suptitle("Fold-by-fold validation of the four formulation rules",
                 fontsize=19, fontweight="bold", color=INK, y=1.03)
    fig.savefig(f"{OUT}/03_fold_validation.png", dpi=DPI)
    plt.close(fig)
    print("03_fold_validation.png")


# ========================================================== FIG 4 — TIME DOMINANCE
def fig4():
    proc = pd.read_excel("mp_dataset_processed.xlsx")
    corr = {"Time": abs(np.corrcoef(np.log1p(proc["Time"]), proc["Release"])[0, 1])}
    for c in ["Particle Size", "Polymer MW", "LA/GA"]:
        corr[c] = abs(np.corrcoef(proc[c], proc["Release"])[0, 1])
    # must match the notebook's printed values exactly
    for k_, want in [("Time", 0.6777), ("Particle Size", 0.0392),
                     ("Polymer MW", 0.0347), ("LA/GA", 0.0535)]:
        assert abs(corr[k_] - want) < 5e-4, (k_, corr[k_], want)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.0),
                                 gridspec_kw={"wspace": 0.20})

    a1.scatter(np.log1p(proc["Time"]), proc["Release"], s=9, alpha=0.10,
               color=BLUE, edgecolors="none", rasterized=False)
    q = proc.assign(b=pd.qcut(np.log1p(proc["Time"]), 12, duplicates="drop")) \
            .groupby("b", observed=True)
    a1.plot([np.log1p(x) for x in q["Time"].median()], q["Release"].median(),
            color=ORANGE, marker="o", ms=8, lw=2.8, zorder=3)
    a1.set_xlabel("log(1 + Time in days)", fontsize=12)
    a1.set_ylabel("Release (fraction)", fontsize=12)
    a1.set_title("Release is driven overwhelmingly by elapsed time", fontsize=15, pad=12)
    a1.grid(axis="y", lw=0.8); a1.set_axisbelow(True)
    a1.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ls="", ms=8, alpha=.55, label="measurement (n = 4,913)"),
        Line2D([], [], color=ORANGE, marker="o", ms=8, lw=2.8, label="median by time decile")],
        loc="lower right", fontsize=11)

    names = list(corr); vals = [corr[n] for n in names]
    colr = [ORANGE] + [BLUE] * 3
    b = a2.barh(names[::-1], vals[::-1], color=colr[::-1], height=0.58)
    for rect, v in zip(b, vals[::-1]):
        a2.text(v + 0.014, rect.get_y() + rect.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=12, fontweight="bold", color=INK)
    a2.set_xlim(0, max(vals) * 1.28)
    a2.set_xlabel("|Pearson r| with Release", fontsize=12)
    a2.set_title("Time dwarfs every feature used by the rules", fontsize=15, pad=12)
    a2.grid(axis="x", lw=0.8); a2.set_axisbelow(True)
    a2.tick_params(axis="y", labelsize=12.5)
    a2.text(max(vals) * 0.52, 2.35,
            f"Time's association is\n{corr['Time']/max(corr['Particle Size'], corr['Polymer MW']):.0f}× "
            "the strongest\nrule feature",
            fontsize=12.5, fontweight="bold", color=ORANGE, ha="center", va="center")

    fig.suptitle("The time confound — why every rule effect must be time-controlled",
                 fontsize=19, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.035,
             "Any rule conditioning on Time will appear to work. This is why the four pure-Time rules "
             "(R1, R2, R7, R8) show large uncontrolled effects that vanish under control.",
             ha="center", fontsize=11.5, color=INK2)
    fig.savefig(f"{OUT}/04_time_dominance.png", dpi=DPI)
    plt.close(fig)
    print("04_time_dominance.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print(f"\nsaved to {OUT}")
