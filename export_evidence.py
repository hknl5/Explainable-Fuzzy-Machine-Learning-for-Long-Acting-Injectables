"""
Export raw evidence from the executed notebook. Nothing is redesigned or recreated.

Two export modes, both loss-free with respect to content:
  * FIGURE outputs -> the PNG bytes embedded in the .ipynb are written straight
    to disk. Byte-identical to what the notebook cell produced.
  * TEXT outputs (stdout and DataFrame reprs) -> the captured text is written
    verbatim as .txt, and typeset to .png in a plain monospace font so it can be
    used as an image. No titles, colours, styling or reordering are added.
"""
import base64, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = "validation_of_8_predefined_fuzzy_rules.ipynb"
OUT = os.path.expanduser("~/Desktop/Rule_Validation_Evidence")
os.makedirs(OUT, exist_ok=True)

nb = json.load(open(NB))
CELLS = nb["cells"]


def cell_by_exec(n):
    for i, c in enumerate(CELLS):
        if c["cell_type"] == "code" and c.get("execution_count") == n:
            return i, c
    raise KeyError(n)


def figures(n):
    """PNG bytes embedded in cell In[n], in output order."""
    _, c = cell_by_exec(n)
    return [base64.b64decode(o["data"]["image/png"])
            for o in c.get("outputs", []) if "image/png" in o.get("data", {})]


def text(n):
    """All non-image output of cell In[n], concatenated in the order shown."""
    _, c = cell_by_exec(n)
    parts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o["text"]))
        elif "image/png" in o.get("data", {}):
            continue
        elif "text/plain" in o.get("data", {}):
            parts.append("".join(o["data"]["text/plain"]))
    return "\n".join(p.rstrip("\n") for p in parts if p.strip())


def save_fig(name, data):
    with open(f"{OUT}/{name}", "wb") as f:
        f.write(data)
    print(f"  {name}   (embedded figure bytes, unmodified)")


def save_text(name, body):
    with open(f"{OUT}/{name}.txt", "w") as f:
        f.write(body + "\n")
    lines = body.split("\n")
    width = max(len(l) for l in lines)
    # monospace typesetting only: 0.098 in per char, 0.20 in per line
    fig = plt.figure(figsize=(max(6.0, width * 0.098) + 0.6,
                              max(1.0, len(lines) * 0.20) + 0.4),
                     facecolor="white")
    fig.text(0.012, 0.985, body, family="DejaVu Sans Mono", fontsize=10,
             va="top", ha="left", color="black", linespacing=1.35)
    fig.savefig(f"{OUT}/{name}.png", dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  {name}.png + {name}.txt   (verbatim text of the cell output)")


print("ITEM 1 — decision tree  [cell In[3]]  ** TEXT OUTPUT, NOT A FIGURE **")
save_text("01_decision_tree_export_text_In3", text(3))

print("ITEM 2 — the 8 predefined rules")
save_text("02_the_8_predefined_rules_In4", text(4))
save_text("02b_rule_structure_table_In5", text(5))

print("ITEM 3 — fold-by-fold validation")
save_text("03_fold_by_fold_results_In12", text(12))
save_fig("03b_per_rule_per_fold_figure_In13.png", figures(13)[0])

print("ITEM 4 — Polymer MW across the 5 folds")
save_text("04_phase_summary_by_claim_In14", text(14))
save_fig("04b_phase_effect_by_fold_figure_In15.png", figures(15)[0])

print("ITEM 5 — time confounding")
save_fig("05_time_dominance_figure_In7.png", figures(7)[0])
save_text("05b_time_correlations_In7", text(7))
save_text("05c_time_control_shrinkage_In13", text(13))

print("ITEM 6 — final summary table")
save_text("06_final_summary_table_In17", text(17))

print(f"\nwritten to {OUT}")
