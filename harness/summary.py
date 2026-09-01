"""
Reads results/results.csv and produces:
  - a printed heatmap-style table: injection success rate by task (rows) x payload
    category (columns)
  - results/heatmap.png: the same table rendered as a compliance-heatmap-style figure
  - a short printed text summary of the most/least vulnerable task and most/least
    effective payload category

Usage:
    python harness/summary.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
HEATMAP_PNG = ROOT / "results" / "heatmap.png"

CATEGORY_ORDER = ["override", "suppress", "falsify", "schema_break", "exfiltrate"]


def load_attack_rows() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["success_bool"] = df["success_bool"].astype(str).str.lower().isin(["true", "1"])
    return df[df["payload_category"] != "baseline"].copy()


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="task", columns="payload_category", values="success_bool", aggfunc="mean"
    )
    ordered_cols = [c for c in CATEGORY_ORDER if c in pivot.columns]
    return (pivot[ordered_cols] * 100).round(0)


def print_text_heatmap(pivot: pd.DataFrame) -> None:
    print("\nInjection success rate (%) by task x payload category\n")
    col_width = 14
    header = "task".ljust(24) + "".join(c.ljust(col_width) for c in pivot.columns)
    print(header)
    print("-" * len(header))
    for task, row in pivot.iterrows():
        line = task.ljust(24) + "".join(
            (f"{v:.0f}%".ljust(col_width) if pd.notna(v) else "n/a".ljust(col_width))
            for v in row
        )
        print(line)


def plot_heatmap(pivot: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 3, 0.7 * len(pivot.index) + 2))
    data = pivot.values.astype(float)

    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if pd.isna(value):
                continue
            text_color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title("Prompt-Injection Success Rate by Task x Payload Category")
    fig.colorbar(im, ax=ax, label="Attack success rate (%)")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150)
    print(f"\nSaved heatmap figure to {HEATMAP_PNG}")


def print_text_summary(pivot: pd.DataFrame) -> None:
    task_avg = pivot.mean(axis=1).sort_values(ascending=False)
    category_avg = pivot.mean(axis=0).sort_values(ascending=False)

    print("\n--- Summary ---")
    print(f"Most vulnerable task:   {task_avg.index[0]} ({task_avg.iloc[0]:.0f}% avg success)")
    print(f"Least vulnerable task:  {task_avg.index[-1]} ({task_avg.iloc[-1]:.0f}% avg success)")
    print(f"Most effective payload category:  {category_avg.index[0]} ({category_avg.iloc[0]:.0f}% avg success)")
    print(f"Least effective payload category: {category_avg.index[-1]} ({category_avg.iloc[-1]:.0f}% avg success)")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found -- run harness/run.py first.")

    df = load_attack_rows()
    pivot = build_pivot(df)

    print_text_heatmap(pivot)
    try:
        plot_heatmap(pivot)
    except ImportError:
        print("\n(matplotlib not installed -- skipping heatmap.png; text table above still applies)")
    print_text_summary(pivot)


if __name__ == "__main__":
    main()
