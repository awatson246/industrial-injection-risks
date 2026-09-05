"""
Reads results/results.csv and produces:
  - a printed heatmap-style table: injection success rate by task (rows) x payload
    category (columns), averaged across all evaluated models
  - results/heatmap_task_category.png: the same table rendered as a compliance-heatmap figure
  - a printed heatmap-style table: injection success rate by model (rows) x payload
    category (columns), averaged across all tasks -- the cross-model comparison
  - results/heatmap_model_category.png: the same table rendered as a figure
  - a short printed text summary of the most/least vulnerable task, model, and most/least
    effective payload category

Usage:
    python harness/summary.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
TASK_HEATMAP_PNG = ROOT / "results" / "heatmap_task_category.png"
MODEL_HEATMAP_PNG = ROOT / "results" / "heatmap_model_category.png"

CATEGORY_ORDER = ["override", "suppress", "falsify", "schema_break", "exfiltrate"]


def load_attack_rows() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["success_bool"] = df["success_bool"].astype(str).str.lower().isin(["true", "1"])
    return df[df["payload_category"] != "baseline"].copy()


def build_pivot(df: pd.DataFrame, index: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=index, columns="payload_category", values="success_bool", aggfunc="mean")
    ordered_cols = [c for c in CATEGORY_ORDER if c in pivot.columns]
    pivot = (pivot[ordered_cols] * 100).round(0)
    return pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]


def print_text_heatmap(pivot: pd.DataFrame, index_label: str, title: str) -> None:
    print(f"\n{title}\n")
    col_width = 14
    header = index_label.ljust(28) + "".join(c.ljust(col_width) for c in pivot.columns)
    print(header)
    print("-" * len(header))
    for label, row in pivot.iterrows():
        line = str(label).ljust(28) + "".join(
            (f"{v:.0f}%".ljust(col_width) if pd.notna(v) else "n/a".ljust(col_width))
            for v in row
        )
        print(line)


def plot_heatmap(pivot: pd.DataFrame, title: str, png_path: Path) -> None:
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

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attack success rate (%)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved heatmap figure to {png_path}")


def print_text_summary(task_pivot: pd.DataFrame, model_pivot: pd.DataFrame) -> None:
    task_avg = task_pivot.mean(axis=1).sort_values(ascending=False)
    category_avg = task_pivot.mean(axis=0).sort_values(ascending=False)
    model_avg = model_pivot.mean(axis=1).sort_values(ascending=False)

    print("\n--- Summary ---")
    print(f"Most vulnerable task:   {task_avg.index[0]} ({task_avg.iloc[0]:.0f}% avg success)")
    print(f"Least vulnerable task:  {task_avg.index[-1]} ({task_avg.iloc[-1]:.0f}% avg success)")
    print(f"Most effective payload category:  {category_avg.index[0]} ({category_avg.iloc[0]:.0f}% avg success)")
    print(f"Least effective payload category: {category_avg.index[-1]} ({category_avg.iloc[-1]:.0f}% avg success)")
    if len(model_avg) > 1:
        print(f"Most vulnerable model:  {model_avg.index[0]} ({model_avg.iloc[0]:.0f}% avg success)")
        print(f"Least vulnerable model: {model_avg.index[-1]} ({model_avg.iloc[-1]:.0f}% avg success)")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found -- run harness/run.py first.")

    df = load_attack_rows()
    task_pivot = build_pivot(df, index="task")
    model_pivot = build_pivot(df, index="model")

    n_models = df["model"].nunique()
    task_title = (
        f"Injection success rate (%) by task x payload category (averaged across {n_models} model(s))"
        if n_models > 1
        else "Injection success rate (%) by task x payload category"
    )
    print_text_heatmap(task_pivot, "task", task_title)
    try:
        plot_heatmap(task_pivot, "Prompt-Injection Success Rate by Task x Payload Category", TASK_HEATMAP_PNG)
    except ImportError:
        print("\n(matplotlib not installed -- skipping heatmap PNGs; text tables above still apply)")

    if n_models > 1:
        print_text_heatmap(
            model_pivot, "model", "Injection success rate (%) by model x payload category (averaged across 6 tasks)"
        )
        try:
            plot_heatmap(model_pivot, "Prompt-Injection Success Rate by Model x Payload Category", MODEL_HEATMAP_PNG)
        except ImportError:
            pass

    print_text_summary(task_pivot, model_pivot)


if __name__ == "__main__":
    main()
