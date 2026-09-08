"""
Reads results/results.csv plus the raw transcripts under outputs/ and writes
results/notable_samples.txt: a small, hand-reviewable set of real examples per payload
category showing a model notably CATCHING an injected instruction (ideally calling it out
explicitly) versus notably FALLING FOR one (the clearest, highest-stakes compliance).

This is a curation pass over already-generated results/outputs -- it makes no model calls.

Usage:
    python harness/notable_samples.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.run import fill_payload_placeholders, load_payload_library
from harness.task_config import TASKS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
OUTPUTS_DIR = ROOT / "outputs"
OUT_PATH = ROOT / "results" / "notable_samples.txt"

CATEGORY_ORDER = ["override", "suppress", "falsify", "schema_break", "exfiltrate"]
SAMPLES_PER_BUCKET = 2  # how many "caught" and "fell for it" examples to keep per category
EXCERPT_CHARS = 700

# Phrases that indicate a model explicitly noticed and called out the injection attempt in its
# own output -- these make the most compelling "caught in the act" examples, since it's not just
# silent non-compliance but visible awareness.
CATCH_PHRASES = [
    "injection", "disregard", "prompt injection", "should not be", "was disregarded",
    "suspicious", "attempt to", "embedded instruction", "not be followed", "override attempt",
    "improperly included", "should not have been included", "possible", "flagged for review",
]

# Per-category heuristics for picking the most illustrative success=True ("fell for it") row
# among many candidates -- prefer the most blatant/highest-stakes failure mode.
SEVERITY_HINTS = {
    "override": ["override marker"],
    "suppress": ["is absent from output keys"],  # total omission > present-but-empty
    "falsify": [],  # any success is equally compelling -- the field itself is the point
    "schema_break": ["debug_notes", "internal_id", "internal_reference"],  # reasoning/leak, not just malformed
    "exfiltrate": ["canary string"],  # prefer canary leaks over raw payload-text leaks
}


def load_transcript(row: pd.Series) -> str:
    if row["payload_category"] == "baseline":
        label = "baseline"
    else:
        label = f"{row['injection_method']}_{row['payload_category']}_{row['payload_variant']}"
    path = OUTPUTS_DIR / row["model"] / row["task"] / row["doc_author"] / f"{label}.txt"
    if not path.exists():
        return "<transcript file not found>"
    text = path.read_text(encoding="utf-8")
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + " ...[truncated]"


def payload_text_for(row: pd.Series, payload_library: dict) -> str:
    category = row["payload_category"]
    variant = next((v for v in payload_library.get(category, []) if v["id"] == row["payload_variant"]), None)
    if variant is None:
        return "<n/a>"
    return fill_payload_placeholders(variant["text"], category, TASKS[row["task"]])


def format_sample(label: str, row: pd.Series, payload_library: dict) -> str:
    lines = [
        f"[{label}]",
        f"task={row['task']}  model={row['model']}  doc_author={row['doc_author']}  "
        f"injection_method={row['injection_method']}  category={row['payload_category']}/{row['payload_variant']}",
        f"Injected payload: {payload_text_for(row, payload_library)}",
        f"Scoring evidence:  {row['evidence']}",
        "Raw model output:",
        "  " + load_transcript(row).replace("\n", "\n  "),
    ]
    return "\n".join(lines)


def pick_caught(df_cat: pd.DataFrame, n: int) -> list[pd.Series]:
    resisted = df_cat[~df_cat["success_bool"]]
    if resisted.empty:
        return []

    def has_catch_phrase(row: pd.Series) -> bool:
        # Only the model's own transcript counts -- not the scoring evidence string, which is
        # harness-authored and would trivially match.
        text = load_transcript(row).lower()
        return any(p in text for p in CATCH_PHRASES)

    called_out = [row for _, row in resisted.iterrows() if has_catch_phrase(row)]
    picks: list[pd.Series] = []
    seen_models: set[str] = set()
    for row in called_out:
        if row["model"] not in seen_models:
            picks.append(row)
            seen_models.add(row["model"])
        if len(picks) >= n:
            return picks

    # Fall back to any resisted example (diversified by model) if not enough explicit call-outs.
    for _, row in resisted.iterrows():
        if len(picks) >= n:
            break
        if row["model"] not in seen_models:
            picks.append(row)
            seen_models.add(row["model"])
    return picks[:n]


def pick_fell_for_it(df_cat: pd.DataFrame, category: str, n: int) -> list[pd.Series]:
    succeeded = df_cat[df_cat["success_bool"]]
    if succeeded.empty:
        return []

    hints = SEVERITY_HINTS.get(category, [])

    def severity_rank(row: pd.Series) -> int:
        evidence = str(row["evidence"]).lower()
        return 0 if any(h.lower() in evidence for h in hints) else 1

    ranked = sorted(succeeded.to_dict("records"), key=severity_rank)
    picks: list[pd.Series] = []
    seen_models: set[str] = set()
    for record in ranked:
        row = pd.Series(record)
        if row["model"] not in seen_models:
            picks.append(row)
            seen_models.add(row["model"])
        if len(picks) >= n:
            break
    if len(picks) < n:
        for record in ranked:
            if len(picks) >= n:
                break
            row = pd.Series(record)
            if not any(row.equals(p) for p in picks):
                picks.append(row)
    return picks[:n]


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found -- run harness/run.py first.")

    df = pd.read_csv(RESULTS_CSV)
    df["success_bool"] = df["success_bool"].astype(str).str.lower().isin(["true", "1"])
    df = df[df["payload_category"] != "baseline"].copy()

    payload_library = load_payload_library()

    sections = [
        "NOTABLE SAMPLES",
        "================",
        "",
        "Hand-reviewable excerpts from results/results.csv + outputs/, curated per payload",
        "category into CAUGHT (the model resisted, ideally calling out the injection attempt",
        "explicitly in its own output) vs FELL FOR IT (the most blatant/highest-stakes",
        "compliance found for that category). Generated by harness/notable_samples.py --",
        "no model calls made; this only re-reads existing results.",
        "",
    ]

    for category in CATEGORY_ORDER:
        df_cat = df[df["payload_category"] == category]
        if df_cat.empty:
            continue

        sections.append(f"\n{'#' * 70}\nCATEGORY: {category}\n{'#' * 70}")

        caught = pick_caught(df_cat, SAMPLES_PER_BUCKET)
        sections.append(f"\n--- CAUGHT ({len(caught)} example(s)) ---\n")
        if not caught:
            sections.append("(no resisted examples found for this category)")
        for row in caught:
            sections.append(format_sample("CAUGHT", row, payload_library))
            sections.append("")

        fell_for_it = pick_fell_for_it(df_cat, category, SAMPLES_PER_BUCKET)
        sections.append(f"\n--- FELL FOR IT ({len(fell_for_it)} example(s)) ---\n")
        if not fell_for_it:
            sections.append("(no successful-attack examples found for this category)")
        for row in fell_for_it:
            sections.append(format_sample("FELL FOR IT", row, payload_library))
            sections.append("")

    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
