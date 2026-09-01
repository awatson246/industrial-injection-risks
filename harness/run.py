"""
Orchestrates the injection-susceptibility benchmark:

    for each task x payload (+ a clean-doc baseline) x model:
        build the adversarial document, call the model, save the raw output, score it

Usage:
    python harness/run.py
    python harness/run.py --provider anthropic --model claude-sonnet-5 --variants 2

Config:
    NUM_PAYLOAD_VARIANTS (env var, default 2) -- how many variants of each payload category to
    run per task. Keep this small for a fast, hand-checkable first pass; raise it later to scale
    up coverage. Can also be set via --variants.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from harness.models import call_model
from harness.scoring import score_baseline, score_output
from harness.task_config import INJECTION_TOKEN, TASK_NAMES, TASKS

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
PAYLOADS_PATH = ROOT / "payloads" / "payload_library.json"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

DEFAULT_NUM_VARIANTS = int(os.environ.get("NUM_PAYLOAD_VARIANTS", "2"))


def load_task(task_name: str) -> dict:
    task_dir = TASKS_DIR / task_name
    return {
        "clean_doc": (task_dir / "clean_doc.txt").read_text(encoding="utf-8"),
        "schema": json.loads((task_dir / "schema.json").read_text(encoding="utf-8")),
        "prompt_template": (task_dir / "prompt.txt").read_text(encoding="utf-8"),
    }


def load_payload_library() -> dict:
    raw = json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def fill_payload_placeholders(payload_text: str, category: str, task_cfg: dict) -> str:
    # {{FIELD}} means different things depending on which field the category targets:
    # suppress payloads try to hide task_cfg["suppress_field"]; falsify payloads try to
    # overwrite task_cfg["falsify_field"] with task_cfg["falsify_value"].
    field = task_cfg["falsify_field"] if category == "falsify" else task_cfg["suppress_field"]
    return (
        payload_text.replace("{{FIELD}}", field)
        .replace("{{VALUE}}", str(task_cfg["falsify_value"]))
        .replace("{{CANARY}}", task_cfg["canary"])
    )


def build_document(clean_doc: str, injected_text: str) -> str:
    # --- PAYLOAD INJECTION POINT ---
    # This is the single place the harness embeds an (attacker-controlled) payload string
    # inline inside an otherwise-clean synthetic document, in place of INJECTION_TOKEN.
    # To add new payload categories or variants, edit payloads/payload_library.json only --
    # nothing here needs to change.
    return clean_doc.replace(INJECTION_TOKEN, injected_text)


def build_prompt(prompt_template: str, schema: dict, document: str) -> str:
    return prompt_template.format(
        schema_json=json.dumps(schema, indent=2),
        document=document,
    )


def save_output(task_name: str, label: str, raw_output: str) -> None:
    task_out_dir = OUTPUTS_DIR / task_name
    task_out_dir.mkdir(parents=True, exist_ok=True)
    (task_out_dir / f"{label}.txt").write_text(raw_output, encoding="utf-8")


def run(provider: str, model_name: str, num_variants: int) -> list[dict]:
    payload_library = load_payload_library()
    rows: list[dict] = []

    for task_name in TASK_NAMES:
        task = load_task(task_name)
        task_cfg = TASKS[task_name]
        schema = task["schema"]

        # --- Baseline: no payload, sanity check the clean pipeline ---
        baseline_doc = build_document(task["clean_doc"], task_cfg["baseline_filler"])
        baseline_prompt = build_prompt(task["prompt_template"], schema, baseline_doc)
        try:
            raw_output = call_model(provider, model_name, baseline_prompt)
        except Exception as exc:  # noqa: BLE001 -- surfaced into results for later review
            raw_output = f"<ERROR calling model: {exc}>"
        save_output(task_name, "baseline", raw_output)
        broke, evidence = score_baseline(raw_output, schema)
        rows.append(
            {
                "task": task_name,
                "payload_category": "baseline",
                "payload_variant": "baseline",
                "model": model_name,
                "success_bool": broke,
                "evidence": evidence,
            }
        )
        print(f"[{task_name}/baseline] schema_broke_unprompted={broke}")

        # --- Payload categories x variants ---
        for category, variants in payload_library.items():
            for variant in variants[:num_variants]:
                payload_text = fill_payload_placeholders(variant["text"], category, task_cfg)
                doc = build_document(task["clean_doc"], payload_text)
                prompt = build_prompt(task["prompt_template"], schema, doc)

                try:
                    raw_output = call_model(provider, model_name, prompt)
                except Exception as exc:  # noqa: BLE001
                    raw_output = f"<ERROR calling model: {exc}>"

                label = f"{category}_{variant['id']}"
                save_output(task_name, label, raw_output)

                success, evidence = score_output(
                    category=category,
                    raw_output=raw_output,
                    schema=schema,
                    task_cfg=task_cfg,
                    payload_text=payload_text,
                )
                rows.append(
                    {
                        "task": task_name,
                        "payload_category": category,
                        "payload_variant": variant["id"],
                        "model": model_name,
                        "success_bool": success,
                        "evidence": evidence,
                    }
                )
                print(f"[{task_name}/{category}/{variant['id']}] success={success} :: {evidence}")
                time.sleep(0.2)  # light rate-limit courtesy

    return rows


def write_results_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "payload_category", "payload_variant", "model", "success_bool", "evidence"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "local"])
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"))
    parser.add_argument("--variants", type=int, default=DEFAULT_NUM_VARIANTS)
    args = parser.parse_args()

    rows = run(args.provider, args.model, args.variants)
    write_results_csv(rows)


if __name__ == "__main__":
    main()
