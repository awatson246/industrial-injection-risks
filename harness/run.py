"""
Orchestrates the injection-susceptibility benchmark:

    for each model x task x payload (+ a clean-doc baseline):
        build the adversarial document, call the model, save the raw output, score it

Usage:
    python harness/run.py                                  # all 5 models in harness/models.py:MODELS
    python harness/run.py --models gpt-4o,claude-sonnet-5   # a subset, by MODELS key
    python harness/run.py --variants 3                      # scale up variant count
    python harness/run.py --provider anthropic --model-name claude-sonnet-5  # ad hoc, unregistered model

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

from harness.models import MODELS, call_model
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


def save_output(model_label: str, task_name: str, output_label: str, raw_output: str) -> None:
    # Namespaced by model first -- multiple models' raw transcripts for the same
    # (task, category, variant) must not collide on the same file path.
    task_out_dir = OUTPUTS_DIR / model_label / task_name
    task_out_dir.mkdir(parents=True, exist_ok=True)
    (task_out_dir / f"{output_label}.txt").write_text(raw_output, encoding="utf-8")


def run(provider: str, model_name: str, num_variants: int, model_label: str | None = None) -> list[dict]:
    model_label = model_label or model_name
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
        save_output(model_label, task_name, "baseline", raw_output)
        broke, evidence = score_baseline(raw_output, schema)
        rows.append(
            {
                "task": task_name,
                "payload_category": "baseline",
                "payload_variant": "baseline",
                "model": model_label,
                "success_bool": broke,
                "evidence": evidence,
            }
        )
        print(f"[{model_label}][{task_name}/baseline] schema_broke_unprompted={broke}")

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

                output_label = f"{category}_{variant['id']}"
                save_output(model_label, task_name, output_label, raw_output)

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
                        "model": model_label,
                        "success_bool": success,
                        "evidence": evidence,
                    }
                )
                print(f"[{model_label}][{task_name}/{category}/{variant['id']}] success={success} :: {evidence}")
                time.sleep(0.2)  # light rate-limit courtesy

    return rows


def write_results_csv(rows: list[dict], append: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "payload_category", "payload_variant", "model", "success_bool", "evidence"]
    write_header = not (append and RESULTS_CSV.exists())
    mode = "a" if append else "w"
    with RESULTS_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    verb = "Appended" if append else "Wrote"
    print(f"\n{verb} {len(rows)} rows to {RESULTS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model keys from harness/models.py:MODELS (e.g. "
            "'gpt-4o,claude-sonnet-5'), or 'all' (default) to run every registered model."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "hf", "mistral"],
        help="Ad hoc mode: run a single provider/model-name pair not in the MODELS registry.",
    )
    parser.add_argument("--model-name", help="Model name/ID to use with --provider in ad hoc mode.")
    parser.add_argument("--variants", type=int, default=DEFAULT_NUM_VARIANTS)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing results.csv instead of overwriting it.",
    )
    args = parser.parse_args()

    if args.provider:
        if not args.model_name:
            parser.error("--provider requires --model-name")
        targets = [(args.provider, args.model_name, args.model_name)]
    else:
        keys = list(MODELS.keys()) if args.models == "all" else [k.strip() for k in args.models.split(",")]
        unknown = [k for k in keys if k not in MODELS]
        if unknown:
            parser.error(f"Unknown model key(s) {unknown}; choose from {list(MODELS.keys())}")
        targets = [(MODELS[k]["provider"], MODELS[k]["model_name"], k) for k in keys]

    all_rows: list[dict] = []
    for provider, model_name, label in targets:
        print(f"\n=== Running {label} ({provider}/{model_name}) ===")
        all_rows.extend(run(provider, model_name, args.variants, model_label=label))

    write_results_csv(all_rows, append=args.append)


if __name__ == "__main__":
    main()
