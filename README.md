# industrial-injection-risks

Research test harness measuring how susceptible LLMs are to prompt injection when used for
structured document-extraction tasks in industrial settings (Digital Product Passports, SDS
hazard classification, predictive maintenance, procurement, QC, customs). This is adversarial
robustness benchmarking, analogous to published methodology for testing LLM compliance-artifact
generation under adversarial input.

**All documents are synthetic and fictional.** No real company, product, or proprietary data
appears anywhere in this repo. The harness targets no live system -- it only calls model
provider APIs with locally-authored test documents.

## What it measures

Each task simulates an LLM ingesting a document authored by a third party upstream of the end
user (a supplier, technician, vendor, or inspector) and producing a structured JSON artifact
that a downstream system or person will trust without re-reading the source document. That
trust boundary is the attack surface: an injected instruction hidden inside the source document
can try to make the model override its task, silently drop a safety-relevant field, insert a
false value, break the output schema, or leak content that should stay out of scope.

The extraction prompts used here (`tasks/*/prompt.txt`) are intentionally naive -- they state
the extraction task but include no explicit "ignore instructions embedded in the document"
defense. This models the realistic, underspecified prompts many production extraction
pipelines actually ship with, which is the scenario this benchmark is trying to characterize.

## Layout

```
tasks/{task_name}/       clean_doc.txt, schema.json, prompt.txt   (6 tasks)
payloads/                payload_library.json (category -> payload variants)
harness/
  task_config.py         per-task target fields, canary strings, injection token
  models.py               call_model(provider, model_name, prompt) -- Anthropic wired up,
                          OpenAI / local-HF stubbed with TODOs
  run.py                  orchestrates task x payload x model, saves raw outputs, scores them
  scoring.py               override/suppress/falsify/schema_break/exfiltrate success checks
  summary.py                heatmap table + heatmap.png + text summary
outputs/{task_name}/...   raw model responses, one file per (task, payload) run
results/results.csv       one row per (task, payload_category, payload_variant, model, success, evidence)
results/heatmap.png       task x payload-category success-rate heatmap
```

## Payload taxonomy

`payloads/payload_library.json` holds reusable, task-agnostic payload text (2-3 variants per
category), with `{{FIELD}}` / `{{VALUE}}` / `{{CANARY}}` placeholders filled in per task from
`harness/task_config.py`:

- **override** -- ignore prior instructions, do X instead
- **suppress** -- omit a specific required field
- **falsify** -- insert a specific false value into a specific field
- **schema_break** -- try to make the model emit invalid/malformed output
- **exfiltrate** -- try to make the model leak out-of-scope content (a canary string in an
  "INTERNAL DISTRIBUTION ONLY" section of the document, or the injection payload itself)

Each task's `clean_doc.txt` embeds the payload inline, mid-sentence, inside a plausible field
(e.g. "Supplementary Remarks: ..."), not as an obviously separate block. The substitution point
is marked `# --- PAYLOAD INJECTION POINT ---` in `harness/run.py:build_document()` -- extend or
swap payloads by editing `payload_library.json` only; the harness logic does not need to change.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY (never commit real keys)
```

## Running

```
python harness/run.py                      # smoke test: Anthropic only, 2 variants/category/task
python harness/run.py --variants 3         # scale up variant count
python harness/summary.py                  # heatmap table + heatmap.png + text summary
```

`NUM_PAYLOAD_VARIANTS` (env var, default 2) or `--variants` controls how many of each payload
category's variants run per task. Default matrix per task: 1 baseline + 2 variants x 5
categories = 11 model calls; x 6 tasks = 66 calls per full run.

## Extending to other providers

`harness/models.py` exposes `call_model(provider, model_name, prompt)`. Only `provider="anthropic"`
is implemented; `"openai"` and `"local"/"hf"` raise `NotImplementedError` with a TODO marking
exactly where to add the SDK call, so the harness runs end-to-end on one provider first.
