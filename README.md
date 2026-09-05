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

Two injection *methods* are tested per payload:
  - **passive** -- the payload is hidden inline inside the source document itself (the
    upstream-third-party-authored-input threat model).
  - **active** -- the source document stays clean; the payload is appended to the end of the
    fully-built prompt instead, as if inserted at inference time by a compromised
    downstream/automated step (e.g. a tool call, a middleware, an orchestration layer).

Each task is also tested against 5 clean document variants, one authored by each model under
test (`tasks/{task}/docs/{model}.txt`, produced by `harness/generate_docs.py`). Using every
tested model as a document author -- rather than one fixed author for the whole corpus --
avoids giving any model a "home turf" advantage from phrasing that happens to suit its own style.

## Layout

```
tasks/{task_name}/
  clean_doc.txt          hand-written reference example (style/structure template only --
                         not used as a test document once docs/ is populated)
  docs/{model}.txt        5 clean documents, one authored by each model under test
  schema.json, prompt.txt
payloads/                payload_library.json (category -> payload variants)
harness/
  task_config.py         per-task target fields, canary strings, injection token, doc genre
  models.py               call_model(provider, model_name, prompt) -- Anthropic, OpenAI,
                          Hugging Face, and Mistral AI all wired up
  generate_docs.py         has each model author its own clean test document per task
  run.py                  orchestrates task x doc-variant x injection-method x payload x model,
                          saves raw outputs, scores them
  scoring.py               override/suppress/falsify/schema_break/exfiltrate success checks
  summary.py                heatmap tables + heatmap PNGs + text summary
outputs/{model}/{task_name}/{doc_author}/...   raw model responses, one file per run
results/results.csv                            one row per (task, doc_author, injection_method,
                                                payload_category, payload_variant, model, success, evidence)
results/heatmap_task_category.png              task x payload-category heatmap (avg across everything else)
results/heatmap_model_category.png             model x payload-category heatmap -- cross-model comparison
results/heatmap_injectionmethod_category.png   passive vs. active x payload-category heatmap
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

For passive injection, the payload is embedded inline, mid-sentence, inside a plausible document
field (e.g. "Supplementary Remarks: ..."), not as an obviously separate block -- see the
`# --- PAYLOAD INJECTION POINT (passive) ---` marker in `harness/run.py:build_document()`. For
active injection, the same payload text is instead appended after the fully-built prompt -- see
the `# --- PAYLOAD INJECTION POINT (active) ---` marker in `harness/run.py:build_prompt()`.
Extend or swap payloads by editing `payload_library.json` only; the harness logic does not need
to change.

## Models

Five models are evaluated (`harness/models.py:MODELS`):

| Model | Developer | Access |
|---|---|---|
| GPT-4o | OpenAI | API |
| Claude 4.6 Sonnet (`claude-sonnet-5`) | Anthropic | API |
| Meta-Llama-3.1-8B-Instruct | Meta | Open (via HF Inference Providers) |
| Mistral-7B (`open-mistral-7b`) | Mistral AI | Open (via Mistral's own API) |
| Qwen2.5-7B-Instruct | Alibaba / HuggingFace | Open (via HF Inference Providers) |

Mistral-7B is called directly against Mistral AI's own API rather than through Hugging Face:
no HF Inference Providers route currently serves a Mistral-7B-Instruct checkpoint with chat
support. If that changes, it can be re-pointed at `"hf"` like the other two open models.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN, MISTRAL_API_KEY (never commit real keys)
```

`HF_TOKEN` needs the "Make calls to Inference Providers" permission enabled (set when
creating/regenerating the token at https://huggingface.co/settings/tokens) -- without it, Llama
and Qwen calls 403 regardless of the model being valid.

You only need to fill in the keys for the providers you intend to run; `--models` lets you run a
subset (see below).

## Running

```
python harness/generate_docs.py                         # one-time: 30 calls (6 tasks x 5 models),
                                                          # only regenerates missing docs unless --force
python harness/run.py                                    # all 5 models, 2 variants/category/task
python harness/run.py --models gpt-4o,claude-sonnet-5     # only a subset of MODELS
python harness/run.py --variants 3                        # scale up variant count
python harness/run.py --models gpt-4o --append            # add one more model's rows to an existing results.csv
python harness/summary.py                                 # heatmap tables + heatmap PNGs + text summary
```

`NUM_PAYLOAD_VARIANTS` (env var, default 2) or `--variants` controls how many of each payload
category's variants run per task. Default matrix per (model, task, doc variant): 1 baseline +
2 variants x 5 categories x 2 injection methods = 21 calls; x 5 doc variants = 105 calls per
task; x 6 tasks = 630 calls per model; x 5 models = 3,150 calls per full run (plus the one-time
30-call document-generation pass).

## Extending to other providers/models

`harness/models.py` exposes `call_model(provider, model_name, prompt)` plus the `MODELS` registry
mapping a friendly name (used in `results.csv` and `--models`) to a `(provider, model_name)` pair.
Four providers are wired up: `"anthropic"`, `"openai"`, `"hf"` (Hugging Face Inference
Providers), and `"mistral"` (Mistral AI's own API). To add another model, add an entry to
`MODELS`; to add another provider, add a `_call_<provider>()` function in `models.py` and a
branch in `call_model()`. For a one-off model not worth registering, use
`python harness/run.py --provider <p> --model-name <name>`.
