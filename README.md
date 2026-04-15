# FVRuleLearner

FVRuleLearner is the rule-learning pipeline for `NL2SVA`. The current release mainline is the prompt-only path under `src/`, with JasperGold used for syntax/functionality evaluation.

Paper: https://arxiv.org/abs/2604.03245

BibTeX:

```bibtex
@article{wan2026fvrulelearner,
  title={FVRuleLearner: Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification},
  author={Wan, Lily Jiaxin and Ho, Chia-Tung and Bai, Yunsheng and Yu, Cunxi and Chen, Deming and Ren, Haoxing},
  journal={arXiv preprint arXiv:2604.03245},
  year={2026}
}
```

The supported tasks in this release are:

- `nl2sva_human`
- `nl2sva_machine`
- `nl2sva_opencore`

## Installation

1. Create and activate an environment.

   ```bash
   conda create -n fvrulelearner python=3.10
   conda activate fvrulelearner
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

## Runtime Setup

FVRuleLearner expects:

- either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, depending on `LLM_gateaway`
- `jg` available on `PATH`
- `LM_LICENSE_FILE` and `CDS_LIC_FILE` for JasperGold
- optional `FVRULELEARNER_TRAIN_LOGDIR` to override the training log used by inference
- optional `FVRULELEARNER_EVAL_LOGDIR` to override the inference log used by eval-only runs

Example:

```bash
export OPENAI_API_KEY=...
# or: export ANTHROPIC_API_KEY=...
export PATH=/path/to/jasper/bin:$PATH
export LM_LICENSE_FILE=5280@cadence.webstore.illinois.edu
export CDS_LIC_FILE=5280@cadence.webstore.illinois.edu
```

Run from the repo root:

```bash
cd /home/jiaxin/Documents/FVRuleLearner
```

The main entrypoint is:

```bash
python3 src/main.py
```

Configuration lives in `src/config.py`.

To switch benchmarks, update:

```python
task = "nl2sva_human"
# task = "nl2sva_machine"
# task = "nl2sva_opencore"
```

Dataset mapping in the current release:

- `nl2sva_human` -> `FVEval/data_nl2sva/data/nl2sva_human.csv`
- `nl2sva_machine` -> `FVEval/data_nl2sva/data/nl2sva_machine_updated.csv`
- `nl2sva_opencore` -> `FVEval/data_1k/module_sva_nl_manual_editing.csv`

## Quick Start

### 1. Training

Training learns suggestions and Q-Trees from the selected NL2SVA training split and writes them into a `train_*` log folder.

Use these core settings in `src/config.py`:

```python
global_task = "train"
task = "nl2sva_opencore"

LLM_gateaway = "openai"
llm_model = "gpt-4o"

# or
# LLM_gateaway = "claude"
# llm_model = "claude-sonnet-4-5-20250929"
```

Then run:

```bash
python3 src/main.py
```

For `nl2sva_human` or `nl2sva_machine`, change only the `task` value and keep the rest of the flow the same.

Training outputs are written to:

```text
src/logs/train_<timestamp>_<hostname>_<user>/
```

Important files in that folder:

- `suggestions.pkl`
- `qtrees.pkl`
- `qtrees.json`
- `log.txt`
- `FLAGS.klepto`

### 2. Testing / Inference

Inference uses a previously trained log folder as the rule source.

Use these core settings in `src/config.py`:

```python
global_task = "inference"
task = "nl2sva_opencore"

LLM_gateaway = "openai"
llm_model = "gpt-4o"

# or
# LLM_gateaway = "claude"
# llm_model = "claude-sonnet-4-5-20250929"

use_RAG = True
RAG_content = ["Examples", "Suggestions"]
load_suggestions_path = "/absolute/path/to/src/logs/train_<...>"
```

For the current mainline, the inference settings that were used successfully are:

```python
similarity_str = ["Question"]
retrieve_str = ["Suggestions"]
Examples_top_k = 1
prompting_instruction = 0

Suggestions_top_k = 3
Suggestions_Reasoning = False

use_qtree_inference = True
retrieval_on_ranking = True
qtree_similarity_top_k = 5
qtree_ranking_mode = "prompt"
rule_source = "generate"

deduplication = True
operator_explanation = False
filter_functionality = True
```

Then run:

```bash
python3 src/main.py
```

### Optional Debug Settings

These are only for short smoke tests or targeted debugging. They are not required for normal release runs.

```python
debug = True

# Only meaningful when debug=True.
# Restrict inference/eval to selected dataset row indices.
only_test_ids = [4]

# Optional: restrict the training pool used by retrieval/Q-Tree building.
training_cases = [0, 1, 2, 3]
```

Notes:

- `debug` is optional
- `only_test_ids` is only used when `debug=True`
- `test_mode` is not part of the current release path and is no longer needed

Again, `task` can be switched to `nl2sva_human` or `nl2sva_machine` with the same inference flow.

Inference writes to:

```text
src/logs/inference_<timestamp>_<hostname>_<user>/
```

and automatically runs evaluation at the end.

### 3. Eval Only

If you already have an inference output folder and only want to re-run postprocessing/evaluation:

```python
global_task = "eval"
task = "nl2sva_opencore"
folder_to_eval = "/absolute/path/to/src/logs/inference_<...>"
```

Then run:

```bash
python3 src/main.py
```

## How The PKL Files Are Used

Training saves learned artifacts into the training log folder for the selected task:

- `suggestions.pkl`: learned suggestion traces/rules
- `qtrees.pkl`: serialized Q-Tree corrections
- `qtrees.json`: JSON view of the Q-Trees for inspection/debugging

During inference:

- `load_suggestions_path` should point to the training log directory, not directly to a single `.pkl` file
- the code first looks for `suggestions.pkl`
- if that file is absent, it falls back to `all_suggestions.pkl`
- Q-Tree retrieval also loads `qtrees.json` from the same folder
- `embeddings.npy` may also be created there for retrieval caching

So the flow is:

1. Run `train`
2. Note the generated `src/logs/train_<...>` folder
3. Set `load_suggestions_path` in `inference` to that folder
4. Run `inference`

The training and inference task should match. For example:

- train with `task = "nl2sva_opencore"` -> infer with `task = "nl2sva_opencore"`
- train with `task = "nl2sva_human"` -> infer with `task = "nl2sva_human"`
- train with `task = "nl2sva_machine"` -> infer with `task = "nl2sva_machine"`

## Output Layout

### Training output

```text
src/logs/train_<...>/
  suggestions.pkl
  qtrees.pkl
  qtrees.json
  log.txt
  FLAGS.klepto
```

### Inference output

```text
src/logs/inference_<...>/
  log.txt
  eval/
    *_sim.csv
    *_jg.csv
    *.csv
    suggestion_usage_analysis.json
```

Typical evaluation files:

- `*_sim.csv`: BLEU / ROUGE / exact match
- `*_jg.csv`: JasperGold syntax / functionality
- `*.csv`: merged summary

## Notes

- `inference` already triggers evaluation automatically from `src/main.py`
- `eval` is mainly for re-running evaluation on an existing inference folder
