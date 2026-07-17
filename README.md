# Trainproof

A deterministic linter for ML training runs. Run it on your dataset, tokenizer, and first-epoch logs — it gives a deterministic PASS/WARN/FAIL verdict with named findings and suggested fixes, BEFORE you burn weeks of GPU time.

## Installation
```bash
pip install .
```

## Usage

Exactly three subcommands:

### 1. Dataset Preflight
```bash
trainproof data /path/to/dataset
```
Checks audio integrity (clipping, silence, duration distribution) and transcript quality (unnormalized text, charset audit, duration correlation).

### 2. Tokenizer Preflight
```bash
trainproof tokenizer my_model.model transcripts.txt
```
Checks vocabulary coverage, tokens-per-second, and suspicious splits on numbers/dates.

### 3. First-Epoch Verification
```bash
trainproof epoch logs/epoch1.jsonl
```
Analyzes loss curves (divergence, flatlines, NaN), grad norms, learning rate response, and throughput.

## Supported log formats
- **Generic JSONL / CSV**
- **HuggingFace** (`trainer_state.json`)
- **Coqui TTS Trainer** (plain text `trainer_0_log.txt`)

*Roadmap: TensorBoard event files planned (v0.2) — Lightning console captures are TTY dumps, not logs, and will not be supported.*

## Verdict Rules
All verdict rules are deterministic thresholds defined centrally in `src/trainproof/rules.py`. Examples include:
- `MAX_CLIPPING_PEAK = 0.99`
- `MAX_LOSS_DIVERGENCE_RATIO = 1.5`
- `MIN_VOCAB_COVERAGE = 0.999`

## Explicit Non-Goals
- No live learning-rate auto-adjustment.
- No PyTorch/Lightning callbacks or framework hooks (log files only).
- No MCP server, no LLM integration.
- No dashboards/wandb-style UI.
- No extra features beyond this spec.
