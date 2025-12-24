# ERR-EVAL Benchmark Engine

This directory contains the evaluation engine for the ERR-EVAL benchmark.

## Quick Start

### 1. Install Dependencies

```bash
cd bench
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -e .
```

### 2. Configure API Key

Copy the example environment file and add your OpenRouter API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get your API key at: https://openrouter.ai/keys

### 3. Run Evaluations

**Single Model:**
```bash
python -m erreval evaluate --model "openai/gpt-4o" --limit 25
```

**All Models (from config/models.yaml):**
```bash
python -m erreval run-all --skip-existing
```

## Project Structure

```
bench/
├── config/
│   └── models.yaml      # Model configurations
├── data/
│   └── canonical/       # Benchmark test items (JSONL)
│       ├── trackA.jsonl # Noisy Perception
│       ├── trackB.jsonl # Ambiguous Semantics
│       ├── trackC.jsonl # False Premise Traps
│       ├── trackD.jsonl # Underspecified Tasks
│       └── trackE.jsonl # Conflicting Constraints
├── prompts/
│   └── judge_prompt.txt # LLM judge scoring criteria
├── erreval/             # Core evaluation engine
│   ├── cli.py           # CLI commands
│   ├── runner.py        # Evaluation orchestration
│   ├── openrouter.py    # API client
│   ├── scorer.py        # Score calculation
│   ├── reporter.py      # Results output
│   └── models.py        # Pydantic data models
├── .env.example         # Environment template
└── pyproject.toml       # Package configuration
```

## CLI Commands

### `evaluate`
Run evaluation on a single model.

```bash
python -m erreval evaluate --model "anthropic/claude-3.5-sonnet" --limit 25 --seed 42
```

Options:
- `--model`: Model ID (required)
- `--limit`: Items per track (default: 25)
- `--seed`: Random seed (default: 42)
- `--judge`: Judge model (default: openai/gpt-5.2)

### `run-all`
Batch evaluate all enabled models.

```bash
python -m erreval run-all --skip-existing
```

Options:
- `--skip-existing`: Skip models already in results.json

## Output

- **JSON Results**: `../frontend/data/results.json` (leaderboard data)
- **Detailed Reports**: `results/<model>_<seed>.json` and `.md`

## Adding New Models

Edit `config/models.yaml`:

```yaml
models:
  - id: provider/model-name
    name: Friendly Model Name
    enabled: true
```

## License

MIT License - See [LICENSE](../LICENSE) for details.
