# Contributing to ERR-EVAL

Thank you for your interest in contributing to ERR-EVAL! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment:

```bash
cd bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

## Development Workflow

### Running Evaluations

```bash
# Single model
python -m erreval evaluate --model "openai/gpt-4o" --limit 25

# All models
python -m erreval run-all --skip-existing
```

### Code Style

- Use Python 3.10+ features
- Follow PEP 8 guidelines
- Use type hints throughout

### Testing

Before submitting a PR, ensure:
- The code runs without errors
- New test items follow the existing JSONL schema
- Results are validated against the Pydantic models

## Contributing Test Items

If you want to contribute new benchmark items:

1. Follow the existing JSONL schema in `bench/data/canonical/`
2. Each item must have:
   - A clear `user_turn` prompt
   - Well-defined `ambiguity_profile`
   - Specific `gold_behavior` with `must_do` and `must_not_do`
3. Items should target one of the five tracks:
   - Track A: Noisy Perception
   - Track B: Ambiguous Semantics
   - Track C: False Premise Traps
   - Track D: Underspecified Tasks
   - Track E: Conflicting Constraints

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Full error traceback
- Steps to reproduce

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Update documentation if needed
4. Submit a PR with a clear description

## Questions?

Open an issue or reach out to the maintainers.
