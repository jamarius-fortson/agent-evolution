# Contributing to agent-evolution

## Setup
```bash
git clone https://github.com/daniellopez882/agent-evolution.git
cd agent-evolution
pip install -e ".[dev,all]"
pytest tests/ -v
```

## High-Impact Contributions

- **NSGA-II** — True Pareto multi-objective optimization
- **New operators** — topology-aware crossover, adaptive mutation rates
- **Framework engines** — evaluate via LangGraph/CrewAI directly
- **Visualization** — convergence curves, architecture diagrams
- **Benchmark configs** — standardized evolution configs for common tasks
- **Distributed evolution** — run evaluations across multiple machines

## Code Style
- Python 3.10+ with type hints and dataclasses
- Lint: `ruff check src/ tests/`
- Tests required for all genetic operators
