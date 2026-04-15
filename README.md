<div align="center">

# agent-evolution

**Let evolution design your agent architecture.**

Define a fitness function. Provide a design space. Let genetic algorithms discover
the optimal agent topology, prompts, model assignments, and tool configurations
that you'd never find by hand.

[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue?style=flat-square)](https://pypi.org/project/agent-evolution/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Design Space](#-design-space) · [Fitness Functions](#-fitness-functions) · [Examples](#-examples)

</div>

---

## 🔥 The Problem

Building multi-agent systems today is manual trial-and-error:

*"Should I use 3 agents or 5? Should the researcher use GPT-4o or Claude? Should the analyst get web_search or code_exec? Should I use a Supervisor pattern or Orchestrator-Workers? What temperature works best for the judge?"*

You test a configuration. It takes 2 minutes and costs $0.15. You tweak one variable. Test again. After 50 iterations over 3 days, you've explored 0.001% of the design space and found a "good enough" setup.

**What if evolution could search the space for you?**

```python
from agent_evolution import Evolver, DesignSpace, FitnessFunction

# Define what can vary
space = DesignSpace(
    num_agents=(2, 6),                         # 2 to 6 agents
    models=["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
    patterns=["pipeline", "supervisor", "debate", "map_reduce"],
    tools=[["web_search"], ["code_exec"], ["web_search", "code_exec"], []],
    temperatures=(0.0, 1.0),
)

# Define what "good" means
fitness = FitnessFunction(
    task="Compare LangGraph vs CrewAI for production use",
    criteria={
        "quality": {"weight": 0.5, "evaluator": "llm_judge", "threshold": 7},
        "cost":    {"weight": 0.3, "evaluator": "token_cost", "target": "minimize"},
        "speed":   {"weight": 0.2, "evaluator": "latency", "target": "minimize"},
    },
)

# Let evolution find the best architecture
evolver = Evolver(space, fitness, population_size=20, generations=10)
best = evolver.run()

print(best.architecture)   # The winning topology
print(best.fitness_score)   # How well it performed
print(best.config)          # Full reproducible configuration
```

After 200 evaluations (20 population × 10 generations), evolution discovers an architecture you'd never have designed by hand — and proves it's optimal across quality, cost, and speed simultaneously.

---

## ⚡ Quick Start

### Install

```bash
pip install agent-evolution
```

### Minimal Example

```python
from agent_evolution import Evolver, DesignSpace, FitnessFunction

space = DesignSpace(
    num_agents=(2, 4),
    models=["gpt-4o", "gpt-4o-mini"],
    patterns=["pipeline", "supervisor"],
)

fitness = FitnessFunction(
    task="Summarize the latest developments in AI agent frameworks",
    criteria={"quality": {"weight": 1.0, "evaluator": "llm_judge"}},
)

best = Evolver(space, fitness, population_size=10, generations=5).run()
print(best)
```

### CLI

```bash
# Run evolution from a config file
agent-evolve run --config configs/research.yaml

# Resume a paused evolution
agent-evolve resume --checkpoint evolution_checkpoint.json

# Analyze results
agent-evolve analyze --results evolution_results.json

# View visual report
# Open <config_name>_report.html in your browser

# Bootstrap a new project
agent-evolve init "market-analysis"

# Estimate costs before running
agent-evolve estimate --config market-analysis_config.yaml
```

---

## 🧬 How It Works

```
Generation 0: Random Population of Agent Architectures
┌─────────┐  ┌─────────┐  ┌─────────┐      ┌─────────┐
│ Arch #1 │  │ Arch #2 │  │ Arch #3 │ ...  │Arch #20 │
│ 3 agents│  │ 5 agents│  │ 2 agents│      │ 4 agents│
│ pipeline│  │ debate  │  │ superv. │      │ map_red │
│ gpt-4o  │  │ mixed   │  │ claude  │      │ gpt-mini│
└────┬────┘  └────┬────┘  └────┬────┘      └────┬────┘
     │            │            │                 │
     ▼            ▼            ▼                 ▼
┌────────────────────────────────────────────────────────┐
│              FITNESS EVALUATION                         │
│  Run each architecture on the task                      │
│  Score: quality (LLM judge) × cost × speed              │
│  Arch #3 scores highest: 0.87                           │
└────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────┐
│              SELECTION + REPRODUCTION                   │
│  Top 50% survive (tournament selection)                 │
│  Crossover: Arch #3's topology + Arch #7's models       │
│  Mutation: Random model swap, tool addition, temp shift  │
└────────────────────────────────────────────────────────┘
     │
     ▼
Generation 1: Evolved Population
┌─────────┐  ┌─────────┐  ┌─────────┐      ┌─────────┐
│ Arch #3 │  │ Child 1 │  │ Child 2 │ ...  │Mutant 5 │
│(elite)  │  │(cross.) │  │(cross.) │      │(mutated)│
└─────────┘  └─────────┘  └─────────┘      └─────────┘
     │
     ▼
  ... repeat for N generations ...
     │
     ▼
┌────────────────────────────────────────────────────────┐
│              BEST ARCHITECTURE FOUND                    │
│  Pattern: supervisor (3 agents)                         │
│  Router: gpt-4o-mini (temp 0.0)                         │
│  Worker 1: claude-sonnet-4 + web_search (temp 0.2)      │
│  Worker 2: gpt-4o + code_exec (temp 0.1)                │
│  Fitness: 0.94 (quality 9.1/10, $0.034, 6.2s)           │
└────────────────────────────────────────────────────────┘
```

### The Evolutionary Loop

1. **Initialize** — Generate a random population of agent architectures from the design space
2. **Evaluate** — Run each architecture on the target task, measure fitness (quality + cost + speed)
3. **Select** — Tournament selection picks the fittest architectures
4. **Crossover** — Combine traits from two parent architectures (topology from A, models from B)
5. **Mutate** — Random perturbations: swap a model, add/remove a tool, adjust temperature, restructure topology
6. **Repeat** — New generation. Elitism preserves the best architecture unchanged.
7. **Converge** — After N generations, return the fittest architecture ever seen

---

## 🔧 Design Space

The design space defines what evolution can vary. Every axis is optional — fix what you know, evolve what you don't.

```python
from agent_evolution import DesignSpace

space = DesignSpace(
    # ── Topology ──
    num_agents=(2, 8),              # Min/max agents in the pipeline
    patterns=[                      # Orchestration patterns to try
        "pipeline",                 # A → B → C
        "supervisor",               # Supervisor → Workers
        "debate",                   # Advocates → Judge
        "map_reduce",               # Fan-out → Reduce
        "parallel_merge",           # Independent → Merge
    ],

    # ── Models ──
    models=[                        # Models agents can use
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-flash",
    ],
    temperatures=(0.0, 1.0),        # Temperature range (continuous)

    # ── Tools ──
    tools=[                         # Tool sets to assign
        [],                         # No tools
        ["web_search"],
        ["code_exec"],
        ["web_search", "code_exec"],
        ["file_read", "file_write"],
    ],

    # ── Prompting ──
    system_prompt_variants=[        # System prompt options to test
        "You are a thorough researcher. Cite all sources.",
        "You are a concise analyst. Focus on key findings only.",
        "You are a creative thinker. Find unexpected connections.",
    ],

    # ── Constraints ──
    max_tokens_per_agent=(1000, 8000),  # Output length range
    max_total_cost=1.00,                # Hard cost ceiling per evaluation
)
```

### Fixing Variables

Don't want to evolve the model? Fix it:

```python
space = DesignSpace(
    num_agents=(2, 5),
    models=["gpt-4o"],              # Fixed — only one option
    patterns=["pipeline", "supervisor", "debate"],  # Evolve pattern
    temperatures=(0.0, 0.5),        # Evolve temperature
)
```

---

## 📊 Fitness Functions

Fitness functions define what "good" means. They combine multiple objectives into a single scalar score.

### Built-in Evaluators

| Evaluator | What It Measures | Type |
|-----------|-----------------|------|
| `llm_judge` | Output quality scored 0-10 by a judge LLM | Maximize |
| `token_cost` | Total tokens consumed × pricing | Minimize |
| `latency` | Wall-clock execution time | Minimize |
| `contains` | Output contains required keywords | Binary |
| `json_valid` | Output is valid JSON | Binary |
| `similarity` | Semantic similarity to reference answer | Maximize |
| `custom` | Your own Python function | Configurable |

### Multi-Objective Fitness

```python
fitness = FitnessFunction(
    task="Research the competitive landscape of AI agent frameworks in 2026",
    criteria={
        "quality": {
            "weight": 0.5,
            "evaluator": "llm_judge",
            "prompt": "Rate accuracy and depth (0-10)",
            "threshold": 6,          # Minimum acceptable score
        },
        "cost": {
            "weight": 0.3,
            "evaluator": "token_cost",
            "target": "minimize",
            "max": 0.20,             # Hard ceiling: $0.20 per run
        },
        "speed": {
            "weight": 0.2,
            "evaluator": "latency",
            "target": "minimize",
            "max": 30,               # Max 30 seconds
        },
    },
    num_evaluations=3,               # Run each architecture 3x for consistency
    judge_model="gpt-4o-mini",       # Model used for LLM judge evaluations
)
```

### Pareto-Optimal Search

For true multi-objective optimization, use Pareto mode:

```python
evolver = Evolver(
    space, fitness,
    strategy="nsga2",                # Non-dominated Sorting GA
    population_size=30,
    generations=15,
)
results = evolver.run()

# Returns the Pareto frontier — all non-dominated architectures
for arch in results.pareto_front:
    print(f"Quality: {arch.quality:.1f}  Cost: ${arch.cost:.3f}  Speed: {arch.speed:.1f}s")
```

### Custom Evaluators

```python
from agent_evolution import FitnessFunction

def check_citations(output: str) -> float:
    """Score 0-1 based on citation count."""
    import re
    citations = re.findall(r'\[\d+\]', output)
    return min(len(citations) / 5.0, 1.0)

fitness = FitnessFunction(
    task="Write a research report with citations",
    criteria={
        "citations": {
            "weight": 0.4,
            "evaluator": "custom",
            "function": check_citations,
            "target": "maximize",
        },
        "quality": {"weight": 0.6, "evaluator": "llm_judge"},
    },
)
```

---

## 🧪 Architecture Genome

Every agent architecture is encoded as a **genome** — a data structure that genetic operators can manipulate.

```python
@dataclass
class AgentGene:
    """One agent in the architecture."""
    role: str                    # "researcher", "analyst", "writer"
    model: str                   # "gpt-4o", "claude-sonnet-4-20250514"
    tools: list[str]             # ["web_search", "code_exec"]
    temperature: float           # 0.0 - 1.0
    max_tokens: int              # 1000 - 8000
    system_prompt: str           # System message

@dataclass
class ArchitectureGenome:
    """Complete agent architecture — the unit of evolution."""
    pattern: str                 # "pipeline", "supervisor", "debate"
    agents: list[AgentGene]      # The agents in this architecture
    topology: list[tuple]        # (source, target) edges
    
    # Fitness (filled after evaluation)
    fitness_score: float = 0.0
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    generation: int = 0
```

### Genetic Operators

| Operator | What It Does | Example |
|----------|-------------|---------|
| **Crossover (Topology)** | Combine graph structure from two parents | Parent A's pipeline + Parent B's agent models |
| **Crossover (Model)** | Swap model assignments between parents | Agent 1 gets Parent B's model, Agent 2 keeps Parent A's |
| **Mutation (Model Swap)** | Replace one agent's model | gpt-4o → claude-sonnet-4 |
| **Mutation (Tool Add/Remove)** | Add or remove a tool from an agent | +web_search or -code_exec |
| **Mutation (Temperature)** | Shift temperature by ±0.1 | 0.3 → 0.4 |
| **Mutation (Agent Add)** | Insert a new agent into the pipeline | Add a "fact_checker" between researcher and writer |
| **Mutation (Agent Remove)** | Remove an agent from the pipeline | Remove the "editor" agent |
| **Mutation (Pattern Shift)** | Change orchestration pattern | pipeline → supervisor |
| **Mutation (Prompt Swap)** | Replace system prompt variant | "thorough researcher" → "concise analyst" |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     agent-evolution                           │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  Design Space   │  │   Fitness      │  │   Search      │  │
│  │  • Models       │  │   Function     │  │   Strategy    │  │
│  │  • Patterns     │  │   • Quality    │  │   • GA        │  │
│  │  • Tools        │  │   • Cost       │  │   • NSGA-II   │  │
│  │  • Temps        │  │   • Speed      │  │   • Random    │  │
│  │  • Prompts      │  │   • Custom     │  │   • Grid      │  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬───────┘  │
│          │                   │                    │           │
│  ┌───────▼───────────────────▼────────────────────▼───────┐  │
│  │                    Evolver                              │  │
│  │  1. Initialize random population                        │  │
│  │  2. Evaluate fitness (parallel)                         │  │
│  │  3. Select (tournament)                                 │  │
│  │  4. Crossover + Mutate                                  │  │
│  │  5. Checkpoint & log                                    │  │
│  │  6. Repeat → convergence                                │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼────────────────────────────────┐  │
│  │                  Results                                │  │
│  │  • Best architecture (YAML export)                      │  │
│  │  • Fitness history (convergence curve)                   │  │
│  │  • Population diversity metrics                         │  │
│  │  • Pareto frontier (multi-objective)                     │  │
│  │  • Checkpoint (resume later)                             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 📘 Examples

### Example 1: Find the Best Research Pipeline

```bash
agent-evolve run --config configs/research-evolution.yaml
```

```yaml
# configs/research-evolution.yaml
task: "Compare LangGraph vs CrewAI for production multi-agent systems"

design_space:
  num_agents: [2, 5]
  models:
    - gpt-4o
    - gpt-4o-mini
    - claude-sonnet-4-20250514
  patterns:
    - pipeline
    - supervisor
    - debate
  tools:
    - [web_search]
    - [web_search, code_exec]
    - []
  temperatures: [0.0, 0.7]

fitness:
  quality:
    weight: 0.5
    evaluator: llm_judge
    prompt: "Rate for accuracy, depth, and actionable insights (0-10)"
  cost:
    weight: 0.3
    evaluator: token_cost
    target: minimize
  speed:
    weight: 0.2
    evaluator: latency
    target: minimize

evolution:
  population_size: 15
  generations: 8
  mutation_rate: 0.3
  crossover_rate: 0.7
  elitism: 2
  strategy: genetic_algorithm
```

**Output:**
```
agent-evolution v0.1.0 — research-evolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Gen 0: Best 0.62 │ Avg 0.41 │ Pop 15
  Gen 1: Best 0.71 │ Avg 0.53 │ Pop 15
  Gen 2: Best 0.78 │ Avg 0.61 │ Pop 15
  Gen 3: Best 0.83 │ Avg 0.69 │ Pop 15
  Gen 4: Best 0.87 │ Avg 0.74 │ Pop 15
  Gen 5: Best 0.89 │ Avg 0.77 │ Pop 15
  Gen 6: Best 0.91 │ Avg 0.80 │ Pop 15
  Gen 7: Best 0.91 │ Avg 0.82 │ Pop 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Best Architecture (fitness: 0.91)

  Pattern: supervisor
  Agents:
    ├─ router (gpt-4o-mini, temp=0.0)
    ├─ researcher (claude-sonnet-4, temp=0.2, tools=[web_search])
    ├─ analyst (gpt-4o, temp=0.1)
    └─ writer (gpt-4o-mini, temp=0.3)

  Quality: 8.7/10 │ Cost: $0.041 │ Speed: 11.3s
  Total evaluations: 120 │ Total cost: $4.92

  📄 Exported to: best_architecture.yaml
```

### Example 2: Optimize for Minimal Cost

```python
from agent_evolution import Evolver, DesignSpace, FitnessFunction

space = DesignSpace(
    num_agents=(1, 3),
    models=["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash"],
    patterns=["pipeline"],
)

fitness = FitnessFunction(
    task="Summarize this document into 3 bullet points",
    criteria={
        "quality": {"weight": 0.4, "evaluator": "llm_judge", "threshold": 6},
        "cost":    {"weight": 0.6, "evaluator": "token_cost", "target": "minimize"},
    },
)

best = Evolver(space, fitness, population_size=10, generations=5).run()
# Discovers that a single claude-haiku agent at temp 0.0 is optimal
# (adding more agents wastes tokens without improving quality)
```

### Example 3: Pareto Search (Quality vs Cost)

```python
evolver = Evolver(space, fitness, strategy="nsga2", population_size=30, generations=10)
results = evolver.run()

for arch in results.pareto_front:
    print(f"  Quality: {arch.quality_score:.1f}/10  Cost: ${arch.cost_usd:.3f}")

# Pareto frontier:
#   Quality: 9.2/10  Cost: $0.089  (best quality)
#   Quality: 8.5/10  Cost: $0.041  (balanced)
#   Quality: 7.1/10  Cost: $0.008  (cheapest)
```

---

## 🆚 How This Compares

| Tool | Approach | What It Optimizes | Open Source |
|------|----------|------------------|:-:|
| **agent-evolution** | Genetic algorithms on architecture space | Topology + models + tools + prompts + temps | ✅ |
| AgentSquare (2025) | Modular design space + evolutionary search | Component combinations | Research paper |
| ADAS (ICLR'25) | Meta-agent designs new agents in code | Agent code via LLM | Research paper |
| AFlow (ICLR'25) | MCTS over workflow graphs | Workflow structure | Research paper |
| ARTEMIS (2025) | Black-box evolutionary optimization | Agent configs (prompts, params) | Limited release |
| EvoAgentX | Framework for evolving agentic workflows | Workflow + prompts | ✅ (early) |
| DSPy | Compiler-style prompt optimization | Prompts only | ✅ |

**agent-evolution differentiates by:**
1. **Full-stack search**: topology + models + tools + prompts + temperatures — not just prompts
2. **Framework-agnostic**: generates architectures that can run on any framework
3. **Multi-objective**: quality × cost × speed simultaneously via NSGA-II
4. **Execution Graph**: runs independent agents in parallel based on discovered topology
5. **Production-ready**: Parallel evaluation, genome caching, CLI, and premium HTML reports
6. **Simple API**: 10 lines of Python to start evolving

---

## 🛡️ Cost Management

Evolution is expensive — each fitness evaluation runs an agent pipeline. Built-in safeguards:

```python
evolver = Evolver(
    space, fitness,
    population_size=15,
    generations=8,
    # Cost controls
    max_total_budget=10.00,          # Stop if total spend exceeds $10
    max_per_evaluation=0.50,         # Kill any single eval over $0.50
    cache_identical=True,            # Don't re-evaluate identical genomes
    parallel_evaluations=4,          # Run 4 evaluations concurrently
)
```

**Cost estimation before running:**
```bash
agent-evolve estimate --config configs/research.yaml

# Estimated cost:
#   Population: 15 × Generations: 8 = 120 evaluations
#   Avg cost per eval: ~$0.04 (based on design space)
#   Estimated total: $4.80 ± $1.50
#   Estimated time: ~25 minutes
```

---

## 📁 Output Formats

### Best Architecture as YAML (agent-compose compatible)

```yaml
# best_architecture.yaml (auto-generated)
name: evolved-research-pipeline
description: "Discovered by agent-evolution (gen 7, fitness 0.91)"

agents:
  router:
    model: gpt-4o-mini
    temperature: 0.0
    system_prompt: "Classify the query and route to the appropriate specialist."
    connects_to: [researcher, analyst]

  researcher:
    model: claude-sonnet-4-20250514
    temperature: 0.2
    tools: [web_search]
    system_prompt: "You are a thorough researcher. Cite all sources."
    connects_to: [writer]

  analyst:
    model: gpt-4o
    temperature: 0.1
    system_prompt: "You are a concise analyst. Focus on key findings only."
    connects_to: [writer]

  writer:
    model: gpt-4o-mini
    temperature: 0.3
    system_prompt: "Write a clear, actionable executive summary."
    output: report.md
```

### Evolution History as JSON

```json
{
  "generations": 8,
  "total_evaluations": 120,
  "total_cost_usd": 4.92,
  "best_fitness": 0.91,
  "convergence_generation": 6,
  "history": [
    {"generation": 0, "best": 0.62, "avg": 0.41, "worst": 0.18},
    {"generation": 1, "best": 0.71, "avg": 0.53, "worst": 0.29}
  ]
}
```

---

## 🗺️ Roadmap

- [x] Architecture genome representation
- [x] Design space definition
- [x] Genetic operators (crossover, mutation)
- [x] Tournament selection with elitism
- [x] NSGA-II for Pareto optimization (True Multi-Objective)
- [x] Graph-based evaluation engine (Parallel agents!)
- [x] Parallel genome evaluation (asyncio)
- [x] Genome caching (skip duplicates)
- [x] Interactive HTML reports (Mermaid + Chart.js + Trace Explorer)
- [x] Standard Benchmarks (research, summarization, code_review)
- [x] Cost estimation and Bootstrapping commands
- [x] Checkpointing and resume
- [x] Mock provider for no-cost testing

### v0.2 (Planned)
- [ ] LangGraph/CrewAI evaluation engines
- [ ] Convergence curve visualization (advanced analytics)
- [ ] Distributed evolution (Redis/Celery backend)
- [ ] Co-evolution (evolve prompts + architecture jointly)
- [ ] Agent-compose YAML export (native integration)
- [ ] Population diversity metrics (advanced entropy tracking)

### v0.3 (Vision)
- [ ] AlphaEvolve integration for algorithm discovery
- [ ] Self-correcting evolution (agents feedback into mutation)
- [ ] Cross-model knowledge transfer (evolve on mini, deploy on pro)

---

## 📚 Academic References

This tool builds on these research foundations:

- **AgentSquare** (Shang et al., 2025) — Modular design space with evolutionary component search
- **ADAS: Automated Design of Agentic Systems** (Hu et al., ICLR 2025) — Meta-agent that designs agents in code
- **AFlow** (Zhang et al., ICLR 2025) — MCTS over workflow graph structures
- **ARTEMIS** (Brookes et al., 2025) — Black-box evolutionary agent configuration tuning
- **EvoFlow** (Zhang et al., 2025) — Evolutionary heterogeneous workflow construction
- **AlphaEvolve** (Novikov et al., 2025) — Evolutionary coding for algorithm discovery
- **Symbolic Learning for Self-Evolving Agents** (ScienceDirect, 2025) — Framework for agent self-optimization
- **Promptbreeder** (Fernando et al., 2024) — Self-referential prompt evolution

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

**High-impact contributions:**
- **NSGA-II implementation** for true Pareto optimization
- **New genetic operators** — topology-aware crossover, adaptive mutation rates
- **Framework engines** — evaluate architectures via LangGraph/CrewAI directly
- **Visualization** — convergence curves, architecture diagrams, population heatmaps
- **Benchmark tasks** — standardized fitness functions for common use cases

---

## License

[MIT](LICENSE) — Evolve freely.

---

<div align="center">

**[agent-evolution](https://github.com/jamarius-fortson/agent-evolution)** by [Jamarius Fortson](https://github.com/jamarius-fortson/)

*Don't design your agents. Evolve them.*

</div>
