"""Standard benchmarks for agent evolution."""

from __future__ import annotations

from typing import Dict, Any
from ..fitness import FitnessFunction

def get_benchmark(name: str) -> FitnessFunction:
    """Return a predefined fitness function for a benchmark task."""
    benchmarks = {
        "research": FitnessFunction(
            task="Research the competitive landscape of AI vector databases in 2026.",
            criteria={
                "quality": {"weight": 0.5, "evaluator": "llm_judge", "prompt": "Score for accuracy, depth and source citations."},
                "cost": {"weight": 0.3, "evaluator": "token_cost", "target": "minimize"},
                "speed": {"weight": 0.2, "evaluator": "latency", "target": "minimize"}
            }
        ),
        "summarization": FitnessFunction(
            task="Summarize the core technical findings of the 'Attention is All You Need' paper into 5 bullet points.",
            criteria={
                "quality": {"weight": 0.6, "evaluator": "llm_judge"},
                "contains": {"weight": 0.4, "evaluator": "contains", "values": ["Transformer", "Self-attention", "Multi-head"]}
            }
        ),
        "code_review": FitnessFunction(
            task="Review this Python function for security vulnerabilities and performance bottlenecks.",
            criteria={
                "quality": {"weight": 0.7, "evaluator": "llm_judge", "prompt": "Identify vulnerabilities like SQL injection or race conditions."},
                "latency": {"weight": 0.3, "evaluator": "latency", "target": "minimize"}
            }
        )
    }
    return benchmarks.get(name, benchmarks["research"])

def list_benchmarks() -> list[str]:
    return ["research", "summarization", "code_review"]
