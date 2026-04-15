"""Genome representation for agent architectures.

An ArchitectureGenome encodes a complete multi-agent system:
- Pattern (pipeline, supervisor, debate, map_reduce)
- List of AgentGenes (model, tools, temperature, prompt)
- Topology edges
- Fitness scores
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentGene:
    """One agent in the architecture — a single gene."""

    role: str
    model: str = "gpt-4o-mini"
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 2000
    system_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "tools": self.tools,
            "temperature": round(self.temperature, 2),
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentGene:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ArchitectureGenome:
    """Complete agent architecture — the unit of evolution."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    pattern: str = "pipeline"
    agents: list[AgentGene] = field(default_factory=list)
    topology: list[tuple[str, str]] = field(default_factory=list)

    # Fitness (filled after evaluation)
    fitness_score: float = 0.0
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    generation: int = 0
    evaluation_count: int = 0
    raw_scores: dict[str, float] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def clone(self) -> ArchitectureGenome:
        """Deep copy for mutation without affecting the original."""
        cloned = copy.deepcopy(self)
        cloned.id = uuid.uuid4().hex[:8]
        cloned.fitness_score = 0.0
        cloned.quality_score = 0.0
        cloned.cost_usd = 0.0
        cloned.latency_seconds = 0.0
        cloned.evaluation_count = 0
        cloned.raw_scores = {}
        return cloned

    def build_default_topology(self) -> None:
        """Construct edges from the pattern and agent list."""
        if not self.agents:
            return

        roles = [a.role for a in self.agents]
        self.topology = []

        if self.pattern == "pipeline":
            for i in range(len(roles) - 1):
                self.topology.append((roles[i], roles[i + 1]))

        elif self.pattern == "supervisor":
            if len(roles) >= 2:
                supervisor = roles[0]
                for worker in roles[1:]:
                    self.topology.append((supervisor, worker))
                # Last worker connects back (for synthesis)
                if len(roles) >= 3:
                    self.topology.append((roles[-1], f"{supervisor}_output"))

        elif self.pattern == "debate":
            if len(roles) >= 3:
                judge = roles[-1]
                for advocate in roles[:-1]:
                    self.topology.append((advocate, judge))

        elif self.pattern == "map_reduce":
            if len(roles) >= 3:
                dispatcher = roles[0]
                reducer = roles[-1]
                for worker in roles[1:-1]:
                    self.topology.append((dispatcher, worker))
                    self.topology.append((worker, reducer))

        elif self.pattern == "parallel_merge":
            if len(roles) >= 2:
                merger = roles[-1]
                for parallel in roles[:-1]:
                    self.topology.append((parallel, merger))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "num_agents": self.num_agents,
            "agents": [a.to_dict() for a in self.agents],
            "topology": self.topology,
            "fitness": {
                "score": round(self.fitness_score, 4),
                "quality": round(self.quality_score, 2),
                "cost_usd": round(self.cost_usd, 4),
                "latency_s": round(self.latency_seconds, 2),
            },
            "generation": self.generation,
        }

    def to_agent_compose_yaml(self) -> dict:
        """Export as agent-compose compatible YAML structure."""
        agents_dict = {}
        for agent in self.agents:
            spec: dict[str, Any] = {"model": agent.model}
            if agent.temperature > 0:
                spec["temperature"] = round(agent.temperature, 2)
            if agent.tools:
                spec["tools"] = agent.tools
            if agent.system_prompt:
                spec["system_prompt"] = agent.system_prompt
            if agent.max_tokens != 2000:
                spec["max_tokens"] = agent.max_tokens

            # Find downstream connections
            connects = [t[1] for t in self.topology if t[0] == agent.role]
            if connects:
                spec["connects_to"] = connects

            agents_dict[agent.role] = spec

        return {
            "name": f"evolved-{self.pattern}-{self.id}",
            "description": (
                f"Discovered by agent-evolution "
                f"(gen {self.generation}, fitness {self.fitness_score:.3f})"
            ),
            "agents": agents_dict,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        models = ", ".join(set(a.model.split("/")[-1] for a in self.agents))
        return (
            f"[{self.id}] {self.pattern} ({self.num_agents} agents: {models}) "
            f"fitness={self.fitness_score:.3f}"
        )
