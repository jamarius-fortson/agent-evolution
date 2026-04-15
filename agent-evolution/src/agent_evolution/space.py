"""Design space — defines what evolution can vary."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .genome import AgentGene, ArchitectureGenome

# Default agent roles by pattern
_PATTERN_ROLES = {
    "pipeline": ["researcher", "analyst", "writer"],
    "supervisor": ["supervisor", "worker_1", "worker_2"],
    "debate": ["advocate_for", "advocate_against", "judge"],
    "map_reduce": ["dispatcher", "worker_1", "worker_2", "reducer"],
    "parallel_merge": ["parallel_1", "parallel_2", "merger"],
}


@dataclass
class DesignSpace:
    """Define the searchable space for agent architectures."""

    num_agents: tuple[int, int] = (2, 5)
    models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    patterns: list[str] = field(
        default_factory=lambda: ["pipeline", "supervisor", "debate"]
    )
    tools: list[list[str]] = field(
        default_factory=lambda: [[], ["web_search"]]
    )
    temperatures: tuple[float, float] = (0.0, 0.7)
    max_tokens_per_agent: tuple[int, int] = (1000, 4000)
    system_prompt_variants: list[str] = field(default_factory=list)

    # Constraints
    max_total_cost: Optional[float] = None

    def random_genome(self, generation: int = 0) -> ArchitectureGenome:
        """Generate a random architecture from this design space."""
        pattern = random.choice(self.patterns)
        n_agents = random.randint(*self.num_agents)

        # Get default roles for this pattern, extend if needed
        base_roles = list(_PATTERN_ROLES.get(pattern, ["agent"]))
        while len(base_roles) < n_agents:
            base_roles.append(f"agent_{len(base_roles)}")
        roles = base_roles[:n_agents]

        agents = []
        for role in roles:
            gene = AgentGene(
                role=role,
                model=random.choice(self.models),
                tools=random.choice(self.tools),
                temperature=round(
                    random.uniform(*self.temperatures), 2
                ),
                max_tokens=random.randint(*self.max_tokens_per_agent),
                system_prompt=(
                    random.choice(self.system_prompt_variants)
                    if self.system_prompt_variants
                    else ""
                ),
            )
            agents.append(gene)

        genome = ArchitectureGenome(
            pattern=pattern,
            agents=agents,
            generation=generation,
        )
        genome.build_default_topology()
        return genome

    def random_population(
        self, size: int, generation: int = 0
    ) -> list[ArchitectureGenome]:
        """Generate a random population of architectures."""
        return [self.random_genome(generation) for _ in range(size)]
