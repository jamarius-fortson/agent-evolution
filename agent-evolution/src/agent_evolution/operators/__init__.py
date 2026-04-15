"""Genetic operators for evolving agent architectures."""

from __future__ import annotations

import copy
import random
from typing import Optional

from ..genome import AgentGene, ArchitectureGenome
from ..space import DesignSpace


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def tournament_select(
    population: list[ArchitectureGenome],
    tournament_size: int = 3,
) -> ArchitectureGenome:
    """Select one individual via tournament selection."""
    candidates = random.sample(population, min(tournament_size, len(population)))
    return max(candidates, key=lambda g: g.fitness_score)


def select_parents(
    population: list[ArchitectureGenome],
    num_parents: int,
    tournament_size: int = 3,
) -> list[ArchitectureGenome]:
    """Select multiple parents via tournament selection."""
    return [
        tournament_select(population, tournament_size)
        for _ in range(num_parents)
    ]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(
    parent_a: ArchitectureGenome,
    parent_b: ArchitectureGenome,
    space: DesignSpace,
) -> ArchitectureGenome:
    """Create a child by combining traits from two parents.

    Strategy:
    - Pattern from parent_a (or parent_b with 50% chance)
    - Agents from both parents (interleaved)
    - Models may come from either parent
    """
    child = parent_a.clone()

    # 50% chance to take pattern from parent_b
    if random.random() < 0.5:
        child.pattern = parent_b.pattern

    # Interleave agents: take agent i from parent_a if i is even, parent_b if odd
    min_agents = min(len(parent_a.agents), len(parent_b.agents))
    max_agents = max(len(parent_a.agents), len(parent_b.agents))
    target_count = random.randint(
        max(space.num_agents[0], min_agents),
        min(space.num_agents[1], max_agents),
    )

    child_agents = []
    for i in range(target_count):
        if i < len(parent_a.agents) and i < len(parent_b.agents):
            # Both parents have this index — crossover individual gene
            donor = parent_a if random.random() < 0.5 else parent_b
            gene = copy.deepcopy(donor.agents[i])
        elif i < len(parent_a.agents):
            gene = copy.deepcopy(parent_a.agents[i])
        elif i < len(parent_b.agents):
            gene = copy.deepcopy(parent_b.agents[i])
        else:
            # Need more agents — generate a random one
            gene = AgentGene(
                role=f"agent_{i}",
                model=random.choice(space.models),
                tools=random.choice(space.tools),
                temperature=round(random.uniform(*space.temperatures), 2),
            )
        child_agents.append(gene)

    child.agents = child_agents

    # Model crossover: randomly swap some models from parent_b
    for i, agent in enumerate(child.agents):
        if random.random() < 0.3 and i < len(parent_b.agents):
            agent.model = parent_b.agents[i].model

    # Rebuild topology for the new pattern
    child.build_default_topology()
    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def mutate(
    genome: ArchitectureGenome,
    space: DesignSpace,
    mutation_rate: float = 0.3,
) -> ArchitectureGenome:
    """Apply random mutations to an architecture genome.

    Each mutation type is applied independently with probability = mutation_rate.
    """
    mutant = genome.clone()

    # 1. Model swap — change one agent's model
    if random.random() < mutation_rate and mutant.agents:
        idx = random.randrange(len(mutant.agents))
        mutant.agents[idx].model = random.choice(space.models)

    # 2. Tool modification — add or remove a tool
    if random.random() < mutation_rate and mutant.agents:
        idx = random.randrange(len(mutant.agents))
        mutant.agents[idx].tools = random.choice(space.tools)

    # 3. Temperature shift — adjust by ±0.1
    if random.random() < mutation_rate and mutant.agents:
        idx = random.randrange(len(mutant.agents))
        delta = random.uniform(-0.15, 0.15)
        new_temp = mutant.agents[idx].temperature + delta
        new_temp = max(space.temperatures[0], min(space.temperatures[1], new_temp))
        mutant.agents[idx].temperature = round(new_temp, 2)

    # 4. System prompt swap
    if (
        random.random() < mutation_rate
        and space.system_prompt_variants
        and mutant.agents
    ):
        idx = random.randrange(len(mutant.agents))
        mutant.agents[idx].system_prompt = random.choice(
            space.system_prompt_variants
        )

    # 5. Pattern shift — change orchestration pattern
    if random.random() < mutation_rate * 0.5:  # Less frequent
        mutant.pattern = random.choice(space.patterns)
        mutant.build_default_topology()

    # 6. Agent add — insert a new agent
    if (
        random.random() < mutation_rate * 0.3
        and mutant.num_agents < space.num_agents[1]
    ):
        new_gene = AgentGene(
            role=f"evolved_{mutant.num_agents}",
            model=random.choice(space.models),
            tools=random.choice(space.tools),
            temperature=round(random.uniform(*space.temperatures), 2),
        )
        mutant.agents.append(new_gene)
        mutant.build_default_topology()

    # 7. Agent remove — remove a non-essential agent
    if (
        random.random() < mutation_rate * 0.2
        and mutant.num_agents > space.num_agents[0]
    ):
        idx = random.randrange(len(mutant.agents))
        mutant.agents.pop(idx)
        mutant.build_default_topology()

    return mutant


# ---------------------------------------------------------------------------
# Elitism
# ---------------------------------------------------------------------------


def apply_elitism(
    population: list[ArchitectureGenome],
    elite_count: int = 2,
) -> list[ArchitectureGenome]:
    """Return the top N individuals unchanged (elitism)."""
    sorted_pop = sorted(population, key=lambda g: g.fitness_score, reverse=True)
    elites = []
    for g in sorted_pop[:elite_count]:
        elite = copy.deepcopy(g)
        elites.append(elite)
    return elites
