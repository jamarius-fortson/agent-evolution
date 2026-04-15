"""Tests for agent-evolution core components."""

import json
import random
import pytest

from agent_evolution.genome import AgentGene, ArchitectureGenome
from agent_evolution.space import DesignSpace
from agent_evolution.operators import (
    apply_elitism,
    crossover,
    mutate,
    tournament_select,
    select_parents,
)
from agent_evolution.fitness import FitnessFunction


# ───────────────────────────────────────────
# Genome
# ───────────────────────────────────────────

class TestGenome:
    def test_agent_gene_to_dict(self):
        gene = AgentGene(role="researcher", model="gpt-4o", temperature=0.3)
        d = gene.to_dict()
        assert d["role"] == "researcher"
        assert d["model"] == "gpt-4o"
        assert d["temperature"] == 0.3

    def test_agent_gene_from_dict(self):
        gene = AgentGene.from_dict({"role": "writer", "model": "gpt-4o-mini"})
        assert gene.role == "writer"
        assert gene.model == "gpt-4o-mini"

    def test_genome_creation(self):
        genome = ArchitectureGenome(
            pattern="pipeline",
            agents=[
                AgentGene(role="a", model="gpt-4o"),
                AgentGene(role="b", model="gpt-4o-mini"),
            ],
        )
        assert genome.num_agents == 2
        assert genome.pattern == "pipeline"

    def test_genome_clone_independent(self):
        original = ArchitectureGenome(
            pattern="pipeline",
            agents=[AgentGene(role="a", model="gpt-4o")],
            fitness_score=0.9,
        )
        clone = original.clone()
        assert clone.id != original.id
        assert clone.fitness_score == 0.0  # Reset
        clone.agents[0].model = "changed"
        assert original.agents[0].model == "gpt-4o"  # Independent

    def test_build_pipeline_topology(self):
        genome = ArchitectureGenome(
            pattern="pipeline",
            agents=[
                AgentGene(role="a"),
                AgentGene(role="b"),
                AgentGene(role="c"),
            ],
        )
        genome.build_default_topology()
        assert ("a", "b") in genome.topology
        assert ("b", "c") in genome.topology

    def test_build_debate_topology(self):
        genome = ArchitectureGenome(
            pattern="debate",
            agents=[
                AgentGene(role="for"),
                AgentGene(role="against"),
                AgentGene(role="judge"),
            ],
        )
        genome.build_default_topology()
        assert ("for", "judge") in genome.topology
        assert ("against", "judge") in genome.topology

    def test_build_map_reduce_topology(self):
        genome = ArchitectureGenome(
            pattern="map_reduce",
            agents=[
                AgentGene(role="dispatch"),
                AgentGene(role="w1"),
                AgentGene(role="w2"),
                AgentGene(role="reduce"),
            ],
        )
        genome.build_default_topology()
        assert ("dispatch", "w1") in genome.topology
        assert ("dispatch", "w2") in genome.topology
        assert ("w1", "reduce") in genome.topology
        assert ("w2", "reduce") in genome.topology

    def test_to_dict_serializable(self):
        genome = ArchitectureGenome(
            pattern="pipeline",
            agents=[AgentGene(role="a", model="gpt-4o")],
            fitness_score=0.85,
        )
        d = genome.to_dict()
        json.dumps(d)  # Must not raise
        assert d["pattern"] == "pipeline"
        assert d["fitness"]["score"] == 0.85

    def test_to_agent_compose_yaml(self):
        genome = ArchitectureGenome(
            pattern="pipeline",
            agents=[
                AgentGene(role="researcher", model="gpt-4o", tools=["web_search"]),
                AgentGene(role="writer", model="gpt-4o-mini"),
            ],
            topology=[("researcher", "writer")],
            generation=5,
            fitness_score=0.87,
        )
        compose = genome.to_agent_compose_yaml()
        assert "agents" in compose
        assert "researcher" in compose["agents"]
        assert compose["agents"]["researcher"]["connects_to"] == ["writer"]
        assert compose["agents"]["researcher"]["tools"] == ["web_search"]

    def test_summary_string(self):
        genome = ArchitectureGenome(
            pattern="debate",
            agents=[AgentGene(role="a", model="gpt-4o")],
            fitness_score=0.72,
        )
        s = genome.summary()
        assert "debate" in s
        assert "0.720" in s


# ───────────────────────────────────────────
# Design Space
# ───────────────────────────────────────────

class TestDesignSpace:
    def test_random_genome(self):
        space = DesignSpace(
            num_agents=(2, 4),
            models=["gpt-4o", "gpt-4o-mini"],
            patterns=["pipeline", "debate"],
        )
        genome = space.random_genome()
        assert 2 <= genome.num_agents <= 4
        assert genome.pattern in ["pipeline", "debate"]
        assert all(a.model in ["gpt-4o", "gpt-4o-mini"] for a in genome.agents)

    def test_random_population(self):
        space = DesignSpace(num_agents=(2, 3), models=["gpt-4o-mini"])
        pop = space.random_population(10)
        assert len(pop) == 10
        assert all(isinstance(g, ArchitectureGenome) for g in pop)

    def test_temperature_range(self):
        space = DesignSpace(temperatures=(0.0, 0.5))
        random.seed(42)
        genome = space.random_genome()
        for a in genome.agents:
            assert 0.0 <= a.temperature <= 0.5

    def test_single_model_fixed(self):
        space = DesignSpace(models=["only-this-model"])
        genome = space.random_genome()
        assert all(a.model == "only-this-model" for a in genome.agents)


# ───────────────────────────────────────────
# Operators
# ───────────────────────────────────────────

class TestOperators:
    def _make_genome(self, fitness: float, n_agents: int = 3) -> ArchitectureGenome:
        return ArchitectureGenome(
            pattern="pipeline",
            agents=[AgentGene(role=f"a{i}", model="gpt-4o") for i in range(n_agents)],
            fitness_score=fitness,
        )

    def test_tournament_select(self):
        pop = [self._make_genome(f) for f in [0.1, 0.5, 0.9, 0.3, 0.7]]
        random.seed(42)
        selected = tournament_select(pop, tournament_size=3)
        assert isinstance(selected, ArchitectureGenome)

    def test_select_parents(self):
        pop = [self._make_genome(f) for f in [0.2, 0.4, 0.6, 0.8]]
        parents = select_parents(pop, num_parents=2)
        assert len(parents) == 2

    def test_crossover_produces_child(self):
        space = DesignSpace(num_agents=(2, 4), models=["gpt-4o", "gpt-4o-mini"])
        parent_a = self._make_genome(0.8)
        parent_b = self._make_genome(0.6, n_agents=4)
        random.seed(42)
        child = crossover(parent_a, parent_b, space)
        assert isinstance(child, ArchitectureGenome)
        assert child.id != parent_a.id
        assert 2 <= child.num_agents <= 4

    def test_mutate_changes_something(self):
        space = DesignSpace(
            models=["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
            patterns=["pipeline", "debate"],
        )
        original = self._make_genome(0.5)
        random.seed(1)
        # High mutation rate to ensure something changes
        mutant = mutate(original, space, mutation_rate=0.99)
        assert mutant.id != original.id
        # At least one thing should differ (model, temp, or pattern)
        differences = 0
        if mutant.pattern != original.pattern:
            differences += 1
        if mutant.num_agents != original.num_agents:
            differences += 1
        for i in range(min(len(mutant.agents), len(original.agents))):
            if mutant.agents[i].model != original.agents[i].model:
                differences += 1
            if mutant.agents[i].temperature != original.agents[i].temperature:
                differences += 1
        assert differences > 0

    def test_elitism_preserves_best(self):
        pop = [self._make_genome(f) for f in [0.3, 0.9, 0.1, 0.7, 0.5]]
        elites = apply_elitism(pop, elite_count=2)
        assert len(elites) == 2
        assert elites[0].fitness_score == 0.9
        assert elites[1].fitness_score == 0.7


# ───────────────────────────────────────────
# Fitness Function
# ───────────────────────────────────────────

class TestFitnessFunction:
    def test_compute_weighted_fitness(self):
        ff = FitnessFunction(
            task="test",
            criteria={
                "quality": {"weight": 0.6, "evaluator": "llm_judge"},
                "cost": {"weight": 0.4, "evaluator": "token_cost"},
            },
        )
        score = ff.compute_fitness({"quality": 0.8, "cost": 0.6})
        expected = (0.6 * 0.8 + 0.4 * 0.6) / (0.6 + 0.4)
        assert abs(score - expected) < 0.001

    def test_compute_single_criterion(self):
        ff = FitnessFunction(
            task="test",
            criteria={"quality": {"weight": 1.0}},
        )
        assert ff.compute_fitness({"quality": 0.9}) == 0.9

    def test_compute_missing_score(self):
        ff = FitnessFunction(
            task="test",
            criteria={"quality": {"weight": 1.0}},
        )
        assert ff.compute_fitness({}) == 0.0

    def test_compute_empty_criteria(self):
        ff = FitnessFunction(task="test", criteria={})
        assert ff.compute_fitness({"anything": 1.0}) == 0.0
