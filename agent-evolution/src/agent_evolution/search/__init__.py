"""Evolver — the main evolutionary search engine."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

from .fitness import EvaluationResult, FitnessFunction, evaluate_genome
from .genome import ArchitectureGenome
from .operators import apply_elitism, crossover, mutate, select_parents
from .space import DesignSpace

logger = logging.getLogger("agent-evolution")


@dataclass
class GenerationStats:
    """Statistics for one generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_genome_id: str
    population_size: int
    evaluations: int
    cost_usd: float
    elapsed_seconds: float


@dataclass
class EvolutionResult:
    """Final result from an evolution run."""

    best: ArchitectureGenome
    history: list[GenerationStats] = field(default_factory=list)
    all_evaluated: list[ArchitectureGenome] = field(default_factory=list)
    total_evaluations: int = 0
    total_cost_usd: float = 0.0
    total_elapsed_seconds: float = 0.0
    converged_at_generation: Optional[int] = None
    best_trace: list[dict] = field(default_factory=list)

    @property
    def pareto_front(self) -> list[ArchitectureGenome]:
        """Return non-dominated solutions (for multi-objective)."""
        front = []
        for g in self.all_evaluated:
            dominated = False
            for other in self.all_evaluated:
                if (
                    other.quality_score >= g.quality_score
                    and other.cost_usd <= g.cost_usd
                    and other.latency_seconds <= g.latency_seconds
                    and (
                        other.quality_score > g.quality_score
                        or other.cost_usd < g.cost_usd
                        or other.latency_seconds < g.latency_seconds
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(g)
        return sorted(front, key=lambda g: g.quality_score, reverse=True)

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "total_evaluations": self.total_evaluations,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_elapsed_s": round(self.total_elapsed_seconds, 1),
            "converged_at": self.converged_at_generation,
            "history": [
                {
                    "gen": gs.generation,
                    "best": round(gs.best_fitness, 4),
                    "avg": round(gs.avg_fitness, 4),
                    "worst": round(gs.worst_fitness, 4),
                    "cost": round(gs.cost_usd, 4),
                }
                for gs in self.history
            ],
        }


class Evolver:
    """Main evolutionary search engine.

    Usage::

        evolver = Evolver(space, fitness, population_size=20, generations=10)
        result = evolver.run()
        print(result.best.to_dict())
    """

    def __init__(
        self,
        space: DesignSpace,
        fitness: FitnessFunction,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        tournament_size: int = 3,
        strategy: str = "genetic_algorithm",
        max_total_budget: Optional[float] = None,
        max_per_evaluation: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        on_generation: Optional[callable] = None,
    ):
        self.space = space
        self.fitness = fitness
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.strategy = strategy
        self.max_total_budget = max_total_budget
        self.max_per_evaluation = max_per_evaluation
        self.checkpoint_path = checkpoint_path
        self.on_generation = on_generation

        self._total_cost = 0.0
        self._total_evals = 0
        self._all_evaluated: list[ArchitectureGenome] = []
        self._cache: Dict[str, EvaluationResult] = {}
        self._best_trace: list[dict] = []

    def _get_genome_hash(self, genome: ArchitectureGenome) -> str:
        """Generate a stable hash for a genome to use as cache key."""
        # Cleaned dict for hashing (exclude ID and scores)
        data = genome.to_dict()
        data.pop("id", None)
        data.pop("fitness", None)
        data.pop("generation", None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def run(self) -> EvolutionResult:
        """Run the evolutionary search (synchronous wrapper)."""
        return asyncio.run(self._evolve())

    async def _evolve(self) -> EvolutionResult:
        """Main evolutionary loop."""
        start_time = time.monotonic()
        history: list[GenerationStats] = []
        best_ever: Optional[ArchitectureGenome] = None
        converged_at: Optional[int] = None

        # Initialize population
        population = self.space.random_population(self.population_size, generation=0)

        # Semaphore for parallel evaluations
        sema = asyncio.Semaphore(4)  # Default to 4 parallel evals

        for gen in range(self.generations):
            gen_start = time.monotonic()
            gen_cost = 0.0

            # Find genomes that need evaluation
            to_evaluate = []
            for genome in population:
                if genome.evaluation_count == 0:
                    g_hash = self._get_genome_hash(genome)
                    if g_hash in self._cache:
                        # Cache hit!
                        cached = self._cache[g_hash]
                        genome.fitness_score = cached.fitness_score
                        genome.quality_score = cached.quality_score
                        genome.cost_usd = cached.cost_usd
                        genome.latency_seconds = cached.latency_seconds
                        genome.raw_scores = cached.raw_scores
                        genome.evaluation_count = 1
                        genome.generation = gen
                    else:
                        to_evaluate.append((genome, g_hash))

            # Evaluate in parallel
            if to_evaluate:
                async def eval_with_sema(genome, g_hash):
                    async with sema:
                        # Skip if budget already hit
                        if self.max_total_budget and self._total_cost >= self.max_total_budget:
                            return None
                        
                        res = await evaluate_genome(genome, self.fitness)
                        self._cache[g_hash] = res
                        return res

                tasks = [eval_with_sema(g, h) for g, h in to_evaluate]
                results = await asyncio.gather(*tasks)

                for (genome, _), result in zip(to_evaluate, results):
                    if result:
                        genome.fitness_score = result.fitness_score
                        genome.quality_score = result.quality_score
                        genome.cost_usd = result.cost_usd
                        genome.latency_seconds = result.latency_seconds
                        genome.raw_scores = result.raw_scores
                        genome.evaluation_count = 1
                        genome.generation = gen

                        gen_cost += result.cost_usd
                        self._total_cost += result.cost_usd
                        self._total_evals += 1
                        self._all_evaluated.append(genome)
                        
                        # Store best trace
                        if best_ever is None or genome.fitness_score > best_ever.fitness_score:
                            self._best_trace = result.traces

            # Track best
            sorted_pop = sorted(
                population, key=lambda g: g.fitness_score, reverse=True
            )
            current_best = sorted_pop[0]
            if best_ever is None or current_best.fitness_score > best_ever.fitness_score:
                best_ever = current_best

            # Generation stats
            fitnesses = [g.fitness_score for g in population]
            stats = GenerationStats(
                generation=gen,
                best_fitness=max(fitnesses),
                avg_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                best_genome_id=current_best.id,
                population_size=len(population),
                evaluations=self._total_evals,
                cost_usd=gen_cost,
                elapsed_seconds=time.monotonic() - gen_start,
            )
            history.append(stats)

            # Callback
            if self.on_generation:
                self.on_generation(stats, current_best)

            logger.info(
                f"Gen {gen}: best={stats.best_fitness:.3f} "
                f"avg={stats.avg_fitness:.3f} "
                f"cost=${gen_cost:.3f}"
            )

            # Convergence detection
            if gen >= 3:
                recent = [h.best_fitness for h in history[-3:]]
                if max(recent) - min(recent) < 0.005:
                    converged_at = gen
                    logger.info(f"Converged at generation {gen}")

            # Checkpoint
            if self.checkpoint_path and best_ever:
                self._save_checkpoint(gen, population, history, best_ever)

            # Last generation — don't evolve
            if gen == self.generations - 1:
                break

            # Budget check before next generation
            if self.max_total_budget and self._total_cost >= self.max_total_budget:
                break

            # --- Create next generation ---
            population = self._create_next_generation(population, gen)

        elapsed = time.monotonic() - start_time
        return EvolutionResult(
            best=best_ever or population[0],
            history=history,
            all_evaluated=self._all_evaluated,
            total_evaluations=self._total_evals,
            total_cost_usd=self._total_cost,
            total_elapsed_seconds=elapsed,
            converged_at_generation=converged_at,
            best_trace=self._best_trace,
        )

    def _create_next_generation(
        self, population: list[ArchitectureGenome], gen: int
    ) -> list[ArchitectureGenome]:
        """Create the next generation using the selected strategy."""
        if self.strategy == "nsga2":
            return self._create_nsga2_generation(population, gen)
        return self._create_ga_generation(population, gen)

    def _create_ga_generation(
        self, population: list[ArchitectureGenome], gen: int
    ) -> list[ArchitectureGenome]:
        """Classic Genetic Algorithm with Elitism."""
        next_population: list[ArchitectureGenome] = []

        # Elitism — carry over top N unchanged
        elites = apply_elitism(population, self.elitism)
        next_population.extend(elites)

        # Fill remaining slots with offspring
        while len(next_population) < self.population_size:
            if random_chance(self.crossover_rate):
                parents = select_parents(population, 2, self.tournament_size)
                child = crossover(parents[0], parents[1], self.space)
            else:
                parent = select_parents(population, 1, self.tournament_size)[0]
                child = parent.clone()

            child = mutate(child, self.space, self.mutation_rate)
            child.generation = gen + 1
            next_population.append(child)

        return next_population

    def _create_nsga2_generation(
        self, population: list[ArchitectureGenome], gen: int
    ) -> list[ArchitectureGenome]:
        """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""
        # This implementation follows the standard NSGA-II procedure:
        # 1. Non-dominated sorting (Fronts)
        # 2. Crowding distance within fronts
        # 3. Selection based on Rank then Crowding Distance

        # For NSGA-II, we combine parents and offspring to select the next population.
        # But in our current loop structure, we only have the current population.
        # We'll create offspring first, then select from P + Q.
        
        # Step 1: Create offspring Q
        offspring = []
        while len(offspring) < self.population_size:
            parents = select_parents(population, 2, self.tournament_size)
            child = crossover(parents[0], parents[1], self.space)
            child = mutate(child, self.space, self.mutation_rate)
            child.generation = gen + 1
            offspring.append(child)
            
        combined = population + offspring
        
        # Step 2: Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)
        
        next_population = []
        for front in fronts:
            self._assign_crowding_distance(front)
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                # Select based on crowding distance to fill remaining slots
                sorted_front = sorted(
                    front, key=lambda x: getattr(x, "crowding_distance", 0), reverse=True
                )
                needed = self.population_size - len(next_population)
                next_population.extend(sorted_front[:needed])
                break
                
        return next_population

    def _fast_non_dominated_sort(self, population: list[ArchitectureGenome]) -> list[list[ArchitectureGenome]]:
        """Sort population into Pareto fronts."""
        # Simple implementation of fast-non-dominated-sort
        fronts = [[]]
        n = {g.id: 0 for g in population} # Domination count
        s = {g.id: [] for g in population} # Dominated set

        for p in population:
            for q in population:
                if self._dominates(p, q):
                    s[p.id].append(q)
                elif self._dominates(q, p):
                    n[p.id] += 1
            
            if n[p.id] == 0:
                setattr(p, "pareto_rank", 1)
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in s[p.id]:
                    n[q.id] -= 1
                    if n[q.id] == 0:
                        setattr(q, "pareto_rank", i + 2)
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return [f for f in fronts if f]

    def _dominates(self, p: ArchitectureGenome, q: ArchitectureGenome) -> bool:
        """True if p dominates q in multi-objective fitness (Quality, Cost, Latency)."""
        # Quality (maximize), Cost (minimize), Latency (minimize)
        # p dominates q if:
        # 1. p is not worse than q in all objectives
        # 2. p is strictly better than q in at least one objective
        
        # We need to handle cases where scores aren't available
        pq = p.quality_score / 10.0
        pc = p.cost_usd
        pl = p.latency_seconds
        
        qq = q.quality_score / 10.0
        qc = q.cost_usd
        ql = q.latency_seconds

        better_on_any = (pq > qq or pc < qc or pl < ql)
        worse_on_any = (pq < qq or pc > qc or pl > ql)

        return better_on_any and not worse_on_any

    def _assign_crowding_distance(self, front: list[ArchitectureGenome]) -> None:
        """Assign crowding distance for diversity maintenance."""
        if len(front) <= 2:
            for g in front:
                setattr(g, "crowding_distance", float('inf'))
            return

        for g in front:
            setattr(g, "crowding_distance", 0.0)

        # Calculate distances for each objective (Quality, Cost, Latency)
        objectives = [
            ("quality_score", True),   # reverse sort
            ("cost_usd", False),      # normal sort
            ("latency_seconds", False) # normal sort
        ]
        
        for attr, reverse in objectives:
            front.sort(key=lambda x: getattr(x, attr), reverse=reverse)
            setattr(front[0], "crowding_distance", float('inf'))
            setattr(front[-1], "crowding_distance", float('inf'))
            
            val_range = getattr(front[-1], attr) - getattr(front[0], attr)
            if val_range == 0: continue
            
            for i in range(1, len(front) - 1):
                dist = (getattr(front[i+1], attr) - getattr(front[i-1], attr)) / val_range
                setattr(front[i], "crowding_distance", getattr(front[i], "crowding_distance") + dist)

    def _save_checkpoint(
        self,
        generation: int,
        population: list[ArchitectureGenome],
        history: list[GenerationStats],
        best: ArchitectureGenome,
    ) -> None:
        """Save evolution state for resumption."""
        checkpoint = {
            "generation": generation,
            "best": best.to_dict(),
            "total_evals": self._total_evals,
            "total_cost": round(self._total_cost, 4),
            "history": [
                {
                    "gen": s.generation,
                    "best": round(s.best_fitness, 4),
                    "avg": round(s.avg_fitness, 4),
                }
                for s in history
            ],
        }
        Path(self.checkpoint_path).write_text(json.dumps(checkpoint, indent=2))


def random_chance(probability: float) -> bool:
    """Return True with given probability."""
    import random
    return random.random() < probability
