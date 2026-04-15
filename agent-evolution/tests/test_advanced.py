"""Advanced tests for parallel and graph-based execution."""

import pytest
import asyncio
import os
from agent_evolution.genome import AgentGene, ArchitectureGenome
from agent_evolution.search import Evolver
from agent_evolution.space import DesignSpace
from agent_evolution.fitness import FitnessFunction, evaluate_genome

@pytest.mark.asyncio
async def test_graph_execution_parallel():
    """Verify that independent agents run in parallel (simulated)."""
    # Mock LLM calls to take some time
    os.environ["AGENT_EVOLVE_MOCK"] = "true"
    
    # Create a parallel architecture: 
    # Start -> A, Start -> B, (A, B) -> C
    genome = ArchitectureGenome(
        pattern="parallel_merge",
        agents=[
            AgentGene(role="A"),
            AgentGene(role="B"),
            AgentGene(role="C"),
        ]
    )
    genome.build_default_topology()
    
    fitness = FitnessFunction(
        task="Test parallel execution",
        criteria={"quality": {"weight": 1.0}}
    )
    
    import time
    start = time.monotonic()
    result = await evaluate_genome(genome, fitness)
    duration = time.monotonic() - start
    
    # Each mock call takes 0.5s. 
    # A and B are parallel (0.5s total for both).
    # C depends on A and B (another 0.5s).
    # Total should be around 1.0s, NOT 1.5s (sequential).
    
    assert result.error is None
    assert 0.9 <= duration <= 1.2 

@pytest.mark.asyncio
async def test_evolver_parallel():
    """Verify that multiple genomes are evaluated in parallel."""
    os.environ["AGENT_EVOLVE_MOCK"] = "true"
    
    space = DesignSpace(num_agents=(1, 1), models=["gpt-4o-mini"])
    fitness = FitnessFunction(task="test", criteria={"q": {"weight": 1}})
    
    # Pop size 4, 1 generation.
    # Semaphor is 4, so all 4 should run in parallel.
    # Each eval is 0.5s.
    evolver = Evolver(space, fitness, population_size=4, generations=1)
    
    import time
    start = time.monotonic()
    result = await evolver._evolve()
    duration = time.monotonic() - start
    
    # Parallel (4 at once): ~0.5s
    # Sequential (4 in a row): ~2.0s
    assert duration < 1.0

def test_genome_caching():
    """Verify that identical genomes are not re-evaluated."""
    os.environ["AGENT_EVOLVE_MOCK"] = "true"
    
    space = DesignSpace(num_agents=(1, 1), models=["gpt-4o-mini"])
    fitness = FitnessFunction(task="test", criteria={"q": {"weight": 1}})
    
    evolver = Evolver(space, fitness, population_size=2, generations=1)
    
    # Manually create two identical genomes
    g1 = space.random_genome()
    g2 = g1.clone() # clone resets scores but keeps genes
    
    # They should have the same hash
    assert evolver._get_genome_hash(g1) == evolver._get_genome_hash(g2)
    
    # If we evaluate g1, then g2 should hit the cache
    async def run_test():
        # Gen 0 evaluation
        population = [g1, g2]
        # Simulate the evaluation logic in _evolve
        for g in population:
            h = evolver._get_genome_hash(g)
            if h in evolver._cache:
                # Cache hit
                res = evolver._cache[h]
                g.fitness_score = res.fitness_score
                g.evaluation_count = 1
            else:
                res = await evaluate_genome(g, fitness)
                evolver._cache[h] = res
                g.fitness_score = res.fitness_score
                g.evaluation_count = 1
        
        return len(evolver._cache)

    num_cached = asyncio.run(run_test())
    assert num_cached == 1 # Only one actual evaluation for two identical genomes
