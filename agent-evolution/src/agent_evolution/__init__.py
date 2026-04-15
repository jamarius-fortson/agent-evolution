"""agent-evolution: Let evolution design your agent architecture."""

from .genome import AgentGene, ArchitectureGenome
from .space import DesignSpace
from .fitness import FitnessFunction, EvaluationResult
from .search import Evolver, EvolutionResult, GenerationStats

__version__ = "0.1.0"
__all__ = [
    "AgentGene",
    "ArchitectureGenome",
    "DesignSpace",
    "EvaluationResult",
    "Evolver",
    "EvolutionResult",
    "FitnessFunction",
    "GenerationStats",
]
