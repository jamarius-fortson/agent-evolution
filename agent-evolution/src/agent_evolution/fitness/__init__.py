"""Fitness evaluation for agent architectures."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..genome import ArchitectureGenome


@dataclass
class FitnessFunction:
    """Define how to evaluate an agent architecture's quality."""

    task: str
    criteria: dict[str, dict] = field(default_factory=dict)
    num_evaluations: int = 1
    judge_model: str = "gpt-4o-mini"
    reference_answer: str = ""

    def compute_fitness(self, scores: dict[str, float]) -> float:
        """Compute weighted fitness score from individual criteria scores."""
        total = 0.0
        total_weight = 0.0

        for name, config in self.criteria.items():
            weight = config.get("weight", 1.0)
            score = scores.get(name, 0.0)
            total += weight * score
            total_weight += weight

        return total / total_weight if total_weight > 0 else 0.0


@dataclass
class EvaluationResult:
    """Result from evaluating one architecture on the fitness function."""

    genome_id: str
    fitness_score: float = 0.0
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    raw_scores: dict[str, float] = field(default_factory=dict)
    output: str = ""
    traces: list[dict] = field(default_factory=list) # Full execution log
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_COST_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    for prefix, (inp_rate, out_rate) in _COST_PER_M.items():
        if model.startswith(prefix):
            return (in_tok * inp_rate + out_tok * out_rate) / 1_000_000
    return (in_tok * 3.0 + out_tok * 15.0) / 1_000_000


async def evaluate_genome(
    genome: ArchitectureGenome,
    fitness_fn: FitnessFunction,
) -> EvaluationResult:
    """Execute the architecture and score it against the fitness function."""
    start = time.monotonic()
    result = EvaluationResult(genome_id=genome.id)

    try:
        # Build execution graph from topology
        # We need to run agents when their dependencies are met
        # Simplified: Use a queue and a count of incoming edges
        in_degree = {agent.role: 0 for agent in genome.agents}
        for src, dst in genome.topology:
            if dst in in_degree:
                in_degree[dst] += 1
            elif dst.endswith("_output"):
                # Handle special output markers
                pass

        # Map role to gene
        role_to_gene = {agent.role: agent for agent in genome.agents}
        # Store outputs
        outputs: dict[str, str] = {"initial_task": fitness_fn.task}
        
        # Track finished agents
        finished = set()
        total_in_tokens = 0
        total_out_tokens = 0
        total_cost = 0.0

        async def run_agent(role: str):
            nonlocal total_in_tokens, total_out_tokens, total_cost
            gene = role_to_gene[role]
            
            # Combine inputs from all precursors
            precursors = [src for src, dst in genome.topology if dst == role]
            if not precursors:
                input_text = outputs["initial_task"]
            else:
                input_text = "\n\n".join([f"Input from {p}:\n{outputs[p]}" for p in precursors if p in outputs])

            if gene.system_prompt:
                input_text = f"{gene.system_prompt}\n\n{input_text}"

            out, in_tok, out_tok = await _call_llm(
                gene.model,
                input_text,
                gene.temperature,
                gene.max_tokens,
            )
            outputs[role] = out
            total_in_tokens += in_tok
            total_out_tokens += out_tok
            total_cost += _estimate_cost(gene.model, in_tok, out_tok)
            finished.add(role)

            # Trigger downstream agents
            downstream = [dst for src, dst in genome.topology if src == role]
            next_tasks = []
            for d in downstream:
                if d in in_degree:
                    in_degree[d] -= 1
                    if in_degree[d] == 0:
                        next_tasks.append(run_agent(d))
            
            if next_tasks:
                await asyncio.gather(*next_tasks)

        # Start with agents that have no dependencies
        start_agents = [role for role, degree in in_degree.items() if degree == 0]
        if not start_agents and genome.agents:
            # Fallback if topology is empty or has cycles
            start_agents = [genome.agents[0].role]
            
        await asyncio.gather(*[run_agent(r) for r in start_agents])

        # Final output is from the last agent(s)
        final_roles = [r for r in role_to_gene if r not in [src for src, dst in genome.topology]]
        if not final_roles:
            final_roles = [genome.agents[-1].role]
        
        final_output = "\n\n".join([outputs.get(r, "") for r in final_roles])

        latency = time.monotonic() - start
        result.output = final_output
        result.cost_usd = total_cost
        result.latency_seconds = latency
        # Traces are already captured in run_agent via _call_llm (if we update it)
        # Or better: let's populate them here from a local list
        result.traces = [{"role": r, "output": o} for r, o in outputs.items() if r != "initial_task"]

        # Score each criterion
        scores: dict[str, float] = {}
        for crit_name, crit_config in fitness_fn.criteria.items():
            evaluator = crit_config.get("evaluator", "llm_judge")
            target = crit_config.get("target", "maximize")

            if evaluator == "llm_judge":
                raw = await _llm_judge_score(
                    final_output,
                    fitness_fn.task,
                    crit_config.get("prompt", "Rate quality 0-10"),
                    fitness_fn.judge_model,
                )
                scores[crit_name] = raw / 10.0
                if crit_name == "quality":
                    result.quality_score = raw

            elif evaluator == "token_cost":
                max_cost = crit_config.get("max", 0.50)
                if target == "minimize":
                    scores[crit_name] = max(0, 1.0 - (total_cost / max_cost))
                else:
                    scores[crit_name] = min(total_cost / max_cost, 1.0)

            elif evaluator == "latency":
                max_latency = crit_config.get("max", 60.0)
                if target == "minimize":
                    scores[crit_name] = max(0, 1.0 - (latency / max_latency))
                else:
                    scores[crit_name] = min(latency / max_latency, 1.0)

            elif evaluator == "contains":
                values = crit_config.get("values", [])
                found = sum(1 for v in values if v.lower() in final_output.lower())
                scores[crit_name] = found / max(len(values), 1)

            elif evaluator == "custom":
                func = crit_config.get("function")
                if callable(func):
                    scores[crit_name] = func(final_output)
                else:
                    scores[crit_name] = 0.0

            else:
                scores[crit_name] = 0.0

        result.raw_scores = scores
        result.fitness_score = fitness_fn.compute_fitness(scores)

    except Exception as e:
        logger = logging.getLogger("agent-evolution")
        logger.error(f"Evaluation failed for genome {genome.id}: {e}")
        result.error = str(e)
        result.latency_seconds = time.monotonic() - start

    return result


async def _call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> tuple[str, int, int]:
    """Call an LLM and return (content, input_tokens, output_tokens)."""
    # Mocking check (useful for testing or no-cost runs)
    import os
    if os.getenv("AGENT_EVOLVE_MOCK") == "true":
        await asyncio.sleep(0.5)
        return f"MOCK RESPONSE: {prompt[:50]}...", 50, 150

    provider = "openai"
    if model.startswith("claude-"):
        provider = "anthropic"

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            content = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return content, resp.usage.input_tokens, resp.usage.output_tokens
        else:
            import openai
            client = openai.AsyncOpenAI()
            resp = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = resp.usage
            return (
                resp.choices[0].message.content or "",
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
    except ImportError:
        msg = f"Provider '{provider}' not installed. Run 'pip install {provider}'."
        raise ImportError(msg)


async def _llm_judge_score(
    output: str,
    task: str,
    criteria_prompt: str,
    judge_model: str,
) -> float:
    """Have an LLM judge score the output 0-10."""
    import os
    if os.getenv("AGENT_EVOLVE_MOCK") == "true":
        return 7.5

    try:
        import openai
        client = openai.AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate AI agent outputs. Respond with ONLY a JSON "
                        'object: {"score": <0-10>, "reason": "<brief>"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}\n\n"
                        f"Output to evaluate:\n{output[:3000]}\n\n"
                        f"Criteria: {criteria_prompt}\n\nScore (0-10):"
                    ),
                },
            ],
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            return float(parsed.get("score", 0))
        except (json.JSONDecodeError, ValueError):
            numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
            return float(numbers[0]) if numbers else 0.0
    except ImportError:
        # Fallback if judge provider missing
        return 5.0
