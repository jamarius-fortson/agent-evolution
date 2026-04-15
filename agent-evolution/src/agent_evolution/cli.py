"""CLI for agent-evolution."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.live import Live
from rich.status import Status
from rich.tree import Tree

from .fitness import FitnessFunction
from .search import Evolver, GenerationStats
from .space import DesignSpace
from .reporting import generate_html_report
from .benchmarks import get_benchmark, list_benchmarks


def _print_generation(stats: GenerationStats, best) -> None:
    try:
        from rich.console import Console
        console = Console()
        color = "green" if stats.best_fitness > 0.8 else "yellow" if stats.best_fitness > 0.5 else "red"
        console.print(
            f"  Gen {stats.generation:>2}: "
            f"Best [{color}]{stats.best_fitness:.3f}[/{color}] │ "
            f"Avg {stats.avg_fitness:.3f} │ "
            f"Pop {stats.population_size} │ "
            f"${stats.cost_usd:.3f}"
        )
    except ImportError:
        print(
            f"  Gen {stats.generation}: "
            f"Best {stats.best_fitness:.3f} | "
            f"Avg {stats.avg_fitness:.3f}"
        )


@click.group("agent-evolve")
@click.version_option(version="0.1.0", prog_name="agent-evolution")
def cli():
    """Evolutionary agent architecture search."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True),
              help="YAML config file")
@click.option("--benchmark", type=click.Choice(list_benchmarks()),
              help="Run a predefined benchmark task")
@click.option("--output", "-o", default="evolution_results.json",
              help="Output results file")
@click.option("--export-best", default="best_architecture.yaml",
              help="Export best architecture as YAML")
def run(config, benchmark, output, export_best):
    """Run evolutionary search from a config file."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    space_cfg = cfg.get("design_space", {})
    space = DesignSpace(
        num_agents=tuple(space_cfg.get("num_agents", [2, 5])),
        models=space_cfg.get("models", ["gpt-4o-mini"]),
        patterns=space_cfg.get("patterns", ["pipeline", "supervisor"]),
        tools=[t if isinstance(t, list) else [t] for t in space_cfg.get("tools", [[]])],
        temperatures=tuple(space_cfg.get("temperatures", [0.0, 0.7])),
        system_prompt_variants=space_cfg.get("system_prompt_variants", []),
    )

    if benchmark:
        fitness = get_benchmark(benchmark)
        console.print(f"[bold cyan]🚀 Benchmark Mode:[/] [yellow]{benchmark}[/]")
    else:
        fitness_cfg = cfg.get("fitness", {})
        criteria = {}
        for name, crit in fitness_cfg.items():
            if isinstance(crit, dict):
                criteria[name] = crit

        fitness = FitnessFunction(
            task=cfg.get("task", ""),
            criteria=criteria,
            judge_model=cfg.get("judge_model", "gpt-4o-mini"),
        )

    evo_cfg = cfg.get("evolution", {})

    console = Console()
    console.print(f"\n[bold blue]🧬 agent-evolution[/bold blue] — [cyan]{Path(config).stem}[/cyan]")
    console.print("=" * 60)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        gen_task = progress.add_task("[yellow]Evolving architectures...", total=evo_cfg.get("generations", 8))
        
        def on_gen(stats: GenerationStats, best):
            progress.update(gen_task, advance=1, description=f"[yellow]Gen {stats.generation} | Best: {stats.best_fitness:.3f}")
            # Optional: detailed printing if verbosity set
            
        evolver = Evolver(
            space=space,
            fitness=fitness,
            population_size=evo_cfg.get("population_size", 15),
            generations=evo_cfg.get("generations", 8),
            mutation_rate=evo_cfg.get("mutation_rate", 0.3),
            crossover_rate=evo_cfg.get("crossover_rate", 0.7),
            elitism=evo_cfg.get("elitism", 2),
            strategy=evo_cfg.get("strategy", "genetic_algorithm"),
            max_total_budget=evo_cfg.get("max_total_budget"),
            checkpoint_path=evo_cfg.get("checkpoint_path"),
            on_generation=on_gen,
        )

        result = evolver.run()

    # Print best result summary
    console.print("\n" + "━" * 60)
    best = result.best
    
    # Architecture Table
    table = Table(title=f"🏆 Winner: {best.id}", border_style="green")
    table.add_column("Agent", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Temp", justify="right")
    table.add_column("Tools", style="blue")
    
    for agent in best.agents:
        table.add_row(
            agent.role,
            agent.model,
            f"{agent.temperature:.1f}",
            ", ".join(agent.tools) if agent.tools else "-"
        )
    
    # Architecture Tree
    tree = Tree(f"[bold green]Architecture: {best.pattern}[/]")
    
    # Simple tree visualization
    # We find start nodes and build from there
    in_degree = {a.role: 0 for a in best.agents}
    for src, dst in best.topology:
        if dst in in_degree: in_degree[dst] += 1
    
    starts = [r for r, d in in_degree.items() if d == 0]
    processed = set()
    
    def build_tree(node, tree_parent):
        processed.add(node)
        agent = next((a for a in best.agents if a.role == node), None)
        label = f"[cyan]{node}[/] [dim]({agent.model if agent else ''})[/]"
        branch = tree_parent.add(label)
        
        downstream = [dst for src, dst in best.topology if src == node]
        for d in downstream:
            if d not in processed:
                build_tree(d, branch)

    for s in starts:
        build_tree(s, tree)

    console.print("\n[bold]Winning Topology:[/]")
    console.print(tree)
    console.print("")
    
    console.print(Panel(
        f"[bold]Pattern:[/] {best.pattern}\n"
        f"[bold]Fitness:[/] [green]{best.fitness_score:.4f}[/] │ "
        f"[bold]Quality:[/] {best.quality_score:.1f}/10 │ "
        f"[bold]Cost:[/] ${best.cost_usd:.3f} │ "
        f"[bold]Speed:[/] {best.latency_seconds:.1f}s\n"
        f"[bold]Total Spend:[/] ${result.total_cost_usd:.2f} (over {result.total_evaluations} evaluations)",
        title="[green]Final Evaluation Results[/]",
        border_style="green",
    ))

    Path(output).write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"📊 Results JSON → [cyan]{output}[/cyan]")

    # Print Pareto Front if NSGA2
    if evo_cfg.get("strategy") == "nsga2":
        console.print("\n[bold cyan]📐 Pareto Frontier (Non-dominated Solutions):[/]")
        pareto = result.pareto_front[:5] # Top 5
        p_table = Table(border_style="cyan")
        p_table.add_column("ID")
        p_table.add_column("Qual", justify="right")
        p_table.add_column("Cost", justify="right")
        p_table.add_column("Speed", justify="right")
        p_table.add_column("Pattern")
        
        for g in pareto:
            p_table.add_row(
                g.id,
                f"{g.quality_score:.1f}",
                f"${g.cost_usd:.3f}",
                f"{g.latency_seconds:.1f}s",
                g.pattern
            )
        console.print(p_table)
        
        # Diversity Spread
        if len(pareto) > 1:
            spread = max(g.quality_score for g in pareto) - min(g.quality_score for g in pareto)
            console.print(f"  [dim]Pareto Quality Spread: {spread:.1f} pts[/]")

    # Export best as YAML
    compose_yaml = result.best.to_agent_compose_yaml()
    Path(export_best).write_text(yaml.dump(compose_yaml, default_flow_style=False))
    console.print(f"📄 Best architecture configuration → [cyan]{export_best}[/cyan]")
    
    # Generate HTML Report
    report_file = f"{Path(config).stem}_report.html"
    generate_html_report(result, report_file)
    console.print(f"🎨 Visual Evolution Report → [bold green]{report_file}[/bold green]\n")


@cli.command()
@click.option("--results", required=True, type=click.Path(exists=True))
def analyze(results):
    """Analyze evolution results."""
    with open(results) as f:
        data = json.load(f)

    click.echo(f"\nBest fitness: {data['best']['fitness']['score']}")
    click.echo(f"Total evaluations: {data['total_evaluations']}")
    click.echo(f"Total cost: ${data['total_cost_usd']:.2f}")
    click.echo(f"Converged at: gen {data.get('converged_at', 'N/A')}")
    click.echo("\nFitness history:")
    for h in data.get("history", []):
        bar = "█" * int(h["best"] * 30)
        click.echo(f"  Gen {h['gen']:>2}: {h['best']:.3f} {bar}")


@cli.command()
@click.argument("name", default="research")
def init(name):
    """Bootstrap a new evolution config."""
    config = {
        "task": f"Analyze the impact of {name} on the industry.",
        "design_space": {
            "num_agents": [2, 5],
            "models": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
            "patterns": ["pipeline", "supervisor", "debate", "map_reduce"],
            "tools": [["web_search"], ["code_exec"], []],
            "temperatures": [0.0, 0.7]
        },
        "fitness": {
            "quality": {"weight": 0.5, "evaluator": "llm_judge"},
            "cost": {"weight": 0.3, "evaluator": "token_cost", "target": "minimize"},
            "speed": {"weight": 0.2, "evaluator": "latency", "target": "minimize"}
        },
        "evolution": {
            "population_size": 12,
            "generations": 6,
            "strategy": "nsga2"
        }
    }
    filename = f"{name}_config.yaml"
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    from rich.console import Console
    Console().print(f"[bold green]✨ Bootstrapped new config:[/bold green] [cyan]{filename}[/cyan]")


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def estimate(config):
    """Estimate the cost of an evolution run."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    space_cfg = cfg.get("design_space", {})
    evo_cfg = cfg.get("evolution", {})
    
    pop = evo_cfg.get("population_size", 15)
    gens = evo_cfg.get("generations", 8)
    total_evals = pop * gens
    
    # Simple estimate: average $0.10 per run (conservative)
    avg_eval_cost = 0.05
    total_est = total_evals * avg_eval_cost
    
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold yellow]💰 Evolution Cost Estimate:[/bold yellow]")
    console.print(f"  Population: [cyan]{pop}[/]")
    console.print(f"  Generations: [cyan]{gens}[/]")
    console.print(f"  Total Evaluations: [bold]{total_evals}[/]")
    console.print(f"  Estimated Total Cost: [green]~${total_est:.2f}[/]")
    console.print(f"  Estimated Time: [green]~{total_evals * 10 / 60:.1f} minutes[/] (sequential)")
    console.print(f"  Estimated Time: [green]~{total_evals * 2.5 / 60:.1f} minutes[/] (parallel=4)\n")


def main():
    cli()


if __name__ == "__main__":
    main()
