"""Reporting utilities for evolution results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search import EvolutionResult


def generate_html_report(result: EvolutionResult, output_path: str = "evolution_report.html") -> str:
    """Generate a standalone, interactive HTML report for the evolution run."""
    best = result.best
    history = result.history
    
    # Prepare data for charts
    gen_labels = [h.generation for h in history]
    best_fitness = [h.best_fitness for h in history]
    avg_fitness = [h.avg_fitness for h in history]
    costs = [h.cost_usd for h in history]

    # Pareto Front data
    pareto_data = [
        {"x": round(g.cost_usd, 4), "y": round(g.quality_score, 1), "id": g.id}
        for g in result.pareto_front
    ]
    
    # Safe dump best trace
    trace_json = json.dumps(result.best_trace)

    # Mermaid diagram for best architecture
    mermaid_lines = ["graph LR"]
    mermaid_lines.append(f'  Start(({{{"initial_task"}}}))')
    for agent in best.agents:
        mermaid_lines.append(f'  {agent.role}["{agent.role}<br/><small>{agent.model}</small>"]')
    
    for src, dst in best.topology:
        if dst.endswith("_output"):
            mermaid_lines.append(f'  {src} --> End((Output))')
        else:
            mermaid_lines.append(f'  {src} --> {dst}')
            
    # Connect start to agents with no in-degree
    in_degree = {agent.role: 0 for agent in best.agents}
    for src, dst in best.topology:
        if dst in in_degree:
            in_degree[dst] += 1
    
    for role, degree in in_degree.items():
        if degree == 0:
            mermaid_lines.append(f'  Start --> {role}')

    mermaid_code = "\n".join(mermaid_lines)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Evolution Report - {best.id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'forest' }});
    </script>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        .mermaid {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-green {{ background: #dcfce7; color: #166534; }}

        /* Trace Styles */
        .trace-message {{
            margin-bottom: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            background: #f1f5f9;
            border-left: 4px solid var(--primary);
        }}
        .trace-role {{
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: var(--primary);
            display: flex;
            justify-content: space-between;
        }}
        .trace-content {{
            white-space: pre-wrap;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🧬 Agent Evolution Report</h1>
                <p>Run ID: <code>{best.id}</code> | Best Fitness: <strong>{best.fitness_score:.4f}</strong></p>
            </div>
            <div style="text-align: right">
                <span class="badge badge-green">v0.1.0-FAANG</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Evolution Summary</h3>
                <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>
                        <p><small>Generations</small><br/><span class="stat-value">{len(history)}</span></p>
                    </div>
                    <div>
                        <p><small>Total Evals</small><br/><span class="stat-value">{result.total_evaluations}</span></p>
                    </div>
                    <div>
                        <p><small>Total Cost</small><br/><span class="stat-value">${result.total_cost_usd:.2f}</span></p>
                    </div>
                    <div>
                        <p><small>Avg Time/Eval</small><br/><span class="stat-value">{result.total_elapsed_seconds / max(result.total_evaluations, 1):.1f}s</span></p>
                    </div>
                </div>
            </div>
            <div class="card">
                <h3>Winner Performance</h3>
                <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>
                        <p><small>Quality Score</small><br/><span class="stat-value">{best.quality_score:.1f}/10</span></p>
                    </div>
                    <div>
                        <p><small>Arch Cost</small><br/><span class="stat-value">${best.cost_usd:.3f}</span></p>
                    </div>
                    <div>
                        <p><small>Latency</small><br/><span class="stat-value">{best.latency_seconds:.1f}s</span></p>
                    </div>
                    <div>
                        <p><small>Pattern</small><br/><span style="font-size: 1.2rem; font-weight: 600;">{best.pattern}</span></p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 2rem;">
            <h3>Convergence History</h3>
            <canvas id="fitnessChart" height="100"></canvas>
        </div>

        <div class="card" style="margin-bottom: 2rem;">
            <h3>Pareto Frontier: Quality vs Cost</h3>
            <p><small>Points represent the best trade-offs discovered. Lower cost + Higher quality is better.</small></p>
            <canvas id="paretoChart" height="100"></canvas>
        </div>

        <div class="card" style="margin-bottom: 2rem;">
            <h3>Execution Trace (Winning Run)</h3>
            <div id="traceContainer"></div>
        </div>

        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <h3>Winning Architecture: <code>{best.pattern}</code></h3>
                <div class="mermaid">
{mermaid_code}
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Role</th>
                            <th>Model</th>
                            <th>Temp</th>
                            <th>Tools</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f"<tr><td><code>{a.role}</code></td><td>{a.model}</td><td>{a.temperature:.1f}</td><td>{', '.join(a.tools) if a.tools else '-'}</td></tr>" for a in best.agents])}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h3>Raw Configuration (YAML)</h3>
            <pre style="background: #1e293b; color: #f8fafc; padding: 1rem; border-radius: 8px; overflow-x: auto;">
{json.dumps(best.to_agent_compose_yaml(), indent=2)}
            </pre>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('fitnessChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {gen_labels},
                datasets: [{{
                    label: 'Best Fitness',
                    data: {best_fitness},
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.3
                }}, {{
                    label: 'Avg Fitness',
                    data: {avg_fitness},
                    borderColor: '#94a3b8',
                    borderDash: [5, 5],
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{ display: true, text: 'Fitness Score' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Generation' }}
                    }}
                }}
            }}
        }});

        const p_ctx = document.getElementById('paretoChart').getContext('2d');
        new Chart(p_ctx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Pareto Front (Non-dominated)',
                    data: {json.dumps(pareto_data)},
                    backgroundColor: '#10b981',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Cost (USD)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Quality Score' }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `ID: ${{ctx.raw.id}} | $${{ctx.raw.x.toFixed(3)}} | Score: ${{ctx.raw.y}}`
                        }}
                    }}
                }}
            }}
        }});

        // Render Traces
        const traces = {trace_json}; 
        const traceContainer = document.getElementById('traceContainer');
        
        if (traces.length === 0) {{
            traceContainer.innerHTML = '<p>No traces available for this run.</p>';
        }} else {{
            traces.forEach(t => {{
                const div = document.createElement('div');
                div.className = 'trace-message';
                div.innerHTML = `
                    <div class="trace-role">
                        <span>${{t.role}}</span>
                    </div>
                    <div class="trace-content">${{t.output}}</div>
                `;
                traceContainer.appendChild(div);
            }});
        }}
    </script>
</body>
</html>
    """
    Path(output_path).write_text(html_template)
    return output_path
