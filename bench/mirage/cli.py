"""
CLI commands for MIRAGE benchmark.
"""

from __future__ import annotations
import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


@click.group()
def main():
    """MIRAGE Benchmark - Epistemic Reliability Evaluation"""
    pass


@main.command()
@click.option("--model", "-m", required=True, help="OpenRouter model ID to evaluate")
@click.option("--seed", "-s", default=42, help="Random seed for variants")
@click.option("--tracks", "-t", default=None, help="Comma-separated track letters (e.g., A,B,C)")
@click.option("--limit", "-l", default=None, type=int, help="Max items per track")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--judge", default="openai/gpt-4.1", help="Judge model ID")
def evaluate(model: str, seed: int, tracks: str | None, limit: int | None,
             output: str | None, temperature: float, judge: str):
    """Run MIRAGE evaluation on a model."""
    from .runner import MirageRunner
    from .reporter import generate_results_json, generate_markdown_report, generate_leaderboard_entry, update_leaderboard
    
    console.print(f"[bold blue]MIRAGE Benchmark Evaluation[/bold blue]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Seed: {seed}")
    console.print(f"Judge: {judge}")
    console.print()
    
    track_list = tracks.split(",") if tracks else None
    
    runner = MirageRunner(judge_model=judge)
    
    async def run_with_progress():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=None)
            
            def update_progress(current, total):
                progress.update(task, completed=current, total=total)
            
            result = await runner.run_evaluation(
                model_id=model,
                seed=seed,
                tracks=track_list,
                limit=limit,
                temperature=temperature,
                progress_callback=update_progress,
            )
            
            return result
    
    try:
        result = asyncio.run(run_with_progress())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    
    # Display results
    console.print()
    console.print(f"[bold green]Evaluation Complete![/bold green]")
    console.print()
    
    # Overall score
    console.print(f"[bold]Overall Score: {result.overall_score:.2f} / 10[/bold]")
    console.print()
    
    # Track breakdown table
    table = Table(title="Track Breakdown")
    table.add_column("Track", style="cyan")
    table.add_column("Name")
    table.add_column("Items", justify="right")
    table.add_column("Score", justify="right", style="green")
    
    for ts in result.track_summaries:
        table.add_row(ts.track, ts.track_name, str(ts.item_count), f"{ts.mean_score:.2f}")
    
    console.print(table)
    console.print()
    
    # Save outputs
    if output:
        output_path = Path(output)
    else:
        output_path = Path("results") / f"{model.replace('/', '_')}_{seed}.json"
    
    generate_results_json(result, output_path)
    console.print(f"Results saved to: [cyan]{output_path}[/cyan]")
    
    md_path = output_path.with_suffix(".md")
    generate_markdown_report(result, md_path)
    console.print(f"Report saved to: [cyan]{md_path}[/cyan]")
    
    # Update leaderboard
    leaderboard_path = Path(__file__).parent.parent.parent / "frontend" / "data" / "results.json"
    entry = generate_leaderboard_entry(result)
    update_leaderboard(leaderboard_path, entry)
    console.print(f"Leaderboard updated: [cyan]{leaderboard_path}[/cyan]")


@main.command()
def list_models():
    """List available models from configuration."""
    import yaml
    
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    
    if not config_path.exists():
        console.print("[red]Config file not found[/red]")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for provider_key, provider in config.get("providers", {}).items():
        console.print(f"\n[bold cyan]{provider['name']}[/bold cyan]")
        
        for model in provider.get("models", []):
            status = "[green]✓[/green]" if model.get("enabled") else "[dim]✗[/dim]"
            console.print(f"  {status} {model['id']} - {model['name']}")


@main.command()
@click.option("--tracks", "-t", default=None, help="Comma-separated track letters")
def stats(tracks: str | None):
    """Show dataset statistics."""
    from .runner import MirageRunner
    
    runner = MirageRunner()
    track_list = tracks.split(",") if tracks else None
    items = runner.load_dataset(tracks=track_list)
    
    console.print(f"[bold]Dataset Statistics[/bold]")
    console.print(f"Total items: {len(items)}")
    console.print()
    
    # Count by track
    from collections import Counter
    track_counts = Counter(item.track for item in items)
    
    table = Table(title="Items by Track")
    table.add_column("Track", style="cyan")
    table.add_column("Count", justify="right")
    
    for track in ["A", "B", "C", "D", "E"]:
        table.add_row(track, str(track_counts.get(track, 0)))
    
    console.print(table)


if __name__ == "__main__":
    main()
