"""
CLI commands for ERR-EVAL benchmark.
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
    """ERR-EVAL Benchmark - Epistemic Reliability Evaluation"""
    pass


@main.command()
@click.option("--model", "-m", required=True, help="OpenRouter model ID to evaluate")
@click.option("--seed", "-s", default=42, help="Random seed for variants")
@click.option("--tracks", "-t", default=None, help="Comma-separated track letters (e.g., A,B,C)")
@click.option("--limit", "-l", default=25, type=int, help="Max items per track (default: 25)")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--judge", default="openai/gpt-5.2", help="Judge model ID")
def evaluate(model: str, seed: int, tracks: str | None, limit: int | None,
             output: str | None, temperature: float, judge: str):
    """Run ERR-EVAL evaluation on a model."""
    from .runner import ErrevalRunner
    from .reporter import generate_results_json, generate_markdown_report, generate_leaderboard_entry, update_leaderboard
    
    console.print(f"[bold blue]ERR-EVAL Benchmark Evaluation[/bold blue]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Seed: {seed}")
    console.print(f"Judge: {judge}")
    console.print()
    
    track_list = tracks.split(",") if tracks else None
    
    runner = ErrevalRunner(judge_model=judge)
    
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
        import traceback
        console.print(f"[red]Error:[/red] {e}")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
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


@main.command(name="run-all")
@click.option("--seed", "-s", default=42, help="Random seed for variants")
@click.option("--tracks", "-t", default=None, help="Comma-separated track letters (e.g., A,B,C)")
@click.option("--limit", "-l", default=25, type=int, help="Max items per track (default: 25)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--judge", default="openai/gpt-5.2", help="Judge model ID")
@click.option("--skip-existing", is_flag=True, help="Skip models already in results.json")
def run_all(seed: int, tracks: str | None, limit: int, temperature: float, judge: str, skip_existing: bool):
    """Run ERR-EVAL evaluation on ALL enabled models from models.yaml."""
    import yaml
    import json
    from .runner import ErrevalRunner
    from .reporter import generate_results_json, generate_markdown_report, generate_leaderboard_entry, update_leaderboard
    
    # Pre-flight checks
    console.print("[bold]Pre-flight checks...[/bold]")
    
    # Check config file
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    if not config_path.exists():
        console.print("[red]✗ Config file not found: models.yaml[/red]")
        raise click.Abort()
    console.print(f"[green]✓[/green] Config file: {config_path}")
    
    # Check results.json access and validate structure
    leaderboard_path = Path(__file__).parent.parent.parent / "frontend" / "data" / "results.json"
    existing_model_ids = set()
    
    try:
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        if leaderboard_path.exists():
            with open(leaderboard_path, "r") as f:
                data = json.load(f)
            
            # Check required fields exist
            if not isinstance(data, dict) or "entries" not in data:
                console.print(f"[yellow]⚠[/yellow] Results file corrupted, reinitializing...")
                data = {"generated_at": "", "dataset_version": "canonical", "providers": {}, "entries": []}
                with open(leaderboard_path, "w") as f:
                    json.dump(data, f, indent=2)
            
            # Get existing model IDs for skip-existing
            for entry in data.get("entries", []):
                existing_model_ids.add(entry.get("model_id"))
            
            console.print(f"[green]✓[/green] Results file: {leaderboard_path} ({len(existing_model_ids)} existing models)")
        else:
            data = {"generated_at": "", "dataset_version": "canonical", "providers": {}, "entries": []}
            with open(leaderboard_path, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]✓[/green] Results file: {leaderboard_path} (created)")
            
    except PermissionError:
        console.print(f"[red]✗ Cannot write to results.json - permission denied[/red]")
        console.print(f"  Path: {leaderboard_path}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]✗ Cannot access results.json: {e}[/red]")
        raise click.Abort()
    
    console.print()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Collect all enabled models (deduplicated by ID)
    seen_ids = set()
    models_to_run = []
    for provider_key, provider in config.get("providers", {}).items():
        for model in provider.get("models", []):
            if model.get("enabled", False):
                model_id = model["id"]
                if model_id not in seen_ids:
                    seen_ids.add(model_id)
                    models_to_run.append({
                        "id": model_id,
                        "name": model["name"],
                        "provider": provider_key,
                    })
    
    if not models_to_run:
        console.print("[yellow]No enabled models found in models.yaml[/yellow]")
        return
    
    # Filter out existing if --skip-existing
    if skip_existing and existing_model_ids:
        original_count = len(models_to_run)
        models_to_run = [m for m in models_to_run if m["id"] not in existing_model_ids]
        skipped = original_count - len(models_to_run)
        if skipped > 0:
            console.print(f"[dim]Skipping {skipped} models already in results.json[/dim]")
        if not models_to_run:
            console.print("[green]All models already evaluated![/green]")
            return
    
    console.print(f"[bold blue]ERR-EVAL Batch Evaluation[/bold blue]")
    console.print(f"Models to evaluate: [cyan]{len(models_to_run)}[/cyan]")
    console.print(f"Items per model: {limit * 5} (5 tracks × {limit})")
    console.print(f"Judge: {judge}")
    if skip_existing:
        console.print(f"[dim]Mode: Skip existing[/dim]")
    console.print()
    
    # Confirm
    console.print("[yellow]Models:[/yellow]")
    for m in models_to_run:
        console.print(f"  • {m['name']} ({m['id']})")
    console.print()
    
    if not click.confirm("Proceed with evaluation?"):
        raise click.Abort()
    
    track_list = tracks.split(",") if tracks else None
    runner = ErrevalRunner(judge_model=judge)
    
    results = []
    failed = []
    
    for idx, model_info in enumerate(models_to_run):
        model_id = model_info["id"]
        console.print(f"\n[bold]({idx + 1}/{len(models_to_run)}) Evaluating: {model_info['name']}[/bold]")
        
        async def run_eval():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Evaluating {model_info['name']}...", total=None)
                
                def update_progress(current, total):
                    progress.update(task, completed=current, total=total)
                
                result = await runner.run_evaluation(
                    model_id=model_id,
                    seed=seed,
                    tracks=track_list,
                    limit=limit,
                    temperature=temperature,
                    progress_callback=update_progress,
                )
                return result
        
        try:
            result = asyncio.run(run_eval())
            results.append(result)
            
            # Save individual results
            output_path = Path("results") / f"{model_id.replace('/', '_')}_{seed}.json"
            generate_results_json(result, output_path)
            
            md_path = output_path.with_suffix(".md")
            generate_markdown_report(result, md_path)
            
            # Update leaderboard
            entry = generate_leaderboard_entry(result)
            update_leaderboard(leaderboard_path, entry)
            
            console.print(f"  [green]✓[/green] Score: {result.overall_score:.2f}/10")
            
        except Exception as e:
            console.print(f"  [red]✗ Failed:[/red] {e}")
            failed.append({"model": model_id, "error": str(e)})
    
    # Summary
    console.print()
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print("[bold green]Batch Evaluation Complete![/bold green]")
    console.print(f"Successful: {len(results)}/{len(models_to_run)}")
    if failed:
        console.print(f"[red]Failed: {len(failed)}[/red]")
        for f in failed:
            console.print(f"  • {f['model']}: {f['error']}")
    
    console.print(f"\nLeaderboard updated: [cyan]{leaderboard_path}[/cyan]")



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
    
    enabled_count = 0
    total_count = 0
    
    for provider_key, provider in config.get("providers", {}).items():
        console.print(f"\n[bold cyan]{provider['name']}[/bold cyan]")
        
        for model in provider.get("models", []):
            total_count += 1
            if model.get("enabled"):
                enabled_count += 1
            status = "[green]✓[/green]" if model.get("enabled") else "[dim]✗[/dim]"
            console.print(f"  {status} {model['id']} - {model['name']}")
    
    console.print(f"\n[bold]Total: {enabled_count} enabled / {total_count} models[/bold]")


@main.command()
@click.option("--tracks", "-t", default=None, help="Comma-separated track letters")
def stats(tracks: str | None):
    """Show dataset statistics."""
    from .runner import ErrevalRunner
    
    runner = ErrevalRunner()
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
