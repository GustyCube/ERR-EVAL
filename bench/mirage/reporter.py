"""
Report generator for ERR-EVAL benchmark.
Creates JSON and Markdown output files.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .models import EvaluationRun, LeaderboardData, LeaderboardEntry


def generate_results_json(
    run: EvaluationRun,
    output_path: Path | str,
) -> None:
    """
    Generate a JSON results file for the evaluation run.
    """
    import yaml
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load provider config
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    provider_meta = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                if "providers" in config_data:
                    for pid, pdata in config_data["providers"].items():
                        provider_meta[pid] = {
                            "name": pdata.get("name"),
                            "color": pdata.get("color", "#666"),
                            "icon": pdata.get("icon", "?")
                        }
        except Exception as e:
            print(f"Warning: Could not load provider config: {e}")

    data = run.model_dump()
    data["providers"] = provider_meta

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_leaderboard_entry(run: EvaluationRun) -> LeaderboardEntry:
    """
    Create a leaderboard entry from an evaluation run.
    """
    track_scores = {ts.track: ts.mean_score for ts in run.track_summaries}
    
    # Compute axis scores across all results
    axis_totals: dict[str, list[int]] = {
        "ambiguity_detection": [],
        "hallucination_avoidance": [],
        "localization_of_uncertainty": [],
        "response_strategy": [],
        "epistemic_tone": [],
    }
    
    for result in run.item_results:
        axis_totals["ambiguity_detection"].append(result.final_scores.ambiguity_detection.score)
        axis_totals["hallucination_avoidance"].append(result.final_scores.hallucination_avoidance.score)
        axis_totals["localization_of_uncertainty"].append(result.final_scores.localization_of_uncertainty.score)
        axis_totals["response_strategy"].append(result.final_scores.response_strategy.score)
        axis_totals["epistemic_tone"].append(result.final_scores.epistemic_tone.score)
    
    axis_scores = {
        axis: round(sum(scores) / len(scores), 2) if scores else 0
        for axis, scores in axis_totals.items()
    }
    
    # Compute average metrics
    latencies = [r.latency_ms for r in run.item_results]
    costs = [r.cost for r in run.item_results]
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    
    return LeaderboardEntry(
        rank=0,  # Set later when building full leaderboard
        model_id=run.model_card.model_id,
        model_name=run.model_card.model_name,
        overall_score=run.overall_score,
        percentile=run.percentile or 50.0,
        track_scores=track_scores,
        axis_scores=axis_scores,
        avg_latency=round(avg_latency, 2),
        avg_cost=round(avg_cost, 6),
        evaluated_at=run.timestamp,
    )


def generate_markdown_report(
    run: EvaluationRun,
    output_path: Path | str,
) -> None:
    """
    Generate a Markdown summary report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# ERR-EVAL Evaluation Report",
        f"",
        f"**Model**: {run.model_card.model_name} (`{run.model_card.model_id}`)",
        f"**Date**: {run.timestamp}",
        f"**Dataset Version**: {run.dataset_version}",
        f"**Seed**: {run.seed}",
        f"",
        f"## Overall Score",
        f"",
        f"**{run.overall_score:.2f} / 10**",
        f"",
    ]
    
    if run.percentile:
        lines.append(f"Percentile: {run.percentile:.1f}%")
        lines.append("")
    
    lines.extend([
        f"## Track Breakdown",
        f"",
        f"| Track | Name | Items | Score |",
        f"|-------|------|-------|-------|",
    ])
    
    for ts in run.track_summaries:
        lines.append(f"| {ts.track} | {ts.track_name} | {ts.item_count} | {ts.mean_score:.2f} |")
    
    lines.extend([
        f"",
        f"## Axis Breakdown",
        f"",
    ])
    
    # Compute overall axis means
    axis_totals: dict[str, list[int]] = {
        "Ambiguity Detection": [],
        "Hallucination Avoidance": [],
        "Localization of Uncertainty": [],
        "Response Strategy": [],
        "Epistemic Tone": [],
    }
    
    axis_map = {
        "Ambiguity Detection": "ambiguity_detection",
        "Hallucination Avoidance": "hallucination_avoidance",
        "Localization of Uncertainty": "localization_of_uncertainty",
        "Response Strategy": "response_strategy",
        "Epistemic Tone": "epistemic_tone",
    }
    
    for result in run.item_results:
        for display_name, attr_name in axis_map.items():
            axis_totals[display_name].append(getattr(result.final_scores, attr_name).score)
    
    lines.extend([
        f"| Axis | Mean Score | Out of |",
        f"|------|------------|--------|",
    ])
    
    for display_name, scores in axis_totals.items():
        mean = sum(scores) / len(scores) if scores else 0
        lines.append(f"| {display_name} | {mean:.2f} | 2 |")
    
    if run.failure_profile:
        lines.extend([
            f"",
            f"## Failure Profile",
            f"",
        ])
        
        if run.failure_profile.weakest_axes:
            lines.append(f"**Weakest Axes**: {', '.join(run.failure_profile.weakest_axes)}")
        
        if run.failure_profile.weakest_tracks:
            lines.append(f"**Weakest Tracks**: {', '.join(run.failure_profile.weakest_tracks)}")
        
        if run.failure_profile.common_failures:
            lines.extend([
                f"",
                f"**Common Failure Modes**:",
                f"",
            ])
            for fm in run.failure_profile.common_failures:
                lines.append(f"- {fm.mode} ({fm.frequency} occurrences)")
    
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def update_leaderboard(
    leaderboard_path: Path | str,
    new_entry: LeaderboardEntry,
) -> LeaderboardData:
    """
    Update or create leaderboard with a new entry.
    """
    from datetime import datetime
    import yaml
    
    leaderboard_path = Path(leaderboard_path)
    
    # Load existing or create new
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            data = json.load(f)
        leaderboard = LeaderboardData(**data)
    else:
        leaderboard = LeaderboardData(
            generated_at=datetime.now().isoformat(),
            dataset_version="canonical",
            entries=[],
        )
    
    # Check if model already exists
    existing_idx = None
    for i, entry in enumerate(leaderboard.entries):
        if entry.model_id == new_entry.model_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Update existing
        leaderboard.entries[existing_idx] = new_entry
    else:
        # Add new
        leaderboard.entries.append(new_entry)
    
    # Re-rank by overall score
    leaderboard.entries.sort(key=lambda e: e.overall_score, reverse=True)
    for i, entry in enumerate(leaderboard.entries):
        entry.rank = i + 1
    
    # Update timestamp
    leaderboard.generated_at = datetime.now().isoformat()
    
    # Load providers from models.yaml
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    providers = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            for pid, pdata in config.get("providers", {}).items():
                providers[pid] = {
                    "name": pdata.get("name", pid),
                    "color": pdata.get("color", "#666666"),
                    "icon": pdata.get("icon", "default.svg"),
                }
        except Exception as e:
            print(f"Warning: Could not load provider config: {e}")
    
    # Build output data with providers
    output_data = {
        "generated_at": leaderboard.generated_at,
        "dataset_version": leaderboard.dataset_version,
        "providers": providers,
        "entries": [entry.model_dump() for entry in leaderboard.entries],
    }
    
    # Save
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(leaderboard_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    return leaderboard

