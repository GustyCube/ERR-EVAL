"""
Scorer for ERR-EVAL benchmark.
Handles aggregation, percentiles, and failure profiling.
"""

from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

from .models import (
    ItemResult,
    TrackSummary,
    AxisSummary,
    FailureProfile,
    FailureMode,
    TrackType,
)


TRACK_NAMES: dict[TrackType, str] = {
    "A": "Noisy Perception",
    "B": "Ambiguous Semantics",
    "C": "False Premise Traps",
    "D": "Underspecified Tasks",
    "E": "Conflicting Constraints",
}

AXIS_NAMES = [
    "ambiguity_detection",
    "hallucination_avoidance",
    "localization_of_uncertainty",
    "response_strategy",
    "epistemic_tone",
]


def compute_track_summaries(results: list[ItemResult]) -> list[TrackSummary]:
    """
    Compute summary statistics for each track.
    """
    # Group by track
    by_track: dict[TrackType, list[ItemResult]] = defaultdict(list)
    for result in results:
        by_track[result.track].append(result)
    
    summaries = []
    
    for track in ["A", "B", "C", "D", "E"]:
        track_results = by_track.get(track, [])
        if not track_results:
            continue
        
        # Compute axis summaries
        axis_summaries = []
        for axis in AXIS_NAMES:
            scores = [getattr(r.final_scores, axis).score for r in track_results]
            distribution = {0: 0, 1: 0, 2: 0}
            for s in scores:
                distribution[s] += 1
            
            axis_summaries.append(AxisSummary(
                axis_name=axis,
                mean_score=sum(scores) / len(scores) if scores else 0,
                score_distribution=distribution,
            ))
        
        # Compute mean total score
        total_scores = [r.final_scores.total for r in track_results]
        mean_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        summaries.append(TrackSummary(
            track=track,
            track_name=TRACK_NAMES[track],
            item_count=len(track_results),
            mean_score=round(mean_score, 2),
            axis_summaries=axis_summaries,
        ))
    
    return summaries


def compute_overall_score(track_summaries: list[TrackSummary]) -> float:
    """
    Compute weighted overall score from track summaries.
    Uses equal weighting across tracks.
    """
    if not track_summaries:
        return 0.0
    
    total = sum(ts.mean_score for ts in track_summaries)
    return round(total / len(track_summaries), 2)


def compute_percentile(
    score: float,
    baseline_scores: list[float],
) -> float:
    """
    Compute percentile rank of a score against a baseline distribution.
    
    Args:
        score: The score to rank
        baseline_scores: List of baseline scores to compare against
        
    Returns:
        Percentile (0-100)
    """
    if not baseline_scores:
        return 50.0
    
    below = sum(1 for b in baseline_scores if b < score)
    equal = sum(1 for b in baseline_scores if b == score)
    
    percentile = ((below + equal / 2) / len(baseline_scores)) * 100
    return round(percentile, 1)


def compute_failure_profile(results: list[ItemResult]) -> FailureProfile:
    """
    Analyze results to identify systematic failure patterns.
    """
    # Compute mean score per axis
    axis_means: dict[str, float] = {}
    for axis in AXIS_NAMES:
        scores = [getattr(r.final_scores, axis).score for r in results]
        axis_means[axis] = sum(scores) / len(scores) if scores else 0
    
    # Find weakest axes (below average)
    avg = sum(axis_means.values()) / len(axis_means) if axis_means else 0
    weakest_axes = [a for a, m in sorted(axis_means.items(), key=lambda x: x[1]) if m < avg][:3]
    
    # Compute mean score per track
    track_scores: dict[TrackType, list[float]] = defaultdict(list)
    for r in results:
        track_scores[r.track].append(r.final_scores.total)
    
    track_means = {t: sum(s) / len(s) for t, s in track_scores.items() if s}
    avg_track = sum(track_means.values()) / len(track_means) if track_means else 0
    weakest_tracks = [t for t, m in sorted(track_means.items(), key=lambda x: x[1]) if m < avg_track][:2]
    
    # Analyze common failure modes from judge justifications (placeholder for future implementation)
    # Since mechanical checks are removed, we currently don't have structured failure modes
    # beyond score analysis.
    failure_counts: dict[str, list[str]] = defaultdict(list)
    
    common_failures = []
    
    return FailureProfile(
        weakest_axes=weakest_axes,
        weakest_tracks=weakest_tracks,
        common_failures=common_failures,
    )


def compute_axis_percentiles(
    results: list[ItemResult],
    baseline_results: list[list[ItemResult]] | None = None,
) -> dict[str, float]:
    """
    Compute percentile for each axis against baseline.
    """
    if baseline_results is None:
        return {axis: 50.0 for axis in AXIS_NAMES}
    
    percentiles = {}
    for axis in AXIS_NAMES:
        current_scores = [getattr(r.final_scores, axis).score for r in results]
        current_mean = sum(current_scores) / len(current_scores) if current_scores else 0
        
        baseline_means = []
        for baseline in baseline_results:
            scores = [getattr(r.final_scores, axis).score for r in baseline]
            if scores:
                baseline_means.append(sum(scores) / len(scores))
        
        percentiles[axis] = compute_percentile(current_mean, baseline_means)
    
    return percentiles
