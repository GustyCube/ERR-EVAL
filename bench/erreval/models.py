"""
Pydantic models for ERR-EVAL benchmark data structures.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# Canonical Item Schema
# =============================================================================

class UncertaintyPoint(BaseModel):
    """A specific point of uncertainty in the prompt."""
    span: str = Field(..., description="The text span containing uncertainty")
    issue: str = Field(..., description="Type of uncertainty (e.g., 'uncertain phoneme')")
    notes: str = Field(default="", description="Additional notes about this uncertainty")


class Temptation(BaseModel):
    """A trap that models commonly fall into."""
    trap: str = Field(..., description="The incorrect behavior models exhibit")
    why_models_fail: str = Field(..., description="Explanation of why this is tempting")


class AmbiguityProfile(BaseModel):
    """Profile describing the ambiguity structure of an item."""
    type: list[str] = Field(..., description="Types of ambiguity present")
    uncertainty_points: list[UncertaintyPoint] = Field(default_factory=list)
    temptations: list[Temptation] = Field(default_factory=list)


class GoldBehavior(BaseModel):
    """Expected ideal behavior for responding to the item."""
    must_do: list[str] = Field(..., description="Required behaviors for a good response")
    must_not_do: list[str] = Field(..., description="Behaviors that should be avoided")
    ideal_clarifiers: list[str] = Field(default_factory=list, description="Example good clarifying questions")
    acceptable_branches: list[str] = Field(default_factory=list, description="Example valid conditional responses")


class Difficulty(BaseModel):
    """Difficulty ratings for the item."""
    human_expected: int = Field(..., ge=1, le=10, description="Expected human score (1-10)")
    model_expected: int = Field(..., ge=1, le=10, description="Expected model score (1-10)")
    notes: str = Field(default="", description="Notes about difficulty")


class VariantSlots(BaseModel):
    """Slot definitions for generating variants."""
    seeded: bool = Field(default=True, description="Whether variants are deterministic")
    slots: dict[str, list[str]] = Field(default_factory=dict, description="Named slots with possible values")
    constraints: list[str] = Field(default_factory=list, description="Constraints on valid combinations")


TrackType = Literal["A", "B", "C", "D", "E"]


class CanonicalItem(BaseModel):
    """A canonical benchmark item."""
    id: str = Field(..., description="Unique item ID (e.g., 'A-0001')")
    track: TrackType = Field(..., description="Track letter (A-E)")
    title: str = Field(..., description="Short descriptive title")
    prompt: str = Field(..., description="The actual prompt shown to models")
    ambiguity_profile: AmbiguityProfile
    gold_behavior: GoldBehavior
    difficulty: Difficulty
    variants: VariantSlots = Field(default_factory=VariantSlots)
    version: str = Field(default="1.0", description="Item version")


# =============================================================================
# Scoring Schema
# =============================================================================

class AxisScore(BaseModel):
    """Score for a single axis with justification."""
    score: int = Field(..., ge=0, le=2, description="Score 0-2")
    justification: str = Field(..., min_length=1, description="Quote-backed justification")


class JudgeScores(BaseModel):
    """Complete judge output for all 5 axes."""
    ambiguity_detection: AxisScore
    hallucination_avoidance: AxisScore
    localization_of_uncertainty: AxisScore
    response_strategy: AxisScore
    epistemic_tone: AxisScore
    
    @property
    def total(self) -> int:
        """Sum of all axis scores (0-10)."""
        return (
            self.ambiguity_detection.score +
            self.hallucination_avoidance.score +
            self.localization_of_uncertainty.score +
            self.response_strategy.score +
            self.epistemic_tone.score
        )


class ItemResult(BaseModel):
    """Complete result for a single item evaluation."""
    item_id: str
    track: TrackType
    variant_seed: int | None = None
    prompt_used: str = Field(..., description="The actual prompt (may be variant)")
    model_response: str = Field(..., description="Raw model response")
    normalized_response: str = Field(..., description="Normalized response for judging")
    
    # Metrics
    latency_ms: float = Field(default=0.0)
    cost: float = Field(default=0.0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    
    judge_scores: JudgeScores
    final_scores: JudgeScores = Field(..., description="Scores (same as judge_scores now)")
    
    @property
    def total_score(self) -> int:
        """Final total score (0-10)."""
        return self.final_scores.total


# =============================================================================
# Aggregation Schema
# =============================================================================

class AxisSummary(BaseModel):
    """Summary statistics for a single axis."""
    axis_name: str
    mean_score: float
    score_distribution: dict[int, int] = Field(default_factory=dict)  # score -> count


class TrackSummary(BaseModel):
    """Summary statistics for a track."""
    track: TrackType
    track_name: str
    item_count: int
    mean_score: float
    axis_summaries: list[AxisSummary]


class FailureMode(BaseModel):
    """A detected failure pattern."""
    mode: str = Field(..., description="Type of failure")
    frequency: int = Field(..., description="Number of occurrences")
    example_item_ids: list[str] = Field(default_factory=list)


class FailureProfile(BaseModel):
    """Profile of model weaknesses."""
    weakest_axes: list[str]
    weakest_tracks: list[TrackType]
    common_failures: list[FailureMode]


class ModelCard(BaseModel):
    """Metadata about the evaluated model."""
    model_id: str = Field(..., description="OpenRouter model ID")
    model_name: str = Field(..., description="Human-readable name")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)
    system_prompt_used: bool = Field(default=False)


class EvaluationRun(BaseModel):
    """Complete evaluation run with all results."""
    run_id: str
    timestamp: str
    dataset_version: str
    seed: int
    model_card: ModelCard
    judge_model: str = Field(default="openai/gpt-5.2")
    item_results: list[ItemResult]
    track_summaries: list[TrackSummary]
    overall_score: float
    percentile: float | None = None
    failure_profile: FailureProfile | None = None


# =============================================================================
# Leaderboard Schema (for frontend)
# =============================================================================

class LeaderboardEntry(BaseModel):
    """A single entry in the leaderboard."""
    rank: int
    model_id: str
    model_name: str
    overall_score: float
    percentile: float
    track_scores: dict[TrackType, float]
    axis_scores: dict[str, float]
    
    # Average Metrics
    avg_latency: float
    avg_cost: float
    
    evaluated_at: str


class LeaderboardData(BaseModel):
    """Complete leaderboard for frontend consumption."""
    generated_at: str
    dataset_version: str
    entries: list[LeaderboardEntry]
