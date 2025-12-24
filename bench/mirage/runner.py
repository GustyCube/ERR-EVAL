"""
Runner for MIRAGE benchmark evaluation.
Orchestrates the full evaluation pipeline.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    CanonicalItem,
    ItemResult,
    JudgeScores,
    AxisScore,
    MechanicalCaps,
    ModelCard,
    EvaluationRun,
)
from .openrouter import OpenRouterClient, normalize_response
from .variant_engine import VariantEngine
from .mechanical_checks import run_mechanical_checks, apply_caps


class MirageRunner:
    """
    Orchestrates MIRAGE benchmark evaluation runs.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        data_dir: Path | str = Path(__file__).parent.parent / "data",
        judge_model: str = "openai/gpt-4.1",
    ):
        self.client = OpenRouterClient(api_key)
        self.data_dir = Path(data_dir)
        self.judge_model = judge_model
        self.variant_engine = VariantEngine(
            self.data_dir / "slots_library.json"
        )
        
        # Load judge prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt.txt"
        if prompt_path.exists():
            self.judge_prompt = prompt_path.read_text()
        else:
            raise FileNotFoundError(f"Judge prompt not found at {prompt_path}")
    
    def load_dataset(
        self,
        tracks: list[str] | None = None,
        version: str = "canonical",
    ) -> list[CanonicalItem]:
        """
        Load benchmark items from JSONL files.
        
        Args:
            tracks: List of track letters to load (e.g., ["A", "B"]), or None for all
            version: Dataset version directory name
            
        Returns:
            List of CanonicalItem objects
        """
        items = []
        data_path = self.data_dir / version
        
        if tracks is None:
            tracks = ["A", "B", "C", "D", "E"]
        
        for track in tracks:
            track_file = data_path / f"track{track}.jsonl"
            if track_file.exists():
                with open(track_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            items.append(CanonicalItem(**data))
        
        return items
    
    async def evaluate_item(
        self,
        item: CanonicalItem,
        model_id: str,
        seed: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ItemResult:
        """
        Evaluate a single item against a candidate model.
        
        Args:
            item: The canonical item to evaluate
            model_id: OpenRouter model ID for the candidate
            seed: Random seed for variant generation
            temperature: Sampling temperature for candidate
            max_tokens: Max response tokens
            
        Returns:
            ItemResult with all scores and justifications
        """
        # Generate variant if seeded
        if seed is not None:
            prompt_used, _ = self.variant_engine.generate_variant(item, seed)
        else:
            prompt_used = item.prompt
        
        # Get candidate response
        raw_response = await self.client.get_candidate_response(
            model=model_id,
            prompt=prompt_used,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Normalize response
        normalized = normalize_response(raw_response)
        
        # Run mechanical checks
        caps = run_mechanical_checks(item, normalized)
        
        # Get judge scores
        judge_scores = await self.client.judge_response(
            judge_model=self.judge_model,
            judge_prompt=self.judge_prompt,
            original_prompt=prompt_used,
            response=normalized,
            gold_behavior=item.gold_behavior.model_dump(),
        )
        
        # Apply mechanical caps to get final scores
        raw_scores = {
            "ambiguity_detection": judge_scores.ambiguity_detection.score,
            "hallucination_avoidance": judge_scores.hallucination_avoidance.score,
            "localization_of_uncertainty": judge_scores.localization_of_uncertainty.score,
            "response_strategy": judge_scores.response_strategy.score,
            "epistemic_tone": judge_scores.epistemic_tone.score,
        }
        
        capped_scores = apply_caps(raw_scores, caps)
        
        final_scores = JudgeScores(
            ambiguity_detection=AxisScore(
                score=capped_scores["ambiguity_detection"],
                justification=judge_scores.ambiguity_detection.justification,
            ),
            hallucination_avoidance=AxisScore(
                score=capped_scores["hallucination_avoidance"],
                justification=judge_scores.hallucination_avoidance.justification,
            ),
            localization_of_uncertainty=AxisScore(
                score=capped_scores["localization_of_uncertainty"],
                justification=judge_scores.localization_of_uncertainty.justification,
            ),
            response_strategy=AxisScore(
                score=capped_scores["response_strategy"],
                justification=judge_scores.response_strategy.justification,
            ),
            epistemic_tone=AxisScore(
                score=capped_scores["epistemic_tone"],
                justification=judge_scores.epistemic_tone.justification,
            ),
        )
        
        return ItemResult(
            item_id=item.id,
            track=item.track,
            variant_seed=seed,
            prompt_used=prompt_used,
            model_response=raw_response,
            normalized_response=normalized,
            mechanical_caps=caps,
            judge_scores=judge_scores,
            final_scores=final_scores,
        )
    
    async def run_evaluation(
        self,
        model_id: str,
        model_name: str | None = None,
        seed: int = 42,
        tracks: list[str] | None = None,
        limit: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        progress_callback: Any | None = None,
    ) -> EvaluationRun:
        """
        Run a complete evaluation of a model.
        
        Args:
            model_id: OpenRouter model ID
            model_name: Human-readable name (defaults to model_id)
            seed: Random seed for variants
            tracks: Which tracks to evaluate (None = all)
            limit: Max items per track (None = all)
            temperature: Sampling temperature
            max_tokens: Max response tokens
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Complete EvaluationRun with all results
        """
        items = self.load_dataset(tracks=tracks)
        
        if limit:
            # Limit per track
            from collections import defaultdict
            by_track = defaultdict(list)
            for item in items:
                by_track[item.track].append(item)
            
            items = []
            for track_items in by_track.values():
                items.extend(track_items[:limit])
        
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, total)
            
            result = await self.evaluate_item(
                item=item,
                model_id=model_id,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
        
        # Build evaluation run
        from .scorer import compute_track_summaries, compute_overall_score, compute_failure_profile
        
        track_summaries = compute_track_summaries(results)
        overall_score = compute_overall_score(track_summaries)
        failure_profile = compute_failure_profile(results)
        
        return EvaluationRun(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            dataset_version="canonical",
            seed=seed,
            model_card=ModelCard(
                model_id=model_id,
                model_name=model_name or model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            judge_model=self.judge_model,
            item_results=results,
            track_summaries=track_summaries,
            overall_score=overall_score,
            failure_profile=failure_profile,
        )
