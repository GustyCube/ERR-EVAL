"""
Runner for ERR-EVAL benchmark evaluation.
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
    ModelCard,
    EvaluationRun,
)
from .openrouter import OpenRouterClient, normalize_response
from .variant_engine import VariantEngine


class ErrevalRunner:
    """
    Orchestrates ERR-EVAL benchmark evaluation runs.
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
        """
        # Generate variant if seeded
        if seed is not None:
            prompt_used, _ = self.variant_engine.generate_variant(item, seed)
        else:
            prompt_used = item.prompt
        
        # Get candidate response and metadata
        raw_response, metadata = await self.client.get_candidate_response(
            model=model_id,
            prompt=prompt_used,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Fetch generation stats if ID is present
        gen_stats = {}
        if "id" in metadata and metadata["id"]:
            try:
                # Wait briefly for stats to be available
                await asyncio.sleep(1) 
                gen_stats = await self.client.get_generation_stats(metadata["id"])
            except Exception as e:
                print(f"Warning: Could not fetch stats for {metadata['id']}: {e}")

        # Normalize response
        normalized = normalize_response(raw_response)
        
        # Get judge scores
        judge_scores = await self.client.judge_response(
            judge_model=self.judge_model,
            judge_prompt=self.judge_prompt,
            original_prompt=prompt_used,
            response=normalized,
            gold_behavior=item.gold_behavior.model_dump(),
        )

        
        final_scores = judge_scores
        
        return ItemResult(
            item_id=item.id,
            track=item.track,
            variant_seed=seed,
            prompt_used=prompt_used,
            model_response=raw_response,
            normalized_response=normalized,
            judge_scores=judge_scores,
            final_scores=final_scores,
            # Metrics
            latency_ms=float(gen_stats.get("latency", 0) or 0),
            cost=float(gen_stats.get("total_cost", 0) or 0),
            prompt_tokens=int(gen_stats.get("tokens_prompt", 0) or 0),
            completion_tokens=int(gen_stats.get("tokens_completion", 0) or 0),
        )
    
    async def run_evaluation(
        self,
        model_id: str,
        model_name: str | None = None,
        seed: int = 42,
        tracks: list[str] | None = None,
        limit: int | None = 50, # Default limit 50
        temperature: float = 0.0,
        max_tokens: int = 2048,
        progress_callback: Any | None = None,
    ) -> EvaluationRun:
        """
        Run a complete evaluation of a model.
        """
        items = self.load_dataset(tracks=tracks)
        
        if limit:
            # Limit per track (naive) or total?
            # User said "maybe 50 for each model". 
            # If we have 5 tracks, that's 10 per track.
            from collections import defaultdict
            by_track = defaultdict(list)
            for item in items:
                by_track[item.track].append(item)
            
            items = []
            # Distribute limit across tracks roughly equally
            per_track = max(1, limit // 5)
            for track_items in by_track.values():
                items.extend(track_items[:per_track])
        
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
