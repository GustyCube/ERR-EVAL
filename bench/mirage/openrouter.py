"""
OpenRouter API client for MIRAGE benchmark.
Handles both candidate model inference and GPT 5.2 judging with structured outputs.
"""

from __future__ import annotations
import os
import re
import json
import httpx
from typing import Any
from pydantic import BaseModel

from .models import JudgeScores, AxisScore


class OpenRouterClient:
    """Client for OpenRouter API with structured output support."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mirage-benchmark",
            "X-Title": "MIRAGE Benchmark",
        }
    
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a completion request to OpenRouter.
        
        Args:
            model: OpenRouter model ID (e.g., "openai/gpt-4o")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum response tokens
            response_format: Optional structured output schema
            
        Returns:
            The model's response content as a string
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        
        return data["choices"][0]["message"]["content"]
    
    async def get_candidate_response(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """
        Get a response from a candidate model being evaluated.
        
        Args:
            model: OpenRouter model ID
            prompt: The MIRAGE prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Raw model response
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.complete(model, messages, temperature, max_tokens)
    
    async def judge_response(
        self,
        judge_model: str,
        judge_prompt: str,
        original_prompt: str,
        response: str,
        gold_behavior: dict[str, Any],
    ) -> JudgeScores:
        """
        Use GPT 5.2 to judge a model response with structured outputs.
        
        Args:
            judge_model: The judge model ID (e.g., "openai/gpt-5.2")
            judge_prompt: System prompt for the judge
            original_prompt: The original MIRAGE prompt
            response: The normalized model response to judge
            gold_behavior: The gold standard behavior from the item
            
        Returns:
            JudgeScores with all 5 axis scores and justifications
        """
        # Build the judge input
        judge_input = f"""## Original Prompt
{original_prompt}

## Model Response
{response}

## Expected Behaviors
Must Do:
{chr(10).join(f"- {b}" for b in gold_behavior.get("must_do", []))}

Must Not Do:
{chr(10).join(f"- {b}" for b in gold_behavior.get("must_not_do", []))}

Score this response on the 5 MIRAGE axes (0-2 each). Provide specific quotes or paraphrases from the response as justification."""

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": judge_input},
        ]
        
        # Define structured output schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_scores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ambiguity_detection": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                "justification": {"type": "string", "minLength": 1},
                            },
                            "required": ["score", "justification"],
                            "additionalProperties": False,
                        },
                        "hallucination_avoidance": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                "justification": {"type": "string", "minLength": 1},
                            },
                            "required": ["score", "justification"],
                            "additionalProperties": False,
                        },
                        "localization_of_uncertainty": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                "justification": {"type": "string", "minLength": 1},
                            },
                            "required": ["score", "justification"],
                            "additionalProperties": False,
                        },
                        "response_strategy": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                "justification": {"type": "string", "minLength": 1},
                            },
                            "required": ["score", "justification"],
                            "additionalProperties": False,
                        },
                        "epistemic_tone": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                "justification": {"type": "string", "minLength": 1},
                            },
                            "required": ["score", "justification"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "ambiguity_detection",
                        "hallucination_avoidance",
                        "localization_of_uncertainty",
                        "response_strategy",
                        "epistemic_tone",
                    ],
                    "additionalProperties": False,
                },
            },
        }
        
        raw_response = await self.complete(
            model=judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format=response_format,
        )
        
        # Parse the structured JSON response
        scores_data = json.loads(raw_response)
        
        return JudgeScores(
            ambiguity_detection=AxisScore(**scores_data["ambiguity_detection"]),
            hallucination_avoidance=AxisScore(**scores_data["hallucination_avoidance"]),
            localization_of_uncertainty=AxisScore(**scores_data["localization_of_uncertainty"]),
            response_strategy=AxisScore(**scores_data["response_strategy"]),
            epistemic_tone=AxisScore(**scores_data["epistemic_tone"]),
        )


def normalize_response(response: str) -> str:
    """
    Normalize a model response before judging.
    
    Removes:
    - Model identifiers
    - System prompt echoes
    - "As an AI..." boilerplate
    - Excessive whitespace
    
    Preserves semantic content.
    """
    normalized = response
    
    # Remove common AI boilerplate patterns
    boilerplate_patterns = [
        r"^As an AI( language model)?,?\s*",
        r"^I am an AI( assistant)?,?\s*",
        r"^As a language model,?\s*",
        r"^I'm just an AI,?\s*",
    ]
    for pattern in boilerplate_patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove model self-references mid-response
    normalized = re.sub(
        r"\b(I am|I'm) (just )?an? (AI|language model|assistant)\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    
    # Normalize whitespace
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    normalized = normalized.strip()
    
    return normalized
