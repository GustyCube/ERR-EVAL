"""
OpenRouter API client for ERR-EVAL benchmark.
Handles both candidate model inference and GPT 5.2 judging with structured outputs.
"""

from __future__ import annotations
import os
import re
import json
import httpx
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv

from .models import JudgeScores, AxisScore

# Load .env file if present
load_dotenv()


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
            "HTTP-Referer": "https://github.com/GustyCube/ERR-EVAL",
            "X-Title": "ERR-EVAL Benchmark",
        }
    
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 5,
    ) -> tuple[str, str]:
        """
        Send a completion request to OpenRouter.
        
        Returns:
            (content, generation_id)
            
        Raises:
            httpx.HTTPStatusError: On non-retryable errors
        """
        import asyncio
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        last_error = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                gen_id = data.get("id", "")
                return content, gen_id
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) + 1  # 2, 3, 5, 9, 17 seconds
                    print(f"  Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    # Non-retryable error
                    raise
            except httpx.TimeoutException as e:
                # Timeout - retry with backoff
                wait_time = (2 ** attempt) + 1
                print(f"  Timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
                last_error = e
        
        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Max retries exceeded")
    
    async def get_generation_stats(self, generation_id: str) -> dict[str, Any]:
        """
        Fetch stats (cost, latency, tokens) for a generation.
        """
        if not generation_id:
            return {}
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/generation",
                params={"id": generation_id},
                headers=self.headers,
            )
            if response.status_code != 200:
                return {}
                
            data = response.json()
            return data.get("data", {})
    
    async def get_candidate_response(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[str, dict[str, Any]]:
        """
        Get a response from a candidate model being evaluated.
        
        Returns:
            (Raw model response, Metadata dict with id)
        """
        messages = [{"role": "user", "content": prompt}]
        content, gen_id = await self.complete(model, messages, temperature, max_tokens)
        
        return content, {"id": gen_id}
    
    async def judge_response(
        self,
        judge_model: str,
        judge_prompt: str,
        original_prompt: str,
        response: str,
        gold_behavior: dict[str, Any],
        max_parse_retries: int = 3,
    ) -> JudgeScores:
        """
        Use GPT 5.2 to judge a model response with structured outputs.
        Retries on parse failures instead of returning default scores.
        """
        import asyncio
        
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

Score this response on the 5 ERR-EVAL axes (0-2 each). Provide specific quotes or paraphrases from the response as justification."""

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
        
        last_error = None
        for attempt in range(max_parse_retries):
            raw_response, _ = await self.complete(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format=response_format,
            )
            
            # Parse the structured JSON response
            try:
                scores_data = json.loads(raw_response)
                return JudgeScores(
                    ambiguity_detection=AxisScore(**scores_data["ambiguity_detection"]),
                    hallucination_avoidance=AxisScore(**scores_data["hallucination_avoidance"]),
                    localization_of_uncertainty=AxisScore(**scores_data["localization_of_uncertainty"]),
                    response_strategy=AxisScore(**scores_data["response_strategy"]),
                    epistemic_tone=AxisScore(**scores_data["epistemic_tone"]),
                )
            except (json.JSONDecodeError, KeyError) as e:
                # Try to extract JSON from response if wrapped in other content
                json_match = re.search(r'\{[\s\S]*\}', raw_response)
                if json_match:
                    try:
                        scores_data = json.loads(json_match.group())
                        return JudgeScores(
                            ambiguity_detection=AxisScore(**scores_data["ambiguity_detection"]),
                            hallucination_avoidance=AxisScore(**scores_data["hallucination_avoidance"]),
                            localization_of_uncertainty=AxisScore(**scores_data["localization_of_uncertainty"]),
                            response_strategy=AxisScore(**scores_data["response_strategy"]),
                            epistemic_tone=AxisScore(**scores_data["epistemic_tone"]),
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                # Parse failed - retry
                last_error = e
                if attempt < max_parse_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Judge parse failed, retrying in {wait_time}s ({attempt + 1}/{max_parse_retries})...")
                    await asyncio.sleep(wait_time)
        
        # All retries exhausted - raise error instead of returning fake scores
        raise ValueError(f"Failed to parse judge response after {max_parse_retries} attempts: {last_error}")


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
