"""
Mechanical checks for rule-based score capping.

These checks inspect responses and apply caps/penalties before LLM judging,
ensuring obvious failures are caught regardless of judge model behavior.
"""

from __future__ import annotations
import re
from typing import Any

from .models import CanonicalItem, MechanicalCaps


# Certainty indicators that suggest overconfidence
CERTAINTY_PATTERNS = [
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bwithout a doubt\b",
    r"\bno question\b",
    r"\babsolutely\b",
    r"\b100%\b",
    r"\bguaranteed\b",
    r"\bfor sure\b",
    r"\bthere's no way\b",
    r"\bit is\b(?! (possible|likely|unclear|uncertain))",
    r"\bthis is\b(?! (possible|likely|unclear|uncertain))",
    r"\bmust be\b",
    r"\bhas to be\b",
]

# Uncertainty indicators that suggest appropriate epistemic humility
UNCERTAINTY_PATTERNS = [
    r"\bmight\b",
    r"\bcould\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bunclear\b",
    r"\buncertain\b",
    r"\bnot sure\b",
    r"\bhard to say\b",
    r"\bdifficult to determine\b",
    r"\bneed more information\b",
    r"\bwould need to know\b",
    r"\bit depends\b",
    r"\bif .+ then\b",
    r"\bassuming\b",
]

# Questions that indicate appropriate clarification-seeking
CLARIFICATION_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"\bcan you (tell|clarify|explain|confirm)\b",
    r"\bdo you (know|mean|have|remember)\b",
    r"\bwhat (exactly|specifically)\b",
    r"\bcould you (provide|share|tell)\b",
    r"\bis it (possible|the case) that\b",
]


def run_mechanical_checks(
    item: CanonicalItem,
    response: str,
) -> MechanicalCaps:
    """
    Run all mechanical checks on a response and return score caps.
    
    Args:
        item: The canonical item with its ambiguity profile
        response: The normalized model response
        
    Returns:
        MechanicalCaps with per-axis caps and reasons
    """
    caps = MechanicalCaps()
    response_lower = response.lower()
    
    # Check 1: Underspecified prompt but single concrete answer
    if "underspecified" in [t.lower() for t in item.ambiguity_profile.type]:
        if not _has_uncertainty_language(response_lower):
            if not _asks_clarifying_question(response_lower):
                caps.ambiguity_detection = 1
                caps.response_strategy = 1
                caps.reasons.append(
                    "Underspecified prompt but response gives concrete answer without noting missing info"
                )
    
    # Check 2: Contradiction present but not mentioned
    if any("conflict" in t.lower() or "contradict" in t.lower() 
           for t in item.ambiguity_profile.type):
        if not _mentions_contradiction(response_lower, item):
            caps.ambiguity_detection = 1
            caps.localization_of_uncertainty = 1
            caps.reasons.append(
                "Prompt contains contradiction but response doesn't acknowledge it"
            )
    
    # Check 3: Hallucinated entities or details
    hallucinations = _detect_hallucinations(response, item)
    if hallucinations:
        caps.hallucination_avoidance = 1
        caps.reasons.append(
            f"Response introduces unsupported details: {', '.join(hallucinations[:3])}"
        )
    
    # Check 4: Overconfident certainty language when uncertainty expected
    if _has_certainty_language(response_lower):
        if not _certainty_is_justified(response_lower, item):
            # Only cap epistemic_tone, not hard fail
            if caps.epistemic_tone is None or caps.epistemic_tone > 1:
                caps.epistemic_tone = 1
            caps.reasons.append(
                "Uses strong certainty language despite ambiguous prompt"
            )
    
    # Check 5: No acknowledgment of any uncertainty points
    if item.ambiguity_profile.uncertainty_points:
        if not _acknowledges_any_uncertainty(response_lower, item):
            caps.localization_of_uncertainty = 1
            caps.reasons.append(
                "Response doesn't acknowledge any of the key uncertainty points"
            )
    
    return caps


def _has_uncertainty_language(response: str) -> bool:
    """Check if response contains uncertainty indicators."""
    for pattern in UNCERTAINTY_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False


def _has_certainty_language(response: str) -> bool:
    """Check if response contains strong certainty indicators."""
    for pattern in CERTAINTY_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False


def _asks_clarifying_question(response: str) -> bool:
    """Check if response asks clarifying questions."""
    for pattern in CLARIFICATION_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False


def _mentions_contradiction(response: str, item: CanonicalItem) -> bool:
    """Check if response acknowledges a contradiction."""
    contradiction_keywords = [
        "contradict",
        "conflict",
        "inconsistent",
        "don't match",
        "doesn't add up",
        "can't both be",
        "mutually exclusive",
        "incompatible",
    ]
    for keyword in contradiction_keywords:
        if keyword in response:
            return True
    return False


def _detect_hallucinations(response: str, item: CanonicalItem) -> list[str]:
    """
    Detect potential hallucinations - details not present in the prompt.
    
    This is a heuristic check that looks for:
    - Specific names not in the prompt
    - Specific dates/times not in the prompt
    - Specific numeric values not in the prompt
    - Specific locations not in the prompt
    """
    hallucinations = []
    prompt_lower = item.prompt.lower()
    response_lower = response.lower()
    
    # Check for specific dates mentioned in response but not prompt
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
        r"\b\d{4}\b",  # Years
    ]
    for pattern in date_patterns:
        response_dates = set(re.findall(pattern, response_lower))
        prompt_dates = set(re.findall(pattern, prompt_lower))
        new_dates = response_dates - prompt_dates
        if new_dates:
            hallucinations.extend([f"date: {d}" for d in list(new_dates)[:2]])
    
    # Check for specific times
    time_pattern = r"\b\d{1,2}:\d{2}\s*(am|pm)?\b"
    response_times = set(re.findall(time_pattern, response_lower))
    prompt_times = set(re.findall(time_pattern, prompt_lower))
    new_times = response_times - prompt_times
    if new_times:
        hallucinations.extend([f"time: {t}" for t in list(new_times)[:2]])
    
    # Check for specific measurements/numbers with units
    measurement_pattern = r"\b\d+\.?\d*\s*(kg|lbs?|miles?|km|meters?|feet|inches|dollars?|\$|hours?|minutes?|seconds?)\b"
    response_measurements = set(re.findall(measurement_pattern, response_lower))
    prompt_measurements = set(re.findall(measurement_pattern, prompt_lower))
    new_measurements = response_measurements - prompt_measurements
    if new_measurements:
        hallucinations.extend([f"measurement: {m}" for m in list(new_measurements)[:2]])
    
    return hallucinations


def _certainty_is_justified(response: str, item: CanonicalItem) -> bool:
    """
    Check if certainty language might be justified.
    
    For some items, certainty about specific aspects might be appropriate
    (e.g., "It is definitely unclear whether...").
    """
    # Certainty about uncertainty is fine
    meta_uncertainty = [
        "definitely unclear",
        "certainly need more",
        "absolutely require",
        "definitely need to",
    ]
    for phrase in meta_uncertainty:
        if phrase in response:
            return True
    
    return False


def _acknowledges_any_uncertainty(response: str, item: CanonicalItem) -> bool:
    """Check if response acknowledges any of the item's uncertainty points."""
    for up in item.ambiguity_profile.uncertainty_points:
        # Check if response mentions the span or addresses the issue
        span_words = up.span.lower().split()
        issue_words = up.issue.lower().split()
        
        # If any significant word from span or issue appears in response
        for word in span_words + issue_words:
            if len(word) > 3 and word in response:
                return True
    
    # Also check if general uncertainty language is present
    return _has_uncertainty_language(response)


def apply_caps(
    scores: dict[str, int],
    caps: MechanicalCaps,
) -> dict[str, int]:
    """
    Apply mechanical caps to judge scores.
    
    Args:
        scores: Dict of axis_name -> score (0-2)
        caps: MechanicalCaps with per-axis caps
        
    Returns:
        New dict with capped scores
    """
    result = scores.copy()
    
    cap_map = {
        "ambiguity_detection": caps.ambiguity_detection,
        "hallucination_avoidance": caps.hallucination_avoidance,
        "localization_of_uncertainty": caps.localization_of_uncertainty,
        "response_strategy": caps.response_strategy,
        "epistemic_tone": caps.epistemic_tone,
    }
    
    for axis, cap in cap_map.items():
        if cap is not None and axis in result:
            result[axis] = min(result[axis], cap)
    
    return result
