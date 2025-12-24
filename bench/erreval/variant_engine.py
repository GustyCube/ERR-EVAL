"""
Variant engine for generating deterministic perturbations of canonical items.
"""

from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any

from .models import CanonicalItem, VariantSlots


class VariantEngine:
    """
    Generates deterministic variants of canonical ERR-EVAL items.
    
    Variants preserve the same ambiguity structure while changing surface details
    (entity names, contexts, specific values) to prevent memorization.
    """
    
    def __init__(self, slots_library_path: Path | str | None = None):
        """
        Initialize the variant engine.
        
        Args:
            slots_library_path: Path to slots_library.json with global slot definitions.
                              If None, uses only item-specific slots.
        """
        self.global_slots: dict[str, list[str]] = {}
        
        if slots_library_path:
            path = Path(slots_library_path)
            if path.exists():
                with open(path) as f:
                    self.global_slots = json.load(f)
    
    def generate_variant(
        self,
        item: CanonicalItem,
        seed: int,
    ) -> tuple[str, dict[str, str]]:
        """
        Generate a deterministic variant of an item's prompt.
        
        Args:
            item: The canonical item to generate a variant of
            seed: Random seed for deterministic selection
            
        Returns:
            Tuple of (variant_prompt, substitutions_made)
        """
        if not item.variants.seeded:
            # Item doesn't support variants, return original
            return item.prompt, {}
        
        # Combine item-specific slots with global slots
        # Item-specific slots take precedence
        available_slots = {**self.global_slots, **item.variants.slots}
        
        if not available_slots:
            return item.prompt, {}
        
        # Create seeded RNG
        rng = random.Random(seed)
        
        # Select values for each slot
        substitutions: dict[str, str] = {}
        for slot_name, options in available_slots.items():
            if options:
                # Ensure options is a list (might be a dict from JSON)
                if isinstance(options, dict):
                    options = list(options.values())
                elif not isinstance(options, list):
                    options = [str(options)]
                
                selected = rng.choice(options)
                
                # Ensure selected value is a string
                while isinstance(selected, (list, dict)):
                    if isinstance(selected, dict):
                        selected = list(selected.values())[0] if selected else ""
                    elif isinstance(selected, list):
                        selected = selected[0] if selected else ""
                
                substitutions[slot_name] = str(selected)
        
        # Apply substitutions to prompt
        variant_prompt = self._apply_substitutions(item.prompt, substitutions)
        
        return variant_prompt, substitutions
    
    def _apply_substitutions(
        self,
        prompt: str,
        substitutions: dict[str, str],
    ) -> str:
        """
        Apply slot substitutions to a prompt.
        
        Slots in the prompt are marked as {{slot_name}}.
        """
        result = prompt
        for slot_name, value in substitutions.items():
            # Replace {{slot_name}} patterns
            pattern = r"\{\{\s*" + re.escape(slot_name) + r"\s*\}\}"
            result = re.sub(pattern, value, result)
        
        return result
    
    def validate_variant(
        self,
        original: CanonicalItem,
        variant_prompt: str,
        substitutions: dict[str, str],
    ) -> list[str]:
        """
        Validate that a variant maintains the item's constraints.
        
        Args:
            original: The original canonical item
            variant_prompt: The generated variant prompt
            substitutions: The substitutions that were made
            
        Returns:
            List of constraint violations (empty if valid)
        """
        violations: list[str] = []
        
        # Check that the variant is not identical to original (if slots exist)
        if original.variants.slots and variant_prompt == original.prompt:
            violations.append("Variant is identical to original despite having slots")
        
        # Check length constraint (variant shouldn't be drastically different)
        len_diff = abs(len(variant_prompt) - len(original.prompt))
        max_diff = len(original.prompt) * 0.5  # Allow up to 50% length change
        if len_diff > max_diff:
            violations.append(f"Variant length differs too much: {len_diff} chars")
        
        # Check that all expected slots were filled
        remaining_slots = re.findall(r"\{\{[^}]+\}\}", variant_prompt)
        if remaining_slots:
            violations.append(f"Unfilled slots remaining: {remaining_slots}")
        
        return violations


def create_variant_prompt_template(
    base_prompt: str,
    slot_markers: dict[str, str],
) -> str:
    """
    Helper to create a variant-ready prompt template from a base prompt.
    
    Args:
        base_prompt: The original prompt text
        slot_markers: Dict mapping literal text to slot names
                     e.g., {"my dad": "speaker", "washing machine": "noise_source"}
    
    Returns:
        Prompt with literal text replaced by {{slot_name}} markers
    
    Example:
        >>> create_variant_prompt_template(
        ...     "I heard my dad calling while the washing machine was on",
        ...     {"my dad": "speaker", "washing machine": "noise_source"}
        ... )
        "I heard {{speaker}} calling while the {{noise_source}} was on"
    """
    result = base_prompt
    for literal, slot_name in slot_markers.items():
        result = result.replace(literal, "{{" + slot_name + "}}")
    return result
