"""Persona definitions for personalized harness evaluation.

Each persona represents a distinct user archetype with different preferences
for how an agent should communicate, analyze, and present results. The same
task evaluated under different personas will use different claim sets,
enabling measurement of personalization effectiveness.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Persona:
    """A user persona with structured preference dimensions."""

    id: str
    name: str
    role: str
    description: str

    # Output preferences
    verbosity: str  # "concise" | "detailed" | "adaptive"
    format: str  # "tables_first" | "narrative" | "structured_bullets"
    reasoning_display: str  # "hide" | "summary" | "show_full"

    # Analysis preferences
    depth: str  # "quick_summary" | "standard" | "deep_dive"
    validation: str  # "trust_first" | "spot_check" | "always_verify"
    uncertainty: str  # "hide" | "mention" | "quantify"

    # Domain emphasis (what aspects to prioritize)
    emphasis: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Concrete personas
# ---------------------------------------------------------------------------

EXECUTIVE = Persona(
    id="exec",
    name="Executive",
    role="VP of Product",
    description=(
        "Senior leader who needs quick, actionable answers. "
        "Values brevity, bottom-line results, and clear recommendations. "
        "Has no patience for methodology details or code."
    ),
    verbosity="concise",
    format="structured_bullets",
    reasoning_display="hide",
    depth="quick_summary",
    validation="trust_first",
    uncertainty="mention",
    emphasis=["actionable_insights", "bottom_line", "comparisons"],
)

DATA_SCIENTIST = Persona(
    id="ds",
    name="Data Scientist",
    role="Senior Data Scientist",
    description=(
        "Technical expert who wants reproducible, rigorous analysis. "
        "Expects to see code, methodology, statistical tests, and caveats. "
        "Prefers tables and structured output over prose."
    ),
    verbosity="detailed",
    format="tables_first",
    reasoning_display="show_full",
    depth="deep_dive",
    validation="always_verify",
    uncertainty="quantify",
    emphasis=["methodology", "reproducibility", "statistical_rigor", "code"],
)

JUNIOR_ANALYST = Persona(
    id="junior",
    name="Junior Analyst",
    role="Junior Business Analyst (first year)",
    description=(
        "New to data work, needs step-by-step explanations and context. "
        "Wants to understand WHY each step is taken, not just the result. "
        "Appreciates definitions of technical terms and learning-oriented output."
    ),
    verbosity="detailed",
    format="narrative",
    reasoning_display="show_full",
    depth="standard",
    validation="spot_check",
    uncertainty="mention",
    emphasis=["explanations", "definitions", "step_by_step", "learning"],
)

JOURNALIST = Persona(
    id="journalist",
    name="Journalist",
    role="Investigative Data Journalist",
    description=(
        "Writes data-driven stories for a general audience. "
        "Needs compelling narratives with concrete examples and human impact. "
        "Wants key findings framed as a story, with sources cited."
    ),
    verbosity="adaptive",
    format="narrative",
    reasoning_display="summary",
    depth="standard",
    validation="always_verify",
    uncertainty="mention",
    emphasis=["storytelling", "concrete_examples", "human_impact", "sources"],
)

# Registry of all personas
PERSONAS: dict[str, Persona] = {
    p.id: p for p in [EXECUTIVE, DATA_SCIENTIST, JUNIOR_ANALYST, JOURNALIST]
}


def get_persona(persona_id: str) -> Persona:
    """Get a persona by ID. Raises KeyError if not found."""
    return PERSONAS[persona_id]


def list_personas() -> list[str]:
    """Return all registered persona IDs."""
    return list(PERSONAS.keys())
