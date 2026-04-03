"""Persona definitions for personalized harness evaluation.

Personas are grounded in four orthogonal dimensions from established research:

1. **Expertise level** — Dreyfus & Dreyfus (1980) skill acquisition model;
   Chi, Feltovich & Glaser (1981) expert-novice differences. Novices need
   context-free rules and definitions; experts want dense, jargon-rich output.

2. **Scope preference** — Riding & Cheema (1991) Wholist-Analytic cognitive
   style; Pask (1976) Holist-Serialist distinction. Wholists want big-picture
   narratives; analytics want structured decompositions.

3. **Task stage** — Kuhlthau (1991) Information Search Process; Taylor (1968)
   information need levels. Explorers need breadth and multiple perspectives;
   executors need precise, filtered, actionable output.

4. **Register** — Giles (1973) Communication Accommodation Theory. Formal
   register uses professional structure (headings, tables, citations);
   casual register uses conversational prose and accessible language.

Each persona occupies a distinct position in this 4D space, ensuring maximum
differentiation. Personas are also grounded in real professional roles whose
information consumption patterns are empirically documented.

References:
    - Dreyfus, S.E. & Dreyfus, H.L. (1980). A Five-Stage Model of the Mental
      Activities Involved in Directed Skill Acquisition. ORC 80-2, UC Berkeley.
    - Chi, M.T.H., Feltovich, P.J. & Glaser, R. (1981). Categorization and
      representation of physics problems by experts and novices. Cognitive
      Science, 5(2), 121-152.
    - Riding, R. & Cheema, I. (1991). Cognitive styles: An overview and
      integration. Educational Psychology, 11(3-4), 193-215.
    - Pask, G. (1976). Styles and strategies of learning. British Journal of
      Educational Psychology, 46(2), 128-148.
    - Kuhlthau, C.C. (1991). Inside the search process: Information seeking
      from the user's perspective. JASIS, 42(5), 361-371.
    - Taylor, R.S. (1968). Question-negotiation and information seeking in
      libraries. College & Research Libraries, 29(3), 178-194.
    - Giles, H. (1973). Accent mobility: A model and some data. Anthropological
      Linguistics, 15(2), 87-105.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Persona:
    """A user persona positioned in the 4D preference space.

    Dimensions (each rated on a 3-point scale for clarity):

    expertise:  "novice" | "intermediate" | "expert"
    scope:      "holistic" | "balanced" | "analytic"
    task_stage: "exploring" | "focused" | "executing"
    register:   "casual" | "adaptive" | "formal"
    """

    id: str
    name: str
    role: str
    description: str

    # --- 4 theoretically-grounded dimensions ---
    expertise: str   # Dreyfus: novice / intermediate / expert
    scope: str       # Riding: holistic / balanced / analytic
    task_stage: str  # Kuhlthau: exploring / focused / executing
    register: str    # CAT: casual / adaptive / formal

    # --- Derived output preferences (for judge prompt context) ---
    verbosity: str          # "concise" | "standard" | "detailed"
    format: str             # "narrative" | "mixed" | "tables_and_code"
    reasoning_display: str  # "hide" | "summary" | "show_full"

    # What the persona prioritizes in output
    emphasis: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Concrete personas — each occupies a distinct region in 4D space
# ---------------------------------------------------------------------------

EXECUTIVE = Persona(
    id="exec",
    name="Strategic Decision-Maker",
    role="VP of Product",
    description=(
        "Senior leader making resource-allocation decisions under time pressure. "
        "Needs bottom-line results, clear recommendations, and confidence levels. "
        "Methodology and code are noise; what matters is the 'so what' and 'now what'."
    ),
    # 4D position: expert + holistic + executing + formal
    expertise="expert",
    scope="holistic",
    task_stage="executing",
    register="formal",
    # Derived preferences
    verbosity="concise",
    format="mixed",  # bullets + summary table, no code
    reasoning_display="hide",
    emphasis=["bottom_line", "recommendations", "confidence_level", "comparisons"],
)

DATA_SCIENTIST = Persona(
    id="ds",
    name="Technical Analyst",
    role="Senior Data Scientist",
    description=(
        "Domain expert who values reproducibility, statistical rigor, and "
        "methodological transparency. Expects to see code, exact queries, "
        "effect sizes, and explicit discussion of assumptions and limitations. "
        "Trusts structured data (tables, metrics) over narrative claims."
    ),
    # 4D position: expert + analytic + executing + formal
    expertise="expert",
    scope="analytic",
    task_stage="executing",
    register="formal",
    # Derived preferences
    verbosity="detailed",
    format="tables_and_code",
    reasoning_display="show_full",
    emphasis=["methodology", "reproducibility", "code", "statistical_rigor", "limitations"],
)

JUNIOR_ANALYST = Persona(
    id="junior",
    name="Learning Practitioner",
    role="Junior Business Analyst (first year)",
    description=(
        "Early-career professional building mental models of data work. "
        "Needs scaffolded explanations that define terms, explain reasoning, "
        "and connect each step to the bigger picture. Learns best from "
        "worked examples with explicit rationale for each decision."
    ),
    # 4D position: novice + holistic + exploring + casual
    expertise="novice",
    scope="holistic",
    task_stage="exploring",
    register="casual",
    # Derived preferences
    verbosity="detailed",
    format="narrative",
    reasoning_display="show_full",
    emphasis=["definitions", "step_by_step", "rationale", "worked_examples", "context"],
)

JOURNALIST = Persona(
    id="journalist",
    name="Narrative Investigator",
    role="Investigative Data Journalist",
    description=(
        "Translates data into stories for a general audience. Needs to verify "
        "facts independently, find the human angle, and frame findings as a "
        "compelling narrative with concrete examples. Values source attribution "
        "and cross-verification but presents output as accessible prose."
    ),
    # 4D position: intermediate + holistic + focused + adaptive
    expertise="intermediate",
    scope="holistic",
    task_stage="focused",
    register="adaptive",
    # Derived preferences
    verbosity="standard",
    format="narrative",
    reasoning_display="summary",
    emphasis=["storytelling", "human_impact", "concrete_examples", "source_attribution", "verification"],
)

# ---------------------------------------------------------------------------
# 4D position summary (for reference)
#
#              expertise   scope      task_stage   register
# exec:        expert      holistic   executing    formal
# ds:          expert      analytic   executing    formal
# junior:      novice      holistic   exploring    casual
# journalist:  intermediate holistic  focused      adaptive
#
# Key differentiation:
# - exec vs ds:       Same expertise/register, differ on scope (holistic vs analytic)
# - exec vs junior:   Opposite on expertise and task_stage
# - ds vs junior:     Opposite on expertise, scope, register
# - journalist vs all: Unique intermediate expertise + focused stage + adaptive register
# ---------------------------------------------------------------------------

PERSONAS: dict[str, Persona] = {
    p.id: p for p in [EXECUTIVE, DATA_SCIENTIST, JUNIOR_ANALYST, JOURNALIST]
}


def get_persona(persona_id: str) -> Persona:
    """Get a persona by ID. Raises KeyError if not found."""
    return PERSONAS[persona_id]


def list_personas() -> list[str]:
    """Return all registered persona IDs."""
    return list(PERSONAS.keys())
