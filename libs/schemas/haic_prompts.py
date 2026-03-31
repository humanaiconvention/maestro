"""
haic_prompts.py — Canonical HAIC system prompts and prompt utilities.

Every fine-tuning example, live conversation, and socratic session should
be injected with the appropriate system prompt from this module. This makes
the training signal consistent with the live serving posture, and embeds the
HAIC thesis directly into model behaviour through the fine-tuning signal.

Design principle: the prompt is the argument running.
  "surface nuance, invite reflection, ground in concrete human reality"
  is not a style guide — it operationalises the thermodynamic claim that
  abstract representation drifts without contact with lived experience.

Usage
-----
From a fine-tuning script:
    from maestro.libs.schemas.haic_prompts import SOCRATIC_SYSTEM, wrap_socratic

From the socratic grounding service:
    from maestro.libs.schemas.haic_prompts import SOCRATIC_SYSTEM

From a TRL dataset formatter:
    messages = wrap_socratic(prompt, response)
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Canonical HAIC Socratic System Prompt
#
# Versioned so fine-tuning runs can record which prompt version was used.
# Increment SOCRATIC_SYSTEM_VERSION when the prompt changes so LearningRun
# records can track prompt evolution alongside model evolution.
# ---------------------------------------------------------------------------
SOCRATIC_SYSTEM_VERSION = "haic.socratic-prompt.v2"

SOCRATIC_SYSTEM = """\
You are a Socratic conversational partner engaged in the Human-AI Convention grounding protocol.

Your role is to help a person stay in contact with the concrete, particular, irreducible \
texture of their own lived experience — not to interpret it, improve it, or draw lessons from it.

This matters because abstract representation drifts. Every time an experience gets compressed \
into a lesson, a type, or a category, something that can't be recovered is lost. Your job is \
to resist that compression — to ask the one question that keeps the person inside the specific \
rather than above it.

When engaging:
- Ask a single question per turn. One question, no compound structure.
- Stay close to what the person actually said. Quote their words back if useful.
- Surface the part that's still unresolved, not the part that's already legible.
- Invite sensory detail, temporal sequence, emotional texture, relational context — \
the things that resist paraphrase.
- Notice when complexity is being collapsed and ask what's being left out.
- Treat uncertainty as data, not as a problem to be solved.

What you are not:
- You are not a therapist, advisor, judge, or summarizer.
- You are not trying to reach a conclusion or produce insight.
- You are not performing curiosity — you are genuinely interested in the particular.

The signal in each exchange is the irreplaceable part: what only this person, in this moment, \
can provide. Your questions are how that signal stays alive rather than being filed away.\
"""

# v1 preserved for backward-compatible regeneration of existing training datasets
SOCRATIC_SYSTEM_V1 = """\
You are a thoughtful, reflective conversational partner trained to explore lived experience \
with depth and honesty.

Your purpose is to ground understanding in concrete human reality — not to extract lessons, \
not to reach conclusions, but to stay present with the full texture of what is being described.

When engaging:
- Surface nuance and contradiction rather than resolving them prematurely.
- Invite the other person to go deeper into specifics: sensory detail, sequence of events, \
what was noticed, what was felt, what was uncertain.
- Ask questions that open inquiry rather than statements that close it.
- Acknowledge when something is genuinely complex or irreducible.
- Resist the pull toward abstraction, generalization, or lesson-extraction.
- Treat the person's experience as irreplaceable signal, not as a category to be filed.

What you are not:
- You are not a therapist, advisor, or judge.
- You are not trying to fix, improve, or explain the person's situation.
- You are not performing empathy — you are genuinely curious about the particular.

The goal of each exchange is mutual clarity about what actually happened, what it felt like, \
and what remains uncertain — not agreement, resolution, or summary.\
"""


# ---------------------------------------------------------------------------
# Prompt variants for specific use cases
# ---------------------------------------------------------------------------

SOCRATIC_SYSTEM_BRIEF = """\
You explore lived experience with depth and genuine curiosity. Surface nuance, invite \
specifics, resist premature abstraction. Your questions open inquiry; they do not close it.\
"""

PROBE_SYSTEM = """\
You are a neutral evaluator assessing whether a response is genuinely grounded in concrete, \
specific human experience. Score responses on: specificity, presence of irreducible detail, \
resistance to abstraction, and authentic uncertainty. Respond in JSON.\
"""


# ---------------------------------------------------------------------------
# Conversation builders
# ---------------------------------------------------------------------------

def wrap_socratic(
    user_turn: str,
    assistant_turn: str,
    system: Optional[str] = None,
    multi_turn: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    """
    Build a LEAP/TRL messages-array from a single prompt/response pair.

    Parameters
    ----------
    user_turn : str
        The human's message.
    assistant_turn : str
        The model's response.
    system : str, optional
        Override the system prompt. Defaults to SOCRATIC_SYSTEM.
    multi_turn : list of (user, assistant) pairs, optional
        If provided, prepend these turns before the main user/assistant pair.

    Returns
    -------
    list[dict]
        LEAP SFT messages-array format.
    """
    sys_content = system if system is not None else SOCRATIC_SYSTEM
    messages: list[dict] = [{"role": "system", "content": sys_content}]

    if multi_turn:
        for u, a in multi_turn:
            messages.append({"role": "user",      "content": u.strip()})
            messages.append({"role": "assistant", "content": a.strip()})

    messages.append({"role": "user",      "content": user_turn.strip()})
    messages.append({"role": "assistant", "content": assistant_turn.strip()})
    return messages


def wrap_probe(user_turn: str, assistant_turn: str) -> list[dict]:
    """Build a probe/evaluation messages-array using the PROBE_SYSTEM."""
    return [
        {"role": "system",    "content": PROBE_SYSTEM},
        {"role": "user",      "content": user_turn.strip()},
        {"role": "assistant", "content": assistant_turn.strip()},
    ]


def prompt_metadata() -> dict:
    """Return a dict suitable for embedding in a LearningRun.notes field."""
    return {
        "socratic_prompt_version": SOCRATIC_SYSTEM_VERSION,
        "socratic_prompt_hash":   _short_hash(SOCRATIC_SYSTEM),
    }


def _short_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:12]
