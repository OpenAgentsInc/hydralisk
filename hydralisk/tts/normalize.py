"""Spoken-form text normalization for the OAV-3 TTS seam.

TTS models read written form literally: "AI" becomes "eye" (owner-reported on
the OAV-1 take), "API" becomes a word, domains are spelled like sentences.
``normalize_spoken`` rewrites text into the form Sarah should *say*, applied
by every adapter before synthesis so the fix is backend-independent.

Two layers, both conservative:

1. A curated lexicon for our product vocabulary (exact, case-sensitive for
   acronyms; case-insensitive for domains).
2. A heuristic: a standalone 2-4 letter ALL-CAPS token that is not a real
   English word (allowlist) is letter-spaced ("CRM" -> "C R M"). Ambiguous
   tokens that are also real words ("US", "IT", "OK") are left untouched —
   wrongly spelling out a word is worse than missing an acronym. Mixed-case
   and longer tokens ("Sarah", "SARAH", "OpenAgents") are never touched by
   the heuristic.

Never applied to prompt/reference transcripts — those must match their audio.
"""

from __future__ import annotations

import re

__all__ = ["normalize_spoken", "SPOKEN_LEXICON"]

# Written form -> spoken form. Keys are matched as whole words,
# case-sensitively (acronyms) unless noted. Values are what the voice says.
SPOKEN_LEXICON: dict[str, str] = {
    "AI": "A.I.",
    "API": "A P I",
    "APIs": "A P Is",
    "CRM": "C R M",
    "URL": "U R L",
    "URLs": "U R Ls",
    "FAQ": "F A Q",
    "TTS": "T T S",
    "QA": "Q A",
    "MVP": "M V P",
    "B2B": "B two B",
    "SaaS": "sass",
}

# Case-insensitive replacements (domains and dotted names read aloud).
_SPOKEN_LEXICON_CI: dict[str, str] = {
    "openagents.com": "open agents dot com",
}

# 2-4 letter ALL-CAPS tokens that are real words or read naturally as-is;
# the heuristic must not letter-space these. Conservative by design.
_ALLCAPS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI",
        "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OH", "OK", "ON",
        "OR", "PM", "SO", "TO", "UP", "US", "WE",
        "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HER", "HIM",
        "HIS", "HOW", "NEW", "NOT", "NOW", "OUR", "OUT", "SHE", "THE",
        "WAS", "WHO", "WHY", "YES", "YOU",
        "BEST", "DAMN", "DATA", "DEAL", "DEMO", "DOES", "DONE", "FREE", "FROM",
        "HAVE", "HERE", "JUST", "LIVE", "MAKE", "MORE", "MOST", "NEED",
        "ONLY", "OVER", "PAIN", "REAL", "SELL", "STOP", "THAT", "THEY",
        "THIS", "WANT", "WHAT", "WHEN", "WITH", "WORK", "YOUR",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _heuristic_spell_out(token: str) -> str | None:
    """Letter-space a standalone acronym-looking token, or None to keep it."""

    if not token.isupper():
        return None
    if not token.isalpha():
        return None
    if not 2 <= len(token) <= 4:
        return None
    if token in _ALLCAPS_ALLOWLIST:
        return None
    return " ".join(token)


def normalize_spoken(text: str) -> str:
    """Rewrite written-form text into the spoken form a TTS voice should say.

    Idempotent for lexicon outputs ("A.I." is not a bare word token) and safe
    on empty input.
    """

    if not text:
        return text

    for written, spoken in _SPOKEN_LEXICON_CI.items():
        text = re.sub(rf"(?i)\b{re.escape(written)}\b", spoken, text)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in SPOKEN_LEXICON:
            return SPOKEN_LEXICON[token]
        spelled = _heuristic_spell_out(token)
        return spelled if spelled is not None else token

    return _WORD_RE.sub(_replace, text)
