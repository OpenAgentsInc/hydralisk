"""Spoken-form text normalization for the OAV-3 TTS seam.

TTS models read written form literally: "AI" becomes "eye" (owner-reported on
the OAV-1 take), "API" becomes a word, domains are spelled like sentences.
``normalize_spoken`` rewrites text into the form Sarah should *say*, applied
by every adapter before synthesis so the fix is backend-independent.

Style decision (OAV-1 QA3, owner feedback): hard letter-spacing
("A P I") produced staccato, clipped audio that also drove clipped visemes in
the lip-synced render. The preferred form is **punctuation-driven**
("A.P.I.") — CosyVoice2/Chirp read dotted initialisms with natural letter
prosody and punctuation-scale pauses. Letter-spacing survives only as the
explicit ``style="spelled"`` fallback for backends whose frontend drops dots.

Two layers, both conservative:

1. A curated lexicon for our product vocabulary (exact, case-sensitive for
   acronyms; case-insensitive for domains).
2. A heuristic: a standalone 2-4 letter ALL-CAPS token that is not a real
   English word (allowlist) is dotted ("ERP" -> "E.R.P."). Ambiguous tokens
   that are also real words ("US", "IT", "OK") are left untouched — wrongly
   spelling out a word is worse than missing an acronym. Mixed-case and
   longer tokens ("Sarah", "SARAH", "OpenAgents") are never touched by the
   heuristic.

Never applied to prompt/reference transcripts — those must match their audio.
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = ["normalize_spoken", "SPOKEN_LEXICON"]

Style = Literal["punctuated", "spelled"]


def _dotted(token: str) -> str:
    return "".join(f"{ch}." for ch in token)


def _spaced(token: str) -> str:
    return " ".join(token)


# Written form -> spoken form per style. Keys are matched as whole words,
# case-sensitively (acronyms) unless noted. Values are what the voice says.
SPOKEN_LEXICON: dict[str, dict[Style, str]] = {
    "AI": {"punctuated": "A.I.", "spelled": "A I"},
    "API": {"punctuated": "A.P.I.", "spelled": "A P I"},
    "APIs": {"punctuated": "A.P.I.s", "spelled": "A P Is"},
    "CRM": {"punctuated": "C.R.M.", "spelled": "C R M"},
    "URL": {"punctuated": "U.R.L.", "spelled": "U R L"},
    "URLs": {"punctuated": "U.R.L.s", "spelled": "U R Ls"},
    "FAQ": {"punctuated": "F.A.Q.", "spelled": "F A Q"},
    "TTS": {"punctuated": "T.T.S.", "spelled": "T T S"},
    "QA": {"punctuated": "Q.A.", "spelled": "Q A"},
    "MVP": {"punctuated": "M.V.P.", "spelled": "M V P"},
    "B2B": {"punctuated": "B two B", "spelled": "B two B"},
    "SaaS": {"punctuated": "sass", "spelled": "sass"},
}

# Case-insensitive replacements (domains and dotted names read aloud).
_SPOKEN_LEXICON_CI: dict[str, str] = {
    "openagents.com": "open agents dot com",
}

# 2-4 letter ALL-CAPS tokens that are real words or read naturally as-is;
# the heuristic must not spell these out. Conservative by design.
_ALLCAPS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI",
        "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OH", "OK", "ON",
        "OR", "PM", "SO", "TO", "UP", "US", "WE",
        "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HER", "HIM",
        "HIS", "HOW", "NEW", "NOT", "NOW", "OUR", "OUT", "SHE", "THE",
        "WAS", "WHO", "WHY", "YES", "YOU",
        "BEST", "DAMN", "DATA", "DEAL", "DEMO", "DOES", "DONE", "FREE",
        "FROM", "HAVE", "HERE", "JUST", "LIVE", "MAKE", "MORE", "MOST",
        "NEED", "ONLY", "OVER", "PAIN", "REAL", "SELL", "STOP", "THAT",
        "THEY", "THIS", "WANT", "WHAT", "WHEN", "WITH", "WORK", "YOUR",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _heuristic_spell_out(token: str, style: Style) -> str | None:
    """Rewrite a standalone acronym-looking token, or None to keep it."""

    if not token.isupper():
        return None
    if not token.isalpha():
        return None
    if not 2 <= len(token) <= 4:
        return None
    if token in _ALLCAPS_ALLOWLIST:
        return None
    return _dotted(token) if style == "punctuated" else _spaced(token)


def normalize_spoken(text: str, style: Style = "punctuated") -> str:
    """Rewrite written-form text into the spoken form a TTS voice should say.

    ``style="punctuated"`` (default) emits dotted initialisms ("A.I.") for
    natural prosody; ``style="spelled"`` emits hard letter-spacing ("A I") as
    the fallback for frontends that drop dots. Idempotent for lexicon outputs
    and safe on empty input.
    """

    if not text:
        return text

    for written, spoken in _SPOKEN_LEXICON_CI.items():
        text = re.sub(rf"(?i)\b{re.escape(written)}\b", spoken, text)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in SPOKEN_LEXICON:
            return SPOKEN_LEXICON[token][style]
        spelled = _heuristic_spell_out(token, style)
        return spelled if spelled is not None else token

    text = _WORD_RE.sub(_replace, text)
    # A dotted form at sentence end yields ".." — collapse it (keep "...").
    return re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
