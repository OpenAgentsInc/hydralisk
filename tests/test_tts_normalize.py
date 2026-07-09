from __future__ import annotations

import pytest

from hydralisk.tts.normalize import normalize_spoken


class TestLexicon:
    def test_ai_becomes_dotted(self) -> None:
        assert normalize_spoken("I'm an AI, and I sell AI employees.") == (
            "I'm an A.I., and I sell A.I. employees."
        )

    def test_api_letter_spaced(self) -> None:
        assert normalize_spoken("Use the API today.") == "Use the A P I today."

    def test_url_letter_spaced(self) -> None:
        assert normalize_spoken("Paste the URL here.") == "Paste the U R L here."

    def test_domain_read_aloud(self) -> None:
        assert normalize_spoken("Visit openagents.com now.") == (
            "Visit open agents dot com now."
        )

    def test_domain_case_insensitive(self) -> None:
        assert normalize_spoken("Visit OpenAgents.com now.") == (
            "Visit open agents dot com now."
        )

    def test_saas_and_b2b(self) -> None:
        assert normalize_spoken("B2B SaaS teams") == "B two B sass teams"

    def test_plural_acronym(self) -> None:
        assert normalize_spoken("Our APIs are fast.") == "Our A P Is are fast."


class TestHeuristic:
    def test_unknown_acronym_letter_spaced(self) -> None:
        assert normalize_spoken("Sync your CRM and ERP.") == (
            "Sync your C R M and E R P."
        )

    def test_us_left_alone(self) -> None:
        # "US" is a real word in caps contexts ("tell US") — conservative: keep.
        assert normalize_spoken("Talk to US about the US market.") == (
            "Talk to US about the US market."
        )

    def test_it_and_ok_left_alone(self) -> None:
        assert normalize_spoken("IT teams say OK.") == "IT teams say OK."

    def test_sarah_all_caps_left_alone(self) -> None:
        # 5 letters: outside the 2-4 acronym window, never touched.
        assert normalize_spoken("SARAH is live.") == "SARAH is live."

    def test_mixed_case_never_touched(self) -> None:
        assert normalize_spoken("OpenAgents ships Khala.") == (
            "OpenAgents ships Khala."
        )

    def test_shouted_words_left_alone(self) -> None:
        assert normalize_spoken("WORK WITH REAL DEMO DATA") == (
            "WORK WITH REAL DEMO DATA"
        )

    def test_alphanumeric_not_letter_spaced(self) -> None:
        # heuristic requires alphabetic-only; B2B is lexicon, GPT4 stays.
        assert normalize_spoken("GPT4 is here.") == "GPT4 is here."


class TestSafety:
    def test_empty(self) -> None:
        assert normalize_spoken("") == ""

    def test_idempotent_on_lexicon_output(self) -> None:
        once = normalize_spoken("An AI with an API.")
        assert normalize_spoken(once) == once

    def test_full_sarah_opener(self) -> None:
        raw = (
            "Hi, I'm Sarah. I'm an AI, and I sell what I am: AI employees "
            "that actually do work."
        )
        assert normalize_spoken(raw) == (
            "Hi, I'm Sarah. I'm an A.I., and I sell what I am: A.I. employees "
            "that actually do work."
        )


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("TTS output", "T T S output"),
        ("run QA on the MVP", "run Q A on the M V P"),
        ("read the FAQ", "read the F A Q"),
    ],
)
def test_lexicon_table(raw: str, spoken: str) -> None:
    assert normalize_spoken(raw) == spoken
