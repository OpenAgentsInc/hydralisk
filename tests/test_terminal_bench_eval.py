from __future__ import annotations

import json

from hydralisk.evals.terminal_bench import (
    TerminalBenchRunSettings,
    main,
    render_markdown,
    summarize_payload,
)


def test_terminal_bench_summary_computes_denominators_from_counts() -> None:
    receipt = summarize_payload(
        {
            "total": 89,
            "solved": 60,
            "failing": 25,
            "envBroken": 2,
            "notStarted": 2,
            "passAtNAnySolved": 60,
        },
        settings=TerminalBenchRunSettings(claim_status="preliminary"),
        env_broken_task_ids=["qemu-alpine-ssh", "qemu-startup"],
        not_started_task_ids=["queued-a", "queued-b"],
        notable_solved_task_ids=["compile-compcert", "build-pov-ray"],
    )

    assert receipt["counts"] == {
        "total": 89,
        "solved": 60,
        "failing": 25,
        "envBroken": 2,
        "notStarted": 2,
        "attempted": 87,
        "properlyAttempted": 85,
    }
    assert receipt["rates"]["fullDenominatorSolved"] == 0.674157
    assert receipt["rates"]["attemptedSolved"] == 0.689655
    assert receipt["rates"]["properlyAttemptedSolved"] == 0.705882
    assert receipt["taskIds"]["envBroken"] == ["qemu-alpine-ssh", "qemu-startup"]
    assert receipt["taskIds"]["notStarted"] == ["queued-a", "queued-b"]
    assert receipt["publicSafety"]["containsPrompts"] is False
    assert receipt["publicSafety"]["containsResponses"] is False


def test_terminal_bench_summary_separates_task_level_statuses() -> None:
    receipt = summarize_payload(
        {
            "tasks": [
                {
                    "task_id": "compile-compcert",
                    "attempts": [{"status": "failed"}, {"status": "passed"}],
                },
                {"task_id": "qemu-startup", "status": "failed"},
                {"task_id": "video-processing", "attempts": [{"passed": False}]},
                {"task_id": "not-yet", "status": "queued"},
            ]
        },
        settings=TerminalBenchRunSettings(),
        env_broken_task_ids=["qemu-startup"],
    )

    assert receipt["counts"]["solved"] == 1
    assert receipt["counts"]["failing"] == 1
    assert receipt["counts"]["envBroken"] == 1
    assert receipt["counts"]["notStarted"] == 1
    assert receipt["passAt"]["passAt1Solved"] == 0
    assert receipt["passAt"]["passAt1KnownTasks"] == 2
    assert receipt["passAt"]["passAtNAnySolved"] == 1


def test_terminal_bench_markdown_is_public_safe_summary() -> None:
    receipt = summarize_payload(
        {"total": 2, "solved": 1, "failing": 1, "envBroken": 0, "notStarted": 0},
        settings=TerminalBenchRunSettings(),
    )

    rendered = render_markdown(receipt)

    assert "Terminal-Bench 2.0 eval gate" in rendered
    assert "Solved / total" in rendered
    assert "raw prompts" in rendered
    assert "messages" not in rendered.lower()


def test_terminal_bench_cli_writes_json_and_markdown(tmp_path) -> None:
    status = main(
        [
            "--output-dir",
            str(tmp_path),
            "--total",
            "4",
            "--solved",
            "2",
            "--failing",
            "1",
            "--env-broken",
            "1",
            "--not-started",
            "0",
            "--env-broken-task",
            "qemu-startup",
        ]
    )

    assert status == 0
    payload = json.loads((tmp_path / "terminal-bench-summary.json").read_text())
    assert payload["counts"]["properlyAttempted"] == 3
    assert "qemu-startup" in (tmp_path / "terminal-bench-summary.md").read_text()
