from __future__ import annotations

import json

from hydralisk.evals.terminal_bench import (
    TerminalBenchRunSettings,
    harbor_job_to_summary_payload,
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
        settings=TerminalBenchRunSettings(min_p=None),
    )

    rendered = render_markdown(receipt)

    assert "Terminal-Bench 2.0 eval gate" in rendered
    assert "Solved / total" in rendered
    assert "`min_p=omitted`" in rendered
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
            "--omit-min-p",
        ]
    )

    assert status == 0
    payload = json.loads((tmp_path / "terminal-bench-summary.json").read_text())
    assert payload["counts"]["properlyAttempted"] == 3
    assert payload["sampler"]["minP"] is None
    assert "qemu-startup" in (tmp_path / "terminal-bench-summary.md").read_text()


def test_terminal_bench_sanitizes_harbor_job_directory(tmp_path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(
        json.dumps({"job_name": "synthetic-terminal-bench", "n_attempts": 2})
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "258396af-2ab3-41b9-b2b7-25655c0a54be",
                "started_at": "2026-06-25T00:00:00Z",
                "updated_at": "2026-06-25T00:03:00Z",
                "finished_at": "2026-06-25T00:03:00Z",
                "n_total_trials": 3,
                "stats": {},
            }
        )
    )

    pass_dir = job_dir / "compile-compcert__first"
    pass_dir.mkdir()
    (pass_dir / "config.json").write_text(
        json.dumps({"task": {"path": "compile-compcert"}})
    )
    (pass_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "compile-compcert",
                "trial_name": "compile-compcert__first",
                "task_id": {"path": "compile-compcert"},
                "finished_at": "2026-06-25T00:01:00Z",
                "verifier_result": {"rewards": {"reward": 1}},
            }
        )
    )

    fail_dir = job_dir / "video-processing__first"
    fail_dir.mkdir()
    (fail_dir / "config.json").write_text(
        json.dumps({"task": {"path": "video-processing"}})
    )
    (fail_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "video-processing",
                "trial_name": "video-processing__first",
                "task_id": {"path": "video-processing"},
                "finished_at": "2026-06-25T00:02:00Z",
                "verifier_result": {"rewards": {"reward": 0}},
            }
        )
    )

    payload = harbor_job_to_summary_payload(job_dir)
    receipt = summarize_payload(payload, settings=TerminalBenchRunSettings())

    assert payload["source"]["type"] == "harbor-job"
    assert payload["source"]["nCompletedTrials"] == 2
    assert receipt["counts"]["total"] == 3
    assert receipt["counts"]["solved"] == 1
    assert receipt["counts"]["failing"] == 1
    assert receipt["counts"]["notStarted"] == 1
    assert receipt["passAt"]["passAt1Solved"] == 1
    assert receipt["passAt"]["passAt1KnownTasks"] == 2


def test_terminal_bench_cli_accepts_harbor_job_directory(tmp_path) -> None:
    job_dir = tmp_path / "job"
    output_dir = tmp_path / "out"
    trial_dir = job_dir / "task-one__abc"
    trial_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(json.dumps({"job_name": "job-one"}))
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "258396af-2ab3-41b9-b2b7-25655c0a54be",
                "started_at": "2026-06-25T00:00:00Z",
                "updated_at": "2026-06-25T00:01:00Z",
                "finished_at": "2026-06-25T00:01:00Z",
                "n_total_trials": 1,
                "stats": {},
            }
        )
    )
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": "task-one"}}))
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "task-one__abc",
                "task_id": {"path": "task-one"},
                "finished_at": "2026-06-25T00:01:00Z",
                "verifier_result": {"rewards": {"reward": 1}},
            }
        )
    )

    status = main(
        [
            "--harbor-job-dir",
            str(job_dir),
            "--output-dir",
            str(output_dir),
            "--json-name",
            "receipt.json",
            "--markdown-name",
            "receipt.md",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "receipt.json").read_text())
    assert payload["counts"]["solved"] == 1
    assert payload["inputSha256"]
    assert "task-one" not in (output_dir / "receipt.md").read_text()
