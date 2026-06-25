from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


SCHEMA = "hydralisk.evals.terminal_bench.summary.v1"

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.:/-]+")
_STATUS_ALIASES = {
    "pass": "solved",
    "passed": "solved",
    "success": "solved",
    "solved": "solved",
    "fail": "failing",
    "failed": "failing",
    "failure": "failing",
    "failing": "failing",
    "timeout": "failing",
    "error": "env_broken",
    "errored": "env_broken",
    "env_broken": "env_broken",
    "env-broken": "env_broken",
    "environment_broken": "env_broken",
    "environment-broken": "env_broken",
    "not_started": "not_started",
    "not-started": "not_started",
    "pending": "not_started",
    "queued": "not_started",
    "skipped": "not_started",
}
_COUNT_KEYS = ("solved", "failing", "envBroken", "notStarted")


@dataclass(frozen=True)
class TerminalBenchRunSettings:
    benchmark_ref: str = "terminal-bench@2.0"
    benchmark_version: str = "2.0"
    benchmark_repository: str = "https://github.com/harbor-framework/terminal-bench-2"
    harness_repository: str = "https://github.com/harbor-framework/harbor"
    runner: str = "harbor"
    runner_version: str = "unknown"
    agent: str = "terminus-2"
    model: str = "openai/glm-5.2-reap-504b-g4"
    model_alias: str = "glm-5.2-reap-504b-g4"
    model_profile_ref: str = "profiles/glm-5.2-reap-504b-b12x-g4.json"
    model_revision: str = (
        "0xSero/GLM-5.2-504B@cb6b1e0451b9d560cda864f84187869c9a679712"
    )
    hardware_profile: str = "4x RTX PRO 6000 G4 on admitted 8x fallback host"
    n_concurrent: int = 1
    max_attempts: int = 5
    timeout_seconds: int = 3600
    retry_policy: str = "pass@1 plus up to 4 queued retries for pass@5"
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    max_tokens: int = 1024
    enable_thinking: bool = False
    claim_status: str = "preliminary"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_task_id(value: object) -> str:
    text = str(value or "unknown").strip()
    text = _TASK_ID_RE.sub("_", text)
    return text[:160] or "unknown"


def _normalize_status(value: object) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return _STATUS_ALIASES.get(text, "failing")


def _attempt_passed(attempt: object) -> bool:
    if isinstance(attempt, bool):
        return attempt
    if not isinstance(attempt, Mapping):
        return False
    if isinstance(attempt.get("passed"), bool):
        return bool(attempt["passed"])
    if isinstance(attempt.get("success"), bool):
        return bool(attempt["success"])
    return _normalize_status(attempt.get("status")) == "solved"


def _task_status(
    task: Mapping[str, Any],
    *,
    env_broken_task_ids: set[str],
    not_started_task_ids: set[str],
) -> tuple[str, int, int, bool | None]:
    task_id = _clean_task_id(task.get("task_id") or task.get("id") or task.get("name"))
    if task_id in env_broken_task_ids:
        return "env_broken", 0, 0, None
    if task_id in not_started_task_ids:
        return "not_started", 0, 0, None

    attempts_raw = task.get("attempts")
    attempts: list[Any]
    if isinstance(attempts_raw, list):
        attempts = attempts_raw
    else:
        attempts = []

    if attempts:
        passed_attempts = sum(1 for attempt in attempts if _attempt_passed(attempt))
        first_attempt_passed = _attempt_passed(attempts[0])
        if passed_attempts:
            return "solved", len(attempts), passed_attempts, first_attempt_passed
        return "failing", len(attempts), 0, first_attempt_passed

    status = _normalize_status(task.get("status") or task.get("result"))
    if status == "solved":
        return "solved", int(task.get("attempt_count") or 1), 1, None
    if status == "not_started":
        return "not_started", 0, 0, None
    if status == "env_broken":
        return "env_broken", int(task.get("attempt_count") or 0), 0, None
    return "failing", int(task.get("attempt_count") or 1), 0, None


def _round_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _counts_from_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    env_broken_task_ids: set[str],
    not_started_task_ids: set[str],
) -> dict[str, Any]:
    counts = {"solved": 0, "failing": 0, "envBroken": 0, "notStarted": 0}
    solved_task_ids: list[str] = []
    failing_task_ids: list[str] = []
    env_task_ids: list[str] = []
    not_started_ids: list[str] = []
    attempts_total = 0
    pass_at_1_known = 0
    pass_at_1_solved = 0
    pass_at_n_solved = 0

    for task in tasks:
        task_id = _clean_task_id(task.get("task_id") or task.get("id") or task.get("name"))
        status, attempts, passed_attempts, first_attempt_passed = _task_status(
            task,
            env_broken_task_ids=env_broken_task_ids,
            not_started_task_ids=not_started_task_ids,
        )
        attempts_total += attempts
        if first_attempt_passed is not None:
            pass_at_1_known += 1
            pass_at_1_solved += int(first_attempt_passed)
        pass_at_n_solved += int(passed_attempts > 0 or status == "solved")

        if status == "solved":
            counts["solved"] += 1
            solved_task_ids.append(task_id)
        elif status == "env_broken":
            counts["envBroken"] += 1
            env_task_ids.append(task_id)
        elif status == "not_started":
            counts["notStarted"] += 1
            not_started_ids.append(task_id)
        else:
            counts["failing"] += 1
            failing_task_ids.append(task_id)

    counts["total"] = sum(counts[key] for key in _COUNT_KEYS)
    return {
        "counts": counts,
        "attemptsTotal": attempts_total,
        "passAt1KnownTasks": pass_at_1_known,
        "passAt1Solved": pass_at_1_solved,
        "passAtNAnySolved": pass_at_n_solved,
        "taskIds": {
            "solved": solved_task_ids,
            "failing": failing_task_ids,
            "envBroken": env_task_ids,
            "notStarted": not_started_ids,
        },
    }


def _counts_from_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    counts_raw = payload.get("counts", payload)
    if not isinstance(counts_raw, Mapping):
        raise ValueError("counts payload must be a JSON object")

    def count(*names: str) -> int:
        for name in names:
            if name in counts_raw:
                value = int(counts_raw[name])
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                return value
        return 0

    counts = {
        "solved": count("solved", "passed"),
        "failing": count("failing", "failed"),
        "envBroken": count("envBroken", "env_broken", "environment_broken"),
        "notStarted": count("notStarted", "not_started"),
    }
    provided_total = count("total", "totalTasks")
    computed_total = sum(counts[key] for key in _COUNT_KEYS)
    if provided_total and provided_total != computed_total:
        raise ValueError(
            f"total {provided_total} does not match status counts {computed_total}"
        )
    counts["total"] = computed_total
    return {
        "counts": counts,
        "attemptsTotal": int(payload.get("attemptsTotal") or 0),
        "passAt1KnownTasks": int(payload.get("passAt1KnownTasks") or 0),
        "passAt1Solved": int(payload.get("passAt1Solved") or 0),
        "passAtNAnySolved": int(payload.get("passAtNAnySolved") or counts["solved"]),
        "taskIds": {
            "solved": [_clean_task_id(item) for item in payload.get("solvedTaskIds", [])],
            "failing": [_clean_task_id(item) for item in payload.get("failingTaskIds", [])],
            "envBroken": [
                _clean_task_id(item) for item in payload.get("envBrokenTaskIds", [])
            ],
            "notStarted": [
                _clean_task_id(item) for item in payload.get("notStartedTaskIds", [])
            ],
        },
    }


def _task_id_from_harbor_mapping(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None

    path = value.get("path")
    if path:
        return _clean_task_id(Path(str(path)).name)

    org = value.get("org")
    name = value.get("name")
    if org and name:
        return _clean_task_id(f"{org}/{name}")

    if name:
        return _clean_task_id(name)

    return None


def _task_id_from_trial_dir_name(value: object) -> str:
    text = str(value or "unknown").strip()
    if "__" in text:
        text = text.split("__", 1)[0]
    return _clean_task_id(text)


def _harbor_trial_task_id(trial_dir: Path, trial: Mapping[str, Any]) -> str:
    task_id = _task_id_from_harbor_mapping(trial.get("task_id"))
    if task_id:
        return task_id

    config = trial.get("config")
    if isinstance(config, Mapping):
        task_id = _task_id_from_harbor_mapping(config.get("task"))
        if task_id:
            return task_id

    task_name = trial.get("task_name")
    if task_name:
        return _clean_task_id(task_name)

    config_path = trial_dir / "config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, Mapping):
            task_id = _task_id_from_harbor_mapping(payload.get("task"))
            if task_id:
                return task_id

    return _task_id_from_trial_dir_name(trial.get("trial_name") or trial_dir.name)


def _harbor_trial_status(trial: Mapping[str, Any]) -> str:
    if not trial.get("finished_at"):
        return "not_started"

    verifier_result = trial.get("verifier_result")
    rewards = (
        verifier_result.get("rewards")
        if isinstance(verifier_result, Mapping)
        else None
    )
    if isinstance(rewards, Mapping):
        if "reward" in rewards:
            try:
                return "solved" if float(rewards["reward"]) > 0 else "failing"
            except (TypeError, ValueError):
                return "failing"
        if len(rewards) == 1:
            value = next(iter(rewards.values()))
            try:
                return "solved" if float(value) > 0 else "failing"
            except (TypeError, ValueError):
                return "failing"

    exception_info = trial.get("exception_info")
    if isinstance(exception_info, Mapping):
        if exception_info.get("exception_type") == "CancelledError":
            return "not_started"
        return "failing"

    return "failing"


def _harbor_job_config_value(job_config: Mapping[str, Any], name: str) -> Any:
    value = job_config.get(name)
    if value is not None:
        return value
    return None


def harbor_job_to_summary_payload(job_dir: Path | str) -> dict[str, Any]:
    """Convert a Harbor job directory into a sanitized Terminal-Bench payload.

    This intentionally reads only Harbor's JSON metadata files:
    job-level ``result.json``/``config.json`` and child trial ``result.json`` /
    ``config.json`` files. It does not read agent trajectories, terminal panes,
    recordings, logs, verifier artifacts, or environment files.
    """

    root = Path(job_dir)
    job_result_path = root / "result.json"
    job_config_path = root / "config.json"
    if not job_result_path.exists():
        raise FileNotFoundError(f"missing Harbor job result: {job_result_path}")
    if not job_config_path.exists():
        raise FileNotFoundError(f"missing Harbor job config: {job_config_path}")

    job_result = json.loads(job_result_path.read_text())
    job_config = json.loads(job_config_path.read_text())
    if not isinstance(job_result, Mapping):
        raise ValueError("Harbor job result must be a JSON object")
    if not isinstance(job_config, Mapping):
        raise ValueError("Harbor job config must be a JSON object")

    attempts_by_task: dict[str, list[dict[str, str]]] = {}
    for child in sorted(root.iterdir()):
        result_path = child / "result.json"
        if not child.is_dir() or not result_path.exists():
            continue
        trial = json.loads(result_path.read_text())
        if not isinstance(trial, Mapping):
            continue
        task_id = _harbor_trial_task_id(child, trial)
        attempts_by_task.setdefault(task_id, []).append(
            {"status": _harbor_trial_status(trial)}
        )

    tasks = [
        {"task_id": task_id, "attempts": attempts}
        for task_id, attempts in sorted(attempts_by_task.items())
    ]

    n_total_trials = int(job_result.get("n_total_trials") or len(tasks))
    missing = max(n_total_trials - len(tasks), 0)
    if missing:
        job_label = _clean_task_id(job_config.get("job_name") or job_result.get("id"))
        for index in range(1, missing + 1):
            tasks.append(
                {
                    "task_id": f"{job_label}.unknown-not-started-{index}",
                    "status": "not_started",
                }
            )

    return {
        "source": {
            "type": "harbor-job",
            "jobName": _harbor_job_config_value(job_config, "job_name"),
            "jobId": str(job_result.get("id") or uuid4()),
            "finished": job_result.get("finished_at") is not None,
            "startedAt": job_result.get("started_at"),
            "finishedAt": job_result.get("finished_at"),
            "nTotalTrials": n_total_trials,
            "nCompletedTrials": len(attempts_by_task),
        },
        "tasks": tasks,
    }


def _public_payload_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def summarize_payload(
    payload: Mapping[str, Any],
    *,
    settings: TerminalBenchRunSettings,
    env_broken_task_ids: Iterable[str] = (),
    not_started_task_ids: Iterable[str] = (),
    notable_solved_task_ids: Iterable[str] = (),
    input_sha256: str | None = None,
) -> dict[str, Any]:
    env_ids = {_clean_task_id(item) for item in env_broken_task_ids}
    not_started_ids = {_clean_task_id(item) for item in not_started_task_ids}

    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        summary = _counts_from_tasks(
            [task for task in tasks if isinstance(task, Mapping)],
            env_broken_task_ids=env_ids,
            not_started_task_ids=not_started_ids,
        )
    else:
        summary = _counts_from_summary(payload)
        summary["taskIds"]["envBroken"] = sorted(
            set(summary["taskIds"]["envBroken"]) | env_ids
        )
        summary["taskIds"]["notStarted"] = sorted(
            set(summary["taskIds"]["notStarted"]) | not_started_ids
        )

    counts = summary["counts"]
    attempted = counts["total"] - counts["notStarted"]
    properly_attempted = counts["total"] - counts["notStarted"] - counts["envBroken"]
    pass_at_1_known = summary["passAt1KnownTasks"]
    pass_at_1_solved = summary["passAt1Solved"] if pass_at_1_known else None

    return {
        "schema": SCHEMA,
        "createdAt": _now_iso(),
        "publicSafe": True,
        "benchmark": {
            "name": "Terminal-Bench",
            "datasetRef": settings.benchmark_ref,
            "version": settings.benchmark_version,
            "repository": settings.benchmark_repository,
            "harnessRepository": settings.harness_repository,
        },
        "runner": {
            "name": settings.runner,
            "version": settings.runner_version,
            "agent": settings.agent,
            "model": settings.model,
            "nConcurrent": settings.n_concurrent,
            "timeoutSeconds": settings.timeout_seconds,
            "maxAttempts": settings.max_attempts,
            "retryPolicy": settings.retry_policy,
        },
        "model": {
            "alias": settings.model_alias,
            "profileRef": settings.model_profile_ref,
            "revision": settings.model_revision,
            "hardwareProfile": settings.hardware_profile,
        },
        "sampler": {
            "minP": settings.min_p,
            "repetitionPenalty": settings.repetition_penalty,
            "maxTokens": settings.max_tokens,
            "enableThinking": settings.enable_thinking,
        },
        "counts": {
            **counts,
            "attempted": attempted,
            "properlyAttempted": properly_attempted,
        },
        "rates": {
            "fullDenominatorSolved": _round_rate(counts["solved"], counts["total"]),
            "attemptedSolved": _round_rate(counts["solved"], attempted),
            "properlyAttemptedSolved": _round_rate(counts["solved"], properly_attempted),
            "knownPassAt1": _round_rate(pass_at_1_solved or 0, pass_at_1_known),
            "passAtN": _round_rate(summary["passAtNAnySolved"], counts["total"]),
        },
        "passAt": {
            "passAt1Solved": pass_at_1_solved,
            "passAt1KnownTasks": pass_at_1_known,
            "passAtNAnySolved": summary["passAtNAnySolved"],
            "maxAttempts": settings.max_attempts,
        },
        "denominatorDefinitions": {
            "total": "all Terminal-Bench 2.0 task IDs in the run set",
            "attempted": "total minus not-started tasks",
            "properlyAttempted": "attempted minus environment-broken tasks",
            "fullDenominatorSolved": "solved / total",
            "attemptedSolved": "solved / attempted",
            "properlyAttemptedSolved": "solved / properlyAttempted",
        },
        "taskIds": {
            "envBroken": summary["taskIds"]["envBroken"],
            "notStarted": summary["taskIds"]["notStarted"],
            "notableSolved": [_clean_task_id(item) for item in notable_solved_task_ids],
        },
        "claimStatus": settings.claim_status,
        "inputSha256": input_sha256,
        "comparisonBoundary": (
            "Do not compare publicly unless benchmark version, Harbor version, agent, "
            "model alias, sampler settings, retry policy, timeout, and denominator "
            "definitions are all named."
        ),
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsHiddenReasoning": False,
            "containsPrivateSource": False,
            "containsRawBenchmarkLogs": False,
        },
    }


def render_markdown(receipt: Mapping[str, Any]) -> str:
    counts = receipt["counts"]
    rates = receipt["rates"]
    task_ids = receipt["taskIds"]
    runner = receipt["runner"]
    sampler = receipt["sampler"]
    benchmark = receipt["benchmark"]
    model = receipt["model"]

    def pct(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "unknown"
        return f"{value * 100:.1f}%"

    env_ids = task_ids.get("envBroken", [])
    not_started_ids = task_ids.get("notStarted", [])
    notable_ids = task_ids.get("notableSolved", [])

    return f"""# GLM-5.2 504B REAP Terminal-Bench 2.0 eval gate

Date: {receipt["createdAt"]}

Schema: `{receipt["schema"]}`

Public-safety boundary: this report contains benchmark identifiers, counts,
denominator definitions, sanitized task IDs, settings, hashes, and aggregate
metrics only. It contains no raw prompts, raw responses, private source, hidden
reasoning traces, bearer tokens, model-provider credentials, raw benchmark
logs, weights, checkpoints, compiled engines, or profiler dumps.

## Benchmark

- Dataset: `{benchmark["datasetRef"]}`
- Version: `{benchmark["version"]}`
- Dataset repository: {benchmark["repository"]}
- Harness repository: {benchmark["harnessRepository"]}
- Runner: `{runner["name"]}`
- Runner version: `{runner["version"]}`
- Agent: `{runner["agent"]}`
- Model argument: `{runner["model"]}`
- Model alias: `{model["alias"]}`
- Profile: `{model["profileRef"]}`
- Hardware profile: `{model["hardwareProfile"]}`
- Concurrent tasks: `{runner["nConcurrent"]}`
- Timeout seconds: `{runner["timeoutSeconds"]}`
- Max attempts: `{runner["maxAttempts"]}`
- Retry policy: `{runner["retryPolicy"]}`

## Sampler

- `min_p={sampler["minP"]}`
- `repetition_penalty={sampler["repetitionPenalty"]}`
- `max_tokens={sampler["maxTokens"]}`
- `enable_thinking={str(sampler["enableThinking"]).lower()}`

## Counts

| Category | Count |
| --- | ---: |
| Total tasks | {counts["total"]} |
| Solved | {counts["solved"]} |
| Failing | {counts["failing"]} |
| Environment-broken | {counts["envBroken"]} |
| Not started | {counts["notStarted"]} |
| Attempted | {counts["attempted"]} |
| Properly attempted | {counts["properlyAttempted"]} |

Rates:

- Solved / total: `{pct(rates["fullDenominatorSolved"])}`
- Solved / attempted: `{pct(rates["attemptedSolved"])}`
- Solved / properly attempted: `{pct(rates["properlyAttemptedSolved"])}`
- pass@N / total: `{pct(rates["passAtN"])}`
- Known pass@1: `{pct(rates["knownPassAt1"])}`

Denominators:

- Total: {receipt["denominatorDefinitions"]["total"]}.
- Attempted: {receipt["denominatorDefinitions"]["attempted"]}.
- Properly attempted: {receipt["denominatorDefinitions"]["properlyAttempted"]}.

Environment-broken task IDs:

{_render_id_list(env_ids)}

Not-started task IDs:

{_render_id_list(not_started_ids)}

Notable solved task IDs:

{_render_id_list(notable_ids)}

## Claim boundary

Claim status: `{receipt["claimStatus"]}`.

{receipt["comparisonBoundary"]}

The full leaderboard-style claim is not admitted until the run is final, all
queued retries are accounted for, and the committed receipt names the exact
Harbor version and agent settings used for the rollout.
"""


def _render_id_list(ids: Iterable[str]) -> str:
    values = list(ids)
    if not values:
        return "- none"
    return "\n".join(f"- `{value}`" for value in values)


def _settings_from_args(args: argparse.Namespace) -> TerminalBenchRunSettings:
    return TerminalBenchRunSettings(
        benchmark_ref=args.benchmark_ref,
        benchmark_version=args.benchmark_version,
        benchmark_repository=args.benchmark_repository,
        harness_repository=args.harness_repository,
        runner=args.runner,
        runner_version=args.runner_version,
        agent=args.agent,
        model=args.model,
        model_alias=args.model_alias,
        model_profile_ref=args.model_profile_ref,
        model_revision=args.model_revision,
        hardware_profile=args.hardware_profile,
        n_concurrent=args.n_concurrent,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        retry_policy=args.retry_policy,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        claim_status=args.claim_status,
    )


def _payload_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    if args.harbor_job_dir:
        payload = harbor_job_to_summary_payload(Path(args.harbor_job_dir))
        return payload, _public_payload_sha256(payload)

    if args.input:
        raw = Path(args.input).read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    required = [args.total, args.solved, args.failing, args.env_broken, args.not_started]
    if any(value is None for value in required):
        raise SystemExit(
            "provide --input or all count flags: --total --solved --failing "
            "--env-broken --not-started"
        )
    return (
        {
            "total": args.total,
            "solved": args.solved,
            "failing": args.failing,
            "envBroken": args.env_broken,
            "notStarted": args.not_started,
            "passAtNAnySolved": args.pass_at_n_any_solved,
            "passAt1Solved": args.pass_at_1_solved or 0,
            "passAt1KnownTasks": args.pass_at_1_known_tasks or 0,
        },
        None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a public-safe Terminal-Bench summary receipt."
    )
    parser.add_argument("--input", help="Sanitized Harbor/Terminal-Bench JSON summary")
    parser.add_argument(
        "--harbor-job-dir",
        help=(
            "Harbor job directory to sanitize into a Terminal-Bench summary input. "
            "Only result/config JSON files are read."
        ),
    )
    parser.add_argument("--output-dir", default=".hydralisk/terminal-bench-summary")
    parser.add_argument("--json-name", default="terminal-bench-summary.json")
    parser.add_argument("--markdown-name", default="terminal-bench-summary.md")
    parser.add_argument("--benchmark-ref", default="terminal-bench@2.0")
    parser.add_argument("--benchmark-version", default="2.0")
    parser.add_argument(
        "--benchmark-repository",
        default="https://github.com/harbor-framework/terminal-bench-2",
    )
    parser.add_argument(
        "--harness-repository",
        default="https://github.com/harbor-framework/harbor",
    )
    parser.add_argument("--runner", default="harbor")
    parser.add_argument("--runner-version", default="unknown")
    parser.add_argument("--agent", default="terminus-2")
    parser.add_argument("--model", default="openai/glm-5.2-reap-504b-g4")
    parser.add_argument("--model-alias", default="glm-5.2-reap-504b-g4")
    parser.add_argument(
        "--model-profile-ref",
        default="profiles/glm-5.2-reap-504b-b12x-g4.json",
    )
    parser.add_argument(
        "--model-revision",
        default="0xSero/GLM-5.2-504B@cb6b1e0451b9d560cda864f84187869c9a679712",
    )
    parser.add_argument(
        "--hardware-profile",
        default="4x RTX PRO 6000 G4 on admitted 8x fallback host",
    )
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--retry-policy",
        default="pass@1 plus up to 4 queued retries for pass@5",
    )
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--claim-status", default="preliminary")
    parser.add_argument("--total", type=int)
    parser.add_argument("--solved", type=int)
    parser.add_argument("--failing", type=int)
    parser.add_argument("--env-broken", type=int)
    parser.add_argument("--not-started", type=int)
    parser.add_argument("--pass-at-n-any-solved", type=int)
    parser.add_argument("--pass-at-1-solved", type=int)
    parser.add_argument("--pass-at-1-known-tasks", type=int)
    parser.add_argument("--env-broken-task", action="append", default=[])
    parser.add_argument("--not-started-task", action="append", default=[])
    parser.add_argument("--notable-solved-task", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload, input_sha256 = _payload_from_args(args)
    settings = _settings_from_args(args)
    receipt = summarize_payload(
        payload,
        settings=settings,
        env_broken_task_ids=args.env_broken_task,
        not_started_task_ids=args.not_started_task,
        notable_solved_task_ids=args.notable_solved_task,
        input_sha256=input_sha256,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(receipt))
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
