from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


FABLE_SCHEMA = "hydralisk.deepseek-v4-fable.adapter-compatibility.v1"
FABLE_LOAD_CANARY_SCHEMA = "hydralisk.deepseek-v4-fable.load-canary.v1"
FABLE_LAB_EVAL_SCHEMA = "hydralisk.deepseek-v4-fable.lab-eval-decision.v1"
FABLE_RETARGET_SCHEMA = "hydralisk.deepseek-v4-fable.retarget-plan.v1"
FABLE_OPROJ_OWNERSHIP_SCHEMA = "hydralisk.deepseek-v4-fable.o-proj-ownership.v1"
FABLE_TRANSFORM_SMOKE_SCHEMA = "hydralisk.deepseek-v4-fable.transform-smoke.v1"
FABLE_REPO = "Chunjiang-Intelligence/DeepSeek-v4-Fable"
FABLE_REVISION = "999909137c15e0b5539fee887431824fa7cb5b10"
FABLE_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROFILE_REF = "profiles/deepseek-v4-fable-adapter-g4.json"
ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/67"
LOAD_CANARY_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/68"
POLICY_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/69"
LAB_EVAL_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/70"
RETARGET_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/71"
OPROJ_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/72"
TRANSFORM_SMOKE_ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/73"

SMALL_METADATA_FILES = (
    "adapter_config.json",
    "generation_config.json",
    "merge_info.json",
    "config.json",
    "model.safetensors.index.json",
)
ADAPTER_FILE = "adapter_model.safetensors"
DEFAULT_RECORDED_FILES = (*SMALL_METADATA_FILES, ADAPTER_FILE)
MERGED_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
LORA_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
FABLE_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FABLE_PACKED_FAMILIES = {
    "attention_fused_wqa_wkv": ("q_proj", "k_proj", "v_proj"),
    "swiglu_gate_up_proj": ("gate_proj", "up_proj"),
    "attention_output_o_proj": ("o_proj",),
    "direct_down_proj": ("down_proj",),
}
EXPECTED_LORA_CONTEXTS = {
    "q_proj": ("attention",),
    "k_proj": ("attention",),
    "v_proj": ("attention",),
    "o_proj": ("attention",),
    "gate_proj": ("mlp",),
    "up_proj": ("mlp",),
    "down_proj": ("mlp",),
}


class FableProbeError(ValueError):
    pass


@dataclass(frozen=True)
class RecordedFile:
    name: str
    bytes: int | None
    sha256: str | None
    source: str
    downloadedPayload: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterTargetMatch:
    target: str
    matchedRuntimeModules: tuple[str, ...]
    status: str


def validate_requested_files(
    files: Iterable[str],
    *,
    allow_merged_shards: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(files))
    forbidden = [name for name in normalized if MERGED_SHARD_RE.match(name)]
    if forbidden and not allow_merged_shards:
        raise FableProbeError(
            "refusing merged checkpoint shard fetch without "
            "--allow-merged-shard-files: " + ", ".join(forbidden)
        )
    allowed = set(DEFAULT_RECORDED_FILES)
    unknown = [
        name
        for name in normalized
        if name not in allowed and not (allow_merged_shards and MERGED_SHARD_RE.match(name))
    ]
    if unknown:
        raise FableProbeError("unsupported Fable probe file(s): " + ", ".join(unknown))
    return normalized


def load_runtime_module_names(path: Path) -> tuple[str, ...]:
    text = path.read_text().strip()
    if not text:
        return ()
    if text.startswith("["):
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise FableProbeError("runtime module JSON must be a list")
        return tuple(str(item).strip() for item in loaded if str(item).strip())
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def adapter_targets(adapter_config: dict[str, Any]) -> tuple[str, ...]:
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise FableProbeError("adapter_config.json is missing target_modules")
    return tuple(str(target) for target in targets)


def compare_adapter_targets(
    targets: Iterable[str],
    runtime_modules: Iterable[str],
) -> tuple[AdapterTargetMatch, ...]:
    module_names = tuple(sorted(set(runtime_modules)))
    matches = []
    for target in targets:
        matched = tuple(
            module
            for module in module_names
            if module == target or module.endswith(f".{target}")
        )
        matches.append(
            AdapterTargetMatch(
                target=target,
                matchedRuntimeModules=matched,
                status="matched" if matched else "missing",
            )
        )
    return tuple(matches)


def load_metadata_from_dir(directory: Path) -> tuple[dict[str, Any], list[RecordedFile]]:
    data: dict[str, Any] = {}
    records: list[RecordedFile] = []
    for name in DEFAULT_RECORDED_FILES:
        path = directory / name
        if not path.exists():
            raise FableProbeError(f"missing metadata file: {path}")
        payload = path.read_bytes()
        if name.endswith(".json"):
            data[name] = json.loads(payload.decode("utf-8"))
            downloaded = True
        else:
            downloaded = True
        records.append(
            RecordedFile(
                name=name,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                source=str(path),
                downloadedPayload=downloaded,
            )
        )
    return data, records


def load_metadata_from_huggingface(
    *,
    repo: str = FABLE_REPO,
    revision: str = FABLE_REVISION,
) -> tuple[dict[str, Any], list[RecordedFile]]:
    data: dict[str, Any] = {}
    records: list[RecordedFile] = []
    for name in SMALL_METADATA_FILES:
        url = _hf_resolve_url(repo, revision, name)
        payload, headers = _http_get(url)
        data[name] = json.loads(payload.decode("utf-8"))
        records.append(
            RecordedFile(
                name=name,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                source=url,
                downloadedPayload=True,
                notes=_header_notes(headers),
            )
        )

    adapter_url = _hf_resolve_url(repo, revision, ADAPTER_FILE)
    size, etag, headers = _http_head_size(adapter_url)
    records.append(
        RecordedFile(
            name=ADAPTER_FILE,
            bytes=size,
            sha256=None,
            source=adapter_url,
            downloadedPayload=False,
            notes=tuple(
                note
                for note in (
                    "adapter payload was not downloaded by the public-safe probe",
                    f"etag={etag}" if etag else None,
                    *_header_notes(headers),
                )
                if note
            ),
        )
    )
    return data, records


def build_report(
    *,
    metadata: dict[str, Any],
    files: list[RecordedFile],
    runtime_modules: tuple[str, ...],
    runtime_source: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    adapter_config = metadata["adapter_config.json"]
    merge_info = metadata["merge_info.json"]
    config = metadata["config.json"]
    index = metadata["model.safetensors.index.json"]
    targets = adapter_targets(adapter_config)
    matches = compare_adapter_targets(targets, runtime_modules)
    missing = tuple(match.target for match in matches if match.status != "matched")
    merged_shards = tuple(
        sorted(
            {
                value
                for value in (index.get("weight_map") or {}).values()
                if isinstance(value, str) and MERGED_SHARD_RE.match(value)
            }
        )
    )
    status = (
        "adapter_targets_match_runtime_surface"
        if not missing
        else "rejected_adapter_incompatible"
    )
    return {
        "schema": FABLE_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": ISSUE_URL,
        "profileRef": PROFILE_REF,
        "status": status,
        "model": {
            "repository": FABLE_REPO,
            "revision": FABLE_REVISION,
            "baseModel": adapter_config.get("base_model_name_or_path")
            or FABLE_BASE_MODEL,
            "declaredArchitecture": config.get("architectures"),
            "modelType": config.get("model_type"),
            "quantizationConfig": config.get("quantization_config"),
            "contextWindowTokens": config.get("max_position_embeddings"),
            "experts": config.get("n_routed_experts"),
            "expertsPerToken": config.get("num_experts_per_tok"),
        },
        "adapter": {
            "peftType": adapter_config.get("peft_type"),
            "taskType": adapter_config.get("task_type"),
            "rank": adapter_config.get("r"),
            "alpha": adapter_config.get("lora_alpha"),
            "targetModules": list(targets),
            "mergeInfo": {
                "loraR": merge_info.get("lora_r"),
                "loraAlpha": merge_info.get("lora_alpha"),
                "outputDtype": merge_info.get("output_dtype"),
                "numShards": merge_info.get("num_shards"),
            },
        },
        "files": [asdict(record) for record in files],
        "mergedCheckpoint": {
            "indexTotalSizeBytes": (index.get("metadata") or {}).get("total_size"),
            "shardCount": len(merged_shards),
            "shardsFetchedByDefault": False,
            "fetchPolicy": "refuse_model_safetensors_shards_without_explicit_unsafe_flag",
        },
        "runtime": {
            "source": runtime_source,
            "moduleCount": len(runtime_modules),
            "moduleNames": list(runtime_modules),
        },
        "targetCompatibility": {
            "status": status,
            "matches": [asdict(match) for match in matches],
            "missingTargets": list(missing),
        },
        "decision": {
            "status": status,
            "canAttemptPrivateAdapterLoad": not missing,
            "canAttemptMergedCheckpointLoad": False,
            "publicAliasesAllowed": False,
            "khalaGeneralRouteAllowed": False,
            "mppPublicSaleAllowed": False,
            "nextStep": (
                "run_private_adapter_load_canary"
                if not missing
                else "stop_before_load_until_adapter_runtime_mapping_exists"
            ),
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    file_rows = [
        "| File | Bytes | Payload downloaded | Notes |",
        "| --- | ---: | --- | --- |",
    ]
    for item in report["files"]:
        notes = "; ".join(item.get("notes") or ())
        file_rows.append(
            f"| `{item['name']}` | {item.get('bytes') or 'unknown'} | "
            f"`{str(item['downloadedPayload']).lower()}` | {notes.replace('|', '/')} |"
        )

    target_rows = [
        "| Adapter target | Status | Matched runtime modules |",
        "| --- | --- | --- |",
    ]
    for item in report["targetCompatibility"]["matches"]:
        modules = ", ".join(f"`{module}`" for module in item["matchedRuntimeModules"])
        target_rows.append(f"| `{item['target']}` | `{item['status']}` | {modules or '-'} |")
    file_table = "\n".join(file_rows)
    target_table = "\n".join(target_rows)

    return f"""# DeepSeek-V4-Fable adapter compatibility evidence

Date: {report["createdAt"]}

Issue: {report["issue"]}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Private adapter load can be attempted: `{str(report["decision"]["canAttemptPrivateAdapterLoad"]).lower()}`
- Merged checkpoint load can be attempted: `{str(report["decision"]["canAttemptMergedCheckpointLoad"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["publicAliasesAllowed"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["khalaGeneralRouteAllowed"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["mppPublicSaleAllowed"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Model

- Repository: `{report["model"]["repository"]}`
- Revision: `{report["model"]["revision"]}`
- Base model: `{report["model"]["baseModel"]}`
- Architecture: `{report["model"]["declaredArchitecture"]}`
- Model type: `{report["model"]["modelType"]}`
- Context window: `{report["model"]["contextWindowTokens"]}`
- Experts: `{report["model"]["experts"]}`
- Experts per token: `{report["model"]["expertsPerToken"]}`

## Files

{file_table}

Merged checkpoint shards detected in index:
`{report["mergedCheckpoint"]["shardCount"]}`

Merged shard fetch policy:
`{report["mergedCheckpoint"]["fetchPolicy"]}`

## Adapter

- PEFT type: `{report["adapter"]["peftType"]}`
- Task type: `{report["adapter"]["taskType"]}`
- Rank: `{report["adapter"]["rank"]}`
- Alpha: `{report["adapter"]["alpha"]}`
- Merge output dtype: `{report["adapter"]["mergeInfo"]["outputDtype"]}`
- Merge shard count: `{report["adapter"]["mergeInfo"]["numShards"]}`

## Runtime target compatibility

Runtime source:
`{report["runtime"]["source"]}`

Runtime module count:
`{report["runtime"]["moduleCount"]}`

{target_table}

Missing targets:
`{", ".join(report["targetCompatibility"]["missingTargets"]) or "none"}`

## Interpretation

The Fable adapter is not admitted unless every target module can be mapped to
the exact Hydralisk DeepSeek V4 runtime surface. A missing target means the
probe fails closed before any adapter load smoke. This does not download or
serve the merged Fable checkpoint.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
"""


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-adapter-compatibility.json"
    md_path = output_dir / "deepseek-v4-fable-adapter-compatibility.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def build_load_canary_report(
    *,
    compatibility_report: dict[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    compatibility_status = str(compatibility_report.get("status"))
    can_attempt = bool(
        (compatibility_report.get("decision") or {}).get("canAttemptPrivateAdapterLoad")
    )
    if can_attempt:
        status = "ready_for_private_adapter_load_canary"
        blocker = None
        next_step = "run_private_no_public_ingress_load_smoke"
    else:
        status = "blocked_adapter_incompatible"
        blocker = {
            "code": "adapter_runtime_targets_missing",
            "message": (
                "Compatibility issue #67 did not admit the Fable adapter target "
                "modules on the current patched G4 runtime."
            ),
            "missingTargets": (
                compatibility_report.get("targetCompatibility") or {}
            ).get("missingTargets", []),
        }
        next_step = "do_not_start_load_smoke_until_adapter_mapping_exists"

    return {
        "schema": FABLE_LOAD_CANARY_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": LOAD_CANARY_ISSUE_URL,
        "dependsOn": [ISSUE_URL],
        "profileRef": PROFILE_REF,
        "status": status,
        "model": compatibility_report.get("model", {}),
        "compatibility": {
            "status": compatibility_status,
            "reportSchema": compatibility_report.get("schema"),
            "reportIssue": compatibility_report.get("issue"),
            "missingTargets": (
                compatibility_report.get("targetCompatibility") or {}
            ).get("missingTargets", []),
        },
        "loadCanary": {
            "attempted": False,
            "noPublicIngress": True,
            "mergedCheckpointServed": False,
            "adapterPayloadRequired": False,
            "reason": (
                "blocked_by_adapter_compatibility"
                if not can_attempt
                else "awaiting_live_private_canary_operator"
            ),
            "timing": {
                "ttftP50Seconds": None,
                "ttftP95Seconds": None,
                "decodeTokensPerSecondP50": None,
                "decodeTokensPerSecondP95": None,
                "endToEndTokensPerSecondP50": None,
                "endToEndTokensPerSecondP95": None,
            },
        },
        "blockers": [blocker] if blocker else [],
        "decision": {
            "status": status,
            "canAttemptPrivateAdapterLoad": can_attempt,
            "canRouteKhalaGeneralTraffic": False,
            "canExposePublicAliases": False,
            "canExposeMppPublicSale": False,
            "nextStep": next_step,
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
        },
    }


def render_load_canary_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    if blockers:
        blocker_lines = "\n".join(
            f"- `{item['code']}`: {item['message']}"
            for item in blockers
        )
    else:
        blocker_lines = "- None"
    missing = ", ".join(report["compatibility"].get("missingTargets") or ()) or "none"
    return f"""# DeepSeek-V4-Fable private load canary evidence

Date: {report["createdAt"]}

Issue: {report["issue"]}

Depends on: {", ".join(report["dependsOn"])}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Private adapter load can be attempted: `{str(report["decision"]["canAttemptPrivateAdapterLoad"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["canRouteKhalaGeneralTraffic"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["canExposePublicAliases"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["canExposeMppPublicSale"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Compatibility input

- Compatibility status: `{report["compatibility"]["status"]}`
- Compatibility issue: {report["compatibility"]["reportIssue"]}
- Missing adapter targets: `{missing}`

## Load canary

- Attempted: `{str(report["loadCanary"]["attempted"]).lower()}`
- No public ingress: `{str(report["loadCanary"]["noPublicIngress"]).lower()}`
- Merged checkpoint served: `{str(report["loadCanary"]["mergedCheckpointServed"]).lower()}`
- Adapter payload required by this gate: `{str(report["loadCanary"]["adapterPayloadRequired"]).lower()}`
- Reason: `{report["loadCanary"]["reason"]}`

Timing metrics are intentionally empty because the canary did not start.

## Blockers

{blocker_lines}

## Interpretation

The private load canary is blocked before host/model interaction because the
adapter compatibility gate did not admit the Fable LoRA targets on the current
patched G4 runtime. This is the expected safe outcome after issue #67.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
"""


def write_load_canary_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-load-canary.json"
    md_path = output_dir / "deepseek-v4-fable-load-canary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_load_canary_markdown(report))
    return json_path, md_path


def build_lab_eval_report(
    *,
    load_canary_report: dict[str, Any],
    policy_status: str = "policy_harness_implemented_fail_closed",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    load_decision = load_canary_report.get("decision") or {}
    load_canary = load_canary_report.get("loadCanary") or {}
    can_attempt_load = bool(load_decision.get("canAttemptPrivateAdapterLoad"))
    load_attempted = bool(load_canary.get("attempted"))
    blockers = list(load_canary_report.get("blockers") or ())

    if can_attempt_load and load_attempted:
        status = "rejected_quality_or_safety_gate_failed"
        reason = "lab_eval_metrics_not_recorded"
        next_step = "run_public_safe_authorized_lab_eval_with_metric_capture"
        blockers.append(
            {
                "code": "lab_eval_metrics_missing",
                "message": (
                    "The private load canary reported an attempted load, but no "
                    "public-safe lab eval metrics were supplied to this gate."
                ),
            }
        )
    else:
        status = "rejected_runtime_unstable"
        reason = "blocked_before_eval_by_private_load_canary"
        next_step = "map_or_retarget_fable_lora_modules_before_any_lab_eval"

    return {
        "schema": FABLE_LAB_EVAL_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": LAB_EVAL_ISSUE_URL,
        "dependsOn": [LOAD_CANARY_ISSUE_URL, POLICY_ISSUE_URL],
        "profileRef": PROFILE_REF,
        "status": status,
        "model": load_canary_report.get("model", {}),
        "prerequisites": {
            "loadCanaryStatus": load_canary_report.get("status"),
            "loadCanaryAttempted": load_attempted,
            "canAttemptPrivateAdapterLoad": can_attempt_load,
            "policyHarnessStatus": policy_status,
            "authorizedSecurityUnscopedRequestsBlocked": (
                policy_status == "policy_harness_implemented_fail_closed"
            ),
        },
        "labEval": {
            "attempted": can_attempt_load and load_attempted,
            "authorizedSandboxTasksOnly": True,
            "productionTargetsUsed": False,
            "thirdPartyTargetsUsed": False,
            "rawPromptsCommitted": False,
            "rawOutputsCommitted": False,
            "reason": reason,
            "metrics": {
                "taskCategories": [],
                "verifierResults": [],
                "turnCountSummary": None,
                "toolCallCountSummary": None,
                "timeoutOrErrorClasses": [],
                "ttftSecondsSummary": None,
                "decodeTokensPerSecondSummary": None,
                "baseDeepSeekV4FlashComparison": None,
            },
        },
        "blockers": blockers,
        "decision": {
            "status": status,
            "admittedPrivateAuthorizedSecurityLabCanary": False,
            "canRouteKhalaGeneralTraffic": False,
            "canExposePublicAliases": False,
            "canExposeMppPublicSale": False,
            "nextStep": next_step,
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
            "containsTargetDetails": False,
        },
    }


def render_lab_eval_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    if blockers:
        blocker_lines = "\n".join(
            f"- `{item['code']}`: {item['message']}"
            for item in blockers
        )
    else:
        blocker_lines = "- None"
    metrics = report["labEval"]["metrics"]
    turn_count = _metric_value(metrics["turnCountSummary"])
    tool_call_count = _metric_value(metrics["toolCallCountSummary"])
    ttft = _metric_value(metrics["ttftSecondsSummary"])
    decode_tps = _metric_value(metrics["decodeTokensPerSecondSummary"])
    base_comparison = _metric_value(metrics["baseDeepSeekV4FlashComparison"])
    return f"""# DeepSeek-V4-Fable lab eval decision

Date: {report["createdAt"]}

Issue: {report["issue"]}

Depends on: {", ".join(report["dependsOn"])}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Private authorized-security lab canary admitted: `{str(report["decision"]["admittedPrivateAuthorizedSecurityLabCanary"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["canRouteKhalaGeneralTraffic"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["canExposePublicAliases"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["canExposeMppPublicSale"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Prerequisites

- Load canary status: `{report["prerequisites"]["loadCanaryStatus"]}`
- Load canary attempted: `{str(report["prerequisites"]["loadCanaryAttempted"]).lower()}`
- Private adapter load can be attempted: `{str(report["prerequisites"]["canAttemptPrivateAdapterLoad"]).lower()}`
- Authorized-security policy harness: `{report["prerequisites"]["policyHarnessStatus"]}`
- Unscoped requests blocked by policy harness: `{str(report["prerequisites"]["authorizedSecurityUnscopedRequestsBlocked"]).lower()}`

## Lab eval

- Attempted: `{str(report["labEval"]["attempted"]).lower()}`
- Authorized sandbox tasks only: `{str(report["labEval"]["authorizedSandboxTasksOnly"]).lower()}`
- Production targets used: `{str(report["labEval"]["productionTargetsUsed"]).lower()}`
- Third-party targets used: `{str(report["labEval"]["thirdPartyTargetsUsed"]).lower()}`
- Raw prompts committed: `{str(report["labEval"]["rawPromptsCommitted"]).lower()}`
- Raw outputs committed: `{str(report["labEval"]["rawOutputsCommitted"]).lower()}`
- Reason: `{report["labEval"]["reason"]}`

No lab eval traffic was run because the model never reached an admitted private
load canary state.

## Public-safe metrics

- Task categories: `{", ".join(metrics["taskCategories"]) or "none"}`
- Verifier results: `{", ".join(metrics["verifierResults"]) or "none"}`
- Turn-count summary: `{turn_count}`
- Tool-call-count summary: `{tool_call_count}`
- Timeout/error classes: `{", ".join(metrics["timeoutOrErrorClasses"]) or "none"}`
- TTFT summary: `{ttft}`
- Decode throughput summary: `{decode_tps}`
- Base DeepSeek V4 Flash comparison: `{base_comparison}`

## Blockers

{blocker_lines}

## Interpretation

Issue #70 cannot honestly run or admit a Fable lab eval while issue #68 remains
blocked. The final decision is therefore `rejected_runtime_unstable`: the
adapter-backed runtime path is not stable/admitted enough to evaluate. Fable
remains disallowed for general Khala routing, public aliases, and MPP sale.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
"""


def _metric_value(value: Any) -> str:
    return "none" if value is None else str(value)


def write_lab_eval_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-lab-eval-decision.json"
    md_path = output_dir / "deepseek-v4-fable-lab-eval-decision.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_lab_eval_markdown(report))
    return json_path, md_path


def build_retarget_plan_report(
    *,
    compatibility_report: dict[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    target_matches = {
        item["target"]: tuple(item.get("matchedRuntimeModules") or ())
        for item in (compatibility_report.get("targetCompatibility") or {}).get(
            "matches",
            [],
        )
    }
    runtime_modules = tuple(
        str(item)
        for item in (compatibility_report.get("runtime") or {}).get("moduleNames", [])
    )
    targets = tuple((compatibility_report.get("adapter") or {}).get("targetModules", []))
    classifications = [
        _classify_retarget_target(
            str(target),
            direct_matches=target_matches.get(str(target), ()),
            runtime_modules=runtime_modules,
        )
        for target in targets
    ]
    source_inventory_required = [
        item["target"]
        for item in classifications
        if item["status"] == "source_inventory_required"
    ]
    packed_required = [
        item["target"]
        for item in classifications
        if item["status"] == "packed_transform_required"
    ]
    unknown = [
        item["target"]
        for item in classifications
        if item["status"] == "unmapped_unknown"
    ]

    if source_inventory_required or unknown:
        status = "blocked_source_inventory_required"
        next_step = "prove_attention_output_projection_owner_then_build_packed_lora_transform"
        primary_path = "packed_runtime_retarget_after_source_inventory"
    elif packed_required:
        status = "blocked_packed_transform_implementation_missing"
        next_step = "implement_packed_lora_tensor_transform_smoke"
        primary_path = "packed_runtime_retarget"
    else:
        status = "ready_for_private_adapter_load_canary"
        next_step = "rerun_private_adapter_load_canary"
        primary_path = "direct_peft_adapter_load"

    return {
        "schema": FABLE_RETARGET_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": RETARGET_ISSUE_URL,
        "dependsOn": [ISSUE_URL, LOAD_CANARY_ISSUE_URL, LAB_EVAL_ISSUE_URL],
        "profileRef": PROFILE_REF,
        "status": status,
        "model": compatibility_report.get("model", {}),
        "compatibility": {
            "status": compatibility_report.get("status"),
            "runtimeSource": (compatibility_report.get("runtime") or {}).get("source"),
            "runtimeModuleCount": len(runtime_modules),
            "missingTargets": (
                compatibility_report.get("targetCompatibility") or {}
            ).get("missingTargets", []),
        },
        "retargetPlan": {
            "primaryPath": primary_path,
            "fallbackPath": "canonical_base_runtime_feasibility_probe",
            "targetClassifications": classifications,
            "packedTransformRequiredTargets": packed_required,
            "sourceInventoryRequiredTargets": source_inventory_required,
            "unknownTargets": unknown,
            "weightsRead": False,
            "adapterPayloadDownloaded": False,
            "transformImplemented": False,
        },
        "decision": {
            "status": status,
            "canAttemptPackedRetargetSmoke": status == "ready_for_private_adapter_load_canary",
            "canAttemptCanonicalRuntimeProbe": True,
            "canRouteKhalaGeneralTraffic": False,
            "canExposePublicAliases": False,
            "canExposeMppPublicSale": False,
            "nextStep": next_step,
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
            "containsTargetDetails": False,
        },
    }


def _classify_retarget_target(
    target: str,
    *,
    direct_matches: tuple[str, ...],
    runtime_modules: tuple[str, ...],
) -> dict[str, Any]:
    if direct_matches:
        return {
            "target": target,
            "status": "direct_attachable",
            "runtimeModules": list(direct_matches),
            "packedFamily": None,
            "requiredWork": "rerun_private_adapter_load_canary",
            "implementationReady": True,
            "notes": [
                "Exact suffix match exists in the inspected runtime module inventory."
            ],
        }

    if target in {"q_proj", "k_proj", "v_proj"}:
        modules = _runtime_modules_ending(runtime_modules, ".attn.fused_wqa_wkv")
        return {
            "target": target,
            "status": "packed_transform_required",
            "runtimeModules": list(modules),
            "packedFamily": "attention_fused_wqa_wkv",
            "requiredWork": (
                "derive DeepSeek-V4 MLA projection slice ownership, then repack "
                "the LoRA delta into the fused attention input projection"
            ),
            "implementationReady": False,
            "notes": [
                "The current NVIDIA runtime packs attention projection ownership into fused_wqa_wkv.",
                "Vanilla PEFT cannot attach this target by module name.",
            ],
        }

    if target in {"gate_proj", "up_proj"}:
        modules = _runtime_modules_ending(runtime_modules, ".mlp.gate_up_proj")
        modules = (*modules, *_runtime_modules_ending(runtime_modules, ".mlp.shared_experts.gate_up_proj"))
        return {
            "target": target,
            "status": "packed_transform_required",
            "runtimeModules": list(dict.fromkeys(modules)),
            "packedFamily": "swiglu_gate_up_proj",
            "requiredWork": (
                "prove gate/up ordering and repack paired SwiGLU LoRA deltas "
                "into fused gate_up_proj weights"
            ),
            "implementationReady": False,
            "notes": [
                "The current NVIDIA runtime fuses gate_proj and up_proj as gate_up_proj.",
                "A transform must preserve SwiGLU ordering and expert/shared-expert ownership.",
            ],
        }

    if target == "o_proj":
        return {
            "target": target,
            "status": "source_inventory_required",
            "runtimeModules": [],
            "packedFamily": "attention_output_o_proj",
            "requiredWork": (
                "inspect the runtime source for attention output projection ownership; "
                "local evidence shows an o_proj provider/kernel path but no "
                "adapter-addressable module inventory entry"
            ),
            "implementationReady": False,
            "notes": [
                "The G4 evidence contains vllm.models.deepseek_v4.nvidia.ops.o_proj traces.",
                "The model module inventory used by the adapter probe did not expose an o_proj module.",
            ],
        }

    return {
        "target": target,
        "status": "unmapped_unknown",
        "runtimeModules": [],
        "packedFamily": None,
        "requiredWork": "add an explicit target mapping before any adapter load",
        "implementationReady": False,
        "notes": ["No direct or known packed-runtime mapping is encoded for this target."],
    }


def _runtime_modules_ending(
    runtime_modules: tuple[str, ...],
    suffix: str,
) -> tuple[str, ...]:
    return tuple(module for module in runtime_modules if module.endswith(suffix))


def render_retarget_plan_markdown(report: dict[str, Any]) -> str:
    target_rows = [
        "| Adapter target | Status | Packed family | Runtime modules | Required work |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["retargetPlan"]["targetClassifications"]:
        modules = ", ".join(f"`{module}`" for module in item["runtimeModules"]) or "-"
        target_rows.append(
            "| `{target}` | `{status}` | `{family}` | {modules} | {work} |".format(
                target=item["target"],
                status=item["status"],
                family=item["packedFamily"] or "-",
                modules=modules,
                work=item["requiredWork"].replace("|", "/"),
            )
        )
    target_table = "\n".join(target_rows)
    packed = ", ".join(report["retargetPlan"]["packedTransformRequiredTargets"]) or "none"
    source_required = (
        ", ".join(report["retargetPlan"]["sourceInventoryRequiredTargets"]) or "none"
    )
    unknown = ", ".join(report["retargetPlan"]["unknownTargets"]) or "none"

    return f"""# DeepSeek-V4-Fable packed-runtime retargeting plan

Date: {report["createdAt"]}

Issue: {report["issue"]}

Depends on: {", ".join(report["dependsOn"])}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Packed retarget smoke can be attempted: `{str(report["decision"]["canAttemptPackedRetargetSmoke"]).lower()}`
- Canonical runtime probe can be attempted: `{str(report["decision"]["canAttemptCanonicalRuntimeProbe"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["canRouteKhalaGeneralTraffic"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["canExposePublicAliases"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["canExposeMppPublicSale"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Compatibility input

- Compatibility status: `{report["compatibility"]["status"]}`
- Runtime source: `{report["compatibility"]["runtimeSource"]}`
- Runtime module count: `{report["compatibility"]["runtimeModuleCount"]}`
- Missing targets from direct match probe: `{", ".join(report["compatibility"]["missingTargets"]) or "none"}`

## Target retargeting plan

{target_table}

Packed transform required targets:
`{packed}`

Source inventory required targets:
`{source_required}`

Unknown targets:
`{unknown}`

## Interpretation

The path to getting Fable working on the current Google G4 lane is a packed
LoRA retarget, not a vanilla PEFT adapter load. The current runtime can only
claim `down_proj` as directly attachable. Attention `q_proj`, `k_proj`, and
`v_proj` need an architecture-aware transform into `fused_wqa_wkv`; MLP
`gate_proj` and `up_proj` need a paired SwiGLU transform into `gate_up_proj`.
`o_proj` remains blocked until the attention output projection owner is proven
from runtime source or a live module inventory, because current evidence shows
an `o_proj` kernel/provider path but no adapter-addressable module entry.

If `o_proj` ownership cannot be proven quickly, the fallback path is a
canonical DeepSeek-V4-Flash base runtime feasibility probe that exposes the
Fable PEFT target names and reruns admission from scratch.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
"""


def write_retarget_plan_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-retarget-plan.json"
    md_path = output_dir / "deepseek-v4-fable-retarget-plan.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_retarget_plan_markdown(report))
    return json_path, md_path


def build_o_proj_ownership_report(
    *,
    source_inventory: dict[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    modules = tuple(source_inventory.get("modules") or ())
    model_modules = tuple(
        module for module in modules if str(module.get("module", "")).endswith(".model")
    )
    attention_modules = tuple(
        module
        for module in modules
        if str(module.get("module", "")).endswith((".flashmla", ".flashinfer_sparse"))
    )
    kernel_modules = tuple(
        module for module in modules if str(module.get("module", "")).endswith(".ops.o_proj")
    )
    adapter_module_exposed = any(
        "o_proj" in set(_strings_from_module(module, "attrs"))
        or "o_proj" in set(_strings_from_module(module, "names"))
        for module in model_modules
    )
    attention_o_proj_methods = sorted(
        {
            fn
            for module in attention_modules
            for fn in _strings_from_module(module, "functions")
            if fn == "_o_proj"
        }
    )
    kernel_calls = sorted(
        {
            call
            for module in attention_modules
            for call in _strings_from_module(module, "calls")
            if call == "deep_gemm_fp8_o_proj"
        }
    )
    kernel_functions = sorted(
        {
            fn
            for module in kernel_modules
            for fn in _strings_from_module(module, "functions")
            if fn == "deep_gemm_fp8_o_proj"
        }
    )
    kernel_provider_owned = bool(attention_o_proj_methods and kernel_calls and kernel_functions)

    if adapter_module_exposed:
        status = "adapter_addressable_o_proj_module"
        classification = "adapter_addressable_module"
        next_step = "rerun_private_adapter_load_canary_with_o_proj_direct_match"
    elif kernel_provider_owned:
        status = "o_proj_owner_proven_kernel_provider"
        classification = "kernel_provider_owned_projection"
        next_step = "implement_offline_packed_lora_delta_transform_smoke"
    else:
        status = "blocked_o_proj_owner_unknown"
        classification = "unknown_projection_owner"
        next_step = "inspect_live_runtime_source_for_o_proj_owner"

    return {
        "schema": FABLE_OPROJ_OWNERSHIP_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": OPROJ_ISSUE_URL,
        "dependsOn": [RETARGET_ISSUE_URL],
        "profileRef": PROFILE_REF,
        "status": status,
        "sourceInventory": {
            "image": source_inventory.get("image"),
            "inspection": source_inventory.get("inspection"),
            "moduleCount": len(modules),
            "modules": [
                {
                    "module": module.get("module"),
                    "file": module.get("file"),
                    "classes": _strings_from_module(module, "classes"),
                    "functions": _strings_from_module(module, "functions"),
                    "names": _strings_from_module(module, "names"),
                    "attrs": _strings_from_module(module, "attrs"),
                    "calls": _strings_from_module(module, "calls"),
                }
                for module in modules
            ],
        },
        "ownership": {
            "classification": classification,
            "adapterAddressableModule": adapter_module_exposed,
            "kernelProviderOwned": kernel_provider_owned,
            "attentionOProjMethods": attention_o_proj_methods,
            "attentionKernelCalls": kernel_calls,
            "kernelFunctions": kernel_functions,
            "modelModuleOProjAttrs": sorted(
                {
                    attr
                    for module in model_modules
                    for attr in _strings_from_module(module, "attrs")
                    if "o_proj" in attr
                }
            ),
        },
        "decision": {
            "status": status,
            "canUseVanillaPeftOProj": adapter_module_exposed,
            "canProceedToPackedLoraTransformSmoke": kernel_provider_owned
            and not adapter_module_exposed,
            "canRouteKhalaGeneralTraffic": False,
            "canExposePublicAliases": False,
            "canExposeMppPublicSale": False,
            "nextStep": next_step,
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
            "containsTargetDetails": False,
            "containsFullThirdPartySource": False,
        },
    }


def _strings_from_module(module: dict[str, Any], key: str) -> list[str]:
    value = module.get(key) or []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def render_o_proj_ownership_markdown(report: dict[str, Any]) -> str:
    module_rows = [
        "| Module | Relevant functions/classes | Relevant names/attrs/calls |",
        "| --- | --- | --- |",
    ]
    for module in report["sourceInventory"]["modules"]:
        functions = [
            item
            for item in (*module["classes"], *module["functions"])
            if "o_proj" in item
            or "Attention" in item
            or item in {"DeepseekV4Model", "DeepseekV4ForCausalLM"}
        ]
        symbols = [
            item
            for item in (*module["names"], *module["attrs"], *module["calls"])
            if "o_proj" in item or "fused_wqa_wkv" in item or "gate_up_proj" in item
        ]
        module_rows.append(
            "| `{module}` | {functions} | {symbols} |".format(
                module=module["module"],
                functions=", ".join(f"`{item}`" for item in functions) or "-",
                symbols=", ".join(f"`{item}`" for item in symbols) or "-",
            )
        )
    module_table = "\n".join(module_rows)

    return f"""# DeepSeek-V4-Fable o_proj ownership evidence

Date: {report["createdAt"]}

Issue: {report["issue"]}

Depends on: {", ".join(report["dependsOn"])}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Vanilla PEFT `o_proj` can be used: `{str(report["decision"]["canUseVanillaPeftOProj"]).lower()}`
- Packed-LoRA transform smoke can proceed: `{str(report["decision"]["canProceedToPackedLoraTransformSmoke"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["canRouteKhalaGeneralTraffic"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["canExposePublicAliases"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["canExposeMppPublicSale"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Source inventory

- Image: `{report["sourceInventory"]["image"]}`
- Inspection: `{report["sourceInventory"]["inspection"]}`
- Module count: `{report["sourceInventory"]["moduleCount"]}`

{module_table}

## Ownership result

- Classification: `{report["ownership"]["classification"]}`
- Adapter-addressable module: `{str(report["ownership"]["adapterAddressableModule"]).lower()}`
- Kernel/provider owned: `{str(report["ownership"]["kernelProviderOwned"]).lower()}`
- Attention `_o_proj` methods: `{", ".join(report["ownership"]["attentionOProjMethods"]) or "none"}`
- Attention kernel calls: `{", ".join(report["ownership"]["attentionKernelCalls"]) or "none"}`
- Kernel functions: `{", ".join(report["ownership"]["kernelFunctions"]) or "none"}`
- Model module `o_proj` attrs: `{", ".join(report["ownership"]["modelModuleOProjAttrs"]) or "none"}`

## Interpretation

Fable's `o_proj` target is not vanilla-PEFT-addressable on the current packed
NVIDIA runtime. The attention output projection is owned by backend attention
classes through `_o_proj` methods that call the `deep_gemm_fp8_o_proj` provider
function in `vllm.models.deepseek_v4.nvidia.ops.o_proj`.

That unblocks the retargeting plan's source-inventory question, but it does
not admit Fable for serving. The next implementation step is an offline
packed-LoRA delta transform smoke that proves we can map Fable adapter tensors
into `fused_wqa_wkv`, `gate_up_proj`, and the kernel/provider-owned `o_proj`
path without running public traffic.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
- Contains full third-party source: false
"""


def write_o_proj_ownership_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-o-proj-ownership.json"
    md_path = output_dir / "deepseek-v4-fable-o-proj-ownership.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_o_proj_ownership_markdown(report))
    return json_path, md_path


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise FableProbeError(f"{path} is too small to be a safetensors file")
        (header_length,) = struct.unpack("<Q", raw_length)
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise FableProbeError(f"{path} has a truncated safetensors header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FableProbeError(f"{path} has an invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise FableProbeError(f"{path} safetensors header is not an object")
    return header


def safetensor_manifest_from_header(header: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for key, value in sorted(header.items()):
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise FableProbeError(f"safetensors entry {key} is not an object")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
            raise FableProbeError(f"safetensors entry {key} has invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise FableProbeError(f"safetensors entry {key} has invalid data offsets")
        manifest.append(
            {
                "key": key,
                "dtype": str(value.get("dtype")),
                "shape": shape,
                "dataOffsets": offsets,
                "bytes": offsets[1] - offsets[0],
            }
        )
    return manifest


def collect_lora_pairs(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for tensor in manifest:
        key = tensor["key"]
        match = LORA_KEY_RE.match(key)
        if not match:
            continue
        module_path = match.group("module")
        side = match.group("side")
        target = _target_from_module_path(module_path)
        if target is None:
            continue
        layer_match = LAYER_RE.search(module_path)
        module = pairs.setdefault(
            module_path,
            {
                "module": module_path,
                "target": target,
                "layer": int(layer_match.group(1)) if layer_match else None,
                "context": _lora_context(module_path),
                "loraA": None,
                "loraB": None,
            },
        )
        module[f"lora{side}"] = {
            "key": key,
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "bytes": tensor["bytes"],
        }
    return pairs


def build_transform_smoke_report(
    *,
    adapter_path: Path,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    header = read_safetensors_header(adapter_path)
    manifest = safetensor_manifest_from_header(header)
    pairs = collect_lora_pairs(manifest)
    target_summaries = _summarize_lora_targets(pairs)
    family_summaries = _summarize_packed_families(target_summaries)
    blockers = [
        blocker
        for family in family_summaries.values()
        for blocker in family["blockers"]
    ]
    if blockers:
        status = (
            "blocked_adapter_config_payload_mismatch"
            if _has_config_payload_mismatch(blockers)
            else "blocked_adapter_shape_manifest_incomplete"
        )
        next_step = (
            "map_actual_adapter_payload_contexts_or_pivot_canonical_runtime"
            if status == "blocked_adapter_config_payload_mismatch"
            else "fix_or_reinspect_adapter_shape_manifest"
        )
    else:
        status = "shape_manifest_ready_for_transform_writer"
        next_step = "implement_actual_packed_lora_delta_writer"

    return {
        "schema": FABLE_TRANSFORM_SMOKE_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "issue": TRANSFORM_SMOKE_ISSUE_URL,
        "dependsOn": [OPROJ_ISSUE_URL],
        "profileRef": PROFILE_REF,
        "status": status,
        "adapter": {
            "path": str(adapter_path),
            "fileBytes": adapter_path.stat().st_size,
            "tensorCount": len(manifest),
            "loraModuleCount": len(pairs),
            "tensorValuesRead": False,
            "headerOnly": True,
        },
        "targets": target_summaries,
        "packedFamilies": family_summaries,
        "blockers": blockers,
        "decision": {
            "status": status,
            "canWritePackedDeltaNow": False,
            "canImplementPackedDeltaWriter": not blockers,
            "canRouteKhalaGeneralTraffic": False,
            "canExposePublicAliases": False,
            "canExposeMppPublicSale": False,
            "nextStep": next_step,
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsTensorValues": False,
            "containsHiddenReasoning": False,
            "containsExploitPayloads": False,
            "containsTargetDetails": False,
        },
    }


def _target_from_module_path(module_path: str) -> str | None:
    segments = module_path.split(".")
    for target in FABLE_LORA_TARGETS:
        if target in segments:
            return target
    return None


def _summarize_lora_targets(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for target in FABLE_LORA_TARGETS:
        modules = [
            module
            for module in sorted(pairs.values(), key=lambda item: item["module"])
            if module["target"] == target
        ]
        complete = [module for module in modules if module["loraA"] and module["loraB"]]
        bad_shape = [
            module["module"]
            for module in complete
            if not _lora_shapes_compatible(module["loraA"]["shape"], module["loraB"]["shape"])
        ]
        context_counts = {
            context: sum(1 for module in modules if module["context"] == context)
            for context in sorted({module["context"] for module in modules})
        }
        unsupported_contexts = [
            module["module"]
            for module in modules
            if module["context"] not in EXPECTED_LORA_CONTEXTS[target]
        ]
        ranks = sorted(
            {
                module["loraA"]["shape"][0]
                for module in complete
                if len(module["loraA"]["shape"]) == 2
            }
        )
        summaries[target] = {
            "moduleCount": len(modules),
            "completePairCount": len(complete),
            "missingPairCount": len(modules) - len(complete),
            "ranks": ranks,
            "contextCounts": context_counts,
            "layers": sorted(
                {
                    module["layer"]
                    for module in modules
                    if module["layer"] is not None
                }
            ),
            "shapeCompatible": not bad_shape,
            "badShapeModules": bad_shape,
            "unsupportedContextModules": unsupported_contexts,
            "sampleModules": [module["module"] for module in modules[:3]],
        }
    return summaries


def _lora_shapes_compatible(shape_a: list[int], shape_b: list[int]) -> bool:
    return (
        len(shape_a) == 2
        and len(shape_b) == 2
        and shape_a[0] > 0
        and shape_b[1] > 0
        and shape_a[0] == shape_b[1]
    )


def _summarize_packed_families(target_summaries: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family, targets in FABLE_PACKED_FAMILIES.items():
        blockers = []
        layers_by_target = {
            target: set(target_summaries[target]["layers"])
            for target in targets
        }
        for target in targets:
            summary = target_summaries[target]
            if summary["completePairCount"] == 0:
                blockers.append(
                    {
                        "code": "missing_lora_pairs",
                        "target": target,
                        "message": f"No complete LoRA A/B pairs found for {target}.",
                    }
                )
            if summary["missingPairCount"]:
                blockers.append(
                    {
                        "code": "incomplete_lora_pairs",
                        "target": target,
                        "message": f"Incomplete LoRA A/B pairs found for {target}.",
                    }
                )
            if not summary["shapeCompatible"]:
                blockers.append(
                    {
                        "code": "incompatible_lora_shapes",
                        "target": target,
                        "message": f"LoRA A/B rank dimensions do not match for {target}.",
                    }
                )
            if summary["unsupportedContextModules"]:
                blockers.append(
                    {
                        "code": "unsupported_lora_context",
                        "target": target,
                        "message": (
                            f"{target} appears in unsupported adapter contexts for "
                            "the current packed-family transform smoke."
                        ),
                    }
                )
        if len(targets) > 1:
            layer_sets = [layers for layers in layers_by_target.values() if layers]
            if layer_sets and any(layers != layer_sets[0] for layers in layer_sets):
                blockers.append(
                    {
                        "code": "target_layer_sets_differ",
                        "target": ",".join(targets),
                        "message": (
                            "Packed-family targets are present on different layer sets."
                        ),
                    }
                )
        families[family] = {
            "targets": list(targets),
            "complete": not blockers,
            "layers": sorted(set().union(*layers_by_target.values()))
            if layers_by_target
            else [],
            "blockers": blockers,
        }
    return families


def _has_config_payload_mismatch(blockers: list[dict[str, str]]) -> bool:
    mismatch_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    return any(
        blocker["code"] == "missing_lora_pairs" and blocker["target"] in mismatch_targets
        for blocker in blockers
    ) or any(blocker["code"] == "unsupported_lora_context" for blocker in blockers)


def _lora_context(module_path: str) -> str:
    if ".mlp." in module_path:
        return "mlp"
    if ".self_attn.compressor.indexer." in module_path:
        return "attention_compressor_indexer"
    if ".self_attn.compressor." in module_path:
        return "attention_compressor"
    if ".self_attn." in module_path:
        return "attention"
    return "unknown"


def render_transform_smoke_markdown(report: dict[str, Any]) -> str:
    if report["status"] == "blocked_adapter_config_payload_mismatch":
        interpretation = """This smoke inspects only safetensors header metadata: tensor keys, dtypes,
shapes, and byte ranges. It does not record tensor values and it does not write
transformed model artifacts.

The adapter payload does not match the module-family assumptions needed by the
current packed-LoRA transform plan. The next step is to map the actual adapter
payload contexts against the runtime, especially attention-compressor and
shared-expert MLP targets, or pivot to a canonical runtime that can load the
adapter as published."""
    elif report["status"] == "shape_manifest_ready_for_transform_writer":
        interpretation = """This smoke inspects only safetensors header metadata: tensor keys, dtypes,
shapes, and byte ranges. It does not record tensor values and it does not write
transformed model artifacts.

The shape manifest is complete enough to implement the actual packed delta
writer for `fused_wqa_wkv`, `gate_up_proj`, and the kernel/provider-owned
`o_proj` path, then rerun a private no-public-ingress load canary."""
    else:
        interpretation = """This smoke inspects only safetensors header metadata: tensor keys, dtypes,
shapes, and byte ranges. It does not record tensor values and it does not write
transformed model artifacts.

The manifest is incomplete or internally inconsistent. Reinspect the adapter
payload and fix the shape manifest before implementing a packed delta writer."""
    target_rows = [
        "| Target | Modules | Complete pairs | Contexts | Ranks | Layers | Shape compatible |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for target, summary in report["targets"].items():
        contexts = ", ".join(
            f"{context}:{count}"
            for context, count in sorted(summary["contextCounts"].items())
        )
        target_rows.append(
            "| `{target}` | {modules} | {pairs} | `{contexts}` | `{ranks}` | `{layers}` | `{shape}` |".format(
                target=target,
                modules=summary["moduleCount"],
                pairs=summary["completePairCount"],
                contexts=contexts or "none",
                ranks=", ".join(str(rank) for rank in summary["ranks"]) or "none",
                layers=", ".join(str(layer) for layer in summary["layers"]) or "none",
                shape=str(summary["shapeCompatible"]).lower(),
            )
        )
    family_rows = [
        "| Family | Targets | Complete | Layers | Blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family, summary in report["packedFamilies"].items():
        blocker_codes = ", ".join(blocker["code"] for blocker in summary["blockers"])
        family_rows.append(
            "| `{family}` | `{targets}` | `{complete}` | `{layers}` | {blockers} |".format(
                family=family,
                targets=", ".join(summary["targets"]),
                complete=str(summary["complete"]).lower(),
                layers=", ".join(str(layer) for layer in summary["layers"]) or "none",
                blockers=blocker_codes or "-",
            )
        )
    target_table = "\n".join(target_rows)
    family_table = "\n".join(family_rows)
    blocker_lines = "\n".join(
        f"- `{item['code']}` on `{item['target']}`: {item['message']}"
        for item in report["blockers"]
    ) or "- None"

    return f"""# DeepSeek-V4-Fable packed-LoRA transform smoke

Date: {report["createdAt"]}

Issue: {report["issue"]}

Depends on: {", ".join(report["dependsOn"])}

Profile: `{report["profileRef"]}`

Status: `{report["status"]}`

## Decision

- Packed delta can be written now: `{str(report["decision"]["canWritePackedDeltaNow"]).lower()}`
- Packed delta writer can be implemented from this manifest: `{str(report["decision"]["canImplementPackedDeltaWriter"]).lower()}`
- Khala general route allowed: `{str(report["decision"]["canRouteKhalaGeneralTraffic"]).lower()}`
- Public aliases allowed: `{str(report["decision"]["canExposePublicAliases"]).lower()}`
- MPP public sale allowed: `{str(report["decision"]["canExposeMppPublicSale"]).lower()}`
- Next step: `{report["decision"]["nextStep"]}`

## Adapter inspection

- Adapter path: `{report["adapter"]["path"]}`
- Adapter file bytes: `{report["adapter"]["fileBytes"]}`
- Tensor count: `{report["adapter"]["tensorCount"]}`
- LoRA module count: `{report["adapter"]["loraModuleCount"]}`
- Header-only inspection: `{str(report["adapter"]["headerOnly"]).lower()}`
- Tensor values read: `{str(report["adapter"]["tensorValuesRead"]).lower()}`

## Target shape summary

{target_table}

## Packed family summary

{family_table}

## Blockers

{blocker_lines}

## Interpretation

{interpretation}

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains tensor values: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
"""


def write_transform_smoke_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-fable-transform-smoke.json"
    md_path = output_dir / "deepseek-v4-fable-transform-smoke.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_transform_smoke_markdown(report))
    return json_path, md_path


def _hf_resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _http_get(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "hydralisk-admission-probe"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read(), dict(response.headers.items())
    except HTTPError as exc:
        raise FableProbeError(f"failed to fetch {url}: HTTP {exc.code}") from exc


def _http_head_size(url: str) -> tuple[int | None, str | None, dict[str, str]]:
    request = Request(
        url,
        method="HEAD",
        headers={"User-Agent": "hydralisk-admission-probe"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            headers = dict(response.headers.items())
    except HTTPError as exc:
        raise FableProbeError(f"failed to inspect {url}: HTTP {exc.code}") from exc
    size_value = headers.get("x-linked-size") or headers.get("content-length")
    size = int(size_value) if size_value and size_value.isdigit() else None
    return size, headers.get("x-linked-etag") or headers.get("etag"), headers


def _header_notes(headers: dict[str, str]) -> tuple[str, ...]:
    notes: list[str] = []
    commit = headers.get("x-repo-commit")
    if commit:
        notes.append(f"x-repo-commit={commit}")
    linked_size = headers.get("x-linked-size")
    if linked_size:
        notes.append(f"x-linked-size={linked_size}")
    return tuple(notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe DeepSeek-V4-Fable adapter compatibility without merged weights."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Directory containing cached Fable metadata files.",
    )
    parser.add_argument(
        "--runtime-modules-file",
        type=Path,
        required=True,
        help="Text or JSON list of runtime module names from the patched serving image.",
    )
    parser.add_argument(
        "--runtime-source-label",
        help="Human-readable source label for the runtime module inventory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-adapter-compatibility"),
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=(),
        help="Additional Fable file to request. Merged shards require the unsafe flag.",
    )
    parser.add_argument(
        "--allow-merged-shard-files",
        action="store_true",
        help="Unsafe: allow explicit model-*.safetensors shard requests.",
    )
    args = parser.parse_args(argv)

    validate_requested_files(
        (*DEFAULT_RECORDED_FILES, *args.extra_file),
        allow_merged_shards=args.allow_merged_shard_files,
    )
    if args.metadata_dir:
        metadata, files = load_metadata_from_dir(args.metadata_dir)
    else:
        metadata, files = load_metadata_from_huggingface()
    runtime_modules = load_runtime_module_names(args.runtime_modules_file)
    report = build_report(
        metadata=metadata,
        files=files,
        runtime_modules=runtime_modules,
        runtime_source=args.runtime_source_label or str(args.runtime_modules_file),
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["canAttemptPrivateAdapterLoad"] else 2


def load_canary_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a public-safe DeepSeek-V4-Fable private load canary decision."
    )
    parser.add_argument(
        "--compatibility-report",
        type=Path,
        required=True,
        help="JSON report from hydralisk-deepseek-v4-fable-adapter-probe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-load-canary"),
    )
    args = parser.parse_args(argv)

    compatibility_report = json.loads(args.compatibility_report.read_text())
    report = build_load_canary_report(compatibility_report=compatibility_report)
    json_path, md_path = write_load_canary_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["canAttemptPrivateAdapterLoad"] else 2


def lab_eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a public-safe DeepSeek-V4-Fable lab eval admit/reject decision."
    )
    parser.add_argument(
        "--load-canary-report",
        type=Path,
        required=True,
        help="JSON report from hydralisk-deepseek-v4-fable-load-canary.",
    )
    parser.add_argument(
        "--policy-status",
        default="policy_harness_implemented_fail_closed",
        help="Public-safe status from the authorized-security policy harness.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-lab-eval"),
    )
    args = parser.parse_args(argv)

    load_canary_report = json.loads(args.load_canary_report.read_text())
    report = build_lab_eval_report(
        load_canary_report=load_canary_report,
        policy_status=args.policy_status,
    )
    json_path, md_path = write_lab_eval_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["admittedPrivateAuthorizedSecurityLabCanary"] else 2


def retarget_plan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a public-safe DeepSeek-V4-Fable packed-runtime retargeting plan."
    )
    parser.add_argument(
        "--compatibility-report",
        type=Path,
        required=True,
        help="JSON report from hydralisk-deepseek-v4-fable-adapter-probe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-retarget-plan"),
    )
    args = parser.parse_args(argv)

    compatibility_report = json.loads(args.compatibility_report.read_text())
    report = build_retarget_plan_report(compatibility_report=compatibility_report)
    json_path, md_path = write_retarget_plan_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["canAttemptPackedRetargetSmoke"] else 2


def o_proj_ownership_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a public-safe DeepSeek-V4-Fable o_proj ownership decision."
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        required=True,
        help="JSON AST/source inventory for relevant NVIDIA DeepSeek V4 runtime modules.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-o-proj-ownership"),
    )
    args = parser.parse_args(argv)

    source_inventory = json.loads(args.source_inventory.read_text())
    report = build_o_proj_ownership_report(source_inventory=source_inventory)
    json_path, md_path = write_o_proj_ownership_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["canProceedToPackedLoraTransformSmoke"] else 2


def transform_smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a public-safe DeepSeek-V4-Fable packed-LoRA transform smoke."
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Local Fable adapter_model.safetensors path in ignored evidence space.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-fable-transform-smoke"),
    )
    args = parser.parse_args(argv)

    report = build_transform_smoke_report(adapter_path=args.adapter_path)
    json_path, md_path = write_transform_smoke_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, indent=2))
    return 0 if report["decision"]["canImplementPackedDeltaWriter"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
