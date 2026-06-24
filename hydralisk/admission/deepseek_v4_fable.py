from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


FABLE_SCHEMA = "hydralisk.deepseek-v4-fable.adapter-compatibility.v1"
FABLE_REPO = "Chunjiang-Intelligence/DeepSeek-v4-Fable"
FABLE_REVISION = "999909137c15e0b5539fee887431824fa7cb5b10"
FABLE_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROFILE_REF = "profiles/deepseek-v4-fable-adapter-g4.json"
ISSUE_URL = "https://github.com/OpenAgentsInc/hydralisk/issues/67"

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

{"\n".join(file_rows)}

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

{"\n".join(target_rows)}

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
