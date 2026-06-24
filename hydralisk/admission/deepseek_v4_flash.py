from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import Any, BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_SCHEMA = "hydralisk.gguf.metadata.v1"
PREFLIGHT_SCHEMA = "hydralisk.deepseek-v4-flash.gce-preflight.v1"
MIB = 1024 * 1024
MAX_ARRAY_ITEMS = 64

GGUF_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

DEEPSEEK_KEYS = {
    "general.architecture",
    "general.name",
    "general.size_label",
    "general.license",
    "tokenizer.ggml.model",
    "deepseek4.block_count",
    "deepseek4.context_length",
    "deepseek4.embedding_length",
    "deepseek4.feed_forward_length",
    "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length",
    "deepseek4.attention.value_length",
    "deepseek4.attention.compress_ratios",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.expert_shared_count",
    "deepseek4.sliding_window",
    "deepseek4.indexer.head_count",
    "deepseek4.indexer.key_length",
    "deepseek4.indexer.top_k",
    "deepseek4.hca.count",
    "deepseek4.sinkhorn_iterations",
}


@dataclass(frozen=True)
class GgufMetadata:
    path: str
    fileBytes: int
    fileMiB: int
    schema: str
    version: int
    tensorCount: int
    metadataCount: int
    values: dict[str, Any]
    parser: dict[str, Any]


@dataclass(frozen=True)
class GpuLane:
    laneId: str
    provider: str
    zones: tuple[str, ...]
    machineType: str
    accelerator: str
    gpuName: str
    gpuCount: int
    gpuMemoryMiBEach: int
    hostMemoryGiB: int
    observedStatus: str
    runtimeFit: str
    notes: tuple[str, ...]

    @property
    def gpuMemoryMiBTotal(self) -> int:
        return self.gpuMemoryMiBEach * self.gpuCount

    @property
    def hostMemoryMiB(self) -> int:
        return self.hostMemoryGiB * 1024


@dataclass(frozen=True)
class LaneAdmission:
    laneId: str
    machineType: str
    zones: tuple[str, ...]
    accelerator: str
    gpuName: str
    gpuCount: int
    gpuMemoryMiBTotal: int
    hostMemoryGiB: int
    observedStatus: str
    admissionClass: str
    decision: str
    weightsMiB: int
    smokeKvReserveMiB: int
    runtimeReserveMiB: int
    usableGpuMiBAt90Pct: int
    marginMiB: int
    reasons: tuple[str, ...]
    notes: tuple[str, ...]


DEFAULT_GCE_LANES = (
    GpuLane(
        laneId="live-a3-highgpu-1g-h100-gptoss120b",
        provider="gce",
        zones=("us-central1-b",),
        machineType="a3-highgpu-1g",
        accelerator="nvidia-h100-80gb",
        gpuName="NVIDIA H100 80GB HBM3",
        gpuCount=1,
        gpuMemoryMiBEach=81559,
        hostMemoryGiB=234,
        observedStatus="running_reserved_for_gpt_oss_120b",
        runtimeFit="do_not_disturb",
        notes=(
            "This is the live Hydralisk GPT-OSS 120B probe host.",
            "Do not use it for DeepSeek smoke work unless it is intentionally drained.",
        ),
    ),
    GpuLane(
        laneId="g4-standard-48-rtxpro6000-1g",
        provider="gce",
        zones=("us-central1-b", "us-central1-f"),
        machineType="g4-standard-48",
        accelerator="nvidia-rtx-pro-6000",
        gpuName="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpuCount=1,
        gpuMemoryMiBEach=97887,
        hostMemoryGiB=180,
        observedStatus="admitted_for_glm_5_2_probe",
        runtimeFit="offload_or_tight_low_context",
        notes=(
            "Cheapest plausible Google lane for an Exobyt-style hot expert cache.",
            "Single-card all-GPU is too tight once runtime and KV cache reserve are included.",
        ),
    ),
    GpuLane(
        laneId="g4-standard-96-rtxpro6000-2g",
        provider="gce",
        zones=("us-central1-b", "us-central1-f"),
        machineType="g4-standard-96",
        accelerator="nvidia-rtx-pro-6000",
        gpuName="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpuCount=2,
        gpuMemoryMiBEach=97887,
        hostMemoryGiB=360,
        observedStatus="quota_visible_capacity_unproven",
        runtimeFit="first_all_gpu_smoke_candidate_if_capacity_admits",
        notes=(
            "The most useful next G4 probe if capacity is available.",
            "Still a Blackwell compatibility risk for vLLM/SGLang FP4 kernels.",
        ),
    ),
    GpuLane(
        laneId="g4-standard-192-rtxpro6000-4g",
        provider="gce",
        zones=("us-central1-b", "us-central1-f"),
        machineType="g4-standard-192",
        accelerator="nvidia-rtx-pro-6000",
        gpuName="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpuCount=4,
        gpuMemoryMiBEach=97887,
        hostMemoryGiB=720,
        observedStatus="quota_visible_capacity_unproven",
        runtimeFit="strong_all_gpu_smoke_candidate_if_capacity_admits",
        notes=("More headroom than 2x G4; higher burn rate.",),
    ),
    GpuLane(
        laneId="g4-standard-384-rtxpro6000-8g",
        provider="gce",
        zones=("us-central1-b", "us-central1-f"),
        machineType="g4-standard-384",
        accelerator="nvidia-rtx-pro-6000",
        gpuName="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpuCount=8,
        gpuMemoryMiBEach=97887,
        hostMemoryGiB=1440,
        observedStatus="admitted_for_glm_5_2_load_smoke_then_deleted",
        runtimeFit="overprovisioned_for_flash_admission_but_useful_for_kernel_debug",
        notes=(
            "G4 admitted previously, but GLM-5.2 hit SGLang/FlashInfer DSA support blockers.",
            "Use only if smaller G4 lanes fail for capacity or topology reasons.",
        ),
    ),
    GpuLane(
        laneId="a3-highgpu-2g-h100",
        provider="gce",
        zones=("us-central1-a", "us-central1-b", "us-central1-c"),
        machineType="a3-highgpu-2g",
        accelerator="nvidia-h100-80gb",
        gpuName="NVIDIA H100 80GB HBM3",
        gpuCount=2,
        gpuMemoryMiBEach=81559,
        hostMemoryGiB=468,
        observedStatus="quota_visible_capacity_unproven",
        runtimeFit="all_gpu_candidate_if_capacity_admits",
        notes=("Hopper path is more mature, but capacity has been volatile.",),
    ),
    GpuLane(
        laneId="a3-highgpu-4g-h100",
        provider="gce",
        zones=("us-central1-a", "us-central1-b", "us-central1-c"),
        machineType="a3-highgpu-4g",
        accelerator="nvidia-h100-80gb",
        gpuName="NVIDIA H100 80GB HBM3",
        gpuCount=4,
        gpuMemoryMiBEach=81559,
        hostMemoryGiB=936,
        observedStatus="quota_visible_capacity_unproven",
        runtimeFit="strong_all_gpu_candidate_if_capacity_admits",
        notes=("Better headroom than 2x H100, but do not assume immediate capacity.",),
    ),
    GpuLane(
        laneId="a2-highgpu-4g-a100",
        provider="gce",
        zones=("us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"),
        machineType="a2-highgpu-4g",
        accelerator="nvidia-tesla-a100",
        gpuName="NVIDIA A100 40GB",
        gpuCount=4,
        gpuMemoryMiBEach=40536,
        hostMemoryGiB=340,
        observedStatus="quota_visible_capacity_unproven",
        runtimeFit="memory_possible_kernel_risk",
        notes=(
            "Enough aggregate memory for a low-context smoke.",
            "A100 is not a good target for FP4/NVFP4 Blackwell paths.",
        ),
    ),
    GpuLane(
        laneId="g2-standard-96-l4-8g",
        provider="gce",
        zones=("us-central1-a", "us-central1-b", "us-central1-c"),
        machineType="g2-standard-96",
        accelerator="nvidia-l4",
        gpuName="NVIDIA L4",
        gpuCount=8,
        gpuMemoryMiBEach=23034,
        hostMemoryGiB=384,
        observedStatus="existing_l4_capacity_visible",
        runtimeFit="deprioritized_multi_l4_smoke_only",
        notes=(
            "Aggregate memory is plausible; bandwidth and interconnect make it a poor first lane.",
            "Do not use active Khala/GPT-OSS L4 hosts for this experiment.",
        ),
    ),
)


class GgufParseError(ValueError):
    pass


def parse_gguf_metadata(path: Path) -> GgufMetadata:
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        magic = _read_exact(stream, 4)
        if magic != GGUF_MAGIC:
            raise GgufParseError(f"{path} is not a GGUF file")
        version = _read_u32(stream)
        tensor_count = _read_u64(stream)
        metadata_count = _read_u64(stream)
        values: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(stream)
            value_type = _read_u32(stream)
            value = _read_value(stream, value_type)
            if key in DEEPSEEK_KEYS:
                values[key] = value

    return GgufMetadata(
        path=str(path),
        fileBytes=file_bytes,
        fileMiB=math.ceil(file_bytes / MIB),
        schema=GGUF_SCHEMA,
        version=version,
        tensorCount=tensor_count,
        metadataCount=metadata_count,
        values=values,
        parser={
            "loadsWeights": False,
            "readsTensorData": False,
            "keptMetadataKeys": sorted(DEEPSEEK_KEYS),
            "arrayItemLimit": MAX_ARRAY_ITEMS,
        },
    )


def classify_lanes(
    *,
    model_file_mib: int,
    lanes: tuple[GpuLane, ...] = DEFAULT_GCE_LANES,
    smoke_kv_reserve_mib: int = 8192,
    runtime_reserve_mib: int = 4096,
    gpu_memory_utilization: float = 0.90,
) -> list[LaneAdmission]:
    admissions: list[LaneAdmission] = []
    for lane in lanes:
        usable_gpu_mib = math.floor(lane.gpuMemoryMiBTotal * gpu_memory_utilization)
        usable_after_reserve = usable_gpu_mib - runtime_reserve_mib
        required_mib = model_file_mib + smoke_kv_reserve_mib
        margin_mib = usable_after_reserve - required_mib
        reasons: list[str] = []

        if lane.runtimeFit == "do_not_disturb":
            admission_class = "blocked_reserved_live_host"
            decision = "reject_for_deepseek_until_drained"
            reasons.append("live host is reserved for GPT-OSS 120B")
        elif margin_mib >= 0:
            if lane.runtimeFit in {
                "memory_possible_kernel_risk",
                "deprioritized_multi_l4_smoke_only",
            }:
                admission_class = "deprioritized_memory_only_smoke"
                decision = "do_not_start_here"
                reasons.append("aggregate GPU memory is enough only on paper")
            else:
                admission_class = "candidate_all_gpu_low_context_smoke"
                decision = "proceed_if_capacity_admits"
                reasons.append("aggregate GPU memory covers weights plus smoke KV reserve")
        elif (
            lane.hostMemoryMiB >= math.ceil(model_file_mib * 1.25)
            and lane.gpuMemoryMiBTotal >= 32768
            and "l4" not in lane.accelerator
        ):
            admission_class = "candidate_offload_prefetch_smoke"
            decision = "proceed_only_for_custom_offload_prefetch_validation"
            reasons.append("host RAM can hold the artifact while GPU memory is too tight")
            reasons.append("requires hot expert cache or CPU-offload bridge")
        elif lane.hostMemoryMiB >= math.ceil(model_file_mib * 1.25):
            admission_class = "deprioritized_offload_smoke"
            decision = "do_not_start_here"
            reasons.append("host RAM can hold the artifact, but GPU lane is too weak")
        else:
            admission_class = "rejected_memory"
            decision = "reject"
            reasons.append("neither GPU memory nor host RAM margin is sufficient")

        if "a100" in lane.accelerator:
            reasons.append("A100 lacks the Blackwell FP4/NVFP4 path expected by the official recipe")
        if "l4" in lane.accelerator:
            reasons.append("multi-L4 has poor bandwidth/interconnect for this MoE path")
        if "rtx-pro-6000" in lane.accelerator:
            reasons.append("Blackwell GPU is directionally aligned with FP4 work but kernel support must be proven")

        admissions.append(
            LaneAdmission(
                laneId=lane.laneId,
                machineType=lane.machineType,
                zones=lane.zones,
                accelerator=lane.accelerator,
                gpuName=lane.gpuName,
                gpuCount=lane.gpuCount,
                gpuMemoryMiBTotal=lane.gpuMemoryMiBTotal,
                hostMemoryGiB=lane.hostMemoryGiB,
                observedStatus=lane.observedStatus,
                admissionClass=admission_class,
                decision=decision,
                weightsMiB=model_file_mib,
                smokeKvReserveMiB=smoke_kv_reserve_mib,
                runtimeReserveMiB=runtime_reserve_mib,
                usableGpuMiBAt90Pct=usable_after_reserve,
                marginMiB=margin_mib,
                reasons=tuple(reasons),
                notes=lane.notes,
            )
        )
    return admissions


def build_preflight_report(
    *,
    metadata: GgufMetadata,
    admissions: list[LaneAdmission],
    gcloud_inventory: list[dict[str, Any]] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(UTC)
    best = _recommended_next_step(admissions)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "profileRef": "profiles/deepseek-v4-flash-gce-preflight.json",
        "issue": "https://github.com/OpenAgentsInc/hydralisk/issues/5",
        "model": {
            "servedModel": "deepseek-ai/DeepSeek-V4-Flash",
            "localArtifact": {
                "path": metadata.path,
                "fileBytes": metadata.fileBytes,
                "fileMiB": metadata.fileMiB,
            },
            "gguf": asdict(metadata),
            "expectedArchitecture": "deepseek4",
            "officialScale": {
                "totalParameters": "284B",
                "activeParameters": "13B",
                "contextWindowTokens": 1048576,
                "quantization": "official FP4 + FP8 mixed path; local GGUF is IQ2XXS/w2Q2K/AProjQ8/SExpQ8/OutQ8",
            },
        },
        "admissionPolicy": {
            "loadsWeights": False,
            "smokeKvReserveMiB": 8192,
            "runtimeReserveMiB": 4096,
            "gpuMemoryUtilization": 0.90,
            "fullContextStatus": "not_admitted_by_this_preflight",
            "publicClaim": "no_public_khala_or_model_selector_claim_until_load_generation_eval_and_capacity_receipts_pass",
        },
        "lanes": [asdict(admission) for admission in admissions],
        "gcloudInventory": gcloud_inventory or [],
        "recommendation": best,
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    model = report["model"]
    values = model["gguf"]["values"]
    recommendation = report["recommendation"]
    lanes = report["lanes"]
    rows = [
        "| Lane | Decision | Class | GPU memory | Margin | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for lane in lanes:
        notes = "; ".join([*lane["reasons"], *lane["notes"]])
        rows.append(
            f"| {lane['laneId']} | {lane['decision']} | "
            f"{lane['admissionClass']} | {lane['gpuMemoryMiBTotal']} MiB | "
            f"{lane['marginMiB']} MiB | {notes.replace('|', '/')} |"
        )

    inventory = report.get("gcloudInventory") or []
    if inventory:
        inventory_lines = "\n".join(
            f"- `{item.get('name')}`: `{item.get('machineType')}`, "
            f"`{item.get('status')}`, `{item.get('zone')}`, "
            f"`{item.get('accelerators')}`"
            for item in inventory
        )
    else:
        inventory_lines = "- No live gcloud inventory was collected by this run."

    values_json = json.dumps(values, indent=2, sort_keys=True)
    rows_text = "\n".join(rows)
    artifact_path = _public_path(str(model["localArtifact"]["path"]))
    return f"""# DeepSeek-V4-Flash GCE admission preflight evidence

Date: {report["createdAt"]}

Issue: {report["issue"]}

Profile: `{report["profileRef"]}`

## Result

Recommendation: `{recommendation["nextStep"]}`

{recommendation["summary"]}

Do not disturb the live single-H100 GPT-OSS 120B host for this experiment.
The first useful Hydralisk step is a G4 or multi-H100 admission/load preflight,
not a product route and not a public model selector.

## Parsed local artifact

- Path: `{artifact_path}`
- Size: {model["localArtifact"]["fileBytes"]} bytes ({model["localArtifact"]["fileMiB"]} MiB)
- GGUF version: {model["gguf"]["version"]}
- Tensor count: {model["gguf"]["tensorCount"]}
- Metadata entries: {model["gguf"]["metadataCount"]}

Selected metadata:

```json
{values_json}
```

## Lane classification

{rows_text}

## Live gcloud GPU inventory

{inventory_lines}

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
"""


def collect_gcloud_inventory() -> list[dict[str, Any]]:
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        "--filter=guestAccelerators:*",
        "--format=json(name,zone,machineType,status,guestAccelerators,scheduling.provisioningModel)",
    ]
    raw = subprocess.check_output(cmd, text=True)
    instances = json.loads(raw)
    sanitized = []
    for instance in instances:
        accelerators = []
        for accelerator in instance.get("guestAccelerators") or []:
            accelerators.append(
                {
                    "type": _basename(str(accelerator.get("acceleratorType", ""))),
                    "count": accelerator.get("acceleratorCount"),
                }
            )
        sanitized.append(
            {
                "name": instance.get("name"),
                "zone": _basename(str(instance.get("zone", ""))),
                "machineType": _basename(str(instance.get("machineType", ""))),
                "status": instance.get("status"),
                "accelerators": accelerators,
                "provisioningModel": (instance.get("scheduling") or {}).get(
                    "provisioningModel"
                ),
            }
        )
    return sanitized


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-flash-gce-preflight.json"
    md_path = output_dir / "deepseek-v4-flash-gce-preflight.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a public-safe DeepSeek-V4-Flash GCE admission preflight."
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="Path to the local DeepSeek-V4-Flash GGUF artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk/deepseek-v4-flash-preflight"),
        help="Directory for generated public-safe preflight evidence.",
    )
    parser.add_argument(
        "--collect-gcloud",
        action="store_true",
        help="Collect sanitized live gcloud GPU instance inventory.",
    )
    args = parser.parse_args(argv)

    metadata = parse_gguf_metadata(args.gguf)
    admissions = classify_lanes(model_file_mib=metadata.fileMiB)
    inventory = collect_gcloud_inventory() if args.collect_gcloud else None
    report = build_preflight_report(
        metadata=metadata,
        admissions=admissions,
        gcloud_inventory=inventory,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Recommendation: {report['recommendation']['nextStep']}")
    return 0


def _recommended_next_step(admissions: list[LaneAdmission]) -> dict[str, str]:
    by_id = {admission.laneId: admission for admission in admissions}
    g4_single = by_id["g4-standard-48-rtxpro6000-1g"]
    g4_two = by_id["g4-standard-96-rtxpro6000-2g"]
    h100_two = by_id["a3-highgpu-2g-h100"]
    return {
        "nextStep": "try_g4_standard_96_all_gpu_or_g4_standard_48_offload_prefetch_smoke",
        "summary": (
            "Hydralisk should proceed, but only as an admission/load experiment. "
            f"Two RTX PRO 6000 GPUs have {g4_two.gpuMemoryMiBTotal} MiB aggregate memory "
            f"and clear the low-context all-GPU preflight by {g4_two.marginMiB} MiB. "
            f"One RTX PRO 6000 misses the conservative all-GPU reserve by {-g4_single.marginMiB} MiB, "
            "but is the cheapest candidate for the custom hot-expert-cache/offload bridge "
            "described in the DeepSeek-V4-Flash thread. "
            f"Two H100s also clear the all-GPU memory preflight by {h100_two.marginMiB} MiB "
            "if Google has capacity. The live single-H100 host is rejected because it is "
            "reserved for GPT-OSS 120B."
        ),
        "hydraliskVsPsionic": (
            "Start in Hydralisk to validate vLLM/SGLang/CUDA admission and collect receipts; "
            "port the proven scheduling and expert-prefetch behavior into Psionic later."
        ),
    }


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise GgufParseError("truncated GGUF metadata")
    return data


def _read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_string(stream: BinaryIO) -> str:
    length = _read_u64(stream)
    return _read_exact(stream, length).decode("utf-8", errors="replace")


def _read_value(stream: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return _read_string(stream)
    if value_type == 9:
        return _read_array(stream)
    if value_type not in SCALAR_FORMATS:
        raise GgufParseError(f"unsupported GGUF metadata type {value_type}")
    fmt = "<" + SCALAR_FORMATS[value_type]
    return struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _read_array(stream: BinaryIO) -> Any:
    element_type = _read_u32(stream)
    count = _read_u64(stream)
    element_name = GGUF_TYPES.get(element_type, f"unknown:{element_type}")
    if count <= MAX_ARRAY_ITEMS:
        return [_read_value(stream, element_type) for _ in range(count)]

    if element_type == 8:
        for _ in range(count):
            _skip_string(stream)
    elif element_type in SCALAR_FORMATS:
        size = struct.calcsize("<" + SCALAR_FORMATS[element_type])
        stream.seek(size * count, os.SEEK_CUR)
    else:
        raise GgufParseError(f"unsupported GGUF array element type {element_type}")
    return {"type": f"array<{element_name}>", "length": count}


def _skip_string(stream: BinaryIO) -> None:
    length = _read_u64(stream)
    stream.seek(length, os.SEEK_CUR)


def _basename(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _public_path(value: str) -> str:
    home = str(Path.home())
    if value.startswith(home + os.sep):
        return "~" + value[len(home) :]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
