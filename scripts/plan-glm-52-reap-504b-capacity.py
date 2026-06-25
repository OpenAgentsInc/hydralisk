#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any


MONTH_HOURS = 730
MONTH_SECONDS = MONTH_HOURS * 60 * 60
TOKENS_PER_SECOND_PER_SLOT = 46.7
UTILIZATION_BANDS = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class Price:
    key: str
    label: str
    hourly_usd: float
    monthly_usd: float
    reliability: str


PRICES = (
    Price("spot", "Spot", 3.69344, 2696.21, "cheap_interruptible"),
    Price("dws_flex", "DWS Flex-start", 9.00000, 6570.00, "queued_flex_start"),
    Price("on_demand", "On-demand", 17.99972, 13139.80, "durable_if_stock_exists"),
    Price("cud_1y", "1-year CUD/reserved stock", 12.42000, 9066.60, "procured_capacity"),
    Price("cud_3y", "3-year CUD/reserved stock", 7.91780, 5779.99, "procured_capacity"),
)


def scenario_rows(slots: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monthly_tokens_at_full_util = TOKENS_PER_SECOND_PER_SLOT * MONTH_SECONDS
    for slot_count in slots:
        for price in PRICES:
            monthly_vm_usd = price.monthly_usd * slot_count
            row: dict[str, Any] = {
                "slots": slot_count,
                "replicas": slot_count,
                "gpus": slot_count * 4,
                "shape": "g4-standard-192 per replica",
                "pricing": price.key,
                "pricingLabel": price.label,
                "reliability": price.reliability,
                "monthlyVmUsd": round(monthly_vm_usd, 2),
                "hourlyVmUsd": round(price.hourly_usd * slot_count, 5),
                "monthlyOutputTokensAtFullUtilization": round(
                    monthly_tokens_at_full_util * slot_count
                ),
                "costPerMillionOutputTokens": {
                    f"{int(util * 100)}pct_util": round(
                        monthly_vm_usd
                        / ((monthly_tokens_at_full_util * slot_count * util) / 1_000_000),
                        2,
                    )
                    for util in UTILIZATION_BANDS
                },
            }
            rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# GLM-5.2 REAP capacity scenarios",
        "",
        "| Slots | Replicas | GPUs | Pricing | VM $/mo | $/M output tok @10% | @25% | @50% | Reliability |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        c = row["costPerMillionOutputTokens"]
        lines.append(
            "| {slots} | {replicas} | {gpus} | {pricingLabel} | ${monthlyVmUsd:,.2f} | ${c10:,.2f} | ${c25:,.2f} | ${c50:,.2f} | {reliability} |".format(
                slots=row["slots"],
                replicas=row["replicas"],
                gpus=row["gpus"],
                pricingLabel=row["pricingLabel"],
                monthlyVmUsd=row["monthlyVmUsd"],
                c10=c["10pct_util"],
                c25=c["25pct_util"],
                c50=c["50pct_util"],
                reliability=row["reliability"],
            )
        )
    lines.extend(
        [
            "",
            "Assumptions:",
            "",
            f"- one slot = one warmed 4 x G4 GLM replica;",
            f"- per-slot decode throughput = {TOKENS_PER_SECOND_PER_SLOT} output tokens/s;",
            f"- month = {MONTH_HOURS} hours;",
            "- costs include VM only, before model-disk storage, egress, logging, control plane, and idle warm probes;",
            "- token-cost rows use output/decode tokens only; long-prefill workloads can be dominated by TTFT and should be modeled separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GLM REAP capacity scenarios")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--slots", default="2,4,8", help="comma-separated slot counts")
    args = parser.parse_args()
    slots = tuple(int(part) for part in args.slots.split(",") if part.strip())
    rows = scenario_rows(slots)
    if args.json:
        print(
            json.dumps(
                {"schema": "hydralisk.glm52_reap.capacity_plan.v1", "rows": rows},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_markdown(rows))


if __name__ == "__main__":
    main()
