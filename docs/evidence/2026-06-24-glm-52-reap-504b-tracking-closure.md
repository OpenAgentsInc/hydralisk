# GLM-5.2 504B REAP tracking closure

Date: 2026-06-24

Tracking issue: https://github.com/OpenAgentsInc/hydralisk/issues/93

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Integration receipt:
[`docs/evidence/2026-06-24-glm-52-reap-504b-integration-receipt.json`](2026-06-24-glm-52-reap-504b-integration-receipt.json)

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this closure ledger contains issue links, commit
hashes, public-safe evidence references, statuses, and claim boundaries only.
It contains no bearer token, model-provider credentials, raw prompts, raw
responses, private source, hidden reasoning traces, weights, checkpoints,
compiled engines, profiler dumps, raw benchmark folders, or large logs.

## Final Status

Status: `private_canary`

Hydralisk has completed the first-pass integration path for
`0xSero/GLM-5.2-504B` REAP/NVFP4 on OpenAgents Google Cloud infrastructure.

Definition of done is satisfied for a private Hydralisk canary:

- Pinned model profile
- Google Cloud G4 hardware admission with explicit 4x capacity boundary
- Durable checkpoint staging and public-safe manifest
- Pinned b12x/vLLM runtime image and launch profile
- Passing private load smoke
- Private OpenAI-compatible proxy
- Tuned context/concurrency/MTP envelope
- Public-safe Terminal-Bench 2.0 pilot receipt
- Fallback matrix and claim guardrails
- Operator hardening, metrics, and recovery runbook
- Consolidated final runbook and public-safe integration receipt

## Work Plan Ledger

| Issue | Result | Commit | Evidence |
| --- | --- | --- | --- |
| #82 Profile and evidence contract | Done | `20cb9e1` | [`2026-06-24-glm-52-reap-504b-profile.md`](2026-06-24-glm-52-reap-504b-profile.md) |
| #83 4x G4 admission | Done with 8x fallback | `6bd6130` | [`2026-06-24-glm-52-reap-504b-g4-admission.md`](2026-06-24-glm-52-reap-504b-g4-admission.md) |
| #84 Checkpoint staging and manifest | Done | `298b418` | [`2026-06-24-glm-52-reap-504b-staging.md`](2026-06-24-glm-52-reap-504b-staging.md) |
| #85 b12x/vLLM launch profile | Done | `5740176` | [`2026-06-24-glm-52-reap-504b-b12x-launch-profile.md`](2026-06-24-glm-52-reap-504b-b12x-launch-profile.md) |
| #86 Conservative load smoke | Done | `ef5a6ac` | [`2026-06-24-glm-52-reap-504b-load-smoke.md`](2026-06-24-glm-52-reap-504b-load-smoke.md) |
| #87 Private OpenAI-compatible endpoint | Done | `76b27c6` | [`2026-06-24-glm-52-reap-504b-private-endpoint.md`](2026-06-24-glm-52-reap-504b-private-endpoint.md) |
| #88 Long-context/concurrency/MTP tuning | Done | `4f6a92b` | [`2026-06-24-glm-52-reap-504b-tuning.md`](2026-06-24-glm-52-reap-504b-tuning.md) |
| #89 Terminal-Bench 2.0 validation | Preliminary pilot | `177f61c` | [`2026-06-24-glm-52-reap-504b-terminal-bench-20.md`](2026-06-24-glm-52-reap-504b-terminal-bench-20.md) |
| #90 Fallback lanes | Done | `f6b6ad9` | [`2026-06-24-glm-52-reap-504b-fallback-matrix.md`](2026-06-24-glm-52-reap-504b-fallback-matrix.md) |
| #91 Operator hardening | Done | `dfb8451` | [`2026-06-24-glm-52-reap-504b-operator-hardening.md`](2026-06-24-glm-52-reap-504b-operator-hardening.md) |
| #92 Final runbook and receipt | Done | `f083d5c` | [`../glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md) |

## Claim Boundary

Admitted:

- Private Hydralisk canary serving path.
- Exact model revision:
  `0xSero/GLM-5.2-504B@cb6b1e0451b9d560cda864f84187869c9a679712`.
- Exact runtime image digest:
  `sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997`.
- 250K admitted context, `max_num_seqs=2`, `max_num_batched_tokens=4096`.
- One full-250K request at a time through the proxy.
- Four selected RTX PRO 6000 GPUs inside the admitted 8x G4 fallback host.
- Public-safe Terminal-Bench 2.0 pilot summary with named denominators.

Not admitted:

- Public production SLA.
- Public endpoint.
- Billing, credits, customer routing, settlement, payout, or product promise.
- Fresh standalone 4x `g4-standard-192` capacity availability.
- Two concurrent full-250K requests.
- Final Terminal-Bench leaderboard claim.
- Committed model weights, compiled engines, hidden reasoning, raw prompts, raw
  responses, raw benchmark logs, or profiler dumps.

## Next Honest Work

- Retry standalone 4x G4 admission when capacity is available.
- Finish a final Terminal-Bench run with all retries and not-started tasks
  resolved.
- Add a separate product-repo issue for any OpenAgents UI/API/customer routing
  work; do not promote Hydralisk evidence into product promises without that
  boundary review.
- Keep operator canaries receipt-safe and private until SLA, abuse controls,
  billing, and customer routing exist outside Hydralisk.
