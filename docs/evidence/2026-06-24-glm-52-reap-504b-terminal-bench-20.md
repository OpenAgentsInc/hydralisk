# GLM-5.2 504B REAP Terminal-Bench 2.0 eval gate

Date: 2026-06-24T23:35:47Z

Schema: `hydralisk.evals.terminal_bench.summary.v1`

Public-safety boundary: this report contains benchmark identifiers, counts,
denominator definitions, sanitized task IDs, settings, hashes, and aggregate
metrics only. It contains no raw prompts, raw responses, private source, hidden
reasoning traces, bearer tokens, model-provider credentials, raw benchmark
logs, weights, checkpoints, compiled engines, or profiler dumps.

## Benchmark

- Dataset: `terminal-bench@2.0`
- Version: `2.0`
- Dataset repository: https://github.com/harbor-framework/terminal-bench-2
- Harness repository: https://github.com/harbor-framework/harbor
- Runner: `harbor`
- Runner version: `0.15.0`
- Agent: `terminus-2`
- Model argument: `openai/glm-5.2-reap-504b-g4`
- Model alias: `glm-5.2-reap-504b-g4`
- Profile: `profiles/glm-5.2-reap-504b-b12x-g4.json`
- Hardware profile: `4x RTX PRO 6000 G4 on admitted 8x fallback host`
- Concurrent tasks: `1`
- Timeout seconds: `3600`
- Max attempts: `5`
- Retry policy: `pass@1 plus up to 4 queued retries for pass@5`

Primary sources checked on 2026-06-24:

- The Terminal-Bench 2.0 repository describes Terminal-Bench as a containerized
  agent benchmark and documents installing Harbor with `uv tool install harbor`
  or `pip install harbor`.
- The same repository documents the official dataset command shape:
  `uv run harbor run --dataset terminal-bench@2.0 --agent oracle --n-concurrent 4`
  for oracle testing, and `uv run harbor run --dataset terminal-bench@2.0
  --agent terminus-2 --model ... --n-concurrent 4` for model runs.
- The Harbor repository describes Harbor as the official harness for
  Terminal-Bench 2.0 and documents the equivalent
  `harbor run --dataset terminal-bench@2.0 --agent ... --model ...` recipe.

## Reproducible recipe

Preflight the private endpoint before starting any Terminal-Bench tasks:

```bash
export HYDRALISK_TB_BASE_URL=http://127.0.0.1:8080
export HYDRALISK_TB_MODEL=glm-5.2-reap-504b-g4
export HYDRALISK_BEARER_TOKEN=<remote proxy token, never committed>

scripts/run-glm-52-reap-terminal-bench-preflight.sh
```

For operator access from a local workstation, create an IAP/SSH tunnel to the
private proxy and point Harbor/LiteLLM at the forwarded `/v1` endpoint:

```bash
gcloud compute ssh hydralisk-glm52-reap-504b-g4-8g-b-20260624214500 \
  --project openagentsgemini \
  --zone us-central1-b \
  -- -L 18080:127.0.0.1:8080

export OPENAI_API_KEY=<remote proxy token, never committed>
export OPENAI_API_BASE=http://127.0.0.1:18080/v1
export OPENAI_BASE_URL=http://127.0.0.1:18080/v1

harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/glm-5.2-reap-504b-g4 \
  --n-concurrent 1
```

Keep `n-concurrent=1` for the first full run because issue #88 admits one full
250K request at a time. Increase concurrency only after a separate Hydralisk
concurrency receipt says the serving lane can absorb it.

After the run, reduce the Harbor job directory directly through Hydralisk's
public-safe reducer. The reducer reads only job/trial `result.json` and
`config.json`; it does not read agent trajectories, terminal panes, recordings,
raw logs, verifier artifacts, or environment files. Then build the committed
receipt:

```bash
uv run hydralisk-terminal-bench-summary \
  --harbor-job-dir <raw-harbor-jobs-dir>/<job-name> \
  --output-dir docs/evidence \
  --json-name 2026-06-24-glm-52-reap-504b-terminal-bench-20-receipt.json \
  --markdown-name 2026-06-24-glm-52-reap-504b-terminal-bench-20.md \
  --runner-version 0.15.0 \
  --agent terminus-2 \
  --model openai/glm-5.2-reap-504b-g4 \
  --model-alias glm-5.2-reap-504b-g4 \
  --n-concurrent 1 \
  --max-attempts 5 \
  --timeout-seconds 3600
```

Do not commit Harbor raw run folders, raw task prompts, terminal transcripts,
agent messages, hidden reasoning, model output, or container logs.

## Sampler

- `min_p=0.05`
- `repetition_penalty=1.05`
- `max_tokens=1024`
- `enable_thinking=false`

## Counts

| Category | Count |
| --- | ---: |
| Total tasks | 89 |
| Solved | 60 |
| Failing | 25 |
| Environment-broken | 2 |
| Not started | 2 |
| Attempted | 87 |
| Properly attempted | 85 |

Rates:

- Solved / total: `67.4%`
- Solved / attempted: `69.0%`
- Solved / properly attempted: `70.6%`
- pass@N / total: `67.4%`
- Known pass@1: `unknown`

Denominators:

- Total: all Terminal-Bench 2.0 task IDs in the run set.
- Attempted: total minus not-started tasks.
- Properly attempted: attempted minus environment-broken tasks.

Environment-broken task IDs:

- `qemu-alpine-ssh`
- `qemu-startup`

Not-started task IDs:

- not captured in the observed pilot summary

The pilot summary visible to Hydralisk did not include the two not-started task
IDs. This is why the claim status remains preliminary.

Notable solved task IDs:

- `compile-compcert`
- `build-pov-ray`
- `caffe-cifar-10`
- `crack-7z-hash`
- `feal-differential-cryptanalysis`
- `merge-diff-arc-agi-task`
- `pytorch-model-recovery`

## Claim boundary

Claim status: `preliminary_pilot_partial`.

Do not compare publicly unless benchmark version, Harbor version, agent, model alias, sampler settings, retry policy, timeout, and denominator definitions are all named.

The full leaderboard-style claim is not admitted until the run is final, all
queued retries are accounted for, and the committed receipt names the exact
Harbor version and agent settings used for the rollout.
