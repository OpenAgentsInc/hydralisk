#!/usr/bin/env bash
# Arm one freshly-admitted GLM-5.2-REAP-504B G4 host end to end:
#   1. stage the pinned public checkpoint (HF download)
#   2. launch raw vLLM (b12x MTP-2 profile) on GPUs 0-3, localhost:8000
#   3. start the bearer-gated private proxy bound to the VM private IP:8080
#   4. expose an authenticated HTTPS origin via Caddy + per-host static IP
#   5. smoke the HTTPS endpoint with a real tiny completion
# Each host is independent and this script is safe to run in parallel per host.
# Public-safe: prints status + shapes only. Bearer tokens stay on the host.
set -uo pipefail

export PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:?TARGET_INSTANCE required}"
TARGET_ZONE="${TARGET_ZONE:?TARGET_ZONE required}"
REGION="${REGION:-${TARGET_ZONE%-*}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
REPLICA_SLUG="${REPLICA_SLUG:-$(echo "$TARGET_INSTANCE" | sed 's/^hydralisk-glm52-reap-504b-//')}"
SKIP_STAGE="${SKIP_STAGE:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Per-replica private TMPDIR so parallel arm chains do not collide on the
# fixed-name mktemp templates used inside the staging/proxy helper scripts
# (macOS mktemp can fail "File exists" when many run at once).
export TMPDIR="${TMPDIR:-/tmp}/hydralisk-arm-${REPLICA_SLUG}-$$"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Per-host unique HTTPS ingress resources so replicas do not collide.
export ADDRESS_NAME="hydralisk-glm52-${REPLICA_SLUG}-ingress"
export TARGET_TAG="hydralisk-glm52-https-${REPLICA_SLUG}"
export FIREWALL_RULE="hydralisk-glm52-https-${REPLICA_SLUG}"

log() { echo "[$REPLICA_SLUG] $*"; }

log "=== arming start ($TARGET_INSTANCE $TARGET_ZONE) ==="

if [[ "$SKIP_STAGE" != "1" ]]; then
  log "step 1/5 stage model (HF download, ~288GB)"
  TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" RUN_ID="$RUN_ID" \
    bash scripts/stage-glm-52-reap-504b-gce.sh || { log "STAGE FAILED"; exit 11; }
else
  log "step 1/5 stage skipped (SKIP_STAGE=1)"
fi

log "step 2/5 launch raw vLLM"
ACTION=start TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" RUN_ID="$RUN_ID" \
  bash scripts/launch-glm-52-reap-504b-b12x-gce.sh || { log "LAUNCH FAILED"; exit 12; }

log "step 2b/5 wait for raw vLLM /v1/models ready (up to ~25 min)"
ready=0
for i in $(seq 1 75); do
  st=$(gcloud compute ssh "$TARGET_INSTANCE" --project "$PROJECT_ID" --zone "$TARGET_ZONE" \
        --tunnel-through-iap --quiet \
        --command='curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && echo READY || echo WAIT' 2>/dev/null | tail -1)
  if [[ "$st" == "READY" ]]; then ready=1; log "raw vLLM READY after ~$((i*20))s"; break; fi
  sleep 20
done
[[ "$ready" == "1" ]] || { log "VLLM NOT READY (timeout)"; exit 13; }

log "step 3/5 start private proxy"
ACTION=start TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" \
  PROXY_HOST=0.0.0.0 RUN_ID="$RUN_ID" \
  bash scripts/expose-glm-52-reap-504b-private-proxy-gce.sh || { log "PROXY FAILED"; exit 14; }

log "step 4/5 expose public HTTPS"
ACTION=setup TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" REGION="$REGION" \
  RUN_ID="$RUN_ID" \
  bash scripts/expose-glm-52-reap-504b-public-https-gce.sh || { log "HTTPS SETUP FAILED"; exit 15; }

log "step 5/5 HTTPS smoke"
ACTION=smoke TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" REGION="$REGION" \
  RUN_ID="$RUN_ID" \
  bash scripts/expose-glm-52-reap-504b-public-https-gce.sh || { log "HTTPS SMOKE FAILED"; exit 16; }

log "=== ARMED OK ==="
