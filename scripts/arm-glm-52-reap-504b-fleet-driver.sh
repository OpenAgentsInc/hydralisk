#!/usr/bin/env bash
# Drive parallel arming of all currently-RUNNING new GLM G4 hosts.
# Stays alive and waits for every per-host arm chain so background children
# are not orphaned/killed when a short-lived launcher shell exits.
set -uo pipefail
export PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p .hydralisk/arm-logs

# host-slug => "instance zone"
HOSTS=(
  "central1f hydralisk-glm52-reap-504b-g4-4g-central1f-spot-20260625203000 us-central1-f"
  "east1b hydralisk-glm52-reap-504b-g4-4g-east1b-spot-20260625203000 us-east1-b"
  "east1d hydralisk-glm52-reap-504b-g4-4g-east1d-spot-20260625203000 us-east1-d"
  "east5a hydralisk-glm52-reap-504b-g4-4g-east5a-spot-20260625203000 us-east5-a"
  "east5b hydralisk-glm52-reap-504b-g4-4g-east5b-spot-20260625203000 us-east5-b"
  "east5c hydralisk-glm52-reap-504b-g4-4g-east5c-spot-20260625211500 us-east5-c"
  "south1a hydralisk-glm52-reap-504b-g4-4g-south1a-spot-20260625203000 us-south1-a"
  "south1b hydralisk-glm52-reap-504b-g4-4g-south1b-spot-20260625211500 us-south1-b"
  "west1a hydralisk-glm52-reap-504b-g4-4g-west1a-spot-20260625203000 us-west1-a"
)

# Allow restricting to a subset via SLUGS="east1b east5a"
ONLY="${SLUGS:-}"

pids=()
for row in "${HOSTS[@]}"; do
  read -r slug inst zone <<< "$row"
  if [[ -n "$ONLY" ]] && ! grep -qw "$slug" <<< "$ONLY"; then continue; fi
  echo "$(date -u +%FT%TZ) launching $slug ($zone)"
  TARGET_INSTANCE="$inst" TARGET_ZONE="$zone" RUN_ID="arm$slug" \
    bash scripts/arm-glm-52-reap-504b-replica-gce.sh > ".hydralisk/arm-logs/$slug.log" 2>&1 &
  pids+=("$!:$slug")
  sleep 3
done

echo "$(date -u +%FT%TZ) waiting on ${#pids[@]} arm chains"
fail=0
for ps in "${pids[@]}"; do
  pid="${ps%%:*}"; slug="${ps##*:}"
  if wait "$pid"; then echo "$(date -u +%FT%TZ) DONE_OK $slug"; else echo "$(date -u +%FT%TZ) DONE_FAIL $slug rc=$?"; fail=$((fail+1)); fi
done
echo "$(date -u +%FT%TZ) fleet driver finished; failures=$fail"
