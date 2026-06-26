#!/usr/bin/env bash
# Collect the live GLM-5.2-REAP-504B endpoint roster for the Khala arming agent.
# For every RUNNING GLM G4 host that has a public HTTPS origin + bearer + ready
# proxy, write one roster record to the gitignored secrets env file.
# Bearer tokens are written to the file but NEVER printed to stdout/logs.
set -uo pipefail
export PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
OUT_FILE="${OUT_FILE:-$HOME/work/.secrets/hydralisk-glm-endpoints.env}"
MODEL_ID="${MODEL_ID:-openagents/glm-5.2-reap-504b}"
PROXY_STATE_DIR="${PROXY_STATE_DIR:-/var/lib/hydralisk/glm52-reap-private-proxy}"

mkdir -p "$(dirname "$OUT_FILE")"
chmod 700 "$(dirname "$OUT_FILE")" 2>/dev/null || true

tmp="$(mktemp)"
{
  echo "# Hydralisk GLM-5.2-REAP-504B live endpoint roster"
  echo "# Generated $(date -u +%FT%TZ) by collect-glm-52-reap-504b-roster.sh"
  echo "# Consumed by the OpenAgents Khala arming agent. Bearer tokens are secret."
  echo "# Each replica: REPLICA_<n>_{ID,BASE_URL,MODEL_ID,BEARER,ZONE,MACHINE,PROVISIONING}"
  echo ""
} > "$tmp"

mapfile -t ROWS < <(gcloud compute instances list --project "$PROJECT_ID" \
  --filter="status=RUNNING AND name~glm52-reap-504b-g4" \
  --format="value(name,zone,machineType.basename(),scheduling.provisioningModel)" 2>/dev/null | sort)

n=0
healthy=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r inst zone machine prov <<< "$row"
  slug="$(echo "$inst" | sed 's/^hydralisk-glm52-reap-504b-//')"
  # Resolve external IP (public HTTPS origin uses <ip>.sslip.io)
  ip="$(gcloud compute instances describe "$inst" --project "$PROJECT_ID" --zone "$zone" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null | tr -d '\r\n')"
  if [[ -z "$ip" ]]; then
    echo "skip $slug: no external IP (HTTPS origin not set up)" >&2
    continue
  fi
  host="hydralisk-glm52-reap-504b.${ip}.sslip.io"
  base_url="https://${host}"
  # Read bearer from host (never printed)
  bearer="$(gcloud compute ssh "$inst" --project "$PROJECT_ID" --zone "$zone" --quiet \
            --command="sudo cat ${PROXY_STATE_DIR}/bearer-token 2>/dev/null" 2>/dev/null | tr -d '\r\n')"
  if [[ -z "$bearer" ]]; then
    echo "skip $slug: no bearer token on host" >&2
    continue
  fi
  # Verify HTTPS health is reachable + completion works (no token in output)
  code="$(curl -sS --max-time 60 -o /dev/null -w '%{http_code}' \
          -H "authorization: Bearer $bearer" -H 'content-type: application/json' \
          "${base_url}/v1/chat/completions" \
          --data "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply READY only.\"}],\"max_tokens\":4,\"temperature\":0}" 2>/dev/null)"
  if [[ "$code" != "200" ]]; then
    echo "skip $slug: completion HTTP $code (not healthy yet)" >&2
    continue
  fi
  n=$((n+1)); healthy=$((healthy+1))
  {
    echo "REPLICA_${n}_ID=$slug"
    echo "REPLICA_${n}_BASE_URL=$base_url"
    echo "REPLICA_${n}_MODEL_ID=$MODEL_ID"
    echo "REPLICA_${n}_BEARER=$bearer"
    echo "REPLICA_${n}_ZONE=$zone"
    echo "REPLICA_${n}_MACHINE=$machine"
    echo "REPLICA_${n}_PROVISIONING=$prov"
    echo ""
  } >> "$tmp"
  echo "roster: $slug ($zone $machine $prov) HEALTHY 200"
done

echo "REPLICA_COUNT=$healthy" >> "$tmp"
install -m 600 "$tmp" "$OUT_FILE"
rm -f "$tmp"
echo "wrote $healthy healthy replica(s) to $OUT_FILE (bearer tokens not printed)"
