#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
KEY_FILES="${KEY_FILES:-}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-service-account-keys-$TS}"

mkdir -p "$OUTPUT_DIR"

RUNNER_PERMISSIONS=(
  compute.instances.create
  compute.instances.get
  compute.instances.setLabels
  compute.instances.setMetadata
  compute.instances.setTags
  compute.disks.create
  compute.subnetworks.use
)
GRANT_PROJECT_PERMISSIONS=(
  iam.roles.create
  iam.roles.update
  resourcemanager.projects.get
  resourcemanager.projects.getIamPolicy
  resourcemanager.projects.setIamPolicy
)
GRANT_SERVICE_ACCOUNT_PERMISSIONS=(
  iam.serviceAccounts.get
  iam.serviceAccounts.getIamPolicy
  iam.serviceAccounts.setIamPolicy
)

RESULTS_TSV="$OUTPUT_DIR/service-account-keys.tsv"
RESULTS_JSONL="$OUTPUT_DIR/service-account-keys.jsonl"
token_file=""
gcloud_config_dir=""
trap 'rm -f "${token_file:-}"; rm -rf "${gcloud_config_dir:-}"' EXIT

if [[ -z "$KEY_FILES" ]]; then
  echo "KEY_FILES is required; pass comma or space separated service-account key paths" >&2
  exit 2
fi

permissions_json() {
  python3 -c 'import json, sys; print(json.dumps({"permissions": sys.argv[1:]}))' "$@"
}

extract_key_metadata() {
  local key_file="$1"
  local output_file="$2"
  python3 - "$key_file" "$output_file" <<'PY'
import json
import sys
from pathlib import Path

key_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
try:
    data = json.loads(key_path.read_text())
except Exception as exc:
    output_path.write_text(json.dumps({
        "readable": False,
        "error": type(exc).__name__,
    }, sort_keys=True))
    sys.exit(0)

output_path.write_text(json.dumps({
    "readable": True,
    "type": data.get("type") or "",
    "projectId": data.get("project_id") or "",
    "clientEmail": data.get("client_email") or "",
}, sort_keys=True))
PY
}

metadata_value() {
  local metadata_file="$1"
  local key="$2"
  python3 - "$metadata_file" "$key" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text() or "{}")
print(data.get(sys.argv[2]) or "")
PY
}

missing_permissions() {
  local response_file="$1"
  local missing_file="$2"
  shift 2
  python3 - "$response_file" "$missing_file" "$@" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text() or "{}")
present = set(response.get("permissions") or [])
missing = [permission for permission in sys.argv[3:] if permission not in present]
Path(sys.argv[2]).write_text("\n".join(missing) + ("\n" if missing else ""))
PY
}

read_project_number() {
  local response_file="$1"
  python3 - "$response_file" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text() or "{}")
value = response.get("projectNumber") or response.get("project_number") or ""
print(value)
PY
}

test_permissions_with_token_file() {
  local token_file="$1"
  local resource_url="$2"
  local output_file="$3"
  shift 3
  curl -fsS -X POST "$resource_url:testIamPermissions" \
    -H "Authorization: Bearer $(cat "$token_file")" \
    -H 'Content-Type: application/json' \
    -d "$(permissions_json "$@")" \
    > "$output_file"
}

fetch_project_with_token_file() {
  local token_file="$1"
  local output_file="$2"
  curl -fsS "https://cloudresourcemanager.googleapis.com/v1/projects/$PROJECT_ID" \
    -H "Authorization: Bearer $(cat "$token_file")" \
    > "$output_file"
}

key_slug() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.@=-' '_'
}

collect_key_files() {
  tr ', ' '\n\n' <<< "$KEY_FILES" | sed '/^$/d'
}

recommendation_for() {
  local metadata_status="$1" activation_status="$2" runner_status="$3" grant_status="$4"
  if [[ "$metadata_status" != "ok" ]]; then
    echo "use a readable service-account JSON key with client_email and project_id metadata"
  elif [[ "$activation_status" != "ok" ]]; then
    echo "key activation failed; use a valid active service-account key or another credential surface"
  elif [[ "$runner_status" = "ok" ]]; then
    echo "rerun scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh with this key in an isolated gcloud config"
  elif [[ "$grant_status" = "ok" ]]; then
    echo "run APPLY=1 scripts/plan-deepseek-v4-g4-iam-grant.sh with this key in an isolated gcloud config, then rerun the DeepSeek G4 wrapper"
  else
    echo "this key cannot run or grant the DeepSeek G4 smoke"
  fi
}

printf 'key_label\tclient_email\tproject_id\tmetadata_status\tactivation_status\trunner_status\trunner_missing\tgrant_status\tgrant_missing\trecommendation\n' > "$RESULTS_TSV"
: > "$RESULTS_JSONL"

index=0
while IFS= read -r key_file; do
  index=$((index + 1))
  key_label="key-$index"
  slug="$(key_slug "$key_label")"
  metadata_file="$OUTPUT_DIR/metadata-$slug.json"
  activation_err="$OUTPUT_DIR/activate-$slug.err"
  token_err="$OUTPUT_DIR/mint-$slug.err"
  project_response="$OUTPUT_DIR/project-$slug.json"
  project_err="$OUTPUT_DIR/project-$slug.err"
  runner_response="$OUTPUT_DIR/runner-$slug.json"
  runner_missing="$OUTPUT_DIR/runner-missing-$slug.txt"
  grant_project_response="$OUTPUT_DIR/grant-project-$slug.json"
  grant_service_response="$OUTPUT_DIR/grant-service-$slug.json"
  grant_missing="$OUTPUT_DIR/grant-missing-$slug.txt"

  extract_key_metadata "$key_file" "$metadata_file"
  client_email="$(metadata_value "$metadata_file" clientEmail)"
  key_project_id="$(metadata_value "$metadata_file" projectId)"
  key_type="$(metadata_value "$metadata_file" type)"
  readable="$(metadata_value "$metadata_file" readable)"

  metadata_status="ok"
  activation_status="skipped"
  runner_status="skipped"
  grant_status="skipped"

  if [[ "$readable" != "True" && "$readable" != "true" ]] || [[ "$key_type" != "service_account" ]] || [[ -z "$client_email" ]]; then
    metadata_status="invalid"
    printf 'metadata invalid\n' > "$runner_missing"
    printf 'metadata invalid\n' > "$grant_missing"
  else
    gcloud_config_dir="$(mktemp -d "${TMPDIR:-/tmp}/hydralisk-key-gcloud.XXXXXX")"
    if ! CLOUDSDK_CONFIG="$gcloud_config_dir" gcloud auth activate-service-account \
      "--key-file=$key_file" \
      --quiet > /dev/null 2> "$activation_err"; then
      activation_status="failed"
      printf 'activation failed\n' > "$runner_missing"
      printf 'activation failed\n' > "$grant_missing"
    else
      activation_status="ok"
      token_file="$(mktemp "${TMPDIR:-/tmp}/hydralisk-key-token.XXXXXX")"
      if ! CLOUDSDK_CONFIG="$gcloud_config_dir" gcloud auth print-access-token > "$token_file" 2> "$token_err"; then
        activation_status="token_failed"
        printf 'token mint failed\n' > "$runner_missing"
        printf 'token mint failed\n' > "$grant_missing"
      else
        if test_permissions_with_token_file \
          "$token_file" \
          "https://cloudresourcemanager.googleapis.com/v1/projects/$PROJECT_ID" \
          "$runner_response" \
          "${RUNNER_PERMISSIONS[@]}" 2> "$OUTPUT_DIR/runner-$slug.err"; then
          missing_permissions "$runner_response" "$runner_missing" "${RUNNER_PERMISSIONS[@]}"
          if [[ -s "$runner_missing" ]]; then
            runner_status="missing_permissions"
          else
            runner_status="ok"
          fi
        else
          runner_status="check_failed"
          printf 'runner testIamPermissions failed\n' > "$runner_missing"
        fi

        if fetch_project_with_token_file "$token_file" "$project_response" 2> "$project_err"; then
          project_number="$(read_project_number "$project_response")"
          default_compute_sa="${project_number}-compute@developer.gserviceaccount.com"

          if test_permissions_with_token_file \
            "$token_file" \
            "https://cloudresourcemanager.googleapis.com/v1/projects/$PROJECT_ID" \
            "$grant_project_response" \
            "${GRANT_PROJECT_PERMISSIONS[@]}" 2> "$OUTPUT_DIR/grant-project-$slug.err"; then
            missing_permissions "$grant_project_response" "$grant_missing" "${GRANT_PROJECT_PERMISSIONS[@]}"
          else
            printf 'project grant testIamPermissions failed\n' > "$grant_missing"
          fi

          if [[ -n "$project_number" ]] && test_permissions_with_token_file \
            "$token_file" \
            "https://iam.googleapis.com/v1/projects/$PROJECT_ID/serviceAccounts/$default_compute_sa" \
            "$grant_service_response" \
            "${GRANT_SERVICE_ACCOUNT_PERMISSIONS[@]}" 2> "$OUTPUT_DIR/grant-service-$slug.err"; then
            missing_permissions "$grant_service_response" "$OUTPUT_DIR/grant-service-missing-$slug.txt" "${GRANT_SERVICE_ACCOUNT_PERMISSIONS[@]}"
          else
            printf '%s\n' "${GRANT_SERVICE_ACCOUNT_PERMISSIONS[@]}" > "$OUTPUT_DIR/grant-service-missing-$slug.txt"
          fi
          cat "$OUTPUT_DIR/grant-service-missing-$slug.txt" >> "$grant_missing"
        else
          printf 'project lookup failed\n' > "$grant_missing"
        fi

        if [[ -s "$grant_missing" ]]; then
          grant_status="missing_permissions"
        else
          grant_status="ok"
        fi
        rm -f "$token_file"
      fi
    fi
    rm -rf "$gcloud_config_dir"
  fi

  runner_missing_compact="$(paste -sd, "$runner_missing" 2>/dev/null || true)"
  grant_missing_compact="$(paste -sd, "$grant_missing" 2>/dev/null || true)"
  recommendation="$(recommendation_for "$metadata_status" "$activation_status" "$runner_status" "$grant_status")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key_label" "$client_email" "$key_project_id" "$metadata_status" "$activation_status" \
    "$runner_status" "$runner_missing_compact" "$grant_status" "$grant_missing_compact" "$recommendation" >> "$RESULTS_TSV"

  python3 - "$key_label" "$client_email" "$key_project_id" "$metadata_status" "$activation_status" "$runner_status" "$runner_missing" "$grant_status" "$grant_missing" "$recommendation" <<'PY' >> "$RESULTS_JSONL"
import json
import sys
from pathlib import Path

(
    key_label,
    client_email,
    project_id,
    metadata_status,
    activation_status,
    runner_status,
    runner_missing_path,
    grant_status,
    grant_missing_path,
    recommendation,
) = sys.argv[1:]

def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

print(json.dumps({
    "keyLabel": key_label,
    "clientEmail": client_email,
    "projectId": project_id,
    "metadataStatus": metadata_status,
    "activationStatus": activation_status,
    "runnerStatus": runner_status,
    "runnerMissing": read_lines(runner_missing_path),
    "grantStatus": grant_status,
    "grantMissing": read_lines(grant_missing_path),
    "recommendation": recommendation,
}, sort_keys=True))
PY
done < <(collect_key_files)

MD="$OUTPUT_DIR/service-account-keys.md"
{
  echo "# DeepSeek-V4-Flash service-account key probe"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "- Project: \`$PROJECT_ID\`"
  echo "- Key file paths written to evidence: \`false\`"
  echo "- Private key material written to evidence: \`false\`"
  echo "- Tokens written to evidence: \`false\`"
  echo
  echo "## Required Runner Permissions"
  echo
  echo '```text'
  printf '%s\n' "${RUNNER_PERMISSIONS[@]}"
  echo '```'
  echo
  echo "## Required Grant Permissions"
  echo
  echo '```text'
  printf '%s\n' "${GRANT_PROJECT_PERMISSIONS[@]}"
  printf '%s\n' "${GRANT_SERVICE_ACCOUNT_PERMISSIONS[@]}"
  echo '```'
  echo
  echo "## Results"
  echo
  echo '```tsv'
  cat "$RESULTS_TSV"
  echo '```'
  echo
  echo "## Public safety"
  echo
  echo "- Contains secrets: false"
  echo "- Contains key file paths: false"
  echo "- Contains private key material: false"
  echo "- Contains private prompts: false"
  echo "- Contains private responses: false"
  echo "- Contains weights: false"
  echo "- Contains hidden reasoning: false"
} > "$MD"

echo "Wrote $MD"
echo "OUTPUT_DIR=$OUTPUT_DIR"
