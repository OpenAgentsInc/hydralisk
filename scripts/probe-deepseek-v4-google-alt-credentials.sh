#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
IMPERSONATE_ACCOUNTS="${IMPERSONATE_ACCOUNTS:-}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-google-alt-credentials-$TS}"

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

RESULTS_TSV="$OUTPUT_DIR/google-alt-credentials.tsv"
RESULTS_JSONL="$OUTPUT_DIR/google-alt-credentials.jsonl"
token_file=""
trap 'rm -f "${token_file:-}"' EXIT

permissions_json() {
  python3 -c 'import json, sys; print(json.dumps({"permissions": sys.argv[1:]}))' "$@"
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

surface_slug() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.@=-' '_'
}

collect_surfaces() {
  printf 'adc\tapplication-default\n'

  configured_impersonation="$(
    gcloud config get-value auth/impersonate_service_account 2>/dev/null || true
  )"
  if [[ -n "$configured_impersonation" && "$configured_impersonation" != "(unset)" ]]; then
    printf 'configured_impersonation\t%s\n' "$configured_impersonation"
  fi

  if [[ -n "$IMPERSONATE_ACCOUNTS" ]]; then
    tr ', ' '\n\n' <<< "$IMPERSONATE_ACCOUNTS" | sed '/^$/d' | while IFS= read -r account; do
      printf 'explicit_impersonation\t%s\n' "$account"
    done
  fi
}

mint_token() {
  local surface="$1"
  local identity="$2"
  local output_file="$3"
  if [[ "$surface" = "adc" ]]; then
    gcloud auth application-default print-access-token > "$output_file"
  else
    gcloud auth print-access-token \
      "--impersonate-service-account=$identity" \
      > "$output_file"
  fi
}

recommendation_for() {
  local token_status="$1" runner_status="$2" grant_status="$3" surface="$4" identity="$5"
  if [[ "$token_status" != "ok" ]]; then
    if [[ "$surface" = "adc" ]]; then
      echo "refresh ADC with gcloud auth application-default login or use another credential surface"
    else
      echo "grant serviceAccountTokenCreator for $identity or use a credential that can mint this impersonated token"
    fi
  elif [[ "$runner_status" = "ok" ]]; then
    echo "rerun scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh using this credential surface"
  elif [[ "$grant_status" = "ok" ]]; then
    echo "run APPLY=1 scripts/plan-deepseek-v4-g4-iam-grant.sh with this credential surface, then rerun the DeepSeek G4 wrapper"
  else
    echo "this surface cannot run or grant the DeepSeek G4 smoke"
  fi
}

printf 'surface\tidentity\ttoken_status\trunner_status\trunner_missing\tgrant_status\tgrant_missing\trecommendation\n' > "$RESULTS_TSV"
: > "$RESULTS_JSONL"

while IFS=$'\t' read -r surface identity; do
  [[ -z "${surface:-}" ]] && continue
  slug="$(surface_slug "$surface-$identity")"
  token_file="$(mktemp "${TMPDIR:-/tmp}/hydralisk-google-alt-token.XXXXXX")"
  mint_err="$OUTPUT_DIR/mint-$slug.err"
  project_response="$OUTPUT_DIR/project-$slug.json"
  project_err="$OUTPUT_DIR/project-$slug.err"
  runner_response="$OUTPUT_DIR/runner-$slug.json"
  runner_missing="$OUTPUT_DIR/runner-missing-$slug.txt"
  grant_project_response="$OUTPUT_DIR/grant-project-$slug.json"
  grant_service_response="$OUTPUT_DIR/grant-service-$slug.json"
  grant_missing="$OUTPUT_DIR/grant-missing-$slug.txt"

  token_status="ok"
  runner_status="skipped"
  grant_status="skipped"

  if ! mint_token "$surface" "$identity" "$token_file" 2> "$mint_err"; then
    rm -f "$token_file"
    token_status="failed"
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

  runner_missing_compact="$(paste -sd, "$runner_missing" 2>/dev/null || true)"
  grant_missing_compact="$(paste -sd, "$grant_missing" 2>/dev/null || true)"
  recommendation="$(recommendation_for "$token_status" "$runner_status" "$grant_status" "$surface" "$identity")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$surface" "$identity" "$token_status" "$runner_status" "$runner_missing_compact" \
    "$grant_status" "$grant_missing_compact" "$recommendation" >> "$RESULTS_TSV"

  python3 - "$surface" "$identity" "$token_status" "$runner_status" "$runner_missing" "$grant_status" "$grant_missing" "$recommendation" <<'PY' >> "$RESULTS_JSONL"
import json
import sys
from pathlib import Path

surface, identity, token_status, runner_status, runner_missing_path, grant_status, grant_missing_path, recommendation = sys.argv[1:]
def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

print(json.dumps({
    "surface": surface,
    "identity": identity,
    "tokenStatus": token_status,
    "runnerStatus": runner_status,
    "runnerMissing": read_lines(runner_missing_path),
    "grantStatus": grant_status,
    "grantMissing": read_lines(grant_missing_path),
    "recommendation": recommendation,
}, sort_keys=True))
PY
done < <(collect_surfaces)

MD="$OUTPUT_DIR/google-alt-credentials.md"
{
  echo "# DeepSeek-V4-Flash alternate Google credential probe"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "- Project: \`$PROJECT_ID\`"
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
  echo "- Contains private prompts: false"
  echo "- Contains private responses: false"
  echo "- Contains weights: false"
  echo "- Contains hidden reasoning: false"
} > "$MD"

echo "Wrote $MD"
echo "OUTPUT_DIR=$OUTPUT_DIR"
