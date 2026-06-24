#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ACCOUNTS="${ACCOUNTS:-}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-gcloud-credentials-$TS}"

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

RESULTS_TSV="$OUTPUT_DIR/gcloud-credentials.tsv"
RESULTS_JSONL="$OUTPUT_DIR/gcloud-credentials.jsonl"
token_file=""
trap 'rm -f "${token_file:-}"' EXIT

permissions_json() {
  python3 -c 'import json, sys; print(json.dumps({"permissions": sys.argv[1:]}))' "$@"
}

json_array_from_file() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = [line.strip() for line in path.read_text().splitlines() if line.strip()]
print(json.dumps(items))
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

account_slug() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.@-' '_'
}

collect_accounts() {
  if [[ -n "$ACCOUNTS" ]]; then
    tr ', ' '\n\n' <<< "$ACCOUNTS" | sed '/^$/d'
    return 0
  fi
  gcloud auth list --format='value(account)' 2>/dev/null | sed '/^$/d'
}

recommendation_for() {
  local auth_status="$1" runner_status="$2" grant_status="$3"
  if [[ "$auth_status" != "ok" ]]; then
    echo "refresh this account with gcloud auth login or choose a working service account"
  elif [[ "$runner_status" = "ok" ]]; then
    echo "run scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh with GCLOUD_ACCOUNT set to this account"
  elif [[ "$grant_status" = "ok" ]]; then
    echo "run APPLY=1 scripts/plan-deepseek-v4-g4-iam-grant.sh with this account, then rerun the DeepSeek G4 wrapper"
  else
    echo "use an IAM-capable account to grant runner permissions, or grant this account the missing runner and grant-authority permissions"
  fi
}

printf 'account\tauth_status\trunner_status\trunner_missing\tgrant_status\tgrant_missing\trecommendation\n' > "$RESULTS_TSV"
: > "$RESULTS_JSONL"

account_list=()
while IFS= read -r account; do
  account_list+=("$account")
done < <(collect_accounts)

for account in "${account_list[@]}"; do
  slug="$(account_slug "$account")"
  token_file="$(mktemp "${TMPDIR:-/tmp}/hydralisk-gcloud-token.XXXXXX")"
  auth_err="$OUTPUT_DIR/auth-$slug.err"
  runner_response="$OUTPUT_DIR/runner-$slug.json"
  runner_missing="$OUTPUT_DIR/runner-missing-$slug.txt"
  grant_project_response="$OUTPUT_DIR/grant-project-$slug.json"
  grant_service_response="$OUTPUT_DIR/grant-service-$slug.json"
  grant_missing="$OUTPUT_DIR/grant-missing-$slug.txt"
  project_number_file="$OUTPUT_DIR/project-number-$slug.txt"
  project_number_err="$OUTPUT_DIR/project-number-$slug.err"

  auth_status="ok"
  runner_status="skipped"
  grant_status="skipped"

  if ! CLOUDSDK_CORE_ACCOUNT="$account" gcloud auth print-access-token > "$token_file" 2> "$auth_err"; then
    rm -f "$token_file"
    auth_status="failed"
    printf 'auth failed\n' > "$runner_missing"
    printf 'auth failed\n' > "$grant_missing"
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

    if CLOUDSDK_CORE_ACCOUNT="$account" gcloud projects describe "$PROJECT_ID" \
      --format='value(projectNumber)' > "$project_number_file" 2> "$project_number_err"; then
      project_number="$(cat "$project_number_file")"
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

      if test_permissions_with_token_file \
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
      printf 'project number lookup failed\n' > "$grant_missing"
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
  recommendation="$(recommendation_for "$auth_status" "$runner_status" "$grant_status")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$account" "$auth_status" "$runner_status" "$runner_missing_compact" \
    "$grant_status" "$grant_missing_compact" "$recommendation" >> "$RESULTS_TSV"

  python3 - "$account" "$auth_status" "$runner_status" "$runner_missing" "$grant_status" "$grant_missing" "$recommendation" <<'PY' >> "$RESULTS_JSONL"
import json
import sys
from pathlib import Path

account, auth_status, runner_status, runner_missing_path, grant_status, grant_missing_path, recommendation = sys.argv[1:]
def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

print(json.dumps({
    "account": account,
    "authStatus": auth_status,
    "runnerStatus": runner_status,
    "runnerMissing": read_lines(runner_missing_path),
    "grantStatus": grant_status,
    "grantMissing": read_lines(grant_missing_path),
    "recommendation": recommendation,
}, sort_keys=True))
PY
done

MD="$OUTPUT_DIR/gcloud-credentials.md"
{
  echo "# DeepSeek-V4-Flash gcloud credential authority probe"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "- Project: \`$PROJECT_ID\`"
  echo "- Account count: \`${#account_list[@]}\`"
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
