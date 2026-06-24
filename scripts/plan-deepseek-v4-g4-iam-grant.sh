#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
ROLE_ID="${ROLE_ID:-HydraliskDeepseekG4Runner}"
ROLE_FILE="${ROLE_FILE:-deploy/gce/deepseek-v4-g4-runner-role.yaml}"
APPLY="${APPLY:-0}"
GRANT_AUTHORITY_PREFLIGHT="${GRANT_AUTHORITY_PREFLIGHT:-1}"

role_name="projects/$PROJECT_ID/roles/$ROLE_ID"
GRANT_REQUIRED_PROJECT_PERMISSIONS=(
  iam.roles.create
  iam.roles.update
  resourcemanager.projects.get
  resourcemanager.projects.getIamPolicy
  resourcemanager.projects.setIamPolicy
)
GRANT_REQUIRED_SERVICE_ACCOUNT_PERMISSIONS=(
  iam.serviceAccounts.get
  iam.serviceAccounts.getIamPolicy
  iam.serviceAccounts.setIamPolicy
)

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

permissions_json() {
  python3 -c 'import json, sys; print(json.dumps({"permissions": sys.argv[1:]}))' "$@"
}

test_permissions() {
  local resource_url="$1"
  local output_file="$2"
  shift 2

  local token
  token="$(run_gcloud auth print-access-token)"
  curl -fsS -X POST "$resource_url:testIamPermissions" \
    -H "Authorization: Bearer $token" \
    -H 'Content-Type: application/json' \
    -d "$(permissions_json "$@")" \
    > "$output_file"
  unset token
}

missing_permissions() {
  local response_file="$1"
  shift
  python3 - "$response_file" "$@" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text() or "{}")
present = set(response.get("permissions") or [])
missing = [permission for permission in sys.argv[2:] if permission not in present]
print("\n".join(missing))
PY
}

default_compute_service_account() {
  local project_number
  if ! project_number="$(run_gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"; then
    return 1
  fi
  printf '%s-compute@developer.gserviceaccount.com\n' "$project_number"
}

grant_authority_preflight() {
  if [[ "$GRANT_AUTHORITY_PREFLIGHT" != "1" ]]; then
    return 0
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/hydralisk-grant-preflight.XXXXXX")"
  local project_response="$tmp_dir/project-iam.json"
  local service_account_response="$tmp_dir/service-account-iam.json"
  local missing_file="$tmp_dir/missing.txt"
  local service_account
  if ! service_account="$(default_compute_service_account 2> "$tmp_dir/default-compute-service-account.err")"; then
    echo "blocked_grant_auth: current gcloud account cannot resolve the default Compute Engine service account for $PROJECT_ID" >&2
    echo >&2
    sed -n '1,12p' "$tmp_dir/default-compute-service-account.err" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  if ! test_permissions \
    "https://cloudresourcemanager.googleapis.com/v1/projects/$PROJECT_ID" \
    "$project_response" \
    "${GRANT_REQUIRED_PROJECT_PERMISSIONS[@]}" 2> "$tmp_dir/project-test-iam-permissions.err"; then
    echo "blocked_grant_auth: current gcloud account could not call project testIamPermissions" >&2
    echo >&2
    sed -n '1,12p' "$tmp_dir/project-test-iam-permissions.err" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  missing_permissions "$project_response" "${GRANT_REQUIRED_PROJECT_PERMISSIONS[@]}" \
    > "$missing_file"

  test_permissions \
    "https://iam.googleapis.com/v1/projects/$PROJECT_ID/serviceAccounts/$service_account" \
    "$service_account_response" \
    "${GRANT_REQUIRED_SERVICE_ACCOUNT_PERMISSIONS[@]}" || printf '{}\n' > "$service_account_response"

  missing_permissions "$service_account_response" "${GRANT_REQUIRED_SERVICE_ACCOUNT_PERMISSIONS[@]}" \
    >> "$missing_file"

  if [[ -s "$missing_file" ]]; then
    echo "blocked_grant_iam: current gcloud account is missing permissions needed to apply the DeepSeek G4 runner IAM grant" >&2
    echo >&2
    echo "Missing grant permissions:" >&2
    sed 's/^/- /' "$missing_file" >&2
    echo >&2
    echo "Run this helper from an IAM-capable account, or grant these permissions first." >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  rm -rf "$tmp_dir"
}

print_plan() {
  cat <<EOF
# Hydralisk DeepSeek-V4-Flash G4 IAM grant plan

Project: $PROJECT_ID
Runner service account: $SERVICE_ACCOUNT
gcloud account override: ${GCLOUD_ACCOUNT:-default}
Custom role id: $ROLE_ID
Custom role file: $ROLE_FILE
Grant authority preflight: $GRANT_AUTHORITY_PREFLIGHT

## 0. Grant-authority preflight

With APPLY=1, this helper first checks that the current gcloud account has:

Project permissions:

$(printf '  - %s\n' "${GRANT_REQUIRED_PROJECT_PERMISSIONS[@]}")

Default Compute Engine service-account policy permissions:

$(printf '  - %s\n' "${GRANT_REQUIRED_SERVICE_ACCOUNT_PERMISSIONS[@]}")

## 1. Create or update the custom project role

if gcloud iam roles describe "$ROLE_ID" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam roles update "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
else
  gcloud iam roles create "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
fi

## 2. Bind the custom role to the runner service account

gcloud projects add-iam-policy-binding "$PROJECT_ID" \\
  --member "serviceAccount:$SERVICE_ACCOUNT" \\
  --role "$role_name"

## 3. Allow OS Login over gcloud compute ssh

gcloud projects add-iam-policy-binding "$PROJECT_ID" \\
  --member "serviceAccount:$SERVICE_ACCOUNT" \\
  --role "roles/compute.osAdminLogin"

## 4. If the VM uses the default Compute Engine service account, allow actAs

PROJECT_NUMBER="\$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
DEFAULT_COMPUTE_SA="\${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding "\$DEFAULT_COMPUTE_SA" \\
  --project "$PROJECT_ID" \\
  --member "serviceAccount:$SERVICE_ACCOUNT" \\
  --role "roles/iam.serviceAccountUser"

## 5. Rerun the canonical DeepSeek G4 smoke

GCLOUD_ACCOUNT="$SERVICE_ACCOUNT" \\
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh

Set APPLY=1 to execute steps 1-4 from this helper. The smoke command is always
printed but not executed by this helper.
EOF
}

apply_plan() {
  if run_gcloud iam roles describe "$ROLE_ID" --project "$PROJECT_ID" >/dev/null 2>&1; then
    run_gcloud iam roles update "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
  else
    run_gcloud iam roles create "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
  fi

  run_gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "$role_name"

  run_gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "roles/compute.osAdminLogin"

  local project_number
  project_number="$(run_gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  local default_compute_sa="${project_number}-compute@developer.gserviceaccount.com"
  run_gcloud iam service-accounts add-iam-policy-binding "$default_compute_sa" \
    --project "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "roles/iam.serviceAccountUser"
}

print_plan

if [[ "$APPLY" = "1" ]]; then
  grant_authority_preflight
  apply_plan
fi
