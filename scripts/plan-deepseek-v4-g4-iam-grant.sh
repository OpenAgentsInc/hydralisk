#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com}"
ROLE_ID="${ROLE_ID:-HydraliskDeepseekG4Runner}"
ROLE_FILE="${ROLE_FILE:-deploy/gce/deepseek-v4-g4-runner-role.yaml}"
APPLY="${APPLY:-0}"

role_name="projects/$PROJECT_ID/roles/$ROLE_ID"

print_plan() {
  cat <<EOF
# Hydralisk DeepSeek-V4-Flash G4 IAM grant plan

Project: $PROJECT_ID
Runner service account: $SERVICE_ACCOUNT
Custom role id: $ROLE_ID
Custom role file: $ROLE_FILE

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
  if gcloud iam roles describe "$ROLE_ID" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam roles update "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
  else
    gcloud iam roles create "$ROLE_ID" --project "$PROJECT_ID" --file "$ROLE_FILE"
  fi

  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "$role_name"

  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "roles/compute.osAdminLogin"

  local project_number
  project_number="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  local default_compute_sa="${project_number}-compute@developer.gserviceaccount.com"
  gcloud iam service-accounts add-iam-policy-binding "$default_compute_sa" \
    --project "$PROJECT_ID" \
    --member "serviceAccount:$SERVICE_ACCOUNT" \
    --role "roles/iam.serviceAccountUser"
}

print_plan

if [[ "$APPLY" = "1" ]]; then
  apply_plan
fi
