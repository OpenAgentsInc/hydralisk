# DeepSeek-V4-Flash Tailnet alternate executor check

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/44

## Summary

Issue #44 checked whether another Tailnet-connected machine could run the
canonical issue #41 DeepSeek G4 smoke while this Mac's local gcloud credentials
require interactive reauthentication.

The workspace Tailnet SSH runbook was read before probing. The runbook's
route-around guidance is relevant, but no usable alternate executor was found
from this shell.

## Probes

Local Tailscale CLI:

```text
/opt/homebrew/bin/tailscale status -> failed to connect to local Tailscale service
/opt/homebrew/bin/tailscale ping archlinux -> failed to connect to local Tailscale service
```

Direct SSH checks:

```text
christopherdavid@archlinux -> connection refused on port 22
christopherdavid@100.108.56.85 -> timed out on port 22
christopherdavid@100.72.151.98 -> timed out on port 22
christopherdavid@100.97.233.57 -> timed out on port 22
christopherdavid@imac-pro-bertha -> hostname did not resolve
christopherdavid@macbook-pro-m2 -> hostname did not resolve
```

All SSH probes used `BatchMode=yes` and a short connect timeout so they would
not prompt for passwords or hang.

## Plain-English read

There is currently no alternate Tailnet machine available from this shell that
can run Hydralisk's issue #41 G4 smoke. This does not say those machines are
permanently unusable; it only says they were not reachable with noninteractive
SSH in this run.

The current live DeepSeek path is still:

```bash
gcloud auth login
gcloud auth application-default login
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

Alternative unblock: bring `archlinux`, `macbook-pro-m2`, or
`imac-pro-bertha` online with SSH and valid gcloud auth, then rerun the same
wrapper from a Hydralisk checkout on that host.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
