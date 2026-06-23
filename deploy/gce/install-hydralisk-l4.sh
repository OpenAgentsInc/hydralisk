#!/usr/bin/env bash

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/opt/hydralisk}"
ENV_FILE="${ENV_FILE:-/etc/hydralisk/hydralisk.env}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: run as root on the Hydralisk host" >&2
  exit 1
fi

install -d -o root -g root -m 0755 /etc/hydralisk /var/lib/hydralisk/receipts /var/log/hydralisk
if ! id hydralisk >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/hydralisk --shell /usr/sbin/nologin hydralisk
fi
chown -R hydralisk:hydralisk /var/lib/hydralisk /var/log/hydralisk

apt-get update -y
apt-get install -y ca-certificates curl git jq

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:${PATH}"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/OpenAgentsInc/hydralisk.git "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin main
  git -C "${REPO_DIR}" checkout --force origin/main
fi

cd "${REPO_DIR}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv-python}"
uv python install 3.12
uv venv --python 3.12 --seed
uv pip install vllm --torch-backend=auto
uv pip install .
chmod -R a+rX "${UV_PYTHON_INSTALL_DIR}" "${REPO_DIR}/.venv"
nvidia_lib_path="$(find "${REPO_DIR}/.venv/lib" -path '*/site-packages/nvidia/*/lib' -type d | paste -sd: -)"
engine_version="$(LD_LIBRARY_PATH="${nvidia_lib_path}:${LD_LIBRARY_PATH:-}" "${REPO_DIR}/.venv/bin/vllm" --version | awk '{print $NF}')"

if [[ ! -f "${ENV_FILE}" ]]; then
  install -o root -g hydralisk -m 0640 /dev/null "${ENV_FILE}"
  cat >"${ENV_FILE}" <<ENV
HYDRALISK_SERVED_MODEL=openai/gpt-oss-20b
HYDRALISK_PUBLIC_MODEL_ALIASES=openagents/khala-oss-20b,gpt-oss-20b
HYDRALISK_VLLM_BASE_URL=http://127.0.0.1:8000
HYDRALISK_RECEIPT_DIR=/var/lib/hydralisk/receipts
HYDRALISK_ENGINE_VERSION=${engine_version}
HYDRALISK_GPU_NAME=NVIDIA L4
HYDRALISK_GPU_COUNT=1
HYDRALISK_QUANTIZATION_WEIGHTS=MXFP4
HYDRALISK_MAX_OUTPUT_TOKENS=1024
LD_LIBRARY_PATH=${nvidia_lib_path}
# Set this out-of-band before enabling the proxy:
# HYDRALISK_BEARER_TOKEN=
ENV
fi

install -o root -g root -m 0644 deploy/systemd/vllm-gpt-oss-20b.service /etc/systemd/system/vllm-gpt-oss-20b.service
install -o root -g root -m 0644 deploy/systemd/hydralisk-proxy.service /etc/systemd/system/hydralisk-proxy.service
systemctl daemon-reload
systemctl enable vllm-gpt-oss-20b.service hydralisk-proxy.service

echo "Installed Hydralisk. Add HYDRALISK_BEARER_TOKEN to ${ENV_FILE}, then run:"
echo "  systemctl start vllm-gpt-oss-20b.service"
echo "  systemctl start hydralisk-proxy.service"
