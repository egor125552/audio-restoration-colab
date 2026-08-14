#!/usr/bin/env bash
set -euo pipefail

XVC_HOME="${1:-${XVC_HOME:-/content/x-vc}}"
UPSTREAM_URL="https://github.com/Jerrister/X-VC.git"
UPSTREAM_COMMIT="49df8c591eafc48b096e466d96f9839f9c0dd739"
PYTHON_VERSION="3.10"
VENV_DIR="${XVC_HOME}/.venv"
SOURCE_DIR="${XVC_HOME}/src"

log() {
  printf '\n[x-vc-colab] %s\n' "$*"
}

ensure_command() {
  local command_name="$1"
  local package_name="$2"
  if command -v "${command_name}" >/dev/null 2>&1; then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Missing ${command_name}; apt-get is unavailable." >&2
    exit 1
  fi
  log "Installing system package ${package_name}"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${package_name}"
}

mkdir -p "${XVC_HOME}"
ensure_command git git
ensure_command ffmpeg ffmpeg
ensure_command curl curl

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  python3 -m pip install --quiet --disable-pip-version-check "uv==0.12.1"
fi

log "Preparing Python ${PYTHON_VERSION}"
uv python install "${PYTHON_VERSION}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  rm -rf "${SOURCE_DIR}"
  log "Cloning official X-VC"
  git clone --quiet "${UPSTREAM_URL}" "${SOURCE_DIR}"
fi

log "Pinning official X-VC commit ${UPSTREAM_COMMIT}"
git -C "${SOURCE_DIR}" fetch --quiet origin "${UPSTREAM_COMMIT}"
git -C "${SOURCE_DIR}" checkout --quiet --detach "${UPSTREAM_COMMIT}"

log "Installing official X-VC dependencies"
uv pip install --python "${PYTHON}" --upgrade "pip<26" "setuptools<81" wheel
uv pip install --python "${PYTHON}" -r "${SOURCE_DIR}/requirements.txt"

log "Installing Colab UI/download helpers compatible with X-VC Transformers"
uv pip install --python "${PYTHON}" \
  "gradio==5.49.1" \
  "huggingface-hub==0.36.2" \
  "modelscope==1.29.2"

log "Checking dependency consistency"
"${PYTHON}" -m pip check

cat > "${XVC_HOME}/INSTALLATION.txt" <<EOF
X-VC source: ${UPSTREAM_URL}
Pinned commit: ${UPSTREAM_COMMIT}
Python: $(${PYTHON} --version 2>&1)
Virtualenv: ${VENV_DIR}
Source: ${SOURCE_DIR}
EOF

log "X-VC environment is ready at ${XVC_HOME}"
