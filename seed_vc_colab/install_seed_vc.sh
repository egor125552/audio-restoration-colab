#!/usr/bin/env bash
set -euo pipefail

SEED_VC_HOME="${1:-${SEED_VC_HOME:-/content/seed-vc}}"
UPSTREAM_URL="https://github.com/Plachtaa/seed-vc.git"
UPSTREAM_COMMIT="51383efd921027683c89e5348211d93ff12ac2a8"
PYTHON_VERSION="3.10"
VENV_DIR="${SEED_VC_HOME}/.venv"
SOURCE_DIR="${SEED_VC_HOME}/src"
NORMALIZED_REQUIREMENTS="${SEED_VC_HOME}/requirements.colab.txt"

log() {
  printf '\n[seed-vc-colab] %s\n' "$*"
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

mkdir -p "${SEED_VC_HOME}"
ensure_command git git
ensure_command ffmpeg ffmpeg
ensure_command curl curl

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  python3 -m pip install --quiet --disable-pip-version-check uv
fi

log "Preparing Python ${PYTHON_VERSION}"
uv python install "${PYTHON_VERSION}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  rm -rf "${SOURCE_DIR}"
  log "Cloning Seed-VC"
  git clone --quiet "${UPSTREAM_URL}" "${SOURCE_DIR}"
fi

log "Pinning upstream commit ${UPSTREAM_COMMIT}"
git -C "${SOURCE_DIR}" fetch --quiet origin "${UPSTREAM_COMMIT}"
git -C "${SOURCE_DIR}" checkout --quiet --detach "${UPSTREAM_COMMIT}"

log "Normalizing upstream requirements"
"${PYTHON}" - "${SOURCE_DIR}/requirements.txt" "${NORMALIZED_REQUIREMENTS}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
lines = source.read_text(encoding="utf-8").splitlines()

# The archived upstream file contains three nightly CUDA requirements and then
# pins torch/torchvision/torchaudio again to 2.4.0. Installing both is not a
# useful compatibility test: it asks one environment for two PyTorch lines.
# Keep the stable pins and every other upstream dependency unchanged.
skip_prefixes = (
    "torch --pre --index-url ",
    "torchvision --pre --index-url ",
    "torchaudio --pre --index-url ",
)
normalized = [line for line in lines if not line.startswith(skip_prefixes)]
normalized.append("setuptools<81")
target.write_text("\n".join(normalized) + "\n", encoding="utf-8")

removed = [line for line in lines if line.startswith(skip_prefixes)]
if len(removed) != 3:
    raise SystemExit(
        f"Upstream requirements changed: expected exactly 3 duplicate nightly lines, got {len(removed)}"
    )
print("Removed only duplicate nightly PyTorch lines:")
for line in removed:
    print("  ", line)
PY

log "Installing every normalized Seed-VC dependency"
uv pip install --python "${PYTHON}" --upgrade "pip<26" "setuptools<81" wheel
uv pip install --python "${PYTHON}" -r "${NORMALIZED_REQUIREMENTS}"

log "Checking dependency consistency"
"${PYTHON}" -m pip check

cat > "${SEED_VC_HOME}/INSTALLATION.txt" <<EOF
Seed-VC source: ${UPSTREAM_URL}
Pinned commit: ${UPSTREAM_COMMIT}
Python: $(${PYTHON} --version 2>&1)
Virtualenv: ${VENV_DIR}
Source: ${SOURCE_DIR}
Normalized requirements: ${NORMALIZED_REQUIREMENTS}
EOF

log "Seed-VC environment is ready at ${SEED_VC_HOME}"
