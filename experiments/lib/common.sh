#!/usr/bin/env bash

# Shared shell helpers for LatentASR experiment runners.
# This file is sourced by scripts under experiments/.

latent_asr_repo_root() {
  local script_dir="$1"
  cd "${script_dir}/.." && pwd
}

latent_asr_python_bin() {
  local requested="${1:-}"
  if [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
  elif [[ -x "/user_data/miniconda3/envs/py311/bin/python" ]]; then
    printf '%s\n' "/user_data/miniconda3/envs/py311/bin/python"
  else
    printf '%s\n' "python"
  fi
}

latent_asr_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[error] ${label} not found: ${path}" >&2
    exit 1
  fi
}

latent_asr_print_kv() {
  local key="$1"
  local value="$2"
  printf '%-16s: %s\n' "${key}" "${value}"
}
