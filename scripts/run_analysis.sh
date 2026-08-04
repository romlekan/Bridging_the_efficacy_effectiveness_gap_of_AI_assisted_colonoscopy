#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python -m colonoscopy_fidelity.cli analyze --root "$project_root" --output "$project_root/outputs"
