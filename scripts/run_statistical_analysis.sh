#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
Rscript code/statistical_analysis.R "$project_root" "$project_root/outputs"
