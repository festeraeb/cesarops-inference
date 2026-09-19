#!/bin/bash
set -euo pipefail
grep -nE 'printf|LOG_|tensor|filter|verbose|ggml_debug|callback' \
  /codebase/repos/wreckhunter2000-1/llama.cpp/examples/debug/debug.cpp | head -80
