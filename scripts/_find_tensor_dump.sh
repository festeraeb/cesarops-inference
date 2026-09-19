#!/bin/bash
set -euo pipefail
ROOT=/codebase/repos/wreckhunter2000-1/llama.cpp
grep -rn "tensor_filter" "$ROOT/common" "$ROOT/examples/debug" "$ROOT/include" "$ROOT/src" 2>/dev/null | head -40
echo '===='
grep -rn "print_tensor\|dump_tensor\|ggml_print\|tensor_to_str" "$ROOT/common" "$ROOT/examples/debug" 2>/dev/null | head -40
