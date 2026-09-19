#!/bin/bash
set -euo pipefail
F=/codebase/repos/wreckhunter2000-1/llama.cpp/common/debug.cpp
python3 - <<'PY'
from pathlib import Path
p = Path("/codebase/repos/wreckhunter2000-1/llama.cpp/common/debug.cpp")
text = p.read_text()
old = '''    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = { 0 };'''
new = '''    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    if (ask) {
        // Only request host readback for tensors we will print.
        // Full-graph readback on 26B MoE is prohibitively slow.
        return matches_filter;
    }

    if (!matches_filter) {
        return true;
    }

    char src1_str[128] = { 0 };'''
if old not in text:
    raise SystemExit('pattern not found — already patched?')
p.write_text(text.replace(old, new, 1))
print('patched ok')
PY

cd /codebase/repos/wreckhunter2000-1/llama.cpp/build-vulkan
cmake --build . --target llama-debug -j8 2>&1 | tail -20
ls -la bin/llama-debug
