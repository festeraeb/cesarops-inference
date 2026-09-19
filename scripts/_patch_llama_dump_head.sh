#!/bin/bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
p = Path("/codebase/repos/wreckhunter2000-1/llama.cpp/common/debug.cpp")
text = p.read_text()
# Force print-first-16 consecutive along dim0 for the first token/row only,
# when CESAROPS_DUMP_HEAD=1 (avoids head/tail ellipsis which breaks compares).
old = '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \\n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \\n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\\n");
            }
            LOG(INDENT INDENT "],\\n");
        }
        LOG(INDENT "]\\n");
        LOG(INDENT "sum = %f\\n", sum);
    }

    if (abort_on_nan) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\\n");
            exit(0);
        }
    }
}'''

new = '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    const char * dump_head = std::getenv("CESAROPS_DUMP_HEAD");
    if (dump_head && dump_head[0] == '1') {
        // Print first HEAD floats of token/row 0 along dim0 (consecutive).
        const int64_t head = 16;
        LOG(INDENT "CESAROPS_HEAD dim0[:%lld] (ne=%lld,%lld,%lld,%lld) [",
            (long long) head, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        const int64_t lim = head < ne[0] ? head : ne[0];
        for (int64_t i0 = 0; i0 < lim; i0++) {
            const float v = common_ggml_get_float_value(data, type, nb, i0, 0, 0, 0);
            LOG("%12.6f", v);
            if (i0 + 1 < lim) LOG(", ");
        }
        LOG("]\\n");
        return;
    }
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \\n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \\n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\\n");
            }
            LOG(INDENT INDENT "],\\n");
        }
        LOG(INDENT "]\\n");
        LOG(INDENT "sum = %f\\n", sum);
    }

    if (abort_on_nan) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\\n");
            exit(0);
        }
    }
}'''

# The file uses real newlines; use a surgical insert instead.
needle = 'static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {\n    GGML_ASSERT(n > 0);\n    float sum = 0;'
insert = '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    const char * dump_head = std::getenv("CESAROPS_DUMP_HEAD");
    if (dump_head && dump_head[0] == '1') {
        const int64_t head = 16;
        LOG(INDENT "CESAROPS_HEAD dim0[:%lld] (ne=%lld,%lld,%lld,%lld) [",
            (long long) head, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        const int64_t lim = head < ne[0] ? head : ne[0];
        for (int64_t i0 = 0; i0 < lim; i0++) {
            const float v = common_ggml_get_float_value(data, type, nb, i0, /*i1*/0, /*i2*/0, /*i3*/0);
            LOG("%12.6f", v);
            if (i0 + 1 < lim) LOG(", ");
        }
        LOG("]\\n");
        return;
    }
    float sum = 0;'''
# fix escaped newline in LOG
insert = insert.replace('LOG("]\\n");', 'LOG("]\\n");')  # noop keep
insert = insert.replace('LOG("]\\n");', 'LOG("]");\n        LOG("\\n");') 
# Actually just use proper C string
insert = '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    const char * dump_head = std::getenv("CESAROPS_DUMP_HEAD");
    if (dump_head && dump_head[0] == '1') {
        const int64_t head = 16;
        LOG(INDENT "CESAROPS_HEAD dim0[:%lld] (ne=%lld,%lld,%lld,%lld) [",
            (long long) head, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        const int64_t lim = head < ne[0] ? head : ne[0];
        for (int64_t i0 = 0; i0 < lim; i0++) {
            const float v = common_ggml_get_float_value(data, type, nb, i0, 0, 0, 0);
            LOG("%12.6f", v);
            if (i0 + 1 < lim) LOG(", ");
        }
        LOG("]\\n");
        return;
    }
    float sum = 0;'''

# C++ needs real \n in source for the format string end - use:
insert = insert.replace('LOG("]\\n");', 'LOG("]\\n");')

if needle not in text:
    if 'CESAROPS_DUMP_HEAD' in text:
        print('already patched')
    else:
        raise SystemExit('needle missing')
else:
    # Fix the LOG line to be valid C++
    insert_fixed = '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    const char * dump_head = std::getenv("CESAROPS_DUMP_HEAD");
    if (dump_head && dump_head[0] == '1') {
        const int64_t head = 16;
        LOG(INDENT "CESAROPS_HEAD dim0[:%lld] (ne=%lld,%lld,%lld,%lld) [",
            (long long) head, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        const int64_t lim = head < ne[0] ? head : ne[0];
        for (int64_t i0 = 0; i0 < lim; i0++) {
            const float v = common_ggml_get_float_value(data, type, nb, i0, 0, 0, 0);
            LOG("%12.6f", v);
            if (i0 + 1 < lim) LOG(", ");
        }
        LOG("]\\n");
        return;
    }
    float sum = 0;'''
    # In the written file, `\\n` in the python string becomes `\n` in C source which is wrong inside quotes.
    # We need the C source to contain: LOG("]\n");
    insert_fixed = insert_fixed.replace('LOG("]\\n");', 'LOG("]" "\\n");')
    text = text.replace(needle, insert_fixed, 1)
    # Ensure cstdlib for getenv
    if '#include <cstdlib>' not in text:
        text = text.replace('#include <cmath>', '#include <cmath>\n#include <cstdlib>')
    p.write_text(text)
    print('patched print_tensor head dump')
PY

cd /codebase/repos/wreckhunter2000-1/llama.cpp/build-vulkan
cmake --build . --target llama-debug -j8 2>&1 | tail -15
