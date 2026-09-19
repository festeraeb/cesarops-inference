#!/bin/bash
set -euo pipefail
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
CARGO_TARGET_DIR=/data/cargo-target /home/cesarops/.cargo/bin/cargo build --release --bin cesarops-inference --bin run_gemma4
echo BUILD_OK
