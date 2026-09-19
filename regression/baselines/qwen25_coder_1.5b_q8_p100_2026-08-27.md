# Regression baseline — Qwen2.5-Coder-1.5B on T440 P100

**Status:** PASS — lock this build as the Qwen native-generate regression point.

## Identity

| Field | Value |
|---|---|
| Date (UTC) | 2026-08-27 |
| Host | T440 (`Tesla P100-PCIE-16GB` index 0) |
| Git HEAD (committed) | `01dabe777588f77dea7798b3bd4a8073aacb6b38` (`01dabe77`) |
| HEAD message | `chore: remove data and build artifacts from tracking` (2026-08-21) |
| Working tree | **dirty** — Qwen/Gemma fixes not yet committed |
| Binary | `/data/cargo-target/release/cesarops-inference` |
| Binary mtime (UTC) | 2026-08-27 05:58 |
| Binary sha256 | `90b4ff5e04e88f897a3948ce60a15b59ec39321938fc5699b195306a87f6def7` |

### Dirty source fingerprint (sha256 of files that built this binary)

```
968ebd5b9be2df1e0c584a4400fc4bde662e10f4e1ab09d3d06b555ddbb8ada0  src/main.rs
38ded51c8ff1509a0bcf6bcb97ad89b2dd3bcad1ae431285421ab2b0da6674ee  src/tensor_loader_safe.rs
963156671b27a45c50a6fd54f197f64e1133aa44b50aec01e5529e132735da82  src/gemma4_gpu_runner.rs
64582599c02991074a62c13ff3f0a343b2aaa951e3990a8b911c606db2410b5c  src/gemma4_layer.rs
4ea0080c2469fcb018956a64f2a2bc0cd83eaedfd8ef21cf9af0635e2dc4931d  src/gemma4_runner.rs
```

`git diff --stat` vs `01dabe77` for those files: `5 files, +80 / −42`.

> Re-run gate must match **binary sha256** (or rebuild from the same dirty tree / a future commit that includes these hashes). Matching HEAD alone is **not** enough until the dirty fixes are committed.

## Locked results

### Short smoke (correctness)

- Model: `/mnt/raid0/models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf`
- Prompt: `2+2=`
- Max tokens: 8
- Output: `2+2 equals 4.`
- Rate: **2.4 t/s** (7 tokens / 2.93s; includes first-token / diagnostic overhead)

### Long regression (correctness + throughput)

- Same model, GPU 0
- Prompt: `Write a short Python function that returns the factorial of n. Include a docstring and one example.`
- Max tokens: **128**
- Output: coherent recursive `factorial` with docstring + `print(factorial(5))  # Output: 120`
- Wall: **22.45s** for 128 tokens → **5.7 t/s**
- Log: `/tmp/qwen_15b_long.log` on T440 (also mirrored in agent terminal 667953)

### Fixes included in this lock (vs `01dabe77`)

1. `--gpu N` selects discrete Vulkan adapter by index (not HighPerformance lottery).
2. Storage buffer limit raised to P100 max (~2 GiB−1) so embd/LM-head f32 inflate fits.
3. K-quants always dequant to f32 before f32 matvec (`needs_dequant`).
4. Tied `token_embd` cloned to `output.weight` so LM head take_buffer cannot steal embd.
5. Load errors are hard failures (no silent `_`).

## Repro

```bash
sudo systemctl stop cesarops-ornith
/data/cargo-target/release/cesarops-inference generate \
  --model /mnt/raid0/models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf \
  --gpu 0 \
  --prompt "Write a short Python function that returns the factorial of n. Include a docstring and one example." \
  --max-tokens 128
# expect: ~5.7 t/s, valid Python factorial
sudo systemctl start cesarops-ornith
```

Pass criteria for future optimization:

- Text remains correct factorial-class answer (no NaN / `!!!!` / empty).
- Sustained rate ≥ **5.0 t/s** on this prompt/length on P100 #0 (allow 10% noise).
- Binary sha256 either matches lock **or** is explicitly annotated as a new baseline.

## Related Gemma note (not a PASS lock)

- Model: CelesteImperia `Gemma-4-26B-MoE-IQ4_XS` → `/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf`
- Binary: `/data/cargo-target/release/run_gemma4` sha256 `6d28c87abffd6c0fbcf803e4b7e6a315030d9620d42e84c30d3be91ecd3711d7`
- Loads; no `scratch.q` panic; **output still wrong** for `3x3=` (finite logits, wrong tokens). See companion notes in chat / `docs/GPU_RESIDENT_RUNNER_REPORT.md` §3.
