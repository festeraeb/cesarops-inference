You are a Python specialist. Verify the argparse contracts of three SAR/wreck-detection scripts and report any mismatches with our Rust forge tool wrappers.

## Background

We just wired forge tools that shell out to these Python scripts. Need to verify the CLI contracts match what we're sending. If any don't, we need to know exactly what to fix (in the Rust wrapper or the Python script).

## Scripts to verify

All at `/mnt/data-external/cesarops/repo/`:

### 1. scan_engine.py
Our Rust wrapper invokes:
```
python /mnt/data-external/cesarops/repo/scan_engine.py \
  --bbox <lat_min,lon_min,lat_max,lon_max> \
  --days <int> \
  --mode <wreck|sar|downed_aircraft> \
  --output <path>
```

### 2. universal_downloader.py
Our wrapper invokes:
```
python /mnt/data-external/cesarops/repo/universal_downloader.py \
  --bbox <lat_min,lon_min,lat_max,lon_max> \
  --provider <auto|sentinel|landsat|ecostress|swot> \
  --days <int> \
  --output-dir <path>
```

### 3. weather_service.py
Our wrapper invokes:
```
python /mnt/data-external/cesarops/repo/weather_service.py \
  --bbox <lat_min,lon_min,lat_max,lon_max> \
  --classify <post_storm|calm|any>
```

## Deliverable

For EACH script, report:

1. **Actual argparse args** — what flags does it really accept?
2. **Required vs optional** — what fails if missing?
3. **Output behavior** — does it print to stdout? Write a file? Both?
4. **Exit code on success vs failure**
5. **Any environment variables it requires** (API keys, paths, etc.)
6. **Mismatches with our wrapper assumption** above

If a script DOESN'T have argparse and instead reads stdin or env vars or hardcoded paths, report that as a critical mismatch with a fix recommendation.

## Mode of operation

You have access to read these files. Use the read_file tool to inspect each script's argparse setup (typically in `def main()` or `if __name__ == '__main__':`). Look at imports too — if it uses `click` or `typer` or `sys.argv` directly instead of argparse, report that.

For files larger than 3000 lines, focus on:
- The CLI argument parsing block (search for "argparse", "ArgumentParser", "@click", "@app.command")
- The main() function entry point
- Any global env var reads (search for "os.environ", "os.getenv")

## Output format

Three sections, one per script:

```
=== scan_engine.py ===
**Actual CLI args:** --foo, --bar, ...
**Required:** ...
**Optional:** ...
**Output:** stdout / file / both
**Env vars needed:** ...
**Wrapper mismatches:**
  1. We send `--mode wreck` but the script doesn't have a --mode flag
  2. ...
**Fix recommendation:** ...

=== universal_downloader.py ===
[same shape]

=== weather_service.py ===
[same shape]
```

## Constraints

- Don't write any code yet, just report findings
- If a script is missing entirely or unreadable, report that clearly
- If a script has hardcoded paths/values that should be CLI args, flag those

If you can't access /mnt/data-external/cesarops/repo/ via your tools, note that — the operator can run a verification script locally.
