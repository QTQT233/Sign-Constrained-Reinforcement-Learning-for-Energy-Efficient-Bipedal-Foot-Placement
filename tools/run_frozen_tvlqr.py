#!/usr/bin/env python3
"""Run frozen Paper2 TVLQR entry points without editing archived source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tvlqr_release.json"
NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_case_id(case_id: str) -> str:
    return (
        case_id.replace("flat_1.145", "flat_L1145")
        .replace("raised_1.145", "raised_L1145")
        .replace("flat_1.28", "flat_L128")
        .replace("raised_1.28", "raised_L128")
    )


def load_release() -> tuple[dict, list[dict], dict[str, str]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["case_manifest"]).open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    with (ROOT / config["source_audit"]).open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    source_hashes = {
        row["case_id"]: row["source_sha256"]
        for row in audit_rows
        if row["method"] == "TVLQR tracking" and row["script"] == "LQR.py"
    }
    source_hashes.update(config.get("source_hash_overrides", {}))
    return config, cases, source_hashes


def verify_release(config: dict, cases: list[dict], source_hashes: dict[str, str]) -> None:
    errors: list[str] = []
    for basename, record in config["models"].items():
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"missing model: {path}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"model hash mismatch: {basename}")
    for case in cases:
        path = ROOT / config["entrypoint_pattern"].format(case_id=case["case_id"])
        expected = source_hashes.get(audit_case_id(case["case_id"]))
        if not path.is_file():
            errors.append(f"missing source: {path}")
        elif not expected:
            errors.append(f"missing audit hash: {case['case_id']}")
        elif sha256(path) != expected:
            errors.append(f"source hash mismatch: {case['case_id']}")
    if errors:
        raise RuntimeError("\n".join(errors))


def portable_source(source: Path, config: dict, seed: int | None) -> str:
    text = source.read_text(encoding="utf-8", errors="strict")
    replacement_count = 0
    for basename, record in config["models"].items():
        model_path = (ROOT / record["path"]).resolve()
        pattern = rf"policy_path\s*=\s*['\"][^'\"]*{re.escape(basename)}['\"]"
        replacement = f"policy_path = {str(model_path)!r}"
        text, count = re.subn(pattern, lambda _match, value=replacement: value, text)
        replacement_count += count
    if replacement_count != 2 or "D:/L&S/Mas/Project/Paper2" in text:
        raise RuntimeError(f"expected two model-path substitutions in {source}, found {replacement_count}")
    if seed is not None:
        marker = "import numpy as np"
        injected = f"{marker}\n\nnp.random.seed({seed})\ntorch.manual_seed({seed})"
        if marker not in text:
            raise RuntimeError("could not find RNG injection marker")
        text = text.replace(marker, injected, 1)
    return text


def parse_cmt(stdout: str) -> str:
    matches = re.findall(rf"LQR[^\r\n]*?Cmt\s*=\s*({NUMBER})", stdout, flags=re.I)
    return matches[-1] if matches else ""


def sanitize_process_text(text: str, temp_source: Path) -> str:
    """Remove the random workstation temporary path from retained logs."""
    sanitized = text
    for value in {str(temp_source), temp_source.as_posix()}:
        sanitized = sanitized.replace(value, "<TEMP>/LQR.py")
    return sanitized


def run_case(case: dict, config: dict, output_dir: Path, timeout: int, seed: int | None) -> dict:
    source = ROOT / config["entrypoint_pattern"].format(case_id=case["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tvlqr_") as temporary:
        temp_source = Path(temporary) / "LQR.py"
        temp_source.write_text(portable_source(source, config, seed), encoding="utf-8")
        env = os.environ.copy()
        conda_prefix = Path(sys.executable).resolve().parent
        conda_paths = [
            conda_prefix,
            conda_prefix / "Library" / "mingw-w64" / "bin",
            conda_prefix / "Library" / "usr" / "bin",
            conda_prefix / "Library" / "bin",
            conda_prefix / "Scripts",
            conda_prefix / "bin",
            conda_prefix.parent.parent / "condabin",
        ]
        env.update(
            {
                "MPLBACKEND": "Agg",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
                "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
                "PATH": os.pathsep.join(str(path) for path in conda_paths)
                + os.pathsep
                + env.get("PATH", ""),
            }
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-B", str(temp_source)],
                cwd=temporary,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            status = "ok" if completed.returncode == 0 else "error"
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nRUNNER TIMEOUT after {timeout} seconds.\n"
            status, returncode = "timeout", ""
        stdout = sanitize_process_text(stdout, temp_source)
        stderr = sanitize_process_text(stderr, temp_source)
        elapsed = time.perf_counter() - started
    stdout_path = output_dir / f"{case['case_id']}.stdout.txt"
    stderr_path = output_dir / f"{case['case_id']}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "case_id": case["case_id"],
        "step_height_m": case["step_height_m"],
        "nominal_length_group": case["nominal_length_group"],
        "replicate_label": case["replicate_label"],
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": f"{elapsed:.6f}",
        "seed_override": "" if seed is None else seed,
        "cmt": parse_cmt(stdout),
        "source_sha256": sha256(source),
        "stdout": stdout_path.relative_to(ROOT).as_posix() if stdout_path.is_relative_to(ROOT) else str(stdout_path),
        "stderr": stderr_path.relative_to(ROOT).as_posix() if stderr_path.is_relative_to(ROOT) else str(stderr_path),
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(values: list[float]) -> dict[str, str | int]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    result: dict[str, str | int] = {
        "n": n,
        "mean_cmt": f"{mean:.12f}",
        "sample_sd": "" if sd is None else f"{sd:.12f}",
    }
    if n == 12 and sd is not None:
        half_width = 2.200985160082949 * sd / (n**0.5)
        result["ci95_low"] = f"{mean - half_width:.12f}"
        result["ci95_high"] = f"{mean + half_width:.12f}"
    else:
        result["ci95_low"] = ""
        result["ci95_high"] = ""
    return result


def write_release_outputs(output: Path, rows: list[dict], config: dict, seed: int | None, timeout: int) -> None:
    write_summary(output / "tvlqr_cases.csv", rows)
    groups = [
        ("all", rows),
        ("flat_1.280", [row for row in rows if row["step_height_m"] == "0" and row["nominal_length_group"] == "1.28"]),
        ("flat_1.145", [row for row in rows if row["step_height_m"] == "0" and row["nominal_length_group"] == "1.145"]),
        ("raised_0.01_1.280", [row for row in rows if row["step_height_m"] == "0.01" and row["nominal_length_group"] == "1.28"]),
        ("raised_0.01_1.145", [row for row in rows if row["step_height_m"] == "0.01" and row["nominal_length_group"] == "1.145"]),
    ]
    group_rows = []
    for label, selected in groups:
        values = [float(row["cmt"]) for row in selected if row["status"] == "ok" and row["cmt"]]
        if not values:
            continue
        group_rows.append({"group": label, **numeric_summary(values)})
    write_summary(output / "tvlqr_group_summary.csv", group_rows)
    manifest = {
        "runner": Path(__file__).relative_to(ROOT).as_posix(),
        "python": sys.version,
        "configured_environment": config["validated_environment"],
        "seed_override": seed,
        "timeout_seconds_per_case": timeout,
        "case_count": len(rows),
        "ok_count": sum(row["status"] == "ok" and bool(row["cmt"]) for row in rows),
        "source_and_model_hashes_verified": True,
        "runner_modified_sources": False,
        "source_hash_overrides": config.get("source_hash_overrides", {}),
    }
    (output / "tvlqr_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", help="case_id from configs/paper2_cases.csv")
    selection.add_argument("--all", action="store_true", help="run all 12 cases")
    parser.add_argument("--check-only", action="store_true", help="verify all sources and models without running")
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed injected only into a temporary copy")
    parser.add_argument("--timeout", type=int, default=600, help="per-case timeout in seconds")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tvlqr_rerun")
    args = parser.parse_args()

    config, cases, source_hashes = load_release()
    verify_release(config, cases, source_hashes)
    print(f"verified {len(cases)} frozen TVLQR sources and {len(config['models'])} checkpoints")
    if args.check_only or (not args.case and not args.all):
        return 0
    selected = cases if args.all else [case for case in cases if case["case_id"] == args.case]
    if not selected:
        parser.error(f"unknown case: {args.case}")
    output = args.output.resolve()
    rows = []
    for case in selected:
        print(f"RUN {case['case_id']}", flush=True)
        row = run_case(case, config, output, args.timeout, args.seed)
        rows.append(row)
        print(f"DONE {row['status']} Cmt={row['cmt'] or 'NA'}", flush=True)
    write_release_outputs(output, rows, config, args.seed, args.timeout)
    return 0 if all(row["status"] == "ok" and row["cmt"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
