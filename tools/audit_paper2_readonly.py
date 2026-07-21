"""Read-only reconstruction of Paper2 non-MPC result scripts.

The child processes run from the original case folders because the research
scripts use relative imports and relative data paths.  All captured output and
audit metadata are written under the workspace, bytecode generation is
disabled, and each source tree is snapshotted before and after execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


PYTHON = Path(sys.executable)
PAPER2_ROOT = Path("paper2_case_bundle")
OUTPUT_ROOT = Path("paper2_rerun_logs")

CASES = [
    ("flat_L1145_r1", "60,30(0.33m),1.145", "flat", "1.145", 1),
    ("flat_L1145_r2", "60,30(0.33m),1.145-2", "flat", "1.145", 2),
    ("flat_L1145_r3", "60,30(0.33m),1.145-3", "flat", "1.145", 3),
    ("raised_L1145_r1", "60,30(0.33m),0.01m,1.145", "raised_0.01m", "1.145", 1),
    ("raised_L1145_r2", "60,30(0.33m),0.01m,1.145-2", "raised_0.01m", "1.145", 2),
    ("raised_L1145_r3", "60,30(0.33m),0.01m,1.145-3", "raised_0.01m", "1.145", 3),
    ("flat_L128_r1", "60,30(0.33m),1.28", "flat", "1.28", 1),
    ("flat_L128_r2", "60,30(0.33m),1.28-2", "flat", "1.28", 2),
    ("flat_L128_r3", "60,30(0.33m),1.28-3", "flat", "1.28", 3),
    ("raised_L128_r1", "60,30(0.33m),0.01m,1.28", "raised_0.01m", "1.28", 1),
    ("raised_L128_r2", "60,30(0.33m),0.01m,1.28-2", "raised_0.01m", "1.28", 2),
    ("raised_L128_r3", "60,30(0.33m),0.01m,1.28-3", "raised_0.01m", "1.28", 3),
]

METHODS = [
    ("Proposed one-sided selector", "Qi_multi_passive_sim.py", "passive"),
    ("Discrete active PPO", "Qi_multi_ac_discrete_sim.py", "discrete"),
    ("Continuous active PPO", "Multi_continuous.py", "continuous"),
    ("LIPM COM", "LIPM.py", "lipm"),
    ("TVLQR tracking", "LQR.py", "lqr"),
]

# Conservative text-level screening.  Any hit prevents execution.  These
# scripts are simple research entry points, so a conservative false positive
# is preferable to writing into the source directory.
WRITER_PATTERNS = [
    ("open_write_mode", re.compile(r"\bopen\s*\([^\n]*,[^\n]*['\"][wax+][^'\"]*['\"]", re.I)),
    ("path_write", re.compile(r"\.(?:write_text|write_bytes|touch|mkdir|rename|replace|unlink)\s*\(", re.I)),
    ("save_api", re.compile(r"\b(?:torch|np|numpy|joblib|pickle)\.(?:save|savez|dump)\s*\(", re.I)),
    ("plot_save", re.compile(r"\b(?:savefig|imsave|imwrite|VideoWriter)\s*\(", re.I)),
    ("pandas_write", re.compile(r"\.(?:to_csv|to_json|to_pickle|to_excel|to_hdf|to_parquet)\s*\(", re.I)),
    ("hdf5_write_mode", re.compile(r"\b(?:h5py\.)?File\s*\([^\n]*,[^\n]*['\"][wax+][^'\"]*['\"]", re.I)),
    (
        "filesystem_mutation",
        re.compile(
            r"\b(?:os\.(?:makedirs|mkdir|remove|unlink|rmdir|removedirs|rename|replace)|"
            r"shutil\.(?:rmtree|copy|copy2|copyfile|copytree|move))\s*\(",
            re.I,
        ),
    ),
]

NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> Dict[str, Tuple[int, int]]:
    snapshot: Dict[str, Tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def scan_source(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [name for name, pattern in WRITER_PATTERNS if pattern.search(text)]


def last_match(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.I | re.M)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def parse_metrics(kind: str, stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    cmt_patterns = {
        "passive": rf"离散被动力矩Cmt\s*[：:]\s*({NUMBER})",
        "discrete": rf"离散主动力矩Cmt\s*[：:]\s*({NUMBER})",
        "continuous": rf"连续力矩Cmt\s*[：:]\s*({NUMBER})",
        "lipm": rf"LIPM\s*的\s*Cmt\s*=\s*({NUMBER})",
        "lqr": rf"LQR\s*的\s*Cmt\s*=\s*({NUMBER})",
    }
    cmt = last_match(cmt_patterns[kind], stdout)
    time_s = last_match(rf"^Time\s*:?[ \t]*({NUMBER})\s*$", stdout)
    foot_error = last_match(rf"^Foot_error\s*:?[ \t]*({NUMBER})\s*$", stdout)
    missing = [name for name, value in (("Cmt", cmt),) if value is None]
    notes = "" if not missing else "missing " + ", ".join(missing)
    return cmt, time_s, foot_error, notes


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def run_one(case, method, timeout_s: int) -> dict:
    case_id, folder_name, terrain, length_m, replicate = case
    method_name, script_name, kind = method
    case_dir = PAPER2_ROOT / folder_name
    source = case_dir / script_name
    stem = f"{case_id}__{kind}"
    stdout_log = OUTPUT_ROOT / f"{stem}.stdout.txt"
    stderr_log = OUTPUT_ROOT / f"{stem}.stderr.txt"

    row = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "terrain": terrain,
        "nominal_length_m": length_m,
        "replicate": replicate,
        "method": method_name,
        "script": script_name,
        "status": "",
        "returncode": "",
        "elapsed_seconds": "",
        "cmt": "",
        "time_s": "",
        "foot_error_m": "",
        "source_sha256": "",
        "source_scan_hits": "",
        "source_tree_changed": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "parse_notes": "",
    }

    if not source.is_file():
        row.update(status="missing_source", parse_notes="source file not found")
        write_text(stdout_log, "")
        write_text(stderr_log, f"Missing source: {source}\n")
        return row

    row["source_sha256"] = sha256_file(source)
    hits = scan_source(source)
    row["source_scan_hits"] = ";".join(hits)
    if hits:
        row.update(status="skipped_unsafe", parse_notes="static writer pattern hit")
        write_text(stdout_log, "")
        write_text(stderr_log, "Skipped by static safety scan: " + ", ".join(hits) + "\n")
        return row

    before = snapshot_tree(case_dir)
    env = os.environ.copy()
    conda_prefix = PYTHON.parent
    conda_path_entries = [
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
            "MPLCONFIGDIR": str(OUTPUT_ROOT / "mplconfig"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCH_HOME": str(OUTPUT_ROOT / "torch_home"),
            # The managed shell omits these ordinary Windows variables; old
            # Matplotlib calls WINDIR while building its font cache.
            "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
            "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
            # Directly invoking a Conda interpreter does not activate its DLL
            # search path.  NumPy/SciPy linalg otherwise terminate with the
            # Windows delay-load status C06D007F before Python can emit a trace.
            "PATH": os.pathsep.join(str(path) for path in conda_path_entries)
            + os.pathsep
            + env.get("PATH", ""),
        }
    )

    started = time.perf_counter()
    stdout = ""
    stderr = ""
    returncode: Optional[int] = None
    try:
        completed = subprocess.run(
            [str(PYTHON), "-B", str(source)],
            cwd=str(case_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        row["returncode"] = returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\nAUDIT TIMEOUT after {timeout_s} seconds.\n"
        row["status"] = "timeout"
        row["parse_notes"] = f"exceeded {timeout_s}s timeout"
    except Exception as exc:  # retain diagnostics without mutating research code
        stderr += f"\nAUDIT RUNNER ERROR: {type(exc).__name__}: {exc}\n"
        row["status"] = "runner_error"
        row["parse_notes"] = f"{type(exc).__name__}: {exc}"

    row["elapsed_seconds"] = f"{time.perf_counter() - started:.6f}"
    write_text(stdout_log, stdout)
    write_text(stderr_log, stderr)

    after = snapshot_tree(case_dir)
    if after != before:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        row["source_tree_changed"] = json.dumps(
            {"added": added, "removed": removed, "changed": changed}, ensure_ascii=False
        )
        if not row["status"]:
            row["status"] = "unsafe_runtime_write"
        row["parse_notes"] = (row["parse_notes"] + "; " if row["parse_notes"] else "") + "source tree changed"
    else:
        row["source_tree_changed"] = "false"

    cmt, time_s, foot_error, parse_notes = parse_metrics(kind, stdout)
    row["cmt"] = "" if cmt is None else repr(cmt)
    row["time_s"] = "" if time_s is None else repr(time_s)
    row["foot_error_m"] = "" if foot_error is None else repr(foot_error)
    if parse_notes:
        row["parse_notes"] = (row["parse_notes"] + "; " if row["parse_notes"] else "") + parse_notes

    if not row["status"]:
        if returncode != 0:
            row["status"] = "error"
        elif cmt is None:
            row["status"] = "parse_failed"
        else:
            row["status"] = "ok"
    return row


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    global PYTHON, PAPER2_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter from the locked evaluation environment",
    )
    parser.add_argument(
        "--paper2-root",
        type=Path,
        required=True,
        help="root of the original case bundle containing the 12 folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper2_rerun_logs"),
        help="directory for captured logs and the tidy result CSV",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--smoke", action="store_true", help="run all five methods for the first case only")
    args = parser.parse_args()

    PYTHON = args.python.resolve()
    PAPER2_ROOT = args.paper2_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    if not PYTHON.is_file():
        parser.error(f"Python interpreter not found: {PYTHON}")
    if not PAPER2_ROOT.is_dir():
        parser.error(f"Paper2 bundle root not found: {PAPER2_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "torch_home").mkdir(parents=True, exist_ok=True)

    selected_cases = CASES[:1] if args.smoke else CASES
    rows = []
    for case in selected_cases:
        for method in METHODS:
            print(f"RUN {case[0]} :: {method[0]}", flush=True)
            row = run_one(case, method, args.timeout)
            rows.append(row)
            print(
                f"DONE {row['status']} cmt={row['cmt'] or 'NA'} elapsed={row['elapsed_seconds']}s",
                flush=True,
            )
            # Continuously checkpoint audit metadata in case a later script hangs.
            write_csv(OUTPUT_ROOT / ("paper2_smoke_results.csv" if args.smoke else "paper2_rerun_results.csv"), rows)

    child_version = subprocess.run(
        [str(PYTHON), "-c", "import sys; print(sys.version)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout.strip()
    manifest = {
        "python": str(PYTHON),
        "python_version": child_version,
        "paper2_root": str(PAPER2_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "timeout_seconds": args.timeout,
        "smoke": args.smoke,
        "environment_overrides": {
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        },
        "row_count": len(rows),
        "status_counts": {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
    }
    (OUTPUT_ROOT / ("smoke_manifest.json" if args.smoke else "rerun_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
