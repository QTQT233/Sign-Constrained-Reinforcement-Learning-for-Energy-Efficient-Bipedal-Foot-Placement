"""Run the frozen five-batch route banks or hard-mask replays with a bounded pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
SEED_TO_BATCH = {
    int(seed): batch
    for batch, seeds in PROTOCOL["batches"].items()
    for seed in seeds
}
ROUTE_RUNNER = HERE / "run_route_bank_seed.py"
HARD_MASK_RUNNER = HERE / "replay_hard_mask.py"
DEFAULT_PYTHON = Path(sys.executable)


def route_dir(seed: int) -> Path:
    return ROOT / "raw" / SEED_TO_BATCH[seed] / f"seed_{seed}"


def is_complete(seed: int, phase: str) -> bool:
    directory = route_dir(seed)
    if phase == "route":
        return (directory / "route_bank_manifest.json").exists() and (
            directory / "CHECKSUMS.sha256"
        ).exists()
    return (directory / "hard_mask_replay.csv").exists() and (
        directory / "hard_mask_replay_manifest.json"
    ).exists()


def command(seed: int, phase: str, python: Path) -> list[str]:
    directory = route_dir(seed)
    if phase == "route":
        return [
            str(python),
            str(ROUTE_RUNNER),
            "--seed",
            str(seed),
            "--output-dir",
            str(directory),
        ]
    return [
        str(python),
        str(HARD_MASK_RUNNER),
        "--seed",
        str(seed),
        "--route-bank-dir",
        str(directory),
    ]


def validate_finished(seed: int, phase: str) -> None:
    directory = route_dir(seed)
    if not is_complete(seed, phase):
        raise RuntimeError(f"{phase} seed {seed} lacks completion artifacts")
    if phase == "hard":
        path = directory / "hard_mask_replay.csv"
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        if rows != 720:
            raise RuntimeError(f"hard-mask seed {seed}: expected 720 rows, got {rows}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("route", "hard"))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")

    seeds = sorted(SEED_TO_BATCH)
    pending = [seed for seed in seeds if not is_complete(seed, args.phase)]
    for seed in seeds:
        if seed not in pending:
            validate_finished(seed, args.phase)
            print(f"SKIP complete {args.phase} seed {seed}", flush=True)

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    running: dict[int, tuple[subprocess.Popen, object, float]] = {}
    failures: list[tuple[int, int]] = []

    while pending or running:
        while pending and len(running) < args.max_workers:
            seed = pending.pop(0)
            directory = route_dir(seed)
            directory.parent.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"{args.phase}_seed_{seed}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command(seed, args.phase, args.python),
                cwd=str(ROOT.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running[seed] = (process, log_handle, time.monotonic())
            print(
                f"START {args.phase} seed {seed} pid={process.pid} log={log_path}",
                flush=True,
            )
            time.sleep(2)

        time.sleep(2)
        for seed, (process, log_handle, started) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            elapsed = time.monotonic() - started
            log_handle.close()
            del running[seed]
            if return_code == 0:
                try:
                    validate_finished(seed, args.phase)
                except Exception:
                    failures.append((seed, 99))
                    print(
                        f"INVALID {args.phase} seed {seed} after {elapsed:.1f}s",
                        flush=True,
                    )
                    continue
                print(
                    f"DONE {args.phase} seed {seed} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            else:
                failures.append((seed, return_code))
                print(
                    f"FAILED {args.phase} seed {seed} rc={return_code} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if failures:
        print(f"FAILURES: {failures}", flush=True)
        return 1
    print(f"ALL {args.phase} SEEDS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
