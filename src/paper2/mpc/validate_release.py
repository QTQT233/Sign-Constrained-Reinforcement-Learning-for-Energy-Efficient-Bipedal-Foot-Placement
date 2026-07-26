"""Validate the repository-relative unified 12-case MPC release."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path


MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[2]
CONFIG = REPO_ROOT / "configs" / "paper2_mpc_unified_12_cases.json"
RESULT_DIR = REPO_ROOT / "results" / "paper2_mpc_unified_12case"
CASES = RESULT_DIR / "accepted_cases.csv"
SUMMARY = RESULT_DIR / "accepted_summary.json"
FORMAL = RESULT_DIR / "formal_validation.json"
CHECKSUMS = RESULT_DIR / "CHECKSUMS.sha256"
CURRENT = REPO_ROOT / "results" / "paper2_mpc_current_12_cases.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    path = MPC_DIR / "run_unified_12case.py"
    spec = importlib.util.spec_from_file_location("paper2_mpc_unified_release", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def main() -> int:
    runner = load_runner()
    config = runner.verify_release_config()
    pinned = runner.verify_pinned_sources()
    require(len(pinned) == 3, "expected three pinned MPC source files")

    expected_hashes = {key: value.lower() for key, value in config["source_files"].items()}
    require(
        {key: value.lower() for key, value in pinned.items()} == expected_hashes,
        "runner source hashes differ from the release config",
    )
    for case in runner.CASES.values():
        runner.audit_case(case)
    raised_r1 = REPO_ROOT / "src" / "paper2" / "entrypoints" / "raised_1.145_r1"
    for filename in (
        "Qi_multi_passive_sim.py",
        "Qi_multi_ac_discrete_sim.py",
        "Multi_continuous.py",
        "LIPM.py",
        "LQR.py",
    ):
        config_values = runner._literal_source_config(raised_r1 / filename)
        require(
            tuple(config_values.get("init_idx", ())) == (19, 11, 15, 1),
            f"raised_1.145_r1 initial index mismatch: {filename}",
        )

    rows = read_csv(CASES)
    require(len(rows) == 12, f"expected 12 accepted cases, found {len(rows)}")
    require(len({row["case_id"] for row in rows}) == 12, "accepted case IDs are not unique")
    require(all(row["method"] == "Continuous-torque MPC" for row in rows), "MPC method label mismatch")
    require(all(row["success"] == "true" for row in rows), "an accepted case is unsuccessful")
    require(all(row["replay_1_exact"] == "true" for row in rows), "replay 1 mismatch")
    require(all(row["replay_2_exact"] == "true" for row in rows), "replay 2 mismatch")
    require(
        all(len(row["accepted_result_sha256"]) == 64 for row in rows),
        "an accepted result hash is malformed",
    )

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    require(formal["integrity_pass"], "formal post-run validation did not pass")
    require(formal["error_count"] == 0, "formal post-run validation contains errors")
    require(summary["successful_cases"] == 12, "summary success count is not 12")
    require(summary["method"] == "Continuous-torque MPC", "summary method label mismatch")
    require(summary["formal_validation_integrity_pass"], "summary validation flag is false")

    for line in CHECKSUMS.read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split("  ", 1)
        require(sha256(RESULT_DIR / relative) == expected, f"release checksum mismatch: {relative}")

    cmt = [float(row["cmt"]) for row in rows]
    landing_case_mae = [
        float(row["foot_placement_mae_per_transition_m"]) for row in rows
    ]
    landing_transitions = sum(int(row["landing_transition_count"]) for row in rows)
    require(landing_transitions == 36, "expected 36 MPC landing transitions")
    require(close(statistics.fmean(cmt), summary["overall_cmt"]["mean"]), "overall Cmt mean mismatch")
    require(close(statistics.stdev(cmt), summary["overall_cmt"]["sample_sd"]), "overall Cmt SD mismatch")
    require(
        close(
            statistics.fmean(landing_case_mae),
            summary["overall_foot_placement_mae"][
                "pooled_per_transition_mae_m"
            ],
        ),
        "pooled per-transition foot-placement MAE mismatch",
    )

    current = {row["case_id"]: row for row in read_csv(CURRENT)}
    accepted_by_id = {row["case_id"]: row for row in rows}
    require(set(current) == set(accepted_by_id), "current MPC table case IDs differ from accepted cases")
    for case_id, row in current.items():
        require(close(float(row["cmt"]), float(accepted_by_id[case_id]["cmt"])), f"Cmt mismatch: {case_id}")
        require(close(float(row["time_s"]), float(accepted_by_id[case_id]["time_s"])), f"time mismatch: {case_id}")
        require(
            close(
                float(row["foot_placement_mae_per_transition_m"]),
                float(
                    accepted_by_id[case_id][
                        "foot_placement_mae_per_transition_m"
                    ]
                ),
            ),
            f"foot-placement MAE mismatch: {case_id}",
        )

    public_files = [CONFIG, CASES, SUMMARY, FORMAL, CURRENT]
    forbidden = ("C:\\Users\\", "D:\\L&S\\", "D:\\L-Environment\\")
    for path in public_files:
        text = path.read_text(encoding="utf-8-sig")
        require(not any(token in text for token in forbidden), f"machine-local path in {path.relative_to(REPO_ROOT)}")

    print("Unified MPC release validation: PASS")
    print(f"Cases: {len(rows)}/12 successful; replay pairs exact")
    print(f"Cmt mean: {statistics.fmean(cmt):.12f}; sample SD: {statistics.stdev(cmt):.12f}")
    print(
        "Pooled per-transition foot-placement MAE: "
        f"{statistics.fmean(landing_case_mae):.12f} m"
    )
    print(f"Pinned sources: {len(pinned)}; formal validation errors: {formal['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
