#!/usr/bin/env python3
"""Validate the released 12-case learned-controller recovery-grid archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "results/paper2_push_off_grid_12case"
CONFIG_PATH = ROOT / "configs/paper2_push_off_grid_12_cases.csv"
TOLERANCE = 1e-10

CONTROLLERS = {
    "continuous": {
        "slug": "continuous_torque_ppo",
        "label": "Continuous-torque PPO",
        "entrypoint": "Multi_continuous.py",
    },
    "discrete": {
        "slug": "discrete_active_ppo",
        "label": "Unrestricted discrete PPO",
        "entrypoint": "Qi_multi_ac_discrete_sim.py",
    },
    "passive": {
        "slug": "proposed_selector",
        "label": "Transition-start lookup router",
        "entrypoint": "Qi_multi_passive_sim.py",
    },
}


def canonical_case_id(case_id: str) -> str:
    replacements = {
        "flat_1.145_": "flat_L1145_",
        "flat_1.28_": "flat_L1280_",
        "raised_1.145_": "raised_L1145_",
        "raised_1.28_": "raised_L1280_",
    }
    for prefix, canonical in replacements.items():
        if case_id.startswith(prefix):
            return canonical + case_id[len(prefix) :]
    return case_id


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def entrypoint_pair(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = [
        float(value)
        for value in re.findall(
            r"dtheta1_new(?:_fun)?\s*-=\s*([0-9]+(?:\.[0-9]+)?)", text
        )
    ]
    if len(values) != 2:
        raise RuntimeError(f"{path}: expected two recovery decrements, found {values}")
    return values[0], values[1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_checksums() -> str:
    lines = []
    for path in sorted(item for item in RESULT_DIR.rglob("*") if item.is_file()):
        if path.name == "CHECKSUMS.sha256":
            continue
        relative = path.relative_to(RESULT_DIR).as_posix()
        lines.append(f"{sha256(path)}  {relative}")
    return "\n".join(lines) + "\n"


def validate(refresh_metadata: bool) -> dict[str, object]:
    accepted = read_rows(RESULT_DIR / "accepted_cases.csv")
    selected = read_rows(RESULT_DIR / "selected_minima.csv")
    if len(accepted) != 36 or len(selected) != 36:
        raise RuntimeError(
            f"expected 36 accepted and selected rows, found {len(accepted)} and "
            f"{len(selected)}"
        )

    accepted_by_key = {
        (row["case_id"], row["controller"]): row for row in accepted
    }
    selected_by_key = {
        (row["case_id"], row["controller"]): row for row in selected
    }
    if set(accepted_by_key) != set(selected_by_key):
        raise RuntimeError("accepted and selected key sets differ")

    case_ids = {row["case_id"] for row in accepted}
    if len(case_ids) != 12:
        raise RuntimeError(f"expected 12 cases, found {sorted(case_ids)}")
    for case_id in case_ids:
        observed = {
            controller for case, controller in accepted_by_key if case == case_id
        }
        if observed != set(CONTROLLERS):
            raise RuntimeError(f"{case_id}: controller set is {sorted(observed)}")

    config_rows: list[dict[str, object]] = []
    provenance_files: list[dict[str, object]] = []
    controller_values: dict[str, list[float]] = defaultdict(list)
    boundary_optima: list[dict[str, object]] = []
    max_replay_delta = {"cmt": 0.0, "energy_J": 0.0, "displacement_m": 0.0}
    total_candidates = 0
    total_successful = 0

    for key in sorted(accepted_by_key):
        case_id, controller = key
        accepted_row = accepted_by_key[key]
        selected_row = selected_by_key[key]
        metadata = CONTROLLERS[controller]
        canonical = canonical_case_id(case_id)
        candidate_rel = (
            Path("results/paper2_push_off_grid_12case/candidates")
            / canonical
            / f"{metadata['slug']}.csv"
        )
        candidate_path = ROOT / candidate_rel
        rows = read_rows(candidate_path)
        if len(rows) != 14_400:
            raise RuntimeError(f"{candidate_rel}: expected 14,400 rows")

        seen: set[tuple[int, int]] = set()
        eligible: list[dict[str, str]] = []
        for row in rows:
            i = int(row["th1_index"])
            j = int(row["th2_index"])
            seen.add((i, j))
            if int(row["seed"]) != 0:
                raise RuntimeError(f"{candidate_rel}: nonzero seed")
            assert_close(float(row["push_off_1_rad_s"]), i * 0.01, "grid r1")
            assert_close(float(row["push_off_2_rad_s"]), j * 0.01, "grid r2")
            if (
                int(row["success"]) == 1
                and float(row["com_displacement_m"]) > 0.0
                and math.isfinite(float(row["cmt"]))
            ):
                eligible.append(row)
        if seen != {(i, j) for i in range(120) for j in range(120)}:
            raise RuntimeError(f"{candidate_rel}: incomplete 120x120 grid")
        if not eligible:
            raise RuntimeError(f"{candidate_rel}: no eligible candidate")

        best = min(
            eligible,
            key=lambda row: (
                float(row["cmt"]),
                int(row["th1_index"]),
                int(row["th2_index"]),
            ),
        )
        for column in (
            "push_off_1_rad_s",
            "push_off_2_rad_s",
            "energy_J",
            "com_displacement_m",
            "cmt",
        ):
            assert_close(
                float(selected_row[column]),
                float(best[column]),
                f"{case_id}/{controller}/{column}",
            )
            assert_close(
                float(accepted_row[column]),
                float(best[column]),
                f"{case_id}/{controller}/replay/{column}",
            )
        if int(accepted_row["completed_steps"]) != 3:
            raise RuntimeError(f"{case_id}/{controller}: replay did not complete 3 steps")
        if int(selected_row["successful_candidate_count"]) != len(eligible):
            raise RuntimeError(f"{case_id}/{controller}: successful-count mismatch")
        if int(selected_row["candidate_count"]) != len(rows):
            raise RuntimeError(f"{case_id}/{controller}: candidate-count mismatch")
        for flag in ("cmt_reproduced", "energy_reproduced", "displacement_reproduced"):
            if accepted_row[flag] != "True":
                raise RuntimeError(f"{case_id}/{controller}: {flag} is false")

        replay_deltas = {
            "cmt": abs(float(accepted_row["cmt_minus_search"])),
            "energy_J": abs(float(accepted_row["energy_minus_search_J"])),
            "displacement_m": abs(
                float(accepted_row["displacement_minus_search_m"])
            ),
        }
        for name, value in replay_deltas.items():
            max_replay_delta[name] = max(max_replay_delta[name], value)
            if value > TOLERANCE:
                raise RuntimeError(f"{case_id}/{controller}: replay {name} mismatch")

        entrypoint_rel = (
            Path("src/paper2/entrypoints")
            / case_id
            / str(metadata["entrypoint"])
        )
        entrypoint_path = ROOT / entrypoint_rel
        source_pair = entrypoint_pair(entrypoint_path)
        assert_close(source_pair[0], float(best["push_off_1_rad_s"]), "source r1")
        assert_close(source_pair[1], float(best["push_off_2_rad_s"]), "source r2")

        r1 = float(best["push_off_1_rad_s"])
        r2 = float(best["push_off_2_rad_s"])
        if math.isclose(r1, 1.19) or math.isclose(r2, 1.19):
            boundary_optima.append(
                {
                    "case_id": canonical,
                    "controller": str(metadata["label"]),
                    "push_off_1_rad_s": r1,
                    "push_off_2_rad_s": r2,
                }
            )

        controller_values[controller].append(float(accepted_row["cmt"]))
        total_candidates += len(rows)
        total_successful += len(eligible)
        config_rows.append(
            {
                "case_id": canonical,
                "controller": metadata["label"],
                "entrypoint": entrypoint_rel.as_posix(),
                "candidate_record": candidate_rel.as_posix(),
                "selected_r1_rad_s": f"{r1:.2f}",
                "selected_r2_rad_s": f"{r2:.2f}",
                "seed": 0,
                "grid_min_rad_s": "0.00",
                "grid_max_rad_s": "1.19",
                "grid_step_rad_s": "0.01",
                "candidate_count": len(rows),
                "successful_candidate_count": len(eligible),
            }
        )
        provenance_files.append(
            {
                "case_id": canonical,
                "controller": metadata["label"],
                "candidate_record": candidate_rel.as_posix(),
                "candidate_sha256": sha256(candidate_path),
                "entrypoint": entrypoint_rel.as_posix(),
                "entrypoint_sha256": sha256(entrypoint_path),
            }
        )

    released_summary = {
        row["controller"]: row
        for row in read_rows(RESULT_DIR / "controller_summary.csv")
    }
    controller_means = {}
    for controller, values in controller_values.items():
        mean = statistics.fmean(values)
        sample_sd = statistics.stdev(values)
        controller_means[controller] = {
            "n_cases": len(values),
            "mean_cmt": mean,
            "sample_sd_cmt": sample_sd,
        }
        assert_close(
            float(released_summary[controller]["mean_cmt"]),
            mean,
            f"{controller}/mean_cmt",
        )
        assert_close(
            float(released_summary[controller]["sample_sd_cmt"]),
            sample_sd,
            f"{controller}/sample_sd",
        )

    validation = {
        "schema": "paper2-push-off-grid-validation/v1",
        "all_checks_passed": True,
        "case_count": len(case_ids),
        "controller_case_count": len(accepted),
        "candidate_count": total_candidates,
        "eligible_candidate_count": total_successful,
        "fixed_replay_count": len(accepted),
        "fixed_replays_completed_three_steps": len(accepted),
        "grid": {
            "r1_rad_s": {"min": 0.0, "max": 1.19, "step": 0.01, "count": 120},
            "r2_rad_s": {"min": 0.0, "max": 1.19, "step": 0.01, "count": 120},
            "pairs_per_controller_case": 14_400,
            "seed_per_candidate": 0,
        },
        "selection_rule": (
            "complete all three transitions, positive COM displacement, finite "
            "Cmt; minimize Cmt, breaking exact ties by r1 then r2 grid index"
        ),
        "max_absolute_fixed_replay_difference": max_replay_delta,
        "boundary_optima": boundary_optima,
        "controller_cmt": controller_means,
    }
    provenance = {
        "schema": "paper2-push-off-grid-provenance/v1",
        "description": (
            "Complete candidate-level records and fixed-parameter replay outputs "
            "for 12 two-link cases and three learned controllers."
        ),
        "candidate_records": 518_400,
        "fixed_parameter_replays": 36,
        "source_code_boundary": (
            "The case entry points contain the selected recovery decrements. "
            "The complete candidate records and fixed-parameter replays are "
            "validated independently of the environment-specific search "
            "orchestration used for the completed runs."
        ),
        "files": provenance_files,
    }

    if refresh_metadata:
        write_csv(CONFIG_PATH, config_rows)
        (RESULT_DIR / "validation_summary.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (RESULT_DIR / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (RESULT_DIR / "CHECKSUMS.sha256").write_text(
            build_checksums(), encoding="utf-8"
        )
    else:
        stored_validation = json.loads(
            (RESULT_DIR / "validation_summary.json").read_text(encoding="utf-8")
        )
        stored_provenance = json.loads(
            (RESULT_DIR / "provenance.json").read_text(encoding="utf-8")
        )
        if stored_validation != validation:
            raise RuntimeError("validation_summary.json is stale")
        if stored_provenance != provenance:
            raise RuntimeError("provenance.json is stale")
        stored_checksums = (RESULT_DIR / "CHECKSUMS.sha256").read_text(
            encoding="utf-8"
        )
        if stored_checksums != build_checksums():
            raise RuntimeError("scoped CHECKSUMS.sha256 is stale")

    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="rewrite the release config, provenance, validation summary, and checksums",
    )
    args = parser.parse_args()
    validation = validate(args.refresh_metadata)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
