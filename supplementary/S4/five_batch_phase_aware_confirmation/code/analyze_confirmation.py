"""Final analysis for the frozen five-batch phase-aware confirmation.

This script is deliberately analysis-only. It reads the immutable route banks
and hard-mask replays, reconstructs the frozen phase-aware three-expert route,
and writes new files only below ``analysis_final``.

Run:
    python supplementary/S4/five_batch_phase_aware_confirmation/code/analyze_confirmation.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw"
PROTOCOL_PATH = ROOT / "protocol.json"
OUTPUT = ROOT / "reproduced_results"
REPO_ROOT = HERE.parents[3]
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_three_expert_selector_evaluation.py"
)
TWO_EXPERT_PATH = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "paper_four_link_passive_sign_selector_v22_3_grid.pth"
)

SOURCE_LABELS = ("positive", "negative", "active")
CONTROLLERS = (
    "active",
    "positive_expert",
    "negative_expert",
    "proposed_phase_aware_three_expert",
    "released_two_expert_selector",
    "true_hard_mask",
)
PAIR_SPECS = (
    ("active", "proposed_phase_aware_three_expert"),
    ("true_hard_mask", "proposed_phase_aware_three_expert"),
    ("released_two_expert_selector", "proposed_phase_aware_three_expert"),
    ("active", "true_hard_mask"),
)
BOOTSTRAP_PAIRS = PAIR_SPECS[:3]
MATCH_FIELDS = (
    "walk_direction",
    "reset_phase",
    "commanded_step_length_m",
    "commanded_step_height_m",
    "initial_q1_deg",
    "initial_q2_deg",
    "initial_q3_deg",
    "initial_q4_deg",
    "initial_dq1_rad_s",
    "initial_dq2_rad_s",
    "initial_dq3_rad_s",
    "initial_dq4_rad_s",
)


@dataclass(frozen=True)
class Outcome:
    success: bool
    valid: bool
    cmt: float
    terminal_reason: str


@dataclass
class Case:
    batch: str
    seed: int
    episode: int
    direction: float
    length: float
    height: float
    reset_phase: str
    base_features: np.ndarray
    source_outcomes: tuple[Outcome, Outcome, Outcome]
    hard_outcome: Outcome
    gate_probabilities: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_outcome(row: dict[str, str]) -> Outcome:
    return Outcome(
        success=parse_bool(row["success"]),
        valid=parse_bool(row["valid_for_cmt"]),
        cmt=parse_float(row["cmt"]),
        terminal_reason=str(row["terminal_reason"]),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def import_evaluator():
    spec = importlib.util.spec_from_file_location(
        "frozen_five_batch_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DEVICE = torch.device("cpu")
    return module


def compare_match_fields(
    reference: dict[str, str],
    candidate: dict[str, str],
    *,
    seed: int,
    episode: int,
) -> None:
    for field in MATCH_FIELDS:
        if field == "reset_phase":
            equal = str(reference[field]) == str(candidate[field])
        else:
            left = parse_float(reference[field])
            right = parse_float(candidate[field])
            equal = math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        if not equal:
            raise RuntimeError(
                f"Matched-case mismatch at seed={seed}, episode={episode}, "
                f"field={field}: {reference[field]!r} != {candidate[field]!r}"
            )


def validate_seed_files(
    directory: Path,
    protocol: dict,
    batch: str,
    seed: int,
) -> dict:
    route_manifest = load_json(directory / "route_bank_manifest.json")
    hard_manifest = load_json(directory / "hard_mask_replay_manifest.json")
    route_schema = route_manifest.get("schema")
    supported_route_schemas = {
        "validated-three-source-route-bank/v1",
        "phase-aware-three-expert-route-bank-seed/v1",
    }
    if route_schema not in supported_route_schemas:
        raise RuntimeError(
            f"Unsupported route-bank manifest schema {route_schema!r}: "
            f"{directory}"
        )
    if (
        route_schema == "validated-three-source-route-bank/v1"
        and route_manifest.get("validation") != "PASS"
    ):
        raise RuntimeError(f"Route-bank validation is not PASS: {directory}")
    if int(route_manifest["seed"]) != seed or int(hard_manifest["seed"]) != seed:
        raise RuntimeError(f"Manifest seed mismatch: {directory}")
    expected_cases = int(protocol["design"]["cases_per_seed"])
    route_case_count = route_manifest.get(
        "cases", route_manifest.get("n_matched_cases")
    )
    if (
        int(route_case_count) != expected_cases
        or int(hard_manifest["n_cases"]) != expected_cases
    ):
        raise RuntimeError(f"Manifest case-count mismatch: {directory}")
    rollouts = directory / "rollouts_active_vs_passive.csv"
    hard = directory / "hard_mask_replay.csv"
    rollout_hash = sha256(rollouts)
    hard_hash = sha256(hard)
    route_output_hashes = route_manifest.get(
        "output_sha256", route_manifest.get("csv_sha256", {})
    )
    if (
        route_output_hashes["rollouts_active_vs_passive.csv"]
        != rollout_hash
    ):
        raise RuntimeError(f"Route-bank checksum mismatch: {rollouts}")
    if hard_manifest["source_route_bank_sha256"] != rollout_hash:
        raise RuntimeError(f"Hard replay source checksum mismatch: {directory}")
    if hard_manifest["output_sha256"] != hard_hash:
        raise RuntimeError(f"Hard replay checksum mismatch: {hard}")
    frozen = protocol["frozen_sha256"]
    if route_manifest["evaluator_sha256"] != frozen["evaluator"]:
        raise RuntimeError(f"Evaluator hash mismatch: {directory}")
    expected_source_hashes = {
        "evaluator": frozen["evaluator"],
        "active_policy": frozen["active_policy"],
        "positive_policy": frozen["non_negative_policy"],
        "negative_policy": frozen["non_positive_policy"],
    }
    if "frozen_artifact_sha256" in route_manifest:
        source_hashes = route_manifest["frozen_artifact_sha256"]
    else:
        source_hashes = {
            "evaluator": route_manifest["evaluator_sha256"],
            "active_policy": route_manifest["model_sha256"]["active"],
            "positive_policy": route_manifest["model_sha256"]["positive"],
            "negative_policy": route_manifest["model_sha256"]["negative"],
        }
    if source_hashes != expected_source_hashes:
        raise RuntimeError(f"Frozen source-artifact mismatch: {directory}")
    if hard_manifest["evaluator_sha256"] != frozen["evaluator"]:
        raise RuntimeError(f"Hard replay evaluator mismatch: {directory}")
    if hard_manifest["hard_mask_model_sha256"] != frozen["hard_mask_policy"]:
        raise RuntimeError(f"Hard-mask model mismatch: {directory}")
    return {
        "batch": batch,
        "seed": seed,
        "route_manifest_schema": route_schema,
        "route_bank_sha256": rollout_hash,
        "hard_mask_replay_sha256": hard_hash,
        "validation": "PASS",
    }


def load_cases(protocol: dict, evaluator) -> tuple[list[Case], list[dict]]:
    cases: list[Case] = []
    integrity: list[dict] = []
    expected_per_seed = int(protocol["design"]["cases_per_seed"])
    for batch, seeds in protocol["batches"].items():
        for seed_value in seeds:
            seed = int(seed_value)
            directory = RAW / batch / f"seed_{seed}"
            integrity.append(
                validate_seed_files(directory, protocol, batch, seed)
            )
            rollouts = read_csv(directory / "rollouts_active_vs_passive.csv")
            candidates = read_csv(directory / "passive_candidate_audit.csv")
            hard_rows = read_csv(directory / "hard_mask_replay.csv")
            active: dict[tuple[int, int], dict[str, str]] = {}
            signs: dict[
                tuple[int, int], dict[str, dict[str, str]]
            ] = defaultdict(dict)
            hard: dict[tuple[int, int], dict[str, str]] = {}
            for row in rollouts:
                if row["controller"] == "active":
                    key = (int(row["seed"]), int(row["episode_index"]))
                    if key in active:
                        raise RuntimeError(f"Duplicate Active key: {key}")
                    active[key] = row
            for row in candidates:
                label = str(row["candidate_for_mode"])
                if label not in {"positive", "negative"}:
                    continue
                key = (int(row["seed"]), int(row["episode_index"]))
                if label in signs[key]:
                    raise RuntimeError(f"Duplicate sign key: {key}/{label}")
                signs[key][label] = row
            for row in hard_rows:
                key = (int(row["seed"]), int(row["episode_index"]))
                if key in hard:
                    raise RuntimeError(f"Duplicate hard-mask key: {key}")
                hard[key] = row
            expected_keys = {(seed, episode) for episode in range(1, 721)}
            if set(active) != expected_keys:
                raise RuntimeError(
                    f"Seed {seed}: Active keys differ from 1..720"
                )
            if set(signs) != expected_keys or set(hard) != expected_keys:
                raise RuntimeError(f"Seed {seed}: unmatched controller keys")
            if len(active) != expected_per_seed:
                raise RuntimeError(f"Seed {seed}: unexpected case count")
            for key in sorted(active):
                active_row = active[key]
                sign_pair = signs[key]
                if set(sign_pair) != {"positive", "negative"}:
                    raise RuntimeError(f"Incomplete sign pair: {key}")
                compare_match_fields(
                    active_row,
                    sign_pair["positive"],
                    seed=seed,
                    episode=key[1],
                )
                compare_match_fields(
                    active_row,
                    sign_pair["negative"],
                    seed=seed,
                    episode=key[1],
                )
                compare_match_fields(
                    active_row,
                    hard[key],
                    seed=seed,
                    episode=key[1],
                )
                state = np.asarray(
                    [
                        np.deg2rad(parse_float(active_row["initial_q1_deg"])),
                        parse_float(active_row["initial_dq1_rad_s"]),
                        np.deg2rad(parse_float(active_row["initial_q2_deg"])),
                        parse_float(active_row["initial_dq2_rad_s"]),
                        np.deg2rad(parse_float(active_row["initial_q3_deg"])),
                        parse_float(active_row["initial_dq3_rad_s"]),
                        np.deg2rad(parse_float(active_row["initial_q4_deg"])),
                        parse_float(active_row["initial_dq4_rad_s"]),
                    ],
                    dtype=float,
                )
                direction = parse_float(active_row["walk_direction"])
                length = parse_float(
                    active_row["commanded_step_length_m"]
                )
                height = parse_float(
                    active_row["commanded_step_height_m"]
                )
                features = np.asarray(
                    evaluator.normalize_state(
                        state, 0.0, direction, length, height
                    ),
                    dtype=np.float32,
                )
                cases.append(
                    Case(
                        batch=batch,
                        seed=seed,
                        episode=key[1],
                        direction=direction,
                        length=length,
                        height=height,
                        reset_phase=str(active_row["reset_phase"]),
                        base_features=features,
                        source_outcomes=(
                            as_outcome(sign_pair["positive"]),
                            as_outcome(sign_pair["negative"]),
                            as_outcome(active_row),
                        ),
                        hard_outcome=as_outcome(hard[key]),
                    )
                )
    expected_total = int(protocol["design"]["total_cases_per_controller"])
    if len(cases) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} matched cases, found {len(cases)}"
        )
    return cases, integrity


def attach_gate_probabilities(cases: list[Case], evaluator) -> None:
    selector = evaluator.load_three_expert_selector(
        evaluator.THREE_EXPERT_SELECTOR_PATH,
        evaluator.THREE_EXPERT_SELECTOR_NAME,
    )
    features = np.stack([case.base_features for case in cases])
    selector_device = next(selector.parameters()).device
    with torch.no_grad():
        probabilities = torch.softmax(
            selector(
                torch.as_tensor(
                    features,
                    dtype=torch.float32,
                    device=selector_device,
                )
            ),
            dim=1,
        ).cpu().numpy()
    for case, values in zip(cases, probabilities):
        case.gate_probabilities = values.astype(np.float64)


def load_two_expert_selector():
    checkpoint = torch.load(
        TWO_EXPERT_PATH, map_location="cpu", weights_only=False
    )
    model = nn.Sequential(
        nn.Linear(12, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if list(checkpoint["label_names"]) != ["positive", "negative"]:
        raise RuntimeError("Unexpected two-expert label order")
    return model


def build_routes(
    cases: list[Case], protocol: dict
) -> dict[str, np.ndarray | None]:
    gate = np.stack([case.gate_probabilities for case in cases])
    base = np.stack([case.base_features for case in cases]).astype(np.float32)
    proposed = np.full(len(cases), 2, dtype=np.int8)
    thresholds = protocol["proposed_rule"]["thresholds"]
    for phase in ("pre_cross", "mid_cross", "touchdown"):
        indices = np.flatnonzero(
            np.asarray([case.reset_phase == phase for case in cases])
        )
        positive_threshold = float(
            thresholds[phase]["non_negative"]
        )
        negative_threshold = float(
            thresholds[phase]["non_positive"]
        )
        positive_ok = gate[indices, 0] >= positive_threshold
        negative_ok = gate[indices, 1] >= negative_threshold
        selected = np.full(len(indices), 2, dtype=np.int8)
        selected[positive_ok & ~negative_ok] = 0
        selected[negative_ok & ~positive_ok] = 1
        both = positive_ok & negative_ok
        selected[both] = np.where(
            gate[indices[both], 0] >= gate[indices[both], 1], 0, 1
        )
        proposed[indices] = selected
    two_model = load_two_expert_selector()
    with torch.no_grad():
        two = torch.argmax(
            two_model(torch.as_tensor(base)), dim=1
        ).cpu().numpy().astype(np.int8)
    return {
        "active": np.full(len(cases), 2, dtype=np.int8),
        "positive_expert": np.full(len(cases), 0, dtype=np.int8),
        "negative_expert": np.full(len(cases), 1, dtype=np.int8),
        "proposed_phase_aware_three_expert": proposed,
        "released_two_expert_selector": two,
        "true_hard_mask": None,
    }


def controller_arrays(
    cases: list[Case], route: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if route is None:
        selected = [case.hard_outcome for case in cases]
    else:
        selected = [
            case.source_outcomes[int(route[index])]
            for index, case in enumerate(cases)
        ]
    return (
        np.asarray([value.success for value in selected], dtype=bool),
        np.asarray([value.valid for value in selected], dtype=bool),
        np.asarray([value.cmt for value in selected], dtype=float),
        np.asarray([value.terminal_reason for value in selected], dtype=object),
    )


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return centre - half, centre + half


def summarize_controller(
    indices: np.ndarray,
    route: np.ndarray | None,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    success, valid, cmt, _ = arrays
    success_subset = success[indices]
    valid_subset = valid[indices]
    values = cmt[indices][valid_subset]
    success_ci = wilson_interval(int(success_subset.sum()), len(indices))
    valid_ci = wilson_interval(int(valid_subset.sum()), len(indices))
    if len(values):
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        median = float(np.median(values))
        q25, q75 = (float(x) for x in np.quantile(values, [0.25, 0.75]))
    else:
        mean = sd = median = q25 = q75 = float("nan")
    if route is None:
        route_counts = {
            "positive": 0,
            "negative": 0,
            "active": 0,
            "hard_mask": int(len(indices)),
        }
    else:
        route_counts = {
            label: int(np.sum(route[indices] == source_index))
            for source_index, label in enumerate(SOURCE_LABELS)
        }
        route_counts["hard_mask"] = 0
    return {
        "n": int(len(indices)),
        "success_count": int(success_subset.sum()),
        "success_rate": float(success_subset.mean()),
        "success_wilson_95_low": finite(success_ci[0]),
        "success_wilson_95_high": finite(success_ci[1]),
        "valid_count": int(valid_subset.sum()),
        "valid_rate": float(valid_subset.mean()),
        "valid_wilson_95_low": finite(valid_ci[0]),
        "valid_wilson_95_high": finite(valid_ci[1]),
        "self_valid_mean_cmt": finite(mean),
        "self_valid_sample_sd_cmt": finite(sd),
        "self_valid_median_cmt": finite(median),
        "self_valid_q25_cmt": finite(q25),
        "self_valid_q75_cmt": finite(q75),
        "route_counts": route_counts,
    }


def exact_mcnemar(
    first_success: np.ndarray, second_success: np.ndarray
) -> tuple[int, int, float]:
    second_only = int(np.sum(second_success & ~first_success))
    first_only = int(np.sum(first_success & ~second_success))
    discordant = second_only + first_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                min(second_only, first_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    return second_only, first_only, p_value


def summarize_pair(
    indices: np.ndarray,
    first_name: str,
    second_name: str,
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> dict:
    first_success, first_valid, first_cmt, _ = arrays[first_name]
    second_success, second_valid, second_cmt, _ = arrays[second_name]
    fs = first_success[indices]
    ss = second_success[indices]
    fv = first_valid[indices]
    sv = second_valid[indices]
    fc = first_cmt[indices]
    sc = second_cmt[indices]
    common = fv & sv
    second_only, first_only, mcnemar_p = exact_mcnemar(fs, ss)
    if common.any():
        first_values = fc[common]
        second_values = sc[common]
        delta = second_values - first_values
        first_mean = float(first_values.mean())
        second_mean = float(second_values.mean())
        result = {
            "first_common_mean_cmt": first_mean,
            "first_common_sample_sd_cmt": float(
                first_values.std(ddof=1)
            )
            if len(first_values) > 1
            else 0.0,
            "second_common_mean_cmt": second_mean,
            "second_common_sample_sd_cmt": float(
                second_values.std(ddof=1)
            )
            if len(second_values) > 1
            else 0.0,
            "second_minus_first_mean_cmt": float(delta.mean()),
            "ratio_of_means_reduction": (
                float(1.0 - second_mean / first_mean)
                if first_mean != 0.0
                else None
            ),
            "second_lower_fraction": float(np.mean(delta < -1e-12)),
            "cmt_tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        }
    else:
        result = {
            "first_common_mean_cmt": None,
            "first_common_sample_sd_cmt": None,
            "second_common_mean_cmt": None,
            "second_common_sample_sd_cmt": None,
            "second_minus_first_mean_cmt": None,
            "ratio_of_means_reduction": None,
            "second_lower_fraction": None,
            "cmt_tie_fraction": None,
        }
    return {
        "first": first_name,
        "second": second_name,
        "n": int(len(indices)),
        "first_success_count": int(fs.sum()),
        "second_success_count": int(ss.sum()),
        "success_difference_pp": float(100.0 * (ss.mean() - fs.mean())),
        "second_only_success_count": second_only,
        "first_only_success_count": first_only,
        "exact_two_sided_mcnemar_p": mcnemar_p,
        "common_valid_count": int(common.sum()),
        "common_valid_fraction_of_first_valid": float(
            common.sum() / max(1, fv.sum())
        ),
        "common_valid_fraction_of_second_valid": float(
            common.sum() / max(1, sv.sum())
        ),
        **result,
    }


def bootstrap_pairs(
    cases: list[Case],
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pair_specs: Iterable[tuple[str, str]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict]:
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        strata[
            (case.seed, case.direction, case.length, case.height)
        ].append(index)
    ordered = [np.asarray(strata[key], dtype=np.int32) for key in sorted(strata)]
    sizes = {len(group) for group in ordered}
    if sizes != {40}:
        raise RuntimeError(f"Expected 40 cases per bootstrap stratum, got {sizes}")
    group_matrix = np.stack(ordered)
    n_groups, group_size = group_matrix.shape
    n_cases = n_groups * group_size
    if n_cases != len(cases):
        raise RuntimeError("Bootstrap strata do not partition all cases")
    prepared = {}
    for first_name, second_name in pair_specs:
        fs, fv, fc, _ = arrays[first_name]
        ss, sv, sc, _ = arrays[second_name]
        common = fv & sv
        prepared[(first_name, second_name)] = {
            "success_delta": ss.astype(np.int8) - fs.astype(np.int8),
            "common": common,
            "first_cmt": np.where(common, fc, 0.0),
            "second_cmt": np.where(common, sc, 0.0),
        }
    storage = {
        pair: {
            "success": np.empty(repetitions, dtype=np.float64),
            "cmt_delta": np.empty(repetitions, dtype=np.float64),
            "ratio_reduction": np.empty(repetitions, dtype=np.float64),
        }
        for pair in prepared
    }
    rng = np.random.default_rng(seed)
    chunk_size = 100
    group_axis = np.arange(n_groups)[None, :, None]
    for start in range(0, repetitions, chunk_size):
        stop = min(repetitions, start + chunk_size)
        count = stop - start
        local = rng.integers(
            0,
            group_size,
            size=(count, n_groups, group_size),
            dtype=np.int16,
        )
        sampled = group_matrix[group_axis, local].reshape(count, n_cases)
        for pair, values in prepared.items():
            success = values["success_delta"][sampled].mean(axis=1)
            common = values["common"][sampled]
            common_count = common.sum(axis=1)
            if np.any(common_count == 0):
                raise RuntimeError("Bootstrap replicate has no common-valid case")
            first_sum = values["first_cmt"][sampled].sum(axis=1)
            second_sum = values["second_cmt"][sampled].sum(axis=1)
            first_mean = first_sum / common_count
            second_mean = second_sum / common_count
            storage[pair]["success"][start:stop] = success
            storage[pair]["cmt_delta"][start:stop] = (
                second_sum - first_sum
            ) / common_count
            storage[pair]["ratio_reduction"][start:stop] = (
                1.0 - second_mean / first_mean
            )
        if stop % 2000 == 0 or stop == repetitions:
            print(f"bootstrap {stop}/{repetitions}", flush=True)
    result = {}
    for pair, values in storage.items():
        first_name, second_name = pair
        success = values["success"]
        cmt_delta = values["cmt_delta"]
        reduction = values["ratio_reduction"]
        result[f"{first_name}_vs_{second_name}"] = {
            "replicates": repetitions,
            "seed": seed,
            "strata": [
                "evaluation_seed",
                "walking_direction",
                "commanded_horizontal_target",
                "commanded_height_target",
            ],
            "success_difference_95_ci_pp": [
                float(100.0 * np.quantile(success, 0.025)),
                float(100.0 * np.quantile(success, 0.975)),
            ],
            "success_difference_one_sided_95_lower_pp": float(
                100.0 * np.quantile(success, 0.05)
            ),
            "cmt_difference_95_ci": [
                float(np.quantile(cmt_delta, 0.025)),
                float(np.quantile(cmt_delta, 0.975)),
            ],
            "ratio_of_means_reduction_95_ci": [
                float(np.quantile(reduction, 0.025)),
                float(np.quantile(reduction, 0.975)),
            ],
        }
    return result


def flatten_controller(
    group_fields: dict, name: str, summary: dict
) -> dict:
    routes = summary["route_counts"]
    return {
        **group_fields,
        "controller": name,
        **{key: value for key, value in summary.items() if key != "route_counts"},
        "route_positive_count": routes["positive"],
        "route_negative_count": routes["negative"],
        "route_active_count": routes["active"],
        "route_hard_mask_count": routes["hard_mask"],
    }


def write_group_metrics(
    cases: list[Case],
    routes: dict[str, np.ndarray | None],
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> dict:
    all_indices = np.arange(len(cases), dtype=int)
    batch = np.asarray([case.batch for case in cases], dtype=object)
    seed = np.asarray([case.seed for case in cases], dtype=int)
    direction = np.asarray([case.direction for case in cases], dtype=float)
    length = np.asarray([case.length for case in cases], dtype=float)
    height = np.asarray([case.height for case in cases], dtype=float)
    group_specs = {
        "overall": [({}, all_indices)],
        "by_batch": [
            ({"batch": value}, np.flatnonzero(batch == value))
            for value in sorted(set(batch))
        ],
        "by_seed": [
            ({"seed": int(value)}, np.flatnonzero(seed == value))
            for value in sorted(set(seed))
        ],
        "by_cell": [
            (
                {
                    "commanded_step_length_m": float(l_value),
                    "commanded_step_height_m": float(h_value),
                },
                np.flatnonzero((length == l_value) & (height == h_value)),
            )
            for l_value in sorted(set(length))
            for h_value in sorted(set(height))
        ],
        "by_cell_direction": [
            (
                {
                    "walk_direction": float(d_value),
                    "commanded_step_length_m": float(l_value),
                    "commanded_step_height_m": float(h_value),
                },
                np.flatnonzero(
                    (direction == d_value)
                    & (length == l_value)
                    & (height == h_value)
                ),
            )
            for d_value in sorted(set(direction))
            for l_value in sorted(set(length))
            for h_value in sorted(set(height))
        ],
    }
    overall_controllers = {}
    overall_pairs = {}
    for group_name, groups in group_specs.items():
        controller_records = []
        pair_records = []
        for fields, indices in groups:
            for name in CONTROLLERS:
                summary = summarize_controller(
                    indices, routes[name], arrays[name]
                )
                controller_records.append(
                    flatten_controller(fields, name, summary)
                )
                if group_name == "overall":
                    overall_controllers[name] = summary
            for first_name, second_name in PAIR_SPECS:
                summary = summarize_pair(
                    indices, first_name, second_name, arrays
                )
                pair_records.append({**fields, **summary})
                if group_name == "overall":
                    overall_pairs[
                        f"{first_name}_vs_{second_name}"
                    ] = summary
        pd.DataFrame(controller_records).to_csv(
            OUTPUT / f"controller_metrics_{group_name}.csv", index=False
        )
        pd.DataFrame(pair_records).to_csv(
            OUTPUT / f"paired_metrics_{group_name}.csv", index=False
        )
    return {
        "controllers": overall_controllers,
        "pairs": overall_pairs,
    }


def terminal_reason_records(
    cases: list[Case],
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> list[dict]:
    records = []
    for name, (success, _, _, reasons) in arrays.items():
        counts: dict[tuple[bool, str], int] = defaultdict(int)
        for ok, reason in zip(success, reasons):
            counts[(bool(ok), str(reason))] += 1
        for (ok, reason), count in sorted(counts.items()):
            records.append(
                {
                    "controller": name,
                    "success": ok,
                    "terminal_reason": reason,
                    "count": count,
                    "fraction_of_controller_cases": count / len(cases),
                }
            )
    return records


def write_matched_trials(
    cases: list[Case],
    routes: dict[str, np.ndarray | None],
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> None:
    fields = [
        "batch",
        "seed",
        "episode_index",
        "walk_direction",
        "commanded_step_length_m",
        "commanded_step_height_m",
        "reset_phase",
        "gate_probability_positive",
        "gate_probability_negative",
        "gate_probability_active",
    ]
    for name in CONTROLLERS:
        fields.extend(
            [
                f"{name}_route",
                f"{name}_success",
                f"{name}_valid",
                f"{name}_cmt",
                f"{name}_terminal_reason",
            ]
        )
    with gzip.open(
        OUTPUT / "matched_trials_10800.csv.gz",
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(cases):
            row = {
                "batch": case.batch,
                "seed": case.seed,
                "episode_index": case.episode,
                "walk_direction": case.direction,
                "commanded_step_length_m": case.length,
                "commanded_step_height_m": case.height,
                "reset_phase": case.reset_phase,
                "gate_probability_positive": case.gate_probabilities[0],
                "gate_probability_negative": case.gate_probabilities[1],
                "gate_probability_active": case.gate_probabilities[2],
            }
            for name in CONTROLLERS:
                success, valid, cmt, reason = arrays[name]
                route = routes[name]
                route_label = (
                    "hard_mask"
                    if route is None
                    else SOURCE_LABELS[int(route[index])]
                )
                row.update(
                    {
                        f"{name}_route": route_label,
                        f"{name}_success": bool(success[index]),
                        f"{name}_valid": bool(valid[index]),
                        f"{name}_cmt": float(cmt[index]),
                        f"{name}_terminal_reason": str(reason[index]),
                    }
                )
            writer.writerow(row)


def table_i_diagnostics(
    arrays: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]
) -> dict:
    active = arrays["active"][0]
    positive = arrays["positive_expert"][0]
    negative = arrays["negative_expert"][0]
    proposed = arrays["proposed_phase_aware_three_expert"][0]
    return {
        "proposed_success_active_failure_count": int(
            np.sum(proposed & ~active)
        ),
        "active_success_proposed_failure_count": int(
            np.sum(active & ~proposed)
        ),
        "proposed_success_and_all_three_source_experts_fail_count": int(
            np.sum(proposed & ~active & ~positive & ~negative)
        ),
        "interpretation": (
            "The last count must be zero because the proposed controller "
            "selects exactly one of the three frozen source experts. Thus an "
            "'only three-expert feasible' category relative to all three "
            "constituents is not a meaningful extra Table-I partition. A "
            "'proposed feasible / Active infeasible' comparison is meaningful "
            "but is a matched evaluation result, not an independent-expert "
            "reachability category."
        ),
    }


def write_checksums() -> None:
    entries = []
    for path in sorted(OUTPUT.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{sha256(path)}  {path.name}")
    (OUTPUT / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty final output: {OUTPUT}"
        )
    evaluator = import_evaluator()
    cases, integrity = load_cases(protocol, evaluator)
    attach_gate_probabilities(cases, evaluator)
    routes = build_routes(cases, protocol)
    arrays = {
        name: controller_arrays(cases, routes[name]) for name in CONTROLLERS
    }
    group_results = write_group_metrics(cases, routes, arrays)
    bootstrap = bootstrap_pairs(
        cases,
        arrays,
        BOOTSTRAP_PAIRS,
        repetitions=int(protocol["analysis"]["bootstrap_replicates"]),
        seed=int(protocol["analysis"]["bootstrap_seed"]),
    )
    for pair_name, inference in bootstrap.items():
        group_results["pairs"][pair_name].update(inference)
    active_pair = group_results["pairs"][
        "active_vs_proposed_phase_aware_three_expert"
    ]
    hard_pair = group_results["pairs"][
        "true_hard_mask_vs_proposed_phase_aware_three_expert"
    ]
    criteria = {
        "proposed_minus_active_success_point_estimate_at_least_zero": (
            active_pair["success_difference_pp"] >= 0.0
        ),
        "proposed_minus_active_success_one_sided_95_lower_at_least_minus_1pp": (
            active_pair["success_difference_one_sided_95_lower_pp"] >= -1.0
        ),
        "proposed_vs_active_common_valid_fraction_of_active_valid_at_least_0_95": (
            active_pair["common_valid_fraction_of_first_valid"] >= 0.95
        ),
        "proposed_minus_active_paired_cmt_95_upper_below_zero": (
            active_pair["cmt_difference_95_ci"][1] < 0.0
        ),
        "proposed_vs_active_ratio_of_means_cmt_reduction_at_least_0_20": (
            active_pair["ratio_of_means_reduction"] >= 0.20
        ),
        "proposed_minus_hard_mask_success_point_estimate_at_least_zero": (
            hard_pair["success_difference_pp"] >= 0.0
        ),
        "proposed_minus_hard_mask_paired_cmt_95_upper_below_zero": (
            hard_pair["cmt_difference_95_ci"][1] < 0.0
        ),
    }
    artifact_hashes = {
        "protocol": sha256(PROTOCOL_PATH),
        "analysis_script": sha256(Path(__file__).resolve()),
        "evaluator": sha256(EVALUATOR_PATH),
        "active_policy": sha256(evaluator.ACTIVE_POLICY_PATH),
        "non_negative_policy": sha256(evaluator.PASSIVE_POS_POLICY_PATH),
        "non_positive_policy": sha256(evaluator.PASSIVE_NEG_POLICY_PATH),
        "three_expert_gate": sha256(evaluator.THREE_EXPERT_SELECTOR_PATH),
        "released_two_expert_gate": sha256(TWO_EXPERT_PATH),
        "hard_mask_policy": sha256(evaluator.TRUE_ACTION_MASK_POLICY_PATH),
    }
    frozen = protocol["frozen_sha256"]
    artifact_checks = {
        "evaluator": artifact_hashes["evaluator"] == frozen["evaluator"],
        "active_policy": artifact_hashes["active_policy"]
        == frozen["active_policy"],
        "non_negative_policy": artifact_hashes["non_negative_policy"]
        == frozen["non_negative_policy"],
        "non_positive_policy": artifact_hashes["non_positive_policy"]
        == frozen["non_positive_policy"],
        "released_two_expert_gate": artifact_hashes[
            "released_two_expert_gate"
        ]
        == frozen["released_two_expert_gate"],
        "hard_mask_policy": artifact_hashes["hard_mask_policy"]
        == frozen["hard_mask_policy"],
    }
    if not all(artifact_checks.values()):
        raise RuntimeError(f"Frozen artifact check failed: {artifact_checks}")
    report = {
        "schema": "phase-aware-three-expert-five-batch-final-analysis/v1",
        "protocol": protocol,
        "n_cases_per_controller": len(cases),
        "n_batches": len(protocol["batches"]),
        "n_evaluation_seeds": sum(
            len(values) for values in protocol["batches"].values()
        ),
        "raw_integrity": {
            "all_15_seed_banks_and_hard_replays_pass": True,
            "seed_files": integrity,
        },
        "artifact_hashes": artifact_hashes,
        "artifact_checks_against_frozen_protocol": artifact_checks,
        "controllers": group_results["controllers"],
        "pairs": group_results["pairs"],
        "bootstrap": {
            "replicates": int(
                protocol["analysis"]["bootstrap_replicates"]
            ),
            "seed": int(protocol["analysis"]["bootstrap_seed"]),
            "strata": protocol["analysis"]["bootstrap_strata"],
        },
        "primary_acceptance": criteria,
        "all_primary_acceptance_criteria_met": bool(all(criteria.values())),
        "table_i_diagnostic": table_i_diagnostics(arrays),
    }
    (OUTPUT / "final_confirmation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(terminal_reason_records(cases, arrays)).to_csv(
        OUTPUT / "terminal_reason_summary.csv", index=False
    )
    pd.DataFrame(integrity).to_csv(
        OUTPUT / "raw_integrity_by_seed.csv", index=False
    )
    write_matched_trials(cases, routes, arrays)
    concise = {
        "all_primary_acceptance_criteria_met": report[
            "all_primary_acceptance_criteria_met"
        ],
        "primary_acceptance": criteria,
        "controllers": {
            name: {
                "success_count": values["success_count"],
                "success_rate": values["success_rate"],
                "valid_count": values["valid_count"],
                "self_valid_mean_cmt": values["self_valid_mean_cmt"],
                "self_valid_sample_sd_cmt": values[
                    "self_valid_sample_sd_cmt"
                ],
                "route_counts": values["route_counts"],
            }
            for name, values in group_results["controllers"].items()
        },
        "primary_pairs": {
            name: group_results["pairs"][name]
            for name in (
                "active_vs_proposed_phase_aware_three_expert",
                "true_hard_mask_vs_proposed_phase_aware_three_expert",
                "released_two_expert_selector_vs_proposed_phase_aware_three_expert",
            )
        },
        "table_i_diagnostic": report["table_i_diagnostic"],
    }
    (OUTPUT / "RESULTS_SUMMARY.json").write_text(
        json.dumps(concise, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# Frozen five-batch final analysis

This directory was produced by:

```
{sys.executable} {Path(__file__).resolve()}
```

- Matched cases per controller: {len(cases):,}
- Bootstrap: {protocol['analysis']['bootstrap_replicates']:,} replicates,
  seed {protocol['analysis']['bootstrap_seed']}, stratified by evaluation seed,
  walking direction, commanded horizontal target, and commanded height target.
- Binary paired test: exact two-sided McNemar test.
- All frozen primary acceptance criteria met: **{all(criteria.values())}**

See `RESULTS_SUMMARY.json` for the concise result,
`final_confirmation_report.json` for the complete machine-readable report,
and the `controller_metrics_*` / `paired_metrics_*` CSV files for all
batches, seeds, endpoint cells, and directions. Raw inputs were read only.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    write_checksums()
    print(json.dumps(concise, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
