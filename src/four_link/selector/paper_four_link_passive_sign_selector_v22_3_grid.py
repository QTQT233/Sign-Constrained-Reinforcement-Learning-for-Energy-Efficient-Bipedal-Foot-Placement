"""Train an online passive hip-sign selector for the V22_3 3x3-grid Cmt evaluator.

The selector is a supervised classifier. It samples initial states from the
same explicit evaluation domain as the released V22_3/V9 evaluator, rolls
out the positive-only and negative-only passive policies, labels the state with
the lower-Cmt successful sign, and trains a small network that can be used
online without trying both signs at evaluation time.
"""

import csv
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# User-editable selector settings
# =============================================================================
# This script imports the CMT evaluator only to reuse the exact same dynamics,
# reset sampler, rollout and Cmt calculation. All experiment settings that affect
# selector dataset generation are explicitly defined below and then applied to
# the imported CMT module.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CMT_SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "src/four_link/evaluation/"
    "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
OUTPUT_SELECTOR_PATH = (
    REPOSITORY_ROOT
    / "results/_scratch/selector_training/paper_four_link_passive_sign_selector_v22_3_grid.pth"
)
OUTPUT_DATASET_CSV = (
    REPOSITORY_ROOT
    / "results/_scratch/selector_training/"
    "paper_four_link_passive_sign_selector_v22_3_grid_dataset.csv"
)

# Passive policy files used to generate labels. These override the paths inside
# CMT_SCRIPT_PATH after it is imported.
PASSIVE_POS_POLICY_NAME = "v22_3_passive_positive_4Nm_grid"
PASSIVE_POS_POLICY_PATH = (
    REPOSITORY_ROOT
    / "models/four_link/v22_3_v9_uni_pos_400_grid/"
    "V22_3V9UniPos400Grid_c097_Policy_best.pth"
)
PASSIVE_NEG_POLICY_NAME = "v22_3_passive_negative_4Nm_grid"
PASSIVE_NEG_POLICY_PATH = (
    REPOSITORY_ROOT
    / "models/four_link/v22_3_v9_uni_neg_400_grid/"
    "V22_3V9UniNeg400Grid_c100_Policy_best.pth"
)

# Actuator/action-table settings. These must match the passive policies above.
HIP_TORQUE_NM = 4.0
KNEE_TORQUE_NM = 0.20

CURRICULUM_PROGRESS = 1.0
MAX_STEPS = 500
DETERMINISTIC_POLICY = True
MIN_NONZERO_TORQUE_STEPS = 1
MIN_COM_DISPLACEMENT_M = 0.006
REQUIRE_NONZERO_TORQUE_FOR_CMT = True
REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT = True
TRAIN_SAMPLES_PER_GRID_CELL_DIRECTION = 1800
VALIDATION_SAMPLES_PER_GRID_CELL_DIRECTION = 600
TRAIN_RANDOM_SEEDS = [20260811, 20260812]
VALIDATION_RANDOM_SEEDS = [20260901]
WALK_DIRECTIONS = [1.0, -1.0]
RESET_PHASE_MODE = "random"  # "random", "pre_cross", "mid_cross", "touchdown"

# Explicit command grid. Set either list to a subset such as [0.35] or [0.0]
# for a faster targeted selector run.
COMMAND_STEP_LENGTH_GRID_M = [0.30, 0.35, 0.40]
COMMAND_STEP_HEIGHT_GRID_M = [0.00, 0.01, 0.02]

# If both signs are valid and the Cmt difference is smaller than this margin,
# the sample is treated as ambiguous and skipped.
CMT_TIE_MARGIN = 0.01

# Label priority is success first, then Cmt among successful signs. If both
# signs succeed but neither has a valid Cmt denominator, skip by default instead
# of injecting a non-Cmt label into a Cmt selector.
ALLOW_BOTH_SUCCESS_WITHOUT_VALID_CMT_LABELS = False

TRAIN_EPOCHS = 120
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def load_cmt_module():
    for required in (CMT_SCRIPT_PATH, PASSIVE_POS_POLICY_PATH, PASSIVE_NEG_POLICY_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required released artifact not found: {required}")
    spec = importlib.util.spec_from_file_location("paper_cmt", CMT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apply_cmt_overrides(module)
    return module


def apply_cmt_overrides(cmt):
    cmt.PASSIVE_POS_POLICY_NAME = PASSIVE_POS_POLICY_NAME
    cmt.PASSIVE_POS_POLICY_PATH = PASSIVE_POS_POLICY_PATH
    cmt.PASSIVE_NEG_POLICY_NAME = PASSIVE_NEG_POLICY_NAME
    cmt.PASSIVE_NEG_POLICY_PATH = PASSIVE_NEG_POLICY_PATH

    cmt.HIP_TORQUE_NM = float(HIP_TORQUE_NM)
    cmt.KNEE_TORQUE_NM = float(KNEE_TORQUE_NM)
    cmt.torque_hip = float(HIP_TORQUE_NM)
    cmt.torque_knee = float(KNEE_TORQUE_NM)
    cmt.PASSIVE_ACTIONS = cmt.build_passive_actions()
    cmt.ACTIVE_ACTIONS = cmt.build_active_actions()
    cmt.ACTIONS = cmt.ACTIVE_ACTIONS

    cmt.CURRICULUM_PROGRESS = float(CURRICULUM_PROGRESS)
    cmt.MAX_STEPS = int(MAX_STEPS)
    cmt.DETERMINISTIC_POLICY = bool(DETERMINISTIC_POLICY)
    cmt.MIN_NONZERO_TORQUE_STEPS = int(MIN_NONZERO_TORQUE_STEPS)
    cmt.MIN_COM_DISPLACEMENT_M = float(MIN_COM_DISPLACEMENT_M)
    cmt.REQUIRE_NONZERO_TORQUE_FOR_CMT = bool(REQUIRE_NONZERO_TORQUE_FOR_CMT)
    cmt.REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT = bool(REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT)
    cmt.RESET_PHASE_MODE = RESET_PHASE_MODE
    cmt.COMMAND_STEP_LENGTH_GRID_M = list(map(float, COMMAND_STEP_LENGTH_GRID_M))
    cmt.COMMAND_STEP_HEIGHT_GRID_M = list(map(float, COMMAND_STEP_HEIGHT_GRID_M))
    cmt.commanded_step_length_options = np.asarray(COMMAND_STEP_LENGTH_GRID_M, dtype=float)
    cmt.commanded_step_height_options = np.asarray(COMMAND_STEP_HEIGHT_GRID_M, dtype=float)
    cmt.COMMAND_STEP_LENGTH_OVERRIDE_M = None
    cmt.COMMAND_STEP_HEIGHT_OVERRIDE_M = None


def label_from_results(pos_result, neg_result):
    # Primary objective: select a sign that reaches the target. Cmt is used only
    # after success is satisfied, so failed low-energy rollouts never beat
    # successful rollouts.
    if pos_result.success and not neg_result.success:
        return 0, "positive_success_only"
    if neg_result.success and not pos_result.success:
        return 1, "negative_success_only"
    if not pos_result.success and not neg_result.success:
        return None, "no_success"

    # Both signs succeed. Prefer valid Cmt comparisons, then valid-only.
    if pos_result.valid_for_cmt and neg_result.valid_for_cmt:
        if abs(pos_result.cmt - neg_result.cmt) < CMT_TIE_MARGIN:
            return None, "ambiguous_valid_cmt"
        return (0, "positive_lower_cmt") if pos_result.cmt < neg_result.cmt else (1, "negative_lower_cmt")
    if pos_result.valid_for_cmt:
        return 0, "both_success_positive_valid_only"
    if neg_result.valid_for_cmt:
        return 1, "both_success_negative_valid_only"

    if not ALLOW_BOTH_SUCCESS_WITHOUT_VALID_CMT_LABELS:
        return None, "both_success_no_valid_cmt"

    pos_score = (pos_result.signed_com_displacement_m, -pos_result.energy_j)
    neg_score = (neg_result.signed_com_displacement_m, -neg_result.energy_j)
    return (0, "positive_success_score_no_valid_cmt") if pos_score >= neg_score else (1, "negative_success_score_no_valid_cmt")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(cmt):
    cmt.CURRICULUM_PROGRESS = CURRICULUM_PROGRESS
    cmt.RESET_PHASE_MODE = RESET_PHASE_MODE
    cmt.update_curriculum(0, progress_override=CURRICULUM_PROGRESS)
    command_lengths = list(map(float, COMMAND_STEP_LENGTH_GRID_M))
    command_heights = list(map(float, COMMAND_STEP_HEIGHT_GRID_M))

    policies = {
        "passive_pos": cmt.load_policy(cmt.PASSIVE_POS_POLICY_PATH, len(cmt.PASSIVE_ACTIONS), cmt.PASSIVE_POS_POLICY_NAME),
        "passive_neg": cmt.load_policy(cmt.PASSIVE_NEG_POLICY_PATH, len(cmt.PASSIVE_ACTIONS), cmt.PASSIVE_NEG_POLICY_NAME),
    }

    features = []
    labels = []
    split_names = []
    audit_rows = []
    episode_index = 0

    if set(TRAIN_RANDOM_SEEDS) & set(VALIDATION_RANDOM_SEEDS):
        raise ValueError("TRAIN_RANDOM_SEEDS and VALIDATION_RANDOM_SEEDS must not overlap.")

    split_specs = [
        ("train", TRAIN_RANDOM_SEEDS, TRAIN_SAMPLES_PER_GRID_CELL_DIRECTION),
        ("validation", VALIDATION_RANDOM_SEEDS, VALIDATION_SAMPLES_PER_GRID_CELL_DIRECTION),
    ]
    for split_name, seeds, samples_per_cell in split_specs:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for walk_direction in WALK_DIRECTIONS:
                for commanded_step_length in command_lengths:
                    for commanded_step_height in command_heights:
                        for _ in range(samples_per_cell):
                            episode_index += 1
                            reset_phase = cmt.choose_eval_reset_phase(walk_direction)
                            initial_state = cmt.sample_initial_state(
                                walk_direction,
                                reset_phase,
                                commanded_step_length,
                                commanded_step_height,
                            )

                            pos_result = cmt.rollout_policy(
                                initial_state,
                                walk_direction,
                                reset_phase,
                                commanded_step_length,
                                commanded_step_height,
                                policies["passive_pos"],
                                "passive_positive",
                                cmt.PASSIVE_POS_POLICY_NAME,
                                seed,
                                episode_index,
                                passive_selected_from="positive_label_candidate",
                            )
                            neg_result = cmt.rollout_policy(
                                initial_state,
                                walk_direction,
                                reset_phase,
                                commanded_step_length,
                                commanded_step_height,
                                policies["passive_neg"],
                                "passive_negative",
                                cmt.PASSIVE_NEG_POLICY_NAME,
                                seed,
                                episode_index,
                                passive_selected_from="negative_label_candidate",
                            )
                            label, label_reason = label_from_results(pos_result, neg_result)
                            if label is not None:
                                state_net = cmt.normalize_state(
                                    initial_state,
                                    0.0,
                                    walk_direction,
                                    commanded_step_length,
                                    commanded_step_height,
                                )
                                features.append(state_net.astype(np.float32))
                                labels.append(int(label))
                                split_names.append(split_name)

                            audit_rows.append(
                                {
                                    "split": split_name,
                                    "seed": seed,
                                    "episode_index": episode_index,
                                    "walk_direction": walk_direction,
                                    "reset_phase": reset_phase,
                                    "commanded_step_length_m": commanded_step_length,
                                    "commanded_step_height_m": commanded_step_height,
                                    "label": "" if label is None else int(label),
                                    "label_name": "" if label is None else ("positive" if label == 0 else "negative"),
                                    "label_reason": label_reason,
                                    "pos_success": pos_result.success,
                                    "pos_valid": pos_result.valid_for_cmt,
                                    "pos_cmt": pos_result.cmt,
                                    "pos_energy_j": pos_result.energy_j,
                                    "pos_displacement_m": pos_result.signed_com_displacement_m,
                                    "pos_reason": pos_result.terminal_reason,
                                    "neg_success": neg_result.success,
                                    "neg_valid": neg_result.valid_for_cmt,
                                    "neg_cmt": neg_result.cmt,
                                    "neg_energy_j": neg_result.energy_j,
                                    "neg_displacement_m": neg_result.signed_com_displacement_m,
                                    "neg_reason": neg_result.terminal_reason,
                                }
                            )
                            if episode_index % 100 == 0:
                                print(f"generated {episode_index} candidates, labelled {len(labels)}")

    write_csv(OUTPUT_DATASET_CSV, audit_rows)
    if not labels:
        raise RuntimeError("No labelled selector samples were generated.")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(split_names, dtype=object),
        audit_rows,
    )


def train_selector(cmt, features, labels, split_names):
    train_idx = np.where(split_names == "train")[0]
    val_idx = np.where(split_names == "validation")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Both train and validation labelled samples are required.")

    train_class_counts = np.bincount(labels[train_idx], minlength=2).astype(float)
    if np.count_nonzero(train_class_counts) < 2:
        raise RuntimeError(
            "Only one selector label class was generated for the training split; "
            f"counts={train_class_counts.astype(int).tolist()}. Training a selector would be meaningless."
        )

    x_train = torch.as_tensor(features[train_idx], dtype=torch.float32, device=cmt.DEVICE)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.long, device=cmt.DEVICE)
    x_val = torch.as_tensor(features[val_idx], dtype=torch.float32, device=cmt.DEVICE)
    y_val = torch.as_tensor(labels[val_idx], dtype=torch.long, device=cmt.DEVICE)

    selector = cmt.make_sign_selector().to(cmt.DEVICE)
    class_weights = train_class_counts.sum() / np.maximum(train_class_counts, 1.0)
    class_weights = class_weights / np.mean(class_weights)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=cmt.DEVICE)
    )
    optimizer = torch.optim.Adam(selector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_acc = -1.0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        perm = torch.randperm(len(x_train), device=cmt.DEVICE)
        selector.train()
        losses = []
        for start in range(0, len(x_train), BATCH_SIZE):
            batch = perm[start:start + BATCH_SIZE]
            logits = selector(x_train[batch])
            loss = loss_fn(logits, y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        selector.eval()
        with torch.no_grad():
            train_acc = float((selector(x_train).argmax(dim=1) == y_train).float().mean().cpu())
            if len(x_val) > 0:
                val_acc = float((selector(x_val).argmax(dim=1) == y_val).float().mean().cpu())
            else:
                val_acc = train_acc
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
            print(
                f"epoch {epoch:03d}, loss {np.mean(losses):.4f}, "
                f"train_acc {train_acc:.3f}, val_acc {val_acc:.3f}"
            )

    checkpoint = {
        "model_state_dict": best_state,
        "state_dim": cmt.state_dim,
        "label_names": ["positive", "negative"],
        "curriculum_progress": CURRICULUM_PROGRESS,
        "samples": int(len(labels)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "positive_labels": int(np.sum(labels == 0)),
        "negative_labels": int(np.sum(labels == 1)),
        "train_positive_labels": int(np.sum(labels[train_idx] == 0)),
        "train_negative_labels": int(np.sum(labels[train_idx] == 1)),
        "validation_positive_labels": int(np.sum(labels[val_idx] == 0)),
        "validation_negative_labels": int(np.sum(labels[val_idx] == 1)),
        "validation_accuracy": float(best_val_acc),
        "cmt_script_path": CMT_SCRIPT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "passive_pos_policy_path": PASSIVE_POS_POLICY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "passive_neg_policy_path": PASSIVE_NEG_POLICY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "hip_torque_nm": float(HIP_TORQUE_NM),
        "knee_torque_nm": float(KNEE_TORQUE_NM),
        "max_steps": int(MAX_STEPS),
        "deterministic_policy": bool(DETERMINISTIC_POLICY),
        "min_nonzero_torque_steps": int(MIN_NONZERO_TORQUE_STEPS),
        "min_com_displacement_m": float(MIN_COM_DISPLACEMENT_M),
        "require_nonzero_torque_for_cmt": bool(REQUIRE_NONZERO_TORQUE_FOR_CMT),
        "require_min_com_displacement_for_cmt": bool(REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT),
        "command_step_length_grid_m": list(map(float, COMMAND_STEP_LENGTH_GRID_M)),
        "command_step_height_grid_m": list(map(float, COMMAND_STEP_HEIGHT_GRID_M)),
        "train_random_seeds": TRAIN_RANDOM_SEEDS,
        "validation_random_seeds": VALIDATION_RANDOM_SEEDS,
        "label_priority": "success_first_then_cmt",
        "allow_both_success_without_valid_cmt_labels": ALLOW_BOTH_SUCCESS_WITHOUT_VALID_CMT_LABELS,
    }
    OUTPUT_SELECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_SELECTOR_PATH)
    print(f"saved selector: {OUTPUT_SELECTOR_PATH}")
    print(
        f"labels positive/negative: {checkpoint['positive_labels']}/"
        f"{checkpoint['negative_labels']}, best_val_acc {best_val_acc:.3f}"
    )


def main():
    cmt = load_cmt_module()
    random.seed(TRAIN_RANDOM_SEEDS[0])
    np.random.seed(TRAIN_RANDOM_SEEDS[0])
    torch.manual_seed(TRAIN_RANDOM_SEEDS[0])
    features, labels, split_names, _ = generate_dataset(cmt)
    train_selector(cmt, features, labels, split_names)


if __name__ == "__main__":
    main()
