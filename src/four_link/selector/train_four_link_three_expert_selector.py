"""Train the transition-locked four-link positive/negative/active expert gate.

The expensive positive- and negative-expert rollouts are reused from the
frozen 75,600-candidate selector record. Initial normalized states are
reconstructed exactly from the archived seed/episode/command sequence. The
script refuses to train unless this reconstruction reproduces the frozen
two-class selector's validation accuracy.

Routing labels follow the paper rule:

* if only one sign-constrained expert succeeds, select that expert;
* if both succeed, retain the frozen lower-Cmt label, resolving archived
  Cmt ties by the lower Cmt and no-valid-Cmt ties by progress then work;
* if neither sign-constrained expert succeeds, select the frozen Active PPO
  expert.

Only this small gate is trained.  All three experts remain frozen.
Running this file directly starts the full training job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
SOURCE_AUDIT_CSV = (
    REPO_ROOT
    / "data"
    / "four_link"
    / "selector"
    / "paper_four_link_passive_sign_selector_v22_3_grid_dataset.csv"
)
RELEASED_TWO_CLASS_SELECTOR = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "paper_four_link_passive_sign_selector_v22_3_grid.pth"
)
OUTPUT_DIR = REPO_ROOT / "results" / "four_link_three_expert_training"
OUTPUT_MODEL = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "three_expert_selector_v22_3"
    / "V22_3_three_expert_selector.pth"
)

LABEL_NAMES = ("positive", "negative", "active")
EXPECTED_CANDIDATES = 75_600
EXPECTED_RELEASED_LABELS = 20_404
EXPECTED_RELEASED_VALIDATION_ACCURACY = 0.8645833730697632
ACCURACY_TOLERANCE = 1e-7

TRAIN_EPOCHS = 120
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAINING_SEED = 20260728


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_selector(state_dim: int, classes: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(state_dim, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, classes),
    )


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def resolved_three_expert_label(row: dict[str, str]) -> tuple[int, str]:
    """Return a complete three-class label without mislabelling sign success."""
    old_label = str(row.get("label", "")).strip()
    if old_label:
        return int(float(old_label)), f"released_{row['label_reason']}"

    pos_success = parse_bool(row["pos_success"])
    neg_success = parse_bool(row["neg_success"])
    if not pos_success and not neg_success:
        return 2, "neither_sign_expert_succeeded"
    if pos_success and not neg_success:
        return 0, "positive_success_only"
    if neg_success and not pos_success:
        return 1, "negative_success_only"

    pos_valid = parse_bool(row["pos_valid"])
    neg_valid = parse_bool(row["neg_valid"])
    pos_cmt = finite_float(row["pos_cmt"])
    neg_cmt = finite_float(row["neg_cmt"])
    if pos_valid and neg_valid and np.isfinite(pos_cmt) and np.isfinite(neg_cmt):
        if pos_cmt <= neg_cmt:
            return 0, "resolved_valid_cmt_tie_positive"
        return 1, "resolved_valid_cmt_tie_negative"
    if pos_valid:
        return 0, "both_success_positive_valid_only"
    if neg_valid:
        return 1, "both_success_negative_valid_only"

    # Both experts reached the endpoint but neither passed the Cmt denominator.
    # Choose the larger signed progress; use lower positive work as a stable tie
    # breaker.  This branch must never be labelled Active because a sign expert
    # did succeed.
    pos_score = (
        finite_float(row["pos_displacement_m"]),
        -finite_float(row["pos_energy_j"]),
    )
    neg_score = (
        finite_float(row["neg_displacement_m"]),
        -finite_float(row["neg_energy_j"]),
    )
    if pos_score >= neg_score:
        return 0, "both_success_progress_work_tiebreak_positive"
    return 1, "both_success_progress_work_tiebreak_negative"


def configure_reconstruction(evaluator) -> None:
    evaluator.CURRICULUM_PROGRESS = 1.0
    evaluator.RESET_PHASE_MODE = "random"
    evaluator.COMMAND_STEP_LENGTH_GRID_M = [0.30, 0.35, 0.40]
    evaluator.COMMAND_STEP_HEIGHT_GRID_M = [0.00, 0.01, 0.02]
    evaluator.commanded_step_length_options = np.asarray(
        evaluator.COMMAND_STEP_LENGTH_GRID_M, dtype=float
    )
    evaluator.commanded_step_height_options = np.asarray(
        evaluator.COMMAND_STEP_HEIGHT_GRID_M, dtype=float
    )
    evaluator.COMMAND_STEP_LENGTH_OVERRIDE_M = None
    evaluator.COMMAND_STEP_HEIGHT_OVERRIDE_M = None
    evaluator.update_curriculum(0, progress_override=1.0)


def reconstruct_dataset(evaluator):
    if not SOURCE_AUDIT_CSV.is_file():
        raise FileNotFoundError(SOURCE_AUDIT_CSV)
    configure_reconstruction(evaluator)

    features: list[np.ndarray] = []
    labels: list[int] = []
    old_labels: list[int] = []
    old_label_mask: list[bool] = []
    splits: list[str] = []
    reasons: list[str] = []
    phases_mismatched = 0
    episode_mismatched = 0
    current_seed: int | None = None

    with SOURCE_AUDIT_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            seed = int(row["seed"])
            if seed != current_seed:
                current_seed = seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            if int(row["episode_index"]) != row_number:
                episode_mismatched += 1
            walk_direction = float(row["walk_direction"])
            step_length = float(row["commanded_step_length_m"])
            step_height = float(row["commanded_step_height_m"])
            reconstructed_phase = evaluator.choose_eval_reset_phase(walk_direction)
            archived_phase = str(row["reset_phase"])
            if reconstructed_phase != archived_phase:
                phases_mismatched += 1

            initial_state = evaluator.sample_initial_state(
                walk_direction,
                reconstructed_phase,
                step_length,
                step_height,
            )
            state_net = evaluator.normalize_state(
                initial_state,
                0.0,
                walk_direction,
                step_length,
                step_height,
            )
            label, reason = resolved_three_expert_label(row)
            features.append(np.asarray(state_net, dtype=np.float32))
            labels.append(label)
            splits.append(str(row["split"]))
            reasons.append(reason)

            archived_label = str(row.get("label", "")).strip()
            if archived_label:
                old_label_mask.append(True)
                old_labels.append(int(float(archived_label)))
            else:
                old_label_mask.append(False)
                old_labels.append(-1)

    if episode_mismatched:
        raise RuntimeError(f"Episode-order mismatches: {episode_mismatched}")
    if phases_mismatched:
        raise RuntimeError(
            "The archived random sequence was not reconstructed exactly: "
            f"{phases_mismatched} reset-phase mismatches."
        )
    if len(features) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES} candidates, reconstructed {len(features)}"
        )
    if int(np.sum(old_label_mask)) != EXPECTED_RELEASED_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_RELEASED_LABELS} released labels, observed "
            f"{int(np.sum(old_label_mask))}"
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(splits, dtype=object),
        np.asarray(old_labels, dtype=np.int64),
        np.asarray(old_label_mask, dtype=bool),
        np.asarray(reasons, dtype=object),
    )


def verify_reconstruction(
    evaluator,
    features: np.ndarray,
    splits: np.ndarray,
    old_labels: np.ndarray,
    old_label_mask: np.ndarray,
) -> float:
    checkpoint = torch.load(
        RELEASED_TWO_CLASS_SELECTOR, map_location="cpu", weights_only=False
    )
    model = make_selector(evaluator.state_dim, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    validation_mask = (splits == "validation") & old_label_mask
    with torch.no_grad():
        logits = model(torch.as_tensor(features[validation_mask]))
        accuracy = float(
            (logits.argmax(dim=1).numpy() == old_labels[validation_mask]).mean()
        )
    expected = float(
        checkpoint.get(
            "validation_accuracy", EXPECTED_RELEASED_VALIDATION_ACCURACY
        )
    )
    if abs(accuracy - expected) > ACCURACY_TOLERANCE:
        raise RuntimeError(
            "Input reconstruction failed provenance check: "
            f"observed old-selector validation accuracy {accuracy:.12f}, "
            f"checkpoint reports {expected:.12f}."
        )
    return accuracy


def balanced_accuracy(
    predictions: np.ndarray, labels: np.ndarray, classes: int
) -> float:
    recalls = []
    for class_index in range(classes):
        mask = labels == class_index
        recalls.append(float(np.mean(predictions[mask] == class_index)))
    return float(np.mean(recalls))


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, classes: int
) -> list[list[int]]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    return matrix.tolist()


def train(
    evaluator,
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    epochs: int,
) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    x_train = torch.as_tensor(features[train_mask], device=device)
    y_train = torch.as_tensor(labels[train_mask], dtype=torch.long, device=device)
    x_validation = torch.as_tensor(features[validation_mask], device=device)
    y_validation = torch.as_tensor(
        labels[validation_mask], dtype=torch.long, device=device
    )

    train_counts = np.bincount(labels[train_mask], minlength=3).astype(float)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights /= class_weights.mean()
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = make_selector(evaluator.state_dim, 3).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state = None
    best_balanced_accuracy = -1.0
    best_validation_accuracy = -1.0
    best_epoch = -1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_losses = []
        for start in range(0, len(x_train), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            logits = model(x_train[batch])
            loss = loss_function(logits, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_predictions = (
                model(x_validation).argmax(dim=1).detach().cpu().numpy()
            )
        validation_labels = y_validation.detach().cpu().numpy()
        validation_accuracy = float(
            np.mean(validation_predictions == validation_labels)
        )
        validation_balanced_accuracy = balanced_accuracy(
            validation_predictions, validation_labels, 3
        )
        history_row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "validation_accuracy": validation_accuracy,
            "validation_balanced_accuracy": validation_balanced_accuracy,
        }
        history.append(history_row)
        if (
            validation_balanced_accuracy > best_balanced_accuracy
            or (
                validation_balanced_accuracy == best_balanced_accuracy
                and validation_accuracy >= best_validation_accuracy
            )
        ):
            best_balanced_accuracy = validation_balanced_accuracy
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:03d}: loss={history_row['loss']:.5f}, "
                f"val_acc={validation_accuracy:.4f}, "
                f"val_bal_acc={validation_balanced_accuracy:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    best_model = make_selector(evaluator.state_dim, 3)
    best_model.load_state_dict(best_state, strict=True)
    best_model.eval()
    with torch.no_grad():
        train_predictions = (
            best_model(torch.as_tensor(features[train_mask]))
            .argmax(dim=1)
            .numpy()
        )
        validation_predictions = (
            best_model(torch.as_tensor(features[validation_mask]))
            .argmax(dim=1)
            .numpy()
        )
    train_labels = labels[train_mask]
    validation_labels = labels[validation_mask]

    metrics = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "train_accuracy": float(np.mean(train_predictions == train_labels)),
        "train_balanced_accuracy": balanced_accuracy(
            train_predictions, train_labels, 3
        ),
        "train_confusion_matrix_actual_rows_predicted_columns": confusion_matrix(
            train_predictions, train_labels, 3
        ),
        "validation_confusion_matrix_actual_rows_predicted_columns": confusion_matrix(
            validation_predictions, validation_labels, 3
        ),
        "train_class_counts": {
            LABEL_NAMES[index]: int(count)
            for index, count in enumerate(
                np.bincount(train_labels, minlength=3)
            )
        },
        "validation_class_counts": {
            LABEL_NAMES[index]: int(count)
            for index, count in enumerate(
                np.bincount(validation_labels, minlength=3)
            )
        },
    }
    checkpoint = {
        "model_state_dict": best_state,
        "state_dim": int(evaluator.state_dim),
        "label_names": list(LABEL_NAMES),
        "transition_locked": True,
        "experts_frozen": True,
        "expert_policy_names": {
            "positive": evaluator.PASSIVE_POS_POLICY_NAME,
            "negative": evaluator.PASSIVE_NEG_POLICY_NAME,
            "active": evaluator.ACTIVE_POLICY_NAME,
        },
        "routing_rule": (
            "positive/negative when a sign-constrained expert succeeds; "
            "active when neither sign-constrained expert succeeds"
        ),
        "source_audit_csv": SOURCE_AUDIT_CSV.relative_to(REPO_ROOT).as_posix(),
        "source_audit_sha256": sha256(SOURCE_AUDIT_CSV),
        "released_two_class_selector": RELEASED_TWO_CLASS_SELECTOR.relative_to(
            REPO_ROOT
        ).as_posix(),
        "released_two_class_selector_sha256": sha256(
            RELEASED_TWO_CLASS_SELECTOR
        ),
        "training_seed": TRAINING_SEED,
        "epochs": int(epochs),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        **metrics,
    }
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="reconstruct and verify inputs without fitting the three-class gate",
    )
    args = parser.parse_args()

    required = (EVALUATOR_PATH, SOURCE_AUDIT_CSV, RELEASED_TWO_CLASS_SELECTOR)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")
    evaluator = load_module(EVALUATOR_PATH, "four_link_three_expert_training_env")
    (
        features,
        labels,
        splits,
        old_labels,
        old_label_mask,
        reasons,
    ) = reconstruct_dataset(evaluator)
    old_accuracy = verify_reconstruction(
        evaluator, features, splits, old_labels, old_label_mask
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "reconstructed_three_expert_dataset.npz",
        features=features,
        labels=labels,
        splits=splits.astype(str),
        label_names=np.asarray(LABEL_NAMES),
    )
    reconstruction_summary = {
        "candidate_count": int(len(labels)),
        "released_label_count": int(np.sum(old_label_mask)),
        "released_selector_validation_accuracy_reproduced": old_accuracy,
        "split_class_counts": {
            split: {
                LABEL_NAMES[index]: int(count)
                for index, count in enumerate(
                    np.bincount(labels[splits == split], minlength=3)
                )
            }
            for split in ("train", "validation")
        },
        "new_label_reason_counts": dict(Counter(map(str, reasons))),
        "source_audit_csv": SOURCE_AUDIT_CSV.relative_to(REPO_ROOT).as_posix(),
        "source_audit_sha256": sha256(SOURCE_AUDIT_CSV),
    }
    (OUTPUT_DIR / "reconstruction_summary.json").write_text(
        json.dumps(reconstruction_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(reconstruction_summary, indent=2), flush=True)
    if args.audit_only:
        return

    checkpoint = train(evaluator, features, labels, splits, args.epochs)
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(
            {key: value for key, value in checkpoint.items() if key != "model_state_dict"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model: {OUTPUT_MODEL}", flush=True)
    print(
        json.dumps(
            {
                "best_epoch": checkpoint["best_epoch"],
                "validation_accuracy": checkpoint[
                    "best_validation_accuracy"
                ],
                "validation_balanced_accuracy": checkpoint[
                    "best_validation_balanced_accuracy"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
