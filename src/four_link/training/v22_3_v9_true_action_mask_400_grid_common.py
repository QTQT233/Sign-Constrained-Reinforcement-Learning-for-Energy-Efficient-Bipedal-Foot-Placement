"""Shared PPO trainer for the true per-step hip-sign action-mask baselines.

The physical action table and all robot, reward, reset, command-grid, and
curriculum logic come from ``v22_3_v9_per_step_sign_400_grid.py``.  At every
simulation step this policy:

1. samples a latent hip sign z in {-1, +1};
2. masks every physical action whose non-zero hip sign disagrees with z;
3. samples the physical action from the renormalized masked distribution.

PPO uses log p(z | s) + log p(a | s, z), so the selector and masked physical
action are trained as one hierarchical action.  Merely zeroing probabilities
during rollout while updating an unmasked 27-action distribution would not be
a valid action-mask PPO implementation.
"""

import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import v22_3_v9_per_step_sign_400_grid as base


NEGATIVE_SIGN_INDEX = 0
POSITIVE_SIGN_INDEX = 1
SIGN_VALUES = (-1.0, 1.0)
VALID_ACTIONS_PER_SIGN = 18  # 9 zero-hip actions + 9 actions of the selected sign.


class TruePerStepActionMaskPolicy(torch.nn.Module):
    """Hierarchical sign selector followed by a hard-masked physical action."""

    def __init__(self, state_dim, hidden_size, actions):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size * 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Tanh(),
        )
        self.action_head = torch.nn.Linear(hidden_size, len(actions))
        self.sign_head = torch.nn.Linear(hidden_size, 2)
        self.register_buffer(
            "action_hip_signs",
            torch.as_tensor(np.asarray(actions)[:, 0], dtype=torch.float32),
        )

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Both experiments start with the same unbiased 0.5/0.5 selector.
        # The action trunk/head still use the original orthogonal initialization
        # or the legacy checkpoint in the warm-start experiment.
        torch.nn.init.zeros_(self.sign_head.weight)
        torch.nn.init.zeros_(self.sign_head.bias)

    def forward(self, states):
        features = self.trunk(states)
        return self.sign_head(features), self.action_head(features)

    def action_mask(self, sign_indices):
        """Return a Boolean [batch, action_dim] mask for sampled sign indices."""
        sign_indices = sign_indices.reshape(-1).long()
        selected_signs = torch.where(
            sign_indices == NEGATIVE_SIGN_INDEX,
            -torch.ones_like(sign_indices, dtype=self.action_hip_signs.dtype),
            torch.ones_like(sign_indices, dtype=self.action_hip_signs.dtype),
        )
        hip_signs = self.action_hip_signs.reshape(1, -1)
        return (hip_signs == 0.0) | (hip_signs == selected_signs.reshape(-1, 1))

    def masked_action_distribution(self, action_logits, sign_indices):
        mask = self.action_mask(sign_indices)
        masked_logits = action_logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=masked_logits)

    def sample(self, states):
        sign_logits, action_logits = self(states)
        sign_dist = torch.distributions.Categorical(logits=sign_logits)
        sign_indices = sign_dist.sample()
        action_dist = self.masked_action_distribution(action_logits, sign_indices)
        action_indices = action_dist.sample()
        joint_log_prob = sign_dist.log_prob(sign_indices) + action_dist.log_prob(action_indices)
        return sign_indices, action_indices, joint_log_prob

    def joint_log_prob_and_entropy(self, states, sign_indices, action_indices):
        """Return PPO joint log probability and exact latent-joint entropy.

        H[z, a] = H[z] + E_z H[a | z].  This is preferable to computing the
        entropy of the unmasked 27-action softmax, which would reward illegal
        opposite-sign actions.
        """
        sign_logits, action_logits = self(states)
        sign_dist = torch.distributions.Categorical(logits=sign_logits)
        action_dist = self.masked_action_distribution(action_logits, sign_indices)
        joint_log_prob = (
            sign_dist.log_prob(sign_indices.reshape(-1))
            + action_dist.log_prob(action_indices.reshape(-1))
        )

        negative_indices = torch.full_like(sign_indices.reshape(-1), NEGATIVE_SIGN_INDEX)
        positive_indices = torch.full_like(sign_indices.reshape(-1), POSITIVE_SIGN_INDEX)
        negative_entropy = self.masked_action_distribution(
            action_logits, negative_indices
        ).entropy()
        positive_entropy = self.masked_action_distribution(
            action_logits, positive_indices
        ).entropy()
        expected_action_entropy = (
            sign_dist.probs[:, NEGATIVE_SIGN_INDEX] * negative_entropy
            + sign_dist.probs[:, POSITIVE_SIGN_INDEX] * positive_entropy
        )
        joint_entropy = sign_dist.entropy() + expected_action_entropy
        return joint_log_prob, joint_entropy


def build_critic():
    critic = torch.nn.Sequential(
        torch.nn.Linear(base.state_dim, base.hidden_size * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(base.hidden_size * 2, base.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(base.hidden_size, 1),
    )
    for module in critic.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return critic


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")
    for key in ("state_dict", "policy_state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def _compatible_tensor(target, source):
    source = source.to(device=target.device, dtype=target.dtype)
    if target.shape == source.shape:
        return source, False
    can_expand_input = (
        target.ndim == 2
        and source.ndim == 2
        and target.shape[0] == source.shape[0]
        and target.shape[1] == source.shape[1] + 1
    )
    if can_expand_input:
        expanded = target.clone()
        expanded[:, : source.shape[1]] = source
        expanded[:, source.shape[1] :] = 0.0
        return expanded, True
    return None, False


def load_policy_checkpoint(policy, checkpoint_path):
    """Load either this hierarchical policy or the legacy Sequential policy."""
    source = _extract_state_dict(torch.load(checkpoint_path, map_location=base.device))
    target = policy.state_dict()
    expanded_keys = []
    skipped_keys = []

    is_hierarchical = any(key.startswith("trunk.") for key in source)
    if is_hierarchical:
        key_mapping = {key: key for key in source if key in target}
    else:
        # Legacy policy:
        # 0 = first Linear, 2 = second Linear, 4 = 27-action Linear.
        key_mapping = {
            "0.weight": "trunk.0.weight",
            "0.bias": "trunk.0.bias",
            "2.weight": "trunk.2.weight",
            "2.bias": "trunk.2.bias",
            "4.weight": "action_head.weight",
            "4.bias": "action_head.bias",
        }

    loaded_targets = set()
    for source_key, target_key in key_mapping.items():
        if source_key not in source or target_key not in target:
            continue
        compatible, expanded = _compatible_tensor(target[target_key], source[source_key])
        if compatible is None:
            skipped_keys.append(f"{source_key}->{target_key}")
            continue
        target[target_key] = compatible
        loaded_targets.add(target_key)
        if expanded:
            expanded_keys.append(target_key)

    required = {"trunk.0.weight", "trunk.2.weight", "action_head.weight"}
    missing_required = sorted(required - loaded_targets)
    if missing_required:
        raise RuntimeError(
            f"Policy checkpoint is incompatible; missing required mapped keys: {missing_required}"
        )
    policy.load_state_dict(target)
    return expanded_keys, skipped_keys, is_hierarchical


def load_critic_checkpoint(critic, checkpoint_path):
    source = _extract_state_dict(torch.load(checkpoint_path, map_location=base.device))
    target = critic.state_dict()
    expanded_keys = []
    skipped_keys = []
    loaded_targets = set()
    for key, source_tensor in source.items():
        if key not in target:
            skipped_keys.append(key)
            continue
        compatible, expanded = _compatible_tensor(target[key], source_tensor)
        if compatible is None:
            skipped_keys.append(key)
            continue
        target[key] = compatible
        loaded_targets.add(key)
        if expanded:
            expanded_keys.append(key)
    required = {"0.weight", "2.weight", "4.weight"}
    missing_required = sorted(required - loaded_targets)
    if missing_required:
        raise RuntimeError(
            f"Critic checkpoint is incompatible; missing required keys: {missing_required}"
        )
    critic.load_state_dict(target)
    return expanded_keys, skipped_keys


def configure_experiment(init_mode, old_policy_path=None, old_critic_path=None):
    if init_mode not in ("load_old", "scratch"):
        raise ValueError("init_mode must be 'load_old' or 'scratch'.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if init_mode == "load_old":
        output_name = "v22_3_v9_true_action_mask_load_old_400_grid"
        prefix = "V22_3V9TrueActionMaskLoadOld400Grid"
    else:
        output_name = "v22_3_v9_true_action_mask_scratch_400_grid"
        prefix = "V22_3V9TrueActionMaskScratch400Grid"

    output_dir = os.path.join(script_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    policy = TruePerStepActionMaskPolicy(
        base.state_dim, base.hidden_size, base.ACTIONS
    ).to(base.device)
    critic = build_critic().to(base.device)

    source_kind = "random_initialization"
    if init_mode == "load_old":
        if old_policy_path is None:
            old_policy_path = os.path.join(
                script_dir,
                "v22_3_v9_per_step_sign_400_grid",
                "V22_3V9PerStepSign400Grid_Policy_old.pth",
            )
        if old_critic_path is None:
            old_critic_path = os.path.join(
                script_dir,
                "v22_3_v9_per_step_sign_400_grid",
                "V22_3V9PerStepSign400Grid_Critic_old.pth",
            )
        missing = [path for path in (old_policy_path, old_critic_path) if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "The load-old experiment must not silently fall back to scratch. "
                f"Missing checkpoint(s): {missing}"
            )
        policy_expanded, policy_skipped, hierarchical = load_policy_checkpoint(
            policy, old_policy_path
        )
        critic_expanded, critic_skipped = load_critic_checkpoint(critic, old_critic_path)
        source_kind = "hierarchical_checkpoint" if hierarchical else "legacy_sequential_checkpoint"
        print(f"Loaded policy checkpoint: {old_policy_path}")
        print(f"Loaded critic checkpoint: {old_critic_path}")
        if policy_expanded or critic_expanded:
            print(
                "Expanded old input layer with one zero-initialized column: "
                f"policy={policy_expanded}, critic={critic_expanded}"
            )
        if policy_skipped or critic_skipped:
            print(
                "Skipped non-critical incompatible checkpoint keys: "
                f"policy={policy_skipped}, critic={critic_skipped}"
            )

    # Reuse the original save/curriculum/logging helpers without changing the
    # source baseline file or overwriting any active/passive checkpoints.
    base.policy = policy
    base.model = critic
    base.output_dir = output_dir
    base.checkpoint_prefix = prefix
    base.training_log_csv_path = os.path.join(output_dir, f"{prefix}_training_log.csv")
    base.training_log_jsonl_path = os.path.join(output_dir, f"{prefix}_training_log.jsonl")

    policy.train()
    critic.train()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    return {
        "policy": policy,
        "critic": critic,
        "policy_optimizer": policy_optimizer,
        "critic_optimizer": critic_optimizer,
        "output_dir": output_dir,
        "prefix": prefix,
        "init_mode": init_mode,
        "source_kind": source_kind,
        "old_policy_path": old_policy_path,
        "old_critic_path": old_critic_path,
    }


def normalized_state(state_raw):
    # In a per-step selector experiment, sign is a latent action rather than an
    # episode-level observation.  State element 8 therefore stays at zero, as
    # in the original per_step_selector baseline.
    return base.normalize_state(
        state_raw,
        0.0,
        base.env.walk_direction,
        base.env.commanded_step_length,
        base.env.commanded_step_height,
    )


def play_episode(policy, critic, policy_buffer, value_buffer, sign_counter):
    state_raw = base.env.reset()
    over = False
    episode_buffer = []

    while not over:
        state_net = normalized_state(state_raw)
        state_tensor = torch.as_tensor(
            state_net, dtype=torch.float32, device=base.device
        ).reshape(1, base.state_dim)
        with torch.no_grad():
            sign_tensor, action_tensor, joint_log_prob_tensor = policy.sample(state_tensor)
        sign_index = int(sign_tensor.item())
        action_index = int(action_tensor.item())
        joint_log_prob = float(joint_log_prob_tensor.item())

        selected_sign = SIGN_VALUES[sign_index]
        chosen_hip_sign = float(base.ACTIONS[action_index, 0])
        if chosen_hip_sign not in (0.0, selected_sign):
            raise RuntimeError(
                "Action-mask invariant violated: sampled an opposite-sign hip action."
            )
        sign_counter["negative" if sign_index == NEGATIVE_SIGN_INDEX else "positive"] += 1

        next_state_raw, reward, over = base.env.step(action_index)
        next_state_net = normalized_state(next_state_raw)
        episode_buffer.append(
            [
                state_net,
                action_index,
                reward,
                next_state_net,
                over,
                joint_log_prob,
                0.0,
                sign_index,
            ]
        )
        state_raw = np.copy(next_state_raw)

    with torch.no_grad():
        next_states = torch.as_tensor(
            np.array([item[3] for item in episode_buffer]),
            dtype=torch.float32,
            device=base.device,
        )
        states = torch.as_tensor(
            np.array([item[0] for item in episode_buffer]),
            dtype=torch.float32,
            device=base.device,
        )
        rewards = torch.as_tensor(
            np.array([item[2] for item in episode_buffer]).reshape(-1, 1),
            dtype=torch.float32,
            device=base.device,
        )
        dones = torch.as_tensor(
            np.array([item[4] for item in episode_buffer]).reshape(-1, 1),
            dtype=torch.float32,
            device=base.device,
        )
        td_target = rewards + base.gamma * (1.0 - dones) * critic(next_states)
        td_delta = (td_target - critic(states)).reshape(-1).cpu().numpy()

    advantages = []
    running_advantage = 0.0
    for delta in td_delta[::-1]:
        running_advantage = delta + base.gamma * base.gamma_gae * running_advantage
        advantages.append(running_advantage)
    advantages.reverse()
    for index, advantage in enumerate(advantages):
        episode_buffer[index][6] = advantage

    policy_buffer.extend(episode_buffer)
    value_buffer.extend(episode_buffer)
    for _ in range(base.concentrated_sample_times):
        value_buffer.append(episode_buffer[-1].copy())


def train_critic(critic, optimizer, value_buffer):
    last_loss = float("nan")
    for _ in range(base.critic_training_times):
        sample_size = min(base.batch_size, len(value_buffer))
        batch = random.sample(value_buffer, sample_size)
        states = torch.as_tensor(
            np.array([item[0] for item in batch]), dtype=torch.float32, device=base.device
        )
        rewards = torch.as_tensor(
            np.array([item[2] for item in batch]).reshape(-1, 1),
            dtype=torch.float32,
            device=base.device,
        )
        next_states = torch.as_tensor(
            np.array([item[3] for item in batch]), dtype=torch.float32, device=base.device
        )
        dones = torch.as_tensor(
            np.array([item[4] for item in batch]).reshape(-1, 1),
            dtype=torch.float32,
            device=base.device,
        )
        for _ in range(base.critic_training_steps):
            values = critic(states)
            with torch.no_grad():
                targets = rewards + base.gamma * (1.0 - dones) * critic(next_states)
            loss = base.loss_fn(values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return last_loss


def train_policy(policy, optimizer, policy_buffer):
    states = torch.as_tensor(
        np.array([item[0] for item in policy_buffer]),
        dtype=torch.float32,
        device=base.device,
    )
    actions = torch.as_tensor(
        np.array([item[1] for item in policy_buffer]),
        dtype=torch.long,
        device=base.device,
    )
    old_joint_log_probs = torch.as_tensor(
        np.array([item[5] for item in policy_buffer]),
        dtype=torch.float32,
        device=base.device,
    )
    sign_indices = torch.as_tensor(
        np.array([item[7] for item in policy_buffer]),
        dtype=torch.long,
        device=base.device,
    )
    advantages = np.asarray(
        [item[6] for item in policy_buffer], dtype=float
    ).reshape(-1)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=base.device)

    last_loss = float("nan")
    last_entropy = float("nan")
    last_ratio_mean = float("nan")
    for _ in range(base.actor_training_times):
        new_joint_log_probs, joint_entropy = policy.joint_log_prob_and_entropy(
            states, sign_indices, actions
        )
        ratio = torch.exp(new_joint_log_probs - old_joint_log_probs)
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(
            ratio, 1.0 - base.ppo_clip, 1.0 + base.ppo_clip
        ) * advantages
        entropy = joint_entropy.mean()
        loss = (
            -torch.min(surrogate_1, surrogate_2).mean()
            - base.policy_entropy_coefficient * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        last_entropy = float(entropy.item())
        last_ratio_mean = float(ratio.mean().item())
    return last_loss, last_entropy, last_ratio_mean


def _reason_counts():
    reason_counts = dict(Counter(base.terminal_reasons))
    directional_reason_counts = Counter()
    for walk_direction, reason in base.directional_terminal_reasons:
        direction = "forward" if walk_direction >= 0.0 else "backward"
        directional_reason_counts[f"{direction}_{reason}"] += 1
    directional_phase_reason_counts = Counter()
    for walk_direction, reset_phase, reason in base.directional_phase_terminal_reasons:
        direction = "forward" if walk_direction >= 0.0 else "backward"
        directional_phase_reason_counts[f"{direction}_{reset_phase}_{reason}"] += 1
    return reason_counts, dict(directional_reason_counts), dict(directional_phase_reason_counts)


def train(init_mode, old_policy_path=None, old_critic_path=None):
    config = configure_experiment(init_mode, old_policy_path, old_critic_path)
    policy = config["policy"]
    critic = config["critic"]
    policy_optimizer = config["policy_optimizer"]
    critic_optimizer = config["critic_optimizer"]
    output_dir = config["output_dir"]
    prefix = config["prefix"]

    # Structural validation before the long training loop.
    with torch.no_grad():
        test_signs = torch.tensor(
            [NEGATIVE_SIGN_INDEX, POSITIVE_SIGN_INDEX], device=base.device
        )
        valid_counts = policy.action_mask(test_signs).sum(dim=1).cpu().tolist()
    if valid_counts != [VALID_ACTIONS_PER_SIGN, VALID_ACTIONS_PER_SIGN]:
        raise RuntimeError(f"Unexpected action-mask sizes: {valid_counts}")

    print(f"Training true action-mask experiment: {init_mode}")
    print(f"Output directory: {output_dir}")
    print(
        "Mask protocol: refresh sign every simulation step; "
        "18/27 actions valid after each sign selection."
    )
    print(
        f"Comparison settings inherited from per_step/bi: "
        f"action_value_weight={base.action_value_weight}, "
        f"entropy_coefficient={base.policy_entropy_coefficient}."
    )
    best_success_by_curriculum = (
        base.load_existing_curriculum_bests() if init_mode == "load_old" else {}
    )

    for epoch in range(base.episode):
        curriculum_progress = base.update_curriculum(
            epoch,
            base.curriculum_progress_value if base.curriculum_adaptive else None,
        )
        base.reset_epoch_phase_quotas(curriculum_progress)
        base.success = []
        base.terminal_reasons = []
        base.directional_terminal_reasons = []
        base.directional_phase_terminal_reasons = []
        base.reset_readiness_records = []
        policy_buffer = []
        value_buffer = []
        sign_counter = Counter()

        for _ in tqdm(range(base.playing_times)):
            play_episode(policy, critic, policy_buffer, value_buffer, sign_counter)

        reason_counts, directional_reason_counts, phase_reason_counts = _reason_counts()
        success_count = len(base.success)
        success_rate = success_count / base.playing_times
        bucket = base.curriculum_bucket(curriculum_progress)
        if epoch > 1:
            base.save_curriculum_best(
                bucket, success_count, best_success_by_curriculum, reason_counts
            )

        smoothed_success_rate = base.update_adaptive_curriculum_progress(
            success_rate, reason_counts, phase_reason_counts
        )
        q1_rate, q3_rate, clearance_rate, ground_rate, speed_rate = (
            base.curriculum_failure_rates(reason_counts)
        )
        sign_total = max(1, sign_counter["negative"] + sign_counter["positive"])

        critic_loss = train_critic(critic, critic_optimizer, value_buffer)
        policy_loss, joint_entropy, ratio_mean = train_policy(
            policy, policy_optimizer, policy_buffer
        )

        torch.save(policy.state_dict(), os.path.join(output_dir, f"{prefix}_Policy.pth"))
        torch.save(critic.state_dict(), os.path.join(output_dir, f"{prefix}_Critic.pth"))

        print(
            f"epoch {epoch + 1}: curriculum={curriculum_progress:.2f}, "
            f"success={success_count}/{base.playing_times}, "
            f"sign(-/+)={sign_counter['negative']}/{sign_counter['positive']}, "
            f"policy_loss={policy_loss:.6f}, critic_loss={critic_loss:.6f}"
        )
        print(f"terminal reasons {reason_counts}")
        print(f"directional terminal reasons {directional_reason_counts}")

        base.append_training_log(
            {
                "epoch": epoch + 1,
                "script": os.path.basename(sys.argv[0]),
                "checkpoint_prefix": prefix,
                "experiment": "true_per_step_action_mask",
                "initialization": init_mode,
                "checkpoint_source_kind": config["source_kind"],
                "old_policy_checkpoint": config["old_policy_path"],
                "old_critic_checkpoint": config["old_critic_path"],
                "selector_refresh": "every_simulation_step",
                "mask_rule": "hip_sign == 0 or hip_sign == selected_sign",
                "action_dim_before_mask": int(base.action_dim),
                "valid_actions_after_mask": int(VALID_ACTIONS_PER_SIGN),
                "ppo_probability": "joint_selector_and_masked_action",
                "learned_sign_selector": True,
                "action_value_weight": float(base.action_value_weight),
                "policy_entropy_coefficient": float(
                    base.policy_entropy_coefficient
                ),
                "curriculum_progress": float(curriculum_progress),
                "curriculum_bucket": int(bucket),
                "angle_settle_deg": float(np.rad2deg(base.angle_settle)),
                "distance_settle_m": float(base.distance_settle),
                "height_settle_m": float(base.height_settle),
                "playing_times": int(base.playing_times),
                "success_count": int(success_count),
                "success_rate": float(success_rate),
                "success_rate_ema": float(smoothed_success_rate),
                "q1_failure_rate": float(q1_rate),
                "q3_failure_rate": float(q3_rate),
                "clearance_failure_rate": float(clearance_rate),
                "ground_failure_rate": float(ground_rate),
                "speed_failure_rate": float(speed_rate),
                "negative_sign_count": int(sign_counter["negative"]),
                "positive_sign_count": int(sign_counter["positive"]),
                "negative_sign_fraction": float(sign_counter["negative"] / sign_total),
                "positive_sign_fraction": float(sign_counter["positive"] / sign_total),
                "policy_loss": float(policy_loss),
                "critic_loss": float(critic_loss),
                "joint_entropy": float(joint_entropy),
                "ppo_ratio_mean": float(ratio_mean),
                "terminal_reasons": reason_counts,
                "directional_terminal_reasons": directional_reason_counts,
                "directional_phase_terminal_reasons": phase_reason_counts,
                "reset_phase_quota_targets": base.reset_phase_quota_targets,
                "reset_readiness_summary": base.format_reset_readiness_records(),
                "curriculum_decision": base.curriculum_last_progress_debug.get(
                    "decision", "n/a"
                ),
                "curriculum_step": float(
                    base.curriculum_last_progress_debug.get("step", 0.0)
                ),
            }
        )


__all__ = ["train", "TruePerStepActionMaskPolicy"]
