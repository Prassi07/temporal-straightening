#!/usr/bin/env python3
"""
Collect Corridor Push-T rollouts (8-D spline actions, per-step state labels).

Mixed collection (default 70% expert / 30% exploration):
  - Expert: geometric cubic Bézier from agent → block → goal (see env.pusht.corridor_expert).
  - Exploration: large noise on expert splines and/or uniform random splines to cover jams/failures.

Per-step state labels saved: goal_reached, wall_contact, block_goal_distance.
Downstream code computes reward shaping from these raw signals.

Example:
  python scripts/collect_corridor_pusht.py --out data/corridor_pusht --n-episodes 10000 --horizon 50
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.pusht.corridor_expert import (  # noqa: E402
    compute_expert_spline_8d,
    hold_agent_spline_8d,
    jitter_expert_spline,
    perturb_spline_exploration,
    sample_uniform_random_spline_8d,
)
from env.pusht.corridor_pusht_env import CorridorPushTEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/corridor_pusht", help="Base output (train/val created inside)")
    p.add_argument("--n-episodes", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-size", type=int, default=224)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--expert-ratio",
        type=float,
        default=0.7,
        help="Fraction of episodes using heuristic expert actions (rest: noisy exploration).",
    )
    p.add_argument(
        "--expert-jitter-sigma",
        type=float,
        default=6.0,
        help="Std dev (pixels) of Gaussian jitter on expert control points for diversity.",
    )
    p.add_argument(
        "--explore-perturb-prob",
        type=float,
        default=0.55,
        help="Within exploration episodes, probability of large-perturb expert vs fully random spline.",
    )
    p.add_argument(
        "--explore-noise-sigma",
        type=float,
        default=110.0,
        help="Std dev (pixels) for exploration perturbations of the expert spline.",
    )
    p.add_argument(
        "--expert-hold-coverage",
        type=float,
        default=0.6,
        help="Overlap mode: hold when coverage >= this. Set < 0 to disable. Ignored if --success-radius-px set.",
    )
    p.add_argument(
        "--success-radius-px",
        type=float,
        default=None,
        metavar="R",
        help="Position-based goal: success/hold when block center within R px of goal center.",
    )
    return p.parse_args()


def _sample_action_for_timestep(
    env: CorridorPushTEnv,
    rng: np.random.RandomState,
    episode_mode: str,
    margin: float,
    ws: float,
    expert_jitter_sigma: float,
    explore_perturb_prob: float,
    explore_noise_sigma: float,
    expert_hold_coverage: float,
) -> np.ndarray:
    """Build 8-D spline for one env.step() given current poses on env."""
    state = env._get_obs()
    agent_xy = state[0:2]
    block_xy = state[2:4]
    goal_xy = env.goal_pose[:2]

    expert = compute_expert_spline_8d(agent_xy, block_xy, goal_xy, margin, ws)

    if episode_mode == "expert":
        use_position = getattr(env, "_success_radius_px", None) is not None
        should_hold = env.is_goal_reached() if use_position else (
            expert_hold_coverage >= 0.0 and env.measure_block_goal_coverage() >= expert_hold_coverage
        )
        if should_hold:
            return hold_agent_spline_8d(agent_xy, margin, ws)
        if expert_jitter_sigma > 0:
            return jitter_expert_spline(expert, rng, expert_jitter_sigma, margin, ws)
        return expert

    # exploration
    if rng.rand() < explore_perturb_prob:
        return perturb_spline_exploration(expert, rng, explore_noise_sigma, margin, ws)
    return sample_uniform_random_spline_8d(rng, margin, ws)


def collect_one_episode(
    env: CorridorPushTEnv,
    horizon: int,
    rng: np.random.RandomState,
    episode_mode: str,
    expert_jitter_sigma: float,
    explore_perturb_prob: float,
    explore_noise_sigma: float,
    expert_hold_coverage: float,
):
    margin = 20.0
    ws = float(env.window_size)
    obs, _info = env.reset()
    frames = []
    states_before = []
    actions = []
    vels = []
    goal_reached_flags = []   # bool per step: was goal reached after this action?
    wall_contact_flags = []   # bool per step: wall contact during this physics step?
    distances = []            # float per step: block-to-goal distance after action

    for _t in range(horizon):
        img = obs["visual"]
        state = env._get_obs()
        frames.append(img)
        states_before.append(state.astype(np.float32))

        a = _sample_action_for_timestep(
            env,
            rng,
            episode_mode,
            margin,
            ws,
            expert_jitter_sigma,
            explore_perturb_prob,
            explore_noise_sigma,
            expert_hold_coverage,
        )
        actions.append(a)
        obs, _r, _term, _trunc, info = env.step(a)
        if env.with_velocity:
            vels.append(state[5:7].astype(np.float32))
        else:
            vels.append(np.array(env.agent.velocity, dtype=np.float32))

        goal_reached_flags.append(float(info.get("goal_reached", False)))
        wall_contact_flags.append(float(info.get("corridor_contact", False)))
        distances.append(float(info.get("block_goal_distance", 0.0)))

    return (
        np.stack(frames, axis=0),
        np.stack(states_before, axis=0),
        np.stack(actions, axis=0),
        np.stack(vels, axis=0),
        np.array(goal_reached_flags, dtype=np.float32),
        np.array(wall_contact_flags, dtype=np.float32),
        np.array(distances, dtype=np.float32),
    )


def write_video(path: Path, frames: np.ndarray, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w = frames.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for i in range(t):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    base = Path(args.out)
    n_train = int(args.n_episodes * args.train_ratio)
    n_val = args.n_episodes - n_train

    env_kw: dict = {
        "with_velocity": True,
        "render_size": args.render_size,
        "relative": False,
        "action_scale": 1.0,
    }
    if getattr(args, "success_radius_px", None) is not None:
        env_kw["success_radius_px"] = float(args.success_radius_px)
    env = CorridorPushTEnv(**env_kw)

    def episode_modes_for_split(n_ep: int, split_rng: np.random.RandomState):
        modes = []
        for _ in range(n_ep):
            modes.append("expert" if split_rng.rand() < args.expert_ratio else "explore")
        return modes

    for split, n_ep in (("train", n_train), ("val", n_val)):
        split_dir = base / split
        obs_dir = split_dir / "obses"
        obs_dir.mkdir(parents=True, exist_ok=True)

        all_states = []
        all_actions = []
        all_vels = []
        all_goal_reached = []
        all_wall_contact = []
        all_distances = []
        seq_lengths = []
        mix_labels = []

        split_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))
        modes = episode_modes_for_split(n_ep, split_rng)

        for i in range(n_ep):
            env.seed(int(rng.randint(0, 2**31 - 1)))
            mode = modes[i]
            mix_labels.append(mode)
            vid, st, act, vel, goal_r, wall_c, dists = collect_one_episode(
                env,
                args.horizon,
                rng,
                episode_mode=mode,
                expert_jitter_sigma=args.expert_jitter_sigma,
                explore_perturb_prob=args.explore_perturb_prob,
                explore_noise_sigma=args.explore_noise_sigma,
                expert_hold_coverage=args.expert_hold_coverage,
            )
            write_video(obs_dir / f"episode_{i:03d}.mp4", vid, args.fps)
            all_states.append(torch.from_numpy(st))
            all_actions.append(torch.from_numpy(act))
            all_vels.append(torch.from_numpy(vel))
            all_goal_reached.append(torch.from_numpy(goal_r[:, None]))   # (T,1)
            all_wall_contact.append(torch.from_numpy(wall_c[:, None]))   # (T,1)
            all_distances.append(torch.from_numpy(dists[:, None]))       # (T,1)
            seq_lengths.append(args.horizon)

        torch.save(torch.stack(all_states, dim=0), split_dir / "states.pth")
        torch.save(torch.stack(all_actions, dim=0), split_dir / "actions.pth")
        torch.save(torch.stack(all_vels, dim=0), split_dir / "velocities.pth")
        # Per-step raw signals — downstream code derives rewards from these
        torch.save(torch.stack(all_goal_reached, dim=0), split_dir / "goal_reached.pth")
        torch.save(torch.stack(all_wall_contact, dim=0), split_dir / "wall_contact.pth")
        torch.save(torch.stack(all_distances, dim=0), split_dir / "distances.pth")
        with open(split_dir / "seq_lengths.pkl", "wb") as f:
            pickle.dump(seq_lengths, f)
        shapes = ["T"] * n_ep
        with open(split_dir / "shapes.pkl", "wb") as f:
            pickle.dump(shapes, f)
        with open(split_dir / "episode_mix.pkl", "wb") as f:
            pickle.dump(mix_labels, f)

        n_ex = sum(1 for m in mix_labels if m == "expert")
        n_success = sum(1 for gr in all_goal_reached if gr.any())
        n_jam = sum(1 for wc in all_wall_contact if wc.any())
        print(
            f"Wrote {split}: {n_ep} episodes → {split_dir}  "
            f"(expert={n_ex}  success={n_success}  wall_contact_ep={n_jam})"
        )

    print("Done.")


if __name__ == "__main__":
    main()
