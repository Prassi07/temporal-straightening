#!/usr/bin/env python3
"""
Collect Corridor Push-T rollouts (8-D spline actions, per-step state labels).

Mixed collection (default 70% expert / 30% exploration):
  - Expert: geometric cubic Bézier from agent → block → goal (see env.pusht.corridor_expert).
  - Exploration: large noise on expert splines and/or uniform random splines to cover jams/failures.

Actions are open-loop chunked: the policy commits a new spline every --chunk-size steps and
executes it open-loop for that many steps.  Both absolute (world-frame) and robot-centric
(relative to agent position at commit time) representations are saved.

Per-step labels saved: goal_reached, wall_contact, distances, goal_poses.
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
    p.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        metavar="K",
        help="Open-loop chunk length: policy commits a new spline every K steps (default 4).",
    )
    # --- Randomization controls (mirrored from visualize_corridor_pusht) ---
    p.add_argument(
        "--no-random-goal",
        action="store_true",
        help="Disable per-reset random goal; keep fixed goal [256,256,pi/4].",
    )
    p.add_argument(
        "--no-random-corridor-width",
        action="store_true",
        help="Disable corridor half-width resampling every N episodes.",
    )
    p.add_argument(
        "--corridor-width-every-n",
        type=int,
        default=None,
        metavar="N",
        help="Resample corridor half-width every N episodes (default: env default = 3).",
    )
    p.add_argument("--corridor-half-width-min", type=float, default=None, help="Min half-width when randomizing (px)")
    p.add_argument("--corridor-half-width-max", type=float, default=None, help="Max half-width when randomizing (px)")
    p.add_argument(
        "--initial-corridor-half-width",
        type=float,
        default=None,
        help="Starting corridor half-width before any randomization.",
    )
    # --- Aleatoric block dynamics ---
    p.add_argument(
        "--com-shift-range",
        type=float,
        default=None,
        metavar="R",
        help="Max per-axis CoM shift each reset (px). Default: env default (28).",
    )
    p.add_argument(
        "--friction-low",
        type=float,
        default=None,
        help="Lower bound of uniform friction draw each reset. Default: env default (0.15).",
    )
    p.add_argument(
        "--friction-high",
        type=float,
        default=None,
        help="Upper bound of uniform friction draw each reset. Default: env default (1.05).",
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
    chunk_size: int = 1,
):
    """Run one episode and return all logged arrays.

    Actions are open-loop chunked: a new spline is committed every ``chunk_size``
    steps and then replayed open-loop for the remaining steps in the chunk.

    Returns
    -------
    frames         : (T, H, W, 3) uint8
    states_before  : (T, S)       float32  — state before each step
    act_global     : (T, 8)       float32  — absolute control points (world frame)
    act_relative   : (T, 8)       float32  — CPs relative to agent at commit time
    goal_poses     : (T, 3)       float32  — [goal_x, goal_y, goal_theta] at each step
    vels           : (T, 2)       float32
    goal_reached   : (T,)         float32
    wall_contact   : (T,)         float32
    distances      : (T,)         float32
    """
    margin = 20.0
    ws = float(env.window_size)
    obs, _info = env.reset()
    frames = []
    states_before = []
    act_global = []
    act_relative = []
    goal_poses = []
    vels = []
    goal_reached_flags = []
    wall_contact_flags = []
    distances = []

    current_action: np.ndarray | None = None
    commit_agent_xy: np.ndarray | None = None

    for t in range(horizon):
        img = obs["visual"]
        state = env._get_obs()
        frames.append(img)
        states_before.append(state.astype(np.float32))

        # Commit a new action at the start of each chunk
        if t % chunk_size == 0:
            commit_agent_xy = state[0:2].copy()
            current_action = _sample_action_for_timestep(
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

        act_global.append(current_action.astype(np.float32))

        # Relative: subtract the agent position at commit time from each CP
        ctrl_rel = current_action.reshape(4, 2) - commit_agent_xy
        act_relative.append(ctrl_rel.reshape(8).astype(np.float32))

        goal_poses.append(env.goal_pose.astype(np.float32))  # (3,)

        obs, _r, _term, _trunc, info = env.step(current_action)
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
        np.stack(act_global, axis=0),
        np.stack(act_relative, axis=0),
        np.stack(goal_poses, axis=0),
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


def build_env_kwargs(args) -> dict:
    kw: dict = {
        "with_velocity": True,
        "render_size": args.render_size,
        "relative": False,
        "action_scale": 1.0,
        "randomize_goal_pose": not args.no_random_goal,
        "randomize_corridor_width": not args.no_random_corridor_width,
        # Never draw overlays in training data
        "draw_goal_radius": False,
    }
    if args.corridor_width_every_n is not None:
        kw["corridor_width_episodes_per_setting"] = max(1, int(args.corridor_width_every_n))
    if args.corridor_half_width_min is not None:
        kw["corridor_half_width_min"] = float(args.corridor_half_width_min)
    if args.corridor_half_width_max is not None:
        kw["corridor_half_width_max"] = float(args.corridor_half_width_max)
    if args.initial_corridor_half_width is not None:
        kw["corridor_half_width"] = float(args.initial_corridor_half_width)
    if args.com_shift_range is not None:
        kw["com_shift_range"] = float(args.com_shift_range)
    if args.friction_low is not None:
        kw["friction_low"] = float(args.friction_low)
    if args.friction_high is not None:
        kw["friction_high"] = float(args.friction_high)
    if args.success_radius_px is not None:
        kw["success_radius_px"] = float(args.success_radius_px)
    return kw


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    base = Path(args.out)
    n_train = int(args.n_episodes * args.train_ratio)
    n_val = args.n_episodes - n_train

    env = CorridorPushTEnv(**build_env_kwargs(args))

    success_radius = getattr(env, "_success_radius_px", None)
    goal_mode = f"radius={success_radius:.1f}px" if success_radius is not None else f"coverage>={args.expert_hold_coverage}"
    print(
        f"\n=== CorridorPushT collect ===\n"
        f"  output          : {Path(args.out).resolve()}\n"
        f"  episodes        : {args.n_episodes}  (train={n_train}  val={n_val})\n"
        f"  horizon         : {args.horizon}  chunk_size={args.chunk_size}\n"
        f"  seed            : {args.seed}\n"
        f"  expert_ratio    : {args.expert_ratio}  jitter_sigma={args.expert_jitter_sigma}\n"
        f"  explore         : perturb_prob={args.explore_perturb_prob}  noise_sigma={args.explore_noise_sigma}\n"
        f"  goal_mode       : {goal_mode}\n"
        f"  random_goal     : {env.randomize_goal_pose}\n"
        f"  corridor_width  : initial={env.corridor_half_width:.1f}  "
        f"random={env.randomize_corridor_width}  "
        f"range=[{env.corridor_half_width_min:.1f}, {env.corridor_half_width_max:.1f}]  "
        f"every={env.corridor_width_episodes_per_setting} episodes\n"
        f"  aleatoric       : com_shift_range={env.com_shift_range:.1f}  "
        f"friction=[{env.friction_low:.2f}, {env.friction_high:.2f}]\n"
        f"  agent           : max_speed={env.agent_max_speed:.1f}  "
        f"ou_theta={env.ou_theta:.2f}  ou_sigma={env.ou_sigma:.2f}\n"
        f"  render_size     : {args.render_size}  fps={args.fps}\n"
    )

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
        all_act_global = []
        all_act_relative = []
        all_goal_poses = []
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
            vid, st, act_g, act_r, gp, vel, goal_r, wall_c, dists = collect_one_episode(
                env,
                args.horizon,
                rng,
                episode_mode=mode,
                expert_jitter_sigma=args.expert_jitter_sigma,
                explore_perturb_prob=args.explore_perturb_prob,
                explore_noise_sigma=args.explore_noise_sigma,
                expert_hold_coverage=args.expert_hold_coverage,
                chunk_size=args.chunk_size,
            )
            write_video(obs_dir / f"episode_{i:03d}.mp4", vid, args.fps)
            all_states.append(torch.from_numpy(st))
            all_act_global.append(torch.from_numpy(act_g))
            all_act_relative.append(torch.from_numpy(act_r))
            all_goal_poses.append(torch.from_numpy(gp))
            all_vels.append(torch.from_numpy(vel))
            all_goal_reached.append(torch.from_numpy(goal_r[:, None]))   # (T,1)
            all_wall_contact.append(torch.from_numpy(wall_c[:, None]))   # (T,1)
            all_distances.append(torch.from_numpy(dists[:, None]))       # (T,1)
            seq_lengths.append(args.horizon)

        torch.save(torch.stack(all_states, dim=0), split_dir / "states.pth")
        # Absolute control points (world frame) and robot-centric control points
        torch.save(torch.stack(all_act_global, dim=0), split_dir / "actions_global.pth")
        torch.save(torch.stack(all_act_relative, dim=0), split_dir / "actions_relative.pth")
        # Per-step goal target: [goal_x, goal_y, goal_theta]
        torch.save(torch.stack(all_goal_poses, dim=0), split_dir / "goal_poses.pth")
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
            f"(chunk_size={args.chunk_size}  expert={n_ex}  success={n_success}  wall_contact_ep={n_jam})"
        )

    print("Done.")


if __name__ == "__main__":
    main()
