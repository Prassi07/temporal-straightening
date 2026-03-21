#!/usr/bin/env python3
"""
Visualize Corridor Push-T for a few episodes (sanity check).

Goal is rendered as a green circle (blue when not reached, green when reached).
The Bézier control polygon is drawn in orange each step.

  # Live window
  python scripts/visualize_corridor_pusht.py --mode human --episodes 3

  # Headless MP4
  python scripts/visualize_corridor_pusht.py --mode save --out /tmp/corridor_viz --episodes 3

  # Step-by-step with per-step distance log
  python scripts/visualize_corridor_pusht.py --debug-steps --episodes 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from env.pusht.corridor_expert import (  # noqa: E402
    compute_expert_spline_8d,
    hold_agent_spline_8d,
    jitter_expert_spline,
)
from env.pusht.corridor_pusht_env import CorridorPushTEnv  # noqa: E402

# Must match CorridorPushTEnv.step control-point clipping
ACTION_MARGIN = 18.0


def action_is_hold(a: np.ndarray) -> bool:
    ac = a.reshape(4, 2)
    return bool(np.max(np.abs(ac - ac[0:1])) < 1e-5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("human", "save"), default="save", help="human = pygame window; save = MP4 files")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--seed", type=int, default=None, help="RNG seed; default None = random seed each run.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--render-size", type=int, default=512, help="Resolution (larger = clearer preview)")
    p.add_argument("--out", type=str, default="viz_corridor_pusht", help="Output directory for --mode save")
    p.add_argument("--jitter", type=float, default=0.0, help="Expert control-point jitter (0 = none)")
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
        help="Position-based goal: success/hold when block center within R px of goal center. Overrides overlap.",
    )
    p.add_argument(
        "--no-random-goal",
        action="store_true",
        help="Disable per-reset random goal; keep parent fixed goal [256,256,pi/4].",
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
        help="Resample corridor half-width every N episodes (default: env default, usually 3).",
    )
    p.add_argument("--corridor-half-width-min", type=float, default=None, help="Min half-width when randomizing (px)")
    p.add_argument("--corridor-half-width-max", type=float, default=None, help="Max half-width when randomizing (px)")
    p.add_argument(
        "--initial-corridor-half-width",
        type=float,
        default=None,
        help="Starting corridor half-width before any randomization (default: env 72).",
    )
    p.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each env.step (0 = no delay). Useful with --log-every-step.",
    )
    p.add_argument(
        "--log-every-step",
        action="store_true",
        help="Print coverage, hold vs expert, reward, corridor wall flags each timestep.",
    )
    p.add_argument(
        "--debug-steps",
        action="store_true",
        help="Shorthand: --log-every-step --step-delay 0.25",
    )
    return p.parse_args()


def build_env_kwargs(args) -> dict:
    kw: dict = {
        "with_velocity": True,
        "render_size": args.render_size,
        "relative": False,
        "action_scale": 1.0,
        "randomize_goal_pose": not args.no_random_goal,
        "randomize_corridor_width": not args.no_random_corridor_width,
        "draw_goal_radius": True,
    }
    if args.corridor_width_every_n is not None:
        kw["corridor_width_episodes_per_setting"] = max(1, int(args.corridor_width_every_n))
    if args.corridor_half_width_min is not None:
        kw["corridor_half_width_min"] = float(args.corridor_half_width_min)
    if args.corridor_half_width_max is not None:
        kw["corridor_half_width_max"] = float(args.corridor_half_width_max)
    if args.initial_corridor_half_width is not None:
        kw["corridor_half_width"] = float(args.initial_corridor_half_width)
    # Only override the env's default (40 px) when user explicitly passes a value
    if getattr(args, "success_radius_px", None) is not None:
        kw["success_radius_px"] = float(args.success_radius_px)
    return kw


def expert_action(
    env: CorridorPushTEnv,
    rng: np.random.RandomState,
    margin: float,
    ws: float,
    jitter: float,
    expert_hold_coverage: float,
) -> np.ndarray:
    """Same policy logic as collect_corridor_pusht expert branch (hold vs Bézier + jitter)."""
    s = env._get_obs()
    agent_xy = s[0:2]
    block_xy = s[2:4]
    goal_xy = env.goal_pose[:2]

    expert = compute_expert_spline_8d(agent_xy, block_xy, goal_xy, margin, ws)

    use_position = getattr(env, "_success_radius_px", None) is not None
    should_hold = env.is_goal_reached() if use_position else (
        expert_hold_coverage >= 0.0 and env.measure_block_goal_coverage() >= expert_hold_coverage
    )
    if should_hold:
        return hold_agent_spline_8d(agent_xy, margin, ws)
    if jitter > 0:
        return jitter_expert_spline(expert, rng, jitter, margin, ws)

    return expert


def show_frame(env, rgb_img: np.ndarray, fps: int) -> None:
    """Blit an already-annotated RGB numpy image to the pygame window."""
    import pygame
    if env.window is None:
        env.window = pygame.display.set_mode(rgb_img.shape[1::-1])
    if env.clock is None:
        env.clock = pygame.time.Clock()
    # numpy HxWxC RGB → pygame wants WxH, so transpose axes 1,0
    surf = pygame.surfarray.make_surface(rgb_img.transpose(1, 0, 2))
    if surf.get_size() != env.window.get_size():
        surf = pygame.transform.scale(surf, env.window.get_size())
    env.window.blit(surf, (0, 0))
    pygame.display.update()
    pygame.event.pump()
    env.clock.tick(fps)


def write_mp4(path: Path, frames: list, fps: float, size_hw: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = size_hw
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h))
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    args = parse_args()
    if args.debug_steps:
        args.log_every_step = True
        if args.step_delay <= 0.0:
            args.step_delay = 0.25
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**31 - 1)
    rng = np.random.RandomState(seed)
    margin = ACTION_MARGIN

    env = CorridorPushTEnv(**build_env_kwargs(args))
    ws = float(env.window_size)
    success_thr = float(getattr(env, "success_threshold", 0.6))

    success_radius = getattr(env, "_success_radius_px", None)
    goal_mode = f"radius={success_radius:.1f}px" if success_radius is not None else f"coverage>={success_thr}"
    print(
        f"CorridorPushT viz  seed={seed}  goal_mode={goal_mode}  "
        f"random_goal={not args.no_random_goal}  random_corridor_w={not args.no_random_corridor_width}  "
        f"jitter={args.jitter}  log_steps={args.log_every_step}  step_delay={args.step_delay}"
    )

    out_dir = Path(args.out)
    if args.mode == "save":
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in range(args.episodes):
            env.seed(int(rng.randint(0, 2**31 - 1)))
            obs, _info = env.reset()
            goal = env.goal_pose
            hw = env.corridor_half_width
            dist0 = env.block_goal_distance()
            print(
                f"  ep{ep} reset: half_w={hw:.1f}  "
                f"goal=({goal[0]:.1f},{goal[1]:.1f})  "
                f"dist@reset={dist0:.1f}px  radius={success_radius or '(coverage)'}"
            )

            frames = []
            if args.mode == "save":
                frames.append(obs["visual"].copy())
            if args.mode == "human":
                show_frame(env, obs["visual"], args.fps)

            n_hold_steps = 0
            for t in range(args.horizon):
                dist_pre = env.block_goal_distance()
                reached_pre = env.is_goal_reached()
                a = expert_action(
                    env, rng, margin, ws, args.jitter, args.expert_hold_coverage
                )
                is_hold = action_is_hold(a)
                if is_hold:
                    n_hold_steps += 1

                obs, rew, _term, _trunc, info = env.step(a)
                dist_post = float(info.get("block_goal_distance", env.block_goal_distance()))
                reached_post = bool(info.get("goal_reached", False))
                corr_step = bool(info.get("corridor_contact", False))
                corr_ep = bool(info.get("corridor_contact_episode", False))

                if args.log_every_step:
                    act_tag = "HOLD" if is_hold else "push"
                    reach_tag = "REACHED" if reached_post else f"dist={dist_post:.1f}/{success_radius or success_thr}"
                    print(
                        f"  ep{ep} t{t:3d}  dist_pre={dist_pre:.1f}→{dist_post:.1f}  "
                        f"rew={float(rew):.3f}  act={act_tag}  {reach_tag}  "
                        f"wall={corr_step} (ep={corr_ep})",
                        flush=True,
                    )

                if args.step_delay > 0.0:
                    time.sleep(args.step_delay)

                if args.mode == "save":
                    frames.append(obs["visual"].copy())

                if args.mode == "human":
                    show_frame(env, obs["visual"], args.fps)

            cost = env.compute_episode_cost()
            jam = getattr(env, "_corridor_contact_episode", False)
            min_d = getattr(env, "_min_goal_distance_episode", float("inf"))
            success_str = (
                f"reached={min_d <= success_radius:.0f}  min_dist={min_d:.1f}/{success_radius:.1f}px"
                if success_radius is not None
                else f"max_cov={max(env.coverage_arr):.3f}" if env.coverage_arr else "no_steps"
            )
            print(
                f"episode {ep}: half_w={hw:.1f}  "
                f"goal=({goal[0]:.1f},{goal[1]:.1f})  "
                f"{success_str}  hold={n_hold_steps}/{args.horizon}  "
                f"cost={cost:.2f}  jam={jam}"
            )

            if args.mode == "save":
                path = out_dir / f"sanity_ep{ep:02d}.mp4"
                h, w = frames[0].shape[0], frames[0].shape[1]
                write_mp4(path, frames, args.fps, (h, w))
                print(f"  wrote {path}")
    finally:
        env.close()

    if args.mode == "human":
        import pygame
        print("Close the pygame window if it is still open.")
        pygame.quit()


if __name__ == "__main__":
    main()
