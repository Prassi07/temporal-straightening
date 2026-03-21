"""
Corridor Push-T: narrow channel + aleatoric dynamics (CoM, friction), spline actions,
PID tracking with OU execution noise, and per-step state labels.
Physics uses PyMunk (same stack as PushTEnv).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pymunk
import pygame
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from env.pusht.pusht_env import PushTEnv, pymunk_to_shapely, register_collision_post_solve


def _cubic_bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    u = 1.0 - t
    return (
        (u**3) * p0
        + 3 * (u**2) * t * p1
        + 3 * u * (t**2) * p2
        + (t**3) * p3
    )


class OUNoise2D:
    """Discrete-time OU process for 2D velocity perturbation."""

    def __init__(
        self,
        theta: float = 4.0,
        sigma: float = 8.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.theta = theta
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()
        self.state = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, dt: float) -> np.ndarray:
        noise = self.rng.standard_normal(2)
        self.state += self.theta * (-self.state) * dt + self.sigma * np.sqrt(
            max(dt, 1e-9)
        ) * noise
        return self.state.copy()


class CorridorPushTEnv(PushTEnv):
    """
    Corridor Push-T environment.

    - Parallel corridor walls; wall contact is a hard failure signal.
    - Randomized T-block CoM and friction each episode (aleatoric variability).
    - Action: 8-D cubic Bézier control points; PD controller tracks curve; OU noise on velocity.
    - Per-step state labels stored in info: goal_reached, corridor_contact, block_goal_distance.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
    }

    COLLISION_BLOCK = 0
    COLLISION_CORRIDOR = 1

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=False,
        render_size=224,
        reset_to_state=None,
        relative=False,
        action_scale=1.0,
        with_velocity=False,
        with_target=True,
        shape="T",
        color="LightSlateGray",
        # Corridor geometry (pixel coords, y down)
        corridor_center_x: float = 256.0,
        corridor_half_width: float = 100.0,
        wall_thickness: float = 4.0,
        randomize_corridor_width: bool = True,
        corridor_half_width_min: float = 85.0,
        corridor_half_width_max: float = 115.0,
        corridor_width_episodes_per_setting: int = 3,
        # Goal pose (sampled inside corridor; walls + loose reachability)
        randomize_goal_pose: bool = True,
        goal_sample_max_tries: int = 48,
        goal_wall_clearance: float = 6.0,
        goal_reachability_gap_px: float = 18.0,
        # Aleatoric ranges
        com_shift_range: float = 28.0,
        friction_low: float = 0.15,
        friction_high: float = 1.05,
        # Agent velocity cap (px/s, applied each substep)
        agent_max_speed: float = 50.0,
        # OU noise on top of PID velocity
        ou_theta: float = 4.0,
        ou_sigma: float = 10.0,
        # Goal success criterion
        success_threshold: float = 0.6,
        success_radius_px: Optional[float] = 40.0,
        draw_goal_radius: bool = False,
    ):
        self.corridor_center_x = corridor_center_x
        self.corridor_half_width = float(corridor_half_width)
        self.wall_thickness = wall_thickness
        self.randomize_corridor_width = randomize_corridor_width
        self.corridor_half_width_min = float(corridor_half_width_min)
        self.corridor_half_width_max = float(corridor_half_width_max)
        self.corridor_width_episodes_per_setting = max(1, int(corridor_width_episodes_per_setting))
        self.randomize_goal_pose = randomize_goal_pose
        self.goal_sample_max_tries = int(goal_sample_max_tries)
        self.goal_wall_clearance = float(goal_wall_clearance)
        self.goal_reachability_gap_px = float(goal_reachability_gap_px)
        self._corridor_episode_index = 0
        self.com_shift_range = com_shift_range
        self.friction_low = friction_low
        self.friction_high = friction_high
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.agent_max_speed = float(agent_max_speed)
        self._success_radius_px = float(success_radius_px) if success_radius_px is not None else None
        self.draw_goal_radius = bool(draw_goal_radius)

        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_action=render_action,
            render_size=render_size,
            reset_to_state=reset_to_state,
            relative=relative,
            action_scale=action_scale,
            with_velocity=with_velocity,
            with_target=with_target,
            shape=shape,
            color=color,
        )

        ws = self.window_size
        self.action_space = spaces.Box(
            low=np.full(8, 0.0, dtype=np.float64),
            high=np.full(8, float(ws), dtype=np.float64),
            shape=(8,),
            dtype=np.float64,
        )
        self.ou_noise = OUNoise2D(
            theta=self.ou_theta, sigma=self.ou_sigma, rng=self.np_random
        )
        self._corridor_contact_episode = False
        self._corridor_contact_step = False
        self._base_block_com: Optional[np.ndarray] = None
        self.success_threshold = float(success_threshold)

    CORRIDOR_Y0 = 20.0
    CORRIDOR_Y1 = 492.0

    def _corridor_wall_union_geom(self) -> sg.base.BaseGeometry:
        """Axis-aligned strips for the two vertical corridor walls."""
        cx = self.corridor_center_x
        hw = self.corridor_half_width
        t = self.wall_thickness
        y0, y1 = self.CORRIDOR_Y0, self.CORRIDOR_Y1
        xl = cx - hw
        xr = cx + hw
        left = sg.box(xl - t, y0, xl + t, y1)
        right = sg.box(xr - t, y0, xr + t, y1)
        return left.union(right)

    def measure_block_goal_coverage(self) -> float:
        """Overlap ratio between goal T-mask and current block (same as step reward numerator / goal area)."""
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return float(intersection_area / goal_area) if goal_area > 0 else 0.0

    def block_goal_distance(self) -> float:
        """Euclidean distance from block center to goal center (goal_pose[:2])."""
        bx = float(self.block.position.x)
        by = float(self.block.position.y)
        gx, gy = float(self.goal_pose[0]), float(self.goal_pose[1])
        return float(np.hypot(bx - gx, by - gy))

    def is_goal_reached(self) -> bool:
        """True if goal is reached: position within success_radius_px, or overlap >= success_threshold."""
        if self._success_radius_px is not None:
            return self.block_goal_distance() <= self._success_radius_px
        return self.measure_block_goal_coverage() >= self.success_threshold

    def _is_goal_pose_valid(
        self,
        pose: np.ndarray,
        walls: sg.base.BaseGeometry,
        xl: float,
        xr: float,
        y0: float,
        y1: float,
    ) -> bool:
        goal_body = self._get_goal_pose_body(pose)
        try:
            goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        except Exception:
            return False
        if goal_geom.is_empty or goal_geom.area < 1e-6:
            return False
        b = goal_geom.bounds
        if b[0] < 6 or b[2] > 506 or b[1] < 6 or b[3] > 506:
            return False
        inflated = walls.buffer(self.goal_wall_clearance)
        if inflated.intersects(goal_geom):
            inter = inflated.intersection(goal_geom)
            if inter.area > 2.5:
                return False
        cx = 0.5 * (b[0] + b[2])
        if cx < xl + 12.0 or cx > xr - 12.0:
            return False
        return True

    def _sample_valid_goal_pose(self) -> np.ndarray:
        cx = self.corridor_center_x
        hw = self.corridor_half_width
        y0, y1 = self.CORRIDOR_Y0, self.CORRIDOR_Y1
        xl, xr = cx - hw, cx + hw
        margin = 22.0
        walls = self._corridor_wall_union_geom()
        block_y = float(self.block.position.y)
        y_hi = min(block_y - self.goal_reachability_gap_px, y1 - margin)
        y_lo = y0 + margin
        if y_hi < y_lo + 24.0:
            y_hi = min(block_y - 8.0, y1 - margin)
        if y_hi < y_lo + 12.0:
            y_lo, y_hi = y0 + margin, y1 - margin

        for _ in range(self.goal_sample_max_tries):
            x = float(self.np_random.uniform(xl + margin, xr - margin))
            y = float(self.np_random.uniform(y_lo, y_hi))
            theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            pose = np.array([x, y, theta], dtype=np.float64)
            if self._is_goal_pose_valid(pose, walls, xl, xr, y0, y1):
                return pose
        for y_try in (max(y_lo, 140.0), y_lo + 40.0, 0.5 * (y_lo + y_hi)):
            for th in (np.pi / 4, 0.0, np.pi / 2):
                pose = np.array([cx, float(y_try), float(th)], dtype=np.float64)
                if self._is_goal_pose_valid(pose, walls, xl, xr, y0, y1):
                    return pose
        return np.array([cx, float(0.5 * (y_lo + y_hi)), 0.0], dtype=np.float64)

    def _add_corridor_walls(self) -> None:
        cx = self.corridor_center_x
        hw = self.corridor_half_width
        t = self.wall_thickness
        y0, y1 = self.CORRIDOR_Y0, self.CORRIDOR_Y1
        xl = cx - hw
        xr = cx + hw
        # Vertical segments: left and right corridor walls
        for seg in (
            ((xl, y0), (xl, y1)),
            ((xr, y0), (xr, y1)),
        ):
            a, b = seg
            s = pymunk.Segment(self.space.static_body, a, b, t)
            s.friction = 0.8
            s.collision_type = self.COLLISION_CORRIDOR
            s.color = pygame.Color(180, 180, 200)
            self.space.add(s)

    def _add_epistemic_sensor_circle(self) -> None:
        """Sensor-only circle for future epistemic / OOD hooks (no dynamics)."""
        pos = (
            float(self.np_random.uniform(60.0, 452.0)),
            float(self.np_random.uniform(60.0, 452.0)),
        )
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        circ = pymunk.Circle(body, 6.0)
        circ.sensor = True
        circ.collision_type = 9
        circ.color = pygame.Color(255, 255, 255)
        self.space.add(body, circ)

    def _render_frame(self, mode):
        # Always suppress the parent's T-shape goal marker — we use a position-based
        # criterion and draw our own indicator (circle) only when draw_goal_radius=True.
        _saved_goal_color = self.goal_color
        self.goal_color = pygame.Color("White")

        img = super()._render_frame(mode)

        self.goal_color = _saved_goal_color

        if self.draw_goal_radius and self._success_radius_px is not None:
            ws = self.window_size
            rs = self.render_size
            scale = rs / ws

            # --- goal circle ---
            gx = int(round(float(self.goal_pose[0]) * scale))
            gy = int(round(float(self.goal_pose[1]) * scale))
            r = max(1, int(round(self._success_radius_px * scale)))
            reached = self.is_goal_reached()
            circle_color = (0, 200, 0) if reached else (0, 160, 255)
            cv2.circle(img, (gx, gy), r, circle_color, 2)
            cv2.circle(img, (gx, gy), max(2, r // 6), circle_color, -1)

            # --- Bezier curve overlay (skip for degenerate hold action) ---
            if (
                self.latest_action is not None
                and np.max(np.abs(self.latest_action - self.latest_action[0:1])) >= 0.5
            ):
                pts = self.latest_action  # shape (4, 2) in window coords
                p = (pts * scale).astype(np.float32)
                # Sample curve
                N = 40
                curve = np.array(
                    [
                        _cubic_bezier(p[0], p[1], p[2], p[3], i / (N - 1))
                        for i in range(N)
                    ],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(img, [curve], False, (230, 80, 0), 2, cv2.LINE_AA)
                # Control point dots and handle lines
                for i, (px_, py_) in enumerate(p.astype(int)):
                    dot_color = (230, 80, 0) if i in (0, 3) else (180, 60, 0)
                    cv2.circle(img, (int(px_), int(py_)), 4, dot_color, -1)
                # Handle lines P0→P1 and P3→P2
                cv2.line(
                    img,
                    tuple(p[0].astype(int)),
                    tuple(p[1].astype(int)),
                    (180, 60, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    img,
                    tuple(p[3].astype(int)),
                    tuple(p[2].astype(int)),
                    (180, 60, 0),
                    1,
                    cv2.LINE_AA,
                )

        return img

    def _on_corridor_collision(self, arbiter, space, data) -> None:
        self._corridor_contact_step = True
        self._corridor_contact_episode = True
        return None

    def _setup(self):
        super()._setup()
        self._corridor_contact_episode = False
        self._corridor_contact_step = False
        self._min_goal_distance_episode = float("inf")

        # Tag block shapes for collision routing
        for sh in self.block.shapes:
            sh.collision_type = self.COLLISION_BLOCK

        self._add_corridor_walls()
        self._add_epistemic_sensor_circle()

        register_collision_post_solve(
            self.space,
            self.COLLISION_BLOCK,
            self.COLLISION_CORRIDOR,
            self._on_corridor_collision,
        )

        # Default CoM reference (after shapes exist)
        self._base_block_com = np.array(self.block.center_of_gravity, dtype=np.float64)

    def _randomize_aleatoric(self) -> None:
        assert self._base_block_com is not None
        dx = float(self.np_random.uniform(-self.com_shift_range, self.com_shift_range))
        dy = float(self.np_random.uniform(-0.15 * self.com_shift_range, 0.15 * self.com_shift_range))
        self.block.center_of_gravity = (
            float(self._base_block_com[0] + dx),
            float(self._base_block_com[1] + dy),
        )
        mu = float(self.np_random.uniform(self.friction_low, self.friction_high))
        for sh in self.block.shapes:
            sh.friction = mu
        for sh in self.agent.shapes:
            sh.friction = min(1.0, mu + 0.1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.randomize_corridor_width:
            if self._corridor_episode_index % self.corridor_width_episodes_per_setting == 0:
                lo = min(self.corridor_half_width_min, self.corridor_half_width_max)
                hi = max(self.corridor_half_width_min, self.corridor_half_width_max)
                self.corridor_half_width = float(self.np_random.uniform(lo, hi))
        self._corridor_episode_index += 1
        self._setup()
        self._randomize_aleatoric()
        self.ou_noise = OUNoise2D(
            theta=self.ou_theta, sigma=self.ou_sigma, rng=self.np_random
        )
        if self.damping is not None:
            self.space.damping = self.damping

        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            cx = self.corridor_center_x
            # T-block half-width is 60 px; keep 12 px gap from inner wall face
            spread = max(0.0, self.corridor_half_width - self.wall_thickness - 60.0 - 12.0)
            if self.with_velocity:
                state = np.array(
                    [
                        cx + rs.uniform(-spread, spread),
                        rs.randint(380, 440),
                        cx + rs.uniform(-spread, spread),
                        rs.randint(260, 340),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,
                    ]
                )
            else:
                state = np.array(
                    [
                        cx + rs.uniform(-spread, spread),
                        rs.randint(380, 440),
                        cx + rs.uniform(-spread, spread),
                        rs.randint(260, 340),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        self._set_state(state)

        self.coverage_arr = []
        if self.randomize_goal_pose:
            self.goal_pose = self._sample_valid_goal_pose()

        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {"visual": visual, "proprio": proprio}
        return observation, {"state": state}

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        self._corridor_contact_step = False
        n_steps = self.sim_hz // self.control_hz

        margin = 18.0
        ws = float(self.window_size)

        if action is not None:
            ctrl = np.asarray(action, dtype=np.float64).reshape(4, 2)
            ctrl = np.clip(ctrl, margin, ws - margin)
            self.latest_action = ctrl
            # Degenerate spline: all four control points equal → "hold / stop" intent.
            # Zero agent velocity and skip PD+OU so the agent truly stays still.
            is_hold = np.max(np.abs(ctrl - ctrl[0:1])) < 0.5
            if is_hold:
                self.agent.velocity = Vec2d(0.0, 0.0)
                for _ in range(n_steps):
                    self.space.step(dt)
            else:
                p0, p1, p2, p3 = ctrl
                for i in range(n_steps):
                    t = (i + 1) / max(n_steps, 1)
                    target = _cubic_bezier(p0, p1, p2, p3, t)
                    act = Vec2d(float(target[0]), float(target[1]))
                    acceleration = self.k_p * (act - self.agent.position) + self.k_v * (
                        Vec2d(0, 0) - self.agent.velocity
                    )
                    self.agent.velocity += acceleration * dt
                    ou = self.ou_noise.step(dt)
                    self.agent.velocity += Vec2d(float(ou[0]), float(ou[1]))
                    # Hard speed cap to prevent overshooting
                    spd = self.agent.velocity.length
                    if spd > self.agent_max_speed:
                        self.agent.velocity = self.agent.velocity * (self.agent_max_speed / spd)
                    self.space.step(dt)

        dist = self.block_goal_distance()
        self._min_goal_distance_episode = min(
            getattr(self, "_min_goal_distance_episode", float("inf")), dist
        )
        if self._success_radius_px is not None:
            reward = 1.0 if dist <= self._success_radius_px else max(0.0, 1.0 - dist / self._success_radius_px)
        else:
            coverage = self.measure_block_goal_coverage()
            reward = float(np.clip(coverage / self.success_threshold, 0, 1))
            self.coverage_arr.append(coverage)
        done = False

        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {"visual": visual, "proprio": proprio}

        info = self._get_info()
        info["state"] = state
        info["block_goal_distance"] = dist
        info["goal_reached"] = self.is_goal_reached()
        info["corridor_contact"] = self._corridor_contact_step
        info["corridor_contact_episode"] = self._corridor_contact_episode
        speed = float(np.linalg.norm(state[5:7])) if self.with_velocity else float(
            np.linalg.norm(np.array(self.agent.velocity))
        )
        info["block_speed"] = speed
        terminated = bool(done)
        truncated = False
        return observation, reward, terminated, truncated, info

    def episode_goal_reached(self) -> bool:
        """True if the goal was reached at any step during this episode."""
        min_d = self._min_goal_distance_episode
        if self._success_radius_px is not None:
            return min_d <= self._success_radius_px
        cov = max(self.coverage_arr) if self.coverage_arr else 0.0
        return cov >= self.success_threshold
