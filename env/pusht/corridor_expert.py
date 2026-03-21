"""
Heuristic expert for Corridor Push-T data collection.

Builds an 8-D action = four 2D control points for the cubic Bézier used in
CorridorPushTEnv.step: P0→P3 with handles P1, P2.

Geometry: end-effector (agent) → T-block center → target zone center (goal_pose xy).
"""
from __future__ import annotations

import numpy as np


def hold_agent_spline_8d(
    agent_xy: np.ndarray,
    margin: float,
    workspace_size: float,
) -> np.ndarray:
    """
    Degenerate spline: all four control points at the agent → minimal motion (expert "stop").
    """
    a = np.asarray(agent_xy, dtype=np.float64).reshape(2)
    hi = float(workspace_size) - margin
    a = np.clip(a, margin, hi)
    ctrl = np.stack([a, a, a, a], axis=0)
    return ctrl.reshape(8).astype(np.float32)


def compute_expert_spline_8d(
    agent_xy: np.ndarray,
    block_xy: np.ndarray,
    goal_xy: np.ndarray,
    margin: float,
    workspace_size: float,
    p1_along_agent_block: float = 0.45,
    p2_along_block_goal: float = 0.55,
) -> np.ndarray:
    """
    Cubic Bézier control points in pixel coordinates, clipped to the workspace.

    - P0 = agent (spline starts at the end-effector).
    - P3 = goal center (target zone).
    - P1 lies on the segment agent→block; P2 on block→goal so the tangent
      of the curve tends to pass through the corridor toward the goal.

    Parameters
    ----------
    p1_along_agent_block : float in (0,1)
        Position of P1 between agent and block (by vector length).
    p2_along_block_goal : float in (0,1)
        Position of P2 between block and goal.
    """
    a = np.asarray(agent_xy, dtype=np.float64).reshape(2)
    b = np.asarray(block_xy, dtype=np.float64).reshape(2)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    p0 = a.copy()
    p3 = g.copy()
    p1 = a + (b - a) * float(p1_along_agent_block)
    p2 = b + (g - b) * float(p2_along_block_goal)

    ctrl = np.stack([p0, p1, p2, p3], axis=0)
    hi = float(workspace_size) - margin
    ctrl = np.clip(ctrl, margin, hi)
    return ctrl.reshape(8).astype(np.float32)


def jitter_expert_spline(
    spline_8d: np.ndarray,
    rng: np.random.RandomState,
    sigma: float,
    margin: float,
    workspace_size: float,
) -> np.ndarray:
    """Small Gaussian jitter on control points (diversifies expert demos)."""
    c = spline_8d.reshape(4, 2).astype(np.float64)
    c += rng.randn(4, 2) * float(sigma)
    hi = float(workspace_size) - margin
    c = np.clip(c, margin, hi)
    return c.reshape(8).astype(np.float32)


def perturb_spline_exploration(
    spline_8d: np.ndarray,
    rng: np.random.RandomState,
    sigma: float,
    margin: float,
    workspace_size: float,
) -> np.ndarray:
    """Large perturbations to break the expert path (failures, jams, glances)."""
    c = spline_8d.reshape(4, 2).astype(np.float64)
    c += rng.randn(4, 2) * float(sigma)
    hi = float(workspace_size) - margin
    c = np.clip(c, margin, hi)
    return c.reshape(8).astype(np.float32)


def sample_uniform_random_spline_8d(
    rng: np.random.RandomState,
    margin: float,
    workspace_size: float,
) -> np.ndarray:
    hi = float(workspace_size) - margin
    return rng.uniform(margin, hi, size=(8,)).astype(np.float32)
