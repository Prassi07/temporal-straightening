# Corridor Push-T: Walkthrough

This document covers the **Corridor Push-T** additions to the temporal-straightening codebase: environment, data layout, and configuration.

---

## 1. Design goals

- **Corridor**: Narrow channel with parallel walls. Wall contact is a hard failure signal that can be used to derive penalties downstream.
- **Aleatoric variability**: Per-episode randomisation of **center of mass** and **friction** so pushes are hard to predict from vision alone.
- **Action space**: Fixed **8-D** vector = four 2D control points for a **cubic Bézier** curve; a **PD controller** tracks the curve in simulation; **Ornstein–Uhlenbeck (OU) noise** perturbs the executed velocity so longer plans accumulate execution uncertainty.
- **State labels**: Three raw per-step signals stored in the dataset — `goal_reached`, `wall_contact`, `block_goal_distance`. Reward / cost shaping is deferred to downstream code.

Physics uses **PyMunk** (same as the original `PushTEnv`).

---

## 2. Conda environment

| File | Purpose |
|------|---------|
| [`environment-corridor-pusht.yaml`](../environment-corridor-pusht.yaml) | Recommended for Corridor Push-T. |
| [`environment.yaml`](../environment.yaml) | Full legacy stack (MuJoCo, D4RL, TensorFlow, etc.) |

```bash
conda env create -f environment-corridor-pusht.yaml
conda activate corridor-pusht
```

For GPU training, reinstall PyTorch after activation:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify

```bash
python -c "from env.pusht.corridor_pusht_env import CorridorPushTEnv; e=CorridorPushTEnv(); e.reset(); print('ok')"
```

---

## 3. Files

### 3.1 Simulator

| File | Description |
|------|-------------|
| [`env/pusht/corridor_pusht_env.py`](../env/pusht/corridor_pusht_env.py) | **`CorridorPushTEnv`**: subclasses `PushTEnv`; adds vertical corridor walls, random CoM + friction each reset, 8-D Bézier + PD + OU in `step()`, random goal position inside the corridor, optional corridor-width schedule, and per-step state labels in `info`. |
| [`env/pusht/corridor_pusht_wrapper.py`](../env/pusht/corridor_pusht_wrapper.py) | **`CorridorPushTWrapper`**: `action_dim = 8`, corridor-aware init states, `prepare` / `step_multiple` / `rollout`. |
| [`env/pusht/corridor_expert.py`](../env/pusht/corridor_expert.py) | **Heuristic expert**: `compute_expert_spline_8d` (agent → block → goal Bézier), `hold_agent_spline_8d` (all CPs at agent = zero motion), jitter / perturb / random helpers. |
| [`env/pusht/__init__.py`](../env/pusht/__init__.py) | Exports `CorridorPushTEnv`. |
| [`env/__init__.py`](../env/__init__.py) | Registers `gym.make("corridor_pusht")` → `CorridorPushTWrapper`. |

### 3.2 Data pipeline

| File | Description |
|------|-------------|
| [`scripts/collect_corridor_pusht.py`](../scripts/collect_corridor_pusht.py) | Mixed collection (70 % expert / 30 % exploration). Saves per-step state labels alongside frames, states, actions, velocities. |
| [`datasets/corridor_pusht_dset.py`](../datasets/corridor_pusht_dset.py) | **`CorridorPushTDataset`**: loads tensors saved by the collect script. |
| [`conf/env/corridor_pusht.yaml`](../conf/env/corridor_pusht.yaml) | Hydra config: `name: corridor_pusht`, dataset path from `DATASET_DIR`. |

---

## 4. Environment details

### 4.1 Action space

- **Shape**: `(8,)` — control points `p0…p3 ∈ ℝ²` in pixel coordinates, clipped to `[margin, window_size − margin]`.
- **Execution**: Each `step()` runs 10 physics substeps (sim_hz 100, control_hz 10). The PD target moves along the cubic Bézier each substep; OU noise is added to velocity.
- **Speed cap**: `agent_max_speed = 300 px/s` (constructor arg) prevents overshoot.

### 4.2 Goal

- Default **position-based**: goal reached when the block center is within `success_radius_px` (default 40 px) of `goal_pose[:2]`.
- `goal_pose` is sampled each reset inside the corridor strip (Shapely collision check against inflated wall geometry, optional reachability bias ahead of block).
- Fallback coverage-based mode: set `success_radius_px=None`, uses `success_threshold` (default 0.6).

### 4.3 Corridor geometry

- `corridor_half_width` (default 100 px) is the half-width of the open channel; walls are at `cx ± hw`.
- `randomize_corridor_width=True`: re-samples width from `[corridor_half_width_min, corridor_half_width_max]` every `corridor_width_episodes_per_setting` resets.
- T-block half-width is 60 px; spawn spread is clamped so blocks cannot initialise inside walls.

### 4.4 Per-step state labels (`info` dict)

| Key | Type | Description |
|-----|------|-------------|
| `goal_reached` | bool | Block center within `success_radius_px` of goal. |
| `corridor_contact` | bool | Wall contact during **this** physics step. |
| `corridor_contact_episode` | bool | Wall contact at **any** point this episode (cumulative). |
| `block_goal_distance` | float | Euclidean distance from block center to goal center (px). |
| `block_speed` | float | Current agent speed (px/s). |
| `state` | ndarray | Full state vector (agent xy, block xy, block angle, [velocities]). |

Reward shaping, penalties, and curriculum decisions are left to downstream code operating on these signals.

### 4.5 Hold action

When the expert emits `hold_agent_spline_8d`, all four control points equal the agent position. `step()` detects this (max CP spread < 0.5 px), zeroes agent velocity, and skips PD + OU so the agent is truly stationary.

---

## 5. Data collection

### 5.1 Expert / exploration mix

| Mode | Behaviour |
|------|-----------|
| **Expert** (default 70 %) | Each step: if `is_goal_reached()` → `hold_agent_spline_8d`; otherwise cubic Bézier agent → block → goal with optional Gaussian jitter. |
| **Exploration** (default 30 %) | Each step: with prob. `explore_perturb_prob`, large-noise perturbation of expert spline; otherwise uniform random spline. |

### 5.2 Run

```bash
export DATASET_DIR=/path/to/data

python scripts/collect_corridor_pusht.py \
  --out "${DATASET_DIR}/corridor_pusht" \
  --n-episodes 10000 \
  --horizon 50 \
  --seed 0
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--expert-ratio` | `0.7` | Expert episode fraction. |
| `--expert-jitter-sigma` | `6.0` | Gaussian jitter std on expert CPs (px). |
| `--explore-perturb-prob` | `0.55` | Prob. of perturbed-expert vs fully-random in exploration episodes. |
| `--explore-noise-sigma` | `110.0` | Std for exploration perturbation (px). |
| `--success-radius-px` | `None` | Override env default (40 px) goal radius. |
| `--horizon` | `50` | Steps per episode. |
| `--render-size` | `224` | Image resolution for dataset frames. |

### 5.3 Saved files (per split)

```
train/
  states.pth          (N, T, state_dim)   float32
  actions.pth         (N, T, 8)           float32
  velocities.pth      (N, T, 2)           float32
  goal_reached.pth    (N, T, 1)           float32  — 1.0 when goal reached this step
  wall_contact.pth    (N, T, 1)           float32  — 1.0 when wall contact this step
  distances.pth       (N, T, 1)           float32  — block-to-goal distance (px)
  obses/episode_XXX.mp4
  seq_lengths.pkl
  episode_mix.pkl     list["expert"|"explore"]
```

### 5.4 Visualisation

```bash
# Headless MP4
python scripts/visualize_corridor_pusht.py --mode save --episodes 3 --out ./viz

# Live pygame window
python scripts/visualize_corridor_pusht.py --mode human --episodes 3

# Step-by-step distance log
python scripts/visualize_corridor_pusht.py --debug-steps --episodes 1
```

The pygame window shows:
- **Blue circle** = goal zone (turns **green** when block reaches goal).
- **Orange Bézier** = current action spline with handle lines and control point dots.

Per-episode summary:

```
episode 0: half_w=102.3  goal=(256.1,148.4)  reached=1  min_dist=28.4/40.0px  hold=18/80  jam=False
```

---

## 6. Training (Phase 1 world model)

```bash
export DATASET_DIR=/path/to/data
python train.py env=corridor_pusht
```

Hydra picks [`conf/env/corridor_pusht.yaml`](../conf/env/corridor_pusht.yaml). Update `ACTION_MEAN` / `ACTION_STD` in [`datasets/corridor_pusht_dset.py`](../datasets/corridor_pusht_dset.py) after a large collection run.

---

## 7. Gymnasium registration

```python
import gymnasium as gym
import env  # registers environments
env = gym.make("corridor_pusht")
```

The project uses **Gymnasium** (not the unmaintained `gym` package) for NumPy 2.x compatibility.

---

## 8. Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `Space` has no attribute `add_collision_handler` | Pymunk 7+ removed that API; the repo uses `register_collision_post_solve()` in [`pusht_env.py`](../env/pusht/pusht_env.py). |
| `ModuleNotFoundError: gymnasium` | `pip install gymnasium` or recreate the conda env. |
| `decord` / video read errors | Ensure `ffmpeg` is available (`conda install ffmpeg`). |
| Block spawns inside walls | Increase `corridor_half_width_min` or reduce `com_shift_range`. |
| Agent overshoots goal | Lower `agent_max_speed` (default 300 px/s) or reduce OU sigma. |
