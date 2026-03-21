# Corridor Push-T: Walkthrough

This document explains the **Stochastic / Corridor Push-T** additions to the temporal-straightening codebase: environment, data layout, configuration, and how they connect to the master implementation plan (risk-aware evidential planning on a constrained 2D push task).

---

## 1. Goals (recap)

- **Corridor**: Narrow channel with parallel walls so that **wall contact** is a catastrophic failure (“jam”), unlike an open plane where a spinning block is only a soft failure.
- **Aleatoric variability**: Per-episode randomization of **center of mass** and **friction** so pushes are hard to predict from vision alone.
- **Action space**: A fixed **8-D** vector = four 2D control points for a **cubic Bézier** curve; a **PD controller** tracks the curve in simulation; **Ornstein–Uhlenbeck (OU) noise** perturbs executed velocity so longer / more aggressive plans accumulate execution uncertainty.
- **Labels**: Scalar **episode cost** for Evidential learning later: success (0), safe halt (0.5), jam (10).

Physics uses **PyMunk** (same as the original `PushTEnv`), not Box2D as in some written plans—the behavior is analogous.

---

## 2. Conda environment

Two environment options:

| File | Purpose |
|------|---------|
| [`environment-corridor-pusht.yaml`](../environment-corridor-pusht.yaml) | **Recommended** for Corridor Push-T: simulator, collection script, `train.py`. Small footprint. |
| [`environment.yaml`](../environment.yaml) | Full legacy stack (MuJoCo, D4RL, TensorFlow, etc.) from the original paper repo. |

### Create and activate (Corridor Push-T)

```bash
cd temporal-straightening
conda env create -f environment-corridor-pusht.yaml
conda activate corridor-pusht
```

### PyTorch and CUDA

The YAML installs CPU-capable `torch` / `torchvision` from pip. For GPU training, reinstall with your CUDA version, for example:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Alternatively, install [`requirements-corridor-pusht.txt`](../requirements-corridor-pusht.txt) after installing PyTorch manually.

### Verify imports

```bash
python -c "from env.pusht.corridor_pusht_env import CorridorPushTEnv; e=CorridorPushTEnv(); e.reset(); print('ok')"
```

---

## 3. New and touched files

### 3.1 Simulator

| File | Description |
|------|-------------|
| [`env/pusht/corridor_pusht_env.py`](../env/pusht/corridor_pusht_env.py) | **`CorridorPushTEnv`**: subclasses `PushTEnv`; adds vertical **corridor walls** (collision type separation), **random CoM + friction** each `reset()`, **8-D Bézier + PD + OU** in `step()`, **sensor circle** placeholder for future epistemic/OOD hooks, **`compute_episode_cost()`** for jam / success / safe-halt. **Goal pose** can be **sampled each reset** inside the corridor (Shapely check vs. inflated wall geometry + “ahead of block” reachability band). **`corridor_half_width`** can be **re-sampled every N episodes** (`corridor_width_episodes_per_setting`). |
| [`env/pusht/corridor_pusht_wrapper.py`](../env/pusht/corridor_pusht_wrapper.py) | **`CorridorPushTWrapper`**: `action_dim = 8`, corridor-aware init states, `prepare` / `step_multiple` / `rollout` (same pattern as `PushTWrapper`). |
| [`env/pusht/corridor_expert.py`](../env/pusht/corridor_expert.py) | **Heuristic expert**: builds 8-D Bézier control points **agent → block → goal** (`compute_expert_spline_8d`), **`hold_agent_spline_8d`** (all control points at the agent = minimal motion) for “done” behavior when coverage is high, plus jitter / large perturb / uniform random helpers for mixed collection. |
| [`env/pusht/__init__.py`](../env/pusht/__init__.py) | Exports `CorridorPushTEnv`. |
| [`env/__init__.py`](../env/__init__.py) | Registers `gym.make("corridor_pusht")` → `CorridorPushTWrapper`. |

### 3.2 Data pipeline

| File | Description |
|------|-------------|
| [`scripts/collect_corridor_pusht.py`](../scripts/collect_corridor_pusht.py) | Mixed collection (**default 70% expert / 30% exploration**): expert uses `corridor_expert` splines each step; when **`measure_block_goal_coverage()` ≥ `--expert-hold-coverage`** (default `0.95`), expert emits **hold** instead of re-planning. Exploration uses large perturbations of the expert spline or uniform random splines. Saves the same tensor/video layout plus **`episode_mix.pkl`** (`"expert"` / `"explore"` per episode). |
| [`datasets/corridor_pusht_dset.py`](../datasets/corridor_pusht_dset.py) | **`CorridorPushTDataset`**: loads the above; **8-D actions**; optional **`costs.pth`** (exposed in `get_frames` `meta` for future critic training). Default **action normalization** uses identity mean/std (replace with dataset statistics after large collection). |
| [`conf/env/corridor_pusht.yaml`](../conf/env/corridor_pusht.yaml) | Hydra config: `name: corridor_pusht`, dataset target `load_corridor_pusht_slice_train_val`, `DATASET_DIR`-based path. |

### 3.3 Root-level env specs

| File | Description |
|------|-------------|
| [`environment-corridor-pusht.yaml`](../environment-corridor-pusht.yaml) | Conda spec for this workflow. |
| [`requirements-corridor-pusht.txt`](../requirements-corridor-pusht.txt) | Pip mirror for manual or GPU-first installs. |

---

## 4. Behavior details (for implementation alignment)

### 4.1 Action space

- **Shape**: `(8,)`: control points `p0…p3 ∈ ℝ²` in pixel coordinates, clipped to the workspace (see `margin` in code).
- **Execution**: Each environment `step()` runs several physics substeps; along substeps, the PD target moves along a **cubic Bézier** from `p0` to `p3` with intermediates `p1`, `p2`.

### 4.2 Execution noise

- **OU process** (`OUNoise2D`): additive noise on **velocity** after the PD update each substep, parameterized by `theta` and `sigma` in `CorridorPushTEnv`.

### 4.3 Cost label (episode)

Computed in **`compute_episode_cost()`** after a rollout (and stored per timestep in **`costs.pth`** as a constant row for that episode):

| Condition | Cost |
|-----------|------|
| Any **corridor wall** contact during the episode | **10.0** (jam) |
| Else, **goal coverage** ≥ success threshold (same notion as original Push-T) | **0.0** |
| Else | **0.5** (safe halt / did not reach goal without jam) |

### 4.4 Differences from classic `PushTEnv`

- Original action: **2-D** target point (with optional relative scaling). Corridor: **8-D** spline.
- Original dataset often uses **`rel_actions.pth`**; corridor dataset uses **`actions.pth`** (absolute spline parameters).

### 4.5 Random goal, corridor width, and coverage

- **`randomize_goal_pose`** (default `True`): after the block and agent are placed, **`goal_pose`** is sampled in the corridor strip; poses whose **T-shaped goal mask** intersects **inflated wall geometry** (plus workspace bounds) are rejected. A simple **reachability** bias samples **goal y** mostly **above** the block on screen (smaller pixel y) so the target is usually “ahead” in the usual push direction.
- **`randomize_corridor_width`** (default `True`): every **`corridor_width_episodes_per_setting`** resets, **`corridor_half_width`** is drawn uniformly in **`[corridor_half_width_min, corridor_half_width_max]`**; walls are rebuilt in **`_setup()`**.
- **`measure_block_goal_coverage()`**: overlap area between goal and block masks divided by goal area—the same ratio used for **`step()`** reward scaling. Collection/visualization can treat **`coverage ≥ threshold`** as “task satisfied” for the heuristic expert and emit a **hold** spline (see `--expert-hold-coverage`).

---

## 5. Data collection

Set **`DATASET_DIR`** to the parent directory where you want `corridor_pusht/train` and `corridor_pusht/val` (matches `conf/env/corridor_pusht.yaml`).

### 5.1 Expert + exploration mix

Pure random 8-D splines rarely reach the goal in a narrow corridor. The collector uses:

1. **Expert episodes (default 70%)**: each timestep, if **`measure_block_goal_coverage()` ≥ `--expert-hold-coverage`** (default `0.95`, or set **&lt; 0** to disable), emit **`hold_agent_spline_8d`** so the spline degenerates at the agent (minimal motion). Otherwise build a **cubic Bézier** with **P0 = current agent position**, **P3 = goal center** (`goal_pose[:2]`), and handles **P1, P2** along **agent→block** and **block→goal** (see [`corridor_expert.py`](../env/pusht/corridor_expert.py)). Small **Gaussian jitter** on control points diversifies successful demos without destroying the path.
2. **Exploration episodes (default 30%)**: each timestep, with probability **`--explore-perturb-prob`**, apply **large Gaussian noise** to the *same* expert spline (glancing hits, jams); otherwise sample a **fully random** spline in the workspace.

Actions are still applied with **`env.step(action)`**—only the **construction** of the 8-D vector changed. Physics (PD + OU noise) is unchanged.

### 5.2 Command

```bash
export DATASET_DIR=/path/to/your/data   # e.g. .../data

python scripts/collect_corridor_pusht.py \
  --out "${DATASET_DIR}/corridor_pusht" \
  --n-episodes 10000 \
  --horizon 50 \
  --seed 0
```

**Notable flags**

| Flag | Default | Meaning |
|------|---------|--------|
| `--expert-ratio` | `0.7` | Fraction of episodes labeled expert vs exploration. |
| `--expert-jitter-sigma` | `6.0` | Pixel std for expert-only jitter (set `0` to disable). |
| `--explore-perturb-prob` | `0.55` | In exploration episodes, prob. of **perturbed expert** vs **uniform random** spline. |
| `--explore-noise-sigma` | `110.0` | Pixel std when perturbing the expert spline in exploration. |
| `--expert-hold-coverage` | `0.95` | Expert only: hold spline when coverage ≥ this; **&lt; 0** disables. |
| `--out` | `data/corridor_pusht` | Base folder; creates **`train/`** and **`val/`** (`--train-ratio`, default 0.9). |
| `--horizon` | `50` | Steps per episode (fixed; matches `seq_lengths`). |
| `--render-size` | `224` | Should match training `img_size`. |

**Outputs** include **`episode_mix.pkl`**: list of `"expert"` / `"explore"` per episode index (for analysis; the PyTorch dataset does not require it).

### 5.3 Quick visualization (sanity check)

Use [`scripts/visualize_corridor_pusht.py`](../scripts/visualize_corridor_pusht.py):

```bash
# Headless: writes MP4s under ./viz_corridor_pusht/ (good for SSH / no display)
python scripts/visualize_corridor_pusht.py --mode save --episodes 3 --horizon 80 --out ./viz_corridor_pusht

# Live pygame window (local machine with a display)
python scripts/visualize_corridor_pusht.py --mode human --episodes 3 --horizon 80
```

Each run prints **episode cost** (0 / 0.5 / 10), whether a **corridor jam** occurred, and **max goal coverage**. The policy is the same **heuristic expert** as collection (with `--jitter`).

---

## 6. Training (Phase 1 world model)

With data in place:

```bash
export DATASET_DIR=/path/to/your/data
python train.py env=corridor_pusht
```

Hydra picks [`conf/env/corridor_pusht.yaml`](../conf/env/corridor_pusht.yaml). The dataloader builds **8-D** action tensors; the **action encoder** `in_chans` follows `dataset.action_dim` (see `train.py`).

**Normalization**: Update `ACTION_MEAN` / `ACTION_STD` in [`datasets/corridor_pusht_dset.py`](../datasets/corridor_pusht_dset.py) after collecting data (e.g. from `get_all_actions()`), similar to the fixed stats in `pusht_dset.py`.

---

## 7. Gymnasium registration

The project uses **[Gymnasium](https://gymnasium.farama.org/)** (not the unmaintained `gym` package), which supports NumPy 2.x.

After `import env` (or any import that loads [`env/__init__.py`](../env/__init__.py)):

```python
import gymnasium as gym
import env  # registers environments
env = gym.make("corridor_pusht")
```

Point-maze / MuJoCo tasks require the `mujoco` Python package (`pip install mujoco` or `pip install "gymnasium[mujoco]"`). [`env/pointmaze/maze_specs.py`](../env/pointmaze/maze_specs.py) allows `import env` without loading MuJoCo until you use point-maze code.

---

## 8. Roadmap (not yet in repo)

Per the master plan, later phases may add:

- **`models/evidential_critic.py`** and training on frozen world model + `costs`.
- **`planning/evidential_planner.py`** with gradients w.r.t. observations and spline actions.
- Optional **zarr** export (current pipeline uses **torch + MP4** like existing Push-T).

---

## 9. Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `Space` has no attribute `add_collision_handler` | **Pymunk 7+** removed that API; the repo uses `register_collision_post_solve()` in [`pusht_env.py`](../env/pusht/pusht_env.py) (works on Pymunk 6 and 7). |
| `ModuleNotFoundError: gymnasium` | Install with `pip install gymnasium` or recreate the conda env from [`environment-corridor-pusht.yaml`](../environment-corridor-pusht.yaml). |
| `decord` / video read errors | Ensure `ffmpeg` is available (installed by conda in this env). |
| CUDA OOM | Reduce `batch_size` in `conf/train.yaml` or use a smaller encoder. |
| Poor world model | Collect more jam-heavy episodes; tune OU / corridor width so failures are not too rare. |

---

## 10. Related README

The main project [`README.md`](../README.md) covers the original temporal-straightening paper setup; this walkthrough is the **add-on** for Corridor Push-T only.
