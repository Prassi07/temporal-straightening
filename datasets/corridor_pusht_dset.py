import pickle
from pathlib import Path
from typing import Callable, Optional

import torch
from decord import VideoReader
from einops import rearrange

import decord

from .traj_dset import TrajDataset, TrajSlicerDataset

decord.bridge.set_bridge("torch")

# Defaults assume relative actions (offsets from agent pos at commit time) are
# already centered near zero. Override with dataset-computed stats for best results.
ACTION_MEAN = torch.zeros(8)
ACTION_STD = torch.ones(8)
STATE_MEAN = torch.zeros(7)
STATE_STD = torch.ones(7)
PROPRIO_MEAN = torch.zeros(4)
PROPRIO_STD = torch.ones(4)


class CorridorPushTDataset(TrajDataset):
    """
    Corridor Push-T dataset.

    Actions are robot-centric 8-D spline control points (``actions_relative.pth``):
    four 2-D offsets from the agent position at the time the chunk was committed.
    This representation is translation-invariant and better suited for learning.

    Per-step state labels loaded alongside observations:
      - goal_reached  (N, T, 1) float32 — 1.0 when block is within success_radius of goal
      - wall_contact  (N, T, 1) float32 — 1.0 when block touched a corridor wall
      - distances     (N, T, 1) float32 — block-to-goal Euclidean distance in pixels
      - goal_poses    (N, T, 3) float32 — [goal_x, goal_y, goal_theta] for the episode
    """

    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/corridor_pusht/train",
        normalize_action: bool = True,
        with_velocity: bool = True,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth", weights_only=False).float()
        # Robot-centric relative control points
        self.actions = torch.load(self.data_path / "actions_relative.pth", weights_only=False).float()

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                self.shapes = pickle.load(f)
        else:
            self.shapes = ["T"] * len(self.states)

        self.n_rollout = n_rollout
        n = n_rollout if n_rollout else len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        # Per-step state labels
        self.goal_reached = torch.load(self.data_path / "goal_reached.pth", weights_only=False).float()[:n]
        self.wall_contact = torch.load(self.data_path / "wall_contact.pth", weights_only=False).float()[:n]
        self.distances = torch.load(self.data_path / "distances.pth", weights_only=False).float()[:n]
        self.goal_poses = torch.load(self.data_path / "goal_poses.pth", weights_only=False).float()[:n]

        self.proprios = self.states[..., :2].clone()
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth", weights_only=False).float()[:n]
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[: self.state_dim]
            self.state_std = STATE_STD[: self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[: self.proprio_dim]
            self.proprio_std = PROPRIO_STD[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            t = self.seq_lengths[i]
            result.append(self.actions[i, :t, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        frames = list(frames)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]
        image = reader.get_batch(frames)
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        meta = {
            "shape": shape,
            "goal_reached": self.goal_reached[idx, frames],
            "wall_contact": self.wall_contact[idx, frames],
            "distances": self.distances[idx, frames],
            "goal_poses": self.goal_poses[idx, frames],
        }
        return obs, act, state, meta

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_corridor_pusht_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/corridor_pusht",
    normalize_action=True,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
):
    train_dset = CorridorPushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = CorridorPushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
