import numpy as np

from env.pusht.corridor_pusht_env import CorridorPushTEnv
from utils import aggregate_dct


class CorridorPushTWrapper(CorridorPushTEnv):
    """
    Rollout utilities for Corridor Push-T (8-D spline actions).
    """

    def __init__(self, with_velocity=True, with_target=True, **kwargs):
        super().__init__(with_velocity=with_velocity, with_target=with_target, **kwargs)
        self.action_dim = 8

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)

        def generate_state():
            cx = float(self.corridor_center_x)
            spread = max(8.0, float(self.corridor_half_width) - 35.0)
            if self.with_velocity:
                return np.array(
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
            return np.array(
                [
                    cx + rs.uniform(-spread, spread),
                    rs.randint(380, 440),
                    cx + rs.uniform(-spread, spread),
                    rs.randint(260, 340),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )

        return generate_state(), generate_state()

    def update_env(self, env_info):
        self.shape = env_info["shape"]

    def eval_state(self, goal_state, cur_state):
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {"success": success, "state_dist": state_dist}

    def prepare(self, seed, init_state):
        self.seed(seed)
        self.reset_to_state = init_state
        obs, info = self.reset()
        state = info["state"]
        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, term, trunc, info = self.step(action)
            d = term or trunc
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
