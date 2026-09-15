"""Tiger-controlled EPyMARL adapter for PettingZoo 1.15 / MAgent 0.1.14."""

import numpy as np

from .multiagentenv import MultiAgentEnv


# MAgent CircleRange(1) enumerates cells row-major, including the centre.
# Actions are up, left, stay, right, down; attacks start at index 5.
MOVE_DELTAS = np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])
STAY = 2


def deer_action(obs, rng, policy):
    """Fixed opponent; flee sees only the deer's native 3x3 observation."""
    if policy == "random":
        return int(rng.randint(5))
    targets = MOVE_DELTAS + 1
    blocked = (obs[:, :, 0] > 0) | (obs[:, :, 1] > 0) | (obs[:, :, 3] > 0)
    blocked[1, 1] = False
    legal = np.flatnonzero(~blocked[targets[:, 0], targets[:, 1]])
    threats = np.argwhere(obs[:, :, 3] > 0)
    if len(threats):
        distances = ((targets[legal, None, :] - threats[None, :, :]) ** 2).sum(-1)
        scores = distances.min(axis=1)
        legal = legal[scores == scores.max()]
    return int(rng.choice(legal))


class TigerDeerEnv(MultiAgentEnv):
    def __init__(self, map_size=45, max_cycles=300, seed=42,
                 deer_policy="flee", state_grid_size=9, global_reward=True,
                 minimap_mode=False, tiger_step_recover=-0.1, deer_attacked=-0.1):
        if map_size < 15 or max_cycles < 1:
            raise ValueError("Use map_size >= 15 (at least two tigers) and max_cycles >= 1")
        if deer_policy not in ("random", "flee"):
            raise ValueError("deer_policy must be 'random' or 'flee'")
        if not 1 <= state_grid_size <= map_size:
            raise ValueError("state_grid_size must be between 1 and map_size")
        if minimap_mode:
            raise ValueError("Tiger-Deer actors use local observations; minimap_mode must be false")
        if not global_reward:
            raise ValueError("These matched Tiger-Deer experiments require global_reward=true")

        # Same process-local compatibility shim as src/main.py.
        if "bool" not in np.__dict__:
            np.bool = bool
        from pettingzoo.magent import tiger_deer_v3

        self.env = tiger_deer_v3.parallel_env(
            map_size=map_size, max_cycles=max_cycles, minimap_mode=False,
            extra_features=False, tiger_step_recover=tiger_step_recover,
            deer_attacked=deer_attacked,
        )
        self.seed = seed
        self._reset_index = 0
        self.deer_policy = deer_policy
        self.episode_limit = max_cycles
        self.map_size = map_size
        self.state_grid_size = state_grid_size
        self.possible_agents = list(self.env.possible_agents)
        self.agents = [a for a in self.possible_agents if a.startswith("tiger_")]
        self.deer = [a for a in self.possible_agents if a.startswith("deer_")]
        self.n_agents = len(self.agents)
        if self.n_agents < 2:
            raise ValueError("At least two tigers are required for contrastive training")
        self._slots = {a: i for i, a in enumerate(self.agents)}
        self.obs_shape = self.env.observation_spaces[self.agents[0]].shape
        self.n_actions = self.env.action_spaces[self.agents[0]].n
        if self.obs_shape != (9, 9, 5) or self.n_actions != 9:
            raise ValueError("Expected unmodified tiger_deer_v3 with 9x9x5 observations and 9 actions")
        counts = np.bincount(np.arange(map_size) * state_grid_size // map_size,
                             minlength=state_grid_size)
        self._bin_area = counts[:, None] * counts[None, :]
        self.reset()

    def _add_to_state(self, state, positions, channel, weights=1.0):
        if len(positions):
            bins = np.asarray(positions, dtype=np.int64) * self.state_grid_size // self.map_size
            np.add.at(state[:, :, channel], (bins[:, 0], bins[:, 1]), weights)

    def _refresh(self, observations):
        self._raw_obs = observations
        self._obs = np.zeros((self.n_agents, self.get_obs_size()), dtype=np.float32)
        self._positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.alive = np.zeros(self.n_agents, dtype=bool)
        state = self._wall_state.copy()
        self.deer_alive = 0
        # Engine IDs remain stable after deaths, unlike compacted group arrays.
        # Use physical survivors even at the time limit, when env.agents is empty.
        for handle in self.env.handles:
            ids = self.env.env.get_agent_id(handle)
            positions = self.env.env.get_pos(handle)
            names = [self.possible_agents[int(i)] for i in ids]
            if not names:
                continue
            is_tiger = names[0].startswith("tiger_")
            channel = 3 if is_tiger else 1
            health = []
            for name, position in zip(names, positions):
                obs = observations[name]
                health.append(obs[obs.shape[0] // 2, obs.shape[1] // 2, 2])
                if is_tiger:
                    slot = self._slots[name]
                    self.alive[slot] = True
                    self._obs[slot] = obs.reshape(-1)
                    self._positions[slot] = position
            if not is_tiger:
                self.deer_alive += len(names)
            self._add_to_state(state, positions, channel)
            self._add_to_state(state, positions, channel + 1, health)
        self._state = (state / self._bin_area[:, :, None]).astype(np.float32).reshape(-1)

    def reset(self):
        episode_seed = (self.seed + self._reset_index) % (2 ** 32)
        self.env.seed(episode_seed)
        self.rng = np.random.RandomState(episode_seed)
        self._reset_index += 1
        observations = self.env.reset()
        # PettingZoo 1.15 caches walls in state() before reset and cannot handle
        # empty groups there. Rebuild the pooled state from current engine data.
        self._wall_state = np.zeros((self.state_grid_size, self.state_grid_size, 5), dtype=np.float32)
        self._add_to_state(self._wall_state, self.env.env._get_walls_info(), 0)
        self._episode_length = 0
        self._return = 0.0
        self._positive_tiger_rewards = 0
        self._terminated = False
        self._refresh(observations)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if self._terminated:
            raise RuntimeError("reset() is required after a completed episode")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        if len(actions) != self.n_agents:
            raise ValueError("Expected one action per initial tiger slot")
        if not np.all((actions >= 0) & (actions < self.n_actions) & (actions == np.floor(actions))):
            raise ValueError("Tiger actions must be integer indices in [0, 8]")
        action_dict = {}
        for name in self.env.agents:
            if name in self._slots:
                action_dict[name] = int(actions[self._slots[name]])
            else:
                action_dict[name] = deer_action(self._raw_obs[name], self.rng, self.deer_policy)
        obs, rewards, dones, infos = self.env.step(action_dict)
        self._episode_length += 1
        raw_reward = sum(rewards.get(a, 0.0) for a in self.agents)
        self._return += raw_reward
        self._positive_tiger_rewards += sum(rewards.get(a, 0.0) > 0 for a in self.agents)
        self._refresh(obs)
        natural_end = not self.alive.any() or self.deer_alive == 0
        timeout = self._episode_length >= self.episode_limit
        self._terminated = natural_end or timeout or not self.env.agents
        info = {
            "episode_limit": bool(timeout and not natural_end),
            "tigers_alive": int(self.alive.sum()),
            "deer_alive": self.deer_alive,
            "deer_killed": len(self.deer) - self.deer_alive,
            "tiger_survival_fraction": float(self.alive.mean()),
            "tiger_return_total": self._return,
            "tiger_return_per_initial_agent": self._return / self.n_agents,
            "positive_tiger_reward_events": self._positive_tiger_rewards,
        }
        # Fixed denominator: deaths must not increase reward scale. Both learners
        # receive the same shared scalar, repeated for the existing runner API.
        return [raw_reward / self.n_agents] * self.n_agents, self._terminated, info

    def get_obs(self):
        return self._obs.copy()

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        return int(np.prod(self.obs_shape))

    def get_state(self):
        return self._state.copy()

    def get_state_size(self):
        return self.state_grid_size ** 2 * 5

    def get_positions(self):
        return self._positions.reshape(-1).copy()

    def get_avail_actions(self):
        available = np.zeros((self.n_agents, self.n_actions), dtype=np.int32)
        available[self.alive] = 1
        available[:, STAY] = 1
        return available

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return {}

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def save_replay(self):
        raise NotImplementedError("Use render=True for interactive Tiger-Deer evaluation")
