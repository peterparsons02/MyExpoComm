"""Contract checks with a fake simulator. No rollouts, training, or downloads."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# Avoid importing envs/__init__.py and its unrelated simulator dependencies.
package = types.ModuleType("tiger_test_envs")
package.__path__ = [str(ROOT / "src/envs")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("tiger_test_envs.tiger_deer", ROOT / "src/envs/tiger_deer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeWarehouseEngine:
    def __init__(self, parent):
        self.parent = parent

    def get_agent_id(self, handle):
        prefix = "deer_" if handle == 0 else "tiger_"
        return np.array([i for i, a in enumerate(self.parent.possible_agents)
                         if a in self.parent.survivors and a.startswith(prefix)], dtype=int)

    def get_pos(self, handle):
        return np.array([[2 + i, 3 + i] for i in self.get_agent_id(handle)], dtype=int).reshape(-1, 2)

    def _get_walls_info(self):
        return [(self.parent.resets, 0)]


class FakeEnv:
    def __init__(self, max_cycles=300, **kwargs):
        self.possible_agents = ["deer_0", "tiger_0", "deer_1", "tiger_1"]
        self.observation_spaces = {
            a: types.SimpleNamespace(shape=(9, 9, 5) if a.startswith("tiger") else (3, 3, 5))
            for a in self.possible_agents
        }
        self.action_spaces = {a: types.SimpleNamespace(n=9 if a.startswith("tiger") else 5)
                              for a in self.possible_agents}
        self.handles = [0, 1]
        self.env = FakeWarehouseEngine(self)
        self.max_cycles = max_cycles
        self.resets = 0
        self.kill = True
        self.kill_all_tigers = False
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def observations(self, names):
        result = {}
        for a in names:
            obs = np.zeros(self.observation_spaces[a].shape, dtype=np.float32)
            c = obs.shape[0] // 2
            obs[c, c, 1:3] = 1
            result[a] = obs
        return result

    def reset(self):
        self.resets += 1
        self.frames = 0
        self.survivors = set(self.possible_agents)
        self.agents = self.possible_agents[:]
        return self.observations(self.agents)

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.last_actions = actions
        before = self.agents[:]
        self.frames += 1
        if self.kill:
            self.survivors.discard("tiger_0")
            self.survivors.discard("deer_1")
        if self.kill_all_tigers:
            self.survivors.difference_update(["tiger_0", "tiger_1"])
        dones = {a: a not in self.survivors or self.frames >= self.max_cycles for a in before}
        self.agents = [a for a in before if not dones[a]]
        return self.observations(before), {"tiger_1": 2.0}, dones, {}

    def close(self):
        self.closed = True


class TigerDeerContractTests(unittest.TestCase):
    def make_env(self, **kwargs):
        pettingzoo = types.ModuleType("pettingzoo")
        magent = types.ModuleType("pettingzoo.magent")
        magent.tiger_deer_v3 = types.SimpleNamespace(parallel_env=FakeEnv)
        with patch.dict(sys.modules, {"pettingzoo": pettingzoo, "pettingzoo.magent": magent}):
            env = module.TigerDeerEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_stable_slots_and_shared_reward_after_death(self):
        env = self.make_env()
        rewards, done, info = env.step([0, 3])
        self.assertEqual(env.env.last_actions["tiger_1"], 3)
        self.assertEqual(rewards, [1.0, 1.0])
        self.assertFalse(done)
        self.assertEqual(env.get_obs().shape, (2, 405))
        self.assertTrue(np.all(env.get_obs()[0] == 0))
        self.assertTrue(np.all(env.get_positions()[:2] == 0))
        self.assertEqual(env.get_avail_agent_actions(0).tolist(), [0, 0, 1, 0, 0, 0, 0, 0, 0])
        env.step([2, 4])
        self.assertNotIn("tiger_0", env.env.last_actions)
        self.assertEqual(env.env.last_actions["tiger_1"], 4)
        self.assertEqual(info["deer_killed"], 1)

    def test_timeout_keeps_survivor_observations_for_bootstrap(self):
        env = self.make_env(max_cycles=1)
        env.env.kill = False
        _, done, info = env.step([2, 2])
        self.assertTrue(done)
        self.assertTrue(info["episode_limit"])
        self.assertTrue(env.alive.all())
        self.assertTrue(env.get_obs().any())
        with self.assertRaises(RuntimeError):
            env.step([2, 2])

    def test_extinction_is_terminal_even_at_time_limit(self):
        env = self.make_env(max_cycles=1)
        env.env.kill_all_tigers = True
        _, done, info = env.step([2, 2])
        self.assertTrue(done)
        self.assertFalse(info["episode_limit"])
        self.assertFalse(env.get_obs().any())
        self.assertTrue(np.isfinite(env.get_state()).all())

    def test_reset_restores_slots_and_rebuilds_walls(self):
        env = self.make_env(state_grid_size=45)
        initial = env.get_state().reshape(45, 45, 5)
        env.step([2, 2])
        env.reset()
        self.assertTrue(env.alive.all())
        state = env.get_state().reshape(45, 45, 5)
        self.assertEqual(initial[1, 0, 0], 1)
        self.assertEqual(state[1, 0, 0], 0)
        self.assertEqual(state[2, 0, 0], 1)

    def test_invalid_actions_fail_before_simulator_step(self):
        env = self.make_env()
        for actions in ([2], [9, 2], [1.5, 2]):
            with self.assertRaises(ValueError):
                env.step(actions)

    def test_flee_uses_correct_action_direction_and_avoids_walls(self):
        obs = np.zeros((3, 3, 5), dtype=np.float32)
        obs[1, 0, 3] = 1  # Tiger to the left.
        self.assertEqual(module.deer_action(obs, np.random.RandomState(0), "flee"), 3)
        obs[1, 2, 0] = 1  # Right is now blocked.
        self.assertIn(module.deer_action(obs, np.random.RandomState(0), "flee"), (0, 4))

    def test_pooled_state_preserves_occupancy_mass(self):
        env = self.make_env(state_grid_size=9)
        state = env.get_state().reshape(9, 9, 5) * env._bin_area[:, :, None]
        np.testing.assert_allclose(state.sum(axis=(0, 1)), [1, 2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
