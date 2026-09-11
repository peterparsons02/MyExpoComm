"""Synthetic-tensor learning checks; no simulator or experiment is launched."""

import importlib.util
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def learner_types():
    # Avoid loading unrelated learners' optional dependencies in these checks.
    package = types.ModuleType("learners")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/learners")]
    with patch.dict(sys.modules, {"learners": package}):
        from learners.hybrid_comm_learner import HybridCommLearner
        from learners.q_learner import ContQLearner
        from learners.tiger_deer_learner import TigerDeerHybridCommLearner, TigerDeerContQLearner
    return (("hybridcomm", HybridCommLearner, TigerDeerHybridCommLearner),
            ("expocomm", ContQLearner, TigerDeerContQLearner))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for synthetic learning checks")
class TigerDeerLearningTests(unittest.TestCase):
    def test_launcher_keeps_existing_registry_entries(self):
        pairs = learner_types()
        learners = types.ModuleType("learners")
        learners.REGISTRY = {"hybrid_comm_learner": pairs[0][1], "cont_q_learner": pairs[1][1]}
        wrappers = types.ModuleType("learners.tiger_deer_learner")
        wrappers.TigerDeerHybridCommLearner = pairs[0][2]
        wrappers.TigerDeerContQLearner = pairs[1][2]
        envs = types.ModuleType("envs")
        envs.REGISTRY = {"MAgent_Battle": object(), "MAgent_AdvPursuit": object()}
        tiger_env = types.ModuleType("envs.tiger_deer")
        tiger_env.TigerDeerEnv = object()
        main = types.ModuleType("main")
        main.main = Mock()  # Exercise launcher wiring without invoking training.
        old_learners, old_envs = learners.REGISTRY.copy(), envs.REGISTRY.copy()
        modules = {"main": main, "envs": envs, "envs.tiger_deer": tiger_env,
                   "learners": learners, "learners.tiger_deer_learner": wrappers}
        launcher = str(Path(__file__).resolve().parents[1] / "src/tiger_deer_experiment.py")
        for algorithm, original, guarded in pairs:
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", [launcher, "--algorithm", algorithm]):
                runpy.run_path(launcher, run_name="__main__")
            config = main.main.call_args[0][0]
            self.assertIs(learners.REGISTRY[config["learner"]], guarded)
            for name, value in old_learners.items():
                self.assertIs(learners.REGISTRY[name], value)
            for name, value in old_envs.items():
                self.assertIs(envs.REGISTRY[name], value)

    def test_guard_boundary_and_exact_original_loss_on_long_batches(self):
        import torch as th
        for algorithm, original, guarded in learner_types():
            learner = object.__new__(guarded)
            learner.cont_t_interval, learner.neg_num, learner.temperature = 5, 3, 0.07
            self.assertIs(guarded.train, original.train)
            for steps in (1, 11, 12):
                with self.subTest(algorithm=algorithm, steps=steps):
                    features = th.nn.functional.normalize(th.randn(2, steps, 3, 8), dim=-1)
                    features.requires_grad_()
                    mask = th.ones(2, steps, 1)
                    mask[1, 6:] = 0  # One shorter padded episode; one full episode.
                    th.manual_seed(42)
                    rng_before = th.get_rng_state()
                    if steps < 12:
                        with patch.object(original, "info_nce_loss", side_effect=AssertionError("sampler called")):
                            loss = learner.info_nce_loss(features, mask)
                        self.assertEqual(loss.item(), 0)
                        self.assertEqual(loss.dtype, features.dtype)
                        self.assertTrue(th.equal(rng_before, th.get_rng_state()))
                    else:
                        expected = original.info_nce_loss(learner, features, mask)
                        expected_rng = th.get_rng_state()
                        expected_grad, = th.autograd.grad(expected, features)
                        th.manual_seed(42)
                        actual = learner.info_nce_loss(features, mask)
                        actual_grad, = th.autograd.grad(actual, features)
                        th.testing.assert_close(actual, expected, rtol=0, atol=0)
                        th.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                        self.assertTrue(th.equal(expected_rng, th.get_rng_state()))

    def test_guarded_learners_keep_td_updates_with_short_batches_and_deaths(self):
        import torch as th
        from components.episode_buffer import EpisodeBatch
        from components.transforms import OneHot
        from controllers.ExpoComm_controller import ExpoCommMAC
        from tiger_deer_experiment import parse_args, build_config

        class Logger:
            console_logger = SimpleNamespace(info=lambda *args: None)
            def __init__(self):
                self.stats = {}
            def log_stat(self, name, value, step):
                assert th.isfinite(th.tensor(value)), (name, value)
                self.stats[name] = value

        th.set_num_threads(1)
        for algorithm, original, learner_type in learner_types():
            th.manual_seed(42)
            config = build_config(parse_args(["--algorithm", algorithm, "--cpu"]))
            config.update(n_agents=3, n_actions=9, state_shape=405, device="cpu",
                          topk_neighbors=3, neg_num=3)
            args = SimpleNamespace(**config)
            scheme = {
                "obs": {"vshape": 405, "group": "agents"},
                "state": {"vshape": 405},
                "avail_actions": {"vshape": (9,), "group": "agents", "dtype": th.int},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "reward": {"vshape": (1,), "group": "agents"},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            batch = EpisodeBatch(scheme, {"agents": 3}, 2, 17,
                                 preprocess={"actions": ("actions_onehot", [OneHot(9)])})
            batch["obs"][:] = th.randn_like(batch["obs"])
            batch["state"][:] = th.rand_like(batch["state"])
            batch["avail_actions"][:] = 1
            batch["actions"][:] = 2
            batch["actions_onehot"][:, :, :, 2] = 1
            batch["reward"][:] = th.rand(2, 17, 1, 1).expand(-1, -1, 3, -1)
            batch["filled"][:] = 1
            batch["avail_actions"][:, 5:, 0] = 0
            batch["avail_actions"][:, 5:, 0, 2] = 1
            batch["obs"][:, 5:, 0] = 0
            batch["terminated"][1, 10] = 1
            batch["filled"][1, 12:] = 0
            batch["avail_actions"][1, 12:] = 0
            for steps in (1, 5, 16):
                with self.subTest(algorithm=algorithm, steps=steps):
                    mac = ExpoCommMAC(batch.scheme, {"agents": 3}, args)
                    logger = Logger()
                    learner = learner_type(mac, batch.scheme, logger, args)
                    before = [p.detach().clone() for p in learner.params]
                    learner.train(batch[:, :steps + 1], t_env=0, episode_num=0)
                    self.assertEqual(learner.training_steps, 1)
                    self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
                    self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, learner.params)))
                    self.assertGreater(logger.stats["q_loss"], 0)
                    if steps <= 5:
                        self.assertEqual(logger.stats["info_nce_loss"], 0)
                        self.assertEqual(logger.stats["loss"], logger.stats["q_loss"])


class TigerDeerConfigTests(unittest.TestCase):
    def test_small_populations_fit_original_contrastive_sampler(self):
        from tiger_deer_experiment import parse_args, build_config
        for algorithm, learner in (("hybridcomm", "hybrid_comm_learner"),
                                   ("expocomm", "cont_q_learner")):
            config = build_config(parse_args(["--algorithm", algorithm, "--map-size", "15"]))
            self.assertEqual(config["learner"], "tiger_deer_" + learner)
            self.assertEqual(config["mac"], "ExpoComm_mac")
            self.assertEqual(config["neg_num"], 2)
            self.assertNotIn("mask_dead_agents", config)
            self.assertNotIn("shared_reward_mean", config)

    def test_short_training_horizons_use_guarded_learner(self):
        from tiger_deer_experiment import parse_args, build_config
        options = parse_args(["--algorithm", "expocomm", "--max-cycles", "1"])
        config = build_config(options)
        self.assertEqual(config["env_args"]["max_cycles"], 1)
        self.assertEqual(config["learner"], "tiger_deer_cont_q_learner")


if __name__ == "__main__":
    unittest.main()
