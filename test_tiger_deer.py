"""Cluster-side smoke test of the actual Tiger-Deer training wrapper."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-size", type=int, default=45)
    parser.add_argument("--deer-policy", choices=("random", "flee"), default="flee")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Import the wrapper without loading unrelated IMP/Battle/Pursuit dependencies.
    import importlib.util
    import types
    package = types.ModuleType("tiger_smoke_envs")
    package.__path__ = [str(Path(__file__).resolve().parent / "src/envs")]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("tiger_smoke_envs.tiger_deer", Path(package.__path__[0]) / "tiger_deer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.TigerDeerEnv(map_size=args.map_size, deer_policy=args.deer_policy,
                             max_cycles=100, seed=args.seed)
    rng = np.random.RandomState(args.seed)
    try:
        from importlib.metadata import version
        print("Versions:", {p: version(p) for p in ("magent", "pettingzoo", "numpy")})
        print("Initial tigers:", env.n_agents, "deer:", len(env.deer))
        print("Environment info:", env.get_env_info())
        for episode in range(2):
            env.reset()
            done = False
            steps = 0
            while not done:
                available = env.get_avail_actions()
                actions = [int(rng.choice(np.flatnonzero(row))) for row in available]
                reward, done, info = env.step(actions)
                steps += 1
                assert len(reward) == env.n_agents
                assert env.get_obs().shape == (env.n_agents, 405)
                assert np.isfinite(env.get_state()).all()
                assert np.isfinite(reward).all()
                assert np.all(env.get_obs()[~env.alive] == 0)
            print("Episode", episode, "steps:", steps, "stats:", info)
        print("Wrapper smoke test completed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
