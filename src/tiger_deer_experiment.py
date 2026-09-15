"""Launch Tiger-Deer with the original controller and guarded learner subclasses."""

import argparse
import math
from pathlib import Path

import yaml


def merge_config(base, update):
    for key, value in update.items():
        if isinstance(value, dict):
            merge_config(base.setdefault(key, {}), value)
        else:
            base[key] = value
    return base


def build_config(options):
    algorithm = {
        "hybridcomm": "HybridComm_one_peer",
        "expocomm": "qmix_ExpoComm_one_peer_n7_cont",
    }[options.algorithm]
    root = Path(__file__).parent / "config"
    config = {}
    for filename in ("default.yaml", "envs/MAgent_TigerDeer.yaml",
                     "algs/" + algorithm + ".yaml", "profiles/tiger_deer.yaml"):
        with (root / filename).open() as stream:
            merge_config(config, yaml.safe_load(stream))
    config["seed"] = options.seed
    config["env_args"].update(map_size=options.map_size, seed=options.seed,
                              deer_policy=options.deer_policy, max_cycles=options.max_cycles)
    initial_tigers = int(options.map_size ** 2 * 0.01)
    config["topk_neighbors"] = math.ceil(math.log2(initial_tigers)) + 1
    # The original contrastive loss reshapes to exactly neg_num negatives.
    config["neg_num"] = min(config["neg_num"], initial_tigers)
    config["learner"] = "tiger_deer_" + config["learner"]
    config["name"] = "HybridComm" if options.algorithm == "hybridcomm" else "ExpoComm"
    config["run_name"] = "tiger_deer_{}_{}_s{}".format(options.map_size, options.deer_policy, options.seed)
    for key in ("t_max", "batch_size", "buffer_size", "test_nepisode"):
        value = getattr(options, key)
        if value is not None:
            config[key] = value
    if options.cpu:
        config["use_cuda"] = False
    if options.evaluate:
        checkpoint = Path(options.checkpoint).expanduser().resolve()
        if not checkpoint.is_dir() or not any(p.is_dir() and p.name.isdigit() for p in checkpoint.iterdir()):
            raise ValueError("--checkpoint must be the models directory containing numeric step directories")
        config.update(evaluate=True, checkpoint_path=str(checkpoint), load_step=options.load_step,
                      buffer_size=1, batch_size=1, save_model=False)
        config["run_name"] += "_eval"
    if config["buffer_size"] < config["batch_size"]:
        raise ValueError("--buffer-size must be at least --batch-size")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("hybridcomm", "expocomm"), required=True)
    parser.add_argument("--map-size", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deer-policy", choices=("flee", "random"), default="flee")
    parser.add_argument("--max-cycles", type=int, default=300)
    parser.add_argument("--steps", dest="t_max", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--test-nepisode", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", help="Matching Tiger-Deer model's models/ directory")
    parser.add_argument("--load-step", type=int, default=0, help="0 selects latest saved checkpoint")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="Print configuration and exit without importing the simulator or trainer")
    options = parser.parse_args(argv)
    if options.map_size < 15 or options.max_cycles < 1:
        parser.error("--map-size must be >= 15 and --max-cycles must be positive")
    for key in ("t_max", "batch_size", "buffer_size", "test_nepisode"):
        value = getattr(options, key)
        if value is not None and value < 1:
            parser.error(key + " must be positive")
    if not 0 <= options.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if options.evaluate != bool(options.checkpoint):
        parser.error("--evaluate and --checkpoint must be supplied together")
    return options


if __name__ == "__main__":
    options = parse_args()
    config = build_config(options)
    if options.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
    else:
        import torch
        from main import main
        from envs import REGISTRY
        from envs.tiger_deer import TigerDeerEnv
        from learners import REGISTRY as LEARNER_REGISTRY
        from learners.tiger_deer_learner import TigerDeerHybridCommLearner, TigerDeerContQLearner
        # Only this dedicated entry point registers the new environment.
        REGISTRY["MAgent_TigerDeer"] = TigerDeerEnv
        LEARNER_REGISTRY["tiger_deer_hybrid_comm_learner"] = TigerDeerHybridCommLearner
        LEARNER_REGISTRY["tiger_deer_cont_q_learner"] = TigerDeerContQLearner
        torch.set_num_threads(1)
        main(config)
