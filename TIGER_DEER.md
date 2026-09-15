# Tiger-Deer experiments

This integration targets the cluster's existing **Python 3.8, magent 0.1.14,
PettingZoo 1.15.0** environment. It uses `pettingzoo.magent.tiger_deer_v3`.
Do not replace that stack with MAgent2 for these commands. No simulator package
files need to be copied or patched. The wrapper includes the same `np.bool`
compatibility shim used by the existing training entry point.

The shared controller, both original learner files, and the shared environment
and learner registries are unchanged from the original repository version.
Tiger-Deer uses the original `ExpoCommMAC` and two small learner subclasses in
`src/learners/tiger_deer_learner.py`. These override only the contrastive loss
entry point to skip short batches; TD training and all other methods are inherited.
Use the dedicated `src/tiger_deer_experiment.py` launcher: it registers the
environment and the two `tiger_deer_*` learner names in its own process before
calling the existing trainer. It never replaces the original learner entries.
Battle and Adversarial Pursuit continue using their original classes.
The generic `src/main.py` entry point does not register this environment.

## After pulling on the cluster

From the repository root, activate the existing ExpoComm environment and run:

```bash
python -m unittest discover -s tests -p 'test_tiger_deer*.py' -v
python test_tiger_deer.py
```

The unit tests use a fake simulator and synthetic PyTorch tensors. They check
death handling, time limits, opponent actions, and finite gradient updates for
each guarded learner. They verify that short batches still perform TD updates,
and that normal-length contrastive losses, gradients, and RNG consumption match
the original methods exactly. PyTorch tests explicitly skip if PyTorch is missing.

The smoke test uses the actual wrapper for two short random-tiger episodes.
It prints dependency versions, actual population counts, dimensions, and episode
statistics. Expect 20 tigers, 101 deer, 405 observation features, and 9 actions on
the default map. Zero reward can occur with random tigers.

Inspect the resolved configuration without importing the simulator or trainer:

```bash
python src/tiger_deer_experiment.py --algorithm hybridcomm --print-config
python src/tiger_deer_experiment.py --algorithm expocomm --print-config
```

For a short **training-pipeline check on a compute node**, before a full study:

```bash
WANDB_MODE=offline python src/tiger_deer_experiment.py \
  --algorithm hybridcomm --steps 3000 --max-cycles 100 \
  --batch-size 2 --buffer-size 4 --test-nepisode 2 --deer-policy random

WANDB_MODE=offline python src/tiger_deer_experiment.py \
  --algorithm expocomm --steps 3000 --max-cycles 100 \
  --batch-size 2 --buffer-size 4 --test-nepisode 2 --deer-policy random
```

These short checks validate the pipeline, not policy quality. The normal
exploration schedule is much longer. Only you should launch these commands.

## Training runs

Within an allocated GPU job:

```bash
python src/tiger_deer_experiment.py --algorithm hybridcomm --map-size 45 --seed 42
python src/tiger_deer_experiment.py --algorithm expocomm --map-size 45 --seed 42
```

Or submit separate Slurm jobs from the repository root:

```bash
mkdir -p logs
sbatch --job-name=td_HC_45_s42 job_tiger_deer.sh --algorithm hybridcomm --map-size 45 --seed 42
sbatch --job-name=td_EC_45_s42 job_tiger_deer.sh --algorithm expocomm --map-size 45 --seed 42
```

`job_tiger_deer.sh` reuses the Conda paths in your existing `job.sh`, requests one
GPU and 60 GB RAM, and forwards all arguments to the launcher. Add your cluster's
partition, account, or node-exclusion options to `sbatch` if needed. It does not
install anything. Create `logs/` before submission because Slurm opens logs before
the script starts. Set `WANDB_MODE=offline` before submission if the node has no
network access; otherwise the existing W&B logging is retained.

Repeat both algorithms with the same seed set, e.g. 42, 43, 44. Defaults are 5M
joint environment steps, 300-step episodes, 20 evaluation episodes every 50k
steps, batch size 8, and 64 episodes in CPU replay. The existing runner can exceed
the step budget by one episode. Shared settings live in
`src/config/profiles/tiger_deer.yaml`; environment settings live in
`src/config/envs/MAgent_TigerDeer.yaml`.

## Larger populations

Map size controls population at the environment's standard densities: about 1%
tigers and 5% deer. The wrapper derives actual tiger counts from agent names,
without a fixed map-size lookup table.

| Map size | Approximate tigers | Approximate deer |
| --- | ---: | ---: |
| 45 | 20 | 101 |
| 100 | 100 | 500 |
| 200 | 400 | 2,000 |
| 320 | 1,024 | 5,120 |

For example, after validating smaller cases:

```bash
sbatch --job-name=td_HC_320_s42 job_tiger_deer.sh \
  --algorithm hybridcomm --map-size 320 --seed 42 --batch-size 2 --buffer-size 8
sbatch --job-name=td_EC_320_s42 job_tiger_deer.sh \
  --algorithm expocomm --map-size 320 --seed 42 --batch-size 2 --buffer-size 8
```

These are configurations, not verified throughput or memory guarantees. The
replay buffer preallocates full episodes for all initial tiger slots, even after
deaths. Observation storage alone for 1,024 tigers, 301 frames, and 8 episodes is
about 3.7 GiB; other fields, sampled batches, and learning activations add memory.
Reduce batch/buffer sizes together for both models if necessary. Large runs
should start with a short cluster-side pipeline check.

The launcher selects `ceil(log2(initial_tigers)) + 1` topology entries, including
self, from the standard map density. Agent IDs are **not** appended as N-wide
one-hot inputs. A larger map keeps density approximately constant; a larger
population does not automatically increase the coordination distance required.
For maps with fewer than 20 initial tigers, `neg_num` is reduced to that initial
count because the original contrastive sampler requires `neg_num <= n_agents`.
The count remains fixed after deaths, as in the original learners.

## Checkpoint evaluation

Outputs follow the existing layout:

```text
work_dirs/TigerDeer/{HybridComm,ExpoComm}/tiger_deer_<size>_<opponent>_s<seed>/<timestamp>/models/<step>/
```

Use the `models/` directory containing numeric step subdirectories:

```bash
python src/tiger_deer_experiment.py --algorithm hybridcomm --map-size 45 \
  --seed 1000 --deer-policy flee --evaluate --test-nepisode 100 \
  --checkpoint /path/to/the/training/run/models
```

Repeat for ExpoComm using its matching checkpoint and the same evaluation seed,
map, opponent, episode limit, and number of episodes. `--load-step 0` selects the
latest checkpoint; another number selects the closest saved step. Checkpoints
should be evaluated with their training population and architecture. ExpoComm's
QMIX mixer is population-dependent; this launcher does not implement cross-size
checkpoint transfer. Evaluation uses a minimal replay allocation and never
falls through to training when a checkpoint argument is missing.

The wrapper reseeds each reset with `seed + reset_index` (modulo 2^32), also used
for the separate opponent RNG. This makes standalone evaluations reproducible
without one model's action sampling changing the opponent's RNG stream.

## Opponent and learning semantics

- **Only tigers learn.** Default `flee` deer use only native 3x3 observations,
  avoid locally blocked cells, and maximize distance to the nearest visible
  tiger. Ties and movement without visible tigers use a seeded RNG. This is a
  fixed heuristic, not a pretrained or benchmark-winning opponent. `random`
  deer are available for debugging through `--deer-policy random`.
- Tiger observations retain all five native 9x9 channels, including obstacles.
  No global minimap is supplied to actors. Action indices are the native MAgent
  indices; movement/stay occupies 0–4, **stay is 2**, and attacks occupy 5–8.
- The centralized training state is a **9x9 spatially pooled global grid**:
  obstacle occupancy, deer occupancy/health, tiger occupancy/health. Values are
  sums divided by bin area. It is a lossy global summary, not the full grid and
  not a single tiger's local observation. This keeps replay and prediction-head
  dimensions bounded. Both models receive the same state; the QMIX baseline
  consumes it in its mixer. `state_grid_size` is configurable in the environment
  YAML and must remain the same for checkpoint loading.
- Reward is the sum of native tiger rewards divided by the **initial** tiger
  population. That scalar is repeated for the existing per-agent runner format.
  Each learner retains its original processing: HybridComm uses each slot's
  reward for its local TD target, while ExpoComm sums the slots for its QMIX
  target. Both standardize rewards before those target calculations by default.
  Consequently, their training reward aggregation/scales differ; sharing the
  environment and hyperparameters does not make their TD objectives identical.
  Native reward/health rules are unchanged.
- Initial tiger slots and the exponential communication ring remain fixed.
  The environment adapter gives dead slots zero observations and only the stay
  action, and omits their actions when stepping the simulator. The original
  controller continues computing their recurrent states, messages, and Q-values;
  these are **not** forced to zero. Dead slots remain in communication, TD losses,
  and contrastive sampling. The original learners retain their timestep masks
  for padded data. No additional death masking or alternative contrastive loss
  is installed.
- Time limits retain survivor observations for bootstrapping. Extinction is a
  true terminal transition. The adapter rebuilds the global state from engine
  data, avoiding legacy PettingZoo's stale wall cache and empty-group state bug.

The previous `mask_dead_agents`, `shared_reward_mean`, and `alive_contrastive`
additions have been removed. Shared files and existing learner registry entries
remain unchanged.

The original contrastive loss assumes enough timesteps to sample temporally
separated negatives. The Tiger-Deer subclasses return zero auxiliary loss when
the actual sampled sequence has fewer than `2 * (topk_neighbors - 1) + 2`
transition timesteps (12 on the default map). This conservatively skips short
batches that cannot guarantee a valid negative for every possible reference
timestep, including batches shortened by natural extinction. Normal TD learning
continues. The existing runner trims batches to their longest sampled episode;
the final observation used for bootstrapping is not counted as a transition.
At or above the threshold, the original contrastive method runs unchanged,
including its random sampling and padding masks. Positive training horizons
shorter than the threshold are allowed, but will train without contrastive loss.

Track `test_return_mean` (shared reward per initial tiger),
`test_deer_killed_mean`, `test_tiger_survival_fraction_mean`, and
`test_tiger_return_total_mean`. `positive_tiger_reward_events` counts individual
positive reward occurrences, **not** unique joint attacks or kills. Report
opponent type, population, seeds, pooled state, reward normalization, memory,
and wall-clock speed alongside performance.

## Local validation scope

No MAgent rollouts or experiment runs were launched during implementation.
The actual wrapper smoke test and hardware-specific training check must be run
on the cluster. Synthetic tests do not establish simulator throughput or
learning performance.
