# D1 RL Token

A method-level reproduction of the RL Token paper using the π0.5 base model:
finetune the D1 VLA, train the RL token, and run online actor-critic training.

## Environment

```bash
cd RL_token/
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PWD/.venv/bin/python" -e "$PWD/openpi"
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
```

## Step 0: Finetune the VLA on π0.5
```bash
source .venv/bin/activate
bash scripts/finetune.sh
```
- Training parameters, the dataset, the base model, and the output directory are read from the `vla` section of `rlt/config/config.yaml`.
- OpenPI automatically saves checkpoints by step under `vla.output/config_name/exp_name/`.
- Robot hardware and runtime parameters are read from `rlt/config/robot.yaml`.


## Step 1: Train the RL token on the finetuned VLA
cache finetuned VLA features
```bash
bash scripts/train_rlt_d1.sh cache
```
reuse cache and start to train RL token
```bash
bash scripts/train_rlt_d1.sh train
```
run cache + train together
```bash
bash scripts/train_rlt_d1.sh
```

- `scripts/finetune.sh` saves checkpoints under `vla.output/config_name/exp_name/`; Step 1 automatically selects the latest checkpoint.
- `scripts/train_rlt_d1.sh` only caches features and trains the RL token. It does not finetune the VLA again. Token checkpoints are saved automatically under `rl_token.output/step_<steps>.pt`.

## Step 2: Online actor-critic training

```bash
# First training run: VLA warmup -> offline Actor/Critic warmup -> critical-only rollout
bash scripts/online.sh --phase critical-only --execute --human

# Continue the same learner and collect full-task rollouts
bash scripts/online.sh --phase full-task --execute --human
```

Bring up the hardware gradually. Without `--execute`, the program only reads the devices and runs inference; it does not send robot commands:

```bash
# 1. Dry-run with the base VLA only
python -m robots.d1.online_rl_d1 --phase eval-base --episodes 1
# 2. RLT shadow run; press c to hand off from VLA to Actor
python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1
# 3. Execute with the default 12°/tick safety limit
python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute
# 4. Test human takeover (press Space to enter/exit)
python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute --human
# 5. Full online run; the first run performs bootstrap automatically
bash scripts/online.sh --phase critical-only --execute --human
```

- The control frequency is fixed at 10 Hz. VLA proposes `H=20` actions, while the VLA and RL policies execute `C=10` actions at each control boundary. After the initial pose is reached, a separate camera thread opens the `D1 Online RL` window and shows live frames. During an episode, press Enter to end it; the robot stays at its last commanded pose and the camera/data thread pauses. Then press `s` or `f` to label success/failure. Press `c` to hand off to the Actor, `y` to approve one VLA chunk in `critical-only`, Space to toggle human takeover, and `q` or Ctrl+C to abandon the current episode without adding it to replay. Mark success only after the doll is supported by the plate, the hand has released it, and the scene has been stable for 1 second.

- On a training run without a learner checkpoint, the VLA-only bootstrap collects `online_rl.warmup_transitions` stride-generated transitions into `warmup_replay.sqlite`. Actor/Critic then perform `online_rl.offline_warmup_updates` updates from that buffer (default 5000), save `learner_latest.pt`, and start the requested phase. A later run automatically resumes the latest checkpoint and skips bootstrap. Use `--episodes N` to limit only the requested rollout phase.
- `critical-only` waits for `y` before each VLA chunk; `full-task` executes the VLA prefix continuously. In both training phases, only transitions after `c` are written to the formal online replay. Human intervention replaces both the executed action and its reference in those transitions.
- When `wandb_enabled: true`, WandB records bootstrap and online metrics separately. Set `WANDB_API_KEY` directly or put it in `wandb.env` before running `scripts/online.sh`.
- The warmup SQLite buffer is stored at `online_rl.output/warmup_replay.sqlite`; formal online replay is `online_rl.output/replay.sqlite`. Rollout checkpoints are saved under `online_rl.output/rollout/` as `learner_latest.pt` plus periodic `learner_step_<critic_steps>.pt` snapshots. The latest checkpoint is resumed automatically, or choose one with `--resume PATH`.

## Evaluation and checks

Each policy runs for 50 episodes by default. Results are appended to `checkpoints/online/episodes.jsonl`, and the program reports the success rate and successful completions per 10 minutes:

```bash
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task critical-only --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task critical-only --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task full-task --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task full-task --execute
.venv/bin/python -m pytest -q tests/test_rlt.py
git -C openpi status --short
git -C openpi rev-parse HEAD
```

The paper's RL token uses the base camera and two wrist cameras. The current D1 dataset and hardware provide only the base camera, which is explicitly configured as `[base_0_rgb]`.
