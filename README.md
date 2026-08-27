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
```bash
bash scripts/train_rlt_d1.sh
```
- `scripts/finetune.sh` saves checkpoints under `vla.output/config_name/exp_name/`; Step 1 automatically selects the latest checkpoint.
- `scripts/train_rlt_d1.sh` only caches features and trains the RL token. It does not finetune the VLA again. Token checkpoints are saved automatically under `rl_token.output/step_<steps>.pt`.

## Step 2: Online actor-critic training

```bash
bash scripts/online.sh --execute --human
```

Bring up the hardware gradually. Without `--execute`, the program only reads the devices and runs inference; it does not send robot commands:

```bash
# 1. Dry-run with the base VLA only
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --episodes 1
# 2. RLT shadow run
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1
# 3. Execute with the default 12°/tick safety limit
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute
# 4. Test human takeover (press Space to enter/exit)
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute --human
# 5. Full online run (also starts the feature server)
bash scripts/online.sh --execute --human
```

- The control frequency is fixed at 10 Hz. Press `c` to enter the critical phase, Space to toggle human takeover, `s/f` to mark success/failure, and `q` or Ctrl+C to abandon the current unlabeled episode. Mark success only after the doll is supported by the plate, the hand has released it, and the scene has been stable for 1 second.

- In `auto` mode, the VLA first fills the replay buffer to `online_rl.warmup_transitions`. Each subsequent episode starts with the VLA; press `c` to hand control to the RL policy. Press `q` to abandon the current episode or Ctrl+C to stop the run. Use `--episodes N` to limit the number of episodes.
- The SQLite replay buffer and online learner checkpoint are stored under `online_rl.output/`; the checkpoint is automatically named `learner.pt`.

## Evaluation and checks

Each policy runs for 50 episodes by default. Results are appended to `checkpoints/online/episodes.jsonl`, and the program reports the success rate and successful completions per 10 minutes:

```bash
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task full --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task full --execute
.venv/bin/python -m pytest -q tests/test_rlt.py
git -C openpi status --short
git -C openpi rev-parse HEAD
```

The paper's RL token uses the base camera and two wrist cameras. The current D1 dataset and hardware provide only the base camera, which is explicitly configured as `[base_0_rgb]`.
