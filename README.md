# D1 RL Token

论文方法级复现，使用 π0.5 基础模型进行 D1 VLA finetune，再训练 RL token 和 online actor-critic。

## 环境

```bash
cd RL_token/
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PWD/.venv/bin/python" -e "$PWD/openpi"
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
```

## step 0：在 pi0.5 上finetune VLA
```bash
source .venv/bin/activate
bash scripts/finetune.sh
```
- 训练参数、数据集、基础模型和输出目录均读取 `rlt/config/config.yaml` 的 `vla` 配置。
- OpenPI 会自动在 `vla.output/config_name/exp_name/` 下按 step 保存 checkpoint。
- 机器人硬件和 runtime 参数读取 `rlt/config/robot.yaml`。


## step 1：在已 finetune 的 VLA 上训练 RL token
```bash
bash scripts/train_rlt_d1.sh
```
- `scripts/finetune.sh` 会把 checkpoint 保存到 `vla.output/config_name/exp_name/`，step 1 会自动选择其中最新的 checkpoint。
- `scripts/train_rlt_d1.sh` 只做特征缓存和 RL token 训练，不会重新 finetune VLA，输出 checkpoint 自动保存到 `rl_token.output/step_<steps>.pt`。

## step 2：在线 actor/critic

```bash
bash scripts/online.sh --execute --human
```

按照下面顺序逐步开放硬件。省略 `--execute` 时只读设备和推理，不发送命令：

```bash
# 1. dry-run base-only
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --episodes 1
# 2. RLT shadow
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1
# 3. 默认 12°/tick 限速执行
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute
# 4. 人工接管演练（空格进入/退出）
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --episodes 1 --execute --human
# 5. 正式在线流程（同时启动 feature server）
bash scripts/online.sh --execute --human
```

- 控制频率固定 10 Hz，按键为 `c` 进入关键阶段、空格切换人工接管、`s/f` 标记成功/失败、`q` 或 Ctrl+C 放弃当前未标注 episode。成功应在玩偶由盘子承托、手已释放并稳定 1 秒后按 `s`。

- `auto` 先由 VLA 自动填充 replay 至 `online_rl.warmup_transitions`，之后每个 episode 从 VLA 开始，操作者按 `c` 将控制交给 RL，按 `q` 放弃当前 episode、Ctrl+C 停止流程；也可用 `--episodes N` 设置本次实验的 episode 数。
SQLite replay 和 online learner checkpoint 位于 `online_rl.output/`，checkpoint 自动命名为 `learner.pt`。

## 评估与检查

每种策略默认 50 次，结果追加到 `checkpoints/online/episodes.jsonl`，结束时打印成功率和每 10 分钟成功次数：

```bash
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task full --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task full --execute
.venv/bin/python -m pytest -q tests/test_rlt.py
git -C openpi status --short
git -C openpi rev-parse HEAD
```

论文的 RL token 输入是 base camera 加左右两个 wrist camera；当前 D1 数据/硬件只有 base camera，config 中显式设置为 `[base_0_rgb]`。
