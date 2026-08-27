# D1 RL Token

论文方法级复现，使用 π0.5 基础模型进行 D1 VLA finetune，再训练 RL token 和 online
actor-critic。`openpi/` 固定在 `215abfb`，本项目不会修改它。

## 环境

```bash
cd RL_token/
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PWD/.venv/bin/python" -e "$PWD/openpi"
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
```
## 阶段 0：在 pi0.5 上finetune VLA
```bash
source .venv/bin/activate
bash scripts/finetune.sh
```
训练参数、数据集、基础模型和输出目录均读取 `rlt/config/config.yaml` 的 `vla` 配置。
OpenPI 会自动在 `vla.output/config_name/exp_name/` 下按 step 保存 checkpoint。
机器人硬件和 runtime 参数读取 `rlt/config/robot.yaml`。
`checkpoint_base_dir` 是 OpenPI 的内部字段，表示 checkpoint 根目录；本项目由
`vla.output` 填充，不需要再单独配置。

## 阶段 1：在已 finetune 的 VLA 上训练 RL token

`scripts/finetune.sh` 会把 checkpoint 保存到 `vla.output/config_name/exp_name/`；
阶段 1 会自动选择其中最新的 checkpoint。`scripts/train_rlt_d1.sh` 只做特征缓存和
RL token 训练，不会重新 finetune VLA：

```bash
bash scripts/train_rlt_d1.sh
```

默认 5175 帧、5000 steps、batch 4、Adam 1e-4、seed 42，每 500 steps 保存。
输出 checkpoint 自动保存到 `rl_token.output/step_<steps>.pt`，缓存约 5.4 GB。
阶段 1 的 WandB 由顶层 `wandb_enabled` 控制，项目名和日志间隔也使用顶层
`project_name`、`log_interval`；默认记录 `reconstruction_mse` 和 `learning_rate`
曲线，并保存训练超参数。

## 阶段 2：在线 actor/critic

`scripts/online.sh` 会读取 `rl_token.output/step_<steps>.pt`，启动 GPU feature server，
然后进入 online RL。若 feature server 已单独启动，可设置
`RLT_START_FEATURE_SERVER=false`：

```bash
bash scripts/online.sh --phase auto --execute --human --learner-device auto
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
bash scripts/online.sh --phase auto --execute --human --learner-device auto
```

VLA 的 action chunk 在 `rlt/config/config.yaml` 的 `vla.action_horizon` 中直接设置；
RL token 和 online RL 复用同一个值作为 `C`，transition horizon 和 target discount
也会自动使用 `C`。控制频率及硬件/runtime 参数在 `rlt/config/robot.yaml`，当前控制频率为
10 Hz。论文实验使用 50 Hz、C=10，本地 10 Hz 是 D1 硬件约束下的执行频率。

`auto` 检测到 CUDA 时 actor/critic 在 GPU 异步更新；显存不足可传 `--learner-device cpu`。
控制频率固定 10 Hz，按键为 `c` 进入关键阶段、空格切换人工接管、`s/f` 标记成功/失败、
`q` 或 Ctrl+C 放弃当前未标注 episode。成功应在玩偶由盘子承托、手已释放并稳定 1 秒后按 `s`。

`auto` 不预设 warmup/critical/full 的 episode 数或总时长：先由 VLA 自动填充 replay 至
`online_rl.warmup_transitions`，之后每个 episode 从 VLA 开始，操作者按 `c` 将控制交给 RL，按 `q` 放弃当前 episode、
Ctrl+C 停止流程；也可用 `--episodes N` 设置本次实验的 episode 数。SQLite replay 和
online learner checkpoint 位于 `online_rl.output/`，
checkpoint 自动命名为 `learner.pt`。

## 评估与检查

每种策略默认 50 次，结果追加到 `checkpoints/online/episodes.jsonl`，结束时打印成功率和
每 10 分钟成功次数：

```bash
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task critical --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-base --eval-task full --execute
.venv/bin/python -m robots.d1.online_rl_d1 --phase eval-rlt --eval-task full --execute
.venv/bin/python -m pytest -q tests/test_rlt.py
git -C openpi status --short
git -C openpi rev-parse HEAD
```

论文的 RL token 输入是 base camera 加左右两个 wrist camera；当前 D1 数据/硬件只有
base camera，config 中显式设置为 `[base_0_rgb]`，因此仍是单 base camera 的有意降级。
本实现是 π0.5、12D D1、10 Hz 的方法级复现，不宣称复现论文原始数值。
