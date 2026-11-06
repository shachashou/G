# 17: 优化与目标尺度的低算力诊断（不扩网、不加数据）

**What to build:** 工单 15 已回答「等算力下追加 20 万条独立数据是否有收益」（答案：未观察到棋力收益，两组 20 局筛选均 `0W/0D/20L`），但三个解释仍并列。本工单在不改变网络维度、不生成新数据、不消耗正式 200 局开局的前提下，用可证伪的单一变量对照检验「全数据下的优化 / 特征 / 目标尺度限制」，并给出是否值得进入下一轮筛选的证据。

**Blocked by:** None（工单 15 已关闭；本工单是 16 / 10 / 12 僵局的新诊断分支，不占用它们的验收位置）

**Status:** in-progress（阶段 A 已完成，阶段 B 待用户启动）
**Category:** diagnosis

- [x] 交付 `trainer/weight_drift.py`：输入 checkpoint 与初始化 seed，重建同 seed 的初始模型，逐层报告权重相对更新幅度（L2 / max abs / 变化比例）；用 `weights/m5_cmp_small.pt` 与 `weights/m5_cmp_large.pt` 出报告，确认 QAT + `--hidden-lr 0.001` 下隐藏层是否真的几乎不动（结论：**否**，详见 Progress Notes）
- [x] 在同一冻结集上量化标签尺度：±32767 饱和比例、`tanh(score/400)` 目标方差、WDL 类分布、常数预测 MSE 与类别先验 CE，作为后续对照的统一分母
- [ ] 三组单一变量训练：仅分别改 `--hidden-lr`、`--wdl-weight`、`--lr`；其余参数与工单 15 大数据组完全一致（同一冻结验证 JSON、同一 `--train-manifest`、seed 42、batch 1024、`--epochs 300 --max-steps 1600 --checkpoint-policy last`、QAT、`--score-clip 0`）。统一比较 history 每行的 **`val_mse`**，不是 `val_loss`（`--wdl-weight` 会改变后者的构成）
- [ ] 每组一份 `diagnose.py fidelity --frozen-validation`：float / 训练 STE / 最终整数 MSE、常数与类别先验基线、误差比值、C++ scalar/AVX2 逐位一致、全部产物身份
- [ ] **条件化筛选**：仅当某组完整冻结 628 条的 `val_mse` 相对工单 15 大数据组 `0.685776` 有**预先声明**的 ≥5% 相对下降（即 ≤ `0.651487`）时才申请一轮 20 局换色筛选；筛选开局须用**新冻结 seed**（不复用 20261115、不碰 20261109），带 `--verify-backend --strict-gate --min-win-rate 0.65`。否则不跑对打、直接给结论，避免拿阴性结果反复消耗筛选开局
- [ ] 严格筛选入口不变：≥20 局、未舍入得分率严格 >65%、无异常、后端/权重/冻结身份与数值链有效，才把候选交工单 16；验证损失或导出对齐改善都不等于 M5 达标
- [ ] 报告支持的结论与仍并列的未排除解释；预算受首轮剩余约 6,035 秒约束，统一 `budget_run.py` + `data/m5/budget.json`、不静默重置，长任务由用户启动、agent 只分析；不自动扩网、不加数据、不消耗正式开局

**Out of scope:** 改变网络维度 / `GMNNUE01` 布局、生成新数据、真正修改标签变换（`--score-clip` 或分位数/胜负目标会改变冻结身份的 `label_transform` 契约，需另立工单）、改写 runner、正式 200 局验收、M6。

## Staged Plan（用户终端已激活 `.venv`：给用户的命令用 `python`；agent 工具 shell 未激活 venv：agent 执行的命令用 `.venv\Scripts\python.exe`，否则 `train.py` / `diagnose.py` 会因找不到 torch 失败）

### 阶段 A（agent 短任务，不消耗预算硬限）

`trainer/weight_drift.py` **已交付**（验收项 1）：读 checkpoint 的 `config` / `model_state`，以同一 seed 重建 `NNUEModel`，逐层输出 `l2_before` / `l2_delta` / `relative` / `max_abs_delta` / `changed_fraction`，并按 accumulator / L1 / output_head / wdl_bias 分组，写入 JSON。`NNUEModel.reset_parameters()` 使用独立 `torch.Generator`，因此初始权重可精确重建。以下命令已执行：

```powershell
.venv\Scripts\python.exe trainer\weight_drift.py --checkpoint weights\m5_cmp_small.pt --seed 42 --out data\m5\opt_diagnosis\weight_drift_small.json
.venv\Scripts\python.exe trainer\weight_drift.py --checkpoint weights\m5_cmp_large.pt --seed 42 --out data\m5\opt_diagnosis\weight_drift_large.json
.venv\Scripts\python.exe tools\dataset_profile.py --data data\raw\m5_fresh_200k\train.binpack --out data\m5\opt_diagnosis\label_profile.json
```

实际产物：两份权重漂移 JSON、一份全数据质量画像，另由 agent 用 `binpack` + `validation_freeze` 在冻结验证 628 条上算出 `data/m5/opt_diagnosis/frozen_label_scale.json`（目标方差 / 饱和比 / WDL 分布 / 常数与先验基线）。**判定结果：隐藏层并非几乎不动，「只训练输出头」被削弱**，详见文末 Progress Notes；阶段 B 的 H 组改以「优化更新高度偏向输出头的不平衡」为检验对象。

### 阶段 B（用户启动，各 600 秒硬限，顺序执行）

参照组复用工单 15 大数据组产物（`weights/m5_cmp_large.pt` / `data/m5/m5_cmp_large_history.json`，完整冻结 `val_mse=0.685776`），**不重跑**。三组各只改一个参数：

```powershell
# H：隐藏层学习率 0.001 -> 0.1
python tools\budget_run.py --ledger data\m5\budget.json --seconds 600 --out-dir data\m5\opt_h_run -- python trainer\train.py --data data\raw\bench-d4\train.binpack --frozen-validation data\m5\frozen_validation.json --train-manifest data\m5\fresh_200k_manifest.json --output weights\m5_opt_h.pt --history data\m5\m5_opt_h_history.json --epochs 300 --batch-size 1024 --max-steps 1600 --checkpoint-policy last --quant-forward --quant-reg 0 --wdl-weight 0 --lr 0.1 --hidden-lr 0.1 --seed 42

# W：WDL 辅助目标 0 -> 0.2（比较时只看 val_mse）
python tools\budget_run.py --ledger data\m5\budget.json --seconds 600 --out-dir data\m5\opt_w_run -- python trainer\train.py --data data\raw\bench-d4\train.binpack --frozen-validation data\m5\frozen_validation.json --train-manifest data\m5\fresh_200k_manifest.json --output weights\m5_opt_w.pt --history data\m5\m5_opt_w_history.json --epochs 300 --batch-size 1024 --max-steps 1600 --checkpoint-policy last --quant-forward --quant-reg 0 --wdl-weight 0.2 --lr 0.1 --hidden-lr 0.001 --seed 42

# L：输出层学习率 0.1 -> 0.3
python tools\budget_run.py --ledger data\m5\budget.json --seconds 600 --out-dir data\m5\opt_l_run -- python trainer\train.py --data data\raw\bench-d4\train.binpack --frozen-validation data\m5\frozen_validation.json --train-manifest data\m5\fresh_200k_manifest.json --output weights\m5_opt_l.pt --history data\m5\m5_opt_l_history.json --epochs 300 --batch-size 1024 --max-steps 1600 --checkpoint-policy last --quant-forward --quant-reg 0 --wdl-weight 0 --lr 0.3 --hidden-lr 0.001 --seed 42
```

预期产物：三份 `run.json`（`complete: true` / 退出码 0）、三个 checkpoint、三份 history。核对 `config` 除目标变量外一致、`optimisation.optimizer_steps == 1600` / `cap_reached: true`、`dataset.val_records == 628`、冻结身份相同。超时（124）或中断（130）的组不作为有效对照，换新 `--out-dir` 重跑。

### 阶段 C（agent 短任务）

```powershell
.venv\Scripts\python.exe trainer\diagnose.py fidelity --data data\raw\bench-d4\train.binpack --checkpoint weights\m5_opt_h.pt --frozen-validation data\m5\frozen_validation.json --probe build\engine\Release\nnue_probe.exe --samples 64 --out-dir data\m5\m5_opt_h_fidelity
.venv\Scripts\python.exe trainer\diagnose.py fidelity --data data\raw\bench-d4\train.binpack --checkpoint weights\m5_opt_w.pt --frozen-validation data\m5\frozen_validation.json --probe build\engine\Release\nnue_probe.exe --samples 64 --out-dir data\m5\m5_opt_w_fidelity
.venv\Scripts\python.exe trainer\diagnose.py fidelity --data data\raw\bench-d4\train.binpack --checkpoint weights\m5_opt_l.pt --frozen-validation data\m5\frozen_validation.json --probe build\engine\Release\nnue_probe.exe --samples 64 --out-dir data\m5\m5_opt_l_fidelity
```

汇总写 `data/m5/opt_comparison_summary.json`：四组（含参照）的 `val_mse`、常数基线与误差比值、`cpp.exact`、权重/冻结身份、权重漂移结论。**阈值在开跑前已固定为 `val_mse ≤ 0.651487`**，不得事后调整或改挑 `val_loss` / 64 样本口径。

### 阶段 D（仅当阶段 C 达到阈值才执行；开局冻结由 agent 完成，对打由用户启动）

```powershell
# agent 先冻结新 seed 的筛选开局（短任务，不与 20261115 / 20261109 重合）
.venv\Scripts\python.exe tools\match_runner.py --games 20 --paired-openings --seed 20261116 --freeze-openings data\m5\opt_screening_openings.json
python tools\budget_run.py --ledger data\m5\budget.json --seconds 300 --out-dir data\m5\opt_screening_run -- python tools\match_runner.py --games 20 --paired-openings --seed 20261116 --openings data\m5\opt_screening_openings.json --strict-gate --min-win-rate 0.65 --verify-backend --engine-a ".\build\engine\Release\gomoku_nnue.exe --depth 2 --nnue data\m5\m5_opt_h_fidelity\candidate.nnue" --engine-b ".\build\engine\Release\gomoku_nnue.exe --depth 2" --output data\m5\opt_screening.json
```

`--engine-a` 里的权重换成阶段 C 判定的**那一组**。退出码 2 = 完整有效比赛但门控未通过，`run.json` 记 `failed` / `complete: false`，这是真实失败、不得改记；124/130/3/1 才是运行故障。

## 结论边界与队列影响

一组对照只能说明本组固定条件下的结果。即使某组 `val_mse` 改善，也不等于棋力提升或 M5 达标；必须经 ≥20 局严格 >65% 筛选才算候选。**未排除的三个解释仍并列**：数据覆盖不足、全数据下的优化/特征/目标尺度限制、训练目标与棋力指标不同步；不得预设网络容量是根因。

预算：账本当前 `1,164.486 / 7,200` 秒，剩余约 6,035 秒；阶段 B 最多预留 1,800 秒、阶段 D 300 秒，按实际墙钟结算。预算不足时报告未完成项，不新建/重置账本。若全部为阴性，保留原参照、提交下一项诊断建议；工单 16 继续 blocked，工单 10 / 12 的剩余项与工单 11 状态不变，M5 仍未达标，正式 100×2 开局不消耗。

来源：工单 15 `Completion Notes` 的「下一项诊断建议」（2026-11-06），经用户「提交并继续」发布。

## Progress Notes：阶段 A 实测（2026-11-06）

> *This was generated by AI during triage.*

工具已交付：`trainer/weight_drift.py`；`engine/tests/training_tests.py` 新增三条回归断言（已知参数改变量、未改动层零漂移、分组聚合），CTest 复核 **16/16 全绿**。

**权重漂移**（`data/m5/opt_diagnosis/weight_drift_{small,large}.json`，seed 42，两组均 `--hidden-lr 0.001 --lr 0.1 --wdl-weight 0`、1600 步、`checkpoint_policy=last`）：

| 分组 | A 小数据 `relative` | B 大数据 `relative` | `changed_fraction` |
|---|---|---|---|
| accumulator（W0/b0，461,824 参数） | 0.155041 | 0.122519 | 0.999 / 1.000 |
| L1（W1/b1，15,375 参数） | 0.153286 | 0.139985 | 1.000 / 1.000 |
| output_head（W2/b2，32 参数） | **1.574212** | **1.169835** | 1.000 / 1.000 |
| wdl_bias（3 参数） | 0（未训练） | 0（未训练） | 0.000 |

**判定：工单 15 `Completion Notes` 中「隐藏层几乎不动、主要训练输出头」的推测被削弱**。隐藏层并非不动——accumulator 与 L1 的 L2 相对变化约 12–16%，几乎 100% 的元素都发生了改变；只是输出头的相对变化（1.17–1.57）比它高约 **8–10 倍**。准确表述应为「优化更新高度偏向输出头，但隐藏层确实在学」。`wdl_bias` 精确不变（`--wdl-weight 0`）也反向验证了初始状态重建的正确性。阶段 B 的 H 组仍保留：检验的是这种**不平衡**是否限制拟合，而不是「隐藏层完全没学」；W / L 两组设计不变。此处只记录勘误，不改写已提交的工单 15。

**标签尺度**（`data/m5/opt_diagnosis/frozen_label_scale.json`，冻结验证 628 条，同一 `data/m5/frozen_validation.json`，sha256 `730b5c7f…`）：±32767 饱和 `113 / 628 = 0.179936`；`tanh(score/400)` 目标 `mean=-0.261562`、`variance=0.780946`、`min/max=±1`；WDL 仅 `{0: 308, 2: 320}`，**无和棋类**；训练侧拟合常数 `-0.26725` 的验证集 MSE `0.780978`、类别先验 CE `0.693117`。目标方差 `0.780946` 与常数 MSE `0.780978` 几乎一致，说明「常数基线 ≈ 目标方差」在此成立；工单 15 大数据组 `val_mse=0.685776` 只解释了约 **12.2%** 的目标方差，预先声明的阈值 `0.651487` 相当于提升到约 **16.6%**。注意工单 15 的 64 样本保真度基线 `0.750735` 是另一口径，不可混用。

**全数据画像**（`data/m5/opt_diagnosis/label_profile.json`）：199,520 条 / 7,840 局、174,905 原始唯一局面 / 103,180 个 D4 轨道、饱和 35,554 条（0.178198）、`p05=-29669` / `p95=32767`，与工单 14 一致。

阶段 A 未消耗预算账本、未生成新数据、未训练、未跑对打、未触碰正式开局。阶段 B 由用户启动（三组各 600 秒硬限，命令见上）；阈值与 `val_mse` 比较口径已在开跑前固定，不得事后调整。
