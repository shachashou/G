# trainer — Python 训练端（M4）

| 文件 | 职责 |
|---|---|
| `binpack.py` | 工单 07 的二进制记录读端（纯标准库 `mmap`）：魔数/版本校验、64 字节记录解码、按对局切分、全量校验、分片合并 |
| `features.py` | **450 维特征索引与 D4 八重对称的权威规范**（纯标准库，供 09 对齐） |
| `weights_format.py` | **量化权重文件格式与整数参考前向**（纯标准库，供 09 对齐） |
| `dataset.py` | binpack mmap 数据集：按完整对局划分 train/val、DataLoader 内实时 D4 增广 |
| `nnue.py` | PyTorch 模型：450 → 累加器 1024 → ClippedReLU → L1 15 → ClippedReLU → 双输出头 1 |
| `train.py` | 训练脚本：`MSE(tanh(score/400)) + CE(WDL)` 混合损失、收敛曲线 JSON、最佳 checkpoint |
| `export.py` | 量化导出 + 回读自校验 + 输出头偏置校准 + float/int 一致性报告 |
| `requirements.txt` | `numpy` + CUDA 版 `torch`（本项目 Python 一律走 `.venv`） |

## 依赖安装

```powershell
cd e:\G
py -3.12 -m venv .venv          # 已有则跳过
.\.venv\Scripts\python.exe -m pip install -r trainer\requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch==2.6.0+cu124` 对应本机 RTX 4080 SUPER；CPU-only 机器可改用 `pip install torch==2.6.0 numpy`，脚本会自动退到 CPU。`engine/tests/training_tests.py` 的规范/格式/划分测试**不需要 torch** 即可运行，装好依赖后会自动追加模型、训练、导出的实测部分。

---

## 跨语言契约 ①：450 维特征索引（C++ ↔ Python）

权威实现 `features.py`，规范如下（**不做视角翻转**）：

- 格子索引 `cell = y * 15 + x`，`x` 向右、`y` 向下，均 `0..14`；
- 黑子 → 特征 `0..224`，索引 = `cell`；白子 → 特征 `225..449`，索引 = `225 + cell`；
- 行棋方**不是特征**，而是选择输出头：`stm = 0`（黑走）/ `1`（白走）；
- 稀疏索引集合 = 所有有子格子的特征索引，升序；`binpack.Record.feature_indices()` 与此完全一致；
- 增量累加器语义：落子 = 加上对应 `W0` 行，撤子 = 减去，`acc` 存 int16 饱和值。

### D4 八重对称（`t = 0..7`）

| t | 名称 | `(x, y) → (x', y')` |
|---:|---|---|
| 0 | 恒等 | `(x, y)` |
| 1 | 顺时针 90° | `(14 - y, x)` |
| 2 | 180° | `(14 - x, 14 - y)` |
| 3 | 顺时针 270° | `(y, 14 - x)` |
| 4 | 左右镜像 | `(14 - x, y)` |
| 5 | 镜像 + 90° | `(14 - y, 14 - x)` |
| 6 | 上下镜像 | `(x, 14 - y)` |
| 7 | 主对角转置 | `(y, x)` |

`transform_cell(cell,t)` / `transform_feature(index,t)` / `transform_sparse(indices,t)` 是权威实现；`symmetry_table()` 给出 dense 向量的 gather 置换。Renju 规则（五连/长连/四四/三三）只依赖直线几何，在 D4 下不变，因此增广时**分数与 WDL 标签都不变换**，只置换特征索引。验证集按完整对局划分，避免同一局面的镜像泄漏进验证集。

---

## 跨语言契约 ②：量化前向（09 的 C++ 标量实现必须逐位复现）

常量：`QA = 127`（累加器 ClippedReLU 上界）、`QB = 64`（L1 ClippedReLU 上界）、`SCORE_SCALE = 400`、`WEIGHT_CLAMP = 64`（int8 权重绝对值上限）。

```
acc[j] = clamp_i16( b0[j] + Σ_{i∈sparse} W0[i,j] )     # int16 饱和，C++ 用 int32 累加再饱和
a[j]   = clamp(acc[j], 0, QA)                          # ClippedReLU
z[k]   = Σ_j a[j] * W1[j,k]                            # int32
zq[k]  = round_div(z[k], QA) + b1[k]
a2[k]  = clamp(zq[k], 0, QB)                           # ClippedReLU
y      = Σ_k a2[k] * W2[h,k]        h = stm (0 黑 / 1 白，双输出头)
eval   = round_div(y, QB) + b2[h]                       # 引擎分数单位
```

- `round_div(n,d) = sign(n) * ((2*|n| + d) // (2*d))`，即四舍五入、**半数远离零**。C++ 不能用普通整数除法（负数向零截断）。
- 权重全为整数：`W0/W1/W2` int8（限幅 ±64）、`b0/b1/b2` int16。中间整数上界约 `1024*127*64 ≈ 8.3M < 2^24`，float32 也能精确表示，因此 PyTorch 的 `forward(..., quantized=True)`（STE 量化 + STE round_div）与 `weights_format.infer_int()` **逐位一致**。
- 训练默认用连续权重 + 普通除法的可微前向；整数化权重初始化与量化正则让训练后取整几乎无损；导出时用采样局面的中位数差校准两个输出头的 `b2`。`export_report.json` 记录 float vs 整数推理的 MAE/最大误差，并给出 09 的对齐容差建议。

### 权重文件格式 `GMNNUE01` v1（64 字节头 + body）

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0..7 | `8s` | magic `b"GMNNUE01"` |
| 8..11 | `u32` | version = 1 |
| 12..15 | `u32` | header_size = 64 |
| 16..19 | `u32` | n_features = 450 |
| 20..23 | `u32` | n_acc = 1024 |
| 24..27 | `u32` | n_l1 = 15 |
| 28..31 | `u32` | n_heads = 2 |
| 32..35 | `u32` | QA = 127 |
| 36..39 | `u32` | QB = 64 |
| 40..43 | `u32` | score_scale = 400 |
| 44..47 | `u32` | weight_clamp = 64 |
| 48..51 | `u32` | FNV-1a 32 位校验（仅 body）：`h=0x811c9dc5; h=(h^b)*0x01000193 (mod 2^32)` |
| 52..63 | 12 × `u8` | 保留，必须为 0 |

body 从偏移 64 起，全部 little-endian、行主序：`W0 int8[450,1024]`、`b0 int16[1024]`、`W1 int8[1024,15]`、`b1 int16[15]`、`W2 int8[2,15]`、`b2 int16[2]`，合计 **478,272 字节**，整文件 478,336 字节。`read_weights()` 校验魔数/版本/尺寸/常量/保留位/校验和/权重范围，任一不符都抛 `WeightFormatError`。

---

## 用法

```powershell
# 训练（自动选 CUDA；--history 里逐 epoch 记录 train/val loss 曲线）
.\.venv\Scripts\python.exe trainer\train.py `
    --data data\raw\round1\train.binpack `
    --output weights\nnue_v1.pt --history weights\training_history.json `
    --epochs 10 --batch-size 1024 --val-fraction 0.1 --seed 42

# 量化导出 + 回读自校验 + float/int 对齐报告
.\.venv\Scripts\python.exe trainer\export.py `
    --checkpoint weights\nnue_v1.pt --data data\raw\round1\train.binpack `
    --output weights\nnue_v1.nnue --report weights\export_report.json --samples 64

# 快速冒烟（小样本 / CPU / 2 epoch）
.\.venv\Scripts\python.exe trainer\train.py --data data\raw\round1\train.binpack `
    --epochs 2 --batch-size 256 --limit 4096 --device cpu
```

- 损失：`MSE(tanh(eval/400), tanh(score/400)) + wdl_weight * CE(WDL) + quant_reg * 量化正则`；WDL 三类 logits 由单一 eval 派生 `[-eval/T, 0, +eval/T] + wdl_bias`，类别顺序 `[行棋方负, 和, 行棋方胜]`，与 binpack 里的 WDL 编码一致。`T = 100`（`--wdl-temperature`）让 CE 项有有效梯度；`wdl_bias` 是**训练专用的三维类别先验参数，不写入权重文件**——没有它，单标量对称 logits 的 CE 下限约 `ln(3)`，损失几乎不下降。推理网络始终只输出 1 个标量。
- 默认 `--epochs 20 --lr 0.1 --batch-size 1024 --wdl-weight 0.2 --quant-reg 0.05 --val-fraction 0.1`。
- `--val-fraction` 按**完整对局**随机抽取（确定性由 `--seed` 控制），`dataset.assert_disjoint()` 会在训练前断言两个 split 无对局交集。
- `--quant-forward` 可切换成整数等价的 STE 前向做量化感知微调；`--quant-reg` 控制接近整数的正则强度。
- 09 的对齐测试可直接用 `weights_format.read_weights()` + `infer_int()` 作为期望值，或 `NNUEModel.from_quantized_weights()` 重建浮点模型。

## 本机实测（RTX 4080 SUPER，torch 2.6.0+cu124）

用 07 生成的 `data/raw/bench-d4/train.binpack`（280 局 / 7208 局面，depth 4）跑默认配置：

```
device: cuda | 6,580 train / 628 val records, 280 games (28 held out)
epoch  1 | train 1.02475 | val 0.96091 (mse 0.77754 ce 0.91686)
epoch 10 | train 0.89241 | val 0.87141 (mse 0.73043 ce 0.70490)
epoch 18 | train 0.87198 | val 0.85424 (best)
epoch 20 | train 0.86661 | val 0.86081
best val loss 0.85424 @ epoch 18; converged=True
```

验证损失 `0.96091 → 0.85424`（-11.1%），CE `0.91686 → 0.70970`（-22.6%），逐 epoch 曲线留在 `weights/training_history.json` 的 `loss_curve`。导出 `weights/nnue_v1.nnue`（478,336 字节）：回读魔数/版本/FNV-1a/维度全通过、整数前向重放一致；`export_report.json` 记录 `float vs integer MAE 0.356 / max 0.978`、`STE vs integer max 0.0000`（64/64 逐位一致），输出头偏置校准 `[-3, -45]`，给 09 的建议容差 `≤ 2`。
