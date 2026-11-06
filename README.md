# GomokuNNUE — 五子棋（Renju）NNUE 引擎与训练流水线

一个"能玩、能训练、能自我对弈"的五子棋 AI 项目：纯 C++ 命令行引擎（Renju 禁手规则 + NNUE 评估 + Alpha-Beta 搜索），配 Python/PyTorch 训练端，在一台 Windows 机器（RTX 4080 + 多核 CPU）上完成从自我对弈数据生成到量化权重加载的完整闭环。

**设计契约的权威版本见 `.scratch/gomoku-nnue/spec.md`，工单见 `.scratch/gomoku-nnue/issues/`。**

**2026-11-05 M5 诊断决策更新**：先解决 M5 再推进 M6；固定小样本拟合与训练原模型到最终量化导出的保真度检查优先，尚不能认定网络容量是根因。执行契约见 [工单 12](.scratch/gomoku-nnue/issues/12-m5-diagnostics-and-gate.md)。正式门控现约定为 100 个配对开局 / 200 局、Renju / depth 2 / VCF 开启 / 64MB TT，得分率严格 >65%；runner 的 `--m5-gate` 已支持该契约，下文旧比赛命令仅作历史诊断/工具用法示例。首轮实验累计运行预算 2 小时；预计 >5 分钟或无法估时的任务交用户自行运行，agent 准备命令并分析产物。

诊断、配对开局与严格门控工具已准备，命令和结果说明见 [M5 用户运行指南](tools/M5.md)。正式比赛与长时间诊断仍须按阶段由用户运行。

## 核心决定（已达成的共识）

- **规则**：默认 Renju（黑方三三/四四/长连禁手，五连优先于禁手，白无禁手、长连算胜）；`INFO rule` 可切换 freestyle（无禁手）。自由开局，不做开局交换规则。
- **引擎**：C++20 / MSVC / AVX2（无运行时分发），单线程，Zobrist + 64MB 置换表，PVS + 杀手/历史启发，VCF 算杀。
- **交互**：无任何图形，纯 stdin/stdout，说 Gomocup/piskvork 协议子集（`START` / `TURN` / `BOARD` / `INFO` / `ABOUT` / `END`），挂现成 GUI 对弈。
- **NNUE**：450 维颜色绝对特征（黑 225 + 白 225）→ 单累加器 1024（int16）→ ClippedReLU → L1 15 → ClippedReLU → 输出 1（int8 权重）；双输出头按行棋方选择。落地顺序：标量对齐 PyTorch → AVX2 增量推理。
- **训练**：Stockfish 路线（非 AlphaZero）；损失 `MSE(tanh(score/400)) + CE(WDL)`；D4 八重对称实时增广；验证集按对局划分。
- **数据**：自我对弈，开局前 3 手中心 7×7 带温度随机，depth-6 搜索；64 字节定长局面记录；首轮目标 ~1000 万局面。

## 目录结构

```
├── CMakeLists.txt        # 根构建配置（C++20 / MSVC / AVX2）
├── engine/               # C++ 引擎核心
│   └── src/              #   棋盘/规则/搜索/协议/NNUE 推理、datagen 写端（binpack）
├── trainer/              # Python 训练端：binpack 读端（已完成）+ 模型/训练/导出（M4）
├── tools/                # 配套工具：match_runner、datagen 编排（均已完成）
├── data/                 # 数据集（git 忽略，仅保留目录）
└── weights/              # 权重与 checkpoint（git 忽略，仅保留目录）
```

## 编译（Windows + Visual Studio 2022）

前置：Visual Studio 2022（含 C++ 桌面开发工作负载）与 CMake ≥ 3.20。CMake 不一定在 PATH 中——最省事的方式是从开始菜单打开 **"Developer PowerShell for VS 2022"**（自动配好 CMake 与 MSVC 环境），或在普通终端直接使用 VS 自带的 CMake（本机位于 `D:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`）。

```powershell
# 配置（x64）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# 编译
cmake --build build --config Release

# 查看版本（打印引擎标识后退出）
.\build\engine\Release\gomoku_nnue.exe --version

# 默认：Alpha-Beta 搜索 + 手工评估，从 stdin 读 Gomocup 协议命令
# （START / TURN / BOARD / INFO / ABOUT / END），挂现成 GUI 对弈
.\build\engine\Release\gomoku_nnue.exe

# 常用选项
.\build\engine\Release\gomoku_nnue.exe --depth 4      # 固定深度（测试用，忽略时间）
.\build\engine\Release\gomoku_nnue.exe --movetime 500 # 每步思考 500ms（默认 1000）
.\build\engine\Release\gomoku_nnue.exe --random       # 随机合法走子基线，用于强度对照
.\build\engine\Release\gomoku_nnue.exe --no-vcf       # 关闭 VCF 算杀（M2 基线，用于 A/B）
.\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue  # NNUE 评估（默认 AVX2 后端）
.\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue --nnue-backend scalar  # 标量后端（对拍/回退）
.\build\engine\Release\gomoku_nnue.exe --help

# 自我对弈 datagen worker（工单 07，单进程单分片；由 tools/datagen.py 拉起多个）
.\build\engine\Release\datagen.exe --help
```

`--nnue PATH` 加载 `GMNNUE01` v1 量化权重（不给该参数则使用手工威胁评估）。推理后端有三档：**手工**（无 `--nnue`）、**标量 NNUE**（`--nnue-backend scalar`）、**AVX2 NNUE**（默认），两档 NNUE 输出逐位一致；权重只在启动时加载一次。文件缺失或格式不符会在 **stderr** 报清晰错误并回退手工评估，未知后端名回退 AVX2，都不会污染 Gomocup 的 stdout 协议流。`ABOUT` 的 `eval=` 与 `nnue_backend=none|scalar|avx2` 可用于确认当前配置。本机实测单次评估：标量 `3.12 µs`、AVX2 `1.15 µs`（2.70×）。`moveHeuristic()`（候选排序 / datagen 开局采样）始终使用手工评估，保证 datagen 可复现。

时间控制：`INFO timeout_turn`（留 20% 余量）、`INFO time_left`（按剩余时间 1/20 封顶）、`INFO timeout_match` 均被接受。

## 测试

```powershell
# 创建项目独立 Python 环境（一次性；项目 Python 一律走 .venv）
py -3.12 -m venv .venv

ctest --test-dir build -C Release --output-on-failure
```

包含 7 个 C++ 单元测试（`board_tests` / `renju_tests` / `eval_tests` / `search_tests` / `vcf_tests` / `binpack_tests` / `nnue_tests`）与 9 个 Python/黑盒测试（`m5_tests` / `protocol_blackbox` / `match_runner_tests` / `datagen_tests` / `training_tests` / `validation_freeze_tests` / `nnue_parity_tests` / `nnue_backend_tests` / `nnue_blackbox`），共 16 个 CTest；Python 测试都经项目 `.venv` 运行。`m5_tests` 的门控/预算部分以及 `training_tests`、`nnue_parity_tests` 的规范/格式/对齐部分不依赖 torch，装好训练依赖后会自动追加模型、训练、导出与 PyTorch 浮点对齐的实测。也可手动执行：

```powershell
.venv\Scripts\python.exe engine\tests\protocol_blackbox.py build\engine\Release\gomoku_nnue.exe
.venv\Scripts\python.exe engine\tests\match_runner_tests.py build\engine\Release\gomoku_nnue.exe tools\match_runner.py
.venv\Scripts\python.exe engine\tests\datagen_tests.py build\engine\Release\datagen.exe trainer\binpack.py tools\datagen.py
.venv\Scripts\python.exe engine\tests\training_tests.py
.venv\Scripts\python.exe engine\tests\validation_freeze_tests.py
.venv\Scripts\python.exe engine\tests\nnue_parity_tests.py build\engine\Release\nnue_probe.exe
.venv\Scripts\python.exe engine\tests\nnue_backend_tests.py build\engine\Release\nnue_probe.exe
.venv\Scripts\python.exe engine\tests\nnue_blackbox.py build\engine\Release\gomoku_nnue.exe
```

黑盒内含双引擎自对弈、规则切换、时间控制、VCF 杀局与搜索引擎对随机基线 6 局全胜的强度验收；`match_runner_tests` 覆盖 100% 胜率门控、结果落盘、freestyle 规则切换与超时/崩溃/非法着法三类异常计负；`datagen_tests` 用真实 worker 落盘，再校验魔数/64 字节布局/WDL 回填/对局边界/分片合并/多进程隔离，并让 C++ 复盘抽样搜索分、检出被篡改的分数；`training_tests` 校验 450 维特征索引/D4 群性质、量化权重格式与整数前向、对局级数据划分，并在 torch 可用时跑通模型→训练→导出；`nnue_tests` 覆盖权重格式拒绝、`round_div`、手算整数前向、int16 饱和与增量累加器对拍；`nnue_parity_tests` 用 `nnue_probe` 比对 C++ 特征索引/标量推理与 Python 整数参考（逐位）及 PyTorch 浮点前向（容差内）；`nnue_backend_tests` 做 scalar/AVX2 逐位对拍并用 `nnue_probe bench` 给出 ns/eval 实测；`nnue_blackbox` 验证 NNUE 完整对局、三档后端切换与坏权重回退。

## 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 | 棋盘 + Renju 规则 + 着法生成 + 单元测试 | 公开禁手题库判定全对 |
| M2 | Alpha-Beta + 手工评估 + Gomocup 协议 | 挂 GUI 完整对局 |
| M3 | VCF 算杀接入搜索 | 标准 VCF 杀局题全解 |
| M4 | 自我对弈 datagen + 训练流水线 | 训出第一个 NNUE 权重 |
| M5 | C++ NNUE 推理（标量→AVX2）替换手工评估 | 同搜索配置下对 M3 胜率 >65% |
| M6 | 自举迭代：datagen→训练→门控循环 | 强度逐轮爬升 |

VCT（活三威胁算杀）为 M6 之后的可选进阶项。

**最终模型在哪产生**：没有独立的「最终训练」阶段。M4 的 `nnue_v1.nnue` 只验证闭环；M5 门控通过后产生第一个现任权重（incumbent）；M6（工单 11）每轮自举重训并由门控决定是否晋级，至少 3 轮且连续 2 轮未晋级即收敛，把当时的现任权重冻结为 `weights/final.nnue` 并写 `weights/final_manifest.json`。最终模型 = 被门控保留的最后一个晋级版本，而非最后一轮训练产物。详见 `.scratch/gomoku-nnue/spec.md` 与 `issues/11-bootstrap-loop.md`。

**当前进度**：工单 01–09 完成，工单 10 的 AVX2 提速/三档切换/耗时三项已完成，**M4 达成、M5 尚未达成**。自我对弈 datagen「生成→落盘→合并→校验→复盘」闭环 + Python 训练端「加载→D4 增广→GPU 训练→量化导出」闭环均已跑通，RTX 4080 SUPER 上 20 epoch 验证损失 `0.9609 → 0.8542`。C++ 端标量推理跨语言对齐实测 PyTorch 浮点 vs C++ `MAE 0.327 / max 0.737`（容差 ≤2），AVX2 后端逐位对齐且单次评估 `1.15 µs`（标量 `3.12 µs`，2.70×），CTest 16/16 全绿。

**M5 剩余链路（2026-11-06）**：13/14/15/17已关闭；工单18也按完整阴性关闭：手工蒸馏D组/搜索分尺度S组均完成1600步及全量导出验收，但最终整数Top-1仅3.7879%/3.1987%（随机3.0812%、教师43.4203%），未过第一门控，按规则不执行20局筛选。数值一致不等于走子能力提升；16正式200局继续blocked，10/12未完成，11仍受M5约束。结果与复现流程见 [tools/M5.md](tools/M5.md)，预算2244.407/7200秒。

**M5 胜率门控未达标**：旧 100 局相同配置（`--depth 2`，NNUE vs 手工评估 + VCF）实测 `0W / 0D / 100L`，退出码 2，无异常（`data/m5_gate.json`）。7,208 条训练标签里 17.9% 是 ±32767 饱和搜索分，v1 模型输出仅 `[-185, 30]`；三组 100-epoch 重训也未明显改善棋力。但这些证据不足以排除数据覆盖、优化或量化问题，先前「网络容量已确认为瓶颈」及「解释 16% 方差」的推断不再作为结论。导出报告中约 0.36 的误差来自导出权重重建模型与整数推理；训练原模型到最终导出的保真度仍待核验。按工单 12 的诊断顺序，再凭证据选择数据或模型调整；单改训练目标且推理契约不变时，不必改变 `GMNNUE01` 布局。

## 工具用法

### `tools/match_runner.py`（工单 06）

runner 自己做裁判：每步用 `BOARD` 把权威棋盘全量同步给待走方引擎，校验并计时每条回复；两个引擎不共享增量状态。判负/判和遵循五连优先、Renju 黑恰好五连、黑长连非法、白/freestyle 长连胜（三三/四四禁手由引擎自身 `isPlayable` 过滤）。

```powershell
# 相同搜索配置下比较两版引擎（两边务必给相同的 --depth / --movetime）
.venv\Scripts\python.exe tools\match_runner.py `
    --engine-a ".\build\engine\Release\gomoku_nnue.exe --depth 4" `
    --engine-b ".\build\engine\Release\gomoku_nnue.exe --depth 4 --no-vcf" `
    --games 20 --rule 2 --opening-plies 2 --seed 7 `
    --output data\match_results.json

# 强度门控（M5：NNUE 对 M3 胜率 >65%；不达标退出码 2）
# 注意：当前首轮权重实测 0/100 未达标，需先补数据重训（见下文进度说明）
.venv\Scripts\python.exe tools\match_runner.py `
    --engine-a ".\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue --depth 2" `
    --engine-b ".\build\engine\Release\gomoku_nnue.exe --depth 2" `
    --games 100 --min-win-rate 0.65 --output data\m5_gate.json

# 自检：搜索引擎对 02 随机基线，交替先手 4 局应 100% 胜
.venv\Scripts\python.exe tools\match_runner.py `
    --engine-a ".\build\engine\Release\gomoku_nnue.exe --depth 2" `
    --engine-b ".\build\engine\Release\gomoku_nnue.exe --random" `
    --games 4 --min-win-rate 1.0 --output data\match_selfcheck.json
```

- `--engine-a/-b`：完整命令串（可含引号包住的 exe 路径与 CLI 参数）；`--games` 按局号奇偶交替先后手；`--rule 1|2` 下发 `INFO rule`（1=freestyle，2=Renju 默认）。
- `--opening-plies 0..8`（默认 2）、`--opening-region`（默认中心 7×7）、`--seed`：随机开局，≤8 手时黑子 ≤4，Renju 禁手不可能出现。
- `--move-timeout`（每步回复超时秒数，默认 10）、`--timeout-turn`（下发 `INFO timeout_turn` 毫秒）、`--max-plies`（默认 225，到限判和）。
- 超时、崩溃 / stdout 关闭、坏回复、越界或占用点、Renju 黑长连：一律记为**该方失败**并写入 anomaly 明细。
- `--output` 的 JSON 含运行配置、W/D/L 与胜率统计、以及每局完整着法序列（含随机开局标记与走子方归属），可离线复查。`--min-win-rate` 门控使用 score rate `(wins + 0.5*draws)/games`，JSON 里同时给出严格胜率 `decisive_win_rate`。

### `tools/datagen.py`（工单 07）

自我对弈数据生成：C++ worker（`datagen.exe`）单进程独占一个分片文件与一条 RNG 流，开局前 3 手在中心 7×7 按评估启发式做温度 softmax 采样（随机开局局面不记录），此后每步固定深度搜索并落一条 64 字节记录；对局结束时把 WDL 回填进该局每条记录。多进程互不共享状态，分片可合并成单一 `train.binpack`。**二进制格式（魔数、字段布局、WDL、分数饱和、边界位、合并规则）与实测吞吐报告见 `tools/README.md`**，Python 读端为 `trainer/binpack.py`。

```powershell
# 首轮生产配置（本机 28 逻辑核实测 285 局面/秒，1000 万局面约 9.7 小时）
.venv\Scripts\python.exe tools\datagen.py run `
    --workers 28 --games-per-worker 100 --depth 4 `
    --out-dir data\raw --tag round1 --seed 20261105 --merge

# 校验格式与 WDL 回填；复盘抽查搜索分是否与引擎重新搜索一致
.venv\Scripts\python.exe tools\datagen.py validate --input data\raw\round1\train.binpack
.venv\Scripts\python.exe tools\datagen.py verify --input data\raw\round1\train.binpack --samples 64

# 换机器先校准吞吐并按半天预算选 depth / 并行度
.venv\Scripts\python.exe tools\datagen.py calibrate --workers 28 --depths 6,5,4,3 --games-per-worker 3
```

- `--depth` 默认 6（规格值），但本机实测 depth 6 约 3 局面/秒（1000 万需约 870 小时），已按工单预案降为 depth 4 并把并行度从 16 提到 28；详见 `tools/README.md` 的吞吐表。
- `run` 产出 `manifest.json`：每秒局面数、1000 万局面预计耗时、是否达标与调整建议；`calibrate` 比较多个 depth 并给出推荐。
- `--verify` 由 C++ 端从两个 225-bit 棋盘平面复原局面并重新搜索，用来确认每条记录的搜索分可复现（生成时每步清空置换表/启发式保证确定性）。

### `trainer/`（工单 08）

Python 训练端：`binpack.py` 读端之上是 `dataset.py`（mmap + 按完整对局划分 + DataLoader 内实时 D4 八重增广）、`nnue.py`（450→1024→15→双输出头，ClippedReLU，int8/int16 量化域）、`train.py`（`MSE(tanh(score/400)) + CE(WDL)`，收敛曲线 JSON）、`export.py`（带魔数/版本/校验和的 `GMNNUE01` 权重 + 回读自校验）。**450 维特征索引、D4 变换、量化前向与权重文件格式的权威规范见 `trainer/README.md`**，那是工单 09 跨语言对齐的依据。工单 10 追加了可选 `--score-clip N`（默认 0 关闭）：把 datagen 饱和的 ±32767 搜索分裁剪到 ±N 再进 `tanh`，损失形式与权重格式不变；其饱和标签限制与实测结论见 `trainer/README.md` 的「已知限制」。

```powershell
# 依赖（一次性，CUDA 版 torch）
.\.venv\Scripts\python.exe -m pip install -r trainer\requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/cu124

# 训练 → 量化导出
.\.venv\Scripts\python.exe trainer\train.py `
    --data data\raw\round1\train.binpack `
    --output weights\nnue_v1.pt --history weights\training_history.json --epochs 10
.\.venv\Scripts\python.exe trainer\export.py `
    --checkpoint weights\nnue_v1.pt --data data\raw\round1\train.binpack `
    --output weights\nnue_v1.nnue --report weights\export_report.json
```

### 工单 09：C++ 标量 NNUE 推理（`engine/src/nnue.h/.cpp`）

`--nnue PATH` 把叶子评估切到 NNUE。C++ 端逐位复现 `trainer/weights_format.py` 的整数前向：`acc = clamp_i16(b0 + ΣW0)` → `ClippedReLU(0,127)` → `L1` → `round_div(z,127)+b1` → `ClippedReLU(0,64)` → 按行棋方选双输出头；`round_div` 为半数远离零。特征索引、文件格式与校验规则的权威说明仍见 `trainer/README.md`。工单 09 的实现是**纯标量**（无 SIMD）数值基准：搜索内用 int32 累加器做落子/撤子差分更新，只在推理时饱和一次，因此与全量稀疏参考前向逐位一致；工单 10 之后默认改走 AVX2，标量仍可用 `--nnue-backend scalar` 复现。`moveHeuristic()` 仍走手工评估以保证 datagen 确定性。

```powershell
# 跨语言对拍（nnue_probe 把 C++ 索引/推理交给 Python 与 torch 比对）
.venv\Scripts\python.exe engine\tests\nnue_parity_tests.py build\engine\Release\nnue_probe.exe

# 用真实导出权重下棋（ABOUT 里应显示 eval="nnue"）
.\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue --depth 4
```

实测：40 个采样局面上 C++ 稀疏索引与 `features.sparse_indices()` 完全一致、标量/增量两条路径与 `infer_int()` 逐位相等；真实权重上 PyTorch 浮点前向 vs C++ `MAE 0.3273 / max 0.7367`（工单 08 建议容差 ≤2）；两台 NNUE 引擎自对弈完整终局，坏权重/缺失权重只在 stderr 报错并回退手工评估。权重与 `.pt` 产物均被 gitignore，测试自行生成确定性权重，不依赖本地产物。

### 工单 10：AVX2 增量累加器与后端切换（进行中）

累加器仍是 `int32[1024]` 的精确和，落子/撤子用 `_mm256_cvtepi8_epi32` + 一次向量加减完成差分；后段 L1 把 `W1` 每行补到 16 字节，用两组 8-lane 向量累加器算 `z[k]`，其余 `round_div` / ClippedReLU / 双头与标量共用。因此两后端切换不改变任何一个 bit，`nnue_backend_tests.py` 在三种输出路径上做逐位对拍。

```powershell
# scalar / AVX2 对拍 + 单次评估耗时（20,000 次基准）
.venv\Scripts\python.exe engine\tests\nnue_backend_tests.py build\engine\Release\nnue_probe.exe

# 三档后端：手工 / 标量 NNUE / AVX2 NNUE
.\build\engine\Release\gomoku_nnue.exe --depth 2
.\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue --nnue-backend scalar --depth 2
.\build\engine\Release\gomoku_nnue.exe --nnue weights\nnue_v1.nnue --depth 2
```

此前实测：标量 `3,117.55 ns/eval`，AVX2 `1,154.87 ns/eval`（**1.15 µs，2.70×**）；CTest 16/16 全绿。**但旧 M5 门控 100 局 `0W/0D/100L` 未达标**。已测样本中的算术对齐不等于训练到导出的保真度或全部接入行为已获验证；根因与下一步诊断见工单 12。

### 后续工单

- 工单 12：M5 诊断与正式门控准备，长任务交用户运行后分析结果；工单 10 保持进行中，正式 200 局门控通过后才完成 M5；工单 11 在此依赖解除后推进自举迭代与最终模型冻结。
