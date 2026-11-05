# GomokuNNUE — 五子棋（Renju）NNUE 引擎与训练流水线

一个"能玩、能训练、能自我对弈"的五子棋 AI 项目：纯 C++ 命令行引擎（Renju 禁手规则 + NNUE 评估 + Alpha-Beta 搜索），配 Python/PyTorch 训练端，在一台 Windows 机器（RTX 4080 + 多核 CPU）上完成从自我对弈数据生成到量化权重加载的完整闭环。

**设计契约的权威版本见 `.scratch/gomoku-nnue/spec.md`，工单见 `.scratch/gomoku-nnue/issues/`。**

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
.\build\engine\Release\gomoku_nnue.exe --help

# 自我对弈 datagen worker（工单 07，单进程单分片；由 tools/datagen.py 拉起多个）
.\build\engine\Release\datagen.exe --help
```

时间控制：`INFO timeout_turn`（留 20% 余量）、`INFO time_left`（按剩余时间 1/20 封顶）、`INFO timeout_match` 均被接受。

## 测试

```powershell
# 创建项目独立 Python 环境（一次性；项目 Python 一律走 .venv）
py -3.12 -m venv .venv

ctest --test-dir build -C Release --output-on-failure
```

包含 6 个 C++ 单元测试（`board_tests` / `renju_tests` / `eval_tests` / `search_tests` / `vcf_tests` / `binpack_tests`）与 3 个 Python/黑盒测试（`protocol_blackbox` / `match_runner_tests` / `datagen_tests`），共 9 个 CTest；Python 测试都经项目 `.venv` 运行。也可手动执行：

```powershell
.venv\Scripts\python.exe engine\tests\protocol_blackbox.py build\engine\Release\gomoku_nnue.exe
.venv\Scripts\python.exe engine\tests\match_runner_tests.py build\engine\Release\gomoku_nnue.exe tools\match_runner.py
.venv\Scripts\python.exe engine\tests\datagen_tests.py build\engine\Release\datagen.exe trainer\binpack.py tools\datagen.py
```

黑盒内含双引擎自对弈、规则切换、时间控制、VCF 杀局与搜索引擎对随机基线 6 局全胜的强度验收；`match_runner_tests` 覆盖 100% 胜率门控、结果落盘、freestyle 规则切换与超时/崩溃/非法着法三类异常计负；`datagen_tests` 用真实 worker 落盘，再校验魔数/64 字节布局/WDL 回填/对局边界/分片合并/多进程隔离，并让 C++ 复盘抽样搜索分、检出被篡改的分数。

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

**当前进度**：工单 01–07 完成——M3 达成（Alpha-Beta + 手工评估 + VCF 连续冲四算杀），补齐强度度量 `match_runner`，并跑通 M4 前半的自我对弈 datagen「生成→分片落盘→合并→校验→复盘」闭环。NNUE 训练流水线在工单 08 起落地。

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
.venv\Scripts\python.exe tools\match_runner.py `
    --engine-a ".\build\engine\Release\nnue_v1.exe --depth 4" `
    --engine-b ".\build\engine\Release\gomoku_nnue.exe --depth 4" `
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

### 其余工具（待落地）

- `trainer/`：模型、DataLoader（在 `binpack.py` 之上加 D4 增广）、训练、量化导出脚本（工单 08，`venv + requirements.txt`）
