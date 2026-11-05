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
│   └── src/              #   棋盘/规则/搜索/协议/NNUE 推理（按里程碑逐步填充）
├── trainer/              # Python 训练端（模型/数据集/训练/量化导出，M4 落地）
├── tools/                # 配套工具：match_runner（已完成）、datagen（M4）
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
```

时间控制：`INFO timeout_turn`（留 20% 余量）、`INFO time_left`（按剩余时间 1/20 封顶）、`INFO timeout_match` 均被接受。

## 测试

```powershell
# 创建项目独立 Python 环境（一次性；项目 Python 一律走 .venv）
py -3.12 -m venv .venv

ctest --test-dir build -C Release --output-on-failure
```

包含 5 个 C++ 单元测试（`board_tests` / `renju_tests` / `eval_tests` / `search_tests` / `vcf_tests`）、协议黑盒测试（`protocol_blackbox`）与 `match_runner` 自检（`match_runner_tests`），共 7 个 CTest；两个 Python 测试都经项目 `.venv` 运行。也可手动执行：

```powershell
.venv\Scripts\python.exe engine\tests\protocol_blackbox.py build\engine\Release\gomoku_nnue.exe
.venv\Scripts\python.exe engine\tests\match_runner_tests.py build\engine\Release\gomoku_nnue.exe tools\match_runner.py
```

黑盒内含双引擎自对弈、规则切换、时间控制、VCF 杀局与搜索引擎对随机基线 6 局全胜的强度验收；`match_runner_tests` 覆盖 100% 胜率门控、结果落盘、freestyle 规则切换与超时/崩溃/非法着法三类异常计负。

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

**当前进度**：工单 01–06 完成——M3 达成（Alpha-Beta + 手工评估 + VCF 连续冲四算杀），并补齐强度度量基础设施 `match_runner`。NNUE 与训练流水线在 M4 起落地。

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

### 其余工具（待落地）

- `tools/` datagen：多进程自我对弈批量生成训练数据（工单 07）
- `trainer/`：数据加载、训练、量化导出脚本（M4，`venv + requirements.txt`）
