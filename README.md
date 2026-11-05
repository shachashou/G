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
├── tools/                # 配套工具：datagen、match_runner（M4+ 逐步落地）
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

# 协议模式（默认）：从标准输入读取 Gomocup 协议命令
# （START / TURN / BOARD / INFO / ABOUT / END），挂现成 GUI 对弈。
# 当前棋力：随机合法走子（02 号工单交付；搜索随 M2 落地）
```

## 测试

```powershell
# 创建项目独立 Python 环境（一次性；项目 Python 一律走 .venv）
py -3.12 -m venv .venv

ctest --test-dir build -C Release --output-on-failure
```

包含棋盘单元测试（`board_tests`）与协议黑盒测试（`protocol_blackbox`，经项目 `.venv` 运行；也可手动执行：
`.venv\Scripts\python.exe engine\tests\protocol_blackbox.py build\engine\Release\gomoku_nnue.exe`）。

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

## 工具用法（占位，随里程碑填充）

- `tools/` datagen：多进程自我对弈批量生成训练数据（M4）
- `tools/` match_runner：两个引擎 build 批量对打、胜率门控（M4 末）
- `trainer/`：数据加载、训练、量化导出脚本（M4，`venv + requirements.txt`）
