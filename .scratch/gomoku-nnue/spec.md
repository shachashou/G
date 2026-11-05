# Spec: 五子棋（Renju）NNUE 引擎与训练流水线

Status: ready-for-agent

## Problem Statement

我想在自己家里那台 Windows 机器（RTX 4080 + 多核 CPU）上，从零做一个"能玩、能训练、能自我对弈"的五子棋 AI：

- 想通过现成的五子棋 GUI 跟它下棋，而不是自己写界面；
- 想要一个认真强度的 AI（明显强于网上随手可得的五子棋程序），而不是玩具；
- 想体验 Stockfish 式的 NNUE 全流程：自我对弈生成数据 → GPU 训练 → 量化导出 → C++ 引擎加载推理 → 新旧版本对打验证提升。

## Solution

一个纯 C++20 命令行引擎（MSVC 编译，AVX2，无任何图形，仅通过 stdin/stdout 的 Gomocup 协议与外部 GUI 通信）+ 一个 Python/PyTorch 训练端。引擎分六个里程碑演进：

1. **M1** 棋盘 + Renju 规则（禁手判定）+ 合法着法生成 + 单元测试；
2. **M2** Alpha-Beta 搜索（PVS/置换表/杀手启发/历史启发）+ 手工评估函数 + Gomocup 协议，exe 可挂 GUI 完整对局；
3. **M3** VCF 算杀（连续冲四必胜判定）接入搜索；
4. **M4** 自我对弈数据生成（目标首轮 ~1000 万局面，半天内）+ Python 训练流水线，训出第一个 NNUE 权重；
5. **M5** C++ 端 NNUE 推理（先标量版验证与 PyTorch 数值一致，再 AVX2 向量化增量累加器）替换手工评估；验收：相同搜索配置下对 M3 引擎胜率 >65%，以隔离验证 NNUE 相对手工评估的净提升；
6. **M6** 迭代循环：用 NNUE 引擎自举（bootstrapping）生成更高质量数据 → 训练 → match_runner 胜率门控，Elo 稳定爬升。

## User Stories

1. As a 玩家, I want 把引擎 exe 挂到现成五子棋 GUI 上（stdin/stdout）, so that 我不用写界面就能和 AI 对弈。
2. As a 玩家, I want 引擎默认执行 Renju 禁手规则（黑方三三/四四/长连为禁手）, so that 对局符合连珠规则。
3. As a 玩家, I want 通过 `INFO rule 1` 切换到无禁手 freestyle 模式, so that 兼容那些默认无禁手的 Gomocup GUI。
4. As a 玩家, I want 引擎遵守 GUI 下发的时间限制（`INFO timeout_turn` / `timeout_match`）, so that 对局节奏正常。
5. As a 玩家, I want 黑方一子同时形成五连和禁手时判黑胜（五连优先）, so that 符合正式连珠规则。
6. As a 玩家, I want 白方无任何禁手且长连也算胜, so that 符合正式连珠规则。
7. As a 开发者, I want 用公开 Renju 禁手题库对禁手判定做单元测试, so that 规则正确性有客观验收标准。
8. As a 开发者, I want 候选着法只生成已有棋子邻域内的空点, so that 搜索不被无关空点淹没。
9. As a 开发者, I want Zobrist 哈希 + 64MB 置换表 + PVS + 杀手/历史启发, so that 搜索效率达到认真强度所需。
10. As a 开发者, I want VCF（连续冲四）算杀接入搜索, so that 必胜杀法不因深度不够而漏算。
11. As a 开发者, I want 手工评估用棋形打分（活四/冲四/活三等）并支持增量更新, so that M2 就有像样的棋力且 datagen 速度可接受。
12. As a 开发者, I want 引擎先做单线程, so that 避免多线程置换表一致性问题拖慢 MVP。
13. As a 训练者, I want 自我对弈工具多进程批量生成数据（16 进程起步）, so that 半天内拿到首轮 ~1000 万局面。
14. As a 训练者, I want 开局前 3 手在中心 7×7 区域内带温度随机化, so that 数据分布多样、不过拟合单一开局。
15. As a 训练者, I want 每个局面记录 depth-6 搜索分 + 最终对局结果, so that 训练标签混合搜索分与 WDL。
16. As a 训练者, I want 每条记录定长 64 字节（黑白各 225 bit 占位 + int16 分数 + 结果 + 行棋方 + ply）, so that mmap 高效加载且格式简单。
17. As a 训练者, I want DataLoader 内实时做 D4 八重对称增广, so that 等效数据量 ×8 且磁盘不膨胀。
18. As a 训练者, I want 验证集按完整对局（而非局面）划分, so that 增广镜像不会泄漏进验证集。
19. As a 训练者, I want 第二轮用已训练的 NNUE 引擎重新自我对弈, so that 数据质量随模型强度自举提升。
20. As a 开发者, I want PyTorch 模型结构为 450 → Accumulator 1024（int16）→ ClippedReLU → L1 15 → ClippedReLU → 输出 1（int8 权重）, so that 推理微秒级、参数量适中不易过拟合。
21. As a 开发者, I want 特征为颜色绝对的 450 维（黑 225 + 白 225，每子 one-hot）+ 单累加器 + 双输出头（轮到黑/轮到白各一头）, so that 增量更新逻辑最简且天然保留 Renju 黑白规则不对称性。
22. As a 开发者, I want Python `features` 与 C++ 特征提取共享同一份权威索引规范, so that 两端索引永不漂移。
23. As a 开发者, I want 先实现标量版 C++ NNUE 推理并与 PyTorch 输出做数值对齐验证, so that 逻辑错误不与 SIMD 指令 bug 耦合。
24. As a 开发者, I want 之后再上 AVX2 增量累加器, so that NNUE 评估不成为搜索瓶颈。
25. As a 开发者, I want 权重导出文件带魔数与版本号, so that C++ 端加载时能校验格式。
26. As a 开发者, I want match_runner 让新旧版本引擎批量对打, so that 每轮迭代有胜率门控（M5 对 M3 需 >65%）。
27. As a 开发者, I want 训练损失为 `MSE(tanh(score/400)) + CE(WDL)` 混合, so that 兼顾搜索分拟合与终局结果（Stockfish 风格）。
28. As a 开发者, I want README 记录编译指南（CMake + MSVC）与各工具用法, so that 半年后自己还能跑起来。

## Implementation Decisions

- **规则集**：Renju 禁手规则，自由开局（不做塔拉山口/索索夫等开局交换规则）。黑方禁手点直接从 movegen 排除（引擎语义：黑永远不主动下禁手，白方利用禁手做杀由搜索自然涌现）；五连优先于禁手；白无禁手、长连算胜；黑长连为禁手。`INFO rule` 开关：1 = freestyle（禁手检查关闭），2 = Renju（默认），两种模式共享同一套 board/movegen 代码。
- **引擎**：C++20，MSVC，`/arch:AVX2`，不做运行时 SIMD 分发；单线程；置换表 64MB；Zobrist 哈希；Alpha-Beta（PVS + 置换表 + 杀手启发 + 历史启发）；候选点剪枝（只搜已有棋子邻域空点）；手工评估为棋形打分表 + 增量更新。
- **协议**：Gomocup/piskvork 协议子集（`START` / `TURN` / `BOARD` / `INFO` / `ABOUT` / `END`），纯 stdin/stdout，无任何图形。
- **NNUE 架构**：输入 450 维颜色绝对稀疏特征（黑子 225 + 白子 225）→ 单 Accumulator 1024（int16，激活前）→ ClippedReLU → L1 15 → ClippedReLU → 输出 1；权重 int8 量化；双输出头按行棋方选择。不做视角翻转（无国王锚点，颜色绝对特征天然保留不对称）。
- **数据格式**：定长 64 字节/局面：黑棋盘 225 bit + 白棋盘 225 bit + int16 搜索分 + 2 bit 对局结果 + 行棋方 + ply；按对局顺序落盘，mmap 读取；验证集按对局划分。
- **Datagen**：16 进程并行（并行度按实测 CPU 调整，若核数不足降 depth）；开局前 3 手中心 7×7 带温度随机；此后每步 depth-6 固定深度搜索；每个非随机开局局面记录（搜索分，最终结果）；满盘为和棋；首轮目标 ~1000 万局面（约 640MB）。
- **训练**：PyTorch + CUDA（RTX 4080）；损失 `MSE(tanh(score/400)) + CE(WDL)`；D4 八重对称实时增广（Renju 禁手规则在 D4 下不变，标签无需变换）；`venv + requirements.txt`。
- **量化导出**：PyTorch 权重 → int8/int16 二进制文件，带魔数与版本号；C++ 端加载时校验。
- **NNUE 落地顺序**：标量参考实现先行，与 PyTorch 在采样局面上做数值对齐（容差内）；验证通过后再写 AVX2 向量化增量累加器。
- **验收门控**：match_runner 批量对打；M5 验收 = 相同搜索配置下 NNUE 引擎对 M3（手工评估 + VCF）胜率 >65%。
- **目录结构**：`engine/`（C++ 核心：board、movegen、search、protocol、nnue/ 子模块）、`trainer/`（Python：model、dataset、train、export、features）、`tools/`（datagen、match_runner）、`data/`（raw/binpack，gitignore 排除）、`weights/`（checkpoints + 量化权重）。原 README 中的 `play_gui.py`（pygame GUI）与 `datasets_download.sh` 移除，与"无图形、纯协议"的决定矛盾。
- **里程碑 M1–M6** 按上文 Solution 执行，VCT 为 M6 之后的可选进阶项。

## Testing Decisions

- **最高接缝：引擎的 stdin/stdout 协议**。外部黑盒测试直接向 exe 进程写入 Gomocup 命令、断言其输出（着法格式、时间行为、协议错误处理）。M2 起所有功能验收都尽量在这个接缝上做。
- **M1 规则正确性**：在棋盘 API（落子/撤子/胜负判定/禁手查询）接缝上做 C++ 单元测试；素材用公开 Renju 禁手题库（三三/四四/长连、嵌套跳三等经典案例）+ 自构造的边界案例（五连与禁手同现、D4 对称局面禁手不变性）。
- **跨语言契约接缝**（仅两处，尽量少）：
  1. 特征索引：C++ 特征提取与 Python 端在采样局面上必须产出完全一致的稀疏索引集合；
  2. 权重文件：标量版 C++ NNUE 推理与 PyTorch 浮点推理在量化容差内逐局面一致。
- **强度验收**：match_runner 批量对打（M5：>65% 胜率门槛，相同搜索配置）。
- 好的测试只断言外部可观察行为（着法、输出、胜率、数值），不断言内部实现细节。

## Out of Scope

- VCT（活三威胁算杀）—— M6 之后的可选进阶项，不进承诺范围。
- 开局交换规则（塔拉山口、索索夫-8 等）。
- 开局库、Gomocup 竞赛级时间管理、多线程搜索、联网对战、Web 界面。
- 自研 GUI（已明确不做，挂现成 GUI）。
- SSE/NEON 等运行时分发回退（只在 4080 那台机器上跑）。
- 15×15 以外的棋盘尺寸。
- AlphaZero/MCTS 路线（已明确选 Stockfish 路线）。

## Further Notes

- 用户未报 CPU 型号/核数（占位符未填）。datagen 并行度需在 M4 时实测校准：16 线程以上维持 depth 6；若只有 8 线程，降 depth 5 或接受更长时间。跑通"生成→落盘→训练→加载→对弈"最小闭环优先于数据量达标。
- 根目录 `README.md` 仍是最初的草案（含已废弃的 `play_gui.py`、`datasets_download.sh`），实现启动时需同步更新。
- `CMakeLists.txt` 目前为空文件，M1 一并初始化。
