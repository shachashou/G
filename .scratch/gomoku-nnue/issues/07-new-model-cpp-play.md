# 07: 使用新模型实际对弈

**What to build:** 用户在客户端选择新导出的模型后，C++ 引擎真实使用该模型和 Alpha-Beta 搜索完成合法对局。先建立全量重建的正确性参照，为自主对弈和后续增量加速提供可用部署路径。

**Blocked by:** 06 — 新网络从零训练与导出。

**Status:** done

- [x] C++ 加载新架构和格式，检查完整性、规则、特征及价值契约；损坏或不兼容模型明确失败。
- [x] 实现新网络标量前向和全量第一层重建，作为后续增量与 SIMD 的长期数值参照。
- [x] 对同一量化模型与 Python 整数参考完成对拍，训练原模型与 C++ 部署结果按冻结容差检查导出保真度。
- [x] 价值按约定转换到当前行棋方搜索视角；终局和已证明战术结果保持独立语义，不能被普通网络分数覆盖。
- [x] 新模型接入实际 Alpha-Beta 选点，终局、合法落子、开源 VCF 及时间控制共同生效。
- [x] GUI 显示实际加载的模型身份及评估方式；新候选可试玩，但不会因为加载成功自动晋级。
- [x] 没有模型或加载失败时，不静默切换手工评估并继续把结果标为 NNUE。
- [x] 新模型在客户端及无界面进程路径均能完成真实对局；从着法记录重放获得相同终局结果。
- [x] 当前交付不依赖增量优化工单，允许使用全量参照推理先跑通自主对弈的必要基础。

## Verification

通过既有推理对拍边界与真实引擎协议验证模型身份、正负视角、格式拒绝、时间控制和完整对局；GUI 验证模型选择到落子的可见路径。

### 验收记录

**构建环境：完全收敛到 `.venv`（01–06 回归前提）**

- C/C++ 编译器改用 venv 内 pip 安装的 `ziglang==0.17.1`（clang 22 + libc++ + lld + mingw-w64），通过 `tools/build/zig-cc.cmd` / `zig-cxx.cmd` 调用；构建不读取 Visual Studio `cl.exe`、Windows SDK 或系统 cmake。
- Rust 工具链一次性安装到 `.venv/rustup` 与 `.venv/cargo`（`stable-x86_64-pc-windows-gnu`，含 `rust-mingw` 导入库）；proc-macro 主机编译也用 GNU 主机工具链，链接器为 venv 内 `rust-lld`。全局 `~/.cargo` 不参与构建。
- Zig 缓存固定在 `.venv/zig-cache`；CMake 用 `cmake -E env` 显式传入 Rust/Zig 变量。Ninja 的链接池设为 1，避免 Zig/lld 并行链接耗尽内存。
- 入口：`tools/build/bootstrap_toolchain.py`（一次性 Rust 安装）、`tools/build/build.py`（配置/构建/测试）。README、`requirements.txt`、`engine/vcf-bridge/README.md` 已同步。

**加载与格式拒绝（`engine/src/value_net.{h,cpp}`）**

- `GMVALUE1`：64 字节头 + JSON 元数据 + int16/int32 权重体，FNV-1a 覆盖元数据+权重体。加载时校验 magic/版本、`450-1024-64-32-1`、Renju、`absolute-color-450`、`black-absolute-tanh-v1`、`ea∈[1,16]`、激活上限、文件长度、元数据与头一致。
- 拒绝路径：坏 magic、截断头/体、校验和不符、旧 `GMNNUE01`、非 Renju、未知特征/价值契约、旧架构元数据、头/元数据 `ea` 或激活上限不一致、维度被篡改。
- 显式指定 `--value-model` 时加载失败 → stderr 说明原因、进程退出码 2，stdout 不产生任何评估响应；不再有“回退手工评估但继续标 NNUE”的路径。`--value-model` 与旧 `--nnue` 互斥。

**参考推理（全量，不做增量）**

- 稀疏标量前向与全量第一层重建均使用 int64 累加（bias 为 int32，最多 225 行 int16，窄位宽无法在撤销时精确还原）；每层一次 `round_div`，最终 `z` 为黑方绝对定点值，`value = tanh(z/2^ea)`。
- 搜索分数为 `stm_score = ±round(z×1000/2^ea)`，限制在 ±100000，远低于 mate 分（±1e6 带宽）。
- 40 子局面全量重建+前向实测约 **13.9 µs/次**（`value_probe bench`）；本工单明确不把该数字当增量加速或棋力结论，增量与 AVX2 留给工单 11。

**C++ ↔ Python 对拍与导出保真度（`value_parity_tests`，实测）**

- 合成 GMVALUE1 与本地 `weights/value-demo/model.gmvalue` 各 27 个局面（合成 + 样本池抽样）：C++ `features` 与 `trainer/features.py` 一致；`infer`（稀疏）、`board`（全量重建）、`acc`（公共累加器重建）与 `trainer/value_format.infer_int` **逐位一致**；`score` 的正负视角换算符合约定。
- 导出保真度：用重新计算的 `model_id` 确认 `last.pt` 就是导出模型，27 个局面上训练原模型（float）与 C++ 部署值（`tanh(z/2^9)`）的差异为 **mean 0.001292、p95 0.003651、max 0.007584**，全部低于冻结容差（0.010 / 0.030 / 0.100）。报告路径由 `fidelity.json` 独立记录。

**价值视角、终局与 VCF（`value_net_tests` + `value_engine_blackbox`）**

- 常数模型 `b3=+500`（ea=9）：黑方行棋 `think` 分数为 **+977**，白方行棋为 **-977**；`b3=-500` 时黑方为 **-977**。视角转换在真实执行器协议上验证。
- 黑方四连、黑先，负价值模型：VCF 关闭时 Alpha-Beta 仍返回 mate 分 **999999** 并完成五连获胜，网络分数不能覆盖终局。
- 已证明 VCF 局面（首着 `(8,7)`）配负价值模型：`from_vcf=true`、首着正确、分数 999999（`kMate-1`），普通网络分数不能覆盖已证明战术结果。

**真实对局与重放**

- 执行器 + 本地 demo 模型、每步 60 ms、`vcf_policy=root`：完成 **21 手黑胜**真实对局，全部着法属于完整合法落点集合、每步均被权威执行器接受、超时余量正常；用同一执行器 `replay` 该着法记录，得到相同终局与相同棋盘。
- Gomocup 协议无界面自对弈：`--value-model ... --depth 1 --vcf-policy off` 两进程完成 16 手对局；`ABOUT` 返回 `eval="nnue"`、`eval_engine="gmvalue1-scalar-full"`、模型 id 与 `value_model_promoted="false"`。

**GUI（`gui_model_smoke`，offscreen）**

- 「模型与评估方式」面板列出/浏览 `weights/` 下的 `*.gmvalue`；启动 `--model` 与面板「应用模型」均显示实际加载的模型 id、`450-1024-64-32-1`、规则/特征/价值契约与「候选试玩（未晋级正式版本）」。
- 选中新模型后引擎真实落子；切回手工评估显示「手工威胁评估」；再次加载模型恢复身份。
- 损坏模型被拒绝，窗口明确显示「模型加载失败」，当前有效模型继续保留且不会被误标；以损坏模型启动时诚实显示手工回退，但绝不显示 GMVALUE1。
- 加载成功不会写入任何晋级状态；工单 08 之前不存在自动晋级路径。

**测试**

- 新增 C++ `value_net_tests`；新增 Python `value_parity_tests`、`value_engine_blackbox`、`gui_model_smoke`。
- 全量 `ctest`：**31/31 通过**（原 27 项 + 4 项新测试），覆盖工单 01–07；失败的只有预期内的外部依赖跳过路径（Rapfi 已安装时教师测试实际执行）。

**限制与后续**

- 搜索每次叶节点评估都做全量第一层重建，是刻意保留的数值参照；第一层增量累加与 AVX2 加速属于工单 11，本工单不声称搜索吞吐或棋力提升。
- 本地 demo 模型仅由 10 条教师样本训练，只用于证明部署链路与数值一致性，不构成棋力结论；模型仍为候选，未晋级正式版本。
- 旧 `GMNNUE` 仅在显式 `--nnue` 时保留，供旧对拍测试使用；新流程以 `--value-model` 为唯一部署入口。
