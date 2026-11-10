# 03: 接入开源 VCF

**What to build:** 用户对弈时实际使用支持禁手规则的开源 VCF；已证明的战术结果可以影响引擎选点，并能从客户端检查首着、变化或证据及搜索状态，同一能力可供后续数据筛选复用。

**Blocked by:** 02 — GUI 完成一局合法对弈。

**Status:** done

- [x] 选择实际可接入且支持 Renju 的开源实现，记录来源、固定版本、许可证、必要依赖和接入方式；不预设 Rapfi 提供独立 VCF 库。
- [x] 新方案的正式算杀路径复用上游实现，不继续使用自研 VCF 充当交付结果，也不另写算杀算法替代上游。
- [x] 适配接口区分已证明结果、预算内未找到证明、预算耗尽或中断以及输入错误；记录预算、首着和可复核的变化或证据。
- [x] 战术证据中的落点经过权威执行器合法性复查，并覆盖涉及黑方禁手的攻击与防守。
- [x] 只有已证明结果可覆盖正常价值判断；未找到 VCF 不被解释为没有其他强制胜法。
- [x] 在真实搜索与 GUI 中接通状态和结果；演示确定杀棋、合法防守、未知及停止场景。
- [x] 节点或时间限额可配置，实际达到限额时不返回伪造的确定结论；记录所选调用位置和开销。
- [x] 战术适配能力可由数据打标流程直接复用，不依赖 GUI 状态或教师评分字符串。

## Verification

在公开适配边界使用确定性战术题和真实上游实现验证结果、合法性、限额与中断。至少一条真实引擎对局路径证明开源实现已被调用。

### 验收记录

**上游选择**

- [quintet](https://github.com/renju-note/quintet)：Rust 的 Renju 杀棋求解器（VCF/VCT），MIT 许可，明确实现黑方禁手（含真假活三递归规则）与"禁手杀"防守。固定 commit `8219b71be2e12e77b735a4ef1462b3dd804f7b97`（2026-11-03）源码原样 vendored 到 `vendor/quintet/`，`vendor/quintet/UPSTREAM.md` 记录来源、date、许可证和"不要改 src/"。
- 上游依赖：`lru 0.18.5`、`wasm-bindgen 0.2.130`（后者是上游无条件依赖，native 构建不使用）。`engine/vcf-bridge/Cargo.lock` 已提交；首次构建需要 crates.io 网络（之后走 CARGO_HOME 缓存）。构建工具为 Rust 稳定 1.99.1（rustup，msvc toolchain）。
- 接入方式：`engine/vcf-bridge`（本仓库新增的小 crate，`crate-type = staticlib`）把引擎的 225 格棋盘翻译成上游 `Board::from_stones`，调用 `solve_with_stats(SolveMode::VCFDFS, …)`，再把状态、结束证据、完整变化、节点数译回 C ABI。CMake 用 `cargo build --release` 生成 `vcf_bridge.lib` 并链接到所有包含 `vcf.cpp` 的目标；`cargo test` 也注册进 ctest。
  - **环境变更（工单 07）**：Windows 构建已改为 `.venv` 内 Rust 工具链（`stable-x86_64-pc-windows-gnu`，链接器 venv `rust-lld`），产物名为 `libvcf_bridge.a`，与 Zig 构建的 C++（mingw-w64 ABI）一致；上游 vendored 源码、commit、C ABI 与全部 VCF 测试未变，并在工单 07 的全量回归中通过。
- 未采用 Rapfi：GPL-3.0 会传染整个仓库，且其 VCF 与自身搜索/评估耦合，没有独立库接口；旧 MIT 分支同样没有可分离的 VCF 模块。quintet 的 MIT + 独立 `solve` 边界与本工单要求最贴合。

**适配层与结果语义**（`engine/src/vcf.h` / `vcf.cpp`）

- `VcfStatus`：`proven` / `disproven` / `budget_exhausted` / `interrupted` / `unsupported_rule` / `invalid_input`。`disproven` = 给定 `maxAttackPlies` 内无 VCF；`budget_exhausted` 与 `interrupted` 明确是未知，不产生首着或变化。
- 每次查询回传 `VcfLimits`（节点预算、攻击手数、时间上限）、`nodes`、`elapsedMs`、首着、完整变化和结束证据（`fours` 的两个杀点或 `forbidden` 的禁手封点）。
- **证据复查**：`checkVcfEvidence` 在返回前按权威 `Board::isPlayable` 逐手重放变化；首着不匹配、越界、占用、终局后落子、黑方禁手任一出现即整条拒绝，结果降级为 `invalid_input` 并写明原因。搜索只在 `proven` 且首着合法时覆盖正常评估。
- 时间上限由 bridge 用线程 + 超时实现（上游无时钟，为 wasm 设计）：超时即放弃该查询并返回 `interrupted`，绝不返回未完成搜索的"结论"。
- 无禁手（freestyle）模式下上游不适用，适配层返回 `unsupported_rule`，搜索不使用 VCF（游戏规则本身仍由执行器判定）。

**搜索接入与实测选择**

- `SearchLimits.vcfPolicy`：`off` / `root` / `root_and_children`，`vcfNodeLimit`、`vcfChildNodeLimit`、`vcfMaxAttackPlies`、`vcfChildChecks` 可配置；引擎命令行 `--vcf-policy`、`--vcf-nodes`，执行器 `think` 接受 `vcf_policy`/`vcf_nodes`/`vcf_plies`。
- `root_and_children`（默认）：根节点查询一次；未证明时对排名前 16 的根着法各做一次对手 VCF 查询（预算 2000 节点），把"走完这步对手就有已证明 VCF"的着法在根节点剪掉；若全部着法都必败则不剪，保证仍返回合法着法。
- 实测（`tools/vcf_bench.py --engine build/engine/gomoku_nnue.exe --vcf-plies 12 --repeats 5`；`--json` 写出的报告属于本地生成产物，已在 `.gitignore` 中忽略，不随仓库提交）：

| 局面 | off | root | root_and_children |
|---|---|---|---|
| counter-four（已证明 3 攻击手） | 124 搜索节点 / 0.54ms | 5 VCF 节点 / 0.15ms，from_vcf | 同 root（根已证明，不再查子） |
| puzzle 3040（已证明 12 攻击手） | 220 节点 / 0.84ms，未用 VCF | 23 VCF 节点 / 0.20ms，from_vcf | 同 root |
| 安静中局（无 VCF） | 301 节点 / 1.45ms | 1 VCF 节点 / 1.45ms | +16 子查询 / 1.47ms，命中 0 |
| 防守方行棋 | 31 节点 / 0.12ms | VCF 平凡证明 0 节点 | +16 子查询、11 个必败着法 / 0.18ms |
| 根预算扫描 | — | 1000/5000/50000/200000 均用 23 节点证明 | — |

- 结论：根查询开销可以忽略（微秒级），`root_and_children` 的 16 次子查询约 +0.02–0.06ms，能在防守局面识别 11/16 个必败根着法；因此默认启用 `root_and_children`，并用节点预算（根 50000 / 子 2000 / 10 攻击手）作为硬上限，时间上限留给交互查询。实际棋力提升仍需工单 08/09 的测评验证，本工单只证明状态、合法性与开销。
- 采集侧：`binpack` 头只有 `use_vcf` 一个字节，记录不下调用点；datagen 生成与 verify 重放都固定用 `root`，保证旧数据与新数据可复现。交互对局使用 `root_and_children`，两者共用同一适配层。

**GUI 与复用**

- 状态区显示算杀引擎身份（`quintet@8219b71… (MIT)`）；每步 `think` 显示本轮算杀状态；新增「算杀检查（VCF）」按钮 + 节点预算输入，后台线程查询当前局面，显示状态、首着、变化、证据和实际开销；检查期间「停止对局」会终止引擎进程，面板显示"已停止（结果未知）"。
- `tools/executor.py` 新增 `vcf()` 与 `think()` 的算杀参数；`tools/vcf_tactics.py` 提供 `probe/describe/variation/filter_positions` 与 CLI（`check`、`filter`），只依赖执行器客户端，不读 GUI 状态或教师评分。`tools/vcf_bench.py` 记录调用点/预算实测。

**实际执行**

- Rust：`cargo test --release` → 6/6（编码方向、已证明变化、预算耗尽、超时中断、输入校验、空指针、版本串）。
- C++：`vcf_tests` 覆盖已证明（counter/puzzle/白方攻击）、禁手杀防守、深度限制、无解、预算耗尽、freestyle、输入错误、证据复查（占用/禁手/首着不符）、D4 八种变换；`search_tests` 覆盖根节点算杀 A/B、`root_and_children` 计数与"全必败仍返回合法着法"。
- 协议/黑盒：`executor_blackbox` 新增 29 项算杀检查（身份、proven 首着与变化、执行器重放复查、预算回显、预算耗尽为未知、无解、freestyle、非法参数、终局局面、`think` 的 from_vcf 与 policy=off），以及"停止时关闭客户端 → 在飞请求报错、进程消失"。
- GUI：`gui_vcf_smoke`（offscreen）11 项：引擎身份、初始态、真实局面查询后显示已证明/首着/变化/预算、回到 idle、停止后显示未知。
- 打标复用：`vcf_tactics_tests` 13 项（proven 变化与坐标、预算耗尽保持未知、`filter_positions` 逐条打标）。
- 构建：`cmake -S . -B build -G Ninja` + `ninja` 成功（含 cargo 子构建）；`ctest` **20/20 全绿**（原 17 项 + `vcf_bridge_tests`、`vcf_tactics_tests`、`gui_vcf_smoke`）。

**发现与限制**

- **跨实现禁手判定差异**：上游 145 手的长 VCF（"shadows and fog"）在本地复查时被 `renju::classifyForbidden` 判为三三禁手（上游允许），适配层因此把该证明降级为 `invalid_input`。合成局面里也见到同类情况。当前行为是保守的（拒绝证明而不是放行非法落点），但提示本地三三判定可能偏严：`renju.cpp` 的 `countThrees` 明确"递归在此切断"，与 RIF 的"延伸点本身若为禁手则该三为假三"不完全一致。修复属规则引擎工单范围，本工单不改规则实现，避免把接入任务扩成规则重构；已记录为后续核对项。
- 长变化（>64 手）此前会被截断，现提升到 512 点容量；引擎实际使用的 10 攻击手上限远低于此。
- 首次 cargo 构建需要网络取得 crates.io 依赖；离线机器可先 `cargo vendor` 或保留 CARGO_HOME 缓存。
- 未做人工长时间对弈棋力验证；算杀对实际胜率的影响留给后续测评工单。
