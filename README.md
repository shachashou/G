# GomokuNNUE

Windows 本地五子棋训练与对弈工作台：在同一个客户端完成"玩一把、收集数据、从0训练、自我训练、历史版本测评"。

**项目正在按新方案重设计。** 当前已交付执行器、搜索、训练与推理基础、可完成一局合法对弈的桌面 GUI，以及接入开源实现的 VCF 算杀；Rapfi 教师接入、新网络及持续自主训练仍待实现。

- [主规格](.scratch/gomoku-nnue/spec.md)：目标行为、交付顺序、测试与范围，状态 ready-for-agent。
- [术语表](GLOSSARY.md)：教师启动、自主训练、正式版本、候选模型与晋级等定义。
- [训练端说明](trainer/README.md) / [工具说明](tools/README.md) / [客户端说明](gui/README.md)：现有能力及改造边界。
- [实施工单](.scratch/gomoku-nnue/issues/)：已发布12张，每张独立记录验收条件和阻塞关系。

已完成 [01：训练基础脱离旧实验](.scratch/gomoku-nnue/issues/01-retire-legacy-experiments.md)、[02：GUI 完成一局合法对弈](.scratch/gomoku-nnue/issues/02-desktop-play-and-shared-rules.md)、[03：接入开源 VCF](.scratch/gomoku-nnue/issues/03-open-source-vcf-integration.md) 和 [04：教师评分与有效样本落盘](.scratch/gomoku-nnue/issues/04-rapfi-labeling-and-sample-storage.md)。其余工单按各自的 Blocked by 推进；ready-for-agent 不表示依赖已经完成。

## 设计方向

- **规则**：15×15、有禁手、黑先白后、自由开局；黑棋恰好五连获胜，白棋五连及以上获胜。所有模块共用权威规则。
- **执行器**：C++ 棋盘、禁手、判赢、完整合法落子、落子与撤销。
- **算杀**：复用支持 Renju 的开源 VCF。上游为 [quintet](https://github.com/renju-note/quintet)（Rust，MIT），固定在 `vendor/quintet/UPSTREAM.md` 记录的 commit；`engine/vcf-bridge` 提供 C ABI，`engine/src/vcf.cpp` 是适配层。自研 VCF 不再作为交付结果。
- **教师启动**：Rapfi 为经过战术筛选的局面打标，帮助自己的模型建立初始判断能力。
- **网络**：PyTorch，450 → 1024 → 64 → 32 → 1，MSE 或 SmoothL1，D4 八种空间对称增广。
- **部署**：C++ 增量累加器结合 Alpha-Beta；第一层落子加向量、撤销减向量，后续层正常前向。
- **自主训练**：正式版本自我对弈，以实际胜负训练候选模型，最终能够完全不调用教师。
- **测评**：候选模型对战冻结的历史版本；相同思考资源、配对开局、交换黑白，通过预定门槛才晋级。
- **客户端**：本机桌面 GUI，管理对局、任务、训练状态、模型和测评。

与 Rapfi 对战不是晋级要求。训练误差、推理一致性和实战棋力分别验证。

## 训练怎样持续运行

教师启动：

少量合法局面 → 战术筛选 → Rapfi 打标 → 样本池 → 初始模型 v0。

持续自主训练：

正式版本自我对弈 → 完整结果回填 → 样本池 → 候选训练 → 导出检查 → 挑战历史版本 → 晋级或保留正式版本。

采集、打标与训练可以并行；采集使用固定快照，候选模型独立更新。样本池有容量上限并混合不同对局的新旧样本。

不要求先积累500万～2000万条才训练；累计生成量、当前池容量和训练曝光次数分别统计。教师退出依据无教师循环能否运行并相对历史版本进步，不按固定数据量"毕业"。

## 环境与构建

所有 Python 侧工具都在 venv 里，**不使用系统安装的 cmake**。`cmake` / `ctest` / `ninja` 是跨平台的 PyPI 轮子，GUI 依赖 PySide6 也在其中：

~~~powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
~~~

此外还需要 **Rust 稳定工具链**（`cargo`）：VCF 算杀复用 vendored 的开源 quintet，由 CMake 自动执行 `cargo build --release` 生成 `engine/vcf-bridge/target/release/vcf_bridge.lib` 再链接。未安装时 cmake 会直接报错并提示：

~~~powershell
winget install --id Rustlang.Rustup -e   # 或 https://rustup.rs
~~~

Windows 上的 C++ 编译器 `cl.exe` 仍来自 Visual Studio。用其 DevShell PowerShell 模块激活编译器环境后，用 venv 的 cmake + Ninja 构建：

~~~powershell
Import-Module "D:\VisualStudio\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "D:\VisualStudio" -DevCmdArguments '-arch=x64 -host_arch=x64' -SkipAutomaticLocation
.venv\Scripts\cmake.exe -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
.venv\Scripts\cmake.exe --build build
.venv\Scripts\ctest.exe --test-dir build --output-on-failure
~~~

`CMakeLists.txt` 不再强制 MSVC；Linux / macOS 上用系统 C++ 编译器直接执行上面三条 cmake / ctest 命令即可。改了 CMakeLists.txt 后先重新执行 `cmake -S . -B build -G Ninja`。

技能目录 `.agents/skills/` 由根目录 `skills-lock.json` 记录的来源安装，属于安装物而非项目源码，因此不纳入版本控制；新克隆后按该锁文件重装技能即可，版本仍由锁文件固定。

## 启动对弈

~~~powershell
.\gui\play.bat
~~~

或 `.venv\Scripts\python.exe gui\gomoku_gui.py --help`。客户端通过 JSON-lines 协议调用权威执行器：完整合法落点、禁手原因、落子结果和终局都由 C++ 执行器给出，界面与 Python 裁判不另写规则。详见 [客户端说明](gui/README.md)。

## 算杀（VCF）

- **上游**：[quintet](https://github.com/renju-note/quintet)（Rust，MIT，Renju 规则），固定 commit 见 [vendor/quintet/UPSTREAM.md](vendor/quintet/UPSTREAM.md)；源码原样 vendored，适配逻辑只在 `engine/vcf-bridge`（C ABI）与 `engine/src/vcf.cpp`。
- **结果区分**：`proven`（已证明，带首着、完整变化和结束证据）、`disproven`（给定预算内没有 VCF，不代表没有其它强制胜法）、`budget_exhausted`、`interrupted`（两者都是未知，不返回确定结论）、`unsupported_rule`（无禁手规则暂不支持）、`invalid_input`。只有 `proven` 且首着合法时才覆盖正常评估。
- **证据复查**：变化在返回前按权威棋盘逐手重放；任一落点非法（占用、越界、黑方禁手）即拒绝整条证明。上游与本地禁手判定不一致时宁可报告未知，也不使用该证明。
- **搜索调用点**：默认 `root_and_children`——根节点算杀，并对排名靠前的根着法检查对手是否存在已证明 VCF 从而剪掉必败着法。预算按节点计（根 50000、每个子查询 2000、10 个攻击手），可用 `--vcf-policy off|root|root_and_children`、`--vcf-nodes N` 调整；实测开销见[工单 03 验收记录](.scratch/gomoku-nnue/issues/03-open-source-vcf-integration.md)。
- **GUI**：状态区显示算杀引擎身份，`think` 结果会显示本轮算杀状态；「算杀检查（VCF）」按钮可在当前局面单独查询并显示状态、首着、变化、预算与实际开销，检查期间「停止对局」可中断。
- **复用**：`tools/vcf_tactics.py` 提供不依赖 GUI 的查询/打标接口（供工单 05 过滤强制战术局面），`tools/vcf_bench.py` 记录调用点与预算实测。

## 教师打标（Rapfi）

- **本地安装**：Rapfi 250615（`Rapfi-engine.7z`）解压到忽略目录 `tools/rapfi/`（Windows 自带 `tar -xf Rapfi-engine.7z -C tools/rapfi` 即可解 7z）。可执行文件、Renju 权重与 `config.toml` 的 SHA-256 和版本号固定在 `tools/rapfi.lock.json`（入库）；每次运行前逐项核对，缺失或哈希不符立即报错，不产生任何标签。
- **评分方法**：通过 Gomocup 协议 `BOARD ... DONE` 让教师以固定预算（默认 `INFO timeout_turn 800`、`thread_num 1`、Renju `INFO rule 2`）搜索该局面，从 `MESSAGE ... Eval ...` 读取**搜索评分**；引擎建议着法单独记录，不与评分混为同一契约。已实测：评分视角为**行棋方**（白方必胜局面给正分），mate 记作 `±M<plies>`，以落子回复作为完成同步。
- **价值契约（teacher-v1）**：原始分（行棋方视角）→ `value = tanh(score / 600)`，范围 [-1,1]。mate、无搜索评分、超时、异常响应不归一化、不计入教师训练队列；VCF 已证明的局面在调用教师前排除，VCF 预算耗尽只记 `vcf_unknown`，不宣称不存在其它强制胜法。
- **样本存储**：`data/teacher/samples.sqlite3`（SQLite，`application_id` + `user_version=1`）。每行保存棋盘、规则、行棋方、来源（对局/独立采样）、来源 id、ply/末着、采集模型 id、战术状态、过滤原因、教师状态、原始评分/视角/归一化参数/价值、建议着法、搜索深度、预算、线程数、耗时、教师版本与身份哈希、标签版本、可空最终结果。被过滤的请求也保留用于计数，训练只读 `accepted` 行。
- **完整性**：打开时校验 application id、schema 版本与列；`validate()` 跑 `PRAGMA integrity_check` 并检查行棋方与棋子数一致、accepted 必须有 value、独立采样不得伪造和棋。截断文件、更新版本的库、非 Renju 规则都会明确拒绝。
- **GUI**：状态区显示教师身份与安装状态；「打标并保存样本」显示原始评分、归一化价值、过滤原因、教师状态、预算与有效样本数；教师缺失时明确报错且不落盘。
- **命令**：

~~~powershell
.venv\Scripts\python.exe tools\rapfi_teacher.py about
.venv\Scripts\python.exe tools\teacher_label.py label "H8,H9,I8,I9"
.venv\Scripts\python.exe tools\teacher_label.py stats
.venv\Scripts\python.exe tools\teacher_label.py validate
~~~

## 当前代码与目标的差距

| 区域 | 当前情况 | 改造目标 |
|---|---|---|
| 执行器 | 已有 C++ 权威执行器、Renju、协议和 Alpha-Beta | 核验规则并统一裁判、采集和对弈 |
| VCF | 已接入开源 quintet（MIT，固定 commit），适配层区分已证明/未证明/预算耗尽/中断，证据经执行器复查 | 保持上游版本，供工单 05 的战术筛选与打标复用 |
| 标签 | 已接入 Rapfi 教师（固定版本）与版本化样本存储，区分评分/建议着法/胜负标签 | 持续采集与样本池（工单 05）、结果标签 |
| 网络 | 旧15维隐藏层及双输出头 | 新64、32隐藏层与单输出 |
| 推理 | 已有标量、AVX2、增量和对拍基础 | 新架构导出、加载与增量推理 |
| 数据与编排 | 批处理工具和旧自举流程 | 有容量上限的持续样本池和任务管理 |
| GUI | 已可完成一局合法人机对弈（PySide6） | 采集、训练、自我训练、测评任务编排 |

现有工具可用于检查和迁移基础能力，不能当作新工作台已经交付。旧权重不视为新网络的兼容模型。

## 目录

| 目录 | 职责 |
|---|---|
| engine/ | C++ 执行器、搜索、协议与部署推理 |
| trainer/ | PyTorch 数据、训练、验证与导出 |
| tools/ | 采集、对战、任务编排工具及权威执行器共享客户端 |
| gui/ | PySide6 桌面客户端与启动入口 |
| .scratch/gomoku-nnue/ | 当前主规格及审核后发布的本地工单 |
| data/ | 本地产生的数据，不纳入版本控制 |
| weights/ | 本地产生的模型，不纳入版本控制 |

构建产物与旧实验产物清理后可重新生成。
