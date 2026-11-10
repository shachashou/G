# GomokuNNUE

Windows 本地五子棋训练与对弈工作台：在同一个客户端完成"玩一把、收集数据、从0训练、自我训练、历史版本测评"。

**项目正在按新方案重设计。** 当前已交付执行器、搜索、训练与推理基础、可完成一局合法对弈的桌面 GUI、开源 VCF 算杀、Rapfi 教师打标与持续样本池、新价值网络的实际 Alpha-Beta 对弈，以及历史版本测评与正式版本晋级；持续教师训练编排与无教师自主训练仍待实现。

- [主规格](.scratch/gomoku-nnue/spec.md)：目标行为、交付顺序、测试与范围，状态 ready-for-agent。
- [术语表](GLOSSARY.md)：教师启动、自主训练、正式版本、候选模型与晋级等定义。
- [训练端说明](trainer/README.md) / [工具说明](tools/README.md) / [客户端说明](gui/README.md)：现有能力及改造边界。
- [实施工单](.scratch/gomoku-nnue/issues/)：已发布12张，每张独立记录验收条件和阻塞关系。

已完成 [01：训练基础脱离旧实验](.scratch/gomoku-nnue/issues/01-retire-legacy-experiments.md)、[02：GUI 完成一局合法对弈](.scratch/gomoku-nnue/issues/02-desktop-play-and-shared-rules.md)、[03：接入开源 VCF](.scratch/gomoku-nnue/issues/03-open-source-vcf-integration.md)、[04：教师评分与有效样本落盘](.scratch/gomoku-nnue/issues/04-rapfi-labeling-and-sample-storage.md)、[05：持续采集与滚动样本池](.scratch/gomoku-nnue/issues/05-continuous-collection-and-replay-pool.md)、[06：新网络从零训练与导出](.scratch/gomoku-nnue/issues/06-new-network-training-and-export.md)、[07：使用新模型实际对弈](.scratch/gomoku-nnue/issues/07-new-model-cpp-play.md) 和 [08：历史版本测评与晋级](.scratch/gomoku-nnue/issues/08-history-evaluation-and-promotion.md)。其余工单按各自的 Blocked by 推进；ready-for-agent 不表示依赖已经完成。

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

**Windows 上的编译工具链完全收敛在 `.venv` 内，不读取 Visual Studio、Windows SDK 或全局 `~/.cargo`。**

- Python 3.12 安装在项目外的稳定位置（本机为 `D:\Python312`），项目 `.venv` 由它创建；不要删除基础 Python 或把它放在会清理的 `.scratch` 中。Python 依赖（含 PySide6、PyTorch）与 `cmake` / `ctest` / `ninja` / `ziglang` 都是 venv 里的包。`ziglang` 自带 clang、libc++、lld 与 mingw-w64 头文件/库，作为 C/C++ 编译器。
- VCF 算杀复用 vendored 的开源 quintet；Rust 工具链通过一次性引导安装到 `.venv/rustup` 与 `.venv/cargo`（`stable-x86_64-pc-windows-gnu`，含 proc-macro 所需的 rust-mingw 导入库），链接器使用 venv 内的 `rust-lld`。
- CMake 通过 `tools/build/zig-cc.cmd` / `zig-cxx.cmd` 调用 venv 中的 Zig，Rust 桥由 CMake 自动执行 `cargo build --release` 生成 `engine/vcf-bridge/target/release/libvcf_bridge.a` 再链接。

首次配置（Windows x64，Python 3.12）：

~~~powershell
D:\Python312\python.exe -m venv .venv
# 如需代理，先在当前 PowerShell 会话设置（本机代理端口 10808）：
$env:HTTP_PROXY = "http://127.0.0.1:10808"
$env:HTTPS_PROXY = $env:HTTP_PROXY
# 按下表选择一个依赖文件；本机为 AMD。
.venv\Scripts\python.exe -m pip install -r requirements-amd.txt
.venv\Scripts\python.exe tools\build\bootstrap_toolchain.py
~~~

`requirements.txt` 只包含通用依赖，PyTorch 后端按硬件选择；三种配置使用同一套数据、训练和 `GMVALUE1` 导出流程：

| 运行设备 | 安装文件 | PyTorch 后端 | 前提 |
|---|---|---|---|
| AMD GPU（本机 RX 7900 XTX） | `requirements-amd.txt` | 2.9.1 + ROCm 7.2.1，AMD 官方 Windows wheel | Python 3.12，受支持的显卡和驱动 |
| NVIDIA GPU | `requirements-nvidia.txt` | 2.6.0 + CUDA 12.4 | 支持该 CUDA 版本的 NVIDIA 显卡与驱动 |
| CPU | `requirements-cpu.txt` | 2.6.0 CPU | 无 GPU 运行时要求 |

AMD 的支持范围与驱动要求以[官方兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)和[安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)为准；上述 ROCm 7.2.1 发行说明要求 26.2.2 显卡驱动。AMD 配置不安装 NVIDIA CUDA 包。切换后端时重建 `.venv`，再安装所选依赖，避免混装。

**AMD ROCm 也使用 PyTorch 的 `torch.cuda` API 和 `cuda` 设备名。** `--device auto` 对两种 GPU 都会自动选择 `cuda`；`--device cuda` 并不代表只能使用 NVIDIA。用 `torch.version.hip` 区分 ROCm、`torch.version.cuda` 区分 CUDA；CPU 可显式使用 `--device cpu`。安装后先检查实际设备：

~~~powershell
.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__, 'HIP:', torch.version.hip, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

设备识别后还应运行小规模真实训练验证前向、反向和优化器；可识别显卡不等于训练已经验证。C++ 执行器和部署推理运行在 CPU 上，与训练使用哪种 GPU 无关。

本机环境、从零重置的数据状态及实际测试结果见[环境与数据验收记录](.scratch/gomoku-nnue/environment.md)。

构建与测试：

~~~powershell
.venv\Scripts\python.exe tools\build\build.py --clean
.venv\Scripts\python.exe tools\build\build.py --test
.venv\Scripts\python.exe tools\build\build.py --test --test-regex value
~~~

`build.py` 使用 venv 的 cmake + Ninja，并把 Zig 缓存、Rust 工具链与链接器都指向 `.venv`。链接步骤在 Ninja 中串行执行，避免 Zig/lld 并行链接耗尽内存。Linux / macOS 仍使用系统 C++ 编译器与 Rust 工具链，直接执行同样的 `build.py`（脚本只在 Windows 设置 venv 工具链变量）。

技能目录 `.agents/skills/` 由根目录 `skills-lock.json` 记录的来源安装，属于安装物而非项目源码，因此不纳入版本控制；新克隆后按该锁文件重装技能即可，版本仍由锁文件固定。

## 启动对弈

~~~powershell
.\gui\play.bat
~~~

或 `.venv\Scripts\python.exe gui\gomoku_gui.py --help`。客户端通过 JSON-lines 协议调用权威执行器：完整合法落点、禁手原因、落子结果和终局都由 C++ 执行器给出，界面与 Python 裁判不另写规则。详见 [客户端说明](gui/README.md)。

**直接双击 `gui/play.bat` 默认加载 `weights/registry/state.json` 指向的当前正式模型。** 尚未登记 v0 时，显示手工威胁评估 + Alpha-Beta + VCF；`--model PATH` 可显式试玩候选。登记或晋级后，GUI 显示真实正式版本身份，后续启动沿用该版本。

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

## 持续采集与滚动样本池

- **采集**：`tools/collect.py` 用当前引擎（未训练模型时即手工评估）自我对弈：随机开局（区域可配）+ 定深搜索，按间隔抽取开局与中盘局面，经“执行器校验 → VCF 过滤 → Rapfi 教师”打标后按对局暂存并原子提交。对局未完成时写入 NULL 结果，不伪造和棋。
- **配置与身份**：局数、深度、最大手数、开局手数/区域、采样间隔、教师预算、容量、验证局数、D4 去重开关、新旧混合比例、限速都记录在样本池的运行配置里；采集策略版本为 `collect-v1`，模型身份随对局记录。
- **样本池**：`data/pool/pool.sqlite3`（SQLite，`application_id` + `user_version=1`）。对局先 `staged`，`commit_game` 在一个事务里转成可读并执行容量淘汰；中断/崩溃的暂存对局对读取者不可见，停止时主动丢弃。
- **去重与泄漏**：按“棋盘+行棋方”精确去重；默认再按 D4 规范形去重，所以对称增广不计为独立原始数据。验证集成员在首批提交时冻结，训练样本若与验证样本 D4 等价会被拒绝并计数；淘汰只针对最旧的训练对局，验证对局永不淘汰。
- **并发与背压**：读端在 WAL 快照里读，写端淘汰旧对局不会把读取中的行抽走；池利用率过高时可配置 `--throttle-ms` 暂停采集。
- **统计**：`stats` 分别给出累计生成、有效、入池（去重后）、当前池容量、过滤原因、异常/未证明计数、有效样本/秒、淘汰局数与对称增广计数（采集阶段恒为 0）。
- **GUI**：状态区显示样本池容量与训练/验证数量；「启动采集」「停止采集」在后台线程运行，实时显示生成/有效/入池/池容量/有效每秒与过滤原因，停止后不遗留采集或教师进程。
- **命令**：

~~~powershell
.venv\Scripts\python.exe tools\collect.py --games 2 --capacity 80 --sample-every 3 --budget-ms 200
.venv\Scripts\python.exe tools\sample_pool.py stats data\pool\pool.sqlite3
.venv\Scripts\python.exe tools\sample_pool.py validate data\pool\pool.sqlite3
.venv\Scripts\python.exe tools\sample_pool.py evict data\pool\pool.sqlite3 --capacity 40
~~~

## 价值网络训练与导出

- **架构**：450 → 1024 → 64 → 32 → 1，单输出，无行棋方分支。输入是绝对黑白 450 占位特征（`trainer/features.py`），行棋方不作为输入，Renju 黑白角色差异保留。
- **价值契约**：标签统一为**黑方绝对价值**。教师 `teacher-v1` 的行棋方视角值在训练前按行棋方符号转换；`selfplay_result` 直接使用黑方绝对结果。网络输出 `tanh(z)` ∈ [-1,1]，搜索侧按行棋方取反，属于冻结契约。
- **训练**：从滚动样本池的已提交对局读取冻结的训练/验证划分；验证按对局隔离并检查 D4 等价泄漏。八种 D4 空间变换在线增广（不交换黑白），验证不增广。损失可选 MSE / SmoothL1；记录随机种子、配置、实际优化步数、样本曝光、标签版本与架构。
- **checkpoint**：保存模型、优化器、步数、配置、数据身份与历史；可从任意 checkpoint 继续训练。
- **部署格式**：`GMVALUE1`，包含架构、规则、特征、价值契约、标签版本、训练身份与 FNV 校验；旧 `GMNNUE01` 双头权重会被明确拒绝。整数方案为 int16 权重 + int32 bias + 每层一次移位（默认 `2^9`），Python 整数参考前向与训练模型、量化模型三方对拍。
- **导出保真度**：容差在测量前冻结（均值 ≤0.01、p95 ≤0.03、最大 ≤0.10；整数参考均值 ≤0.002、最大 ≤0.01），报告记录实测值与是否通过；不把损失下降称为棋力晋级。
- **GUI**：「价值网络训练」面板选择样本池、步数/批大小/学习率/损失，后台训练并显示进度、训练损失、验证损失与 MAE、导出保真度与输出路径；停止会保存 checkpoint 并导出当前模型。
- **命令**：

~~~powershell
.venv\Scripts\python.exe trainer\value_train.py --pool data\pool\pool.sqlite3 \
    --output-dir weights\value-run1 --steps 200 --batch-size 32 --lr 1e-3 --loss mse
~~~

## 新模型实际对弈（工单 07）

- **加载**：`--value-model PATH` 严格读取 `GMVALUE1`。校验 magic/版本、架构、规则、特征、价值契约、固定点参数、元数据一致性与 FNV 校验；损坏、截断、旧 `GMNNUE01` 或元数据不兼容时进程以非零码退出并在 stderr 说明原因，**不会**静默切回手工评估再标成 NNUE。`--value-model` 与旧 `--nnue` 互斥。
- **推理参照**：`engine/src/value_net.{h,cpp}` 实现 450 → 1024 → 64 → 32 → 1 的标量前向和全量第一层重建（int64 累加，避免饱和后无法撤销）。这是工单 11 增量累加器与 SIMD 的长期数值参照；本工单的搜索评估每次做全量重建。
- **数值契约**：网络输出黑方绝对定点 z；`value = tanh(z / 2^ea)`。搜索使用 `stm_score = ±round(z × 1000 / 2^ea)`（黑方为正、白方取反），并限制在 ±100000，远低于 mate 分。终局与已证明 VCF 分数由搜索先判定，网络分数不能覆盖。
- **搜索接入**：新模型接入现有 Alpha-Beta 选点；合法落点、终局、开源 VCF 与时间控制共同生效。`about`/`think` 返回实际加载的 `model_id`、架构、规则、特征、价值契约与 `backend=scalar-full-rebuild`；模型只是「候选试玩（未晋级）」，加载成功不会自动晋级。
- **GUI**：「模型与评估方式」面板列出/浏览 `weights/` 下的 `*.gmvalue`，应用后重启引擎并显示真实模型身份；加载失败保留当前引擎并明确报错。启动参数 `--model PATH`。
- **无界面路径**：执行器 `--executor --value-model PATH`、Gomocup 协议模式与 `tools/match_runner.py --engine-a "... --value-model PATH"` 均可完成真实对局；从着法记录 `replay` 得到相同终局。
- **验证**：`value_net_tests`（加载/拒绝/前向/全量重建/视角/终局与 VCF 优先级）、`value_parity_tests`（C++ 与 `trainer/value_format.infer_int` 逐位一致；本地 demo 模型与训练原模型按冻结容差对拍）、`value_engine_blackbox`（真实完整对局、重放、时间控制、正负视角、严格失败）、`gui_model_smoke`（offscreen 模型选择到落子）。验收记录见[工单 07](.scratch/gomoku-nnue/issues/07-new-model-cpp-play.md)。
- **性能**：全量参照推理在 40 子局面实测约 14 µs/次；增量与 AVX2 加速属于工单 11，本工单不以此声称棋力提升。

## 历史版本测评与晋级（工单 08）

客户端的「历史版本测评与晋级」面板可选择候选模型、额外历史对手、开局对数、每步时间和得分率基线；当前正式版本自动加入对手列表。没有正式版本时，先「验证并登记 v0」：数值检查通过且自对弈完成一对换色合法对局后建立初始版本，不宣称战胜历史版本。

- **冻结**：开赛前复制候选、对手、引擎、保真度报告、训练局面清单，记录 SHA-256、规则、开局、搜索预算与方法；双方都核验实际加载的模型身份，拒绝回退。
- **公平与隔离**：每个开局换色两局，同一引擎、相同时间与单线程。开局不能与候选训练/验证局面 D4 等价，同一测评目录下已用过的开局也不复用。评测对局只写入测评目录，不写训练样本池。
- **默认统计方案**：64 个开局对（128 局）/对手，至少 32 个完整开局对；对每对两局的平均得分使用 Hoeffding 单侧下界，整体 `alpha=0.05`，多个对手用 Bonferroni 校正。只有每个对手的下界都**严格高于 0.5**才具备晋级资格。这是保守的工程默认值，可在开赛前配置，不继承旧 M5 门槛。
- **异常与停止**：崩溃、超时、非法落子、身份不符或模型加载失败即终止本次测评；异常判负与有效棋力证据分开，达到手数上限只记未完成，不算和棋。停止保存部分报告并回收子进程。证据不足或任何异常均保持现任版本。
- **发布**：发布前复核冻结身份、统计和权威执行器重放的终局，带锁原子替换正式版本指针；若现任版本已变更则拒绝过期发布。历史版本保留模型、数值报告及晋级来源。
- **报告**：`data/evaluations/<run>/manifest.json` 是赛前配置，`report.json` 是封存对局与统计，`result.json` 是发布回执。GUI 显示胜/和/负、严格胜率、得分率、有效局数、配对下界、结论、用时和异常，并可查看已有报告。

候选目录必须有新训练器导出的 `model.gmvalue`、`fidelity.json`、`training_positions.json`。缺少训练清单的旧模型仍可试玩，但不能用于本流程登记或晋级；需要通过新训练链路生成可追溯导出。

~~~powershell
.venv\Scripts\python.exe tools\history_evaluation.py initial weights\value-run1\model.gmvalue
.venv\Scripts\python.exe tools\history_evaluation.py challenge weights\candidate\model.gmvalue --pairs 64 --movetime-ms 200
.venv\Scripts\python.exe tools\history_evaluation.py challenge weights\candidate\model.gmvalue --opponent v0
.venv\Scripts\python.exe tools\history_evaluation.py list
~~~

`--registry` 和 `--output-root` 可指定隔离目录。小规模验收结果见[工单 08](.scratch/gomoku-nnue/issues/08-history-evaluation-and-promotion.md)，不作为棋力提升声明。旧 `match_runner.py --min-win-rate` 仅供旧实验回归，不是本流程的发布入口。

## 当前代码与目标的差距

| 区域 | 当前情况 | 改造目标 |
|---|---|---|
| 执行器 | 已有 C++ 权威执行器、Renju、协议和 Alpha-Beta | 核验规则并统一裁判、采集和对弈 |
| VCF | 已接入开源 quintet（MIT，固定 commit），适配层区分已证明/未证明/预算耗尽/中断，证据经执行器复查 | 保持上游版本，供工单 05 的战术筛选与打标复用 |
| 标签 | 已接入 Rapfi 教师（固定版本）与版本化样本存储，区分评分/建议着法/胜负标签 | 持续采集与样本池（工单 05）、结果标签 |
| 网络 | 新 450 → 1024 → 64 → 32 → 1 单输出网络已训练、导出并可加载对弈 | 无教师自主训练与历史版本晋级 |
| 推理 | `GMVALUE1` 标量全量参照前向已接入搜索；旧 NNUE 保留 | 第一层增量累加、AVX2 与实际搜索吞吐（工单 11） |
| 数据与编排 | 有容量上限的持续样本池与训练闭环 | 持续自主训练、结果标签与任务管理 |
| GUI | 对弈、采集、教师打标、训练、模型选择、历史测评与原子晋级已完成 | 持续教师训练与无教师自主训练编排 |

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
