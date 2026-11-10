# tools：采集、测评、任务编排与执行器客户端

新方案以[主规格](../.scratch/gomoku-nnue/spec.md)为准，术语见[术语表](../GLOSSARY.md)。

## 目标工作流

- **教师启动**：收集合法局面，经过战术筛选，由 Rapfi 打标，训练初始正式版本。
- **自主训练**：正式版本自我对弈，完整结束后回填胜负，持续训练候选模型。
- **历史测评**：候选挑战冻结旧版本，配对换色，使用公平思考资源并记录晋级证据。
- **任务管理**：启动、暂停、恢复、停止、进度和版本发布，由 GUI 与命令入口共用后端能力。

样本池控制当前留存量；累计采集量和实际训练曝光量分别报告。与 Rapfi 对战不是交付要求。

## 现有可复用工具

| 入口 | 当前职责 | 改造重点 |
|---|---|---|
| executor.py | 权威执行器 JSON-lines 共享客户端（工单 02） | 供 GUI、采集、算杀证据复查与比赛裁判复用 |
| vcf_tactics.py | 开源 VCF 算杀的可复用查询/打标边界（工单 03） | 工单 05 用它过滤已识别强制战术局面 |
| vcf_bench.py | 算杀调用点与节点预算实测（工单 03） | 记录搜索接入位置和开销，供后续调整 |
| rapfi_teacher.py | 本地 Rapfi 教师适配器（工单 04） | 固定身份、Renju 配置、预算与评分解析 |
| teacher_store.py | 版本化教师样本存储（工单 04） | 连续性、完整性与来源字段校验 |
| teacher_label.py | 执行器 → VCF → Rapfi → 落盘流水线（工单 04） | 供持续打标复用 |
| collect.py | 自我对弈采集、抽样、打标并提交样本池（工单 05） | 与样本池、教师和 GUI 共用同一适配层 |
| sample_pool.py | 有容量、去重、冻结验证集的滚动样本池（工单 05） | 工单 06 的训练读取边界 |
| datagen.py | 多进程自我对弈采集、合并、校验与吞吐统计 | 标签来源、战术筛选、持续入池 |
| match_runner.py | 双引擎对战、配对开局、结果和异常报告 | 共用完整禁手裁判、冻结版本与新晋级配置 |
| dataset_profile.py | 数据画像 | 统计新标签来源、过滤原因与数据覆盖 |
| run_support.py | 配置、文件身份和结果记录的共享基础 | 供新编排复用 |

旧诊断、蒸馏、局部排序和预算脚本已移除；基础采集、对战和数据画像能力保留。

## 权威执行器客户端

`executor.py` 的 `ExecutorClient` 启动 `gomoku_nnue.exe --executor`，通过每行一个 JSON 请求/响应通信。它提供局面重建校验、完整合法落点、禁手原因、落子、撤销、着法序列重放、搜索候选裁剪和引擎选点。所有规则判定都由 C++ 执行器给出，Python 侧不实现禁手或胜负算法；客户端负责进程启动、请求串行化、诊断收集与进程清理。命令清单见[客户端说明](../gui/README.md)。

## 算杀边界（工单 03）

`executor.py` 新增 `vcf(max_nodes=..., max_attack_plies=..., time_ms=...)`，对应执行器的 `vcf` 命令；`think()` 也接受 `vcf_policy`、`vcf_nodes`、`vcf_plies`。结果词表由上游实现与适配层共同定义：`proven` / `disproven` / `budget_exhausted` / `interrupted` / `unsupported_rule` / `invalid_input`。只有 `proven` 带首着与变化，且变化已经过权威执行器逐手复查。

`vcf_tactics.py` 只依赖这个客户端，不读 GUI 状态、不使用教师评分字符串，因此数据打标与采集进程可直接复用：

~~~powershell
.venv\Scripts\python.exe tools/vcf_tactics.py --engine build\engine\gomoku_nnue.exe check H8 H7 I9 G7
.venv\Scripts\python.exe tools/vcf_tactics.py --engine build\engine\gomoku_nnue.exe filter positions.jsonl --output labels.jsonl
~~~

`filter` 读取 `{cells, side_to_move}` 的 JSONL，逐条输出 `vcf` 与可读说明；是否丢弃已证明局面由调用方（工单 05 的样本池）决定。`budget_exhausted` 与 `interrupted` 必须按未知处理，不能当成"没有杀棋"。

`vcf_bench.py` 用 `think` 的三种 `vcf_policy` 对比根节点、根+子节点与关闭时的用时、节点数和子查询命中数，并扫描根预算；工单 03 的默认配置就是它实测定下的。

## 教师打标（工单 04）

本地安装 Rapfi 250615 到 `tools/rapfi/`（`tar -xf Rapfi-engine.7z -C tools/rapfi`），身份与哈希固定在 `tools/rapfi.lock.json`。`rapfi_teacher.py` 负责启动、`START/INFO rule 2/BOARD` 同步、读取 `MESSAGE ... Eval` 搜索评分并按行棋方视角解释；mate（`±M<plies>`）、无评分、超时和进程异常分别返回状态。评分与建议着法分开记录，raw 分经 `tanh(score/600)` 归一化为 `teacher-v1` 价值。

`teacher_label.py` 按“执行器校验 → 排除终局 → 排除已证明 VCF → Rapfi 评分 → 过滤 mate/超时/异常 → 落盘”的顺序处理；被过滤的请求也写库用于计数，未知（超时、异常）单独统计且默认不进入训练队列。`teacher_store.py` 是 SQLite 存储：`application_id` + `user_version` 固定格式，`validate()` 同时跑完整性检查与语义检查（行棋方与棋子数一致、accepted 必须有 value、独立采样不得伪造和棋）；截断、更新版本、非 Renju 规则都会明确拒绝。

~~~powershell
.venv\Scripts\python.exe tools\rapfi_teacher.py about
.venv\Scripts\python.exe tools\teacher_label.py label "H8,H9,I8,I9"
.venv\Scripts\python.exe tools\teacher_label.py stats
.venv\Scripts\python.exe tools\teacher_label.py validate
~~~

## 持续采集与样本池（工单 05）

`collect.py` 用当前引擎自我对弈：随机开局 + 定深搜索，按 `--sample-every` 抽取局面，走 `teacher_label.py` 的“执行器校验 → VCF 过滤 → Rapfi 评分”流水线，再按对局提交进样本池。`sample_pool.py` 是 SQLite 滚动池：

- 对局先暂存（staged），提交后读取者才可见；中断的对局直接丢弃，读取者永远不会看到半写入数据。
- 精确去重 + 默认 D4 规范形去重；验证集在首批提交时冻结，训练侧与验证 D4 等价的样本被拒绝；容量淘汰只删最旧的训练对局，验证对局永久保留。
- 读取走 WAL 快照，写端淘汰不会影响正在读取的行；`--throttle-ms` 在高利用率时暂停采集。
- `stats/validate/evict` 分别查看计数、跑完整性检查、按新容量触发淘汰。

~~~powershell
.venv\Scripts\python.exe tools\collect.py --games 2 --capacity 80 --sample-every 3 --budget-ms 200
.venv\Scripts\python.exe tools\sample_pool.py stats data\pool\pool.sqlite3
.venv\Scripts\python.exe tools\sample_pool.py validate data\pool\pool.sqlite3
.venv\Scripts\python.exe tools\sample_pool.py evict data\pool\pool.sqlite3 --capacity 40
~~~

## 历史测评与正式版本（工单 08）

`history_evaluation.py` 提供 `initial` / `challenge` / `list` 命令及同名 GUI 复用的 `evaluate()` 接口。`model_registry.py` 保存冻结历史、当前正式指针与晋级来源，带锁发布并检查测评期间现任版本是否变更。完整操作和统计默认值见[根目录说明](../README.md#历史版本测评与晋级工单-08)。

测评复用 `match_runner.Engine` 的 Gomocup 进程传输，但规则裁判仅使用 `executor.py` 对接的 C++ 权威执行器；旧 runner 的截断和棋与旧胜率门槛不能作为正式版本晋级证据。测评不会写入 `SamplePool`，不需要 Rapfi。

## 持续教师训练编排（工单 09）

`training_task.py` 提供 GUI / CLI 共用的 `TrainingTask(TaskConfig).run()`、`stop()`、`snapshot()`。`TeacherDataSource` 负责教师依赖及单盘数据生产；调度器只依赖数据源的 `preflight()` / `produce()`，训练与测评分别调用现有 `train_value()` 和 `evaluate()`，为后续替换监督来源保留边界。

- 每盘开始从正式登记表绑定模型，核验实际加载身份；采集运行配置记录 task id、正式版本与模型哈希，提交事件关联 source ids 和 pool run id。
- `SamplePool.training_snapshot()` 在同一 WAL 读事务中读取分组、提交计数和数据；`value_data.load_pool()` 冻结为内存数据。指纹覆盖完整记录，不再只哈希行号；`commit_sequence` 与 `committed_training_samples` 区分已提交数据和暂存写入。
- 调度器要求启动阈值、后续新提交样本与变化的数据指纹，支持新旧抽样、积压背压、容量限速、候选保留及次数/时间/空转上限；不会无限重复训练未变化的数据。
- 任务持有样本池专属 lease，GUI 禁止与手工任务并行抢占。停止会回收本任务拥有的引擎/教师进程、丢弃暂存对局，保留已提交数据与有效模型；完整任务恢复留给工单 12。

默认值与操作见[根目录说明](../README.md#持续教师训练工单-09)，所有可配置字段见[示例 JSON](teacher-task.example.json)：

~~~powershell
.venv\Scripts\python.exe tools\training_task.py --config tools\teacher-task.example.json
~~~

## 数据格式边界

现有 binpack 是64字节文件头加64字节记录，实际旧格式定义以 C++ 写端和 Python 读端为准。它属于当前实现，不是新数据契约；不能假设能完整表示新标签来源与战术状态。

新格式必须保存规则、行棋方、对局和模型身份、标签来源与版本、筛选状态和结果。升级时显式区分版本，拒绝误读；不会把 mate 或饱和值当成普通静态标签。

## 检查现有入口

~~~powershell
./.venv/Scripts/python.exe tools/datagen.py --help
./.venv/Scripts/python.exe tools/match_runner.py --help
~~~

现有选项只描述当前实现。完全无教师训练与完整暂停恢复仍按工单 10 / 12 交付。
