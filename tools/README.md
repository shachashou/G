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

## 数据格式边界

现有 binpack 是64字节文件头加64字节记录，实际旧格式定义以 C++ 写端和 Python 读端为准。它属于当前实现，不是新数据契约；不能假设能完整表示新标签来源与战术状态。

新格式必须保存规则、行棋方、对局和模型身份、标签来源与版本、筛选状态和结果。升级时显式区分版本，拒绝误读；不会把 mate 或饱和值当成普通静态标签。

## 检查现有入口

~~~powershell
./.venv/Scripts/python.exe tools/datagen.py --help
./.venv/Scripts/python.exe tools/match_runner.py --help
~~~

现有选项只描述当前实现。新样本池、无教师训练和 GUI 任务管理需按新工单交付。
