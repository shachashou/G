# 04: 教师评分与有效样本落盘

**What to build:** 用户在客户端提交合法局面后，能够由本地 Rapfi 评分，看到战术筛选和异常原因，并保存一条可追溯、可被后续训练读取的有效样本。

**Blocked by:** 03 — 接入开源 VCF。

**Status:** done

- [x] 实际启动本地 Rapfi 教师，核验其可执行文件、权重和 Renju 配置，记录版本身份、搜索预算及完成同步方式。
- [x] 选择并验证可用的局面评分方法；不把引擎建议着法、静态评分和搜索评分混为同一契约。
- [x] 原始评分、原始视角、归一化方法与参数、转换后的价值和标签版本可追溯，明确供后续网络与搜索共用的价值约定。
- [x] 使用权威执行器校验输入，调用共用战术适配器，排除终局、已识别算杀、mate 标签和异常响应。
- [x] 超时、中断、预算耗尽等未知情况单独计数，默认不进入教师训练队列；有限 VCF 筛选不宣称证明不存在其他杀棋。
- [x] 保存盘面、规则、行棋方、对局或独立采样来源身份、着法信息、采集模型身份、标签来源与版本、战术状态及可空的最终结果；缺失字段有明确语义。
- [x] 独立采样局面和未完成对局没有真实结果时，不补写和棋；教师标签与最终胜负可以分别存在。
- [x] 明确数据格式版本与完整性规则，读回后内容一致；损坏、截断、错误规则或不兼容版本明确拒绝。
- [x] GUI 展示原始请求状态、评分结果、过滤原因及有效样本数；依赖缺失时明确失败，不伪造标签。

## Verification

在客户端、真实教师进程和样本读写边界完成小批量闭环；验证黑白视角、已知终局、战术命中、超时与损坏数据，兼顾可重放的适配契约检查。

### 验收记录

**教师安装与身份**

- 上游：**Rapfi 250615** 发布包 `Rapfi-engine.7z`（2025-06-15）。使用其中的 `pbrain-rapfi-windows-avx2.exe`，`ABOUT` 返回 `Rapfi 0.43.01 (clang++ 19.1.5 on Windows SSE41 AVX2)`。
- 解压到 `tools/rapfi/`（`.gitignore` 忽略），身份与哈希固定在入库文件 `tools/rapfi.lock.json`：
  - 引擎 `1e02497433fb765626f954c895b0b92e057ac12a76c76318abe74d4050914e8e`
  - Renju 黑权重 `mix9svqrenju_bs15_black.bin.lz4` `7ab8c974…46c5ec`，白权重 `mix9svqrenju_bs15_white.bin.lz4` `a31f3185…e4e4e`
  - `config.toml` `32b9180c…0d5d3c`；标签版本 `teacher-v1`
- 每次打标前 `check_install()` 逐项核对存在性与 SHA-256；缺失/不符返回明确错误，GUI 显示「不可用」且不写任何样本。
- Renju 配置实测：不发送 `INFO rule 2` 时加载 `mix9svqfreestyle_bsmix`（freestyle）；发送后加载 `mix9svqrenju_bs15_black/white`，确认规则确实切到 Renju。
- 同步方式：`START 15` → `INFO rule 2` → `INFO thread_num 1` → `INFO timeout_turn <ms>` → `BOARD`/`DONE`，以引擎返回的 `x,y` 落子行作为本轮完成信号；超时余量 6000ms，超限即杀进程并报告未知。

**评分方法（与建议着法、静态分区分）**

- 使用 Gomocup `BOARD ... DONE` 让教师在固定预算（默认 800ms、1 线程）下搜索该局面，解析 `MESSAGE Depth D1-D2 | Eval V | Time T | PV ...`，取最后一条 `Eval` 作为**根位置搜索评分**；`Speed` 行的深度记入样本。引擎返回的着法单独保存为 `suggested_move`，不参与价值契约。
- 视角实测：白方必胜局面（白先）得到 `+M3`，黑方必胜局面（黑先）得到 `+M25`，空三局面得到 `+M3` → **评分以行棋方视角**、正数表示行棋方更优，mate 单位是 ply。
- mate 记作 `±M<plies>`，不归一化；直接看到立即结果而不输出搜索信息的局面记为 `no_score`，同样过滤。

**价值契约（teacher-v1）**

- 原始分（行棋方视角）→ `value = tanh(score / 600)`，范围 [-1, 1]；参数、视角和标签版本写入 `tools/rapfi.lock.json` 的 `value_contract`，并随每条样本落库。
- 例：安静开局样本 raw `443` → `tanh(443/600) = 0.6281`，读回一致。
- mate / 无评分 / 超时 / 异常不产生 value；后续训练与搜索必须使用同一 `label_version` 换算。

**过滤与未知**

- 顺序：权威执行器 `rebuild` 校验 → 终局排除（`game_over`）→ 共用 VCF 适配器（预算 50000 节点、10 攻击手）→ Rapfi。
- 已证明 VCF：`filter_reason = vcf_proven`，教师不会被调用；VCF 预算耗尽/中断只记 `tactical_status = vcf_unknown`，仍可保留教师标签，且不宣称不存在其它强制胜法；无禁手规则返回 `unsupported_rule`。
- 教师 `mate`、`no_score`、`teacher_timeout`、`teacher_error` 均不进入训练队列；超时/异常计入 `unknown`，与「未找到」分开统计。样本库的 `counts()` 同时给出 `by_reason`、`by_status`、`by_tactical` 和 `vcf_unknown`。

**样本存储**

- `data/teacher/samples.sqlite3`（SQLite；`application_id = 0x47544D31`、`user_version = 1`、`store_format = gomoku-nnue-teacher-store-v1`）。
- 每行字段：棋盘（225 字节）、规则、行棋方、ply、末着、来源种类（game/standalone）、来源 id、采集模型 id、战术状态、教师状态、过滤原因、accepted、可空最终胜负、raw_eval/raw_score/mate_plies、视角、归一化方法/参数、value、建议着法、PV、深度、预算、线程、耗时、教师 release/version 与三份身份哈希、标签版本、note。
- 约束与完整性：`CHECK` 限制棋盘长度、规则、行棋方、accepted 一致性、结果枚举；`validate()` 执行 `PRAGMA integrity_check` 并检查行棋方与棋子数一致、accepted 必须有 value、独立采样不得伪造和棋。
- 拒绝路径实测：截断文件报「not a teacher store or the file is corrupt」；`user_version=99` 报「newer than this reader」；freestyle 规则写入被拒绝；手工插入行棋方不符的行会被 `validate()` 指出。
- 未完成对局与独立采样写 `final_result = NULL`，读回仍为 `NULL`；真实结束的对局才写 `black_win`/`white_win`/`draw`。

**GUI**

- 状态区显示教师身份（版本、Renju 权重、`teacher-v1`）与安装检查结果。
- 「打标并保存样本」使用独立执行器校验当前局面（不改动对局中的执行器历史），后台线程完成 VCF 过滤 + Rapfi 打分 + 落盘；面板显示 accepted/过滤原因、`raw → value`、教师状态、预算、有效样本数与总请求数（含未知、VCF 未知）。
- 教师缺失（模拟 `check_install` 失败）时错误区显示「教师不可用」，样本总数保持不变。
- 停止/关闭窗口会同时终止教师子进程，不留下后台任务。

**实际执行**

- 命令：
  - `.venv\Scripts\python.exe tools\rapfi_teacher.py about`
  - `.venv\Scripts\python.exe tools\teacher_label.py label "H8,H9,I8,I9"`
  - `.venv\Scripts\python.exe tools\teacher_label.py stats`
  - `.venv\Scripts\python.exe tools\teacher_label.py validate`
- `teacher_tests`（真实 Rapfi + 真实执行器 + 临时样本库）：**48 项全过**，覆盖身份/横幅、接受样本与读回一致、视角正负、mate/终局/VCF 过滤、超时未知计数、截断/新版本/错误规则/语义损坏、NULL 结果。
- `gui_teacher_smoke`（offscreen）：**14 项全过**，覆盖教师身份、真实打标后显示评分与有效样本数、教师缺失报错且不落盘。
- `ctest`：**22/22 全绿**（原 20 项 + `teacher_tests`、`gui_teacher_smoke`；未安装 Rapfi 的机器上两者返回 77 记为 skip）。
- 小批量闭环：命令行与 GUI 各自写入真实样本，`stats/validate` 通过；库内同时保留被过滤请求的计数。

**限制与后续**

- Rapfi 二进制与权重不入库（体积大且可重新下载），身份由 `tools/rapfi.lock.json` 固定；新克隆需要按 README 安装 Rapfi 才能打标，缺失时测试 skip、GUI 明确报错。
- 教师仅配置 Renju 15×15；freestyle 局面返回 `unsupported_rule` 且不入库。
- 预算以时间计，标签不是逐位可复现；每条样本记录预算、线程、深度和身份，便于追溯。该 Rapfi 构建未提供 `INFO depth` 固定深度，若要完全可复现的离线标注需后续单独讨论。
- 有限 VCF 过滤只排除「已证明」的强制局面，不宣称其它局面绝无杀棋。
- 尚未实现多局面批量打标、样本池容量/淘汰与训练接入（工单 05/06）。
