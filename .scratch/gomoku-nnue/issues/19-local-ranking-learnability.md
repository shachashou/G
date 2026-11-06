# 19: 原局面—子局面误差分解与局部排序可学性诊断

**What to build:** 用现有工单18权重定位静态蒸馏为何没有形成走子知识，再用真正独立训练的float/QAT小规模对照区分量化训练、全数据拟合和候选子局面分布问题。先建立可复核的局部排序能力，再决定下一轮正式训练目标。最终整数搜索着法Top-1仍是进入棋力筛选的第一门控。

**Status:** in-progress（真实阶段A完成并核验；阶段B独立float/QAT工具已实现、clean128条及初始化/批次已封存；真实B训练待用户启动，C–E尚未实施或运行）
**Category:** diagnosis / controlled training
**Blocked by:** 无；18已完整阴性关闭，原权重/冻结/标签缓存齐全。
**Downstream:** 根据证据形成下一轮训练设计；本工单不自动解除16/10/12/11依赖，也不运行正式比赛。

## 1. 已知证据与待区分假设

工单18的D/S两组各1600步、相同1637392样本曝光；628条冻结验证和21387个合法非终局候选子局面的STE/Python整数/C++ scalar/AVX2逐位一致，导出保留了训练STE。最终整数搜索着法Top-1仅D **3.7879%** / S **3.1987%**，随机 **3.0812%**，手工教师 **43.4203%**，第一门控失败，未执行筛选。

D验证Huber/训练常数基线比0.964153，MSE/常数比1.031833；64条同样本STE拟合比0.085613通过，但那是小训练集上的拟合，不证明全训练池、D4增广或候选局面的可学性。同一QAT master权重的连续float旁路不是独立浮点训练，不能拿它排除量化训练问题。

预声明并列假设：

- **H1 全数据拟合不足**：现有模型在见过的原训练局面上也只接近常数；应先查训练目标、优化和表示，而不能只归因于走后分布。
- **H2 原局面—候选子局面错位**：原局面已拟合，但相同父局面的候选子局面误差/排序显著更差；训练侧子局面蒸馏可能弥补。
- **H3 QAT学习受限**：在相同小数据、初始化和预算上，独立float训练能拟合而独立QAT训练不能；是量化训练路径的证据，区别于最终导出错误。
- **H4 D4增广/局部排序监督不足**：小集无增广能学、加入D4后明显退化，或子局面点值误差改善但局部排序仍差；需单变量检查，不能直接扩网。

阶段A只提供描述性证据，最多决定下一控制实验，不给出唯一根因。父子难度、ply、行棋方和教师幅度不同会混淆总体误差；必须看匹配父局面及分桶。

## 2. 全程不变的契约

- 15×15、Renju、radius1、颜色绝对450维、双stm输出头、1024/15网络、`GMNNUE02`、`G=128`及±900000静态界限不变。
- 教师仍为同一C++ `Evaluator` 无network；不在Python重写棋形标签，不把父局面的搜索分/WDL当作新子局面的标签。
- 原验证28局/628条及其594个主局面/21387个候选子局面保持不变，不重抽更容易的验证局面，不调原Top-1阈值。
- 工单18计划及源码SHA256继续验证；新增19工具使用新文件，禁止修改旧冻结JSON、标签缓存、checkpoint、候选权重或重导出覆盖旧证据。
- 训练仅使用原允许对局派生的诊断小集；不新增自我对弈。派生子局面是新监督样本，须独立manifest和D4扩展泄漏检查，不能声称它与原训练数据完全相同。
- 阶段A针对D/S都计算对手工H的误差；S原来学的是搜索分，所以这里是S对H的迁移诊断，不是S原训练损失的复算。

## 3. 阶段A：分析现有权重（本次交付并可执行）

### A1. 输入与确定性抽样

`trainer/local_ranking_diagnosis.py`默认读取 `data/m5/distill18/`：plan、summary、frozen_validation、train_manifest、target_d、教师sidecar，以及summary绑定的D/S checkpoint、最终.nnue和fidelity。核验文件SHA256/字节数、pool身份、v2/G、checkpoint/最终权重六组张量逐项一致；不根据路径名猜模型身份。

训练侧固定seed **20261119**，抽 **256个父局面**：先按`(source_sha256, game_id)`分组并打乱对局，依次交替选stm0/1各128条，每局最多1条，从相应行棋方的有效记录中均匀抽取。不按教师分、模型误差或走子结果挑样本。游戏编号只在单文件内有效，跨文件必须带源文件SHA256。

验证侧读取原case清单的全部628条；原局面误差使用全部记录，局部排序使用冻结594条主集。训练侧也调用C++ `candidates`，把含立即获胜着法的父局面放入terminal-tactical桶，候选不足2个的放入single-candidate桶；局部排序仅使用所有候选均非终局且候选≥2的primary桶。不得删除不利的预测，记录桶数及覆盖率。

在推理前写入 `sample_manifest.json` 与 `cases.jsonl`，冻结源文件/记录/对局/stm、板面、教师记录着法、候选列表及筛除原因。后续重试只能在新输出目录重放原manifest；不能重抽seed。

### A2. 扩展D4泄漏审计

原18只保证训练原记录与验证原记录D4分离。19还计算以下四种重合，均保留颜色与stm：

1. 训练父局面 ↔ 验证父局面：应为0，否则原冻结契约失效，立即失败。
2. 训练父局面 ↔ 验证primary子局面。
3. 训练primary子局面 ↔ 验证父局面。
4. 训练primary子局面 ↔ 验证primary子局面。

后三者在只读分析中如实计数；不能静默删除来美化诊断结果。后续拟合必须排除有任一重合的**整个来源对局**。分别报告原256样本和 `train_followup_clean` 子集的误差/排序，输出不具备后续训练资格的`game_key`。验证原局面及其子局面不改变。若clean池不足阶段B需求，阶段B停止，另行封存扩充训练侧抽样计划，不自动换seed补样。

### A3. 前向路径与标签视角

对每个原局面与所有primary子局面，用相同probe批量计算实际C++手工H。原局面H需与int32 sidecar逐项一致，否则失败。

每组模型计算：最终C++ AVX2、C++ scalar、checkpoint STE、QAT master连续float旁路。前三者在**全部新增训练样本及子局面、全部验证样本及子局面**必须逐位一致；报告静态上限次数。主报告合并相同的STE/整数路径为`d_integer/s_integer`，另列`d_master_float/s_master_float`，明确后者不是独立训练控制。

原局面训练目标是`H(B,stm)`；候选m的子局面标签是`H(B+m,1-stm)`。比较当前方着法时，同时取负：`h_m=-H_child`、`e_m=-E_child`。由于1.15先手系数，不能从父局面H取负推导子局面标签，也不能将Black标签取负当White标签。

### A4. 点值误差、偏置与局部排序指标

每组分别报告train_sample / validation / train_followup_clean三部分：

- `parents`：所有原局面的MAE、RMSE、归一化MSE、Huber、signed bias、预测/教师标准差及spread ratio。
- `paired_primary_parents`：只保留同时有可比primary子局面的父局面，避免拿全部父局面与少数战术过滤后的子局面混比。
- `children`：所有primary子局面的同口径点值误差；同时报告每父局面先聚合的macro child MAE，防止候选多的父局面支配结论。
- D/S均使用D契约中**完整原训练池拟合的手工常数**作为点值参照。该常数应用到子局面时不是“子局面最佳常数”；不得用验证子局面拟合新常数或校准模型。
- 父局面、子局面和局部排序分别按stm、ply十手桶、`abs(H)`的`≤128 / (128,1024] / (1024,8192] / >8192`分桶，带样本数。局部桶的stm/ply指父局面；子局面点值桶指走后的stm/ply。

对同一父局面n个候选，定义：

```text
h_m = -H(B+m,1-stm); e_m = -E(B+m,1-stm)
centered_error_m = (e_m - mean(e)) - (h_m - mean(h))
centered_MSE_ratio = mean(centered_error^2) / variance(h)
```

同时报告macro centered MAE、centered RMSE、逐父局面ratio均值，以及`sum(centered_MSE)/sum(variance(h))`的pooled ratio。它们只作为误差度量，不是对验证标签拟合校准；给每局所有输出加同一常数应不改变这些指标。

局部排序保留两种不同参照：

1. **搜索记录着法Top-1**：严格沿用18的并列均匀命中规则。验证侧594条的D/S整数值必须复现18；不符即分析失败。
2. **手工静态最优集合命中**：令T为教师最大值候选集合，P为模型最大值集合，命中率为`|T∩P|/|P|`，随机参照为`|T|/n`，excess为`(hit-random)/(1-random)`；另报告教师regret `max(h)-mean(h over P)`。

教师全部候选同分的局面从第二类“可学习排序”均值中排除并单独计数，不能算满分撑指标。模型全部候选同分应等于随机参照，excess=0。另计算所有教师不等分候选对的逆序率：预测反序计1、预测并列计0.5、正确计0，先每父局面归一化再取macro均值；按教师分差`(0,128] / (128,1024] / (1024,8192] / >8192`报告逆序率和对数。

### A5. 分层证据如何指导后续（不作因果结论）

- 训练原局面和验证原局面都弱：优先阶段B的真正float/QAT可拟合性，不先声称是分布迁移。
- 原局面点值较好而子局面差：核对匹配父局面、stm/ply/H桶后，考虑阶段C子局面监督。
- 子局面raw误差大但centered误差/排序好：主要是绝对偏置问题的线索，不能直接认定走子知识缺失。
- 点值误差小但局部teacher regret/逆序率仍差：考虑局部差分监督；区分小分差与大威胁分差，不能盲目提升所有候选对权重。

阶段A不因为某个分桶好就放行棋力筛选，不把训练样本上的排序当泛化证据。

## 4. 阶段B：真正独立训练的float/QAT小规模控制（后续实现）

从阶段A clean池按已冻结顺序取 **128个父局面**，stm各64、每局1条；不足则停止。只用这些父局面的静态H，**初始关闭D4**，不混入任何子局面或验证局面。两组从相同seed42/相同模型初始化独立训练；不是将一个QAT checkpoint用两种前向各评估一次。

共同配置：G=128、线性Huber beta1、batch64、AdamW/lr0.1/hidden_lr0.001、weight_decay=1e-5、WDL=0、quant_reg=0、固定 **1000次更新**、最后一步checkpoint；每100步记录同一固定小训练集误差、两头误差、激活/梯度。float只用连续前向训练；QAT用现有STE训练。报告实际样本曝光和重复次数，不因早期达到阈值缩短一组。

为防止把参数取值范围差异误判为舍入差异，两组每次optimizer更新后都将master W投影到[-64,64]、b投影到int16范围；隐藏激活与输出界限相同。float前向只去掉离散舍入，不额外放开权重域。记录边界占比、STE裁剪及梯度统计；除连续/离散运算外不引入第三个变量。阶段C/D沿用该共同参数域，不修改18已冻结的正式训练代码。

分别在这128条上拟合其自身训练均值/Huber常数，作为**同样本可拟合性分母**；同时报告18完整训练池常数供比较，但不可混用。小集拟合通过须同样本归一化MSE比和Huber比均 **≤0.25**，无静态上限命中、非有限值或单头缺失。常数基线为0的退化池在冻结前拒绝，不以重抽验证修补。

两组都在原594条验证primary子局面上只读报告点值/排序，验证值不用于挑checkpoint或调参。float能拟合而QAT不能，支持继续查量化训练；两者均不能则报告未拟合，不预设容量；两者均能只证明小集可学，不证明泛化。

交付独立 `float.pt/qat.pt`、固定采样/初始化/目标契约、每组1000步曲线、训练/验证前向与排序结果。诊断checkpoint必须带 `diagnostic_only=true`，不覆盖18模型。

**实现入口（本轮prepare-b已完成，不重跑）**：`trainer/local_ranking_fit.py` 与 `tools/run_distill19_fit.ps1`。批次另用独立CPU generator seed42生成固定1000×64索引，封存batch hash；float/QAT从相同master hash分别训练，无warm start。每组输出在 `data/m5/distill19/fit_b/{float,qat}_initial/` 的 `model.pt` / `report.json`，权重均为diagnostic_only。summary只作分流，不启动C/D或比赛。

```powershell
.\tools\run_distill19_fit.ps1 -Stage fit-float
# 第一组完整结束后再执行：
.\tools\run_distill19_fit.ps1 -Stage fit-qat
# 两组产物齐全后汇总（不训练）：
.\tools\run_distill19_fit.ps1 -Stage summary-b
```

各训练300秒硬限、原账本、各1000步/64000样本曝光；报告包括原128条及两头误差、完整验证、同批父局面的局部排序、参数边界/梯度/激活和训练期/评估期静态上限次数。中断时保留原记录，用 `-RunTag retry1` 输出到新目录；使用重试产物汇总时，通过 `tools/local_ranking_summary.py --float-report ... --qat-report ...` 显式选择，不自动挑最好尝试。后续C/D使用新增模块或新派生契约，不修改已封存B源码后冒用旧manifest。

## 5. 阶段C：按证据做子局面监督控制（条件执行）

前提：至少一个阶段B路径通过小集拟合，而且同一批父局面的候选排序仍弱（该路径的训练侧静态best excess **<0.25** 或pooled centered_MSE_ratio **>0.75**）。阈值是实验前声明的工程分流标准，不是统计因果保证。

从同一clean池/固定顺序中取 **64个primary父局面**（stm各32），冻结其全部合法非终局子局面，重新执行扩展D4审计。保留原父局面与子局面来源对局，合并去重时保留全部parent/move映射。不将验证子局面导入训练。

继续做独立float/QAT两组 **各1000步、batch64、同一初始化与阶段B超参**，保持同一Huber损失；训练抽样先均匀选父局面，再均匀选其候选子局面，避免候选多的父局面支配损失。报告原父局面/子局面样本曝光数，且明确相对B改变了训练分布。该阶段不同时改网络、学习率或损失，以免混淆数据分布效应。

通过标准是同样本子局面MSE比与Huber比均≤0.25，训练父局面集合上静态best excess **≥0.75**、macro pair disagreement **≤0.10**，并带regret及教师并列覆盖。它是训练集可学性门槛，不是18的搜索着法门槛；完全同分教师局面仍单独排除。只达到点值门槛而排序差，记录排序监督假设，**不在本阶段临时加入pairwise loss重试**。

若B两组均失败，C跳过并保留原因；若B排序已好，C也可跳过并记录它不是当前最小诊断，不为了做满矩阵消耗预算。

## 6. 阶段D：D4单变量与下一轮训练目标决策（条件执行）

若B或C有通过可拟合性标准的模型，选满足标准的最小阶段和QAT路径（仅float通过则用float），从相同初始化另跑一次 **1000步**，仅开启原D4八重随机增广，其他数据/超参/抽样不变；每组样本曝光相同，记录实际对称变换分布。与对应无增广组比较同一原样本、固定8种D4图像及子局面的点值/排序，不以随机验证增强的均值掩盖方向性退化。

若加入D4后跨过既定拟合/排序失败边界，报告D4训练困难的证据，不能归因于规则不对称（颜色没有交换）。若子局面点值始终可拟合而排序仍弱，下一轮可选择相同候选组内的差分/排序目标，但必须另封存margin、教师并列处理、hard-negative规则和pointwise辅助项权重；本工单不自动实施任意新损失网格。

本工单至少交付一张分流表：原记录是否可拟合、子局面是否可拟合、float/QAT差异、D4影响、验证侧排序是否迁移，明确支持/未排除解释及下一工单的单变量修改建议。不能因失败就自动扩网或追加自我对弈。

## 7. 阶段E：量化及最终门控复核（条件执行）

只有独立QAT路径通过对应小集门槛才导出新的诊断v2权重，在冻结594个验证父局面及全部子局面验证训练STE/整数/C++ scalar/AVX2，报告所有分值/排名差异。独立float结果可以指导训练设计，不能当作当前v2整数部署候选；量化后变差时先记录保真度，不把float命中率带入门控。

原18的阈值保持：最终整数搜索着法Top-1≥20%、较历史最佳≥10个百分点、达到手工参照一半（当前约21.7102%），相对每个bootstrap重采样中最佳历史模型的按局95%区间下界>0，数据/数值链有效且主集不触及静态上限。任何训练集静态teacher-best门槛都不替代这套门控。

19产生的模型首先是 `diagnostic_only`；即使验证Top-1意外达到标准，也只记录候选线索，在单独的训练/部署交接工单完成来源与发布核验后才准许20局筛选。本工单内**不运行20局筛选或正式200局**，不产生incumbent。

## 8. 预算、产物与停止规则

沿用 `data/m5/budget.json`，起始已用 **2244.407/7200秒**，剩余 **4955.593秒**，每阶段启动前重新核对；不重置/复制账本。A预留600秒；B float/QAT各300秒（600）；C两组各300秒（600）；D一次300秒；阶段B–D统一评估汇总300秒；E一次450秒，总最多 **2850秒**，按实际墙钟结算。顺序执行，条件不满足则跳过对应阶段、不消耗预算，预计超过5分钟或无法可靠估时的任务由用户启动。

600/300/450秒是硬上限，不是必须耗时。预算/超时不完整、错误身份、静态溢出、非有限值或前置未满足都必须保留失败，不能冒充完整阴性。中断保留原日志，用新run tag/新输出目录、同一冻结manifest重放；不覆盖原产物或偷偷放宽阈值。

阶段A目录默认 `data/m5/distill19/analysis_initial/`：

- `sample_manifest.json`：训练抽样、seed、来源记录及原验证IDs，推理前封存。
- `cases.jsonl`：所有父局面、候选与分桶、扩展D4重合和后续训练资格。
- `parents.jsonl` / `children.jsonl`：C++ H及D/S integer/master_float逐局面预测，子局面含parent/move键，视角为各自stm。
- `local.jsonl`：逐primary父局面/逐路径的centered误差、teacher-best/search-move两种命中、regret、逆序率/分差桶。
- `report.json`：完成状态、全部输入/脚本身份、原始与clean样本汇总/分桶、数值链一致性、输出界限及解释边界。
- 预算运行记录：`data/m5/distill19_analysis_initial_run/{run.json,console.log}`。

后续B–E实现时另交付独立 `fit_manifest.json`、`experiment_plan.json`、阶段配置/来源/扩展排除清单、两头覆盖、初始化hash、独立checkpoint/history以及最终 `decision.json`。这些运行产物沿用data/weights忽略规则；规范、代码与关键结论入库。阶段A完成不等于B–E已实现或运行。

## 9. 第一步执行命令（已实现）

在 `PS E:\G>` 执行；脚本自动使用项目`.venv`、原账本及600秒硬限：

```powershell
.\tools\run_distill19_analysis.ps1
```

等价完整命令（已激活项目`.venv`的终端）：

```powershell
python tools\budget_run.py --ledger data\m5\budget.json --seconds 600 --out-dir data\m5\distill19_analysis_initial_run -- python trainer\local_ranking_diagnosis.py --experiment-dir data\m5\distill18 --out-dir data\m5\distill19\analysis_initial --train-parents 256 --seed 20261119 --timeout 120 --threads 1
```

首次运行前不用重新prepare/训练/编译，也不执行18的screen。完整结果看 `report.json` 的 `complete=true` 和预算 `run.json`；终端completed只表示运行完成，仍须分析证据，A本身不做棋力门控。

若被中断且原 `sample_manifest.json` 已写出，确认无残留进程/锁后使用新tag（保留原记录与预算）：

```powershell
.\tools\run_distill19_analysis.ps1 -RunTag retry1 -SampleManifest data\m5\distill19\analysis_initial\sample_manifest.json
```

若原manifest未写出，仍用新tag但不传SampleManifest，确定性seed/采样规则不变；所有样本来源/输入身份仍验证。不能复用已存在的输出或run目录。agent工具调用使用显式 `.venv\Scripts\python.exe`；已激活用户终端的完整CLI可用python。

## 10. 具体代码清单与验收

| 文件 | 状态 / 修改 |
|---|---|
| `trainer/local_ranking_diagnosis.py` | 本次新增并实现A：冻结输入/抽样、C++标签与候选、四类D4审计、四个预测路径、原局面/子局面/centered及两种Top-1、分层报告、拒绝覆盖/manifest重放。 |
| `tools/run_distill19_analysis.ps1` | 本次新增：项目venv、600秒硬限、原账本、默认/重试tag及原manifest参数。 |
| `engine/tests/local_ranking_diagnosis_tests.py` | 本次新增：整体偏置与排序分离、常量随机基线、教师并列/逆序、子局面泄漏、真实probe小流水线/身份拒绝/重放。 |
| `engine/CMakeLists.txt` | 本次注册短测试，不修改已有probe或引擎推理语义。 |
| `trainer/local_ranking_fit.py` / `tools/run_distill19_fit.ps1` | 阶段B已实现：clean128条、独立float/QAT、共同参数投影、固定1000步/64000曝光/批次/初始化、完整验证及阶段门槛；C/D控制待新增模块实施。 |
| `tools/local_ranking_summary.py` | 阶段B已实现：核对独立训练/等算力/输入身份，区分拟合通过/失败与incomplete，输出下一阶段资格；不自动启动后续训练或比赛。 |
| 工单19、spec、tools/M5.md | 记录本轮设计/执行边界、阶段A真实产物入口与后续状态。 |

- [x] 完整规范包含可证伪假设、采样/数据身份/视角、分层指标、真正独立float/QAT控制、条件子局面/D4实验、预算/停止/门控边界。
- [x] 阶段A脚本与具体PowerShell命令已交付，小型真实probe测试通过；没有改动工单18已冻结源码/权重。
- [x] 真实阶段A completed报告、原局面与子局面误差/排序分层、扩展D4清单及证据解读。
- [ ] 独立float/QAT小规模控制完整运行，或因前置条件失败记录停止原因。
- [ ] 条件C/D/E按预声明分流执行或明确跳过，提供结论/限制/下一轮单变量建议。
- [ ] 所有实际耗时入原账本，代码检查通过，最终状态分列implementation/experiment/fit/Top-1，16/10/12/11依赖未被误解除。

初次交付仅包含规范与阶段A工具；最新真实执行状态见下方Progress Notes。运行A不会自动开始B–E。本工单可按完整诊断阴性收尾，但不把“脚本写好”当成实验完成。

## Implementation Notes（2026-11-06）

阶段A工具和重试入口已实现；CMake注册新短测试后，CTest **20/20**通过（43.41秒）。新增测试覆盖整体偏置不影响centered排序、常量预测等于随机、反序惩罚/教师全并列、父子D4扩展重合、实际C++ probe的小数据完整分析、旧Top-1复现、manifest重放与身份错误拒绝；PowerShell语法及CLI检查通过。

工单18的冻结计划/源码/模型身份再次验证仍有效，没有改动其任何受封存源码。真实19阶段A输出目录尚未创建，预算仍 **2244.407/7200秒**；这次没有消耗真实实验预算，也没有新训练或比赛。

## Progress Notes：阶段A完成与阶段B准备（2026-11-06）

用户运行 `run_distill19_analysis.ps1`，实际 **18.593秒**、completed/退出0/无残留进程。全部输入/脚本/逐例产物SHA256核验通过，D/S在 **30697个状态**（256训练父+628验证父+8426训练子+21387验证子）上STE/scalar/AVX2逐位一致，无静态上限命中，验证搜索Top-1复现18。分析记录及审阅结果保存在 `analysis_initial/report.json` / `stage_a_review.json`。

训练256条有235条primary、21条terminal-tactical；扩展D4计数：parent-parent=0、parent-child=0、child-parent=0、child-child=3，涉及3个完整来源对局。后续排除这3局，clean池 **253个父局面（黑127/白126）**，其中232个primary/8293子局面。原验证28局/628条和全部候选身份均不变；原诊断保留全256条与clean253条两种口径，未静默删样本。

| D组整数路径 | 全256训练父局面 | clean253训练父局面 | 628验证父局面 |
|---|---:|---:|---:|
| Huber / 完整训练池常数 | 0.959363 | 0.961985 | 0.964153 |
| MSE / 完整训练池常数 | 1.051258 | 1.051218 | 1.031833 |
| 预测标准差 / 教师标准差 | 0.126304 | 0.125425 | 0.130339 |
| primary候选pooled centered MSE比 | 1.001489 | 1.001518 | 1.000549 |
| primary候选macro逆序率（并列计0.5） | 0.493305 | 0.493199 | 0.494064 |

原训练允许池中的原局面也拟合较弱，且训练侧局部候选差分接近零差分参照；排除3个重合来源后结论没有实质变化。这不支持将失败**仅**归因于候选子局面分布错位。数据桶难度不同、D4随机训练不保证每条原始朝向都被见到，因此仍不作唯一因果解释。现有QAT master的float旁路依然不能当独立float控制，下一步按预声明进入B，而非直接加入排序损失或扩网。

阶段B已从clean池按冻结顺序封存 **128条/128局、stm各64**，G=128、关闭D4、只用原父局面。manifest SHA256为 `dc83d0010e780acc64e19f6b6a566f4a34b3bc80204891a8f78d954aba836966`；两组共同initial master SHA256 `2078ec51ae18f0cc718e1123221a50ad12a04f5d5f8a11d5835903a1fe836c13`、batch schedule SHA256 `b3f60210a19d4da387bce22fc30b65fb9b8015f1b1c829d2d1c84d8cd88603ae`。同128条训练均值常数为-3498.9453125，Huber常数-107.3870968；两种本集基线分别MSE **4036.751122** / Huber **38.241565**，与阶段A完整训练池分母明确分开。

B实现的短集成测试已覆盖两组独立更新、相同初始化/批次、不同前向、共同投影、最后一步checkpoint、样本曝光、summary可比性/错误拒绝；smoke fixture显式标记，不能当正式B完成。CTest **20/20**通过；未修改18冻结源码。实际float/qat输出目录尚不存在，真实B训练未启动；预算 **2263.000/7200秒**、剩余 **4937.000秒**。下一步用户运行 `run_distill19_fit.ps1 -Stage fit-float`，再运行fit-qat；不要重跑A或prepare-b。
