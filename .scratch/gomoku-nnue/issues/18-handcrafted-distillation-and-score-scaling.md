# 18: 蒸馏手工评估与修正分值缩放，以走子 Top-1 突破为第一门控

**What to build:** 在既有数据、网络容量和搜索配置下，使 NNUE 学会区分候选着法。主方案回归 C++ 手工静态评估；配一组使用相同新损失与输出缩放的搜索分对照，区分「教师标签来源」和「分值表达链」的影响。先验证最终导出权重的 `move_correlation` Top-1 明显突破，再允许 20 局筛选；M5 正式验收仍由工单 16 承担。

**Status:** draft（详细设计草案；尚未实现、未启动训练或比赛）
**Category:** training / diagnosis
**Blocked by:** 无前置未交付代码；工单 17 已关闭。阶段 A 的度量与尺度审计通过后才能启动训练。
**Downstream:** 合格候选 → 工单 16；工单 10 / 12 仍未完成，工单 11 的 incumbent 依赖不解除。

本工单中的新 CLI、schema 和权重格式均为**待实现接口**，不能当成现有命令运行。实施范围不包含扩网、新自我对弈数据、正式 200 局或 M6。

## 1. 已有证据与本次要检验的假设

冻结验证集有 28 局 / 628 条记录。旧 `move_correlation` 报告：手工静态评估 Top-1 `269/628 = 42.83%`，四组 NNUE 为 `0.2%–1.1%`；工单 17 三组优化对照均未通过原损失阈值。报告见 `data/m5/move_correlation_{handcrafted,v1,small,large,opt_l}.json`。

这些结果支持优先检验目标与尺度，但还不能证明「标签无信息」或「根因唯一确定」。尤其需要先补齐三项事实：

1. `rank_of()` 用「严格更好数量」作为排名，并列第一全部算命中。所有候选同分的常量模型会得到旧 Top-1 **100%**。42.83% 仅是旧口径手工参照，不能直接当新门控阈值或理论上限。
2. 现有诊断枚举全部空位，没有 Renju 禁手和邻域候选过滤；`nnue_probe board/handcrafted` 使用 Freestyle 棋盘做静态评估。旧结果不等于合法搜索候选上的走子正确率。
3. `score_scale=400` 是 `tanh` 损失的分母。`NnueNetwork::load()` 虽读取它，`inferSum()` 并不据此还原输出。现有网络的 `sum(a2*W2)/QB` 在 `15 × 64` 权重约束下，单头可变项绝对值至多 960，两个局面的差至多 1920；`b2` 只能整体平移，无法恢复手工评估中 15000 级冲四、500000 级五连的动态范围。只改 `--score-scale`、导出元数据或输出 bias 不足以修正尺度。

预先声明的假设：**保留幅度差异的监督 + 可表达的输出尺度，能使最终整数 NNUE 的候选排序明显偏离随机；手工静态标签可能比已截断为 int16 的搜索标签更容易学会这一关系。** 这是待实验验证的假设。蒸馏只能先争取逼近教师，不能保证超过教师或通过 M5。

## 2. 标签、视角与数据身份

### 2.1 教师定义

- 标签 `H(B, stm)` 必须来自**同一版本 C++ `Evaluator`，未挂载 network**，对输入棋盘 `init()` 后 `evaluate()`；不在 Python 重写棋形表，不使用搜索结果冒充静态评估。
- 保留现有公式与整数截断：`H = floor(1.15 * myScore) - oppScore`（本实现 `myScore` 非负），单位为手工评估原始引擎分；标签用 int32，不能回写 binpack 的 int16 `score`。
- 分值始终属于**输入局面的行棋方**。原局面训练 `E(B,stm)≈H(B,stm)`；候选 `m` 的当前方价值为 `-E(B+m,1-stm)`。不能简单将 `H(B,Black)` 取负作为 White 标签，1.15 系数使两者不严格反对称。
- D4 只变换特征，`stm` 与教师分不变；不交换颜色。训练标签只覆盖当前记录的 `stm`，双头分别报告覆盖和误差。
- 主训练池只使用已有清单内的原始记录；不在本轮生成候选子局面训练数据。若原局面拟合成功而子局面排序仍失败，将分布差异记为下一工单假设，不能在当前实验中途补数据。

### 2.2 Sidecar 缓存

新增 `trainer/teacher_labels.py build`，按每个源文件的**原始 record index**写出 little-endian int32 教师标签数组与 JSON manifest；不更改 64B binpack、`record.move` 或对局边界。每个槽位与源记录一一对应，另存有效性 mask，禁止跳过记录导致索引错位。

Manifest 至少包含：schema、源 binpack SHA256 / 记录数、标签文件 SHA256 / dtype / 字节序 / 数量、有效性 mask 身份、教师定义版本、`nnue_probe.exe` SHA256、`eval.cpp` SHA256、代码 revision / dirty diff 身份、视角与单位、生成参数、完整性状态。路径不作为唯一身份。验证文件允许生成标签供验收，但不得进入训练侧尺度统计、常数拟合、导出校准。

复用工单 14 的 `fresh_200k_manifest.json` 所列训练文件与允许对局，原池应为 204674 条；有效性检查如发现源记录已经终局，应计数并过滤训练记录，两组使用完全相同的过滤池并记录实际数量。非终局的 ±32767 搜索分记录仍保留，不将其误判为终局。损坏棋盘、重叠占位、非法 `stm`、缓存长度或身份不符必须失败，不能悄悄忽略。

### 2.3 重冻结契约，不重新抽样

保留旧 `frozen_validation.json` 和旧训练清单；派生 `data/m5/distill18/frozen_validation.json`，增加 `parent_frozen_sha256`，**原封保留 28 个验证对局与 628 个 record ID**。重冻结指标签与度量契约升级，不允许换 seed 或重抽更容易的验证局面。

新 schema 将三种身份分开：`split_identity`、`target_contract`、`move_metric_contract`。标签契约包含 D/S 两组定义、sidecar 身份、有效性 mask、损失、所有尺度、教师版本；度量契约包含候选集合、并列规则、逐局面清单、终局规则。重新生成绑定新冻结文件 SHA256 的训练清单，重新执行 D4 完整对局排除；不能只编辑旧 JSON 的 hash 字段。

常数基线分别在**完整有效训练清单**上拟合，再应用于相应验证目标：均值用于 MSE，另拟合最小化 Huber 的常数用于 Huber。不能继续只用旧源文件训练侧的常数代表 20 万记录训练池。保留同一集合上的 WDL 先验供历史报告，不参与本轮损失。

## 3. 统一分值契约：损失归一化与推理还原各司其职

### 3.1 主损失

两组均禁用 `tanh`，使用线性分值空间的 `SmoothL1Loss(beta=1)`：

```text
D 蒸馏组：label = H(B,stm)
S 尺度对照：label = 原 binpack 的 int16 搜索分（保留其截断语义）
loss = mean(smooth_l1((E_engine - label) / loss_normalizer, beta=1))
wdl_weight = 0；quant_reg = 0；score_clip = 0
```

`loss_normalizer` 只调节训练残差的数值范围，不决定引擎输出单位。取下节确定的 `output_gain`，两组完全相同；二者在契约里仍分别命名，防止以后误用。标签不做中心化、不做额外裁剪；记录 MAE/RMSE（原始分）、归一化 MSE / Huber、训练侧常数基线与误差比。新指标不得与旧 `val_mse=0.685776` 或阈值 `0.651487` 直接比较。

### 3.2 输出缩放的确定与范围

在阶段 A、训练开始前，用共同训练池的 `max(abs(H), abs(search_score))` 确定一个固定正整数 `G`：从 `{1,2,4,...,1024}` 取最小值，使所有训练标签绝对值 `≤ 0.75 × 960 × G`。该保守条件为可变项预留余量，不证明网络能拟合全部函数。若没有可用 G，阶段 A 失败，记录范围问题；不能根据验证表现调 G、截标签或扩展格式上限。

两组使用同一个 G，`loss_normalizer=G`；分黑白、ply 与搜索饱和桶报告标签分位数、max、可表达范围和预测范围。仅扩大损失分母不能补回标签已丢失的信息；对旧预测乘正数也不会改变排序，G 的作用是修复**训练与量化中的表达范围/分辨率**。

### 3.3 新权重推理公式与版本

本方案改变整数前向语义，必须版本化。保留 `GMNNUE01/version=1` 的读取与逐位行为；新导出使用 **`GMNNUE02/version=2`**，张量维度、权重类型和 body 布局不变。

Header 仍为 64B，offset 0 magic 改为 `GMNNUE02`，offset 8 version 改为 2；offset 52..55 为 little-endian u32 `output_gain`，合法值为上述集合；56..63 保留零。offset 40 的 `score_scale` 保留原 legacy 字段、默认 400，**不参与 v2 线性损失或推理还原**；新标签契约在 checkpoint / 报告中显式保存。body checksum 算法不变，文件身份 SHA256 覆盖完整 header 和 body。magic/version 必须配对，未知版本、非法 G、非零剩余保留字节均拒绝加载；旧 loader 对新格式明确拒绝。

沿用现有隐藏层运算，令 `y = sum_k(a2[k]*W2[stm,k])`，新最终整数输出：

```text
E_unclamped = round_div(G * y, QB) + G * b2[stm]
E_engine = clamp(E_unclamped, -900000, +900000)
```

乘法及求和用 int64；舍入仍为 half-away-from-zero。**先乘 G 再做最终 round_div**，不能写成 `G * round_div(y,QB)`，否则会人为制造 G 分的粗台阶。b2 为内部单位，校准时 `delta_engine/G` 才是 b2 的修正量。v1 隐含 G=1，继续使用旧路径且不新增截断。

v2 的 ±900000 静态上限留在 `search.cpp` 当前 `kMateBound=999000` 以下，避免被 TT mate 分修正或「已找到杀」逻辑误认。搜索仍负责终局/VCF mate 分，不将静态五连棋形当作 mate。Python float、STE、整数参考及 C++ 都实现相同单位与上限；float 保持连续除法，STE 与整数端必须逐位一致。记录达到上限的次数，冻结非终局候选若发生输出截断，候选不得晋级。

## 4. 第一门控：可审计的 move_correlation Top-1

### 4.1 冻结评估案例

新增 `nnue_probe candidates --rule renju --radius 1`（批量逐行输入、逐行结构化输出），直接复用 C++ `generateCandidates()`、`Board::isPlayable()`，返回按 cell index 排序的候选 ID、走后结果与输入记录 ID。禁止在 Python 重新实现禁手。排序不使用教师启发分，也不随被评模型改变。

从原 628 条生成不可变 `move_cases.jsonl`：父局面必须 ongoing，教师 `record.move` 必须属于合法邻域候选。候选包含立即获胜着法的局面归入单独 `terminal/tactical` 桶；**第一门控只用所有候选均为非终局且候选数 ≥2 的局面**，避免共用终局判断给模型制造不劳而获的提升。所有分桶、record IDs、game IDs、候选列表、排除原因与数量在跑候选前冻结，合计必须覆盖 628 条。

教师 move 不在合法候选、畸形棋盘、终局状态矛盾等属于数据/工具故障，不能静默从分母删除。合理的终局/单候选分桶不改变验证集身份，但必须报告覆盖率。主集要求至少 **200 条 / 20 局**，否则第一门控不可用，停止并报告，不自行换验证集或降低要求。

对主集所有候选子局面，按下一行棋方分别调用 handcrafted / NNUE，并取负用于当前方排序。`move_correlation` 也输出旧 `all_empty + optimistic_rank` 的桥接指标，允许在原 628 条重现历史结果；桥接值只能解释口径变化，不能用来过门控。

### 4.2 并列、随机与聚合

令 `b` 为严格优于教师 move 的候选数，`t` 为与教师 move 同分的候选数（含自身），`n` 为全部候选数：

```text
top1_credit = 1/t if b == 0 else 0
topk_credit = min(1, max(0, k-b)/t)
midrank = b + (t-1)/2
random_topk = min(k,n)/n
```

主指标 **`top1_tie_adjusted`** 为逐局面 `top1_credit` 的算术均值。另报告唯一第一命中率、旧 optimistic Top-1、Top-3/5、midrank 百分位、最大同分组和教师同分组大小、全部同分比例、黑白/ply 分桶、game-macro Top-1。常量模型必须回到 `mean(1/n)` 随机基线；随机基线是逐条均值，不能继续用 `1/mean(n)`。

保存每例的教师 move 分、b/t/n、credit 与输入身份。输出整数计数/credit 总和、分母和未舍入比率；显示用四舍五入值不能参与判定。探针返回长度、顺序、字段或分值解析不符必须失败。主门控只能用**最终导出文件经 C++ AVX2 实际推理**的结果，不用 float checkpoint 的排名代替。

### 4.3 预先声明通过规则

在相同 `move_cases` 上重测手工教师与四个历史 NNUE，令 `H1` 为教师新口径 Top-1，`B1` 为历史 NNUE 的最大值，`R1=mean(1/n)`。手工参照不是理论上限；教师必须满足 `H1 ≥ 0.20` 且 `H1-R1 ≥ 0.10`，否则度量敏感性不足，阶段 A 停止。

候选的第一门控须**同时**满足：

1. `T1 ≥ 0.20`，`T1-B1 ≥ 0.10`，`T1 ≥ 0.50*H1`（百分比差为百分点，不是相对百分比）。
2. 按完整对局成簇 bootstrap（固定 seed 20261118、10000 次），以候选减去**每次重采样中表现最好的历史 NNUE**的 Top-1 为差值，其 percentile 95% 区间下界严格 `>0`。不能把同局多条位置视为独立试验；报告重复使用旧验证集的局限，不宣称独立泛化证明。
3. 度量/数据身份完整；无非有限值、静默遗漏、非法候选；跨语言数值检查有效；主集 v2 输出截断次数为 0。

这是本轮的**最低突破标准**，不是对未来效果的承诺。20%要求明显脱离旧近随机区间，10个百分点要求超过同口径历史参照，教师比例要求恢复有意义的一部分静态走子信号；它们是本轮工程决策阈值，不是由旧42.8%推导出的统计保证。阈值在任何新候选训练之前写入 `experiment_plan.json` 并封存 SHA256。不能为某组失败事后改口径、改阈值或挑最好 epoch。归一化验证损失改善、棋谱观感或旧 Top-1 上升都不替代该门控。

## 5. 分阶段实验与停止规则

### 阶段 A：工具、冻结、基线及数值预检

实现标签/尺度契约、v2 推理、候选清单与新度量。完成新冻结、D4 排除、D/S 常数基线、教师与旧模型新口径重测、G 的训练侧确定；交付 `experiment_plan.json`。旧结果用 schema 1 保留，新报告用 schema 2，不覆盖旧 JSON。

从有效训练侧固定抽 64 条，进行 float/STE 各最多 1000 步的小样本拟合；同一 seed、同一 D 标签，报告标签归一化分布、两头误差、梯度、激活饱和及训练曲线。STE 训练集归一化 MSE / 同样本常数 MSE 必须 `≤0.25`；基线为零的退化样本需在抽样时依训练规则重取，并记录原因。该检查只防止在不可训练链上消耗整轮训练，不能当泛化或棋力验收。未通过则停止，解释优化/量化问题，不静默加步骤调参。

数值预检：v1 历史 fixture 结果不变；v2 人工正负/半数舍入/溢出边界样本与真实子局面上，Python STE、Python 整数参考、C++ scalar/AVX2 逐位一致。特别验证 G=1、G>1 的舍入次序、非零 b2 单位、静态上限、错误 header 拒绝。

### 阶段 B：两组等算力训练，用户启动

| 项 | D：手工蒸馏（主方案） | S：搜索分尺度对照 |
|---|---|---|
| 标签 | C++ H(B,stm) int32 sidecar | 原 int16 search score |
| 损失 / 输出 | 共同 Huber / 同一 G / v2 | 同左 |
| 数据 / 验证 | 新冻结 + 相同有效训练清单 | 同左 |
| 优化 | seed 42、batch 1024、AdamW、lr 0.1、hidden_lr 0.001 | 同左 |
| 预算口径 | epochs 300、max_steps 1600、checkpoint_policy last | 同左 |
| 其他 | QAT、wdl_weight 0、quant_reg 0、score_clip 0、D4 | 同左 |

从同 seed 重建共同初始化，保持优化器其余设置不变。每组记录实际 optimizer steps、samples exposed、epoch、数据/冻结/标签/G/源码身份。未达到 1600 步、超时或被中断的组记 incomplete，不参与正式候选比较。

此设计中 D/S 仅标签来源不同，可比较蒸馏与搜索分；它们与旧模型同时改变损失和输出缩放，**不能**把 D 相对旧模型的变化归结为某一单独因素。若需拆分损失和 G 的独立收益，另立后续实验，本轮不加第三组。

### 阶段 C：最终导出、完整排序验收

分别输出 `.pt`、history、v2 `.nnue`、export report、fidelity report、完整主集 move-correlation JSON 和逐例 JSONL。导出/偏置校准只取训练清单，排除验证对局及 D4 泄漏局；本轮 QAT 原始整数与 STE 应完全一致，默认关闭无必要的 bias 校准。若启用，必须量化到 b2 内部单位并保存前后变化，不能从验证集拟合仿射系数。

`diagnose.py fidelity` 新接口支持完整冻结评估（如 `--all-validation`），不能只抽 64 条代表 628 条。分别报告训练原 float、训练 STE、原始整数、最终整数，按 D/S 正确标签契约计算损失与常数基线。**完整主集所有候选子局面**做最终整数 Python / scalar / AVX2 一致性检查；主集逐例 float/STE/final 的 Top-1 用于定位量化后排序损失。对 float 与整数只报告实测误差，不沿用旧「≤2 原始分」容差，因为输出单位的量化行为已变化。

最后以 C++ 最终 v2 文件计算第一门控并在 `summary.json` 中作出通过/失败决定。若 D/S 都通过，以未舍入最终 `T1` 最大者选一个；相同则优先 D，再相同用固定组名顺序。最多一个候选进入对打。

### 阶段 D：有条件 20 局换色筛选

仅第一门控通过后才生成新筛选开局：seed **20261118**，10 对 / 20 局，与历史筛选和正式 `formal_openings.json` 做内容去重核验后冻结；冲突则失败并报告，不能看到对打结果后换 seed 重试。保持 Renju、depth 2、VCF 开、64MB TT，双方只有评估器不同。

复用 `match_runner.py --games 20 --paired-openings --strict-gate --min-win-rate 0.65 --verify-backend`。有效完整比赛、无异常、后端/可执行文件/最终 v2 权重身份一致，未舍入得分率严格 **>65%** 才可交工单 16。恰好 65% 失败，退出码 2 保留为真实门控失败。第一门控失败则不生成/消耗筛选开局；正式 100 对开局在本工单内始终不运行。

### 预算与执行交接

沿用 `data/m5/budget.json`：当前 `1547.330 / 7200` 秒、剩余约 `5652.670` 秒，实际执行前重新读取账本。后续数据标注、过拟合实验、训练、全候选保真度与批量对打均通过 `budget_run.py` 纳入同一账本；不重置或复制预算绕过上限。

预留上限：A 标注/冻结/历史基线 900 秒 + 小样本拟合 300 秒；B 两组各 600 秒；C 每组 600 秒；D 单次 300 秒，总预留 **3900 秒**。顺序运行，按实际墙钟扣费；剩余预算不足以容纳下一任务上限时停止交接，不自行增加预算。工具要支持阶段断点、完成状态和拒绝覆盖，超时产物只用于诊断。

预计超过 5 分钟或无法可靠估时的任务由用户在已激活项目 `.venv` 的终端启动；agent 仅执行明确可限时的短检查。实施时交付 `tools/M5.md` 中与实际 parser 一致的完整 PowerShell 命令，用户命令用 `python`，agent 用 `.venv\Scripts\python.exe`。本草案不启动这些任务。

### 产物与实施接口

统一使用 `data/m5/distill18/`；至少交付 `teacher_manifest.json`、`target_d.json` / `target_s.json`、`frozen_validation.json`、`train_manifest.json`、`move_cases.jsonl`、`metric_contract.json`、`historical_baselines.json`、`experiment_plan.json`、`preflight/report.json`、`d/history.json` / `s/history.json`、两组 `export_report.json` / `fidelity.json` / `move_correlation.json` / `per_case.jsonl`、`summary.json`。checkpoint 为 `weights/m5_distill18_{d,s}.pt`，最终权重为各组目录的 `candidate.nnue`；运行目录另用 `*_run/` 保存 budget_run 记录，正式比赛文件不移入此处。

`target_{d,s}.json` 至少包含 `schema/name/label_source/perspective/units/loss/output_gain/loss_normalizer/static_limit` 及源文件/sidecar/共同训练池身份。`experiment_plan.json` 同时绑定两份 target contract、新冻结、清单、metric contract、所有超参、候选选择规则、阈值与 bootstrap 配置。`summary.json` 记录这些输入的 SHA256 与每项门控布尔值、失败原因、实际数量和预算用量；缺少任何一组完整产物时 `experiment_complete=false`。

以下为**待实现训练接口模板**，实现完成并通过阶段 A 后才可复制运行。`target_d.json` 中应已经封存计算出的 G 和标签缓存身份；S 组仅替换 target、输出路径和运行目录，其他参数逐项相同：

```powershell
python tools\budget_run.py --ledger data\m5\budget.json --seconds 600 --out-dir data\m5\distill18\d_train_run -- python trainer\train.py --data data\raw\bench-d4\train.binpack --frozen-validation data\m5\distill18\frozen_validation.json --train-manifest data\m5\distill18\train_manifest.json --target-contract data\m5\distill18\target_d.json --output weights\m5_distill18_d.pt --history data\m5\distill18\d\history.json --epochs 300 --batch-size 1024 --max-steps 1600 --checkpoint-policy last --quant-forward --quant-reg 0 --wdl-weight 0 --score-clip 0 --lr 0.1 --hidden-lr 0.001 --seed 42
```

`--target-contract` 存在时，label/loss/G 以契约为准；CLI 显式给出的非零 `score_clip/wdl_weight/quant_reg` 或其他目标覆盖必须拒绝。export/diagnose 从 checkpoint 读取并核验同一契约，不允许用户在导出阶段另传一个 G 重标分值。

## 6. 具体代码修改清单与接缝

以下均为**计划修改**，顺序反映依赖；不应只改 `train.py` 或对旧 `.nnue` 做外部乘法。

| 文件 | 具体修改 |
|---|---|
| `engine/src/nnue_probe.cpp` | 扩展批量标签/候选输出：复用无 network 的 `Evaluator` 标注 int32；新增 Renju/radius=1 候选与走后结果接口，保留旧 handcrafted 输入输出兼容；严格校验参数/逐行标识。 |
| **新增** `trainer/teacher_labels.py` | 生成/读取/校验 record-index int32 sidecar 和 mask；教师/源文件/源码身份；原子写入、超时、不完整产物拒用；标签与 G 范围统计。 |
| **新增** `trainer/targets.py` | 集中定义 legacy / handcrafted-linear / search-linear 契约，目标解析、训练侧尺度选择、归一化损失/基线与 schema；训练、冻结和诊断共用，杜绝三份目标公式漂移。 |
| `trainer/dataset.py` | `BinpackDataset.__init__/__getitem__/iter_samples` 按原 record index 取标签/mask；保留原 score/WDL 来源；多文件各自绑定缓存，D4 后标签不变；关闭 mmap 资源。 |
| `trainer/validation_freeze.py` | 增加从旧冻结派生契约的入口；保留 split，绑定 sidecar/共同训练清单，重建 D4 排除和有效性计数；D/S 基线从完整共同训练池拟合；schema 兼容旧文件，拒绝目标身份不匹配。 |
| `trainer/nnue.py` | `ModelConfig` 增加版本/G/目标配置；`forward` 实现 v2 先乘后舍入、b2 单位和静态上限；分离 legacy `mixed_losses` 与新 Huber 目标；旧 checkpoint 缺字段默认 v1。 |
| `trainer/weights_format.py` | `pack/read_weights` 支持 v1/v2、header offset52 G、元信息校验；`infer_int`、numpy 和纯 Python 两条参考路径落实同一 v2 公式；不丢弃重载权重的 G。 |
| `engine/src/nnue.h`、`engine/src/nnue.cpp` | 网络保存 format version/G；loader 双版本校验；`inferSum` v2 int64 先乘后舍入和静态上限，scalar/AVX2 共用最终输出逻辑；v1 结果不变。 |
| `trainer/train.py` | `parse_args/build_datasets/run_training/evaluate` 接入目标契约与 sidecar，训练前校验 freeze/G；记录新损失、原始分误差、两头分布/上限次数和完整身份；保留 1600 步与 last checkpoint 行为。新增 `--target-contract`，与 legacy `--score-clip` 等冲突时明确报错。 |
| `trainer/export.py` | `load_model/int_evals/main` 贯穿格式/G；`pack_weights` 不再静默使用全局默认；严格训练侧校准、b2 的 engine/internal 单位转换；报告原训练前向到最终文件的误差与格式身份。 |
| `trainer/diagnose.py` | `fidelity/metrics/samples_at` 使用共同 targets；增加完整验证模式，基线从实际训练 manifest 求取；贯通 sidecar/G/格式，报告训练原 float/STE/raw/final 四路及候选子局面保真度。 |
| `trainer/move_correlation.py` | 替换主门控候选源与并列算法；冻结 cases、schema2、逐例结果、精确随机基线、game bootstrap、历史比较/阈值判定；保留 legacy 桥接模式；probe 身份、长度/顺序严格核验，子进程超时可配置且受外层硬限。 |
| **新增** `tools/distill18_summary.py` | 读取冻结 plan 与两组完整产物，核对等算力/身份/逐位结果，计算第一门控、唯一候选选择及 `eligible_for_screening`；区分完整阴性与 incomplete。只汇总，不自动启动比赛。 |
| `engine/tests/training_tests.py` | 检验新损失保留中间分值差异、sidecar 按原 index 对齐、D4/两头、训练配置保存/载入及短步数流程；保留 legacy 回归。 |
| `engine/tests/validation_freeze_tests.py` | 新冻结保留28局/628条、旧文件不变、跨文件D4整局排除、完整训练池常数、标签缓存/G/manifest身份不符拒绝。 |
| `engine/tests/nnue_tests.cpp`、`engine/tests/nnue_parity_tests.py`、`engine/tests/nnue_backend_tests.py` | v1 兼容 + v2 header 错误 + 非单位 G、非零 bias、正负半数、先乘后舍入、静态界限及完整/增量 scalar/AVX2 数值行为。 |
| `engine/tests/move_correlation_tests.py` | 常量模型回到随机、多重并列 Top-k、`mean(1/n)`、negamax 视角、合法邻域/黑禁手、立即获胜分桶、缺失输出失败；引擎真实子进程集成 + 记录清单/门控边界与按局 bootstrap 可复现。 |
| **新增** `engine/tests/distillation_tests.py` | 小型 sidecar→新冻结→短训练→v2导出→C++子局面排序→summary 的完整可限时烟测；损坏缓存和身份失败路径。不要在 CTest 中运行正式1600步或真实20局实验。 |
| `engine/CMakeLists.txt` | 为新增集成测试注册 CTest，并给 move-correlation 测试传真实 probe 路径；设置短测试 TIMEOUT。 |
| `trainer/README.md`、`tools/M5.md`、`README.md`、`.scratch/gomoku-nnue/spec.md` | 更新目标/输出单位/v2格式、重冻结流程、第一门控与长任务交接命令，工单17已关闭及18状态；解释旧42.8%口径与结论边界。 |

`engine/src/eval.cpp` 的棋形表/1.15 系数、`engine/src/search.cpp` 的算法/mate 常量、`engine/src/datagen.cpp`、binpack布局和 `tools/match_runner.py` 门控算法均保持原契约，本轮只调用。若实现发现必须改这些行为，应先记录设计变更，不能与本轮训练结果混在一起。协议已有的后端/权重核验继续使用；format/G 由权重 header、probe 与产物 SHA256 核验，无需添加新的运行时调分开关。

## 7. 验收与交付状态

- [ ] 教师 sidecar/目标契约/共同训练池已冻结，原28局628条成员不变，缓存与 D4 身份可复核。
- [ ] G 由训练侧规则确定；v1兼容与v2跨语言/增量/舍入/静态界限测试通过，短样本训练预检通过。
- [ ] 度量 schema2 已冻结；常量/并列/合法候选测试通过，历史模型与手工参照新口径齐全且满足敏感性和覆盖门槛。
- [ ] D/S 两组完成相同1600步，last checkpoint，独立导出、完整冻结评估和主集所有子局面保真度，身份齐全。
- [ ] `summary.json` 给出未舍入第一门控、cluster区间、等算力核验、两组差异及唯一候选或明确阴性结论。
- [ ] 仅有合格候选时执行一次20局换色筛选；不通过或无合格候选均保留阴性，不重复挑选开局。
- [ ] 预算/运行完成状态完整，CTest原18项与新增检查全绿；文档交接命令与parser一致。

工单可按「实验完整完成、结果为阴性」关闭；必须将 `implementation_complete`、`experiment_complete`、`top1_gate_passed`、`screening_passed`、`eligible_for_ticket16` 分字段记录。实现完成但用户未跑实验时保持 in-progress；阶段 A 预检失败可形成诊断结论，但不能勾选未执行阶段。只有20局筛选严格通过才解除工单16的候选阻塞；第一门控本身不关闭10/12、不产生incumbent、不证明M5达标。

若 D 明显优于 S：支持在该配置下静态教师更有效；若两组都突破：支持共同尺度/目标修复有价值，但不能单独归因；若损失下降而 Top-1 不变：保留训练/候选分布差异、特征/优化限制等解释；若两组都失败：先核对预检与数值证据，再设计下一工单。禁止自动扩网、补数据、延长预算或直接启动正式比赛。
