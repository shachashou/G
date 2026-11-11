# 11: 增量累加器与推理加速

**What to build:** 用户使用新模型对弈和采集时，引擎通过第一层增量累加器减少重复计算，并提供可复查的推理和实际搜索性能数据，同时保持与全量参照一致。

**Blocked by:** 07 — 使用新模型实际对弈。

**Status:** done

- [x] 对450 → 1024 → 64 → 32 → 1网络实现第一层稀疏增量更新：落子增加对应向量，撤销减去对应向量。
- [x] 后续隐藏层和输出层正常前向；不把整网评估描述为只有向量加减。
- [x] 内部累加采用足够宽且可逆的表示或明确状态保存，避免饱和后减法无法还原。
- [x] 新局、局面全量同步、连续落子撤销、搜索分支回退和终局路径保持棋盘与累加器一致；非法操作不改变任何相关状态。
- [x] 保留全量标量参照，覆盖完整对局、多种子随机合法序列、深层撤销与边界局面，增量输出符合冻结数值契约。
- [x] 在现有目标平台接入适用的 AVX2 加速，逐局面核验标量与优化路径一致；不以近似值隐藏错误。
- [x] 在真实引擎选点和采集路径实际启用优化，提供可确认的运行配置与后端身份。
- [x] 在相同模型和资源配置下报告推理耗时、搜索吞吐及配对开局实际表现，区分单次推理提速与整机棋力提升。
- [x] 性能结论基于实测，预先记录测量条件，不编造固定加速倍数；出现数值或性能回退时可明确选择参照路径。

## Verification

在模型推理文件边界、落子撤销公共行为及真实引擎搜索边界对拍。只因本工单引入优化而增加有针对性的随机序列与性能测量，避免重复运行无关训练实验。

### 实施与验收记录（2026-11-11）

**第一层增量与可逆性**

- `engine/src/value_net.{h,cpp}` 新增 `ValueAccumulatorMode`（`Incremental` / `FullRebuild`）和 `ValueBackend`（`Avx2` / `Scalar`）。`updateAccumulator(board, move, to, acc)` 在棋盘变更前验证：索引范围、目标必须是棋子或空格、落子点必须为空、撤销点必须有子；任一条件不满足即在修改前抛 `ValueNetError`，累加器保持原值。
- 累加器继续使用 `std::array<int64_t, 1024>`，全程不饱和。上界为 `|int32 bias| + 225 × 32768 < 2^32`，远小于 int64；落子加 `W0[feature]`、撤销减同一行，任意顺序均精确可逆。AVX2 更新把 8 个 int16 权重扩展为两组 4×int64 后做 `vpaddq`/`vpsubq`，与标量路径逐位相同。
- 64 → 32 → 1 的前向没有拆成向量加减。`finishForward()` 由标量和 AVX2 共用，避免两条路径各写一份后漂移。AVX2 的 1024→64 层使用 `vpmuldq`（有符号 32×32→64，精确）累计；一次处理 32 个输出以减少 W1 重复读取。没有用近似、饱和或浮点替代整数契约。
- `Evaluator` 新增 `ValueAccumulator valueAcc_`：`init()` 在搜索根做一次全量重建，`update()` 在 `board.place`/`board.undo` 前同步加减。搜索分支回退走同一 `Evaluator::update(..., Cell::Empty)`；新局和 `BOARD` 同步由新的 `Evaluator` 全量重建。终局与已证明 VCF 分数仍由 `Search` 在调用评估前决定，网络值不能覆盖。
- `reference` 模式完全绕过缓存：每次评估执行 `rebuildAccumulator + inferAccumulatorScalar`，保留工单 07 的全量标量参照。

**运行配置与身份**

- 新增 `--value-backend avx2|scalar|reference`，默认 `avx2`（增量 + AVX2）。`scalar` 是增量 + 标量前向；`reference` 是全量重建 + 标量前向。Gomocup `ABOUT` 输出 `eval_engine=gmvalue1-incremental-avx2`、`value_model_backend`、`value_model_accumulator`、`value_model_compute`；执行器 `about`/`think` 输出对应字段，`think` 还回显 `value_model_backend`。GUI 的评估方式标签读取实际 `backend` 字段，不再固定写“标量全量参照”。
- 真实选点路径（`Protocol`、`ExecutorProtocol`、自对弈/采集工具启动的引擎实例）默认使用增量 AVX2；测评、自对弈和 GUI 无需额外参数。`--value-backend reference` 可在同一模型、同一资源下回退对拍。

**探针与测试**

- `value_probe` 增加 `--backend`、`walk` 和 `bench` 模式。`walk` 用真实 `load/place/undo/state` 会话语义，逐步输出增量 z、全量参照 z 和 `match`；`bench` 区分 `full` 与 `incremental`。
- `value_net_tests` 新增：4 个种子 × 最多 225 手随机合法对局及全撤销；BOARD 同步的中局开局；非法落子/撤销/越界/非棋子目标不改变累加器；`b0` 接近 int32 极值、`W0` 全为 ±32768/32767 时多子加减精确还原；`Evaluator` 与全量参照逐步一致；avx2 / scalar / reference 三种路径在同一局面固定深度搜索的着法、分数、节点数完全相同；后端名称解析。
- `value_parity_tests` 新增标量与 AVX2 的 `walk` 对拍：38 个局面（合成 + 实际样本池采样），每局面 12 次落子再全部撤销；每一步 `z == reference`，两种后端逐位相同，整段撤销后回到初始 z；并打印单次推理实测条件与数值。
- 新增 `value_backend_tests`：真实执行器 `think` 在三个后端下返回同一着法/分数/节点数，测固定深度 4 的搜索吞吐；再用 `match_runner.py` 跑 AVX2 与 reference 的配对开局，核对每对走法序列、胜方颜色与终局。
- `value_engine_blackbox` 更新后端身份断言，并新增三个后端的固定深度对拍。
- 全量 CTest **39/39 通过，无跳过**（最终一轮 85.73 s）；`git diff --check` 通过。

**实测条件**

- 机器：Intel Core i7-14700KF（20 核 / 28 线程），Windows 11 10.0.26300，Release 构建（cmake 3.31.10，Zig 0.17.1，venv 工具链）。
- 模型：`weights/value-demo/model.gmvalue`，SHA-256 `232FA16629814B4FC4EC9D052C9B0164EC699E73C9136D966E0F59ACE9165885`。
- 推理：单线程，每个配置 20000 次，取总耗时均值；计时前有 warmup。搜索：同一 12 子局面，固定深度 4，关闭 VCF，每个后端 5 个独立进程取中位数；节点数 8113，三后端相同。
- 原始数据：`build/acceptance/ticket11/probe_bench.json`、`search_bench.json`、`paired-openings.json`、`ctest-final.log`（构建产物目录，可重新生成）。

**单次推理**

| 局面子数 | 全量标量参照 | 增量标量 | 增量 AVX2 |
|---:|---:|---:|---:|
| 6 | 3.69 µs | 3.31 µs | 2.27 µs |
| 20 | 5.09 µs | 2.98 µs | 2.32 µs |
| 40 | 6.98 µs | 2.86 µs | 2.24 µs |

40 子时增量 AVX2 约为全量参照的 3.1×；AVX2 相对增量标量约 1.27×。全量重建随棋子数增长，增量前向基本恒定。

**真实搜索吞吐**

| 后端 | 节点数 | 中位耗时 | nps |
|---|---:|---:|---:|
| reference（全量+标量） | 8113 | 65.6 ms | 123.7k |
| scalar（增量+标量） | 8113 | 55.3 ms | 146.7k |
| avx2（增量+AVX2） | 8113 | 48.2 ms | 168.2k |

三后端着法、分数、节点数完全相同；AVX2 约为全量参照的 1.36×。整机提速明显小于单次推理的 3.1×：该局面搜索中合法着法生成、胜负判定和搜索本身占主要成本。该比例随局面、深度和配置变化，不写成固定倍数。

**配对开局**

- 配置：AVX2 对 reference，2 个冻结开局各换色一局，共 4 局；双方同一可执行文件、同一模型、`--depth 2`、`--vcf-policy off`、单线程。
- 结果：每对走法序列完全相同，胜方颜色相同，终局 Reason 相同；A 2 胜 2 负（得分率 50%），无异常。说明优化没有改变选点或制造棋力差异。
- 该结果只证明“优化前后行为一致”，不构成棋力提升证据；本轮没有做不同模型或不同搜索预算的晋级测评。

**边界**

- AVX2 是编译期目标（顶层 CMake 对非 MSVC 工具链加 `-mavx2`），与旧 NNUE 相同；未声称 AVX-512 或其它平台加速。
- 搜索性能随局面子数和分支因子变化；本记录只保留上述条件下可复现的数字。若后续机器或模型出现回退，可加 `--value-backend reference` 固定走全量参照路径，再用 `value_backend_tests` 与 `value_parity_tests` 对拍定位。
- 未修改正式版本指针、样本池或历史测评结果；本工单不发布模型、不宣称棋力提高。
