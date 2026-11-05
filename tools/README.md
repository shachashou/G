# tools — 配套工具

- `match_runner.py`（工单 06）：两个引擎 build 批量对打、交替先手、随机开局、异常计负、每局着法序列 JSON 落盘与 `--min-win-rate` 胜率门控。完整用法见根目录 `README.md` 的「工具用法」。
- `datagen.py`（工单 07）：多进程自我对弈编排（run / merge / validate / verify / calibrate / report）；C++ 写端是 `build/engine/Release/datagen.exe`，Python 读端是 `trainer/binpack.py`。

项目 Python 一律走独立虚拟环境（`py -3.12 -m venv .venv`），不要直接用系统解释器。

---

## binpack 二进制格式（权威定义，v1）

C++ 写端 `engine/src/binpack.h/.cpp`、Python 读端 `trainer/binpack.py` 共用这份规范。所有多字节整数均为 **little-endian**；文件由 64 字节头 + N 条 64 字节定长记录组成，因此 `mmap` 后第 `i` 条记录的偏移恒为 `64 + 64*i`，`data/` 下分片可直接拼接/合并。

### 文件头（64 字节）

| 偏移 | 类型 | 字段 | 说明 |
|---:|---|---|---|
| 0..7 | `8s` | magic | 固定 `b"GOMOKU07"`；不匹配直接拒绝读取 |
| 8..11 | `u32` | version | 当前 `1` |
| 12..15 | `u32` | header_size | `64` |
| 16..19 | `u32` | record_size | `64` |
| 20..23 | `u32` | board_size | `15` |
| 24..27 | `u32` | search_depth | 生成时固定搜索深度（复盘按此值） |
| 28..31 | `u32` | shard_id | worker 编号；合并文件为 `0xFFFFFFFF` |
| 32..39 | `u64` | record_count | 记录条数（`finish()` 时回填） |
| 40..47 | `u64` | game_count | 有记录的对局数 |
| 48..55 | `u64` | base_seed | worker 的 RNG 基种子 |
| 56 | `u8` | rule_mode | `1`=freestyle，`2`=Renju |
| 57 | `u8` | use_vcf | `0/1`，复盘是否开 VCF |
| 58..59 | `u16` | tt_mb | worker 的置换表 MB（复盘按此值） |
| 60..61 | `u16` | vcf_node_limit | VCF 节点上限（默认 5000） |
| 62..63 | `u16` | opening_temp_x100 | 开局温度 ×100（100 = 1.00） |

worker 启动时先写一个 `record_count=0 / game_count=0` 的占位头，正常结束时用 `fstream` 回写真实计数。若 worker 中途崩溃，头部计数会与文件大小不符，`validate` 会报错——这是故意留下的完整性哨兵。

### 记录（64 字节）

| 偏移 | 类型 | 字段 | 说明 |
|---:|---|---|---|
| 0..28 | `225 bit` | black plane | 黑子占位；`bit i` = 格子 `i`（`x=i%15, y=i/15`），字节内 LSB 优先；bit 225..231 为 padding，必须为 0 |
| 29..57 | `225 bit` | white plane | 白子占位，编码同上；与 black plane 不得重叠 |
| 58..59 | `i16` | score | 该局面的搜索分，**行棋方视角（negamax）** |
| 60 | `u8` | flags | bit 0..1 WDL、bit 2 stm、bit 3 game-start、bit 4 game-end、bit 5..7 保留必须为 0 |
| 61 | `u8` | ply | 记录时盘上已有子数（`0..225`），也等于两侧 plane 的 popcount 之和 |
| 62..63 | `u16` | move | 从该局面实际走出的着法索引 `0..224`；`0xFFFF` 表示无着法（正常记录不会出现） |

**WDL（bit 0..1）以行棋方为准**：`0`=行棋方负、`1`=和、`2`=行棋方胜、`3`=未回填（只在对局进行中短暂存在，成品文件里出现即为错误）。`stm`（bit 2）：`0`=黑、`1`=白；黑先手时恒有 `stm == ply & 1`。配合双输出头训练时无需再翻转标签。

**分数饱和**：引擎分数是 `int`（手工评估单窗口最高 500000、VCF/mate 为 ±999999），装不进 `int16`。写盘统一 `clamp(score, -32767, 32767)`；`-32768` 保留不用。`tanh(score/400)` 会把 ±32767 压到 ±1，饱和不损失训练信号；复盘按同样规则比较。VCF 命中较多时饱和记录占比约 15–20%，属正常。

**对局边界**：每条记录带 `game-start` / `game-end` 位，可精确切分对局（验证集按对局划分，不依赖 ply 单调性）。worker 先把记录以 `WDL=3` 顺序写出并记下文件偏移，对局结束时按各记录的 `stm` 把 `WDL` 换算成 `wdlFor(result, stm)`，随机写回偏移 `+60` 的 flags 字节，同时给首/末记录置边界位——这就是「结果回填」。全部记录定长使随机回填只需一次单字节写。

**合并**：`binpack.merge_files` 逐个校验分片头（magic/version/格式/搜索配置必须一致、记录数与文件大小必须吻合），只拼接记录体，重写一个 `shard_id=0xFFFFFFFF` 的新头并累加 `record_count/game_count`；记录内的对局边界位原样保留，合并后仍能按对局切分。

**确定性**：datagen 每步搜索前调用 `Search::reset()` 清空置换表与启发式，使分数只依赖「局面 + depth + VCF + TT 大小」，`datagen.exe --verify` 可以只重建单局面（由两个 bit plane 复原棋盘）重新搜索来复盘，不需要重放整局。

### 用法

```powershell
# 16 进程起步；按实测本机 28 逻辑核 + 降深度预案，生产配置用 28 进程 depth 4
.venv\Scripts\python.exe tools\datagen.py run `
    --workers 28 --games-per-worker 100 --depth 4 `
    --out-dir data\raw --tag round1 --seed 20261105 --merge

# 单独合并 / 校验 / 搜索分复盘（抽查或全量）
.venv\Scripts\python.exe tools\datagen.py merge `
    --inputs "data/raw/round1/shard_*.binpack" --output data\raw\round1\train.binpack
.venv\Scripts\python.exe tools\datagen.py validate --input data\raw\round1\train.binpack
.venv\Scripts\python.exe tools\datagen.py verify --input data\raw\round1\train.binpack --samples 64

# 换机器时重新校准：比较多个 depth 的吞吐并给出推荐
.venv\Scripts\python.exe tools\datagen.py calibrate --workers 28 --depths 6,5,4,3 --games-per-worker 3
.venv\Scripts\python.exe tools\datagen.py report --manifest data\raw\round1\manifest.json
```

`run` 的 `--tag` 在 `--out-dir` 下建独立子目录，每个 worker 独占 `shard_XXXX.binpack` 与 `logs/worker_XXXX.log`，进程间无共享可变状态；`--merge` 额外产出 `train.binpack`；`manifest.json` 记录配置、每分片统计、总吞吐与达标判定。`verify` 直接调用 `datagen.exe --verify`，头部里的 depth/VCF/TT 配置自动带入复盘，篡改分数会被检出并以非零码退出。

### 边界情况

- 满盘：`Board::place` 在第 225 手判和，`playGame` 不再记录终局局面；`ply` 最大 224，仍可放进 `u8`。
- 超长对局：`--max-plies`（默认 225）到限未分胜负判和，已写出的记录照样回填 WDL、置好 game-end 位。
- 无合法着法（例如 Renju 下剩余空点全是黑禁手、或候选为空）：判和并安全收尾，不写无效记录。
- 未回填记录：flags 里 `WDL=3` 会被 `validate` 与 `--verify` 同时报错；头部计数与文件大小不符也会被 `validate` 抓到。

---

## 吞吐实测（本机：28 逻辑核，Windows 11，MSVC Release / AVX2）

`run` / `calibrate` 的统计口径是「从拉起第一个 worker 到最后一个 worker 退出」的墙钟时间除以**总落盘局面数**（不是引擎节点数）。达成 1000 万局面 / 12 小时需要 **231.48 局面/秒**。

| 并行度 | depth | 样本 | 局面/秒 | 1000 万预计 |
|---:|---:|---:|---:|---:|
| 1 | 6 | 1 局 | 2.2 | ~1260 h |
| 16 | 6 | 32 局 | 3.2 | ~870 h |
| 16 | 5 | 32 局 | 43.5 | ~64 h |
| 16 | 4 | 32 局 | 90.9 | ~30.6 h |
| 28 | 5 | 84 局 | 94.2 | ~29.5 h |
| 28 | 4 | 84 局 | 171.8 | ~16.2 h |
| **28** | **4** | **280 局 / 7208 局面** | **285.2** | **9.74 h（达标）** |
| 28 | 3 | 84 局 | 2187.0 | ~1.3 h |

**结论与预案**：规格里的 `depth 6` 在这套 M3 引擎上不可行——固定深度 6 的单步搜索约 0.45 秒，且不同开局的成本方差极大（16 进程 × 2 局时最慢 worker 达 258 秒，长尾拖垮整体）。置换表 16/64MB 与 VCF 节点上限 500/5000 的对比实测对耗时无实质影响，瓶颈是 alpha-beta 节点数（每个节点都要为所有候选点跑 Renju 禁手判定）。按工单预案启用 **降 depth + 调并行度**：首轮生产配置 `--depth 4 --workers 28`，实测 285 局面/秒、1000 万局面约 9.7 小时，落在半天预算内；若要更快跑通「生成→落盘→训练→加载→对弈」闭环，先用同一配置生成较小目标（例如 100 万局面，约 1 小时）。`depth 3` 虽快但棋力过弱，只作应急兜底。后续若要恢复 depth 6，需要先优化引擎的禁手判定/候选生成开销，属于 M3 之后的性能工单，不在 07 范围。

实测校验：`bench-d4` 合并文件 7208 条记录 / 280 局，`validate` 全量零错误（WDL 全部回填、边界位与 ply/stm 自洽）；`verify` 抽查 64 条零差异，另对 `shard_0000.binpack` **210/210 条全量复盘零差异**。
