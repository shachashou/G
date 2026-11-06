# 21: float16 权重链路——用更高精度权重替换 int8 以保留排序信号

**What to build:** 工单 20 已定位根因——int8 权重舍入摧毁走子排序信号，而 float16/bfloat16 精度即可保住（disagreement 0.267 vs int8 0.486）。本工单落地 float16 权重链路：新权重格式 v3（float16 权重 + float32 偏置）、C++ 浮点推理（scalar + AVX2）、导出与跨语言对齐、再用 pairwise 排序损失训练 → 导出 → M5 门控。保留旧 `GMNNUE01`/`GMNNUE02` int8 格式兼容。

**Status:** in-progress（实现中）
**Category:** implementation
**Blocked by:** 无。依赖工单 20 结论（float16 精度足够）。
**Downstream:** M5 门控（工单 16/10/12）；本工单不改动旧格式语义，v1/v2 推理保持逐位不变。

## 1. 契约（新权重格式 v3）

- 魔数 `GMNNUE03`、version=3，沿用 64 字节头（`<8s11I` + 保留 12 字节）。
- 架构维度不变：450 特征 / 1024 累加器 / 15 L1 / 2 头。
- 权重精度：`W0`/`W1`/`W2` 为 **float16**（IEEE 754 half），`b0`/`b1`/`b2` 为 **float32**。
- 推理算术：**全浮点**——累加器 float32、除法用浮点 `/qa`、`/qb`，**无 round_div、无 int16 clamp**；激活仍 ClippedReLU `[0, qa]` / `[0, qb]`；输出 `eval = y*gain/qb + gain*b2[h]`（format_version=2 的 gain 语义保留，加 ±900000 静态上限 clamp）。
- 文件体布局（row-major）：`W0 f16[450,1024]`、`b0 f32[1024]`、`W1 f16[1024,15]`、`b1 f32[15]`、`W2 f16[2,15]`、`b2 f32[2]`。

## 2. 落地清单

| 文件 | 改动 |
|---|---|
| `trainer/weights_format.py` | 新增 v3：float16 打包（`array('e')`）、`infer_float` 浮点参考前向、`unpack_weights` 兼容 v3；v1/v2 路径不变 |
| `trainer/nnue.py` | 复用现有 `quantized=False` 浮点前向（已与 v3 语义一致）；`from_quantized_weights` 支持 v3 float 权重 |
| `trainer/export.py` | 新增 float16 导出路径（`format_version=3`） |
| `engine/src/nnue.h` / `nnue.cpp` | 加载 v3：half→float 转换、float32 累加器、标量 + AVX2 浮点前向；v1/v2 整数路径逐位不变 |
| `engine/tests/nnue_tests.cpp` / `nnue_parity_tests.py` | v3 加载/对齐/坏文件拒绝 + C++ vs Python 浮点对齐 |

## 3. 验收

- [ ] `weights_format.py` v3 pack/unpack/infer_float 自洽（roundtrip + 与 numpy 参考一致）。
- [ ] C++ 加载 v3 并标量/AVX2 前向与 Python 浮点前向在容差内逐位/逐局面一致。
- [ ] v1/v2 整数路径 CTEST 回归全绿（不破坏旧格式）。
- [ ] 用 pairwise 排序损失训练 float 模型 → 导出 v3 → 引擎加载 v3 下完整对局。
- [ ] M5 门控（工单 16 的 20 局筛选入口）用 v3 候选重跑，记录胜率基线。

## 4. 预算与停止

长训练/门控沿用 `tools/budget_run.py`；短对齐检查直接跑。顺序执行，每阶段通过再进下一阶段。
