# 22: 搜索深度可见性

**What to build:** 每次 AI 行棋后，界面显示本轮实际搜索深度（层 / 手）、节点数与评分，并说明默认 8 层硬上限与算杀独立预算，让「AI 能看多少步」可观察。

**Blocked by:** 21 — 训练任务参数持久化。

**Status:** done

- [x] 对局详情新增「AI 搜索」一行：`深度 N 层（约 N/2 手） · Xk 节点 · 评分 S`；着法来自算杀时追加说明。
- [x] 行棋后同时更新顶部活动状态行；tooltip 说明「层 = 半手，默认上限 8 层」。
- [x] 不改变搜索、时间控制、算杀策略或落子逻辑。

## Verification

- `gui_ui_smoke`：调用 `_show_search_summary({"depth": 6, "nodes": 237000, "score": 12})` 后断言出现「深度 6 层」。
- `gui_model_smoke`、`gui_live_board_smoke` 保持通过。

### 实施与验收记录（2026-11-11）

- 实测（本机 Release，`--executor`，1 线程，VCF 默认 `root_and_children`）：

  | 局面 | 200 ms | 1000 ms |
  |---|---:|---:|
  | 空棋盘 | 7 层 / 74k 节点 | 8 层 / 74k 节点（172 ms 达上限） |
  | 开局（4 手） | 6 层 / 54k | 7 层 / 285k |
  | 中盘（12 手） | 5 层 / 42k | 6 层 / 237k |

- 结论：GUI 只传 `movetime_ms`，引擎默认 `maxDepth=8`；中盘 1 秒约 6 层 = 3 手。加时间最多到 8 层，再往上需要 `depth` 参数（GUI 未暴露）。
- `gui/gomoku_gui.py`：新增 `search_label` 与 `_show_search_summary()`；`_apply_engine_move()` 每次落子前刷新。
