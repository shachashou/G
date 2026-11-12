# 18: 历史数据清理

**What to build:** 把清理入口从「候选模型」扩展到历史数据：已完成/失败的任务目录、未被登记表引用的测评记录；与候选包共用同一个对话框（两个标签页）和同一套保护规则。运行中任务、可恢复任务、登记版本证据默认保护，可恢复任务需显式勾选。

**Blocked by:** 17 — 模型清理。

**Status:** done

- [x] 新增 `tools/storage_cleanup.py`：扫描 `weights/*-tasks/<id>` 与 `data/evaluations/<run>`，标注大小、状态、保护原因和警告。
- [x] 任务目录按状态分类：运行中（running/pausing/resuming/stopping）硬保护；completed/failed 默认勾选；paused/stopped 可清理但默认不勾，警告「删除后无法继续恢复」。
- [x] 测评记录：被登记表任何版本引用的 run、被可恢复任务中 interrupted/evaluating 记录引用的 run 视为证据，永不删除；其余可清理但默认不勾，警告「删除后失去历史证据」。
- [x] 删除只接受重新扫描后仍存在、仍可清理、位于对应根目录内、非符号链接的整目录；未知/越界 id 拒绝。
- [x] 对话框改为两页：「候选模型」「历史数据」；各自有说明、树形列表与统计，底部统一显示各类已选数量/大小，一次确认同时删除两类。
- [x] 设置面板入口改为「清理模型与数据…」，摘要同时显示候选与历史数据总量/可清理量。
- [x] 任务运行中拒绝打开；删除后刷新模型胶囊、设置摘要与测评报告列表。
- [x] CLI：`tools/storage_cleanup.py` 默认列出；`--apply --select ID`、`--apply --all`、`--include-resumable`、`--json`。
- [x] 不修改任务状态机、登记表、训练或测评逻辑。

## Verification

- `storage_cleanup_tests`（15 项）：运行中保护、completed/failed 默认目标、stopped 警告、登记表 run 与中断测评 run 保护、旧 run 警告、摘要、越界/未知/受保护拒绝、幂等、CLI `--all` 保留可恢复与证据。
- `gui_cleanup_smoke`：offscreen 下两个标签页、默认勾选（仅 completed 任务与旧候选轮）、删除后运行中/stopped/证据/测试样例/手动输出仍在、显式勾选旧 run 可删除、摘要与胶囊刷新。
- `model_cleanup_tests` 保持通过；全量相关 CTest 通过。

### 实施与验收记录（2026-11-11）

- 新增 `tools/storage_cleanup.py`（`DataEntry` / `scan()` / `summary()` / `delete()` / CLI）与 `gui/cleanup_dialog.py`（两页 `QTabWidget` 共用勾选与删除流程）。
- `gui/gomoku_gui.py`：`_refresh_cleanup_summary()` 同时统计候选与历史数据；`_open_cleanup()` 传入权重、登记表与测评根目录；清理完成后刷新胶囊、摘要与测评列表。
- 本机实测（只扫描未删除）：`weights/` 已完成任务 2 个约 73.8 MB 可删；`data/evaluations/` 10 次记录中 8 次未被登记表引用，约 54 MB；运行中的自主训练任务与 v0/v1 证据受保护。
