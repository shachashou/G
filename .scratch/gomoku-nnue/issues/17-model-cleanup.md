# 17: 模型清理

**What to build:** 在设置面板「模型」分组提供「清理模型…」：列出 `weights/` 下的候选模型包与占用空间，默认勾选可安全清理的旧轮次，确认后整包删除；正式/历史版本、当前加载模型、测试样例和运行中任务永不删除。同时提供 `tools/model_cleanup.py` 命令行入口。

**Blocked by:** 16 — 模型胶囊可读性（同一模型列表的呈现口径）。

**Status:** done

- [x] 只扫描 `weights/` 下的候选包（含 `model.gmvalue` 的目录），`weights/registry` 完全不参与扫描。
- [x] 硬保护：登记表里的正式/历史版本、当前加载模型、本地测试样例 `weights/value-demo`、运行中任务的全部候选。
- [x] 每个任务至少保留最近 `keep` 轮（默认 1），保证暂停/已停止任务仍有最近检查点可恢复。
- [x] 保留给其余候选的删除选项带风险说明：「任务可恢复：只删旧轮」「手动输出（默认不勾选）」。
- [x] 对话框按任务分组展示：模型名、大小、说明；受保护条目不可勾选并置灰。
- [x] 默认勾选已完成/已失败任务的旧候选；手动输出默认不勾选，避免误删未测评的候选。
- [x] 底部实时显示候选总数、总占用、可清理数量与已选大小；删除前二次确认，删除后刷新列表、顶部胶囊和设置面板摘要。
- [x] 任务运行中拒绝打开清理对话框，避免删除正在写入的候选。
- [x] 删除只接受重新扫描后仍存在、仍可清理、仍在 `weights/` 内、非符号链接、不位于登记表下的目录；不存在或越界的 id 一律拒绝。
- [x] `tools/model_cleanup.py` 支持 `list`（默认）、`--apply --select ID`、`--apply --all`、`--include-manual`、`--keep N`、`--loaded PATH`、`--json`。
- [x] 不修改规则、搜索、训练、晋级、注册表和任务状态机。

## Verification

- `model_cleanup_tests`：登记表排除、运行中任务全保护、每任务保留最近 N 轮、可恢复任务旧轮警告、加载中模型保护、测试样例保护、手动输出需显式选择、摘要统计、越界/未知/受保护 id 拒绝、幂等删除、CLI 列表与 `--all`。
- `gui_model_cleanup_smoke`：offscreen 下打开设置面板、摘要与按钮可见、忙碌时拒绝、树形条目与默认勾选、确认后删除两个旧轮且保留最新轮/运行中任务/测试样例/登记表，删除后刷新摘要与胶囊。
- 现有 GUI、模型、任务与恢复测试不回归；全量 CTest 通过。

### 实施与验收记录（2026-11-11）

- 新增 `tools/model_cleanup.py`：`Bundle` / `scan()` / `summary()` / `delete()` / `human_size()` / CLI。硬保护与警告在扫描阶段确定，`delete()` 再次扫描并复检，避免 TOCTOU。
- 新增 `gui/model_cleanup_dialog.py`：按任务分组的 `QTreeWidget`、保留轮数 `QSpinBox`（写回 `QSettings` 的 `models/keep_candidates`）、全选可清理、删除确认与结果提示；`cleaned` 信号通知主窗口。
- `gui/gomoku_gui.py`：设置面板「模型」分组增加「清理模型…」按钮与摘要标签（候选数、占用、可清理数）；忙碌阶段拒绝；清理成功后重载模型胶囊并刷新摘要；`MainWindow` 新增 `weights_root` 注入点，便于离屏测试使用临时目录。
- 测试：新增 `engine/tests/model_cleanup_tests.py`（18 项）与 `engine/tests/gui_model_cleanup_smoke.py`，均注册到 `ctest`。
- 本机实测：`weights/` 共 8 个候选包、约 109.6 MB；可清理 4 个约 53.8 MB（自主训练旧轮 + 3 个手动输出）；正式 v0、测试样例与各任务最近一轮均受保护。
