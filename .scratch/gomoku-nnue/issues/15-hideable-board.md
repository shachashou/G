# 15: 可隐藏棋盘

**What to build:** 顶部提供「隐藏棋盘 / 显示棋盘」开关（快捷键 Ctrl+B），收起左侧棋盘，把宽度让给右侧面板；选择写入 QSettings，下次启动沿用；隐藏只影响显示，不影响对局、实时镜像或后台任务。

**Blocked by:** 14 — 实时棋盘与三模式简化工作台。

**Status:** done

- [x] 顶栏新增「隐藏棋盘」按钮，样式与「设置」一致的次要按钮；点击切换为「显示棋盘」。
- [x] 隐藏时棋子棋盘卡片不占布局空间，右侧面板自动占满整个宽度，不被拉伸或裁切。
- [x] Ctrl+B 与按钮等效；按钮 tooltip 说明快捷键。
- [x] 隐藏/显示不改变对局状态、执行器连接、实时快照或任务；隐藏期间后台任务与镜像照常运行。
- [x] 隐藏时 LIVE/返回对局入口不显示，状态行改为「后台任务运行中」或「棋盘已隐藏（Ctrl+B 显示）」提示，避免指向不存在的按钮。
- [x] 重新显示时按当前状态重绘玩家棋盘或实时镜像，并恢复 LIVE 入口。
- [x] 选择写入 `QSettings`（`window/board_visible`），重启客户端沿用。
- [x] 现有三模式、模型胶囊、设置 sheet、实时棋盘、任务暂停/恢复/测评逻辑与测试不变。

## Verification

- `gui_ui_smoke`：按钮可见、文字切换、棋盘卡片可见性、QSettings 写入、第二个窗口沿用隐藏状态、Ctrl+B 键位、恢复显示。
- `gui_live_board_smoke`：隐藏后 LIVE 入口消失、实时快照仍在更新、重新显示后 LIVE 入口恢复。
- 现有 GUI、任务、恢复、测评、训练测试全部保留；不修改 C++ 规则、协议、搜索、训练或任务状态机。

### 实施与验收记录（2026-11-11）

- `gui/gomoku_gui.py`：新增 `_board_visible`、`btn_board`、`_toggle_board()`、`_set_board_visible()` 与 `QShortcut("Ctrl+B")`；`board_card` 提升为实例属性以便切换可见性。
- `_sync_live_button()` 在棋盘隐藏时不再显示 LIVE/返回对局，并把提示改为「棋盘已隐藏（Ctrl+B 显示）」；`_refresh()` 在隐藏时只显示「后台任务运行中」。
- 重新显示时按 `_live_showing()` 选择重绘镜像或执行 `_refresh()`，不重开对局、不重建执行器。
- `MainWindow` 新增可选 `settings` 参数（默认仍是 `QSettings("GomokuNNUE", "DesktopClient")`），让测试使用临时 ini，避免污染真实客户端偏好。
- 快捷键放在 `_build_ui()` 末尾创建并保存引用，避免被垃圾回收。
- 测试更新：`engine/tests/gui_ui_smoke.py`（开关、持久化、快捷键、重启沿用）、`engine/tests/gui_live_board_smoke.py`（隐藏期间的 LIVE 与快照行为）。
- 验收：两个 offscreen 冒烟测试通过（2026-11-11，Qt offscreen）。
