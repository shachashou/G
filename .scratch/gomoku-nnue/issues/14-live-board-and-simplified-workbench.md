# 14: 实时棋盘与三模式简化工作台

**What to build:** 后台采集、自我训练和测评时，左侧棋盘实时镜像其中一盘真实对局；主界面进一步简化为「对弈 / 训练 / 测评」三个模式、顶部模型胶囊、一行状态和单一主操作，所有高级参数与恢复入口收进右侧设置 sheet。

**Blocked by:** 13 — 简化界面（接近苹果手机式）。

**Status:** done

- [x] 采集、自我训练、测评运行时向 GUI 推送可选的完整 `cells` 快照；回调不阻塞、不重入引擎、不写入 `state.json`。
- [x] 棋盘实时镜像其中一盘：`LIVE` 标记、模式/编号、行棋方、手数、最后一手；多 worker 固定跟随一盘，结束或暂停后切换，绝不混盘。
- [x] 玩家不在对局中时自动显示实时棋盘；玩家正在下自己的棋时保留玩家棋盘，仅显示 `LIVE` 按钮供主动切换，任务结束/停止后回到玩家棋盘。
- [x] 暂停冻结并淡化最后一帧显示「已暂停」；舍弃/停止后短暂停留再回到玩家棋盘；自然终局保留约 2 秒再切下一盘；纯训练显示上一盘终局淡化 + 训练阶段，不伪造落子。
- [x] 实时棋盘只读；玩家点击不出手，不改变权威棋局或任务状态。
- [x] 顶部导航简化为「对弈 / 训练 / 测评」；训练页内部用「采集数据 / 从0训练 / 自我训练」三段切换，当前模式只保留一个主操作卡。
- [x] 模型选择移到顶部胶囊，选中立即应用；忙时禁用并说明原因；模型详情/路径/后端进设置 sheet。
- [x] 删除独立状态大卡片，顶部只保留一行模式/阶段/关键指标/错误；完整指标、日志和技术详情进设置 sheet。
- [x] 新增右侧设置 sheet：对局、采集、训练、测评、任务与资源分组；原高级控件全部迁入，默认值、绑定、暂停/恢复/晋级逻辑不变。
- [x] 训练页在存在可恢复任务时显示上下文入口「恢复 N 个未完成任务 →」，点击打开设置 sheet 的任务区；完整恢复列表仍在 sheet。
- [x] 保留所有既有公共控件属性、模型/对局/教师/采集/训练/测评/恢复功能和现有测试；新增实时棋盘与三模式 GUI 验收。
- [x] 更新 README、GUI 说明和验收记录；不把未实现的行为写成已完成。

## Verification

- 数据边界：对 `collect`、`selfplay_data`、`history_evaluation`、`TrainingTask` 的公共回调验证完整 cells、跟随/暂停/停止、线程安全和不持久化。
- GUI 边界：offscreen 下验证三模式、训练三段、模型胶囊、单行状态、设置 sheet、恢复入口、实时棋盘只读、玩家棋盘恢复。
- 现有 GUI、任务、恢复、测评、训练测试全部保留并通过；全量 CTest 通过。
- 不修改 C++ 规则、协议、搜索、训练、模型晋级或任务状态机。

### 实施与验收记录（2026-11-11）

**数据管线**

- `tools/collect.py`：新增 `position_payload()` / `emit_position()`；`play_game`、`collect` 接受可选 `on_position`，在该手被权威执行器接受后发出 225-cell 快照。采集使用 1-based 全局 `game` 编号；不传回调时不构造 payload。
- `tools/selfplay_data.py`：每次被接受的自对弈落子后调用 `task.publish_preview()`，携带 kind=selfplay、全局 game、模型身份。
- `tools/history_evaluation.py`：`play_pair_game`、`parallel_games`、`evaluate` 接受可选 `on_position`；为并行换色对局新增唯一 `game = pair*2 + colour`，避免把同一 pair 的两盘混在一起；`pair`、`engine`、`model_id` 一并携带。
- `tools/training_task.py`：`TrainingTask(preview=...)`、`resume_from_directory(..., preview=...)` 和 `publish_preview()`；只做浅拷贝/转发，不写 state、events、manifest 或池。
- 回调是 best-effort：异常被吞掉，不阻塞比赛；所有 cells 只在内存/GUI 信号中流动，不持久化（测试验证 `state.json`/`events.jsonl` 无 `cells`）。

**实时棋盘**

- `BoardWidget` 新增只读 `preview_mode`、banner/subtitle/dim；`set_preview()` 清空合法/禁手标记，鼠标按下直接忽略。最后一手用蓝色圆环，LIVE/暂停/训练状态用深色圆角 pill 叠加。
- `MainWindow._on_preview()` 使用 last-wins、GUI 主线程绘制；按 `(kind, game, pair)` 固定跟随一盘，2 秒内不切换到另一盘，之后切换；多 collector/evaluation worker 不混盘。
- 玩家不在进行中的对局时自动显示后台棋盘；玩家正在下自己的棋时保留玩家棋盘，顶栏出现 `LIVE`；点击后切换为后台棋盘，按钮变「返回对局」。
- 任务状态：暂停 → 最后一帧淡化 +「已暂停」；训练阶段 → 最后一帧淡化 +「训练中 · step N」；停止/完成 → 短暂保留停止帧约 1.2 秒后回到玩家棋盘；任务结束清除预览。
- 独立采集、持续教师/自主任务和独立测评都接入；`TrainingTaskPanel.previewReceived`、`HistoryPanel.previewReceived` 通过信号进入主线程。

**导航与设置**

- 顶部只保留三个主模式：对弈 / 训练 / 测评。训练页顶部三段切换：采集数据 / 从0训练 / 自我训练；只显示当前模式的一个主操作卡。
- 模型选择移到顶部胶囊：当前正式/历史/候选/手工评估/浏览文件；选中立即应用，忙时拒绝并恢复显示；模型详情、路径、后端在设置 sheet。
- 删除独立状态大卡片；标题下只保留一行活动状态（模式/阶段/关键指标），错误为内联红色提示。旧状态/详情控件保留在设置 sheet 供测试和查看。
- 新增 `gui/settings_sheet.py`：右侧滑出 sheet + 半透明遮罩，关闭按钮/遮罩/ESC 收起；分组「模型 / 高级对局设置 / 对局详情 / 教师打标 / 采集参数 / 从0训练参数 / 任务与资源 / 恢复未完成任务 / 任务指标 / 测评参数与对手 / 测评报告」。
- 所有原高级控件原样迁入 sheet；默认值、绑定、暂停/恢复/晋级逻辑未改。训练页在存在未完成任务时显示「恢复 N 个未完成任务 →」，点击打开 sheet 的任务区。
- 主界面卡片标题统一为「运行」，避免与标签/段切换重复；旧控件属性名全部保留。

**测试**

- 新增 `gui_live_board_smoke`：注入实时 payload，验证 LIVE 横幅、跟随固定游戏、多 worker 不混盘、暂停淡化、玩家棋局不被抢占、LIVE/返回切换、只读点击不落子、停止清除预览。
- 更新 `gui_ui_smoke`：三模式、训练三段、模型胶囊、单行状态、设置 sheet 分组、高级控件默认不可见/展开可见、主/危险按钮角色、无横向裁切。
- `gui_training_task_smoke` 增加真实教师任务必须产生 225-cell 实时预览的断言；`gui_recovery_smoke` 更新三模式断言。
- 数据侧扩展 `collect_tests`、`selfplay_task_tests`、`history_evaluation_tests`：验证 225 cells、ply/最后着法、1-based game、evaluation pair/engine/唯一 game、停止后不再产生预览、报告/manifest 不变、cells 不持久化。
- 全量 CTest **43/43 通过，无跳过**（最终一轮 123.34 s）；`git diff --check` 通过。
- 视觉截图在 `build/acceptance/ticket14/`（构建产物，可重新生成）；offscreen 缺字体目录时中文显示为方块，本机 Windows 字体下正常。

**边界**

- 实时棋盘是只读观察窗口，不替代玩家对局，也不修改任何权威棋盘状态。
- 多 worker 只跟随一盘（固定后按 last-wins + 2 秒切换）；不是多盘宫格。
- 纯训练无棋局时不伪造落子；只保留上一盘终局并显示训练阶段。
- 预览是 best-effort 可视化，不作为训练、测评或晋级证据；不影响规则、搜索、训练、发布或恢复逻辑。
