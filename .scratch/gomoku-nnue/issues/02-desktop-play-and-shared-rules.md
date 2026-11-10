# 02: GUI 完成一局合法对弈

**What to build:** 用户打开 Windows 本地客户端后，可以选择执黑或执白，与现有引擎完成一局有禁手五子棋，查看合法落点、拒绝原因、对局结果并重新开局。该对局使用后续采集和裁判能够共用的权威执行器。

**Blocked by:** None (can start immediately).

**Status:** done

- [x] 选择并记录适合本机 C++ 引擎与 Python 训练工具的桌面 GUI 和本地控制方案；提供真实可启动入口。
- [x] 支持15×15、黑先白后、自由开局、玩家选择颜色、引擎响应、新局及结束后查看着法记录。
- [x] 客户端通过权威执行器获取完整合法落点、禁手原因、落子结果和终局，不在界面或 Python 裁判另写一套禁手算法。
- [x] 执行器公开局面重建与校验、合法落点查询、落子和撤销能力，足以供采集、算杀证据复查和比赛裁判复用。
- [x] 合法落点集合与搜索候选裁剪区分；非法请求和终局后的落子不改变棋盘、行棋方或历史。
- [x] 对既定规则覆盖三三、四四、真假活三、长连、成五优先、白棋长连、边界与 D4 等价局面，不重新进行产品规则访谈。
- [x] 没有新模型时明确显示当前使用的现有引擎评估方式，不把手工评估标记为已训练 NNUE。
- [x] 引擎思考期间界面保持响应；退出或停止对局能够结束相关工作进程，错误在界面明确显示。
- [x] 用户可从 GUI 完成完整真实对局；着法序列通过同一执行器重放后得到一致结果。

## Verification

在客户端交互与公共执行器边界验证可见行为。复用棋盘、Renju 和协议测试素材，补充非法操作不改状态、终局拒绝落子及完整对局重放。

### 验收记录

- **实际执行**：
  - 环境按本轮要求改为 venv 自包含：`requirements.txt` 加入 `cmake==3.31.10`、`ninja==1.13.2`、`PySide6==6.9.3`，并修正原先指向不存在的 `trainer/requirements.txt` 的注释；`CMakeLists.txt` 去掉「非 MSVC 直接 FATAL_ERROR」，改为按编译器分支设置 flags，venv python 探测兼容 `Scripts/` 与 `bin/`。Windows 构建用 VS DevShell 模块激活 `cl.exe` 环境，再用 `.venv/Scripts/cmake.exe -G Ninja` + `.venv/Scripts/ctest.exe`，不再引用 VS 自带 cmake。
  - 框架与协议：PySide6 6.9.3 + JSON-lines（stdin/stdout 每行一个 JSON）记录在 `gui/README.md`；真实启动入口 `gui/play.bat`。
  - C++：`renju::classifyForbidden` 把禁手判定拆成「分类 + `isForbiddenMove` 包装」，界面显示的原因与 `Board::isPlayable` 同源；新增 `Executor`（局面重建校验、完整合法落点、落子/撤销、重放、搜索候选裁剪）、极简 JSON 解析/生成、`--executor` 协议（`about/new_game/state/legal_moves/move_reason/play/undo/rebuild/replay/candidates/think/quit`）。原 Gomocup 协议与 15 项既有测试未改动。
  - Python 共享客户端 `tools/executor.py`（供 GUI、采集、算杀证据复查、裁判复用）；`engine/tests/executor_blackbox.py` 黑盒验证；`engine/tests/executor_tests.cpp` 覆盖非法不改状态、终局拒绝、撤销、禁手原因分类、legalMoves/candidates 区分、D4、完整重放一致。
  - GUI `gui/gomoku_gui.py`：选色、新局、合法落点/禁手原因标记、着法记录、终局结果、结束自动重放校验、手工评估明确标注「未加载已训练 NNUE 模型」、后台线程思考、停止/退出清理进程、错误显示。
  - cmake 重新生成 + Ninja 编译 135/135 成功；`ctest` **17/17 全绿**（原有 15 项 + `executor_tests` + `executor_blackbox`）；GUI 用 `QT_QPA_PLATFORM=offscreen` 冒烟通过：窗口构造/绘制、评估方式显示、玩家落子→引擎思考→引擎响应→回到可落子状态，无错误。
- **缺少依赖**：无（torch/numpy/tqdm/PySide6/cmake/ninja 均在 venv，MSVC 编译器由本机 VS 提供）。
- **未运行**：未做人工长时间 GUI 对弈或棋力实验；界面交互以 offscreen 冒烟与协议黑盒覆盖，真实桌面点击流程留待用户启动 `gui/play.bat` 体验。未启动大规模采集或训练（不在本工单范围）。
