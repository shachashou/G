# 桌面对弈客户端（工单 02）

Windows 本地客户端：选择执黑或执白，与现有引擎完成一局有禁手五子棋，查看合法落点、禁手原因、着法记录、对局结果与重放校验，并可随时开新局。

## 框架与本地控制方案

- **GUI 框架：PySide6 6.9.3**（Qt for Python）。控件与信号槽完整，适合后台任务与长时间运行；与 `trainer/` 和 `tools/` 同为 Python，后续采集、自我训练、测评工单可以直接复用同一套后端客户端。依赖已写入根目录 `requirements.txt`，离线可运行。
- **控制协议：JSON-lines over stdin/stdout**。GUI 通过 `tools/executor.py` 启动 `gomoku_nnue.exe --executor`，每行一个 JSON 请求、一行一个 JSON 响应。原 Gomocup 协议保持不变，仍供引擎对引擎对战使用。
- **权威执行器**：所有规则（完整合法落点、禁手原因、落子结果、终局）只由 C++ `Executor` 给出。GUI 和 Python 客户端不实现任何禁手或胜负算法。

| 命令 | 作用 |
|---|---|
| `about` | 引擎身份与实际评估方式（手工评估不会标成已训练 NNUE） |
| `new_game` | 15×15、黑先白后、自由开局的新局 |
| `state` | 棋盘、行棋方、结果、规则、着法历史快照 |
| `legal_moves` | **完整**合法落点集合，以及每个禁手点的具体原因 |
| `move_reason` | 单点合法性及拒绝原因 |
| `play` | 落子；非法请求或终局后落子不改变棋盘、行棋方或历史 |
| `undo` | 撤销上一手 |
| `rebuild` | 局面重建与校验（拒绝损坏输入；报告是否与黑先行棋一致） |
| `replay` | 从空棋盘重放着法序列，得到一致终局 |
| `candidates` | 搜索候选裁剪，明确区别于完整合法落点集合 |
| `think` | 引擎选点（只返回着法，不落子），GUI 再显式 `play` |
| `quit` | 结束执行器进程 |

## 行为约定

- 引擎思考放在后台线程，主线程只负责界面；思考期间不向执行器发其它请求，界面仍可响应「新局」「停止对局」和关闭窗口。
- 「停止对局」和关闭窗口会终止引擎子进程并按需重建，不留下后台工作进程；引擎侧诊断与协议错误在状态区明确显示。
- 没有新模型时显示「手工威胁评估（未加载已训练 NNUE 模型）」；加载 NNUE 后显示后端与权重路径，不会把手工评估标记为已训练 NNUE。
- 一局结束后自动用同一执行器重放着法序列并显示校验结果。

## 启动

先按根目录 `README.md` 用 venv 的 cmake + Ninja 构建出 `build/engine/gomoku_nnue.exe`，然后任选一种：

~~~text
双击 gui\play.bat
.venv\Scripts\python.exe gui\gomoku_gui.py
.venv\Scripts\python.exe gui\gomoku_gui.py --engine build\engine\gomoku_nnue.exe --movetime 800
~~~

也可用环境变量 `GOMOKU_ENGINE` 指定引擎路径。

## 验证

- `engine/tests/executor_tests.cpp`：执行器 C++ 行为（非法请求不改状态、终局拒绝、撤销、规则分类、D4、重放一致）。
- `engine/tests/executor_blackbox.py`：通过 JSON 协议黑盒验证完整落点、禁手原因、非法不改状态、终局拒绝、完整对局重放一致与进程清理。
- 两者均注册在 `ctest`；界面验收覆盖选色、新局、引擎响应、合法落点/禁手原因、着法记录、结束重放校验与退出清理。
