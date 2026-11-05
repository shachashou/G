# tools — 配套工具

- `match_runner.py`（工单 06）：两个引擎 build 批量对打、交替先手、随机开局、异常计负、每局着法序列 JSON 落盘与 `--min-win-rate` 胜率门控。完整用法见根目录 `README.md` 的「工具用法」。
- datagen：多进程自我对弈批量生成训练数据（工单 07，待实现）。

项目 Python 一律走独立虚拟环境（`py -3.12 -m venv .venv`），不要直接用系统解释器。
