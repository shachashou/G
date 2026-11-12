# GomokuNNUE

Windows 本地五子棋训练与对弈工作台：PySide6 客户端、C++ 执行器与搜索、Rust VCF 桥、PyTorch 训练端。

## 环境与命令

- venv 固定为 `.venv`。cmake / ninja / ziglang / Rust 工具链都装在 venv 内；不要调用系统 Visual Studio、Windows SDK、全局 cmake 或 `~/.cargo`。
- 依赖按 README 选择 `requirements-amd.txt` / `requirements-nvidia.txt` / `requirements-cpu.txt`；开发工具在 `requirements-dev.txt`。新增或升级依赖先说明，并同步更新对应文件。
- `data/` 与 `weights/` 是运行时产物，不提交；`weights/registry/state.json` 是正式版本的唯一真源。
- 回答与反问用中文。

| 目的 | 命令 |
|---|---|
| 构建 | `.venv\Scripts\python.exe tools\build\build.py` |
| 全量测试 | `.venv\Scripts\python.exe tools\build\build.py --test` |
| 单个测试 | `.venv\Scripts\python.exe tools\build\build.py --test --test-regex <名称>` |
| Lint | `.venv\Scripts\ruff.exe check <文件或目录>` |
| 启动对弈 | `gui\play.bat` |
| 查看正式版本 | `.venv\Scripts\python.exe tools\history_evaluation.py list` |

## 不可协商的规则

- **规则唯一来源是 C++ 执行器**：合法落点、禁手、胜负和着法重放都走 `tools/executor.py`；Python、GUI 和测试不另写 Renju 规则。
- **VCF 只认证明**：只有 `proven` 且首着合法的变化可以覆盖常规评估；`disproven` 只代表预算内没找到，`budget_exhausted` / `interrupted` 是未知，不能当成"没有杀"或"防守成功"。
- **三个概念分开**：算术一致性、导出保真度、实战棋力分别测量。训练/验证损失下降、自对弈能跑、导出通过，都不等于棋力提升。
- **晋级门槛赛前冻结**：候选对冻结版本、配对开局、交换黑白、相同时间与线程，只有配对得分的单侧下界严格高于基线才晋级；不因一次结果放宽门槛或降低证据标准。
- **数据与模型由工具管理**：不要手改 `weights/registry` 或 `data/pool`；清理只用 `tools/model_cleanup.py` / `tools/storage_cleanup.py`，先 dry-run，只删扫描器标记为可清理且不在保护名单内的目录。

## 代码地图

| 路径 | 职责 |
|---|---|
| `engine/` | C++ 执行器、规则、搜索、协议、部署推理、VCF 适配 |
| `trainer/` | PyTorch 数据、训练、验证、导出与权重格式 |
| `tools/` | 采集、教师打标、样本池、任务编排、对战、历史测评、清理 CLI |
| `gui/` | PySide6 桌面客户端与启动入口 |
| `vendor/quintet/` | 固定 commit 的开源 VCF；只经 `engine/src/vcf.cpp` 与 `engine/vcf-bridge` 使用 |
| `.scratch/gomoku-nnue/` | 主规格 `spec.md`、工单 `issues/`、环境验收记录 |
| `data/`、`weights/` | 运行时数据与模型，不纳入版本控制 |

## 开发规范

- 只做任务要求的改动；不顺手重构、加功能或加配置。未改动的代码不加 docstring、注释或类型标注。
- 死代码直接删除，不留兼容垫片；三行重复优于为一次性操作造抽象。内部代码信任约定，只在系统边界做校验。
- 临时脚本放 `.scratch/` 根目录（不入库），实验结束删除；可复用的能力放回 `tools/` 并补测试。
- 目标是 Python 3.12：直接用 `X | None`、`list[int]`，新文件不需要 `from __future__ import annotations`。
- 只对改动文件跑 Ruff；不要为清零而对全库执行 `ruff format` / `--fix`，避免无关的大范围 diff。
- GUI 行为优先用 CTest 的 offscreen smoke 验证；需要人工看界面时给出 `gui\play.bat`，不要主动打开真实桌面窗口。
- 改完先跑相关 CTest；提交前跑全量 `build.py --test`。提交使用 `feat:` / `fix:` / `refactor:` / `chore:`，说明动机与影响。

## 文档与工单

- 主规格：`.scratch/gomoku-nnue/spec.md`
- 术语：`GLOSSARY.md`
- 训练端：`trainer/README.md`；工具：`tools/README.md`；客户端：`gui/README.md`
- 工单与验收：`.scratch/gomoku-nnue/issues/`
- 编辑器与检查：`ruff.toml`、`.vscode/settings.json`

`.agents/skills/` 是安装物、不入库；只有用户点名时才走 `/implement`、`/implement-spec` 等 skill 流程。
