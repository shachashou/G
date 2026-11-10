# 01: 训练基础脱离旧实验

**What to build:** 维护者能够在没有旧实验数据、权重和诊断产物的干净环境中，运行现有基础训练、导出和验证。解除旧实验与共用能力之间的依赖，移除被新规格淘汰的实验代码，为新网络训练提供可独立使用的基础。

**Blocked by:** None (can start immediately).

**Status:** done

- [x] 梳理旧诊断、蒸馏、局部排序、预算脚本及测试与当前训练、导出、数据校验之间的调用关系；保留共用能力，删除不再使用的实验流程。
- [x] 基础能力不再通过实验编号、旧实验计划或旧冻结产物才能调用；需要的校验、文件身份和数值报告能力迁移到通用边界。
- [x] 同步清理命令入口、构建与测试注册、依赖清单和文档引用；没有悬空导入、删除后仍被调用的脚本或只验证已删除行为的测试。
- [x] 旧 M5 的特定预算、固定比赛门槛和阻塞条件不再约束新流程。
- [x] 保留棋盘、规则、搜索、协议、数据读写及推理对拍基础；旧网络仅作为迁移前实现，不要求保留旧权重兼容。
- [x] 不存在历史数据和 checkpoint 时，使用临时生成的小样本完成一次基础训练、导出、回读校验；仅对新生成产物做判断。
- [x] 重新配置、构建并运行与保留能力相关的检查；移除的测试有明确对应的已淘汰行为，不能靠删除失败测试掩盖共用能力损坏。
- [x] 验收记录区分实际执行、缺少依赖及未运行内容，不启动大规模采集或棋力实验作为清理前置条件。

## Verification

优先在现有训练与导出入口、数据边界和引擎协议上验证外部行为。检查干净环境小闭环和剩余导入、构建引用，不为删除文件编写镜像测试。

### 验收记录

- **实际执行**：删除 23 个旧实验文件（trainer 9 个：diagnose/distill18/distill18_fidelity/local_ranking_diagnosis/local_ranking_fit/move_correlation/weight_drift/teacher_labels/targets；tools 9 个：bootstrap/budget_run/distill18_screen/distill18_summary/local_ranking_summary + 4 个 ps1；engine/tests 5 个）。解耦 train.py/export.py 的 `--target-contract` 线性路径、validation_freeze.py 的 `derive` 子命令、dataset.py 的 `target_contract`/`label_store`、match_runner.py 的 `--m5-gate`；清理 CMakeLists 5 个测试注册与 README 引用。cmake 重新生成 + 构建成功，ctest 15/15 全绿。小闭环：datagen（20 局 depth-2，491 局面）→ train（2 epoch CPU，val loss 收敛）→ export 回读 `read_back_replay_matches=true`、magic `GMNNUE01`、parity MAE 0.37。
- **缺少依赖**：无（torch/numpy 已装，MSVC 工具链可用）。
- **环境变更（工单 07）**：Windows 构建已完全收敛到 `.venv`（pip 的 ziglang + venv 内 Rust/rust-lld），不再使用 MSVC/Visual Studio；本工单涉及的训练与测试在工单 07 的全量回归中仍通过。
- **未运行**：未启动大规模采集或棋力实验（按工单要求，清理不以棋力实验为前置）。
