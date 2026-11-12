# 19: 测评候选防护与保留结论说明

**What to build:** 测评页不再把「当前正式版本」自动填成候选（避免 `candidate is identical to a selected historical model`）；启动前给出中文拦截；保留现任时在报告里用一行话解释为什么没晋级；并在测评/训练面板直接显示「当前配置下需要多少得分率才能晋级」。

**Blocked by:** 18 — 历史数据清理。

**Status:** done

- [x] 顶部胶囊切到正式/历史版本时，测评候选框不再自动填入该路径；切到未登记候选时正常填入。
- [x] 测评启动前核验候选哈希：与任何登记版本相同则中文拒绝（说明是哪个版本、应选新训练的候选目录），不启动 worker、不创建空测评目录。
- [x] 保留后端兜底：错误信息带具体版本名（`candidate is identical to the registered version v1`）。
- [x] 报告面板的统计增加「有效配对」「门槛」，结论显示中文，并给出「判定」：通过 / 对局未完整 / 配对不足 / 明显更弱 / 证据不足（含得分率、配对下界、门槛与增加开局对数建议）。
- [x] 测评面板与训练任务面板实时显示晋级条件：`64 对开局、1 个对手，配对得分率需 > 65.3%（单侧 95% 下界 > 50%）`；改开局对数/对手/基线时立即更新，并注明「51% 的观察值并不等于能晋级」。
- [x] 统计基础新增 `hoeffding_margin()` / `required_score_rate()`，GUI 不另写一套门槛公式。
- [x] 不改变统计方法、门槛、冻结开局、晋级原子发布或任务调度。

## Verification

- `gui_history_guard_smoke`：候选等于正式 v1 时 `candidate_problem` 返回中文说明、`start()` 拒绝且无 worker/无目录；不同文件通过；得分率 62.5% / 下界 47.2% / 64 对的保留报告出现「证据不足」与「增加开局对数」；`_sync_history_candidate` 对正式版本清空、对候选填入；门槛提示 64 对=65.3%、128 对=60.8%、基线 0.51 时=66.3%（即提高基线反而更难）。
- `gui_ui_smoke`：训练任务面板显示「晋级条件」。
- `gui_history_smoke`、`gui_model_smoke` 等既有测评/模型测试保持通过。

### 实施与验收记录（2026-11-11）

- 现象：本机 v1 已在 15:31 晋级后，测评页候选框仍默认填当前正式版本，直接点挑战会在 `history_evaluation` 冻结候选时撞上「candidate is identical to a selected historical model」，留下 9 个 3 MB 的空 run 目录（15:38–15:41）。
- `gui/history_panel.py`：新增 `candidate_problem()`；`start()` 在启动线程前调用；`show_report()` 增加判定文案（`judgement()` / `CONCLUSION_TEXT`）；新增 `gate_hint` 与 `_update_gate_hint()`。
- `gui/training_task_panel.py`：新增 `eval_hint`，按任务固定单对手（当前正式版本）显示所需得分率。
- `gui/gomoku_gui.py`：新增 `_sync_history_candidate()` 并在启动与切换模型时调用；正式版本不再写入候选框。
- `tools/history_evaluation.py`：同一候选错误信息改为带版本名；新增 `hoeffding_margin()` / `required_score_rate()` 作为唯一门槛公式。
- 既有训练任务（`selfplay-tasks/d9126e8b…`）实测：两轮候选得分率 50.8% / 62.5%，配对下界 35.5% / 47.2%，均未超过 50% 门槛；随后到达 1000 局采集上限正常结束。未晋级是统计门槛结果，不是训练失败。

