# trainer：训练与导出

项目处于重设计阶段，目标以[主规格](../.scratch/gomoku-nnue/spec.md)为准，术语见[术语表](../GLOSSARY.md)。

## 新训练目标

- 网络为450 → 1024 → 64 → 32 → 1，单价值输出。
- 教师启动学习归一化的 Rapfi 评分；自主阶段学习完整对局的实际胜负。
- 使用 MSE 或 SmoothL1，并记录标签来源、视角、尺度及训练配置。
- 八种 D4 空间变换在线增广；Renju 不允许把交换黑白作为等价增广。
- 持续从有容量上限的样本池训练，无需先收集满数百万条。
- 按对局冻结验证集并排除 D4 等价泄漏。
- 分别验证训练原模型、量化导出模型与 C++ 推理；模型正式晋级依据历史版本对战。

## 可复用基础与待替换内容

| 能力 | 现有基础 | 实施边界 |
|---|---|---|
| 特征与对称 | 450维占位及 D4 变换 | 核验新单输出价值契约 |
| 数据 | 定长记录、数据集、对局划分 | 补标签来源、版本、战术状态与滚动样本池 |
| 训练 | PyTorch、checkpoint和训练记录 | 替换旧网络、目标和批处理编排 |
| 验证 | 冻结验证身份与防泄漏工具 | 保持追加与淘汰数据时验证身份不变 |
| 导出 | 格式校验、整数参考和数值对拍 | 新架构使用可识别的新格式或版本 |

现有模型、训练器和权重格式仍包含旧结构；旧的手工评估标签与线性目标已移除，教师标签将按新工单接入 Rapfi。

## 价值网络与导出（工单 06）

新训练链路只使用以下模块，旧的 `nnue.py` / `train.py` / `export.py` / `weights_format.py` 属于旧 GMNNUE 双头实现，仅为旧 C++ 推理对拍测试保留，新流程不使用它们。

| 模块 | 职责 |
|---|---|
| value_model.py | 450-1024-64-32-1 单输出网络，黑方绝对价值 `tanh(z)`，隐藏层定上限 ReLU |
| value_data.py | 从样本池读取已提交数据，教师视角→黑方绝对值转换，D4 在线增广与泄漏检查 |
| value_format.py | GMVALUE1 文件格式、FNV 校验、int16/int32 整数参考前向、冻结的保真度容差 |
| value_train.py | 训练循环、checkpoint 保存/恢复、验证指标、量化导出与保真度报告、CLI |

~~~powershell
.venv\Scripts\python.exe trainer\value_train.py --pool data\pool\pool.sqlite3 ^
    --output-dir weights\value-run1 --steps 200 --batch-size 32 --lr 1e-3 --loss mse
~~~

输出目录包含 `last.pt` / `best.pt`（可恢复）、`train_history.jsonl`、`train_summary.json`、`model.gmvalue`、`fidelity.json` 与 `training_positions.json`。`GMVALUE1` 记录架构、规则、特征、价值契约、标签版本和训练数据身份；加载旧 `GMNNUE01` 或架构不符的元数据会被拒绝。C++ 已能严格加载该格式并在 Alpha-Beta 中实际对弈（工单 07）；第一层增量累加与 AVX2 在工单 11，当前搜索使用全量重建参照路径。

工单 08 的测评以 `training_positions.json` 排除候选训练/验证局面的 D4 等价开局；清单 SHA-256 写入模型元数据并由保真度报告绑定。checkpoint 保存累计训练清单，换池续训时合并历史身份，池淘汰不会抹掉已用局面。缺少该清单的旧 checkpoint 明确拒绝续训，需从零训练，不能伪造空白训练历史。模型导出通过数值检查后仍是候选；登记 v0 或历史挑战通过后，才进入正式版本登记表，见[测评与晋级说明](../README.md#历史版本测评与晋级工单-08)。

## 环境与现有命令

工单 09 的调度器通过 `train_value(..., data_bundle=...)` 传入一个已冻结的完整训练/验证快照；采集可继续写入、淘汰，训练输入不会随之变化。数据指纹覆盖局面、标签和来源的完整内容，并记录提交序号、采集模型与对局来源。每次候选需有新的已提交数据，优化步数与实际样本曝光分别统计；停止只记录真正完成的优化步骤，小于批大小的样本池按实际批量计曝光。

持续任务可续训前一候选 checkpoint；首次从已有正式版本启动时，以其量化权重初始化浮点训练模型，重新建立优化器并继承已验证的训练清单。谱系写入 checkpoint 和导出元数据；数值或历史测评未通过都不替换正式模型。详见[持续教师任务](../README.md#持续教师训练工单-09)。

Python 使用项目 .venv；通用依赖见[requirements.txt](../requirements.txt)，AMD / NVIDIA / CPU 后端安装步骤见[根目录环境说明](../README.md#环境与构建)。AMD ROCm 与 NVIDIA CUDA 都使用 `--device cuda`，默认 `auto` 自动检测；CPU 使用 `--device cpu`。在已安装依赖的项目环境中，可检查当前入口：

~~~powershell
./.venv/Scripts/python.exe trainer/train.py --help
./.venv/Scripts/python.exe trainer/export.py --help
~~~

这些是当前实现的命令，不是新训练流水线的完成证明。新流程的默认参数和客户端操作随实施工单补充。
