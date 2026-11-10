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

新训练链路只使用以下模块，旧的 `nnue.py` / `train.py` / `export.py` / `weights_format.py` 属于旧 GMNNUE 双头实现，保留是为了旧 C++ 推理测试在工单 07/11 切换前仍可运行，新流程不使用它们。

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

输出目录包含 `last.pt` / `best.pt`（可恢复）、`train_history.jsonl`、`train_summary.json`、`model.gmvalue` 与 `fidelity.json`。`GMVALUE1` 记录架构、规则、特征、价值契约、标签版本和训练数据身份；加载旧 `GMNNUE01` 或架构不符的元数据会被拒绝。C++ 侧的新格式加载与对弈在工单 07/11。

## 环境与现有命令

Python 使用项目 .venv；依赖列表见[requirements.txt](../requirements.txt)。在已安装依赖的项目环境中，可检查当前入口：

~~~powershell
./.venv/Scripts/python.exe trainer/train.py --help
./.venv/Scripts/python.exe trainer/export.py --help
~~~

这些是当前实现的命令，不是新训练流水线的完成证明。新流程的默认参数和客户端操作随实施工单补充。
