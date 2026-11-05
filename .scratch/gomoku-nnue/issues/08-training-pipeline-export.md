# 08: Python 训练流水线与量化导出（里程碑 M4 后半）

**What to build:** 训练端完整闭环：mmap 加载 64 字节记录数据集、DataLoader 内实时 D4 八重对称增广、验证集按完整对局划分、PyTorch（RTX 4080）训练 NNUE 模型（450 → Accumulator 1024 → ClippedReLU → L1 15 → ClippedReLU → 输出 1，双输出头按行棋方选择）、损失为 MSE(tanh(score/400)) + CE(WDL) 混合，训练收敛后导出带魔数与版本号的 int8/int16 量化权重文件。特征索引逻辑必须与 C++ 端共享同一份权威规范。

**Blocked by:** 07（自我对弈 datagen 与 64 字节记录格式）

**Status:** ready-for-agent

- [ ] 数据集读取：mmap 高效加载，D4 实时增广（Renju 规则 D4 不变性下标签无需变换）
- [ ] 验证集按对局划分，无局面级泄漏
- [ ] 模型结构、双输出头、量化参数（int16 累加器 / int8 权重）与 spec 一致
- [ ] 训练脚本在 4080 上跑通并收敛（验证损失下降曲线可复查）
- [ ] 量化导出：权重文件带魔数/版本号，Python 端可回读自校验
- [ ] Python 特征索引与 C++ 端规范文档一致（为 09 的对齐测试做准备）
