# 本机环境与数据重置验收

日期：2026-11-10。对应主规格的环境准备、教师启动数据链路与新网络部署能力；不表示自主训练与晋级工单已完成。

## 环境

- 基础解释器：官方 Python 3.12.10，`D:\Python312\python.exe`；`.venv/pyvenv.cfg` 已指向该目录。
- 显卡：AMD Radeon RX 7900 XTX；PyTorch `2.9.1+rocm7.2.1`，HIP `7.2.53211-158bd99533`，`torch.version.cuda=None`。
- 已实测 GPU 前向、反向、Adam 优化器、新价值网络训练与导出；ROCm 通过 `cuda` 设备名执行。
- 项目内工具链：cmake 3.31.10、ninja 1.13.2、ziglang 0.17.1、Rust 1.99.1 GNU host；GUI 为 PySide6 6.9.3。
- Rapfi 250615 已安装于 `tools/rapfi`，可执行文件、Renju 黑白权重和配置通过 `tools/rapfi.lock.json` 的 SHA-256 校验。
- 下载通过本机 HTTP 代理 `http://127.0.0.1:10808`，仅设置进程环境变量。
- 三种后端的安装区别及依赖入口见[README](../../README.md#环境与构建)。仅 AMD 后端在本次机器上实测，NVIDIA / CPU 提供独立安装配置。

## 数据

按用户要求删除 `data/` 和 `weights/` 的旧实验产物，保留 `.gitkeep`。验收测试结束后再次清理 GUI 测试生成的数据，并初始化：

- `data/teacher/samples.sqlite3`：0 条记录，完整性校验通过。
- `data/pool/pool.sqlite3`：0 条样本、0 次采集，完整性校验通过。
- `weights/`：无模型；没有默认正式版本，后续从零采集、训练。

验收样本和候选模型仅保存在忽略目录 `build/acceptance/`，不进入正式样本池或模型目录。`build.py --clean` 会删除这些可再生成的验收产物。

## 验收结果

- 全新 Zig / Rust 构建完成，后续增量构建也完成。修复了构建脚本要求 CXX-only 项目必须缓存 C 编译器、导致第二次构建误报工具链不符的问题。
- `ctest --test-dir build --output-on-failure -j 2`：31/31 通过，无跳过，包含真实教师、GUI、模型加载与对局测试。
- 真实教师采集：3 局、16 个待筛选局面，5 条有效样本（训练 3、验证 2），D4 泄漏为 0。过滤 mate 8、VCF 已证明 1、无评分 2；这是链路验收数据，不用于吞吐或棋力结论。
- 上述真实样本在 AMD GPU 上训练 20 步，导出 `GMVALUE1` 通过冻结容差；C++ 引擎成功加载，返回正确架构、规则和候选模型身份，未晋级。
- `pip check` 与 `git diff --check` 通过。

本地详细产物：`build/acceptance/environment.json`、`real-pool.sqlite3`、`real-amd-model/train_summary.json`、`real-amd-model/fidelity.json`、`loaded-model.json`。
