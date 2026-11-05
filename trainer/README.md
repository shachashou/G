# trainer — Python 训练端

- `binpack.py`（工单 07）：自我对弈二进制格式的 **Python 读端**，纯标准库（`struct` / `mmap`），提供文件头魔数与版本校验、64 字节记录解码、按对局切分、全量一致性校验和分片合并。格式的权威定义见 `tools/README.md`。工单 08 的 DataLoader 直接在此模块上加 D4 增广与 batch，不重复实现格式解析。

  ```powershell
  .venv\Scripts\python.exe trainer\binpack.py info data\raw\round1\train.binpack
  .venv\Scripts\python.exe trainer\binpack.py validate data\raw\round1\train.binpack
  ```

  `Record.feature_indices()` 返回 450 维颜色绝对特征的激活索引（黑 0..224、白 225..449），与 C++ 端特征提取共用同一份索引规范；`BinpackReader` 支持 `mmap` 随机访问 `__getitem__` / `iter_records` / `iter_games`。`tools/datagen.py` 的 `merge` / `validate` 复用本模块。

- 其余部分（模型、DataLoader、训练脚本、量化导出、`requirements.txt`）随工单 08 填充。

项目 Python 一律走独立虚拟环境（`py -3.12 -m venv .venv`），不要直接用系统解释器。
