# Phase 0 离线数据审计工具

在仓库根目录执行（仅 Python 标准库）：

```powershell
python -B -m scripts.datasets
python -B -m scripts.datasets plan --dataset sleepaccel
python -B -m scripts.datasets verify --dataset sleepaccel
python -B -m unittest discover -s tests -p "test_*.py" -v
```

默认命令为 `audit`，读取已入库 manifest 和 `data/raw` 文件状态。
`plan` 展示已知官方来源、版本、许可、页面报告大小、校验信息及未解决问题。
这两个命令不读取网络、不修改文件、不计算大文件哈希。`-B` 禁止 Python 生成字节码缓存。
`sample`、`download`、`prepare` 尚未实现；传入这些命令会报参数错误，不能将离线计划视为已取得数据。

`verify` 按 1 MiB 块计算 SHA-256，有官方 MD5 时另算 MD5；官方校验值只来自审计证据。
有精确字节数时检查大小，不把网页舍入显示值换算成伪精确值。
缺失文件、空数据文件、大小或官方哈希不匹配均失败；空文件清单也失败。
没有官方 SHA-256 时计算结果明确标为本地值。有 MD5 时仅证明与该 MD5 一致，不能冒称官方 SHA-256。
退出码：0 表示命令成功（verify 仅限清单内文件）；1 表示文件校验失败；2 表示 manifest、路径或读取错误。
`ok` 仅表示列出文件通过当前可用检查；没有官方哈希时也可能为 true。
`official_sha256_verified` 单独表示所有列出文件均与官方 SHA-256 相符；未来下载/准备门槛不能把 `ok` 当作官方完整性已验证。

manifest 的 `inventory_complete: false` 表明清单不完整；校验成功也不能声称整个数据集完整、标签正确或可训练。
`source_status` 区分目录确认、校验清单确认与推导候选。`reported_size` 为官方页面/文档显示值；`exact_bytes` 仅用于明确读到的字节数。
原始数据必须放在 manifest 指定的仓库内 `data/raw/...` 路径，禁止绝对路径、`..`、Windows 盘符以及指向树外的符号链接。
本阶段无自动下载、解压、格式解析、标签/时间对齐检查或模型结果。

真实样本结构审计（先检查官方 SHA-256，失败时拒绝解析）：

```powershell
python -B -m scripts.datasets.inspect_sleepaccel --kind labels
python -B -m scripts.datasets.inspect_sleepaccel --kind heart_rate
python -B -m scripts.datasets.inspect_sleepaccel --kind motion
```

该脚本统计实际列数、有限值、严格递增时间、间隔和原始标签频数，不套用阶段映射；原始标签4/-1的依据见 docs/LABEL_MAPPING_EVIDENCE.md。暂用于已审计的最小样本，读取完整样本到内存，不用于未经规模评估的全库处理。
