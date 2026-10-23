# NightLab Phase 0：可穿戴数据审计

核查日期：2026-10-23。仅阅读官方说明、目录和网页预览，未下载信号、标签、压缩包，未执行 Git 操作。网页目录显示值不等于已在本机校验文件。

## 结论

SleepAccel 是合理的首个可穿戴候选：31 人，Apple Watch 三轴加速度与心率配 PSG 标签，文件可按受试者选择。官方 PhysioNet 镜像已明确 ODC-By 1.0，并实际读到其 LICENSE.txt；NSRR 的“open access / no restrictions”是访问声明，不能据此写成无许可义务。NSRR 自身 LICENSE 正文与跨站文件同一性仍未核实。

建议下一阶段从一个受试者的 motion + heart_rate + labels 三文件开始；5383425 的 motion 为目录中最小（12.8 MB），但另外两文件大小及三文件校验值尚未读到，因此只是最小候选，不是已证明的最小完整样本。尚不能声称已取得或验证可训练数据。

## SleepAccel：证据与范围

- [NSRR 数据页](https://sleepdata.org/datasets/sleepaccel)及[官方文档固定版本](https://github.com/nsrr/sleepaccel-documentation/blob/d6a09dba76c38fd8c9e0429b095096b54224ee2e/pages/README.md)：31 人；时间统一为相对 PSG 开始秒数，0 是 PSG 开始，不是 Unix 时间；标签 0=Wake、1=N1、2=N2、3=N3、5=REM。不能将 5 自动当作 Unknown。文档没有给 4 的明确语义，解析时应保留并报告异常值。
- 文档由 GitHub API 读取，master tree SHA 为 `d6a09dba76c38fd8c9e0429b095096b54224ee2e`，正文 blob SHA 为 `47944ca5b398ae765d9a6c6a079d7504bfd5e424`，文档字节数 7,898；这两个 SHA 是 Git 对象标识，不是数据文件 SHA256。
- [NSRR 根目录](https://sleepdata.org/datasets/sleepaccel/files)确认 `heart_rate/`、`labels/`、`motion/`、`steps/`、`LICENSE.txt`、`SHA256SUMS.txt` 及完整 ZIP 存在。NSRR 文档标 ZIP 563 MB；[PhysioNet 1.0.0](https://physionet.org/content/sleep-accel/1.0.0/)标压缩 550.1 MB、未压缩 2.2 GB。数值不同不能当成版本差异证据，也不能未经哈希就宣称完全相同。
- [NSRR labels 目录](https://sleepdata.org/datasets/sleepaccel/files/labels)列 31 个标签文件，含 `5383425_labeled_sleep.txt`。NSRR 直连发生 TLS/超时，目录依靠官方网页搜索索引获取；这不是已成功完成下载链路的证明。
- [PhysioNet motion 目录](https://www.physionet.org/content/sleep-accel/1.0.0/motion/)实际列出 31 文件，均标 Modified `2019-08-18`。该日期是文件目录修改日期，不是采集日期。最小 `5383425_acceleration.txt` 12.8 MB，次小 `8258170_acceleration.txt` 12.9 MB；`1066528_acceleration.txt` 54.7 MB，`9618981_acceleration.txt` 380.9 MB。这些是页面舍入值，未取得精确字节数。
- 格式依据官方文档：motion 每行相对秒数+x/y/z（g）；heart_rate 每行相对秒数+bpm；labels 每行相对秒数+阶段；steps 为相邻时间戳之间步数。未核实原始分隔符、采样率分布、标签实际间隔、时间交集、缺失率及类分布。

## 许可及最小取得方案

[PhysioNet LICENSE 正文](https://physionet.org/content/sleep-accel/1.0.0/LICENSE.txt)网页显示 20,402 bytes，确认为 ODC Attribution License；§3.1 允许商业使用，§4 规定公开传播时保留许可与相关通知，公开产出需要说明来源。许可另区分数据库权利与内容自身其他权利，不能写成“所有用途无条件”。NSRR 文档还要求引用 NSRR 论文及 Walch 2019，并附其指定资助致谢；实施时应保留来源要求。

最小候选（后续获准取得时逐个下载，当前未执行）：

| 文件 | 官方原始路径 | 当前证据 |
| --- | --- | --- |
| motion | `https://physionet.org/files/sleep-accel/1.0.0/motion/5383425_acceleration.txt` | motion 目录确认文件名、12.8 MB、修改日期 |
| 心率 | `https://physionet.org/files/sleep-accel/1.0.0/heart_rate/5383425_heartrate.txt` | 文档命名规则推导；具体目录未成功读到 |
| 标签 | `https://physionet.org/files/sleep-accel/1.0.0/labels/5383425_labeled_sleep.txt` | NSRR 标签目录存在；镜像对应文件未单独核实 |
| 校验清单 | `https://physionet.org/files/sleep-accel/1.0.0/SHA256SUMS.txt` | 根目录确认 11.6 KB、2019-10-08；正文读取超时 |

建议从许可清楚的 PhysioNet 固定版本取得，先取 LICENSE 与 SHA256SUMS 元数据，再核对三个候选条目与大小后下载三文件，逐项核对 SHA256。一次下载全 ZIP 不属于最小方案。仅一个人可检验解析与时间对齐，不能用于可靠泛化评估；后续训练应按受试者划分，避免同人泄漏。上述是实施建议，不是已完成验收。

## BIDSleep：可选后续扩展

[PhysioNet v1.0.1 官方页](https://physionet.org/content/bidsleep-dataset/1.0.1/)列 47 人、253 夜，发布 2026-09-25；开放访问并要求遵守 ODC-By 1.0，完整未压缩 27.9 GB / ZIP 5.9 GB。参考标签来自 Dreem 2 EEG，含自动与专家修正标签，不能当作完整 PSG 金标准。根目录列 LICENSE 19.9 KB、SHA256SUMS 64.4 KB，均修改于 2026-09-24；未读到本版本 SHA256SUMS 正文，也未逐条核对所有夜。

[官方 README](https://physionet.org/content/bidsleep-dataset/1.0.1/README.md)说明每夜 `motion.csv`、`hr.csv`、`labels.mat`；心率无标题行，约 0.2 Hz；motion 有 Timestamp/x/y/z 标题。信号为绝对 Unix 秒，labels 的 recStart 以美国东部本地时间保存，须正确处理时区/夏令时。每标签 30 秒，0/1/2/3/4/5=Wake/N1/N2/N3/REM/Unknown，REM 编码与 SleepAccel 不同。

已核实的完整单夜候选（未称全库最小）：[Bidslab00/1 目录](https://physionet.org/content/bidsleep-dataset/1.0.1/Bidslab00/1/)三文件均修改 2026-03-20。网页预览显示 [hr.csv](https://physionet.org/content/bidsleep-dataset/1.0.1/Bidslab00/1/hr.csv) 114,314 bytes，[labels.mat](https://physionet.org/content/bidsleep-dataset/1.0.1/Bidslab00/1/labels.mat) 465 bytes，[motion.csv](https://physionet.org/content/bidsleep-dataset/1.0.1/Bidslab00/1/motion.csv) 103,763,554 bytes，合计 **103,878,333 bytes**。labels.mat 仅见元信息，未解析其内部变量；预览不等于全文件质量核验。

建议本轮优先 SleepAccel；BIDSleep 需独立适配标签编码与绝对时间，保留为多夜外部验证来源。两者都没有在本轮产生本地数据、模型结果或真实设备验收。

## 后续主线程复核：校验清单已读取

2026-10-23 内置浏览器实际显示 https://physionet.org/content/sleep-accel/1.0.0/SHA256SUMS.txt 全文，页面标 11,899 bytes。以下三个条目确认，取代前文“清单正文未取得”和候选 HR 路径仅推导的状态：

| 相对文件路径 | 官方 SHA-256 |
|---|---|
| heart_rate/5383425_heartrate.txt | 9b6af9046a4c516f948f307e8e130a823b244056ae705e2bb8abd89bb3e76eb8 |
| labels/5383425_labeled_sleep.txt | 10af34ea9a31f2284dafa034827e3050c0ea584b71107133e694715c5990df1b |
| motion/5383425_acceleration.txt | 40e461c8b2da44a7cd780cbae28b1d7dbcf7d32d9172898ae94c85e3f69b366e |

这证明官方清单包含这些路径与摘要，不证明本地信号已经下载或校验。使用浏览器 Download 保存清单时工具超时，未取得返回路径，故本地保存状态未确认。信号文件精确字节数仍待读取。
