# Phase 0 最小数据可行性验收

日期：2026-10-24。状态：**通过最小可行性审计，进入Phase 1准备。整体项目未完成。**

| 要求 | 权威证据/实际结果 |
|---|---|
| 完整阅读并归档原规格 | docs/PROJECT_SPEC.md；源文件与副本SHA256相同，初始提交记录 |
| Git真实初始化和远端 | main，真实逐步提交；公开仓库fzyz-whx/NightLab，推送后逐次比对GitHub分支hash |
| 数据审计、许可、引用 | DATA_AUDIT、DATASETS、SOUND_AUDIT、PSG_AUDIT、WEARABLE_AUDIT与LABEL_MAPPING_EVIDENCE；区分数据与论文许可 |
| 实际文件和大小/下载路径 | 主声音380数据文件元信息；实际取得PSG EDF/RML与SleepAccel三文件；manifest列精确字节和官方原始路径 |
| 实际标签粒度/字段 | 主声音JSON实际evnet_start；PSG Event属性为Start/Duration，人工和机器阶段分开；SleepAccel实际数字编码经作者代码核实 |
| 实际时间戳和同步可行性 | 主声音01/02时间冲突明确冻结；PSG分片时钟/时长相对RML匹配3600秒；SleepAccel976已评分epoch两类信号都有采样 |
| 完整性校验 | PSG两个文件匹配官方MD5；SleepAccel三文件匹配官方SHA256；原始数据不进Git |
| 下载计划与授权 | 原先禁止直接大规模下载；本次在完成元数据/许可核查后，经用户明确批准仅一个710738976字节EDF，其余大型文件未取；取代原规格此处“尚未下载大数据”的绝对表述，未扩大授权 |
| 工具检查 | 单元测试42项：41通过、1真实symlink因Windows权限跳过；独立越界测试通过；下载/错误响应/坏摘要/取消/续传和实际结构/覆盖均检查 |
| 研究边界 | 两条路线独立；一位受试者/一小时只证明解析可行，不能报告subject-wise泛化或医学准确率 |

## 已执行的实际验收命令

- `python -B -m scripts.datasets verify --dataset sleepaccel`：三文件官方SHA256均匹配。
- `python -B -m scripts.datasets.inspect_sleepaccel --kind motion`：298000行、4列、严格递增时间。
- `python -B -m scripts.datasets.inspect_sleepaccel --kind alignment`：976已评分epoch均有心率和motion采样，明确报告缺口与范围外样本。
- `python -B -m scripts.datasets.inspect_psg`：已校验EDF/RML，20通道、3600秒、46个完整呼吸事件，实际读取两路48kHz短记录。
- `python -B -m scripts.datasets.inspect_sciencedb --subject 01/02`：均返回1，明确复现主声音冲突；这个负面结果被保留，不假装通过。

## Phase 1前必须处理

选定PSG-Audio作为声音基线来源，先在当前样本完成解析/特征准备；扩大到多受试者时，另列文件、大小和来源清单，等待相应大数据授权。不得用单人随机窗口切分冒充受试者泛化。
可穿戴选择缺口/覆盖质量规则与传统特征时只参考训练数据；保留未评分掩码和原始标签。正式实验须有受试者级划分、完整依赖记录和结果产物。
PSG目录重复条目与根README摘要差异、主声音同步异常继续保留为来源限制；不能把全库或全部受试者宣布已验证。
软件LICENSE、CITATION.cff、完整产品架构、依赖锁定、模型/UI/删除/截图/交付仍属于后续工作，本阶段没有声称完成。
