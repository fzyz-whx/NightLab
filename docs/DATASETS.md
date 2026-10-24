# 数据来源、许可与引用

更新：2026-10-24。公开论文、数据访问声明、数据库许可、文件校验是不同证据。原始数据不进入本仓库的软件许可，也不进入Git。

| 数据 | 已核实版本与许可 | 来源与引用 | 当前使用状态 |
|---|---|---|---|
| 多模态睡眠声音 | V5，CC BY 4.0，官方2025-06-20更新 | [官方数据](https://doi.org/10.57760/sciencedb.19070)；Tao, Huang, Miao, Yang. Scientific Data 12,1263 (2025), [论文](https://doi.org/10.1038/s41597-025-05583-8) | 两份JSON与一份阶段CSV校验通过，声音头存在时间冲突，冻结事件训练 |
| PSG-Audio | V3，CC BY 4.0，页面2022-03-14更新 | [官方数据](https://doi.org/10.11922/sciencedb.00345)（现重定向10.57760/sciencedb.00345）；Korompili et al., Scientific Data 8,197 (2021), [论文](https://doi.org/10.1038/s41597-021-00977-w) | 一个已授权EDF分片及对应RML匹配官方MD5，时钟与区间通过最小结构检查；根README差异保留，扩大范围前继续审计 |
| SleepAccel | PhysioNet 1.0.0，ODC-By 1.0正文已读；NSRR与该版本逐文件同一性未验证 | [NSRR](https://sleepdata.org/datasets/sleepaccel)、[PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/)、[固定官方说明及其引用/资助要求](https://github.com/nsrr/sleepaccel-documentation/blob/d6a09dba76c38fd8c9e0429b095096b54224ee2e/pages/README.md) | 心率/标签SHA256已匹配，一人样本仅审计；加速度仍需完成 |
| BIDSleep | 1.0.1，ODC-By 1.0；见可穿戴报告的官方证据范围 | [官方记录](https://physionet.org/content/bidsleep-dataset/1.0.1/)、[DOI](https://doi.org/10.13026/rees-1092) | 可选，不下载全量 |

CC BY 数据的公开使用需署名并标注修改。ODC-By 使用应保留许可、通知和来源要求；不能将“开放访问”改写为无任何义务。SleepAccel研究报告需保留官方要求的NSRR和Walch等引用及资助致谢。
主声音论文的文章许可是CC BY-NC-ND 4.0，不能将它当作数据许可；源站数据记录实际是CC BY 4.0。

获取从 `python -B -m scripts.datasets plan --dataset <id>` 开始。已审计文件才能通过sample门槛；默认dry-run。完整信号、时间对齐、标签适用性需要单独验收，摘要匹配不代表研究任务成立。
