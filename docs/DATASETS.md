# 数据来源、许可与引用

更新：2026-10-26。公开论文、数据访问声明、数据库许可、文件校验是不同证据。原始数据不进入本仓库的软件许可，也不进入Git。

| 数据 | 已核实版本与许可 | 来源与引用 | 当前使用状态 |
|---|---|---|---|
| 多模态睡眠声音 | V5，CC BY 4.0，官方2025-06-20更新 | [官方数据](https://doi.org/10.57760/sciencedb.19070)；Tao, Huang, Miao, Yang. Scientific Data 12,1263 (2025), [论文](https://doi.org/10.1038/s41597-025-05583-8) | 两份JSON与一份阶段CSV校验通过，声音头存在时间冲突，冻结事件训练 |
| PSG-Audio | V3，CC BY 4.0，页面2022-03-14更新 | [官方数据](https://doi.org/10.11922/sciencedb.00345)（现重定向10.57760/sciencedb.00345）；Korompili et al., Scientific Data 8,197 (2021), [论文](https://doi.org/10.1038/s41597-021-00977-w) | 十人各第二小时20文件匹配官方MD5，时钟/区间已核；固定6/2/2试点与域差异保留 |
| SleepAccel | PhysioNet 1.0.0，ODC-By 1.0；NSRR归档93个选定目标成员全部匹配官方SHA | [NSRR](https://sleepdata.org/datasets/sleepaccel)、[PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/)、[固定官方说明及其引用/资助要求](https://github.com/nsrr/sleepaccel-documentation/blob/d6a09dba76c38fd8c9e0429b095096b54224ee2e/pages/README.md) | 31人HR/motion/labels全93文件验收、独立18/6/7实验已完成；steps未提取 |
| BIDSleep | 1.0.1，ODC-By 1.0；见可穿戴报告的官方证据范围 | [官方记录](https://physionet.org/content/bidsleep-dataset/1.0.1/)、[DOI](https://doi.org/10.13026/rees-1092) | 可选，不下载全量 |

CC BY 数据的公开使用需署名并标注修改。ODC-By 使用应保留许可、通知和来源要求；不能将“开放访问”改写为无任何义务。SleepAccel研究报告需保留官方要求的NSRR和Walch等引用及资助致谢。
主声音论文的文章许可是CC BY-NC-ND 4.0，不能将它当作数据许可；源站数据记录实际是CC BY 4.0。

获取从 `python -B -m scripts.datasets plan --dataset <id>` 开始。已审计文件才能通过sample门槛；默认dry-run。完整信号、时间对齐、标签适用性需要单独验收，摘要匹配不代表研究任务成立。

## SleepAccel正式引用与致谢要求

2026-10-24主线程重新读取固定官方README blob（47944ca5b398ae765d9a6c6a079d7504bfd5e424），确认需引用：

- Zhang GQ, Cui L, Mueller R, et al. The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc. 2018;25(10):1351–1358. [doi:10.1093/jamia/ocy064](https://doi.org/10.1093/jamia/ocy064)。
- Walch O, Huang Y, Forger D, Goldstein C. Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device. Sleep. 2019;42(12):zsz180. [doi:10.1093/sleep/zsz180](https://doi.org/10.1093/sleep/zsz180)。

官方还要求在正式产出的致谢中保留NSRR的NIH/NHLBI资助说明（R24 HL114473、75N92019R002），准确英文措辞见上表固定README，不用概括的“公开数据无使用限制”替代它。

## 当前试点数据范围（2026-10-24更新）

PSG-Audio已从原先一小时扩展为用户明确批准的10人各第二小时：20文件总7109932234字节，全部官方MD5匹配，受试者划分6/2/2在窗口处理前冻结。
SleepAccel当前选择3人共9文件，总84815494字节，全部匹配PhysioNet1.0.0官方SHA256；只获取小规模试点，未取完整31人ZIP。NSRR官方LICENSE.txt已实读并与PhysioNet逐字节匹配，原先“NSRR许可未读取”的状态已解决；未验证整个两个站点全库逐文件相同。
两条传统基线和可穿戴输入消融均已真实运行，指标与局限见EXPERIMENTS.md。临床数据和派生特征均保留本地。

正式致谢保留官方要求的文字：

> The National Sleep Research Resource was supported by the U.S. National Institutes of Health, National Heart Lung and Blood Institute (R24 HL114473, 75N92019R002).

## 2026-10-26：当前已验收范围

上方“当前试点”是当时历史状态。现在PSG10人20文件及SleepAccel31人93个HR/motion/labels文件均已校验；SleepAccel完整ZIP576850847字节下载完成，逐成员CRC/官方SHA通过（84新增/9复用）。NSRR归档续传身份由完整既有前缀匹配及最终93目标成员官方SHA核验支持；steps等extras不提取，不宣称整个镜像全库已核。许可正文也匹配官方SHA。详见WEARABLE_EXPANSION_PLAN.md和WEARABLE_CLEANING_V3.md，六模型结果见WEARABLE31_RESULTS.md。

软件MIT不覆盖这些数据库或第三方文章。项目未再分发原始数据/权重，正式文章仍保留上述数据引用及NSRR资助致谢。
