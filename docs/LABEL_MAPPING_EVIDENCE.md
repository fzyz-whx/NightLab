# SleepAccel 标签 4、-1 与五阶段统一映射证据

核查：2026-10-24。只读本地样本、官方说明和作者代码；仅新增本报告，未改变既有文档、数据或解析器，未下载大数据。

## 已解决的映射

**SleepAccel 原始 4 是作者命名的 n4（stage 4），不是 REM，也不是无效值；-1 是 unscored（未评分）。** 将它转换成现代五阶段时，可将原始 3、4 一起映射到 N3，并保留原始列用于追踪。5 始终对应本数据集 REM。-1 应保留未评分标志并从有监督目标/评分分母中排除，不得自动归入 Wake 或 NREM。

| 原始值 | 作者枚举 | 建议统一阶段 | 本地样本行数 |
| --- | --- | --- | ---: |
| -1 | unscored | 未评分，目标掩码为 false | 2 |
| 0 | wake | W | 40 |
| 1 | n1 | N1 | 52 |
| 2 | n2 | N2 | 391 |
| 3 | n3 | N3 | 34 |
| 4 | n4 | N3 | 190 |
| 5 | rem | REM | 269 |

因此五阶段有效标签共 976 行，W/N1/N2/N3/REM 为 40/52/391/**224**/269。这是全标签文件统计，未做信号交集裁剪，不是最终训练可用 epoch 数。

## 作者代码是直接证据

核对官方数据页指向的作者仓库 `ojwalch/sleep_classifiers`，当前 main 固定提交为 `ec9704f0290563016adbe48184365ab102daf9ea`。通过 GitHub API 获取代码 blob 并逐行阅读，没有运行上游程序。

1. [source/sleep_stage.py L4-L11](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/sleep_stage.py#L4-L11)：`n4 = 4`、`rem = 5`、`unscored = -1`。代码直接提供了官方概览遗漏的两项含义。
2. [source/preprocessing/psg/psg_converter.py L5-L27](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/preprocessing/psg/psg_converter.py#L5-L27)：字符串 4/N4 转为 n4，整数 4 转为 n4，-1 与 6 转为 unscored；字符串问号也转为 unscored。因此“未评分”含义有显式转换佐证，但本样本没有原始 6。
3. [psg_service.py L61-L83](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/preprocessing/psg/psg_service.py#L61-L83)：`read_precleaned` 打开 `data/labels/{subject_id}_labeled_sleep.txt`，按空格读取第二列，再调用上述整数转换。该映射确实用于发布标签文件，并非仅用于某个无关数据源。
4. [raw_data_processor.py L17-L29 与 L48-L68](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/preprocessing/raw_data_processor.py#L48-L68)：先读取发布标签并裁剪信号交集，生成有效 epoch 时还明确排除 `SleepStage.unscored`。
5. [sleep_labeler.py L11-L37](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/analysis/setup/sleep_labeler.py#L11-L37)：作者二分类将正值归 Sleep，三分类将 0 归 Wake、5 归 REM，其余归 NREM。**这两个函数本身并不保护 -1**，必须结合前面的有效 epoch 过滤理解；单独照搬会把未评分误算为 Wake 或 NREM。作者原有二/三分类流程未展示一个 3/4→N3 的现代五分类步骤。

## 官方说明与合并规则的证据边界

[NSRR 固定 README](https://github.com/nsrr/sleepaccel-documentation/blob/d6a09dba76c38fd8c9e0429b095096b54224ee2e/pages/README.md)本轮通过官方 GitHub blob `47944ca5b398ae765d9a6c6a079d7504bfd5e424` 再次读取；[PhysioNet 1.0.0 数据页](https://physionet.org/content/sleep-accel/1.0.0/)也检索到同样说明：只列 0/1/2/3/5，且概括为 0–5。它们遗漏了 4 的解释和 -1，不能据此把实测的 4、-1 判为损坏。此前报告中的“4 未有明确语义”只适用于这些概览，本报告用作者代码补足。

[AASM 官方《Scoring Manual v2.1 更新摘要》H. Scoring Stage N3, Note 1](https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf)明确现代 N3 取代旧 Rechtschaffen–Kales 的 stage 3 与 stage 4。故 **3、4→统一 N3** 有标准来源；应记录为 NightLab 的标准化规则，而非宣称原始文件已经采用现代五阶段编码。这是对既有阶段标签的合并，不是重新按 AASM 全部规则重判 EEG。

## 本地证据与未决点

本轮直接统计 `data/raw/sleepaccel/1.0.0/labels/5383425_labeled_sleep.txt`，978 行，与主线程报告一致。两个未评分行的相对 PSG 秒数分别为 21120、25710。本轮计算 SHA256 为 `10af34ea9a31f2284dafa034827e3050c0ea584b71107133e694715c5990df1b`，与当前 `data/manifests/sleepaccel.json` 记录的官方值一致；本轮没有重新远程取得校验清单。

仍未核实：其他 30 名受试者的原始标签全集；两个未评分 epoch 的具体原因；各受试者原始 PSG 实际采用何版评分手册；2019 年发布时与当前作者代码是否完全一致。当前代码足以确定作者对 4、-1 的处理含义，不能据此倒推未评分原因或所有历史评分细节。

建议解析器允许本数据集 -1/0/1/2/3/4/5，分别保存 `raw_stage` 与规范阶段及有效目标掩码；任何其他原始值应单独报告。暂未修改解析器或运行训练。
