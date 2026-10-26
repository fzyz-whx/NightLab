# SleepAccel31 v3 原始信号规范化

2026-10-26，主线程在看到完整质量审计、尚未训练扩展模型时审定此策略。仅适用于新 `sleepaccel-cohort-31`；旧三人划分、v2特征、旧deep输入及其报告保持冻结。

## 证据与理由

完整93文件均匹配官方大小与SHA256。原严格审计完整覆盖31人，`collapse_identical`仅接受16人，拒绝15人；这不等于后15人的原始数据不可使用。历史报告留在本地忽略路径 `data/processed/sleepaccel-cohort-31/quality-audit.json`，SHA256为 `cb5e00d06e015723089877ef37b331442962e7c5a60d29a2a11ed36c22e06cf0`，保留原结果，不以v3覆盖。

[官方固定README](https://github.com/nsrr/sleepaccel-documentation/blob/d6a09dba76c38fd8c9e0429b095096b54224ee2e/pages/README.md)说明时间是相对PSG开始的秒数，没有承诺原始文件行序递增。[作者固定代码utils.py104–107](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/utils.py#L104-L107)先对完整行 `np.unique(axis=0)`，再按原timestamp排序；[心率](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/preprocessing/heart_rate/heart_rate_service.py#L12-L16)和[运动](https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/preprocessing/motion/motion_service.py#L12-L16)的 `load_raw` 都调用该函数。代码固定提交 `ec9704f0290563016adbe48184365ab102daf9ea`。

实际11个motion失败文件均为行序回退，但timestamp本身唯一；2个HR文件有完整块不相同/长度不等的重复，但重复timestamp的值全部相同；另2个HR文件各有1个同时间不同值冲突，均在label范围外。全部31人的label范围内冲突数为0。不能把这些行序问题一律推断为不同clock轨迹。

## 冻结规则

1. Labels先按原顺序严格检查：有限值、正确列数、已知编码、严格递增且连续30秒。坏标签拒绝，不排序或修复。原始4并入N3，-1保留在timeline但排除监督目标，沿用现有提取函数。
2. Signals拒绝空行表或非有限值；完整timestamp/value行精确去重，并按原timestamp递增排序。不舍入，不修改信号值，不根据stage值、训练结果或标签类别选择信号点。
3. 显式裁剪到 `[第一条label时间, 最后一条label时间+30)`，包括-1未评分epoch。报告raw样本数、重复行数、时间回退数、域内外冲突组数及裁剪前后数量。窗外同时间冲突保留计数后随时间窗裁剪。
4. 裁剪后同timestamp不同值仍存在则拒绝；空信号拒绝。所有信号必须通过原严格递增校验，随后原classical/deep提取函数以 `reject` 策略运行。不平均、插值、填补或推断clock偏移。
5. 所有操作使用新数组，不写原始文件。下游提取器的 `original_input_samples` 等字段只指规范化后的输入；另存 `raw_cleaning`，避免将裁剪后数量误称为raw样本数。

## 输出与验收

执行入口是 `python -m scripts.datasets.prepare_sleepaccel31 --execute`；默认仅校验元信息，不写文件。执行重新关联固定intake receipt/原始plan并核对93文件大小与官方SHA。单worker、数值线程1、RAM余量检查，无网络或GPU。

Classical保存到 `data/processed/sleepaccel-cohort-31/{subject}/features-v3/`，包含 `features.npz`、完整 `timeline.npz` 与metadata；feature版本为 `sleepaccel-classic-v3`。Deep保存到 `data/processed/sleepaccel-cohort-31/deep-inputs-v3/{subject}/`，版本为 `deep-inputs-v3`，包含values/masks与同样的监督时间窗。缺失values保留NaN并由masks标注。数据不进入公共Git提交。

记录目录先在同一父目录完整写入临时目录，再原子改名发布；已存在输出仅在来源摘要、artifact摘要、数组内容相同时复用，来源冲突拒绝覆盖。两个根corpus仅在31人全部成功并通过旧三人parity后标记complete；中途失败保留incomplete，不能静默跳人。18/6/7身份分组保持原批准值。

旧三人必须与冻结v2 classical的X/y/原始标签/subject/time，以及旧deep的values/masks/y/subject/time/supervised mask逐项完全相同，NaN按相同缺失处理；允许raw数量/裁剪统计不同。每人metadata记录真实parity结果和旧artifact SHA。这是输入兼容性验证，不是模型成绩、生产或医学验证。

## 本次实际验收

真实执行已完成31/31人，classical与deep两个corpus均为 `complete=true`、无失败记录。使用现有 `scripts.train.baselines.load_corpus` 实际加载成功：classical矩阵为26,773行、13列，受试者31人；冻结train/validation/test分别为16,126/4,422/6,225个已评分epoch。没有拟合模型。

31份deep NPZ均逐份匹配artifact SHA、身份、字段与形状，共26,773×5×30个values；缺失值保留817,884个NaN并以masks对应标注。规范化统计的label域内同时间冲突为0组，窗外冲突为2组。旧1066528/5383425/8258170三人的classical与deep均通过上述exact parity；原严格审计报告摘要未变。

单文件合成测试 `tests/test_sleepaccel31_v3.py`：8 passed、8 subtests passed。覆盖乱序/完全相同重复、域外冲突裁剪及域内拒绝、坏label不修复、raw不变、缺失与字段语义、原子发布/冲突复用拒绝、中途失败不complete、dry-run不写文件。

本次classical corpus SHA256为 `9f8b59db871190ac1142c50b811d227d80d9568efc45866dbe1bf99b0550c920`；deep corpus SHA256为 `af51a76693c38b0e1cd73347fd0c47e0764cfc63d640804cca0aa59be3c03ef7`。来源摘要包含新规范化器/prepare、现有特征与deep提取函数、manifest和intake receipt；公共plan仅保存匿名聚合与摘要，原始信号/特征数组仍仅本地忽略。

旧CNN入口仍硬编码三人pilot与旧版本provenance校验，新31人deep输入需要独立的v3读取入口。此次未修改旧训练代码，也未运行CNN、GPU测试或训练。

后续状态（2026-10-26）：上述“没有训练”专指数据准备验收时点。独立31人loader、六个模型与同seed复跑现已完成，见 [WEARABLE31_RESULTS.md](WEARABLE31_RESULTS.md)；旧三人入口仍冻结。
