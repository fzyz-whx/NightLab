# NightLab

本地优先的夜间声音与可穿戴睡眠时序研究工作台。仅供研究，不用于医疗诊断。

**当前：Phase 0 数据审计。** 已取得一人的可穿戴心率/标签、两份主声音标注，以及用户批准的一个PSG-Audio小时分片及其RML。尚无模型、正式实验结果或产品UI。

声音主候选发现时间对齐冲突，已用PSG-Audio完成最小结构检查；可穿戴加速度仍需完整下载并校验。当前结果和限制见[数据审计](docs/DATA_AUDIT.md)。

研究路线：声音事件与抗噪实验、可穿戴睡眠阶段预测分别训练与评价；按受试者划分，个人数据仅用于探索。原始数据不上传GitHub。

## 当前可运行工具

Windows / Python 3.12标准库，无需安装第三方依赖，在仓库根目录执行：

```powershell
python -B -m scripts.datasets audit
python -B -m scripts.datasets plan --dataset sleepaccel
python -B -m scripts.datasets verify --dataset sleepaccel
python -B -m scripts.datasets.inspect_sleepaccel --kind labels
python -B -m scripts.datasets.inspect_sciencedb --subject 01
python -B -m unittest discover -s tests -p "test_*.py" -v
```

`verify`在文件缺失/不完整时应失败；声音审计在已发现的时间冲突下返回1，这些不能计作模型失败或准确率。样本下载默认dry-run，明确执行前展示许可、来源、精确大小与校验方式，详见[获取工具说明](scripts/datasets/README.md)。全量与大于500MB数据另行确认。

## 研究与工程文档

- [原始任务书](docs/PROJECT_SPEC.md)与[需求完善/决策](docs/DECISIONS.md)
- [数据来源、许可及引用](docs/DATASETS.md)
- [声音实测审计与时间冲突](docs/SOUND_AUDIT.md)
- [PSG-Audio已校验样本](docs/PSG_AUDIT.md)与[隐私和数据流](docs/PRIVACY.md)
- [可穿戴审计](docs/WEARABLE_AUDIT.md)与[标签映射证据](docs/LABEL_MAPPING_EVIDENCE.md)
- [研究方案与训练门槛](docs/RESEARCH_PLAN.md)
- [真实开发记录](docs/PROJECT_LOG.md)

环境已确认Python 3.12.10、Node 24.22.1、Git；尚未安装模型依赖或检查CUDA。下载/审计工具不是完整应用，后续逐阶段补充安装启动、真实结果、截图和局限。当前不生成健康评分，也没有医疗准确率声明。
