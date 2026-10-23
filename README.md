# NightLab

本地优先的夜间声音与可穿戴睡眠时序研究工作台。仅供研究，不用于医疗诊断。

目前处于 Phase 0 数据审计，尚无模型、实验结果或产品 UI；未下载原始数据。
完整需求见 [项目规格](docs/PROJECT_SPEC.md)。

研究路线：声音事件与抗噪实验、可穿戴睡眠阶段预测分别训练与评价；采用受试者级划分，个人数据仅用于探索。

- [数据审计](docs/DATA_AUDIT.md)
- [数据来源与许可](docs/DATASETS.md)
- [开发记录](docs/PROJECT_LOG.md)

环境已确认：Windows、Python 3.12.10、Node 24.22.1、Git。尚未安装项目依赖或检查 CUDA。
目前只需阅读文档，无应用启动命令。后续逐阶段补充运行、测试、真实结果与截图。

需求完善与研究决策见 [DECISIONS](docs/DECISIONS.md)，可穿戴文件级证据见 [WEARABLE_AUDIT](docs/WEARABLE_AUDIT.md)。

## 已实现的离线工具

Python 3.12 标准库即可运行，无需安装依赖：

```powershell
python -B -m scripts.datasets audit
python -B -m scripts.datasets plan --dataset sleepaccel
python -B -m scripts.datasets verify --dataset sleepaccel
python -B -m unittest discover -s tests -p "test_*.py" -v
```

原始文件未取得时 verify 应失败；这不是模型验证。当前仅支持离线审计、计划、完整性检查，下载与数据准备尚未实现。详见 [工具说明](scripts/datasets/README.md) 与 [研究方案](docs/RESEARCH_PLAN.md)。
