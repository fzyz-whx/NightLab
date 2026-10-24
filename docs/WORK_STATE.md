# 压缩后接续：NightLab

更新：2026-10-24。这是进度索引，完整任务仍以PROJECT_SPEC.md及用户后续更新为准，整体goal保持进行中。

## 立即遵守

- 读取根AGENTS.md：GPU共享、功耗受限；默认不启动GPU任务。CPU线程<=2、准备worker默认1，长训练等待明确资源窗口。当前无自动跨项目队列。两个批处理CLI的默认worker已降为1，README示例也使用1；dry-run会显示worker。
- 用户允许改进原始AI任务书，但需记录证据与调整；仓库已明确要求公开。
- 只按阶段推进；本地优先、非诊断、数据不进Git、真实测试/提交/推送。子代理沿用用户要求的Sol低/中，只有独立且值得并行的任务才委派；主代理保留研究判断和整合。

## 已完成并提交

- Phase0最小数据审计，见PHASE0_ACCEPTANCE.md。主声音ScienceDB19070 V5两样本事件跨度超过WAV长度，冻结事件训练；转任务书备用PSG-Audio。
- Phase1传统基线试点，见PHASE1_ACCEPTANCE.md、EXPERIMENTS.md。核心提交31a0dfcfca01ea40dd9599b735a2b73ff52156a6；追记020db1f7ed0f11d1fef620d05893224b2d1a5fdf。两者已推送，远端hash已核验。
- CPU环境：.venv，CPython3.12 AMD64；requirements-cpu.lock与25个wheel官方PyPI摘要核验，scripts/setup_cpu.ps1实际通过，pip check通过。全局Python不修改。
- PSG十人各第二小时，20文件7109932234字节；新增6398951410字节经用户明确批准，所有官方MD5匹配。seed42/PCG64受试者6/2/2。6321个明确标签的10秒窗/5秒hop，67特征。
- SleepAccel三人9文件84815494字节，全部按PhysioNet1.0.0 SHA256匹配；追加两人小样本71259543字节。NSRR官方替代源许可与选中文件身份已核验，未取完整31人ZIP。
- 1066528心率16617行是5539行完整信号重复三次，官方摘要匹配；显式v2仅折叠逐项相同完整块，默认仍拒绝时间回退。WEARABLE_CLEANING.md有证据。三人2900有效30秒epoch，13特征。
- 声音RF测试Macro-F1=.595983、balanced accuracy=.678096、AP=.283317、目标窗Precision=.302372；多数类Macro-F1=.452814。
- 可穿戴融合RF测试Macro-F1=.209617、balanced accuracy=.250946；多数类Macro-F1=.095604。仅HR/仅accel/融合均真实运行，负面结果完整公开。每集合仅1人，不能声称总体效果。
- 两条主基线固定种子重训指标精确一致，自己生成的保存模型加载后预测与重训模型逐项一致；docs/results/baseline-reproducibility.json含可信本地模型SHA。
- 原始/特征/权重/runs均忽略，公开docs/results及artifacts/experiments是实际聚合结果。README有复现命令，架构见ARCHITECTURE.md。尚无API/UI/个人导入删除/深度学习结果，软件LICENSE尚未确定。

## 当前Phase2代码检查点

最新代码已主线程检查并运行完整测试：136 passed、1 Windows真实symlink权限skip、54 subtests passed（19.64秒）。测试时仅CPU，OMP/MKL/OpenBLAS进程线程均限制2。真实事件评价已完成，神经模型结果尚无。

- backend/src/nightlab/audio/events.py：固定5秒bin，取覆盖bin中心的窗口概率均值，阈值>=.5，合并相邻正bin。禁止缺覆盖bin。evaluate_events用一对一IoU>=.5，先最大匹配数量再最大IoU；返回TP/FP/FN、P/R/F1、FP/hour。边界截断参考参与检测，但不进入起止误差均值。38个手算/边界测试。
- backend/src/nightlab/deep/inputs.py：真实STFT的32x100 log-mel图，不能用旧67维摘要伪造图；可穿戴5个1Hz观测值通道+2个观测mask，缺失保持NaN，标签/重复块沿用显式v2策略；MaskedChannelStandardizer只拟合训练观测，transform时缺失填0并保留mask。12测试通过。没有Torch依赖。
- 两名子代理已结束，仅留下以上代码和测试；无子代理训练任务。

## 连续事件评价已完成

- `docs/EVENT_PROTOCOL.md`在计算真实指标前固定5秒网格/概率0.5/一对一IoU0.5。旧模型不重训、不按测试结果调参。
- `scripts/datasets/dense_audio.py`只补齐固定验证/测试4条录音：共2876个窗口，复用2546、补算330；新窗不补监督标签。原始官方MD5/本地SHA256、旧特征/源码摘要、EDF/RML时钟、时间轴与身份核对；每条重算一个已存窗口，逐项相同。缓存位于原cohort录音下`psg-continuous-classic-v1`。
- `scripts/evaluate/audio_events.py`只加载与`baseline-reproducibility.json`摘要相符的自建模型；所有文件先验摘要再joblib。旧声音报告没有后来添加的feature_subset字段，以67个完整特征名和数量核对；不改旧报告/模型。
- 真实`runs/psg-events-pilot-v1`及`-repro`除耗时外所有报告字段完全一致；每个保存匹配独立复核IoU与一对一索引。全窗/bin概率与参考/预测时间轴保持本地忽略。
- RF验证TP/FP/FN=28/58/76，事件F1=.294737，29FP/hour；测试10/81/72，F1=.115607，40.5FP/hour。多数类两集合均无预测、召回0。验证104参考/2小时、测试82参考/2小时，这四条无截断目标。
- RF测试10个成功匹配的起/止MAE=2.65/5.05秒，不能用它掩盖72个漏检。FP中37个无交集、44个IoU不足；FN中20个无交集、52个不足。不能按测试分数改阈值。
- 公共真实摘要/失败统计/复跑凭据在`docs/results/*events*`和`event-reproducibility-v1.json`；PNG/SVG及来源在`artifacts/experiments/psg-events-pilot-v1`，图已查看。不是产品UI截图，也不是Phase2整体完成。

## 环境检查与失败状态

- 实测GPU为NVIDIA RTX5060 Laptop，nvidia-smi驱动617.42、8151MiB；用户随后明确共享/低功耗约束，不能据旧空闲采样启用GPU。
- 系统Python的Torch导入发生DLL错误，未修全局环境。项目venv仍无Torch。
- 已查PyTorch官方安装入口及CPU索引。页面选择器文本可能陈旧，不把“最新版本”当事实。实际官方索引可用的CPU轮子：
  - torch2.14.1+cpu，CPython3.12 WindowsAMD64，123996008字节。
  - https://download-r2.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-win_amd64.whl
  - SHA256 c52adc3bd526ff7fc5bdcc58fc4a76e6214d09951b646ffe5256ebd7ba60cef1。
- pip --dry-run获取该轮子在约33MB处读超时，进程已终止，安装未发生，DL锁尚未生成。不要假设pip临时文件可续传。后续可在低负载窗口用明确partial获取/校验，或先继续事件评价与软件工作。
- 当前没有NightLab训练/下载进程；不要重启旧工具session。以新的进程/工具状态为准。

## 接下来先做什么

1. 完成低负载PyTorch CPU路径；只改本项目venv，global Torch失败不处理，不和其他项目并发安装/训练争抢资源。下载明确partial与校验，不能假设旧pip临时文件仍可续传。
2. 实现候选音频轻量2D-CNN、可穿戴1D-CNN及输入缓存/可信checkpoint。训练/归一化仅用train，checkpoint仅看validation，test不调参。长训练先报预算、等明确资源窗口；GPU仍默认禁用。
3. 当前dense只覆盖验证/测试4人，没补训练6人；CNN图像需要原始STFT，不能用67维旧摘要伪造。可按监督窗口准备真实输入，保留掩码；不要因为事件评价又重训旧基线。
4. 后续仍须抗噪对照、神经基线/消融/失败分析、API/React/SQLite、CSV/JSON/个人音频导入删除、真实UI截图、完整许可/引用/讲解稿和交付验收。不要收缩goal成仅有CLI。

## 避免重读/重做

- 从此文件、AGENTS.md、git status/log、对应源代码与实际产物继续，不要加载整段旧聊天或其他项目记忆。
- docs/EXPERIMENTS.md和docs/results保留冻结分数。旧声学/可穿戴基线不要因后续测试分数修改。
- .venv/Scripts/python.exe -m pytest -q可核验当前完整套件；长任务只能等待已确认存活的具体handle，不能凭partial或旧日志宣布在运行。
- 已从系统压缩后的交接重新核验当前文件、进程和Git；代理本身没有调用强制compaction接口。这里持续保留精简接续记录。
