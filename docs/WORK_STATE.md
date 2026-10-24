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

新增代码已主线程阅读并运行完整测试：122 passed、1 Windows真实symlink权限skip、54 subtests passed。测试时仅CPU，OMP/MKL/OpenBLAS进程线程均限制2。还没有真实事件级指标或神经模型结果。

- backend/src/nightlab/audio/events.py：固定5秒bin，取覆盖bin中心的窗口概率均值，阈值>=.5，合并相邻正bin。禁止缺覆盖bin。evaluate_events用一对一IoU>=.5，先最大匹配数量再最大IoU；返回TP/FP/FN、P/R/F1、FP/hour。边界截断参考参与检测，但不进入起止误差均值。38个手算/边界测试。
- backend/src/nightlab/deep/inputs.py：真实STFT的32x100 log-mel图，不能用旧67维摘要伪造图；可穿戴5个1Hz观测值通道+2个观测mask，缺失保持NaN，标签/重复块沿用显式v2策略；MaskedChannelStandardizer只拟合训练观测，transform时缺失填0并保留mask。12测试通过。没有Torch依赖。
- 两名子代理已结束，仅留下以上代码和测试；无子代理训练任务。

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

1. 完成声音dense缓存/事件评价runner，现有baseline不重训。复用已校验的6321个特征，只计算旧监督准备排除的869窗；必须覆盖每小时全部719窗，不能把无标签窗当负例或填0。
2. 验证原始输入/旧NPZ摘要与特征源码指纹。解码统一5秒/.5，IoU.5；RML四目标类型裁剪到实际分片，标记边界截断。10份RML未见重复目标事件或非目标呼吸类型；1006、1014有边界目标，验证/测试四人无边界目标。
3. 本地模型只加载自己生成且摘要与docs/results/baseline-reproducibility.json一致的joblib。输出事件级指标、每小时误报、未截断匹配的起止误差，模型预测与参考标注分开。
4. 完成低负载PyTorch CPU路径和小CNN训练前，先给资源预算；GPU训练继续等明确窗口。候选架构尚未实现：音频轻量2D-CNN；可穿戴1D-CNN。训练/归一化仅用train，checkpoint仅看validation，test不调参。
5. 后续仍须抗噪对照、事件级/失败分析、API/React/SQLite、CSV/JSON/个人音频导入删除、真实UI截图、完整许可/引用/讲解稿和交付验收。不要收缩goal成仅有CLI。

## 避免重读/重做

- 从此文件、AGENTS.md、git status/log、对应源代码与实际产物继续，不要加载整段旧聊天或其他项目记忆。
- docs/EXPERIMENTS.md和docs/results保留冻结分数。旧声学/可穿戴基线不要因后续测试分数修改。
- .venv/Scripts/python.exe -m pytest -q可核验当前完整套件；长任务只能等待已确认存活的具体handle，不能凭partial或旧日志宣布在运行。
- 上下文系统压缩尚未由代理触发；当前工具没有强制compaction入口。这里是可恢复交接记录，不是假称压缩已执行。
