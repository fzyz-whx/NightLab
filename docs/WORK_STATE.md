# NightLab 接续入口

更新：2026-10-26。完整目标以 PROJECT_SPEC.md 和用户后续指示为准；软件核心已通过验收，当前在最后提交/推送核验，不把算法里程碑当成最终交付。

## 当前授权与资源

- 用户睡前明确要求自主推进、不再请示。公开仓库 https://github.com/fzyz-whx/NightLab；已有推送授权。新的大数据获取仍不得扩大历史清单，当前不需要新下载。
- 本次GPU已归还且满功率，允许单进程轻量训练；31人训练已结束。CPU数值线程最多2、准备worker1。没有跨项目队列，不改其他项目/系统电源/全局Python或代理。
- 子代理按会话指定Sol low/medium，不用Luna。主线程负责最终验收与Git。当前文件先看git status和实际进程。

## 已完成的新阶段（本阶段Git提交前检查已完成）

- API/SQLite/private媒体存储、React三页已实现。WAV原始body导入/知情同意、校验、Range播放、波形/局部谱图、真实RF/CNN CPU任务/状态/取消、用户标记/版本CSV、可穿戴CSV/JSON与缺口、确认关联删除、真实实验浏览。
- 新API明确关闭FastAPI遥测，默认127.0.0.1；只有受控本机请求头允许变更。默认private/app忽略，全部原始/派生/权重/runs不提交。
- 新环境setup_app.ps1实际在private/app-env-check创建CPU环境，pip check、隔离npm ci/typecheck/lint/build及无Torch/无模型的Demo/导入/媒体/数据库/删除通过。首次离线缓存不算空缓存新机器。
- Python完整套件504 passed、4 Windows真实symlink权限skip、117 subtests passed，36.83秒；Starlette/httpx弃用警告保留。前端typecheck/lint/build通过。
- 主线程实际隐藏浏览器已跑WAV文件选择上传（32kHz双声道PCM24，12.3秒）、坏附件拒绝、有效JSON缺口/Unknown/三轴、用户标记、RF和增强CNN完整推理、CSV真实下载、静音播放推进及选段结束暂停。截图/删除/响应式/跨午夜/真实任务取消已验收，见PHASE3_ACCEPTANCE.md。没有真人个人记录或真实设备验证。

## SleepAccel31真实状态

- 已批准ZIP精确576850847字节，下载owner87301已exit0。不要重启旧下载。文件仍名sleep-accel-1.0.0.zip.part，但内容完整；本地SHA f0c1345070823a21e2bcc23a0a2f0cd82b85add5fc7797713dd7ab64a224ff8a。
- intake实际exit0：93成员size/CRC/官方SHA全通过，84新增/9复用；许可正文匹配官方SHA。receipt本地忽略，SHA 0be9ff8c5f313d3cd2e53f9c3548cbe5f18633e7f9c9b8ebdc794cb271569489。仅取HR/motion/labels；不声称steps/归档extras全验证。
- strict v2原审计31/31完整、只接受16；11motion乱序、2HR相同重复、2HR域外冲突。作者固定代码支持完整行去重再排序；独立v3保持raw，按完整label域裁剪、域内冲突拒绝，不插值/改时钟。见WEARABLE_CLEANING_V3.md。旧源码和三人结果不改。
- v3经典/深度31/31complete，26773监督epoch，16126/4422/6225，冻结18/6/7，旧3全train。经典SHA9f8b59db871190ac1142c50b811d227d80d9568efc45866dbe1bf99b0550c920；deep SHAaf51a76693c38b0e1cd73347fd0c47e0764cfc63d640804cca0aa59be3c03ef7。817884 NaN保留，旧三人classical/deep exact parity。
- 六模型与各自独立seed42复跑完成，保存加载/概率/指标/checkpoint与CNN非耗时history exact。新runner scripts/train/wearable31.py；默认严格校验不训练。结果docs/WEARABLE31_RESULTS.md与docs/results/wearable31-*.json。
- test Macro-F1：RF HR .23836/accel .27118/fusion .31161；CNN .20877/.21391/.24739；多数类.12849。三CNN低于对应RF，无测试调参。测试七人仍小，没跨seed/外部设备/医学验证。当前无训练残留。

## 冻结旧研究证据

- 声音主ScienceDB19070 V5两样本时间冲突，冻结；备用PSG10人各第二小时20文件7.11GB官方MD5匹配，6/2/2。RF窗口Macro .595983/AP .283317，CNN .543759/.247170；事件RF F1 .115607/40.5FP/h，CNN .063694/35FP/h。
- 噪声4SNR、white/hum、四分支与增强/等量控制完整实跑复跑。增强没有全面改善：未缩放窗口Macro .5664而事件F1 .0970；白噪声绝大多数漏检，滤波后增强仍全漏。见PHASE2_ACCEPTANCE/EXPERIMENTS与聚合JSON。
- 不修改旧audio/features、wearable/features、deep/inputs、deep_cache、训练基础和噪声协议/源码：缓存和模型绑定源码SHA。新31独立模块，旧3每集合1人的结果只作历史pilot，不与新split直接比提升。

## 下一步

1. 上述UI流程和真实文件/SQLite核验已完成，当前不需要重复运行。
2. README/架构/隐私/实验/许可引用/中文答辩与验收/真实截图、新CPU环境实际启动均已完成。新绑定工具使实际当地intake可合法更新唯一收据摘要，既有训练目录不能换绑；见PORTABLE_REPRODUCTION.md。
3. 所有必要检查通过后，审查Git公共文件、大小/凭据/数据忽略，整体阶段提交并推送，GitHub API比对完整hash。再追记实际hash，不凭旧日志声称干净。
4. 本goal尚未标记complete。Android和时序扩展属于可选；没有真实设备输入时不编造个人实验。

当前验收服务：原主线程exec69865/PID11308与新CPU启动exec4827/PID16960均在完成核验后停止；暂无需要保留的验收服务/训练/下载进程。默认8766端口应空闲，接续以实查为准。实际截图与删除、跨午夜、响应式、真实任务取消均完成，详见PHASE3_ACCEPTANCE.md。

最终复查：主环境504通过/4权限skip/117子测试；无数据无Torch的隔离公共检出417通过/13可选依赖、权重或权限skip/117子测试。没有真人文件，不将这一层标记已验证。完整规格交付核对见DELIVERY_CHECKLIST.md，整体goal仍未标完成。
