# NightLab 接续入口

更新：2026-10-26。先检查工作树和实际进程；完整需求在PROJECT_SPEC.md，用户修订在DECISIONS.md。历史过程见PROJECT_LOG.md，不重放旧下载/训练。

## 当前目标与完成范围

**用户最新明确：只依靠公开数据即可构成完整NightLab；本人手机录音与Android是可选DLC。** 手机未连接、无本人录音、无个人手表与舍友数据都不能再记为主项目阻塞。目标、操作与验收定义见CORE_DELIVERY.md；旧记录中“必须等真人文件才能完成”的解释已撤销。

主项目包含公开数据审计/许可、两条独立研究路线、经典/深度/抗噪与消融、本地工作台、公开样例与真实实验浏览、复现入口、文档/检查/Git交付。弱模型与负面结果保留，完成项目不等于获得可靠个人事件检测器。

## 本轮公开主线完善

- 首页新增两位验证集受试者固定前120秒PSG-Audio入口；每次载入实际核验官方raw摘要和既有准备收据。仅读Mic，有界写入工作台WAV副本，无自动下载。
- PSG RML参考来源reference，与model/user/demo分开；不可作为用户标记删除，重分析保留。无已核验时区则用相对时间，不假配SleepAccel。
- `python -m scripts.train.app_audio_rf --run-name <new>`默认dry-run，加`--execute`才训练固定已准备十人CPU RF，生成private/app-models收据，新模型可在桌面选择；不接收任意pickle/已有模型导入，拒绝覆盖。
- `python -m scripts.register_experiment --run-name <new>`默认只检查，加`--execute`将本地经典run的白名单聚合快照登记到private/local-experiments。坏登记不阻断八个内置实验，不显示身份/原始路径/逐窗预测。
- 已实际训练并登记public-core-audio-rf-v1，权重SHA a20408410200c503928ba3318d2cdada4a4257926e48e39ba4cce90c15865199。实际浏览器载入两样例、新RF完成23窗、查看本地新报告。第二样例2参考/2预测，第一样例0目标参考。CSV与确认删除链通过，四表归零，原始文件SHA不变。
- 公开收据与边界见PUBLIC_CORE_ACCEPTANCE.md、results/public-core-acceptance-v1.json。原始波形界面证据只留private，不上传。

## 冻结研究证据

- 声音原主候选ScienceDB19070 V5两样本时间冲突，冻结；PSG-Audio10人各第二小时20文件7109932234字节匹配官方MD5，6/2/2。RF窗口Macro .595983/AP .283317；CNN .543759/.247170。事件RF F1 .115607/40.5FP/h；CNN .063694/35FP/h，仅两测试小时。
- white/hum四SNR、四分支、增强/等量控制完整实跑复跑，未全面改善。增强未缩放事件F1 .0970低于控制.1304；白噪声/滤波严重漏检。见EXPERIMENTS与PHASE2_ACCEPTANCE。
- SleepAccel31官方1.0.0 ZIP576850847字节已完整下载，仍名.zip.part。本地归档SHA f0c1345070823a21e2bcc23a0a2f0cd82b85add5fc7797713dd7ab64a224ff8a不是官方归档摘要。93目标文件size/CRC/官方SHA通过，84新增9复用；原始收据本地保留。
- v3完整行去重/排序/标签域裁剪，不插值或猜时钟，31人26773监督epoch，18/6/7（旧3全train），旧三人输入精确一致。六模型及各seed42独立复跑，权重/概率/指标和CNN非耗时history一致。RF HR/accel/fusion Macro .23836/.27118/.31161，CNN .20877/.21391/.24739，三个CNN均低于RF。见WEARABLE31_RESULTS。
- 不改旧audio/features、wearable/features、deep/inputs、deep_cache、基础训练和噪声源码，避免破坏冻结来源SHA。新机器实际intake后按PORTABLE_REPRODUCTION在新克隆绑定唯一收据摘要；禁止复制旧私有收据或给已有模型换绑。

## 软件与资源

React/FastAPI/SQLite已有WAV校验、Range/波形/谱图、CPU单worker任务/取消、标记/版本CSV、可选CSV/JSON时序与缺口、确认关联删除、真实实验。默认private/app，127.0.0.1:8766，无启动数据/权重下载、无遥测。Windows setup_app/start_app实际新CPU环境通过；不是空缓存新电脑验收。

本轮没有新下载/GPU，只新增一次小规模CPU RF重训。数值线程<=2/准备worker1；不改系统电源、显卡功耗、其他项目进程、全局Python/代理/DNS。新的>500MB或全量数据仍须清单明确批准。子代理按会话指定Sol low/medium，不用Luna。

## 可选DLC状态

个人仅本人声音，不等手表，不采舍友。Android Kotlin源码与签名debug APK已实现；工具复用USERPROFILE/.zcode/workspace/default/_toolchain，C:/android-sdk为空，不需重下SDK。构建/JVM/合成互通通过，**真实手机麦克风/锁屏/整夜未验收**，不阻塞主项目。

本地APK private/android-delivery/NightLabRecorder-0.1.0-debug.apk，927378字节，SHA 9b7df3cc30e46321c6c09ce2f2b8a513f4b4840c0712dd656c43073f0e0f57c6。private/android-signing/debug.keystore用于后续更新，必须保留，不当普通缓存清理、不上传。android/README.md记录权限/试录与限制。

## 结束检查与接续

本阶段最终测试/提交/远端hash以PROJECT_LOG末条和public-core-acceptance-v1.json为准。接续先实查Git和8766/5178监听；不要重跑已结束的服务或研究下载。主项目按公开主线验收后可完成goal；后续按用户新的具体改进或可选扩展推进，不能用缺个人材料无限延长。
