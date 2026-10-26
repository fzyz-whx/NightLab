# 对照完整规格的交付审计

2026-10-26，依据PROJECT_SPEC.md原始18节及DECISIONS/用户后续授权，不以测试数量代替要求本身。当前软件/研究核心已交付；真实个人文件层没有材料，不计为通过。临床验证、Android和长上下文的范围见末节，不能将其擅自加入必做门槛。

| 要求或边界 | 可直接核对的证据 | 结论与范围 |
|---|---|---|
| 1：本地研究定位、两条独立路线 | README、PRIVACY、两套manifest/训练入口与App页脚 | 已实现；无健康评分或联合模态提升声明 |
| 2：审计先于训练、标签粒度决定目标 | DATA_AUDIT、SOUND_AUDIT、PSG_AUDIT、LABEL_MAPPING_EVIDENCE、Git历史 | 主声音冲突保留并转备用，未缩放标签强行对齐 |
| 3：数据版本、文件、许可与引用 | DATASETS、各manifest、PHASE0/1_ACCEPTANCE、WEARABLE31_RESULTS | 所用声音十人/可穿戴31人经官方摘要验收，BIDSleep未取全量 |
| 4：audit/plan/sample/download/verify/prepare | scripts/datasets/cli.py及sample/fetch/prepare，tests/test_datasets与相关网络夹具 | 入口实际存在，默认dry，大文件执行须明确标志及清单授权 |
| 4：断点/错误/磁盘检查、国内官方源优先 | sample/fetch_manifest/fetch_sleepaccel_archive、PROJECT_LOG、归档intake | 实际官方续传完成；不将partial/HTTP成功当数据就绪 |
| 5：声音传统与轻量DL | psg-rf-pilot-v1/audio-cnn-pilot-v1聚合JSON、CNN复跑收据 | 实际训练/保存加载/独立seed42复跑，无预训练权重 |
| 5：滤波、增强、控制与指标 | NOISE_PROTOCOL、NOISE_AUGMENTATION_PROTOCOL、完整四模型结果/图、EVENT_PROTOCOL | 事件F1/误报/定位及四SNR全保留；大量漏检/不改善未隐藏 |
| 5：真实室友声音使用限制 | PRIVACY、噪声类型记录 | 没有采集室友声音，当前white/hum明确为合成，不冒充宿舍验证 |
| 6：wearable多数类/RF/CNN/三输入消融 | WEARABLE31_RESULTS、六结果JSON、verification-v3 | 31人固定18/6/7真实执行；CNN低于对应RF |
| 6/11：受试者划分/训练拟合/验证选点 | manifests、deep/training、experiments、测试与真实训练记录 | 无随机epoch跨人泄漏；统计/类权重只train、checkpoint只val |
| 6：缺失/未知/标签映射 | wearable/features、deep/inputs、cleaning_v3与针对测试 | NaN/mask保留；旧3输入精确兼容，原始4并N3、-1不监督 |
| 7：并列不等于融合 | README/PRIVACY、Signals与两条独立训练 | 界面并列观察，未声称声音与手表联合训练 |
| 8A：WAV/附件/时间轴/无数据状态 | PHASE3_ACCEPTANCE、app-browser-acceptance-v1与实际截图 | 真实浏览器合成文件、缺口、跨午夜0/2/7秒已核 |
| 8B：播放器/波形/谱图/标记/CSV | RecordViewer/Signals、浏览器收据、Range/CSV API测试 | 静音时钟推进与选段末尾暂停；模型/用户/demo来源和版本可区分 |
| 8C：真实实验与可折叠参数 | Experiments、受控CATALOG、真实聚合报告/图 | 页面显示实际分数；不伪造实验运行或个人准确率 |
| 8/15：错误、加载和恢复 | app-retry-acceptance-v1、两张错误/恢复截图、frontend检查 | 定向阻断模型请求后重试恢复完整数据，1600波形点/两模型/17窗RF通过 |
| 9：业务分层与路径/导入验证 | api/services/storage/audio/wearable/deep、ARCHITECTURE | 非单文件全业务；用户不传服务器路径，原始数据/私有位置可配置 |
| 9/15：长任务/安全停止/CPU路径 | jobs/repository、API回归、真实359窗任务取消、fresh-cpu-start收据 | 单worker；取消不发布部分结果；无Torch环境真实RF17窗完成 |
| 10：同意/本地/删除/无遥测 | records/media/API配置、PRIVACY、实际目录和SQLite关联删除 | synthetic测试无私人数据；无启动下载；删除不是安全擦除 |
| 11：真实记录/图表/复现 | runs本地、docs/results、绘图脚本、各模型复跑JSON | 同种子计算复现已证明；跨硬件、跨seed及外部泛化未做且不作结论 |
| 12：真实Git/公开远端 | PROJECT_LOG、git log、GitHub main hash比对 | 已逐阶段实际提交/推送，数据/权重不入Git |
| 13：README图文/安装/研究问题/结果/局限 | README及所链接文档 | 真实截图标synthetic，含研究问题、架构、许可、安装、结果与贡献说明 |
| 14 Phase0–2 | PHASE0/1/2_ACCEPTANCE、后续31人结果与真实产物 | 对应阶段证据齐全；旧阶段“待做”按当时历史理解，由本索引指向后续 |
| 14 Phase3 | PHASE3_ACCEPTANCE、setup_app/start_app、frontend与API | 软件核心和新CPU环境已验收 |
| 14 Phase4：通用个人适配与删除 | WAV/CSV/JSON导入、时间单位、SQLite与删除链 | 功能以合成文件验证，未以假数据冒充真人设备导出 |
| 14 Phase4：实际本人手机/手表同晚记录 | 用户尚未提供文件 | 未验收；不擅自采集或搜索私人健康文件 |
| 14 Phase5：打磨与讲解 | INSTALLATION、实际UI截图、DEFENSE_ZH、LICENSE/CITATION | 已提供；新虚拟环境用本机包缓存，非空缓存新电脑/全研究重建 |
| 15：数据/模型/API异常与一致性 | 最近完整504通过/4权限skip/117子测试，隔离417通过/13skip/117子测试 | 覆盖范围见PHASE3；skip明确，不以绿色测试推断真人验证 |
| 16：锁定/CPU/CUDA/模板/启动 | requirements锁、pyproject、package-lock、setup脚本、.env.example | 示例无秘密且不自动加载；当前变更前端typecheck/lint/build再通过 |
| 16：新机器收据可迁移 | PORTABLE_REPRODUCTION、binder及12个合成测试 | 真实当地intake后只绑定摘要，不复制私人收据；旧产物拒绝换绑 |
| 17：最终九项交付 | DELIVERY_CHECKLIST逐项映射 | 软件/研究/文档/Git齐备；个人实际设备层仍单列未验收 |
| 18：阶段如实汇报与资源 | PROJECT_LOG、WORK_STATE、AGENTS、当前进程检查 | 本轮无新下载或训练；只临时单CPU验收服务，结束已停 |

临床验证不是本项目声称或要求完成的目标，不能以未做临床试验无限延长goal。Android录音端、CNN-GRU/TCN、BIDSleep扩展为原规格明确的可选项，本轮未做，不将其伪造完成或自动启动大数据获取。

当前唯一需要外部材料才能进行的验收是本人有权使用的实际录音与可用设备导出文件及其时区/时间起点。用户睡前要求不再发问题，故只保留未验收状态，不索取文件、不采集、不编造。整体goal尚未标complete。

## 用户新范围与Android增量

2026-10-26用户明确个人手表不可得，个人先仅本人声音、舍友不纳入；个人手表不再是门槛。Android由原可选项提升为当前交付，现已实现并真实构建、签名/权限/JVM/合成桌面衔接通过，详见android/README.md。原表未覆盖的手机麦克风/锁屏/整夜仍须真人设备验收，不能据APK或JVM测试宣称完成。
