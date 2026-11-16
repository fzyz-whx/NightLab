# 最终交付核对与未覆盖项

更新2026-10-26。**用户已明确：公开数据足以构成完整主项目，个人录音与Android为可选DLC。** 当前目标与操作路径见CORE_DELIVERY.md；真人或医学验收不是主线完成门槛。

| 原任务交付 | 当前可核对产物 | 状态 |
|---|---|---|
| 可运行本地软件 | frontend/src、api/services/storage、setup_app/start_app | 已实现，真实浏览器/新CPU环境验收 |
| 架构、数据、隐私、实验说明 | README、ARCHITECTURE、DATA_AUDIT/DATASETS、PRIVACY、EXPERIMENTS | 已更新 |
| 审计、许可、获取 | manifests/plans、datasets脚本、原主声音时间冲突及备用审计 | 真实执行；不含原始数据 |
| 声音传统/DL比较 | 十人RF/CNN、同seed复跑、连续事件 | 已运行，保留弱事件结果 |
| 可穿戴传统/DL消融 | 31人18/6/7、三RF/三CNN及各复跑 | 已运行，CNN低于RF |
| 抗噪与失败案例 | 固定SNR、四分支、增强/重复控制、JSON/图 | 已运行，未全面改善 |
| 通用个人导入和删除 | WAV、CSV/JSON、绝对/相对时间、确认删除 | 合成数据及实际文件/SQLite链通过 |
| 本人实际手机录音（可选DLC） | 尚未采集；仅本人，不纳入舍友 | 未验收，独立于主项目完成 |
| 实际UI截图 | artifacts/screenshots | 本机真实截图，示例标synthetic |
| 公开GitHub历史 | fzyz-whx/NightLab、PROJECT_LOG | 每阶段实际提交/推送后核hash |
| 中文讲解稿 | DEFENSE_ZH.md | 已提供五分钟稿与问答 |
| Android声音采集端（可选DLC） | 已完成首版源码与签名APK | 源码、签名debug APK、JVM/合成互通已通过；真机待验，见android/README |
| 长上下文 | 原规格可选扩展 | 未实现 |

必要软件测试与数据/实验收据分别在PHASE0/1/2_ACCEPTANCE、PHASE3_ACCEPTANCE、WEARABLE31_RESULTS和INSTALLATION。公开主线增量验收见PUBLIC_CORE_ACCEPTANCE.md。可选DLC的实际设备文件、手机麦克风/锁屏仍保持“未验收”；不要求用户提供材料才能交付主项目。

公开样本载入、新CPU RF到模型注册、本地经典run到实验页均提供显式入口；正式研究与公开聚合报告保留。个人手表不可得、Android真机和本人录音均不阻塞公开主线。
