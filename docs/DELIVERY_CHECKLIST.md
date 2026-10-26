# 最终交付核对与未覆盖项

更新2026-10-26。证据按层面列出，完成软件功能不等于真人或医学验收。

| 原任务交付 | 当前可核对产物 | 状态 |
|---|---|---|
| 可运行本地软件 | frontend/src、api/services/storage、setup_app/start_app | 已实现，真实浏览器/新CPU环境验收 |
| 架构、数据、隐私、实验说明 | README、ARCHITECTURE、DATA_AUDIT/DATASETS、PRIVACY、EXPERIMENTS | 已更新 |
| 审计、许可、获取 | manifests/plans、datasets脚本、原主声音时间冲突及备用审计 | 真实执行；不含原始数据 |
| 声音传统/DL比较 | 十人RF/CNN、同seed复跑、连续事件 | 已运行，保留弱事件结果 |
| 可穿戴传统/DL消融 | 31人18/6/7、三RF/三CNN及各复跑 | 已运行，CNN低于RF |
| 抗噪与失败案例 | 固定SNR、四分支、增强/重复控制、JSON/图 | 已运行，未全面改善 |
| 通用个人导入和删除 | WAV、CSV/JSON、绝对/相对时间、确认删除 | 合成数据及实际文件/SQLite链通过 |
| 本人实际手机录音 | 尚未采集；用户2026-10-26确认仅本人声音 | 安卓采集链待实现/真机验收，不等个人手表数据 |
| 实际UI截图 | artifacts/screenshots | 本机真实截图，示例标synthetic |
| 公开GitHub历史 | fzyz-whx/NightLab、PROJECT_LOG | 每阶段实际提交/推送后核hash |
| 中文讲解稿 | DEFENSE_ZH.md | 已提供五分钟稿与问答 |
| Android声音采集端 | 用户2026-10-26提出作为下一交付 | 源码、签名debug APK、JVM/合成互通已通过；真机待验，见android/README |
| 长上下文 | 原规格可选扩展 | 未实现 |

必要软件测试与数据/实验收据分别在PHASE0/1/2_ACCEPTANCE、PHASE3_ACCEPTANCE、WEARABLE31_RESULTS和INSTALLATION。剩余现场工作是使用本人明确授权的实际设备文件进行操作与时间同步复核；没有材料时保持“未验收”，不擅自采集或询问睡眠中的用户。

用户最新边界：个人手表数据目前无法取得，不再作为门槛；不纳入舍友。公开31人可穿戴研究仍保留。
