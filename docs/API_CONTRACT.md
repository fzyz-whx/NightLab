# 本地API/前端实现契约 v1

2026-10-26，以下接口在本阶段实现，未实现前不作功能完成声明。API前缀`/api`，127.0.0.1:8766；Vite5178代理`/api`，生产构建由后端本机静态提供。变更请求必须`X-Nightlab-Client: 1`，无远程上传/遥测/自动下载。

## 数据形状

Record：`id`(UUID)、`name`、`created_at`(UTC ISO)、`start_time`(带时区ISO或null)、`is_demo`、`consent`、`audio`。audio含`duration_seconds,sample_rate,channels,frames,format,subtype,bytes,sha256`。

Event：`id,record_id,start_seconds,end_seconds,label,source,score,model_id,model_version`；source为`model/user/demo`，用户/合成标记score和model_id为null，模型输出不是参考真值。

Job：`id,record_id,model_id,model_version,status,progress,processed_windows,total_windows,message,error,created_at,finished_at`；status=`queued/running/completed/cancelled/failed`，状态来自实际worker，不以提交成功当完成。

Wearable：`source:'user_import',time_unit:'relative_seconds',heart_rate:[{time_seconds,value}],stages:[{time_seconds,duration_seconds,stage}],acceleration:[{time_seconds,x_g,y_g,z_g}],summary:{...},warnings:[...]`；value/坐标可null，保留缺口，summary-only不生成曲线。导入标签不是PSG真值。

## 记录与媒体

- `GET /health`：`{status:'ok',app:'NightLab',version:'0.1.0.dev0',privacy:'local'}`。
- `GET /records`：`{records:Record[]}`。
- `POST /records/import`：原始WAV二进制body（不是multipart），`X-Filename`为encodeURIComponent后的文件名，`X-Consent:true`，可选`X-Start-Time`带时区ISO；返回Record。XHR可显示实际上传字节/取消；后台校验真实容器/大小/帧/样本，先暂存后原子发布，失败无孤立DB/个人记录。上限4GiB、最长24h、8–96kHz、mono/stereo，接受WAV/WAVEX/RF64内PCM16/24/32或float（非有限/超幅拒绝），分块处理。
- `POST /demo`：生成确定性合成90秒WAV、示例HR/阶段与3个`source=demo`片段标记，返回Record；is_demo始终true，不称真实参考，不自动播放。
- `GET /records/{id}`：`{record:Record,events:Event[],wearable:Wearable|null,latest_job:Job|null}`。
- `GET /records/{id}/audio`：受控本地FileResponse，支持HTTP Range，客户端audio元素正常seek。
- `GET /records/{id}/waveform?points=1600&start=0&end=...`：`{duration_seconds,start_seconds,end_seconds,points:[{time_seconds,min,max}]}`，分块包络，有界点数。
- `GET /records/{id}/spectrogram?start=0&duration=10`：PNG，最多60秒局部；强度为相对频谱，不声称SPL标定。
- `DELETE /records/{id}`：JSON`{confirm:true}`；无确认400，活动job时409（先取消），成功`{deleted:true,id}`；实际删原始/附件/衍生目录及DB/event/job关联。

## 时序与标记

- `POST /records/{id}/wearable?kind=combined&time_unit=relative_seconds`：raw CSV（text/csv）或JSON；最多20MiB，返回canonical Wearable。JSON可包含heart_rate/stages/acceleration/summary。time_unit枚举relative_seconds/epoch_seconds/epoch_milliseconds/iso8601；绝对时间必须已有record.start_time，禁止猜秒/毫秒或排序修复。
- CSV列：`time_seconds`或`timestamp`，可选`heart_rate_bpm,stage,stage_duration_seconds,accel_x_g,accel_y_g,accel_z_g`。存在stage时duration必须明确，规范W/N1/N2/N3/REM/Unknown，数字厂商编码不猜测。HR/轴空值保留null；有限正HR，轴为明确g。JSON条目可用`time`或`time_seconds`、HR的`value`/`bpm`、stage的`duration_seconds`，文档/模板给确切示例。summary-only不要求time。
- 每一signal时序严格递增；标记区间不重叠；允许信号在音频范围前后但warnings明确计数，不裁掉原始观察或虚构同步。
- `GET /templates/wearable.csv`、`/templates/wearable.json`：明确synthetic、相对秒/g模板。
- `POST /records/{id}/annotations`：JSON`{start_seconds,end_seconds,label}`，区间在音频内，source强制user；返回Event。
- `DELETE /records/{id}/annotations/{event_id}`：仅user/demo可删，model标记通过下次分析替换；返回`{deleted:true}`。
- `GET /records/{id}/events.csv`：包含区间、label、source、score、model_id、model_version，用户/示例不造得分。

## 模型、任务与实验

- `GET /models`：`{models:[{id,name,kind,available,reason,version,scope}]}`；至少audio_rf_pilot_v1，其他可信自有CNN可扩展；缺模型仍可浏览音频。默认CPU，不加载未知joblib。
- `POST /records/{id}/analyze`：JSON`{model_id:'audio_rf_pilot_v1'}`，返回Job；同记录活动任务409。单worker/数值两线程。
- `GET /jobs/{id}`：Job；`POST /jobs/{id}/cancel`：请求取消，最终status由worker实际停止后确认。
- 个人音频10秒/5秒hop，末尾若未覆盖，追加sample-grid上的最后10秒窗，确保每个5秒bin均有真实预测；不足10秒明确失败，不补造静音/负例。全部概率再按冻结事件协议解码。显示模型得分/版本，无个人准确率和健康分数。
- `GET /experiments`：`{experiments:[{id,title,description,category,dataset,figures:[{name,url}]}]}`；只注册真实公开JSON/科研图。
- `GET /experiments/{id}`：`{id,title,report,figures:[{name,url}]}`；`GET /experiments/{id}/figures/{name}`受allowlist控制。前端普通表/曲线与折叠参数读真实数据，不造KPI。

## 误差与测试约束

错误使用JSON`{detail:'中文可操作说明'}`，validation422数组前端也可读。每个UUID/路径均在独立private root下，客户端不传服务器路径。删除前解析绝对根并检查，不触研究dataset/其他项目。

SQLite每次操作独立连接/事务；导入保存音频并验证后才提交record，DB失败回滚目录。任务开始/进度/结束与事件发布原子；取消不发布部分model结果。静音headless浏览器验证真实接口、文件/DB、模型与播放器联动，再截图；不外放声音。
