# NightLab 本地工作台

React / TypeScript / Vite。三个页面：夜间记录、事件检查、实验比较。所有数据从 `/api` 读取；没有内置假指标、云服务或自动音频播放。

在本目录使用 Node 24.22.1 / npm 11.19.1：

```powershell
$env:GOMAXPROCS='2'
$env:RAYON_NUM_THREADS='2'
$env:NPM_CONFIG_MAXSOCKETS='2'
npm ci
npm run typecheck
npm run lint
npm run build
npm run dev
```

开发服务只监听 `127.0.0.1:5178`，端口冲突时退出。`/api` 默认代理 `http://127.0.0.1:8766`，可在启动前设置 `NIGHTLAB_API_URL`。生产资源使用相对路径，由后端同源提供 `dist/`。

上传 WAV 使用原始二进制 body 和真实 XHR 字节进度；同意录音使用权后才能导入。绝对时间由本机时间选择器转为带时区的 ISO 时间。附件支持明确相对秒、Unix 秒/毫秒和带时区 ISO；模板来自后端。心率及加速度空值与较长采样间隔断线，阶段 Unknown 为灰色，仅汇总不生成曲线。模型任务状态与完成窗口直接来自服务。删除需确认，有活动任务时等待服务确认任务停止。

合成 Demo 始终显示标记；用户标记、模型预测和 Demo 区别展示。实验页递归读取真实 JSON 的指标与元信息，包含 model/split 或 noise condition 结构；未知指标可在完整报告中查看。图表只接受本地实验受控 URL。

真实端到端与静音媒体联动验证由主项目统一执行；静态检查本身不代表真实个人设备或医学验证。
