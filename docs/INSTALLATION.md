# Windows 本地安装、启动与复现边界

适用平台：Windows AMD64、CPython 3.12。当前 wheel 锁不承诺其他操作系统、ARM64 或其他 Python 版本。前端需要已安装、支持 Vite 8 的 Node.js/npm；本次实际核验 Node.js 24.22.1、npm 11.19.1。脚本不改全局 Python、代理、DNS 或电源设置。

## 安装

在仓库根目录的 PowerShell 中执行：

```powershell
py -3.12 --version
node --version
npm.cmd --version
.\scripts\setup_app.ps1
```

默认建立 `.venv`，也可以指定隔离环境：

```powershell
.\scripts\setup_app.ps1 -VenvPath private/my-app-env
.\scripts\start_app.ps1 -VenvPath private/my-app-env
```

相对路径以仓库根目录为基准，环境父目录由安装脚本创建。已有环境必须匹配 CPython 3.12 AMD64；建议为应用使用专用环境。安装从 `requirements-api.lock` 递归读取 `requirements-cpu.lock`，使用精确版本、wheel-only 和 SHA256 验证。先尝试清华 PyPI 镜像，失败回退官方 PyPI；随后以 `--no-build-isolation --no-deps` 安装本地 editable 包并执行 `pip check`。CPU 锁还包含已有科研绘图、构建和测试依赖，因此不是最小运行时依赖清单。

前端使用 `package-lock.json` 与 `npm ci --ignore-scripts`，随后实际执行类型检查、lint 和生产构建。npm 使用官方 registry，不改变用户全局配置。锁文件完整性由 npm 验证；安装过程需要联网或已有包缓存。setup 会更新指定环境及前端 node_modules/dist，代码回滚使用 Git；默认 `.venv`、node_modules、dist 均为可重建产物。`-FrontendPath` 可指定包含同一套源文件、配置和锁文件的隔离前端目录，用于安装核验；默认仍是 `frontend`。

安装不会请求 Torch、数据集、模型权重或 GPU 运行库，也不会启动训练；已有环境里原先安装的可选包不会被卸载。数据下载必须另走显式数据准备流程。模型文件及训练私有产物不随仓库发布。

## 日常启动与停止

```powershell
.\scripts\start_app.ps1
```

打开 `http://127.0.0.1:8766`。生产前端由 FastAPI 同源提供，不必再启动前端开发服务。启动脚本检查 Python 可执行文件和已构建 `frontend/dist/index.html`，拒绝占用端口，然后以前台 Uvicorn、一个 worker、无 reload 运行。数值库线程环境变量在子进程运行期间限制为 2，结束后恢复当前 PowerShell 的原值；应用自身单任务工作队列保持 CPU 上限。按 Ctrl+C 停止该前台服务。

启动不执行 pip/npm、不重新构建、不下载数据或权重。端口冲突时脚本报错退出，不终止占用者；绑定检查与真正启动间如有竞争，Uvicorn 自身仍会拒绝绑定。不要为了绕过冲突启动第二份后台服务。

默认个人记录目录为 `private/app`，可在当前 PowerShell 用 `NIGHTLAB_DATA_DIR` 设置独立目录。目录包含导入录音及 SQLite 元数据，预览按需生成而不永久缓存，应视为私人数据；演示记录明确标记 synthetic/demo。界面删除需要确认，备份、已有导出和系统级恢复的边界见 PRIVACY.md。

## 没有本地模型时能做什么

干净克隆可安装 CPU/API、构建界面、生成合成演示、导入有使用权的 WAV、查看媒体、导入通用可穿戴表、手动标注、导出与删除。公开聚合研究结果可独立查看。模型注册表只有在本地文件、版本及来源校验通过后才提供真实推理；缺失时显示不可用并拒绝任务，不回退为虚构预测。CNN 另需匹配的可选 Torch 环境和自有权重，该安装脚本没有安装它。

复现实验需要先按 DATASETS.md、各协议及 manifest 获取获准数据，再执行对应准备/训练流程。应用安装通过不等于下载数据通过、训练复现通过或个人设备/医学验证通过。可穿戴三十一人和声音试点的实际研究结果分别见 WEARABLE31_RESULTS.md 与 EXPERIMENTS.md；不在启动时重训。

## 2026-10-26 实际干净环境核验

使用全新忽略目录 `private/app-env-check/venv`，由脚本创建 CPython 3.12.10 环境；前端源码、配置和锁复制到 `private/app-env-check/frontend`，未用这次核验覆盖工作区前端产物。真实执行：

```powershell
.\scripts\setup_app.ps1 -VenvPath private/app-env-check/venv -FrontendPath private/app-env-check/frontend
```

hash 锁安装、editable 安装、`pip check` 均通过。pip 使用了镜像及本机 wheel 缓存，因此这是新虚拟环境安装验收，不是全新机器/空缓存离线验收。npm ci 实际安装 144 个包，typecheck、lint、build 全部退出 0。私有安装日志保存在 `private/app-env-check/setup.log`。

在这个新环境确认 `find_spec('torch') is None`，使用 TestClient 和空的模型根目录实际完成：health、所有模型 unavailable、合成 demo、独立生成的 12 秒 WAV 导入、Range 206、波形、PNG 声谱图、缺失模型分析拒绝、SQLite 两条记录计数、确认删除后记录计数归零。没有用 mock 模型冒充推理；这个验收未执行真实模型推理，也不是浏览器播放验收。TestClient 提示其 httpx 路径有弃用警告，当前请求全部通过，未擅自扩大锁文件依赖。

PowerShell Parser 检查两个新脚本无语法错误。启动脚本实际面对已有 8766 listener 拒绝启动，前后监听 PID 相同；没有终止其他进程。前台服务、真实浏览器交互及有模型环境的验收由软件主流程另行记录，不能从此 TestClient 收据推断。



## 研究输入的重建顺序

应用本身无需研究数据。完整复现31人扩展时，先按归档计划执行明确授权的获取和成员验收，再准备旧三人缓存作为兼容性锚点，最后准备独立v3。已有同名输出若来源冲突会拒绝覆盖；新实验需选新run名或独立克隆，不能删除旧结果以冒充历史。

```powershell
.venv\Scripts\python.exe -m scripts.datasets.sleepaccel_archive --execute
.venv\Scripts\python.exe -m scripts.datasets.bind_sleepaccel31_receipt
.venv\Scripts\python.exe -m scripts.datasets.bind_sleepaccel31_receipt --execute
.venv\Scripts\python.exe -m scripts.datasets.prepare_cohort --dataset sleepaccel-pilot-3 --execute --workers 1 --feature-tag features-v2 --no-reuse-pilot
.venv\Scripts\python.exe -m scripts.datasets.deep_cache --task wearable --execute
.venv\Scripts\python.exe -m scripts.datasets.prepare_sleepaccel31 --execute
.venv\Scripts\python.exe -m scripts.train.baselines --corpus data/processed/sleepaccel-cohort-31/corpus.json --run-name my-wearable31-cpu
```

深度缓存需要先按setup_deep.ps1和DL_PROTOCOL.md准备可选Torch；新31人固定CUDA批次`python -m scripts.train.wearable31 --execute`仅在明确GPU窗口执行，不带参数只做输入核验。上述重建依赖冻结intake receipt及旧三人parity，不能将新环境应用安装验收等同于从零下载和重训全研究链。当前已有输入/权重逐次核验与重训记录，但没有在另一台机器重建全链。

主线程随后实际通过start_app.ps1启动全新CPU venv（无Torch），读取同源生产前端，真实自有RF完成90秒Demo的17窗口分析，CNN明确因缺可选依赖不可用；确认删除后记录归零。收据见docs/results/app-fresh-cpu-start-v1.json。验收服务随后停止。

另一台机器的真实归档路径/耗时会改变intake receipt摘要。不要复制本机私有收据：仅在没有扩展缓存或models/runs的新克隆中，先实际验收当地归档，再用上述默认只读绑定检查及显式执行，唯一更新本地manifest的receipt摘要。官方93文件/划分/策略不变；本机既有实验拒绝换绑，详细边界见[PORTABLE_REPRODUCTION.md](PORTABLE_REPRODUCTION.md)。此流程由93个微型合成文件的真实intake→bind→冻结auditor测试覆盖，未冒充第二台机器重训。
