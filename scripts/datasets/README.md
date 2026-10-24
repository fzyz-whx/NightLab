# 数据审计与样本获取工具

Windows / Python 3.12标准库。在仓库根目录运行：

```powershell
python -B -m scripts.datasets audit
python -B -m scripts.datasets plan --dataset sleepaccel
python -B -m scripts.datasets verify --dataset sleepaccel
python -B -m scripts.datasets sample --dataset sleepaccel --file data/raw/sleepaccel/1.0.0/labels/5383425_labeled_sleep.txt
```

默认命令audit，只读取已提交manifest与本地文件状态。plan只展示来源、版本、许可、大小与校验信息；二者不联网、不计算大文件摘要、不修改文件。`-B`避免字节码缓存。

## sample

sample默认dry-run。确认输出的精确字节数与来源后，同一命令加 `--execute` 才执行。必须明确选择一个manifest路径；许可和来源已审计、HTTPS原始地址、精确大小1至500,000,000字节、官方SHA256或MD5和来源缺一则拒绝。当前只开放实际核查过的SleepAccel三文件、主声音01 JSON/阶段CSV/手机WAV和PSG-Audio单个RML；手机WAV因对齐冲突暂不建议获取。

网络执行按块写入 `.part`，检查磁盘空间、Content-Length、摘要和大小；续传必须返回准确的206 Content-Range。服务器忽略Range返回200会拒绝追加。Ctrl+C、超时、校验失败保留partial；完整且官方摘要匹配后才发布正式文件名。已有完整文件校验通过时直接复用，已有不匹配正式文件拒绝覆盖。单次网络操作有socket30秒、总计约300秒上限，无无限重试；网络慢时再次执行同一命令继续。`.part`的正式路径与续传状态只在data/raw内，不能将其当成已验收数据。

手动curl留下的未完成正式文件不能直接通过sample，须先确认没有进程写入，再在data/raw内改名为.part（不能覆盖已有partial）。批量全量下载和prepare尚未实现；全量或大于500MB的计划必须另行确认。

## verify与结构审计

verify按1MiB计算SHA-256，有官方MD5则另算。缺失、空文件、精确大小或摘要不符、空清单均失败。`ok`仅表示清单内当前检查无错误；无官方摘要时本地SHA256也不能证明官方匹配。`official_sha256_verified`只有全部列出文件与官方SHA256相符时才为true。`inventory_complete: false`表示清单不完整，即使verify通过也不能声称整个数据集完整或可训练。

```powershell
python -B -m scripts.datasets.inspect_sleepaccel --kind labels
python -B -m scripts.datasets.inspect_sleepaccel --kind heart_rate
python -B -m scripts.datasets.inspect_sleepaccel --kind motion
python -B -m scripts.datasets.inspect_sleepaccel --kind alignment
python -B -m scripts.datasets.inspect_sciencedb --subject 01
python -B -m scripts.datasets.inspect_sciencedb --subject 02
python -B -m unittest discover -s tests -p "test_*.py" -v
```

SleepAccel检查必须先匹配官方SHA256，才统计列数、有限值、单调时间、间隔和原始标签。sleepaccel-five-stage-v1保留原始编码，3/4合并N3、-1掩码为无效目标，未文档化编码拒绝；依据见docs/LABEL_MAPPING_EVIDENCE.md。检查脚本读取完整小样本到内存，尚未为未经规模评估的全库处理优化。

ScienceDB检查先匹配JSON官方MD5，再读取从官方206 Range得到的WAV局部头；它明确不声称完整WAV已校验。实际时间冲突返回1，数据读取/格式错误返回2。当前01/02均有时间冲突，不能用裁剪标签或人为缩放时间强行修复。详见docs/SOUND_AUDIT.md。

来源清单中的reported_size是页面显示值；exact_bytes只记录官方精确值。预览和原始下载必须分清：ScienceDB downloadFileMole可变换格式，china.scidb.cn/download才是已核实的原始入口。

## PSG-Audio 检查与明确授权的大文件获取

`download --dataset <id> --file <manifest-path>`默认仍是dry-run。大文件执行需先有用户针对该文件的授权，再加 `--execute --confirm-large-download`；该参数不能代替授权。当前唯一获准的EDF是00000995-100507[002].edf（710738976字节），未批准其余分片或受试者。每次只获取一个清单文件，没有批量全库下载。

`python -B -m scripts.datasets.inspect_psg`先核验完整EDF/RML，再检查头/时钟/事件区间，并实际读取短音频记录。只从UserStaging取得人工阶段来源，MachineStaging独立报告，不把阶段变化条数当作epoch数。输出避免身份字段和绝对日期。见docs/PSG_AUDIT.md。
