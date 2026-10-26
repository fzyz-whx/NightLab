# NightLab Android 本人声音采集端

第一版0.1.0为本机编译的debug APK，包名org.nightlab.recorder，最低Android8.0/API26，compile/target API34。当前已通过编译、JVM文件/状态测试、Android lint和APK签名检查；没有手机或模拟器实录、界面截图、锁屏整夜通过声明。

## 使用

1. 将`private/android-delivery/NightLabRecorder-0.1.0-debug.apk`复制到手机安装。APK不随Git提交；重新构建可得到本机安装包。
2. 独处且有权采集时勾选本人录音确认，允许麦克风与通知，点击开始。看到正在录音状态和通知后再锁屏。
3. 先做30–60秒短录，停止并保存，使用“试听首片”检查。可随时“停止试听”。
4. 选择“导出”，在系统文件选择器选本机目标。ZIP含WAV分片与session.json；原始录音仍保留在App私有目录。导出可取消，取消/失败的目标ZIP应自行删除，原录音不受影响。
5. 将ZIP传回电脑并解压，WAV可直接在NightLab桌面“夜间记录”导入。无手表数据也能正常使用，默认相对秒时间轴即可。完整日期/时区在session.json的各chunk.start_time中；要填桌面绝对时间时应按电脑时区正确换算，不能直接丢掉时区。

短录和桌面导入通过后，再试锁屏20–30分钟，最后才录一夜。前台服务不保证所有厂商的省电策略都放行；任何中断均需据实际文件/状态确认。当前没有声压标定、睡眠分期、医学判断或健康评分。

## 行为与隐私

只由用户点击开始，不开机自启、不后台自动重启录音。前台mic服务有可见通知和停止按钮；没有INTERNET权限。初始格式优先16kHz/PCM16/mono，不支持时试48/44.1kHz并记录实际输出格式。最长12小时、约每小时分片，16kHz正文约115.2MB/小时，8小时约921.6MB；回退48kHz约三倍。

内部流式WAV/JSON只在App私有files/recordings保存。JSON含手机厂商/型号/API、采样格式、起止时区、样本/单调时钟诊断、分片hash及停止原因，不收集账号、定位或设备唯一标识。默认禁止备份，并为Android12+配置cloud/device-transfer排除；用户主动导出的目标提供者可能有自己的同步行为，请选择合适的本机位置。

定期更新临时WAV头。进程中断后修复合法.part前缀；若文件已完成改名但JSON尚未登记，会在格式/连续编号/哈希检查后恢复登记为interrupted。未知、冲突或损坏文件保留并报错，不伪造完整录音或填补静音。分片时间来自样本数与手机墙钟锚点，不声称外部精密同步。录音启动、恢复、导出和删除使用进程内互斥预约，避免恢复误伤正在启动的录音。

确认删除只删除本机该条录音；已导出的文件不会一起删除。卸载会清除App私有数据，卸载前先导出。更新同一手机的APK应保留本项目`private/android-signing/debug.keystore`；该签名私钥不进Git，也不能把它当作可随意清掉的构建缓存。

## Windows 构建

实际复用旧Zcode工具目录`%USERPROFILE%/.zcode/workspace/default/_toolchain/`下SDK34/Build Tools34.0.0、JDK17.0.20.1、Gradle8.5；没有下载整套SDK或模拟器。也可显式传`-SdkPath/-JdkPath/-GradlePath`。只对当前构建子进程设置环境，结束恢复，不改系统PATH或其他项目。

```powershell
.\scripts\build_android.ps1
# 依赖已缓存后，严格校验并离线构建
.\scripts\build_android.ps1 -Offline
```

构建仓库依次为阿里云google/public镜像、官方Google Maven/Maven Central回退。AGP8.3.2、Kotlin1.9.22、JUnit4.13.2等固定版本；根buildscript/app依赖锁与gradle/verification-metadata.xml入库。正常构建使用strict验证；`-InitializeVerification`只用于显式审核后的初次/有依据的依赖更新，不能用于绕过校验错误。

本轮Maven缓存414文件295133117字节，这不是网络抓包流量值。394个已登记artifact逐一匹配冻结SHA256，6个核心依赖另与官方发布checksum匹配；不能称所有传递依赖都做过独立官方签名认证。证据见`docs/ANDROID_DEPENDENCIES.json`。

安装包输出在private/android-delivery，脚本验证签名并打印SHA256；包约0.93MB，开发签名不是应用商店正式发行。本轮APK hash见`docs/results/android-build-v1.json`，构建日志/原始测试报告留忽略目录。

## 复核与跨语言夹具

本次带两个显式夹具参数的offline clean/test/lint/assemble真实通过。24个JUnit用例中22通过，2个真实符号链接测试因Windows权限skip；其中2个是明确启用的Kotlin夹具生成检查。未请求夹具参数时，它们不写输出文件。lint0error/3warning：旧Android忽略API31备份属性（由旧allowBackup配置覆盖）、v26图标目录对min26冗余、缺少Android13主题单色图标；不是麦克风或权限检查被关闭。

为避免覆盖原文件，夹具输出必须使用不存在的文件路径，并事先建立其父目录。下面命令是复核入口，已有同名文件应换新名字：

```powershell
New-Item -ItemType Directory -Force private/android-fixture
.\scripts\build_android.ps1 -Offline -Tasks @('clean','testDebugUnitTest','lintDebug','assembleDebug') -WavFixture "$PWD/private/android-fixture/review.wav.part" -SessionFixture "$PWD/private/android-fixture/review.zip"
.venv\Scripts\python.exe -m scripts.verify_android_interop --archive private/android-fixture/review.zip --wav private/android-fixture/review.wav.part
```

实际Kotlin夹具由两个12秒分片组成，跨午夜时间/PCM头/frames/SHA通过Python和桌面HTTP导入核验，未添加手表附件；本机自建RF实际完成2窗口，再确认删除归零。无本机模型时复核脚本明确省略推理并报告不可用，不生成假得分。该夹具是合成PCM，不能替代真机麦克风验收。

应用内“关于与开源许可”可查看自写代码MIT与运行时Kotlin/JetBrains annotations的Apache2许可全文，许可证也随APK资产打包。开源依赖保持原许可，构建工具/测试依赖不当作App自写代码。
