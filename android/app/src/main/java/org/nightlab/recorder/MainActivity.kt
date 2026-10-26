package org.nightlab.recorder

import android.Manifest
import android.app.*
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.MediaPlayer
import android.net.Uri
import android.os.*
import android.provider.Settings
import android.view.*
import android.widget.*
import org.json.JSONObject
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Local microphone recorder. No account, network, health score or automatic recording. */
class MainActivity : Activity() {
    private lateinit var store: SessionStore
    private lateinit var list: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var counter: TextView
    private lateinit var feedback: TextView
    private lateinit var consent: CheckBox
    private lateinit var start: Button
    private lateinit var stop: Button
    private lateinit var stopPreview: Button
    private lateinit var recover: Button
    private lateinit var exportCancel: Button
    private lateinit var level: ProgressBar
    private val handler = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private val loadingList = AtomicBoolean(false)
    private val cancelExport = AtomicBoolean(false)
    private var exporting = false
    private var visible = false
    private var pendingExportId: String? = null
    private var sessions: List<JSONObject> = emptyList()
    private var lastSignature = ""
    private var lastButtonsBusy: Boolean? = null
    private var player: MediaPlayer? = null
    private val ink = Color.rgb(32, 59, 67)
    private val teal = Color.rgb(40, 125, 116)
    private val muted = Color.rgb(98, 115, 118)
    private val ticker = object : Runnable {
        override fun run() { refreshStatus(); loadSessions(); handler.postDelayed(this, 1000) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = RecorderStorage.get(this)
        pendingExportId = savedInstanceState?.getString("pending_export")
        val scroll = ScrollView(this).apply { isFillViewport = true; setBackgroundColor(Color.rgb(243, 245, 241)) }
        val page = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(22), dp(24), dp(22), dp(24)) }
        scroll.addView(page)
        page.addView(text("NIGHTLAB / 本人声音", 12, teal, true))
        page.addView(text("留下一段夜间声音", 27, ink, true).apply { setPadding(0, dp(12), 0, dp(8)) })
        page.addView(text("本机采集与导出，回到桌面工作台逐段复核。", 14, muted))
        val card = card()
        statusText = text("尚未开始录音", 19, ink, true)
        counter = text("00:00:00 · 0.0 MiB", 25, ink, true).apply { setPadding(0, dp(15), 0, dp(12)) }
        card.addView(statusText); card.addView(counter)
        card.addView(text("输入电平 · 相对幅度，未经声压标定", 12, muted))
        level = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        card.addView(level, LinearLayout.LayoutParams(-1, dp(18)))
        consent = CheckBox(this).apply { text = "确认当前采集仅涉及本人，且有权录音。"; textSize = 14f; setTextColor(ink) }
        card.addView(consent)
        val actions = row()
        start = button("开始录音") { requestStart() }
        stop = button("停止并保存") { stopPlayback(); startService(Intent(this, RecordingService::class.java).setAction(RecordingService.STOP)) }
        addEqual(actions, start); addEqual(actions, stop); card.addView(actions)
        card.addView(text("录音期间保留可见通知；最长12小时，约每小时生成一个WAV分片。手机锁屏可靠性需先做短录测试。", 12, muted).apply { setPadding(0, dp(12), 0, 0) })
        page.addView(card)
        feedback = text("", 13, teal).apply { setPadding(0, dp(8), 0, dp(8)) }
        page.addView(feedback)
        stopPreview = button("停止试听") { stopPlayback(); feedback.text = "试听已停止。" }.apply { visibility = View.GONE }
        page.addView(stopPreview)
        exportCancel = button("取消当前导出") { cancelExport.set(true); feedback.text = "正在停止导出，原始录音保留在本机。" }
        exportCancel.visibility = View.GONE
        page.addView(exportCancel)
        page.addView(text("本机录音", 21, ink, true).apply { setPadding(0, dp(18), 0, dp(6)) })
        recover = button("检查并恢复中断录音") { recoverSessions() }
        page.addView(recover)
        list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        page.addView(list)
        page.addView(text("研究用途 · 非诊断工具\n没有PSG真值，不生成睡眠分期、个人准确率或健康分数。录音只在App私有目录保存；导出目的地由你选择。", 12, muted).apply { setPadding(0, dp(24), 0, 0) })
        page.addView(button("关于与开源许可") {
            val body = assets.open("third_party_notices.txt").bufferedReader().use { it.readText() }
            val view = ScrollView(this).apply { addView(text(body, 12, ink).apply { setPadding(dp(16), dp(12), dp(16), dp(12)) }) }
            AlertDialog.Builder(this).setTitle("NightLab 0.1.0 · 开源许可").setView(view).setPositiveButton("关闭", null).show()
        })
        setContentView(scroll)
        consent.setOnCheckedChangeListener { _, _ -> refreshStatus() }
        if (RecorderStorage.operations.owner == OperationGate.Owner.IDLE) recoverSessions() else loadSessions()
    }

    override fun onStart() { super.onStart(); visible = true }
    override fun onResume() { super.onResume(); handler.post(ticker) }
    override fun onPause() { handler.removeCallbacks(ticker); stopPlayback(); super.onPause() }
    override fun onStop() { visible = false; super.onStop() }
    override fun onSaveInstanceState(outState: Bundle) { outState.putString("pending_export", pendingExportId); super.onSaveInstanceState(outState) }
    override fun onDestroy() {
        cancelExport.set(true); handler.removeCallbacksAndMessages(null); stopPlayback(); io.shutdown()
        super.onDestroy()
    }

    private fun requestStart() {
        if (!consent.isChecked || RecordingService.snapshot.busy || RecorderStorage.operations.owner != OperationGate.Owner.IDLE || exporting) return
        val required = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) required += Manifest.permission.POST_NOTIFICATIONS
        val missing = required.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), 20) else beginCapture()
    }
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != 20) return
        val mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val notice = Build.VERSION.SDK_INT < 33 || checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (mic && notice && visible) beginCapture() else {
            feedback.text = "未开始录音。需要麦克风和通知权限，才能明确显示采集状态。"
            AlertDialog.Builder(this).setTitle("录音尚未开始").setMessage("你可以保留拒绝状态；若要采集，可在应用设置中允许麦克风与通知后，再点击开始。")
                .setNegativeButton("返回", null).setPositiveButton("打开应用设置") { _, _ ->
                    startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
                }.show()
        }
    }
    private fun beginCapture() {
        if (!visible || !consent.isChecked || RecordingService.snapshot.busy || RecorderStorage.operations.owner != OperationGate.Owner.IDLE || exporting) return
        val notifications = getSystemService(NotificationManager::class.java)
        if (!notifications.areNotificationsEnabled() || notifications.getNotificationChannel("nightlab_recording")?.importance == NotificationManager.IMPORTANCE_NONE) {
            feedback.text = "录音通知被关闭，尚未开始。请先在应用设置中允许通知。"
            startActivity(Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE, packageName))
            return
        }
        stopPlayback()
        if (!RecorderStorage.operations.tryAcquire(OperationGate.Owner.CAPTURE)) return
        try {
            startForegroundService(Intent(this, RecordingService::class.java).setAction(RecordingService.START))
            feedback.text = "正在启动；看到录音状态与通知后再离开页面。"
            start.isEnabled = false
            refreshStatus()
        } catch (error: RuntimeException) { RecorderStorage.operations.release(OperationGate.Owner.CAPTURE); refreshStatus(); feedback.text = getString(R.string.start_error, error.message ?: "未知错误") }
    }

    private fun refreshStatus() {
        val state = RecordingService.snapshot
        val owner = RecorderStorage.operations.owner
        val fileBusy = owner != OperationGate.Owner.IDLE
        statusText.text = when (owner) {
            OperationGate.Owner.RECOVERY -> "正在检查中断录音"
            OperationGate.Owner.EXPORT -> "正在导出，原录音保留"
            OperationGate.Owner.DELETE -> "正在删除本机录音"
            OperationGate.Owner.CAPTURE -> if (state.busy) state.message else "正在准备录音服务"
            else -> state.message
        }
        counter.text = getString(R.string.capture_counter, clock(state.frames / state.sampleRate), mib(state.frames * 2))
        level.progress = state.inputLevel
        start.isEnabled = consent.isChecked && !state.busy && !fileBusy && !exporting
        stop.isEnabled = state.busy && state.status != "stopping"
        consent.isEnabled = !state.busy && !fileBusy && !exporting
        recover.isEnabled = !state.busy && !fileBusy && !exporting && !loadingList.get()
        val buttonsBusy = state.busy || fileBusy || exporting
        if (lastButtonsBusy != buttonsBusy) { lastButtonsBusy = buttonsBusy; renderSessions() }
    }
    private fun loadSessions() {
        if (isDestroyed || isFinishing || io.isShutdown || exporting || !loadingList.compareAndSet(false, true)) return
        io.execute {
            try {
                val rows = store.listSessions()
                runOnUiThread {
                    if (!isDestroyed) {
                        val signature = rows.joinToString { it.toString() }
                        sessions = rows
                        if (signature != lastSignature) { lastSignature = signature; renderSessions() }
                    }
                }
            } catch (error: Exception) { showFeedback("录音列表读取失败，文件未删除：${error.message}") }
            finally { loadingList.set(false) }
        }
    }
    private fun recoverSessions() {
        if (isDestroyed || io.isShutdown || RecordingService.snapshot.busy || exporting) return
        if (!RecorderStorage.operations.tryAcquire(OperationGate.Owner.RECOVERY)) return
        recover.isEnabled = false
        refreshStatus()
        io.execute {
            try {
                check(!RecordingService.snapshot.busy) { "录音已经启动，恢复已停止" }
                val recovered = store.recoverInterruptedSessions()
                showFeedback(if (recovered.isEmpty()) "检查完成，没有待恢复的中断文件。" else "已检查${recovered.size}条中断记录；请查看状态和错误说明后导出。")
                runOnUiThread { lastSignature = ""; loadSessions() }
            } catch (error: Exception) { showFeedback("恢复未完成，音频保留：${error.message}") }
            finally { RecorderStorage.operations.release(OperationGate.Owner.RECOVERY); runOnUiThread { if (!isDestroyed) refreshStatus() } }
        }
    }
    private fun renderSessions() {
        list.removeAllViews()
        if (sessions.isEmpty()) { list.addView(text("还没有录音。先做一段短录，检查声音与导出文件。", 14, muted)); return }
        for (session in sessions) {
            val id = session.getString("session_id")
            val chunks = session.getJSONArray("chunks")
            val format = session.getJSONObject("recording_config")
            val rate = format.getInt("sample_rate")
            var frames = 0L
            for (i in 0 until chunks.length()) frames += chunks.getJSONObject(i).getLong("frames")
            val status = session.getString("status")
            val panel = card()
            val date = OffsetDateTime.parse(session.getString("start_time")).format(DateTimeFormatter.ofPattern("MM-dd HH:mm:ss XXX"))
            panel.addView(text(date, 18, ink, true))
            val statusName = mapOf("recording" to "正在录制", "completed" to "已保存", "interrupted" to "中断记录", "failed" to "录音异常")[status] ?: status
            panel.addView(text("$statusName · ${clock(frames / rate)} · ${chunks.length()}个分片", 14, teal))
            panel.addView(text("${rate / 1000.0} kHz · 单声道PCM16", 12, muted))
            if (!session.isNull("error")) panel.addView(text(session.getString("error"), 13, Color.rgb(146, 67, 48)))
            val actions = row()
            val play = button("试听首片") { audition(id, chunks.getJSONObject(0).getString("filename")) }
            val export = button("导出") { chooseExport(id, date) }
            val delete = button("删除") { confirmDelete(id, date) }
            play.isEnabled = chunks.length() > 0 && status != "recording" && RecorderStorage.operations.owner == OperationGate.Owner.IDLE && !exporting && !RecordingService.snapshot.busy
            export.isEnabled = chunks.length() > 0 && status != "recording" && RecorderStorage.operations.owner == OperationGate.Owner.IDLE && !exporting && !RecordingService.snapshot.busy
            delete.isEnabled = status != "recording" && RecorderStorage.operations.owner == OperationGate.Owner.IDLE && !exporting && !RecordingService.snapshot.busy
            addEqual(actions, play); addEqual(actions, export); addEqual(actions, delete); panel.addView(actions)
            list.addView(panel)
        }
    }
    private fun audition(id: String, filename: String) {
        stopPlayback()
        val root = java.io.File(filesDir, "recordings").canonicalFile
        val folder = java.io.File(root, id).canonicalFile
        val file = java.io.File(folder, filename).canonicalFile
        if (folder.parentFile != root || file.parentFile != folder || !file.isFile) { feedback.text = "录音文件位置无效"; return }
        try {
            player = MediaPlayer().apply {
                setDataSource(file.absolutePath)
                setOnPreparedListener { if (player === it) { it.start(); stopPreview.visibility = View.VISIBLE; feedback.text = "正在试听首个分片；离开本页会停止播放。" } }
                setOnCompletionListener { stopPlayback(); feedback.text = "试听结束。" }
                setOnErrorListener { _, _, _ -> stopPlayback(); feedback.text = "试听失败，原文件保留。"; true }
                prepareAsync()
            }
        } catch (error: Exception) { stopPlayback(); feedback.text = getString(R.string.play_error, error.message ?: "未知错误") }
    }
    private fun stopPlayback() { player?.release(); player = null; if (this::stopPreview.isInitialized) stopPreview.visibility = View.GONE }

    private fun chooseExport(id: String, date: String) {
        if (RecordingService.snapshot.busy || exporting) return
        stopPlayback()
        pendingExportId = id
        startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE); type = "application/zip"
            putExtra(Intent.EXTRA_TITLE, "NightLab-${date.take(14).replace(':','-').replace(' ','_')}.zip")
        }, 30)
    }
    @Deprecated("Framework activity result API supports this app's API26 baseline")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != 30 || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val id = pendingExportId ?: return
        pendingExportId = null
        if (RecordingService.snapshot.busy) { feedback.text = "录音中无法导出。请停止后重试；目标位置可能留有空文件。"; return }
        if (!RecorderStorage.operations.tryAcquire(OperationGate.Owner.EXPORT)) { feedback.text = "正在处理其他操作，请稍后重新导出。"; return }
        exporting = true; cancelExport.set(false); exportCancel.visibility = View.VISIBLE
        refreshStatus(); renderSessions()
        feedback.text = "正在校验并导出，原录音保留在本机。"
        io.execute {
            try {
                contentResolver.openOutputStream(uri, "w").use { output ->
                    requireNotNull(output) { "无法写入所选位置" }
                    var lastUpdate = 0L
                    store.exportSession(id, output, { cancelExport.get() }) { bytes ->
                        val now = SystemClock.elapsedRealtime()
                        if (now - lastUpdate >= 500) { showFeedback("已导出 ${mib(bytes)} MiB…"); lastUpdate = now }
                    }
                }
                showFeedback("导出完成。ZIP包含WAV分片和session.json；解压后可导入桌面。")
            } catch (_: CancellationException) {
                showFeedback("导出已取消，原录音保留；请删除所选位置的未完成ZIP。")
            } catch (error: Exception) { showFeedback("导出未完成，原录音保留：${error.message}。请删除未完成ZIP。") }
            finally {
                RecorderStorage.operations.release(OperationGate.Owner.EXPORT)
                runOnUiThread { if (!isDestroyed) { exporting = false; exportCancel.visibility = View.GONE; refreshStatus(); renderSessions() } }
            }
        }
    }
    private fun confirmDelete(id: String, date: String) {
        AlertDialog.Builder(this).setTitle("删除这条本机录音？")
            .setMessage("将删除 $date 的WAV分片与采集元信息，无法撤销。已导出的副本不会一起删除。")
            .setNegativeButton("保留", null).setPositiveButton("确认删除") { _, _ ->
                stopPlayback()
                if (!RecorderStorage.operations.tryAcquire(OperationGate.Owner.DELETE)) { feedback.text = "正在处理其他操作，尚未删除。"; return@setPositiveButton }
                refreshStatus()
                io.execute {
                    try { store.deleteSession(id); showFeedback("本机录音已删除。"); runOnUiThread { lastSignature = ""; loadSessions() } }
                    catch (error: Exception) { showFeedback("删除未完成：${error.message}") }
                    finally { RecorderStorage.operations.release(OperationGate.Owner.DELETE) }
                }
            }.show()
    }
    private fun showFeedback(message: String) { runOnUiThread { if (!isDestroyed) feedback.text = message } }
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun text(value: String, size: Int, color: Int, bold: Boolean = false) = TextView(this).apply {
        text = value; textSize = size.toFloat(); setTextColor(color); setLineSpacing(dp(3).toFloat(), 1f)
        if (bold) setTypeface(typeface, Typeface.BOLD)
    }
    private fun row() = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(8), 0, 0) }
    private fun button(title: String, action: () -> Unit) = Button(this).apply {
        text = title; textSize = 14f; isAllCaps = false; setTextColor(teal); setOnClickListener { action() }
    }
    private fun addEqual(row: LinearLayout, view: View) { row.addView(view, LinearLayout.LayoutParams(0, -2, 1f)) }
    private fun card() = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL; setPadding(dp(17), dp(18), dp(17), dp(18))
        background = GradientDrawable().apply { setColor(Color.WHITE); cornerRadius = dp(16).toFloat(); setStroke(dp(1), Color.rgb(228, 234, 228)) }
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(18) }
    }
    private fun clock(seconds: Long): String = String.format(Locale.US, "%02d:%02d:%02d", seconds / 3600, seconds / 60 % 60, seconds % 60)
    private fun mib(bytes: Long): String = String.format(Locale.US, "%.1f", bytes / 1048576.0)
}
