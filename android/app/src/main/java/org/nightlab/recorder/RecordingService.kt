package org.nightlab.recorder

import android.Manifest
import android.app.*
import android.content.*
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.*
import android.graphics.drawable.Icon
import android.os.*
import org.json.JSONObject
import java.io.File
import java.time.OffsetDateTime
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/** One explicitly started microphone session; never restarts after process death. */
class RecordingService : Service() {
    data class LiveState(
        val status: String = "idle", val sessionId: String? = null,
        val frames: Long = 0, val sampleRate: Int = 16000,
        val inputLevel: Int = 0, val message: String = "尚未开始录音",
    ) {
        val busy: Boolean get() = status in setOf("starting", "recording", "stopping")
    }
    companion object {
        const val START = "org.nightlab.recorder.START"
        const val STOP = "org.nightlab.recorder.STOP"
        private const val CHANNEL = "nightlab_recording"
        private const val NOTICE = 10
        private const val MAX_SECONDS = 12 * 3600L
        @Volatile var snapshot = LiveState()
            private set
    }
    private val worker = Executors.newSingleThreadExecutor()
    private val stopRequested = AtomicBoolean(false)
    private val active = AtomicBoolean(false)
    private val handler = Handler(Looper.getMainLooper())
    @Volatile private var recorder: AudioRecord? = null
    @Volatile private var lastReadNanos = 0L
    @Volatile private var stopReason = "user_stop"
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var store: SessionStore
    private val watchdog = object : Runnable {
        override fun run() {
            if (active.get()) {
                if (!stopRequested.get() && lastReadNanos > 0 && SystemClock.elapsedRealtimeNanos() - lastReadNanos > 15_000_000_000L) {
                    requestStop("input_stalled")
                }
                handler.postDelayed(this, 3000)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        store = RecorderStorage.get(this)
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "本人声音采集", NotificationManager.IMPORTANCE_LOW).apply {
                description = "录音期间保持可见，可随时停止"; setSound(null, null)
            })
    }
    override fun onBind(intent: Intent?) = null
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) { if (active.get()) requestStop("user_stop") else stopSelf(startId); return START_NOT_STICKY }
        if (intent?.action != START || !active.compareAndSet(false, true)) return START_NOT_STICKY
        val gate = RecorderStorage.operations
        if (gate.owner != OperationGate.Owner.CAPTURE && !gate.tryAcquire(OperationGate.Owner.CAPTURE)) {
            active.set(false); stopSelf(startId); return START_NOT_STICKY
        }
        stopRequested.set(false)
        stopReason = "user_stop"
        snapshot = LiveState(status = "starting", message = "正在准备麦克风")
        try {
            check(checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) { "尚未允许麦克风权限" }
            val notification = notification("正在准备录音", true)
            if (Build.VERSION.SDK_INT >= 30) startForeground(NOTICE, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            else startForeground(NOTICE, notification)
            worker.execute { capture(startId) }
        } catch (error: Exception) {
            active.set(false)
            snapshot = LiveState(status = "failed", message = error.message ?: "无法启动录音")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
            gate.release(OperationGate.Owner.CAPTURE)
        }
        return START_NOT_STICKY
    }

    private fun requestStop(reason: String) {
        if (!active.get()) return
        if (!stopRequested.getAndSet(true)) stopReason = reason
        snapshot = snapshot.copy(status = "stopping", message = "正在停止并保存录音")
        try { recorder?.stop() } catch (_: IllegalStateException) { /* Worker owns release. */ }
    }

    @android.annotation.SuppressLint("MissingPermission")
    private fun openRecorder(): Pair<AudioRecord, String> {
        check(checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) { "麦克风权限已撤销" }
        val manager = getSystemService(AudioManager::class.java)
        val unprocessed = manager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val sources = if (unprocessed) listOf(MediaRecorder.AudioSource.UNPROCESSED, MediaRecorder.AudioSource.VOICE_RECOGNITION)
                      else listOf(MediaRecorder.AudioSource.VOICE_RECOGNITION)
        for (source in sources) for (rate in listOf(16000, 48000, 44100)) {
            val minimum = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minimum <= 0) continue
            val candidate = try {
                AudioRecord.Builder().setAudioSource(source)
                    .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build())
                    .setBufferSizeInBytes(maxOf(minimum * 4, rate * 8)).build()
            } catch (_: IllegalArgumentException) { continue }
            if (candidate.state == AudioRecord.STATE_INITIALIZED && candidate.sampleRate == rate && candidate.channelCount == 1) {
                return candidate to if (source == MediaRecorder.AudioSource.UNPROCESSED) "UNPROCESSED" else "VOICE_RECOGNITION"
            }
            candidate.release()
        }
        error("设备未提供可用的单声道PCM录音格式")
    }

    private fun capture(startId: Int) {
        var sessionId: String? = null
        var writer: PcmWavWriter? = null
        var captureStart = 0L
        var sampleRate = 16000
        var totalFrames = 0L
        var chunkStartFrame = 0L
        var chunkIndex = 0
        var failure: String? = null
        var record: AudioRecord? = null
        var callback: AudioManager.AudioRecordingCallback? = null
        fun sealChunk(status: String) {
            val current = writer ?: return
            if (current.framesWritten > 0) {
                val summary = current.finish()
                writer = null
                store.finalizeChunk(sessionId!!, chunkIndex, summary, chunkStartFrame, status)
                chunkIndex++
                chunkStartFrame = totalFrames
            } else {
                current.close(); writer = null
                check(current.file.delete()) { "空录音片段清理失败" }
            }
        }
        try {
            check(StatFs(filesDir.absolutePath).availableBytes > 64L * 1024 * 1024) { "本地存储不足64MiB，未开始录音" }
            store.recoverInterruptedSessions()
            val opened = openRecorder()
            record = opened.first
            recorder = record
            val rate = record.sampleRate
            sampleRate = rate
            captureStart = SystemClock.elapsedRealtimeNanos()
            sessionId = store.createSession(rate, opened.second, JSONObject().put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL).put("android_api", Build.VERSION.SDK_INT), OffsetDateTime.now().toString(), captureStart)
            if (Build.VERSION.SDK_INT >= 29) {
                callback = object : AudioManager.AudioRecordingCallback() {
                    override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>) {
                        if (configs.any { it.isClientSilenced }) requestStop("microphone_silenced")
                    }
                }
                record.registerAudioRecordingCallback(mainExecutor, callback)
            }
            wakeLock = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "NightLab:AudioCapture").apply {
                setReferenceCounted(false); acquire((MAX_SECONDS + 60) * 1000)
            }
            if (stopRequested.get()) return
            record.startRecording()
            check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "麦克风没有进入录音状态" }
            lastReadNanos = SystemClock.elapsedRealtimeNanos()
            handler.post(watchdog)
            val buffer = ShortArray(3200)
            var lastCheckpoint = lastReadNanos
            var lastNotice = lastReadNanos
            snapshot = LiveState("recording", sessionId, 0, rate, 0, "正在录音，仅保存在本机")
            while (!stopRequested.get()) {
                if (writer == null) writer = PcmWavWriter(store.chunkFile(sessionId, chunkIndex), rate)
                val current = writer!!
                val remaining = rate.toLong() * 3600 - current.framesWritten
                val wanted = minOf(buffer.size.toLong(), remaining).toInt()
                val read = record.read(buffer, 0, wanted, AudioRecord.READ_BLOCKING)
                if (stopRequested.get()) break
                check(read >= 0) { "麦克风读取中断（$read）" }
                if (read == 0) { Thread.sleep(10); continue }
                current.append(buffer, read)
                totalFrames += read
                val now = SystemClock.elapsedRealtimeNanos()
                lastReadNanos = now
                var energy = 0.0
                for (sample in 0 until read) { val x = buffer[sample].toDouble() / 32768; energy += x * x }
                snapshot = LiveState("recording", sessionId, totalFrames, rate,
                    (sqrt(energy / read) * 100).toInt().coerceIn(0, 100), "正在录音，仅保存在本机")
                if (now - lastCheckpoint >= 30_000_000_000L) {
                    current.checkpoint(); lastCheckpoint = now
                    if (StatFs(filesDir.absolutePath).availableBytes < 32L * 1024 * 1024) requestStop("low_storage")
                }
                if (now - lastNotice >= 10_000_000_000L) {
                    getSystemService(NotificationManager::class.java).notify(NOTICE,
                        notification("已录制 ${totalFrames / rate / 60} 分钟 · 点击停止可保存", true))
                    lastNotice = now
                }
                if (current.framesWritten == rate.toLong() * 3600) sealChunk("completed")
                if (totalFrames >= rate * MAX_SECONDS) requestStop("duration_limit")
            }
            sealChunk(if (stopReason in setOf("user_stop", "duration_limit")) "completed" else "interrupted")
        } catch (error: Exception) {
            failure = error.message ?: "录音执行失败"
            stopReason = if (stopReason == "user_stop") "capture_error" else stopReason
            try { sealChunk("interrupted") } catch (_: Exception) { /* .part remains for explicit recovery. */ }
        } finally {
            handler.removeCallbacks(watchdog)
            try { record?.stop() } catch (_: IllegalStateException) { }
            try { if (Build.VERSION.SDK_INT >= 29 && callback != null) record?.unregisterAudioRecordingCallback(callback) } catch (_: RuntimeException) { }
            try { record?.release() } catch (_: RuntimeException) { }
            recorder = null
            try { writer?.close() } catch (_: Exception) { }
            if (wakeLock?.isHeld == true) wakeLock?.release()
            wakeLock = null
            val complete = failure == null && totalFrames > 0 && stopReason in setOf("user_stop", "duration_limit")
            val status = if (complete) "completed" else "interrupted"
            if (sessionId != null) {
                try {
                    store.updateSession(sessionId, status, stopReason,
                        (SystemClock.elapsedRealtimeNanos() - captureStart).coerceAtLeast(0) / 1_000_000_000.0, failure)
                } catch (error: Exception) { failure = "音频已保留，但元信息保存失败：${error.message}" }
            }
            active.set(false)
            snapshot = LiveState(if (complete && failure == null) "completed" else "failed", sessionId,
                totalFrames, sampleRate, 0,
                failure ?: if (complete) "录音已保存，可导出到桌面" else "录音已中断；保留已写入的音频，原因：$stopReason")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
            RecorderStorage.operations.release(OperationGate.Owner.CAPTURE)
        }
    }

    private fun notification(text: String, ongoing: Boolean): Notification {
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), flags)
        val stop = PendingIntent.getService(this, 1, Intent(this, RecordingService::class.java).setAction(STOP), flags)
        return Notification.Builder(this, CHANNEL).setSmallIcon(R.drawable.ic_recording).setContentTitle("NightLab 本人声音采集")
            .setContentText(text).setContentIntent(open).setOngoing(ongoing).setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(Icon.createWithResource(this, R.drawable.ic_recording), "停止并保存", stop).build()).build()
    }
    override fun onDestroy() {
        if (active.get()) requestStop("service_destroyed")
        worker.shutdown()
        super.onDestroy()
    }
}
