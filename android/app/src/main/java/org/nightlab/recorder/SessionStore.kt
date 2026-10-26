package org.nightlab.recorder

import java.io.File
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CancellationException
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONArray
import org.json.JSONObject

/** Private, local recording storage. Instances must not be used by concurrent app processes. */
class SessionStore(root: File) {
    private val root: File
    init {
        require(!Files.isSymbolicLink(root.toPath())) { "录音目录不能是符号链接" }
        // The caller supplies a trusted app-private root. Android may alias its ancestors
        // (/data/user/0 and /data/data); pin the canonical directory before creating children.
        this.root = root.canonicalFile
        Files.createDirectories(this.root.toPath())
        require(this.root.isDirectory) { "录音目录不可用" }
    }

    @Synchronized
    fun createSession(sampleRate: Int, audioSource: String, device: JSONObject,
        startTime: String, startedElapsedNanos: Long): String {
        require(sampleRate in RATES) { "不支持的采样率" }
        require(audioSource.isNotBlank() && startedElapsedNanos >= 0) { "录音配置无效" }
        OffsetDateTime.parse(startTime)
        val id = UUID.randomUUID().toString()
        Files.createDirectory(File(root, id).toPath())
        val safeDevice = JSONObject()
        for (key in listOf("manufacturer", "model")) if (device.has(key)) safeDevice.put(key, device.getString(key))
        if (device.has("android_api")) safeDevice.put("android_api", device.getInt("android_api"))
        val metadata = JSONObject().put("schema_version", 1).put("source", "nightlab_android")
            .put("version", 1).put("session_id", id).put("start_time", startTime)
            .put("started_elapsed_realtime_nanos", startedElapsedNanos)
            .put("recording_config", JSONObject().put("sample_rate", sampleRate).put("channels", 1)
                .put("encoding", "PCM16").put("audio_source", audioSource))
            .put("device", safeDevice).put("status", "recording").put("stop_reason", JSONObject.NULL)
            .put("error", JSONObject.NULL).put("elapsed_seconds", 0.0).put("elapsed_seconds_measured", false).put("chunks", JSONArray())
            .put("time_basis", JSONObject().put("chunk_timestamps", "nominal_sample_count")
                .put("wall_clock_anchor", "captured_around_audio_record_start")
                .put("elapsed_clock", "elapsed_realtime_nanos").put("external_synchronization", false))
        save(id, metadata)
        return id
    }

    @Synchronized
    fun chunkFile(id: String, index: Int): File {
        require(index in 0..99999) { "录音分片编号无效" }
        val metadata = read(id)
        require(metadata.getString("status") == "recording") { "该录音已停止，不能继续写入" }
        val part = child(id, chunkName(index) + ".part")
        require(!Files.exists(part.toPath(), LinkOption.NOFOLLOW_LINKS) &&
            !Files.exists(child(id, chunkName(index)).toPath(), LinkOption.NOFOLLOW_LINKS)) { "录音分片已经存在" }
        return part
    }

    @Synchronized
    fun finalizeChunk(id: String, index: Int, summary: WavSummary, startFrame: Long, status: String): JSONObject {
        require(index in 0..99999 && status in setOf("completed", "interrupted")) { "录音分片状态无效" }
        val metadata = read(id)
        require(metadata.getString("status") == "recording") { "该录音已停止，不能提交分片" }
        val chunks = metadata.getJSONArray("chunks")
        require((0 until chunks.length()).none { chunks.getJSONObject(it).getInt("index") == index }) { "录音分片编号重复" }
        require(startFrame >= 0 && startFrame == frames(metadata)) { "录音分片起点与已记录帧数不符" }
        val rate = metadata.getJSONObject("recording_config").getInt("sample_rate")
        val part = child(id, chunkName(index) + ".part")
        val actual = inspectWav(part, rate)
        require(actual == summary) { "录音分片校验失败：摘要或帧数不符" }
        val final = child(id, chunkName(index))
        require(!Files.exists(final.toPath(), LinkOption.NOFOLLOW_LINKS)) { "目标录音分片已经存在" }
        // No REPLACE_EXISTING: a prior final file must never be overwritten.
        Files.move(part.toPath(), final.toPath())
        val chunk = appendChunk(metadata, index, summary, startFrame, status, final)
        save(id, metadata)
        return JSONObject(chunk.toString())
    }

    private fun appendChunk(metadata: JSONObject, index: Int, summary: WavSummary,
        startFrame: Long, status: String, final: File): JSONObject {
        val rate = metadata.getJSONObject("recording_config").getInt("sample_rate")
        val chunk = JSONObject().put("index", index).put("filename", final.name)
            .put("start_frame", startFrame).put("frames", summary.frames).put("sample_rate", rate)
            .put("start_time", frameTime(metadata.getString("start_time"), startFrame, rate))
            .put("end_time", frameTime(metadata.getString("start_time"), startFrame + summary.frames, rate))
            .put("duration_seconds", summary.durationSeconds).put("bytes", final.length())
            .put("sha256", summary.sha256).put("status", status)
        metadata.getJSONArray("chunks").put(chunk)
        return chunk
    }

    @Synchronized
    fun updateSession(id: String, status: String, stopReason: String, elapsedSeconds: Double, error: String? = null) {
        require(status in setOf("completed", "interrupted", "failed")) { "录音结束状态无效" }
        require(elapsedSeconds.isFinite() && elapsedSeconds >= 0) { "录音时长无效" }
        val metadata = read(id)
        require(metadata.getString("status") == "recording") { "该录音已经结束" }
        metadata.put("status", status).put("stop_reason", stopReason).put("elapsed_seconds", elapsedSeconds)
            .put("elapsed_seconds_measured", true).put("error", error ?: JSONObject.NULL)
        addDurationDiagnostics(metadata, elapsedSeconds)
        save(id, metadata)
    }

    @Synchronized
    fun read(id: String): JSONObject {
        val file = child(id, "session.json")
        requireRegular(file)
        val metadata = JSONObject(file.readText(Charsets.UTF_8))
        require(metadata.getInt("schema_version") == 1 && metadata.getString("source") == "nightlab_android" &&
            metadata.getString("session_id") == id) { "录音元数据格式无效" }
        require(metadata.getJSONObject("recording_config").getInt("sample_rate") in RATES) { "录音采样率无效" }
        require(metadata.getString("status") in setOf("recording", "completed", "interrupted", "failed")) { "录音状态未知" }
        require(metadata.getJSONObject("recording_config").getInt("channels") == 1 &&
            metadata.getJSONObject("recording_config").getString("encoding") == "PCM16") { "录音编码格式无效" }
        return metadata
    }

    @Synchronized
    fun totalFrames(id: String): Long = frames(read(id))

    @Synchronized
    fun listSessions(): List<JSONObject> = sessionIds().map { read(it) }
        .sortedByDescending { OffsetDateTime.parse(it.getString("start_time")).toInstant() }

    /** Call on service startup or while idle; activeId protects the live writer's files. */
    @Synchronized
    fun recoverInterruptedSessions(activeId: String? = null): List<String> {
        val recovered = mutableListOf<String>()
        for (id in sessionIds()) {
            if (id == activeId) continue
            var metadata = read(id)
            val files = checkedFiles(id)
            val chunks = metadata.getJSONArray("chunks")
            val known = (0 until chunks.length()).map { chunks.getJSONObject(it).getString("filename") }.toSet()
            val parts = files.filter { PART.matches(it.name) }.sortedBy { it.name }
            val unknownParts = files.filter { it.name.endsWith(".part") && !PART.matches(it.name) }
            val orphans = files.filter { FINAL.matches(it.name) && it.name !in known }
            val unknownFinals = files.filter { it.name.endsWith(".wav") && !FINAL.matches(it.name) }
            if (metadata.getString("status") != "recording" && parts.isEmpty() && unknownParts.isEmpty() && orphans.isEmpty() && unknownFinals.isEmpty()) continue
            val measuredElapsed = if (metadata.optBoolean("elapsed_seconds_measured", false) && !metadata.isNull("elapsed_seconds")) {
                metadata.getDouble("elapsed_seconds")
            } else null
            val errors = mutableListOf<String>()
            for (unknown in unknownParts) {
                errors += "needs_manual_recovery：${unknown.name} 的临时分片文件名未知，已保留"
            }
            for (unknown in unknownFinals) {
                errors += "needs_manual_recovery：${unknown.name} 的最终分片文件名未知，已保留"
            }
            if (parts.isNotEmpty() && metadata.getString("status") != "recording") {
                // A stopped writer can leave valid .part audio after an I/O failure.
                // This transient state is private to the synchronized repair operation.
                metadata.put("recovery_previous_status", metadata.getString("status")).put("status", "recording")
                save(id, metadata)
            }
            // Merge both crash windows in index order: payload still .part, or already renamed
            // but not yet registered. Never infer a missing earlier chunk or overwrite audio.
            val candidates = (parts + orphans).sortedWith(compareBy<File> { it.name.substring(6, 11).toInt() }.thenBy { it.name })
            for (candidate in candidates) {
                try {
                    val index = candidate.name.substring(6, 11).toInt()
                    val currentMetadata = read(id)
                    require(index == nextSequentialIndex(currentMetadata)) { "分片编号缺失、重复或顺序不符，已保留音频" }
                    if (candidate.name.endsWith(".part")) {
                        require(!Files.exists(child(id, chunkName(index)).toPath(), LinkOption.NOFOLLOW_LINKS)) {
                            "分片编号或最终文件冲突，已保留临时音频"
                        }
                        val summary = PcmWavWriter.recover(candidate)
                        finalizeChunk(id, index, summary, totalFrames(id), "interrupted")
                    } else {
                        require(!Files.exists(child(id, chunkName(index) + ".part").toPath(), LinkOption.NOFOLLOW_LINKS)) {
                            "最终分片与临时分片冲突，已保留两份音频"
                        }
                        val summary = inspectWav(candidate, currentMetadata.getJSONObject("recording_config").getInt("sample_rate"))
                        appendChunk(currentMetadata, index, summary, frames(currentMetadata), "interrupted", candidate)
                        save(id, currentMetadata)
                    }
                } catch (failure: Exception) {
                    errors += "needs_manual_recovery：${candidate.name} 恢复失败：${failure.message ?: failure.javaClass.simpleName}"
                }
            }
            metadata = read(id)
            metadata.put("status", "interrupted").put("stop_reason", "process_interrupted")
                .put("elapsed_seconds", measuredElapsed ?: JSONObject.NULL).put("elapsed_seconds_measured", measuredElapsed != null)
                .put("error", if (errors.isEmpty()) JSONObject.NULL else errors.joinToString("；"))
                .put("recovery_errors", JSONArray(errors))
            addDurationDiagnostics(metadata, measuredElapsed)
            save(id, metadata)
            recovered += id
        }
        return recovered
    }

    /** Streams a ZIP without taking ownership of the caller's output stream. Cancellation throws. */
    @Synchronized
    fun exportSession(id: String, output: OutputStream, cancelled: () -> Boolean = { false }, onProgress: (Long) -> Unit = {}) {
        val metadata = read(id)
        require(metadata.getString("status") in setOf("completed", "interrupted", "failed")) { "录音尚未停止，不能导出" }
        val chunks = metadata.getJSONArray("chunks")
        require(chunks.length() > 0) { "没有可导出的录音分片" }
        val allFiles = checkedFiles(id)
        val names = (0 until chunks.length()).map { chunks.getJSONObject(it).getString("filename") }
        require(names.distinct().size == names.size && names.all { FINAL.matches(it) }) { "录音分片文件名无效" }
        require(allFiles.none { it.name.endsWith(".part") || (it.name.endsWith(".wav") && it.name !in names) }) { "录音仍有未恢复的分片，暂不能导出" }
        require(metadata.optJSONArray("recovery_errors")?.length().let { it == null || it == 0 }) { "录音存在未解决的恢复错误，暂不能导出" }
        val rate = metadata.getJSONObject("recording_config").getInt("sample_rate")
        var expectedFrame = 0L
        for (index in 0 until chunks.length()) {
            checkCancellation(cancelled)
            val chunk = chunks.getJSONObject(index)
            require(chunk.getLong("start_frame") == expectedFrame && chunk.getInt("sample_rate") == rate) { "录音分片顺序无效" }
            val file = child(id, chunk.getString("filename"))
            val actual = inspectWav(file, rate, cancelled)
            validateChunk(chunk, file, actual)
            expectedFrame += actual.frames
        }
        checkCancellation(cancelled)
        val zip = object : ZipOutputStream(object : OutputStream() {
            override fun write(value: Int) = output.write(value)
            override fun write(bytes: ByteArray, offset: Int, length: Int) = output.write(bytes, offset, length)
            override fun flush() = output.flush()
            override fun close() = output.flush()
        }) {
            fun releaseCompressor() { def.end() }
        }
        zip.setLevel(0)
        var copied = 0L
        try {
            zip.putNextEntry(ZipEntry("session.json"))
            zip.write((exportSafe(metadata) as JSONObject).toString(2).toByteArray(Charsets.UTF_8))
            zip.closeEntry()
            val buffer = ByteArray(65536)
            for (index in 0 until chunks.length()) {
                checkCancellation(cancelled)
                val chunk = chunks.getJSONObject(index)
                val file = child(id, chunk.getString("filename"))
                requireRegular(file)
                zip.putNextEntry(ZipEntry(file.name))
                val digest = MessageDigest.getInstance("SHA-256")
                var fileBytes = 0L
                file.inputStream().use { input ->
                    while (true) {
                        checkCancellation(cancelled)
                        val count = input.read(buffer)
                        if (count < 0) break
                        zip.write(buffer, 0, count)
                        digest.update(buffer, 0, count)
                        fileBytes += count
                        copied += count
                        onProgress(copied)
                    }
                }
                require(fileBytes == chunk.getLong("bytes") && hex(digest.digest()) == chunk.getString("sha256")) { "导出时录音发生变化，导出已中止" }
                zip.closeEntry()
            }
            checkCancellation(cancelled)
            zip.finish()
            zip.flush()
        } finally {
            // Release native compression state without implicitly finishing a failed ZIP.
            zip.releaseCompressor()
        }
        // Do not close on failure: close() would implicitly finish an incomplete ZIP.
    }

    @Synchronized
    fun deleteSession(id: String) {
        val metadata = read(id)
        require(metadata.getString("status") in setOf("completed", "interrupted", "failed")) { "录音尚未结束，不能删除" }
        val files = checkedFiles(id) // Full preflight before any deletion, including unknown entries.
        for (file in files) Files.delete(file.toPath())
        Files.delete(directory(id).toPath())
    }

    private fun sessionIds(): List<String> {
        require(!Files.isSymbolicLink(root.toPath()) && root.canonicalFile == root.absoluteFile.toPath().normalize().toFile()) { "录音目录不安全" }
        return (root.listFiles() ?: error("无法读取录音目录")).filter { UUID_PATTERN.matches(it.name) }
            .map { directory(it.name); it.name }
    }

    private fun directory(id: String): File {
        require(UUID_PATTERN.matches(id) && UUID.fromString(id).toString() == id) { "录音编号无效" }
        require(!Files.isSymbolicLink(root.toPath()) && root.canonicalFile == root.absoluteFile.toPath().normalize().toFile()) { "录音目录不安全" }
        val directory = File(root, id)
        require(!Files.isSymbolicLink(directory.toPath()) && directory.isDirectory && directory.canonicalFile.parentFile == root.canonicalFile) { "录音目录不存在或不安全" }
        return directory
    }

    private fun child(id: String, name: String): File {
        require(name.isNotBlank() && name != "." && name != ".." && !name.contains('/') && !name.contains('\\')) { "录音文件名无效" }
        val parent = directory(id)
        val file = File(parent, name)
        require(!Files.isSymbolicLink(file.toPath()) && file.canonicalFile.parentFile == parent.canonicalFile) { "录音文件路径不安全" }
        return file
    }

    private fun requireRegular(file: File) {
        require(!Files.isSymbolicLink(file.toPath()) && Files.isRegularFile(file.toPath(), LinkOption.NOFOLLOW_LINKS)) { "录音文件不存在或不安全：${file.name}" }
    }

    private fun checkedFiles(id: String): List<File> {
        return (directory(id).listFiles() ?: error("无法读取录音目录")).map { file ->
            child(id, file.name).also { requireRegular(it) }
        }
    }

    private fun save(id: String, metadata: JSONObject) {
        val target = child(id, "session.json")
        val temporary = child(id, ".session-${UUID.randomUUID()}.tmp")
        try {
            FileChannel.open(temporary.toPath(), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE).use { channel ->
                val bytes = ByteBuffer.wrap(metadata.toString(2).toByteArray(Charsets.UTF_8))
                while (bytes.hasRemaining()) check(channel.write(bytes) > 0) { "无法写入录音元数据" }
                channel.force(true)
            }
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally {
            Files.deleteIfExists(temporary.toPath())
        }
    }

    private fun frames(metadata: JSONObject): Long {
        val chunks = metadata.getJSONArray("chunks")
        var result = 0L
        for (index in 0 until chunks.length()) {
            val count = chunks.getJSONObject(index).getLong("frames")
            require(count >= 0 && count <= Long.MAX_VALUE - result) { "录音帧数无效" }
            result += count
        }
        return result
    }

    private fun nextSequentialIndex(metadata: JSONObject): Int {
        val chunks = metadata.getJSONArray("chunks")
        var startFrame = 0L
        for (index in 0 until chunks.length()) {
            val chunk = chunks.getJSONObject(index)
            require(chunk.getInt("index") == index && chunk.getString("filename") == chunkName(index) &&
                chunk.getLong("start_frame") == startFrame && chunk.getLong("frames") >= 0 &&
                chunk.getLong("frames") <= Long.MAX_VALUE - startFrame) { "已有分片编号、文件名或帧顺序无效" }
            startFrame += chunk.getLong("frames")
        }
        return chunks.length()
    }

    private fun inspectWav(file: File, rate: Int, cancelled: () -> Boolean = { false }): WavSummary {
        requireRegular(file)
        val length = file.length()
        require(length >= 44 && length - 8 <= 0xffffffffL) { "WAV 文件长度无效" }
        val bytes = ByteArray(44)
        RandomAccessFile(file, "r").use { it.readFully(bytes) }
        val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        fun uint(offset: Int) = header.getInt(offset).toLong() and 0xffffffffL
        fun short(offset: Int) = header.getShort(offset).toInt() and 65535
        fun tag(offset: Int, text: String) = String(bytes, offset, 4, Charsets.US_ASCII) == text
        require(tag(0, "RIFF") && tag(8, "WAVE") && tag(12, "fmt ") && tag(36, "data") && uint(16) == 16L &&
            short(20) == 1 && short(22) == 1 && short(32) == 2 && short(34) == 16 && uint(24) == rate.toLong() && uint(28) == rate * 2L &&
            uint(4) == length - 8 && uint(40) == length - 44 && (length - 44) % 2 == 0L) { "WAV 格式或文件长度校验失败" }
        val frames = (length - 44) / 2
        require(frames <= rate * 3600L) { "录音分片超过一小时" }
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(65536)
            while (true) {
                checkCancellation(cancelled)
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return WavSummary(frames, length - 44, hex(digest.digest()), frames.toDouble() / rate)
    }

    private fun validateChunk(chunk: JSONObject, file: File, summary: WavSummary) {
        require(chunk.getLong("bytes") == file.length() && chunk.getLong("frames") == summary.frames &&
            chunk.getString("sha256") == summary.sha256 && chunk.getDouble("duration_seconds") == summary.durationSeconds &&
            chunk.getString("status") in setOf("completed", "interrupted")) { "录音分片已变化或元数据不符：${file.name}" }
    }

    private fun frameTime(start: String, frames: Long, rate: Int): String = OffsetDateTime.parse(start)
        .plusSeconds(frames / rate).plusNanos((frames % rate) * 1_000_000_000L / rate).toString()

    private fun addDurationDiagnostics(metadata: JSONObject, elapsedSeconds: Double?) {
        val audioSeconds = frames(metadata).toDouble() / metadata.getJSONObject("recording_config").getInt("sample_rate")
        metadata.put("audio_duration_seconds", audioSeconds)
            .put("elapsed_minus_audio_seconds", if (elapsedSeconds == null) JSONObject.NULL else elapsedSeconds - audioSeconds)
            .put("duration_diagnostics_note", "经过时长与样本时长之差仅供本机诊断，不代表外部时钟同步或丢帧测量")
    }

    private fun checkCancellation(cancelled: () -> Boolean) {
        if (cancelled()) throw CancellationException("已取消导出")
    }

    private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it.toInt() and 255) }

    private fun exportSafe(value: Any): Any = when (value) {
        is JSONObject -> JSONObject().also { copy ->
            val keys = value.keys()
            while (keys.hasNext()) { val key = keys.next(); copy.put(key, exportSafe(value.get(key))) }
        }
        is JSONArray -> JSONArray().also { copy -> for (index in 0 until value.length()) copy.put(exportSafe(value.get(index))) }
        is String -> value.replace(root.absolutePath, "[local]").replace(root.canonicalPath, "[local]")
        else -> value
    }

    companion object {
        private val RATES = setOf(16000, 44100, 48000)
        private val UUID_PATTERN = Regex("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        private val PART = Regex("chunk_([0-9]{5})\\.wav\\.part")
        private val FINAL = Regex("chunk_[0-9]{5}\\.wav")
        private fun chunkName(index: Int) = "chunk_${index.toString().padStart(5, '0')}.wav"
    }
}
