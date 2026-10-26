package org.nightlab.recorder

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import java.util.zip.ZipInputStream
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeNoException
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SessionStoreTest {
    @get:Rule val temporary = TemporaryFolder()
    private fun store() = SessionStore(File(temporary.root, "recordings"))
    private fun create(store: SessionStore, time: String = "2026-10-26T22:00:00+08:00") = store.createSession(
        16000, "UNPROCESSED", JSONObject().put("manufacturer", "Example").put("model", "Phone")
            .put("android_api", 34).put("private_path", temporary.root.absolutePath), time, 123456L)
    private fun append(store: SessionStore, id: String, index: Int = 0, frames: Int = 16000): JSONObject {
        val summary = PcmWavWriter(store.chunkFile(id, index), 16000).use {
            it.append(ShortArray(frames) { index -> (index - 8000).toShort() }, frames)
            it.finish()
        }
        return store.finalizeChunk(id, index, summary, store.totalFrames(id), "completed")
    }
    private inline fun <reified T : Throwable> refuses(block: () -> Unit) {
        try { block() } catch (error: Throwable) {
            if (error is T) return
            throw AssertionError("Expected ${T::class.java.name}, got $error", error)
        }
        fail("Expected ${T::class.java.name}")
    }
    private fun zipFiles(bytes: ByteArray): Map<String, ByteArray> {
        val files = linkedMapOf<String, ByteArray>()
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                assertFalse(entry.name.contains('/') || entry.name.contains('\\'))
                files[entry.name] = zip.readBytes()
            }
        }
        return files
    }
    private fun uint(bytes: ByteArray, offset: Int) = (0..3).fold(0L) { result, index ->
        result or ((bytes[offset + index].toLong() and 255) shl (8 * index))
    }

    @Test fun metadataFinalizationAndExportAreConsistent() {
        val store = store()
        val id = create(store)
        val initial = store.read(id)
        assertEquals(1, initial.getInt("schema_version"))
        assertEquals("nightlab_android", initial.getString("source"))
        assertEquals("recording", initial.getString("status"))
        assertEquals(123456L, initial.getLong("started_elapsed_realtime_nanos"))
        assertEquals("nominal_sample_count", initial.getJSONObject("time_basis").getString("chunk_timestamps"))
        assertFalse(initial.getJSONObject("time_basis").getBoolean("external_synchronization"))
        assertFalse(initial.getJSONObject("device").has("private_path"))
        val first = append(store, id)
        assertEquals("2026-10-26T22:00+08:00", first.getString("start_time"))
        assertEquals("2026-10-26T22:00:01+08:00", first.getString("end_time"))
        assertEquals(32044L, first.getLong("bytes"))
        assertFalse(File(temporary.root, "recordings/$id/chunk_00000.wav.part").exists())
        refuses<IllegalArgumentException> { store.exportSession(id, ByteArrayOutputStream()) }
        append(store, id, 1, 8000)
        assertEquals(24000L, store.totalFrames(id))
        store.updateSession(id, "completed", "user_stop", 1.6)
        assertEquals(1.5, store.read(id).getDouble("audio_duration_seconds"), 0.0)
        assertEquals(0.1, store.read(id).getDouble("elapsed_minus_audio_seconds"), 0.000001)
        val output = ByteArrayOutputStream()
        var progress = 0L
        store.exportSession(id, output, onProgress = { assertTrue(it >= progress); progress = it })
        val files = zipFiles(output.toByteArray())
        assertEquals(setOf("session.json", "chunk_00000.wav", "chunk_00001.wav"), files.keys)
        val metadata = JSONObject(String(files.getValue("session.json"), Charsets.UTF_8))
        assertFalse(String(files.getValue("session.json"), Charsets.UTF_8).contains(temporary.root.absolutePath))
        for (index in 0..1) {
            val chunk = metadata.getJSONArray("chunks").getJSONObject(index)
            val audio = files.getValue(chunk.getString("filename"))
            assertEquals(audio.size - 8L, uint(audio, 4))
            assertEquals(audio.size - 44L, uint(audio, 40))
            assertEquals(16000L, uint(audio, 24))
            val digest = MessageDigest.getInstance("SHA-256").digest(audio)
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            assertEquals(chunk.getString("sha256"), digest)
        }
        assertEquals(48088L, progress)
        refuses<IllegalArgumentException> { store.chunkFile(id, 2) }
        store.deleteSession(id)
        assertTrue(store.listSessions().isEmpty())
    }

    @Test fun cancellationAndTamperingCannotProduceSuccessfulExportsOrChangeStatus() {
        val store = store()
        val id = create(store)
        append(store, id)
        store.updateSession(id, "completed", "user_stop", 1.0)
        val before = store.read(id).toString()
        var cancel = false
        val output = ByteArrayOutputStream()
        refuses<CancellationException> {
            store.exportSession(id, output, cancelled = { cancel }, onProgress = { cancel = true })
        }
        assertEquals(before, store.read(id).toString())
        val bytes = output.toByteArray()
        // No ZIP end-of-central-directory success record after cancellation.
        assertFalse((0 until (bytes.size - 3).coerceAtLeast(0)).any {
            bytes[it] == 0x50.toByte() && bytes[it + 1] == 0x4b.toByte() && bytes[it + 2] == 5.toByte() && bytes[it + 3] == 6.toByte()
        })
        val wav = File(temporary.root, "recordings/$id/chunk_00000.wav")
        RandomAccessFile(wav, "rw").use { it.seek(44); it.write(77) }
        val tamperedOutput = ByteArrayOutputStream()
        refuses<IllegalArgumentException> { store.exportSession(id, tamperedOutput) }
        assertEquals(0, tamperedOutput.size())
        assertEquals(before, store.read(id).toString())
    }

    @Test fun recoveryPreservesActualFramesAndRetainsMalformedAndOrphanFiles() {
        val store = store()
        val interrupted = create(store)
        val part = store.chunkFile(interrupted, 0)
        PcmWavWriter(part, 16000).use {
            it.append(shortArrayOf(1, -2, 3), 3)
        }
        RandomAccessFile(part, "rw").use { it.seek(it.length()); it.write(42) }
        val malformedId = create(store, "2026-10-26T23:00:00+08:00")
        val malformed = store.chunkFile(malformedId, 0).also { it.writeText("invalid") }
        val orphanId = create(store)
        val orphanPart = store.chunkFile(orphanId, 0)
        PcmWavWriter(orphanPart, 16000).use { it.append(shortArrayOf(4), 1); it.finish() }
        val orphan = File(orphanPart.parentFile, "chunk_00000.wav")
        Files.move(orphanPart.toPath(), orphan.toPath())
        val orphanBytes = orphan.readBytes()
        assertEquals(setOf(interrupted, malformedId, orphanId), store.recoverInterruptedSessions().toSet())
        assertEquals(3L, store.totalFrames(interrupted))
        val metadata = store.read(interrupted)
        assertEquals("interrupted", metadata.getString("status"))
        assertEquals("interrupted", metadata.getJSONArray("chunks").getJSONObject(0).getString("status"))
        assertEquals(50L, File(part.parentFile, "chunk_00000.wav").length())
        assertTrue(metadata.isNull("elapsed_seconds"))
        assertFalse(metadata.getBoolean("elapsed_seconds_measured"))
        assertTrue(metadata.isNull("elapsed_minus_audio_seconds"))
        assertEquals(3.0 / 16000, metadata.getDouble("audio_duration_seconds"), 0.0)
        store.exportSession(interrupted, ByteArrayOutputStream())
        assertTrue(malformed.exists())
        assertEquals("invalid", malformed.readText())
        assertTrue(store.read(malformedId).getString("error").contains("恢复失败"))
        assertTrue(orphan.exists())
        assertArrayEquals(orphanBytes, orphan.readBytes())
        assertTrue(store.read(orphanId).isNull("error"))
        assertEquals(1L, store.totalFrames(orphanId))
        assertEquals("interrupted", store.read(orphanId).getJSONArray("chunks").getJSONObject(0).getString("status"))
        val orphanExport = ByteArrayOutputStream()
        store.exportSession(orphanId, orphanExport)
        assertArrayEquals(orphanBytes, zipFiles(orphanExport.toByteArray()).getValue("chunk_00000.wav"))
        assertEquals(3, store.listSessions().size)
        assertEquals(malformedId, store.listSessions().first().getString("session_id"))
    }

    @Test fun protectsActiveSessionAndRejectsEmptyExportAndInvalidFinalization() {
        val store = store()
        val id = create(store)
        refuses<IllegalArgumentException> { store.deleteSession(id) }
        assertTrue(store.recoverInterruptedSessions(id).isEmpty())
        val part = store.chunkFile(id, 0)
        val summary = PcmWavWriter(part, 16000).use { it.append(shortArrayOf(2), 1); it.finish() }
        refuses<IllegalArgumentException> { store.finalizeChunk(id, 0, summary.copy(sha256 = "wrong"), 0, "completed") }
        refuses<IllegalArgumentException> { store.finalizeChunk(id, 0, summary, 1, "completed") }
        assertTrue(part.exists())
        assertEquals(0L, store.totalFrames(id))
        store.finalizeChunk(id, 0, summary, 0, "completed")
        refuses<IllegalArgumentException> { store.finalizeChunk(id, 0, summary, 1, "completed") }
        val empty = create(store)
        store.updateSession(empty, "failed", "audio_init_failed", 0.0, "麦克风初始化失败")
        refuses<IllegalArgumentException> { store.exportSession(empty, ByteArrayOutputStream()) }
        store.deleteSession(empty)
    }

    @Test fun traversalAndUnexpectedDirectoriesAreRejectedBeforeDeletion() {
        val store = store()
        for (id in listOf("../outside", "/tmp/x", "..", "NOT-A-UUID")) {
            refuses<IllegalArgumentException> { store.read(id) }
            refuses<IllegalArgumentException> { store.deleteSession(id) }
        }
        val id = create(store)
        append(store, id)
        store.updateSession(id, "completed", "user_stop", 1.0)
        val directory = File(temporary.root, "recordings/$id")
        File(directory, "unexpected").mkdir()
        refuses<IllegalArgumentException> { store.deleteSession(id) }
        assertTrue(File(directory, "session.json").exists())
        assertTrue(File(directory, "chunk_00000.wav").exists())
    }

    @Test fun symlinkPreflightLeavesOutsideAndSessionFilesUntouched() {
        val store = store()
        val id = create(store)
        append(store, id)
        store.updateSession(id, "completed", "user_stop", 1.0)
        val outside = temporary.newFile("outside.txt").also { it.writeText("outside") }
        val link = File(temporary.root, "recordings/$id/link.txt")
        try { Files.createSymbolicLink(link.toPath(), outside.toPath()) }
        catch (failure: Exception) { assumeNoException("平台不允许创建符号链接", failure); return }
        try {
            refuses<IllegalArgumentException> { store.deleteSession(id) }
            refuses<IllegalArgumentException> { store.exportSession(id, ByteArrayOutputStream()) }
            assertEquals("outside", outside.readText())
            assertTrue(File(link.parentFile, "session.json").exists())
            assertTrue(File(link.parentFile, "chunk_00000.wav").exists())
        } finally { Files.deleteIfExists(link.toPath()) }
    }

    @Test fun trustedParentAliasIsCanonicalizedButDirectRootLinkIsRejected() {
        val actualParent = temporary.newFolder("actual-parent")
        val alias = File(temporary.root, "parent-alias")
        try { Files.createSymbolicLink(alias.toPath(), actualParent.toPath()) }
        catch (failure: Exception) { assumeNoException("平台不允许创建符号链接", failure); return }
        try {
            val store = SessionStore(File(alias, "recordings"))
            val id = create(store)
            assertTrue(File(actualParent, "recordings/$id/session.json").isFile)
            val part = store.chunkFile(id, 0)
            assertEquals(File(actualParent, "recordings/$id").canonicalFile, part.parentFile)
            assertEquals(id, store.read(id).getString("session_id"))
            refuses<IllegalArgumentException> { SessionStore(alias) }
        } finally { Files.deleteIfExists(alias.toPath()) }
    }

    @Test fun stoppedValidPartCanBeRecoveredWithoutFabricatingElapsedTimeOrOverwritingCollision() {
        val store = store()
        val id = create(store)
        val part = store.chunkFile(id, 0)
        PcmWavWriter(part, 16000).use { it.append(shortArrayOf(1, 2, 3, 4), 4) }
        store.updateSession(id, "interrupted", "finalize_io_error", 0.75, "写入失败，已保留临时音频")
        assertEquals(listOf(id), store.recoverInterruptedSessions())
        val metadata = store.read(id)
        assertEquals("interrupted", metadata.getString("status"))
        assertEquals(4L, store.totalFrames(id))
        assertTrue(metadata.isNull("error"))
        assertEquals(0.75, metadata.getDouble("elapsed_seconds"), 0.0)
        assertEquals(0.75 - 4.0 / 16000, metadata.getDouble("elapsed_minus_audio_seconds"), 0.0)
        store.exportSession(id, ByteArrayOutputStream())

        val collisionId = create(store)
        val collisionPart = store.chunkFile(collisionId, 0)
        PcmWavWriter(collisionPart, 16000).use { it.append(shortArrayOf(9), 1) }
        val before = collisionPart.readBytes()
        val orphan = File(collisionPart.parentFile, "chunk_00000.wav").also { it.writeText("preserve orphan") }
        store.updateSession(collisionId, "failed", "io_error", 0.25)
        store.recoverInterruptedSessions()
        assertArrayEquals(before, collisionPart.readBytes())
        assertEquals("preserve orphan", orphan.readText())
        assertTrue(store.read(collisionId).getString("error").contains("冲突"))
        refuses<IllegalArgumentException> { store.exportSession(collisionId, ByteArrayOutputStream()) }
    }
    @Test fun renamedOrphanAndLaterPartialRecoverInOrderWithoutChangingFinalAudio() {
        val store = store()
        val id = create(store)
        append(store, id, frames = 3)
        val orphanPart = store.chunkFile(id, 1)
        val summary = PcmWavWriter(orphanPart, 16000).use { it.append(shortArrayOf(-9, 77), 2); it.finish() }
        val orphan = File(orphanPart.parentFile, "chunk_00001.wav")
        Files.move(orphanPart.toPath(), orphan.toPath()) // Kill after rename, before metadata save.
        val orphanBytes = orphan.readBytes()
        val laterPart = store.chunkFile(id, 2)
        PcmWavWriter(laterPart, 16000).use { it.append(shortArrayOf(8), 1) }
        assertEquals(listOf(id), store.recoverInterruptedSessions())
        assertArrayEquals(orphanBytes, orphan.readBytes())
        val metadata = store.read(id)
        assertEquals("interrupted", metadata.getString("status"))
        assertEquals(6L, store.totalFrames(id))
        val recoveredChunk = metadata.getJSONArray("chunks").getJSONObject(1)
        assertEquals(summary.sha256, recoveredChunk.getString("sha256"))
        assertEquals(3L, recoveredChunk.getLong("start_frame"))
        assertEquals(2L, recoveredChunk.getLong("frames"))
        assertEquals("interrupted", recoveredChunk.getString("status"))
        assertFalse(metadata.getJSONObject("time_basis").getBoolean("external_synchronization"))
        val output = ByteArrayOutputStream()
        store.exportSession(id, output)
        val entries = zipFiles(output.toByteArray())
        assertEquals(4, entries.size)
        assertArrayEquals(orphanBytes, entries.getValue("chunk_00001.wav"))
        assertTrue(store.recoverInterruptedSessions().isEmpty())
    }

    @Test fun malformedOutOfOrderUnknownAndDuplicateMetadataOrphansRemainUnchanged() {
        val store = store()
        val malformedId = create(store)
        val malformedPart = store.chunkFile(malformedId, 0)
        PcmWavWriter(malformedPart, 16000).use { it.append(shortArrayOf(3), 1); it.finish() }
        val malformed = File(malformedPart.parentFile, "chunk_00000.wav")
        Files.move(malformedPart.toPath(), malformed.toPath())
        RandomAccessFile(malformed, "rw").use { it.seek(20); it.write(3) }

        val gapId = create(store)
        val gapPart = store.chunkFile(gapId, 1)
        PcmWavWriter(gapPart, 16000).use { it.append(shortArrayOf(4), 1); it.finish() }
        val gap = File(gapPart.parentFile, "chunk_00001.wav")
        Files.move(gapPart.toPath(), gap.toPath())

        val unknownId = create(store)
        val unknownPart = store.chunkFile(unknownId, 0)
        PcmWavWriter(unknownPart, 16000).use { it.append(shortArrayOf(5), 1); it.finish() }
        val unknown = File(unknownPart.parentFile, "unexpected.wav")
        Files.move(unknownPart.toPath(), unknown.toPath())

        val duplicateId = create(store)
        append(store, duplicateId, frames = 1)
        val duplicatePart = store.chunkFile(duplicateId, 1)
        PcmWavWriter(duplicatePart, 16000).use { it.append(shortArrayOf(6), 1); it.finish() }
        val duplicate = File(duplicatePart.parentFile, "chunk_00001.wav")
        Files.move(duplicatePart.toPath(), duplicate.toPath())
        val badMetadata = store.read(duplicateId)
        badMetadata.getJSONArray("chunks").put(JSONObject(badMetadata.getJSONArray("chunks").getJSONObject(0).toString()))
        File(duplicate.parentFile, "session.json").writeText(badMetadata.toString())

        val files = listOf(malformed, gap, unknown, duplicate)
        val before = files.associateWith { it.readBytes() }
        store.recoverInterruptedSessions()
        for (file in files) assertArrayEquals(before.getValue(file), file.readBytes())
        for (id in listOf(malformedId, gapId, unknownId, duplicateId)) {
            assertTrue(store.read(id).getString("error").contains("needs_manual_recovery"))
            refuses<IllegalArgumentException> { store.exportSession(id, ByteArrayOutputStream()) }
        }
        assertEquals(0L, store.totalFrames(malformedId))
        assertEquals(0L, store.totalFrames(gapId))
        assertEquals(0L, store.totalFrames(unknownId))
    }

    @Test fun exportCrossLanguageFixtureWhenRequested() {
        val destination = System.getProperty("nightlab.session.fixture") ?: return
        val store = store()
        val id = create(store, "2026-10-25T23:59:59+08:00")
        append(store, id, 0, 192000)
        append(store, id, 1, 192000)
        store.updateSession(id, "completed", "user_stop", 24.05)
        Files.newOutputStream(File(destination).toPath(), java.nio.file.StandardOpenOption.CREATE_NEW).use {
            store.exportSession(id, it)
        }
        assertEquals(384000L, store.totalFrames(id))
    }

}
