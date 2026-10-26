package org.nightlab.recorder

import java.io.File
import java.io.RandomAccessFile
import java.nio.file.FileAlreadyExistsException
import java.nio.file.Files
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class PcmWavWriterTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun target(name: String = "audio.wav.part") = File(temporary.root, name)
    private fun uint(bytes: ByteArray, offset: Int): Long = (0..3).fold(0L) { value, index ->
        value or ((bytes[offset + index].toLong() and 255) shl (8 * index))
    }
    private fun ushort(bytes: ByteArray, offset: Int) =
        (bytes[offset].toInt() and 255) or ((bytes[offset + 1].toInt() and 255) shl 8)
    private fun assertHeader(bytes: ByteArray, rate: Int, dataBytes: Long) {
        assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
        assertEquals("WAVE", String(bytes, 8, 4, Charsets.US_ASCII))
        assertEquals("fmt ", String(bytes, 12, 4, Charsets.US_ASCII))
        assertEquals("data", String(bytes, 36, 4, Charsets.US_ASCII))
        assertEquals(36 + dataBytes, uint(bytes, 4))
        assertEquals(16L, uint(bytes, 16))
        assertEquals(1, ushort(bytes, 20))
        assertEquals(1, ushort(bytes, 22))
        assertEquals(rate.toLong(), uint(bytes, 24))
        assertEquals(rate.toLong() * 2, uint(bytes, 28))
        assertEquals(2, ushort(bytes, 32))
        assertEquals(16, ushort(bytes, 34))
        assertEquals(dataBytes, uint(bytes, 40))
    }
    private inline fun <reified T : Throwable> refuses(block: () -> Unit) {
        try { block() } catch (error: Throwable) {
            if (error is T) return
            throw AssertionError("Expected ${T::class.java.name}, got $error", error)
        }
        fail("Expected ${T::class.java.name}")
    }

    @Test fun signedSamplesHaveCanonicalLittleEndianEncodingAndHash() {
        val file = target()
        val writer = PcmWavWriter(file, 16000)
        writer.append(shortArrayOf(Short.MIN_VALUE, -1, 0, 1, Short.MAX_VALUE), 5)
        val result = writer.finish()
        val bytes = file.readBytes()
        assertHeader(bytes, 16000, 10)
        assertArrayEquals(byteArrayOf(0, -128, -1, -1, 0, 0, 1, 0, -1, 127), bytes.copyOfRange(44, 54))
        assertEquals(5L, result.frames)
        assertEquals(10L, result.dataBytes)
        assertEquals(5.0 / 16000, result.durationSeconds, 0.0)
        val independentHash = MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        assertEquals(independentHash, result.sha256)
        refuses<IllegalStateException> { writer.append(shortArrayOf(1), 1) }
        refuses<IllegalStateException> { writer.checkpoint() }
        writer.close()
    }

    @Test fun streamingAndCheckpointTrackOnlyAppendedFramesAtEverySupportedRate() {
        for (rate in listOf(16000, 44100, 48000)) {
            val file = target("$rate.wav.part")
            PcmWavWriter(file, rate).use { writer ->
                val samples = ShortArray(20003) { (it * 17).toShort() }
                writer.append(samples, samples.size)
                assertEquals(20003L, writer.framesWritten)
                assertEquals(40050L, file.length())
                assertHeader(file.readBytes(), rate, 0)
                writer.checkpoint()
                assertHeader(file.readBytes(), rate, 40006)
                writer.append(shortArrayOf(-77, 8, 999), 2)
                writer.append(shortArrayOf(), 0)
                val result = writer.finish()
                assertEquals(20005L, result.frames)
                val bytes = file.readBytes()
                assertHeader(bytes, rate, 40010)
                for (index in samples.indices) assertEquals(samples[index].toInt() and 65535, ushort(bytes, 44 + index * 2))
                assertEquals((-77).toShort().toInt() and 65535, ushort(bytes, 40050))
                assertEquals(8, ushort(bytes, 40052))
            }
        }
    }

    @Test fun recoveryUsesActualPayloadAndRemovesOnlyOneTrailingByte() {
        val file = target()
        PcmWavWriter(file, 44100).use { writer ->
            writer.append(shortArrayOf(7), 1)
            writer.checkpoint()
            writer.append(shortArrayOf(-9, 32), 2)
        }
        RandomAccessFile(file, "rw").use { it.seek(it.length()); it.write(0x7f) }
        assertEquals(51L, file.length())
        assertHeader(file.readBytes(), 44100, 2)
        val recovered = PcmWavWriter.recover(file)
        assertEquals(3L, recovered.frames)
        assertEquals(6L, recovered.dataBytes)
        assertEquals(3.0 / 44100, recovered.durationSeconds, 0.0)
        assertEquals(50L, file.length())
        assertHeader(file.readBytes(), 44100, 6)
        assertTrue(file.name.endsWith(".wav.part"))
        assertFalse(target("audio.wav").exists())
        assertEquals(recovered, PcmWavWriter.recover(file))
    }

    @Test fun refusesOverwriteWrongSuffixRateAndInvalidCountsWithoutChanges() {
        val existing = target()
        existing.writeText("existing bytes")
        refuses<FileAlreadyExistsException> { PcmWavWriter(existing, 16000) }
        assertEquals("existing bytes", existing.readText())
        refuses<IllegalArgumentException> { PcmWavWriter(target("audio.wav"), 16000) }
        refuses<IllegalArgumentException> { PcmWavWriter(target("rate.wav.part"), 8000) }
        assertFalse(target("rate.wav.part").exists())
        PcmWavWriter(target("counts.wav.part"), 48000).use { writer ->
            refuses<IllegalArgumentException> { writer.append(shortArrayOf(1), -1) }
            refuses<IllegalArgumentException> { writer.append(shortArrayOf(1), 2) }
            assertEquals(0L, writer.framesWritten)
            assertEquals(44L, writer.file.length())
        }
    }

    @Test fun malformedHeadersAreRejectedWithoutMutation() {
        val original = target("original.wav.part")
        PcmWavWriter(original, 16000).use { it.append(shortArrayOf(10, 20), 2); it.finish() }
        // Foreign tags, codec, channels, rate, byte rate, alignment, depth, and lengths.
        val changes = listOf(0 to 0, 8 to 0, 12 to 0, 16 to 18, 20 to 3,
            22 to 2, 24 to 0, 28 to 1, 32 to 4, 34 to 24, 36 to 0, 40 to 3,
            40 to 8, 4 to 0)
        changes.forEachIndexed { index, (offset, value) ->
            val file = target("bad$index.wav.part")
            val bytes = original.readBytes().also { it[offset] = value.toByte() }
            file.writeBytes(bytes)
            refuses<IllegalArgumentException> { PcmWavWriter.recover(file) }
            assertArrayEquals(bytes, file.readBytes())
        }
        val short = target("short.wav.part").also { it.writeBytes(ByteArray(43)) }
        refuses<IllegalArgumentException> { PcmWavWriter.recover(short) }
        refuses<IllegalArgumentException> { PcmWavWriter.recover(target("final.wav")) }
    }

    @Test fun recoveryRejectsOversizedChunksAndRiffOverflowBeforeModification() {
        val file = target()
        PcmWavWriter(file, 16000).close()
        val header = file.readBytes()
        RandomAccessFile(file, "rw").use { it.setLength(44L + 16000L * 3600 * 2 + 2) }
        val tooLong = file.length()
        refuses<IllegalArgumentException> { PcmWavWriter.recover(file) }
        assertEquals(tooLong, file.length())
        val overflow = target("overflow.wav.part")
        // Explicit sparse creation avoids allocating four GiB for this boundary test.
        FileChannel.open(overflow.toPath(), StandardOpenOption.CREATE_NEW,
            StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.SPARSE).use {
            it.write(ByteBuffer.wrap(header), 0)
            it.write(ByteBuffer.wrap(byteArrayOf(0)), 0x100000000L + 7)
        }
        refuses<IllegalArgumentException> { PcmWavWriter.recover(overflow) }
        assertEquals(0x100000000L + 8, overflow.length())
        RandomAccessFile(overflow, "rw").use { it.setLength(44) }
        RandomAccessFile(file, "rw").use { it.setLength(44) }
    }

    /** Opt-in cross-language fixture, consumed by NightLab's Python WAV reader. */
    @Test fun exportFixtureWhenRequested() {
        val path = System.getProperty("nightlab.wav.fixture") ?: return
        val file = File(path)
        val samples = ShortArray(16000) { index ->
            (kotlin.math.sin(2.0 * Math.PI * 440.0 * index / 16000) * 8192).toInt().toShort()
        }
        val summary = PcmWavWriter(file, 16000).use { writer ->
            writer.append(samples, samples.size)
            writer.finish()
        }
        assertEquals(16000L, summary.frames)
        assertEquals(1.0, summary.durationSeconds, 0.0)
        assertHeader(Files.readAllBytes(file.toPath()), 16000, 32000)
    }
}
