package org.nightlab.recorder

import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

data class WavSummary(
    val frames: Long,
    val dataBytes: Long,
    val sha256: String,
    val durationSeconds: Double,
)

/** Streaming PCM16 mono WAV. A .part filename remains provisional even after recovery. */
class PcmWavWriter(val file: File, val sampleRate: Int) : Closeable {
    private val channel: FileChannel
    var framesWritten: Long = 0
        private set
    private var closed = false
    private val conversion = ByteBuffer.allocate(16384).order(ByteOrder.LITTLE_ENDIAN)

    init {
        requirePart(file)
        require(sampleRate in SUPPORTED_RATES) { "Unsupported sample rate: $sampleRate" }
        channel = FileChannel.open(file.toPath(), StandardOpenOption.CREATE_NEW,
            StandardOpenOption.READ, StandardOpenOption.WRITE)
        try {
            writeHeader(channel, sampleRate, 0)
            channel.force(true)
        } catch (failure: Throwable) {
            channel.close()
            throw failure
        }
    }

    @Synchronized
    fun append(samples: ShortArray, count: Int) {
        checkOpen()
        require(count in 0..samples.size) { "count must be within the sample array" }
        val nextFrames = framesWritten + count.toLong()
        require(nextFrames <= sampleRate.toLong() * MAX_SECONDS) { "WAV chunk exceeds one hour" }
        require(nextFrames * 2 <= MAX_DATA_BYTES) { "WAV exceeds RIFF size limit" }
        // Bounded conversion buffer, independent of the size of the caller's array.
        val buffer = conversion
        var index = 0
        while (index < count) {
            buffer.clear()
            val start = index
            val end = minOf(count, index + buffer.capacity() / 2)
            while (index < end) buffer.putShort(samples[index++])
            buffer.flip()
            writeFully(channel, buffer, HEADER_SIZE + framesWritten * 2)
            framesWritten += (end - start).toLong()
        }
    }

    /** Make the current prefix self-describing and durable, retaining the provisional name. */
    @Synchronized
    fun checkpoint() {
        checkOpen()
        writeHeader(channel, sampleRate, framesWritten * 2)
        channel.force(true)
    }

    /** Seal and close this handle. The caller owns metadata persistence and the final rename. */
    @Synchronized
    fun finish(): WavSummary {
        checkOpen()
        checkpoint()
        return try {
            summary(channel, sampleRate, framesWritten * 2)
        } finally {
            close()
        }
    }

    /** Closing alone does not update sizes or assert recording completion. */
    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            channel.close()
        }
    }

    private fun checkOpen() = check(!closed) { "WAV writer is closed" }

    companion object {
        private const val HEADER_SIZE = 44L
        private const val MAX_SECONDS = 3600L
        private const val MAX_DATA_BYTES = 0xffffffffL - 36L
        private val SUPPORTED_RATES = setOf(16000, 44100, 48000)

        /** Repair only a canonical NightLab-compatible PCM prefix; never rename or mark complete. */
        fun recover(file: File): WavSummary {
            requirePart(file)
            FileChannel.open(file.toPath(), StandardOpenOption.READ, StandardOpenOption.WRITE).use { channel ->
                val length = channel.size()
                require(length >= HEADER_SIZE && length - 8 <= 0xffffffffL) { "Invalid RIFF file length" }
                val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                while (header.hasRemaining()) {
                    check(channel.read(header, header.position().toLong()) > 0) { "Incomplete WAV header" }
                }
                header.flip()
                fun tag(offset: Int, value: String) = value.indices.all { header.get(offset + it) == value[it].code.toByte() }
                fun uint(offset: Int) = header.getInt(offset).toLong() and 0xffffffffL
                fun ushort(offset: Int) = header.getShort(offset).toInt() and 0xffff
                require(tag(0, "RIFF") && tag(8, "WAVE") && tag(12, "fmt ") && tag(36, "data")) { "Not a canonical WAV header" }
                val rate = uint(24)
                require(uint(16) == 16L && ushort(20) == 1 && ushort(22) == 1 && ushort(32) == 2 && ushort(34) == 16) { "Unsupported WAV format" }
                require(rate in SUPPORTED_RATES.map { it.toLong() } && uint(28) == rate * 2) { "Invalid WAV sample rate or byte rate" }
                val declared = uint(40)
                val actual = length - HEADER_SIZE
                require(declared % 2 == 0L && declared <= actual && uint(4) == 36 + declared) { "Inconsistent WAV lengths" }
                val dataBytes = actual - actual % 2
                require(dataBytes <= MAX_DATA_BYTES && dataBytes / 2 <= rate * MAX_SECONDS) { "WAV exceeds chunk size limit" }
                if (actual != dataBytes) channel.truncate(HEADER_SIZE + dataBytes)
                writeHeader(channel, rate.toInt(), dataBytes)
                channel.force(true)
                return summary(channel, rate.toInt(), dataBytes)
            }
        }

        private fun requirePart(file: File) {
            require(file.name.endsWith(".wav.part")) { "Expected a .wav.part filename" }
        }

        private fun writeHeader(channel: FileChannel, sampleRate: Int, bytes: Long) {
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt((36 + bytes).toInt())
            header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)).putInt(16)
            header.putShort(1).putShort(1).putInt(sampleRate).putInt(sampleRate * 2)
            header.putShort(2).putShort(16)
            header.put("data".toByteArray(Charsets.US_ASCII)).putInt(bytes.toInt()).flip()
            writeFully(channel, header, 0)
        }

        private fun writeFully(channel: FileChannel, buffer: ByteBuffer, position: Long) {
            var offset = position
            while (buffer.hasRemaining()) {
                val written = channel.write(buffer, offset)
                check(written > 0) { "Unable to write WAV" }
                offset += written
            }
        }

        private fun summary(channel: FileChannel, rate: Int, bytes: Long): WavSummary {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteBuffer.allocate(65536)
            var offset = 0L
            while (offset < channel.size()) {
                buffer.clear()
                val read = channel.read(buffer, offset)
                check(read > 0) { "Unable to hash WAV" }
                digest.update(buffer.array(), 0, read)
                offset += read
            }
            val hash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            return WavSummary(bytes / 2, bytes, hash, (bytes / 2).toDouble() / rate)
        }
    }
}
