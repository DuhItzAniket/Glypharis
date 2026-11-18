package org.glypharis.bench

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Element types of a GWT1 file; `code` is the value stored in the file. */
internal enum class ElementType(val code: Int, val bytes: Int) {
    FLOAT32(1, 4),
    INT64(2, 8);

    companion object {
        fun of(code: Int): ElementType =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalArgumentException("unknown element type $code")
    }
}

/** One named tensor; `data` is a direct little-endian buffer holding exactly its elements. */
internal class NamedTensor(
    val name: String,
    val type: ElementType,
    val shape: LongArray,
    val data: ByteBuffer,
)

/**
 * The GWT1 tensor file, little-endian throughout (README.md): magic "GWT1", u32 tensor count, then
 * for each tensor u32 name length, UTF-8 name, u32 element type, u32 rank, rank i64 dimensions,
 * u64 payload length, payload.
 */
internal object TensorFile {
    private val MAGIC = "GWT1".toByteArray(Charsets.US_ASCII)
    private const val MAX_TENSORS = 16
    private const val MAX_NAME = 256
    private const val MAX_RANK = 8

    fun read(file: File): List<NamedTensor> {
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(MAGIC.size).also { buf.get(it) }
        require(magic.contentEquals(MAGIC)) { "${file.name} is not a GWT1 file" }
        val count = buf.int
        require(count in 1..MAX_TENSORS) { "bad tensor count $count" }
        val tensors = List(count) { readTensor(buf) }
        require(!buf.hasRemaining()) { "${file.name} has ${buf.remaining()} trailing bytes" }
        return tensors
    }

    private fun readTensor(buf: ByteBuffer): NamedTensor {
        val nameLength = buf.int
        require(nameLength in 1..MAX_NAME) { "bad name length $nameLength" }
        val name = String(ByteArray(nameLength).also { buf.get(it) }, Charsets.UTF_8)
        val type = ElementType.of(buf.int)
        val rank = buf.int
        require(rank in 0..MAX_RANK) { "bad rank $rank" }
        val shape = LongArray(rank) { buf.long }
        var elements = 1L
        for (dim in shape) {
            require(dim >= 0) { "negative dimension in $name" }
            elements = Math.multiplyExact(elements, dim)
        }
        val length = buf.long
        require(length == Math.multiplyExact(elements, type.bytes.toLong())) {
            "$name: payload of $length bytes does not fit its shape"
        }
        require(length <= buf.remaining()) { "$name: file is cut short" }
        val payload = ByteArray(length.toInt()).also { buf.get(it) }
        val data = ByteBuffer.allocateDirect(payload.size).order(ByteOrder.LITTLE_ENDIAN)
        data.put(payload)
        data.rewind()
        return NamedTensor(name, type, shape, data)
    }

    fun write(file: File, tensors: List<NamedTensor>) {
        val out = ByteArrayOutputStream()
        out.write(MAGIC)
        out.write(int32(tensors.size))
        for (t in tensors) {
            val name = t.name.toByteArray(Charsets.UTF_8)
            out.write(int32(name.size))
            out.write(name)
            out.write(int32(t.type.code))
            out.write(int32(t.shape.size))
            t.shape.forEach { out.write(int64(it)) }
            val payload = ByteArray(t.data.remaining())
            t.data.duplicate().get(payload)
            out.write(int64(payload.size.toLong()))
            out.write(payload)
        }
        BenchFiles.writeAtomic(file, out.toByteArray())
    }

    private fun int32(v: Int): ByteArray =
        ByteBuffer.allocate(Int.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()

    private fun int64(v: Long): ByteArray =
        ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array()
}
