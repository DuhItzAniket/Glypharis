package org.glypharis.bench

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** Element types of a GWT1 file; `code` is the value stored in the file. */
internal enum class ElementType(val code: Int, val bytes: Int) {
    FLOAT32(1, 4),
    INT64(2, 8),
    UINT8(3, 1);

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
    const val ARENA_ALIGN = 8L // as `phase22_android.ARENA_ALIGN`, which sizes `buffer_bytes`

    /**
     * The tensors of `file`, read through its channel: the header fields one by one, then each
     * payload straight into its own direct buffer, so no copy of a payload is made on the heap.
     */
    fun read(file: File): List<NamedTensor> =
        FileInputStream(file).use { stream ->
            val input = ChannelInput(stream.channel, file.name)
            val magic = input.bytes(MAGIC.size)
            require(magic.contentEquals(MAGIC)) { "${file.name} is not a GWT1 file" }
            val count = input.int()
            require(count in 1..MAX_TENSORS) { "bad tensor count $count" }
            val tensors = List(count) { readTensor(input) }
            val trailing = input.remaining()
            require(trailing == 0L) { "${file.name} has $trailing trailing bytes" }
            tensors
        }

    /**
     * The tensors of `file`, each payload read into `arena` (a direct little-endian buffer) at the
     * next `ARENA_ALIGN`-byte boundary, from its start: the returned tensors are views of `arena`,
     * valid until the next call. A file whose payloads do not fit is refused.
     */
    fun readInto(file: File, arena: ByteBuffer): List<NamedTensor> =
        FileInputStream(file).use { stream ->
            val input = ChannelInput(stream.channel, file.name)
            val magic = input.bytes(MAGIC.size)
            require(magic.contentEquals(MAGIC)) { "${file.name} is not a GWT1 file" }
            val count = input.int()
            require(count in 1..MAX_TENSORS) { "bad tensor count $count" }
            var offset = 0L
            val tensors = List(count) {
                readTensor(input) { length ->
                    require(offset + length <= arena.capacity()) {
                        "${file.name} needs more than the plan's ${arena.capacity()} buffer bytes"
                    }
                    val view = arena.duplicate()
                    view.limit((offset + length).toInt()).position(offset.toInt())
                    offset = (offset + length + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN
                    view.slice().order(ByteOrder.LITTLE_ENDIAN)
                }
            }
            val trailing = input.remaining()
            require(trailing == 0L) { "${file.name} has $trailing trailing bytes" }
            tensors
        }

    private fun readTensor(
        input: ChannelInput,
        buffer: (Long) -> ByteBuffer = { length ->
            ByteBuffer.allocateDirect(length.toInt()).order(ByteOrder.LITTLE_ENDIAN)
        },
    ): NamedTensor {
        val nameLength = input.int()
        require(nameLength in 1..MAX_NAME) { "bad name length $nameLength" }
        val name = String(input.bytes(nameLength), Charsets.UTF_8)
        val type = ElementType.of(input.int())
        val rank = input.int()
        require(rank in 0..MAX_RANK) { "bad rank $rank" }
        val shape = LongArray(rank) { input.long() }
        var elements = 1L
        for (dim in shape) {
            require(dim >= 0) { "negative dimension in $name" }
            elements = Math.multiplyExact(elements, dim)
        }
        val length = input.long()
        require(length == Math.multiplyExact(elements, type.bytes.toLong())) {
            "$name: payload of $length bytes does not fit its shape"
        }
        require(length <= input.remaining()) { "$name: file is cut short" }
        require(length <= Int.MAX_VALUE) { "$name: payload of $length bytes is too large" }
        val data = buffer(length)
        input.fill(data)
        data.rewind()
        return NamedTensor(name, type, shape, data)
    }

    /**
     * `tensors` as a GWT1 file, written through a temporary file (`BenchFiles.writeAtomic`). Each
     * payload goes from its own buffer to the file's channel, with no copy of it on the heap.
     */
    fun write(file: File, tensors: List<NamedTensor>) {
        BenchFiles.writeAtomic(file) { channel ->
            val start = ByteBuffer.allocate(MAGIC.size + Int.SIZE_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN)
            start.put(MAGIC).putInt(tensors.size)
            writeFrom(channel, start)
            for (t in tensors) {
                val name = t.name.toByteArray(Charsets.UTF_8)
                val size = 3 * Int.SIZE_BYTES + name.size + (t.shape.size + 1) * Long.SIZE_BYTES
                val header = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
                header.putInt(name.size).put(name).putInt(t.type.code).putInt(t.shape.size)
                t.shape.forEach { header.putLong(it) }
                header.putLong(t.data.remaining().toLong())
                writeFrom(channel, header)
                val payload = t.data.duplicate()
                while (payload.hasRemaining()) channel.write(payload)
            }
        }
    }

    /** Writes what was put into `buffer`, from its start to its position. */
    private fun writeFrom(channel: FileChannel, buffer: ByteBuffer) {
        buffer.flip()
        while (buffer.hasRemaining()) channel.write(buffer)
    }
}

/** Little-endian fields read in order from `channel`; a file that ends too early is refused. */
private class ChannelInput(private val channel: FileChannel, private val name: String) {
    private val size = channel.size()
    private val field = ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)

    fun remaining(): Long = size - channel.position()

    fun int(): Int = field(Int.SIZE_BYTES).int

    fun long(): Long = field(Long.SIZE_BYTES).long

    fun bytes(n: Int): ByteArray {
        val out = ByteBuffer.allocate(n)
        fill(out)
        return out.array()
    }

    /** Reads from the channel until `buffer` is full. */
    fun fill(buffer: ByteBuffer) {
        require(buffer.remaining() <= remaining()) { "$name is cut short" }
        while (buffer.hasRemaining()) {
            require(channel.read(buffer) >= 0) { "$name is cut short" }
        }
    }

    private fun field(n: Int): ByteBuffer {
        field.clear()
        field.limit(n)
        fill(field)
        field.flip()
        return field
    }
}
