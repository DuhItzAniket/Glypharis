package org.glypharis.bench

import java.io.File
import java.io.FileOutputStream
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** File access for the bench. Every name stays inside the app's external files folder. */
internal object BenchFiles {
    private val SIMPLE_NAME = Regex("[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")

    /** `name` inside `folder`; a name with a path separator or a leading dot is refused. */
    fun resolve(folder: File, name: String): File {
        require(SIMPLE_NAME.matches(name)) { "not a plain file name: $name" }
        return File(folder, name)
    }

    /** `name` inside `folder` as a file that exists, can be read and is not empty. */
    fun readable(folder: File, name: String): File {
        val file = resolve(folder, name)
        require(file.isFile && file.canRead() && file.length() > 0) {
            "$name is missing or cannot be read"
        }
        return file
    }

    /** Writes to a temporary file and renames it, so a reader never sees a partial file. */
    fun writeAtomic(file: File, bytes: ByteArray) = replace(file) { it.write(bytes) }

    /** As `writeAtomic(file, bytes)`, with `write` putting the contents into the file's channel. */
    fun writeAtomic(file: File, write: (FileChannel) -> Unit) = replace(file) { write(it.channel) }

    private inline fun replace(file: File, write: (FileOutputStream) -> Unit) {
        val tmp = File(file.parentFile, "${file.name}.tmp")
        FileOutputStream(tmp).use { out ->
            write(out)
            out.fd.sync()
        }
        check(tmp.renameTo(file)) { "cannot move ${tmp.name} to ${file.name}" }
    }

    fun sha256(file: File): String = sha256(file, ByteArray(1 shl 20))

    /** As `sha256(file)`, reading through `block`, which one caller can reuse for many files. */
    fun sha256(file: File, block: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            while (true) {
                val n = input.read(block)
                if (n < 0) break
                digest.update(block, 0, n)
            }
        }
        return hex(digest.digest())
    }

    fun sha256(bytes: ByteArray): String = hex(MessageDigest.getInstance("SHA-256").digest(bytes))

    private fun hex(digest: ByteArray): String = digest.joinToString("") { "%02x".format(it) }
}
