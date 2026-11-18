package org.glypharis.bench

import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/** File access for the bench. Every name stays inside the app's external files folder. */
internal object BenchFiles {
    private val SIMPLE_NAME = Regex("[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")

    /** `name` inside `folder`; a name with a path separator or a leading dot is refused. */
    fun resolve(folder: File, name: String): File {
        require(SIMPLE_NAME.matches(name)) { "not a plain file name: $name" }
        return File(folder, name)
    }

    /** Writes to a temporary file and renames it, so a reader never sees a partial file. */
    fun writeAtomic(file: File, bytes: ByteArray) {
        val tmp = File(file.parentFile, "${file.name}.tmp")
        FileOutputStream(tmp).use { out ->
            out.write(bytes)
            out.fd.sync()
        }
        check(tmp.renameTo(file)) { "cannot move ${tmp.name} to ${file.name}" }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val block = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(block)
                if (n < 0) break
                digest.update(block, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
