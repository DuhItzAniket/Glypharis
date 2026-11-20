package org.glypharis.bench

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONArray
import org.json.JSONObject

/** A call's input file could not be read: the request is at fault, not the model. */
internal class UnreadableInput(cause: Exception) : Exception(cause.message, cause)

/** A check output could not be written (storage, a rename): the phone is at fault, not the model. */
internal class UnwritableOutput(cause: Exception) : Exception(cause.message, cause)

/**
 * What the modes share: the session options, input tensors over GWT1 buffers, a float output as
 * bytes, error records, and the rule of one timing or check pass per process.
 */
internal object BenchSession {
    private val passStarted = AtomicBoolean(false)

    /**
     * The options of `glypharis.efficiency.threads.ort_session`, for every session of every mode:
     * sequential execution, all graph optimisations, CPU memory arena and memory patterns off,
     * `threads` intra-op threads. Spinning stays at the default.
     */
    fun options(threads: Int): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setCPUArenaAllocator(false)
            setMemoryPatternOptimization(false)
            setIntraOpNumThreads(threads)
        }

    /** What `options` sets, in the names `threads.session_options` reports. */
    fun optionsRecord(threads: Int): JSONObject =
        JSONObject()
            .put("providers", JSONArray(listOf("CPUExecutionProvider")))
            .put("intra_op_num_threads", threads)
            .put("execution_mode", "ORT_SEQUENTIAL")
            .put("graph_optimization_level", "ORT_ENABLE_ALL")
            .put("enable_cpu_mem_arena", false)
            .put("enable_mem_pattern", false)
            .put("allow_spinning", "default")

    /** An ONNX Runtime tensor over `t`'s direct buffer, which ONNX Runtime uses without a copy. */
    fun tensor(env: OrtEnvironment, t: NamedTensor): OnnxTensor =
        when (t.type) {
            ElementType.FLOAT32 -> OnnxTensor.createTensor(env, t.data.asFloatBuffer(), t.shape)
            ElementType.INT64 -> OnnxTensor.createTensor(env, t.data.asLongBuffer(), t.shape)
            ElementType.UINT8 -> OnnxTensor.createTensor(env, t.data, t.shape, OnnxJavaType.UINT8)
        }

    /** The inputs of one call, from its GWT1 file; any failure is an `UnreadableInput`. */
    fun readInputs(file: File): List<NamedTensor> {
        try {
            val tensors = TensorFile.read(file)
            require(tensors.map { it.name }.toSet().size == tensors.size) {
                "${file.name} holds two tensors of one name"
            }
            return tensors
        } catch (e: Exception) {
            throw UnreadableInput(e)
        }
    }

    /** The inputs of one call read into `arena` (`TensorFile.readInto`); failures as `readInputs`. */
    fun readInputsInto(file: File, arena: ByteBuffer): List<NamedTensor> {
        try {
            val tensors = TensorFile.readInto(file, arena)
            require(tensors.map { it.name }.toSet().size == tensors.size) {
                "${file.name} holds two tensors of one name"
            }
            return tensors
        } catch (e: Exception) {
            throw UnreadableInput(e)
        }
    }

    /**
     * The one input buffer of a timing run, `bytes` long, every page of it written once so that
     * it is resident from the start: it adds the same memory to every arm's VmHWM.
     */
    fun arena(bytes: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < bytes) {
            buffer.put(i, 0)
            i += PAGE_BYTES
        }
        return buffer
    }

    private const val PAGE_BYTES = 4096

    /** A float32 output's values as little-endian bytes, and whether every value is finite. */
    fun floatBytes(value: OnnxTensor): Pair<ByteBuffer, Boolean> {
        val values = value.floatBuffer
        val bytes = ByteBuffer.allocateDirect(values.remaining() * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        var finite = true
        while (values.hasRemaining()) {
            val v = values.get()
            if (v.isNaN() || v.isInfinite()) finite = false
            bytes.putFloat(v)
        }
        bytes.flip()
        return bytes to finite
    }

    /** The error as a record. ONNX Runtime's messages name the model's path; it is cut out. */
    fun describe(e: Throwable, folder: File): JSONObject {
        val out = JSONObject()
            .put("type", e.javaClass.name)
            .put("message", e.message?.replace(folder.path, "<app files>") ?: JSONObject.NULL)
        if (e is OrtException) out.put("code", e.code.name)
        return out
    }

    /**
     * Refuses a second timing or check pass in this process. Each pass needs a new process:
     * `VmHWM` never falls within one, and the check must not share a process with a timing run.
     */
    fun claimProcess() {
        require(passStarted.compareAndSet(false, true)) {
            "this process has already started a timing or check pass: force-stop the app first"
        }
    }
}
