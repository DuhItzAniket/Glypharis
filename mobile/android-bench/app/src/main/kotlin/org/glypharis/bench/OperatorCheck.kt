package org.glypharis.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/** What the adb command asks for: plain file names inside the app's folder and a thread count. */
internal class OperatorRequest(
    val model: String,
    val input: String,
    val output: String,
    val threads: Int,
)

/**
 * GW-EFF-015 part a: create a session on the model and run one input pair. The result says what
 * happened and holds no timing. A failure is a result, written like any other.
 */
internal object OperatorCheck {
    const val MODE = "operator_check"
    private const val LOGITS = "logits"

    fun run(folder: File, request: OperatorRequest): JSONObject {
        val result = JSONObject()
            .put("schema", BenchActivity.RESULT_SCHEMA)
            .put("mode", MODE)
            .put("status", "bad_request")
            .put("session_created", false)
            .put("session_error", JSONObject.NULL)
            .put("run_error", JSONObject.NULL)
            .put("request_error", JSONObject.NULL)
            .put("output_shape", JSONObject.NULL)
            .put("logits_finite", JSONObject.NULL)
            .put("session_options", sessionOptionsRecord(request.threads))

        val modelFile: File
        val outputFile: File
        val inputs: List<NamedTensor>
        try {
            require(request.threads >= 1) { "threads must be positive, got ${request.threads}" }
            modelFile = readable(folder, request.model)
            val inputFile = readable(folder, request.input)
            outputFile = BenchFiles.resolve(folder, request.output)
            result.put("model_sha256", BenchFiles.sha256(modelFile))
            result.put("input_sha256", BenchFiles.sha256(inputFile))
            inputs = TensorFile.read(inputFile)
        } catch (e: Exception) {
            return result.put("request_error", describe(e, folder))
        }

        val env: OrtEnvironment
        try {
            env = OrtEnvironment.getEnvironment()
            env.setTelemetry(false)
            result.put("onnxruntime_version", env.version)
        } catch (e: Throwable) {
            return result.put("status", "runtime_failed").put("session_error", describe(e, folder))
        }

        val session: OrtSession
        try {
            session = sessionOptions(request.threads).use { env.createSession(modelFile.path, it) }
        } catch (e: Throwable) {
            return result.put("status", "session_failed").put("session_error", describe(e, folder))
        }
        result.put("session_created", true)

        session.use {
            result.put("input_names", JSONArray(it.inputNames.toList()))
            result.put("output_names", JSONArray(it.outputNames.toList()))
            try {
                runOnce(env, it, inputs, outputFile, result)
                result.put("status", "ok")
            } catch (e: Throwable) {
                result.put("status", "run_failed").put("run_error", describe(e, folder))
            }
        }
        return result
    }

    private fun readable(folder: File, name: String): File {
        val file = BenchFiles.resolve(folder, name)
        require(file.isFile && file.canRead() && file.length() > 0) {
            "$name is missing or cannot be read"
        }
        return file
    }

    /** The options of `glypharis.efficiency.threads.ort_session`; spinning stays at the default. */
    private fun sessionOptions(threads: Int): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setCPUArenaAllocator(false)
            setMemoryPatternOptimization(false)
            setIntraOpNumThreads(threads)
        }

    /** What `sessionOptions` sets, in the names `threads.session_options` reports. */
    private fun sessionOptionsRecord(threads: Int): JSONObject =
        JSONObject()
            .put("providers", JSONArray(listOf("CPUExecutionProvider")))
            .put("intra_op_num_threads", threads)
            .put("execution_mode", "ORT_SEQUENTIAL")
            .put("graph_optimization_level", "ORT_ENABLE_ALL")
            .put("enable_cpu_mem_arena", false)
            .put("enable_mem_pattern", false)
            .put("allow_spinning", "default")

    private fun runOnce(
        env: OrtEnvironment,
        session: OrtSession,
        inputs: List<NamedTensor>,
        outputFile: File,
        result: JSONObject,
    ) {
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            for (t in inputs) tensors[t.name] = onnxTensor(env, t)
            session.run(tensors, setOf(LOGITS)).use { outputs ->
                val logits = outputs.get(LOGITS).get() as OnnxTensor
                val shape = logits.info.shape
                val values = logits.floatBuffer
                val bytes = ByteBuffer.allocateDirect(values.remaining() * Float.SIZE_BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN)
                var finite = true
                while (values.hasRemaining()) {
                    val v = values.get()
                    if (v.isNaN() || v.isInfinite()) finite = false
                    bytes.putFloat(v)
                }
                bytes.flip()
                TensorFile.write(
                    outputFile,
                    listOf(NamedTensor(LOGITS, ElementType.FLOAT32, shape, bytes)),
                )
                result.put("output_shape", JSONArray(shape.toList()))
                result.put("logits_finite", finite)
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    private fun onnxTensor(env: OrtEnvironment, t: NamedTensor): OnnxTensor =
        when (t.type) {
            ElementType.FLOAT32 -> OnnxTensor.createTensor(env, t.data.asFloatBuffer(), t.shape)
            ElementType.INT64 -> OnnxTensor.createTensor(env, t.data.asLongBuffer(), t.shape)
        }

    /** The error as a record. ONNX Runtime's messages name the model's path; it is cut out. */
    private fun describe(e: Throwable, folder: File): JSONObject {
        val out = JSONObject()
            .put("type", e.javaClass.name)
            .put("message", e.message?.replace(folder.path, "<app files>") ?: JSONObject.NULL)
        if (e is OrtException) out.put("code", e.code.name)
        return out
    }
}
