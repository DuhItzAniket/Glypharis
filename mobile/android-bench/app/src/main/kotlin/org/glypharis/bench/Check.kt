package org.glypharis.bench

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/** What the adb command asks for: the check plan's file name and the thread count. */
internal class CheckRequest(
    val plan: String,
    val threads: Int,
)

/**
 * GW-EFF-015 part b: the output check of one arm, in a process of its own (README.md). Only the
 * recognizer session is created. Each call of the check plan runs in order and its logits are
 * written to the call's output file, for the laptop to compare. Nothing is timed and no memory
 * figure is read. A failure is a result, written like any other.
 */
internal object Check {
    const val MODE = "check"
    private const val LOGITS = "logits"

    fun run(folder: File, request: CheckRequest): JSONObject {
        val result = JSONObject()
            .put("schema", BenchActivity.RESULT_SCHEMA)
            .put("mode", MODE)
            .put("status", "bad_request")
            .put("failed_model", JSONObject.NULL)
            .put("request_error", JSONObject.NULL)
            .put("session_error", JSONObject.NULL)
            .put("run_error", JSONObject.NULL)
            .put("pid", android.os.Process.myPid())
            .put("experiment", JSONObject.NULL)
            .put("arm", JSONObject.NULL)
            .put("threads", request.threads)
            .put("plan", request.plan)
            .put("plan_sha256", JSONObject.NULL)
            .put("model_sha256", JSONObject.NULL)
            .put("tensors_checked", JSONObject.NULL)
            .put("onnxruntime_version", JSONObject.NULL)
            .put("session_options", BenchSession.optionsRecord(request.threads))
            .put("session_created", false)
            .put("input_names", JSONObject.NULL)
            .put("output_names", JSONObject.NULL)
            .put("calls_done", 0)
            .put("all_finite", JSONObject.NULL)
            .put("calls", JSONArray())

        val plan: CheckPlan
        val model: File
        val inputs: Map<String, File>
        val outputs: List<File>
        try {
            BenchSession.claimProcess()
            require(request.threads >= 1) { "threads must be positive, got ${request.threads}" }
            val text = BenchFiles.readable(folder, request.plan).readBytes()
            result.put("plan_sha256", BenchFiles.sha256(text))
            plan = Plans.check(Plans.parse(text))
            result.put("experiment", plan.experiment).put("arm", plan.arm)
            require(plan.threads == request.threads) {
                "the plan is for ${plan.threads} threads, the request for ${request.threads}"
            }
            val models = mapOf(Plans.RECOGNIZER to plan.model)
            model = Plans.checkModels(folder, models, result).getValue(Plans.RECOGNIZER)
            outputs = Plans.checkOutputs(folder, plan, request.plan)
            inputs = Plans.checkTensors(folder, plan.tensors, plan.calls.map { it.tensor })
            result.put("tensors_checked", inputs.size)
        } catch (e: Exception) {
            return result.put("request_error", BenchSession.describe(e, folder))
        }

        val env: OrtEnvironment
        try {
            env = OrtEnvironment.getEnvironment()
            env.setTelemetry(false)
            result.put("onnxruntime_version", env.version)
        } catch (e: Throwable) {
            return result.put("status", "runtime_failed")
                .put("session_error", BenchSession.describe(e, folder))
        }

        val session: OrtSession
        try {
            session = BenchSession.options(request.threads).use {
                env.createSession(model.path, it)
            }
        } catch (e: Throwable) {
            return result.put("status", "session_failed")
                .put("failed_model", Plans.RECOGNIZER)
                .put("session_error", BenchSession.describe(e, folder))
        }
        result.put("session_created", true)

        try {
            result.put("input_names", role(JSONArray(session.inputNames.toList())))
                .put("output_names", role(JSONArray(session.outputNames.toList())))
            try {
                require(plan.model.output in session.outputNames) {
                    "the recognizer has no output ${plan.model.output}"
                }
            } catch (e: IllegalArgumentException) {
                return result.put("failed_model", Plans.RECOGNIZER)
                    .put("request_error", BenchSession.describe(e, folder))
            }
            checkCalls(env, session, plan, inputs, outputs, folder, result)
        } finally {
            runCatching { session.close() }
        }
        return result
    }

    private fun checkCalls(
        env: OrtEnvironment,
        session: OrtSession,
        plan: CheckPlan,
        inputs: Map<String, File>,
        outputs: List<File>,
        folder: File,
        result: JSONObject,
    ) {
        val wanted = setOf(plan.model.output)
        val records = JSONArray()
        result.put("calls", records)
        var allFinite = true
        for ((i, call) in plan.calls.withIndex()) {
            try {
                val (shape, finite) = runCall(
                    env,
                    session,
                    wanted,
                    plan.model.output,
                    inputs.getValue(call.tensor),
                    outputs[i],
                )
                allFinite = allFinite && finite
                records.put(
                    JSONObject()
                        .put("tensor", call.tensor)
                        .put("output", call.output)
                        .put("source", call.source)
                        .put("output_shape", JSONArray(shape.toList()))
                        .put("logits_finite", finite),
                )
            } catch (e: UnreadableInput) {
                val error = BenchSession.describe(e.cause ?: e, folder)
                result.put("request_error", at(error, i, call))
                break
            } catch (e: UnwritableOutput) {
                // Storage, not the model: the result stays bad_request and the step stays due.
                val error = BenchSession.describe(e.cause ?: e, folder)
                result.put("request_error", at(error, i, call))
                break
            } catch (e: Throwable) {
                val error = BenchSession.describe(e, folder)
                result.put("status", "run_failed")
                    .put("failed_model", Plans.RECOGNIZER)
                    .put("run_error", at(error, i, call))
                break
            }
        }
        result.put("calls_done", records.length()).put("all_finite", allFinite)
        if (records.length() == plan.calls.size) result.put("status", "ok")
    }

    /** One call; its logits go to `output` as a GWT1 file holding one float32 tensor, `logits`. */
    private fun runCall(
        env: OrtEnvironment,
        session: OrtSession,
        wanted: Set<String>,
        name: String,
        input: File,
        output: File,
    ): Pair<LongArray, Boolean> {
        val tensors = BenchSession.readInputs(input)
        val feeds = LinkedHashMap<String, OnnxTensor>()
        try {
            for (t in tensors) feeds[t.name] = BenchSession.tensor(env, t)
            session.run(feeds, wanted).use { got ->
                val logits = got.get(name).get() as OnnxTensor
                val type = logits.info.type
                require(type == OnnxJavaType.FLOAT) { "the output $name is $type, not float32" }
                val shape = logits.info.shape
                val (bytes, finite) = BenchSession.floatBytes(logits)
                try {
                    TensorFile.write(
                        output,
                        listOf(NamedTensor(LOGITS, ElementType.FLOAT32, shape, bytes)),
                    )
                } catch (e: Exception) {
                    throw UnwritableOutput(e)
                }
                return shape to finite
            }
        } finally {
            feeds.values.forEach { it.close() }
        }
    }

    private fun role(names: JSONArray): JSONObject = JSONObject().put(Plans.RECOGNIZER, names)

    private fun at(error: JSONObject, call: Int, planCall: CheckCall): JSONObject =
        error.put("call", call).put("tensor", planCall.tensor)
}
