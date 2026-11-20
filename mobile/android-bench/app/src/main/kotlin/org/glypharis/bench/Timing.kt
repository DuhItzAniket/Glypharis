package org.glypharis.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import org.json.JSONArray
import org.json.JSONObject

/** What the adb command asks for: the plan's file name, the thread count and the run number. */
internal class TimingRequest(
    val plan: String,
    val threads: Int,
    val run: Int,
)

/**
 * GW-EFF-015 part b: one timing run of one arm, in a new process (README.md). Every file is
 * checked against the plan before a session exists. The items then run in the plan's order; each
 * call's inputs are read from their GWT1 file just before the call and dropped after it, and only
 * `session.run` lies between the two time stamps. Each output is closed as soon as its call is
 * timed, so the run holds no logits, and `VmHWM` is read straight after the last item. A failure
 * is a result, written like any other.
 */
internal object Timing {
    const val MODE = "timing"

    fun run(context: Context, folder: File, request: TimingRequest): JSONObject {
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
            .put("run", request.run)
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
            .put("warmup_pages", JSONObject.NULL)
            .put("timed_pages", JSONObject.NULL)
            .put("lines", JSONObject.NULL)
            .put("interactive_start", JSONObject.NULL)
            .put("thermal_start", JSONObject.NULL)
            .put("battery_start", JSONObject.NULL)
            .put("items_done", 0)
            .put("vm_hwm_kb", JSONObject.NULL)
            .put("vm_rss_kb", JSONObject.NULL)
            .put("interactive_end", JSONObject.NULL)
            .put("thermal_end", JSONObject.NULL)
            .put("battery_end", JSONObject.NULL)
            .put("items", JSONArray())

        val plan: TimingPlan
        val models: Map<String, File>
        try {
            BenchSession.claimProcess()
            require(request.threads >= 1) { "threads must be positive, got ${request.threads}" }
            require(request.run >= 1) { "run must be positive, got ${request.run}" }
            val text = BenchFiles.readable(folder, request.plan).readBytes()
            result.put("plan_sha256", BenchFiles.sha256(text))
            // The tensor list and the files' map are dropped after the checks: only the items
            // stay live through the timed run.
            val parsed = Plans.timing(Plans.parse(text))
            plan = parsed.plan
            result.put("experiment", plan.experiment)
                .put("arm", plan.arm)
                .put("buffer_bytes", plan.bufferBytes)
                .put("warmup_pages", plan.warmupPages)
                .put("timed_pages", plan.timedPages)
                .put("lines", plan.lines)
            require(plan.threads == request.threads) {
                "the plan is for ${plan.threads} threads, the request for ${request.threads}"
            }
            models = Plans.checkModels(folder, plan.models, result)
            val names = plan.items.flatMap { item -> item.calls.map { it.tensor } }
            result.put("tensors_checked", Plans.checkTensors(folder, parsed.tensors, names).size)
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

        val sessions = LinkedHashMap<String, OrtSession>()
        try {
            for ((role, file) in models) {
                result.put("failed_model", role)
                sessions[role] = BenchSession.options(request.threads).use {
                    env.createSession(file.path, it)
                }
            }
        } catch (e: Throwable) {
            sessions.values.forEach { runCatching { it.close() } }
            return result.put("status", "session_failed")
                .put("session_error", BenchSession.describe(e, folder))
        }
        result.put("failed_model", JSONObject.NULL).put("session_created", true)

        try {
            result.put("input_names", names(sessions) { it.inputNames })
                .put("output_names", names(sessions) { it.outputNames })
            for ((role, session) in sessions) {
                val output = plan.models.getValue(role).output
                try {
                    require(output in session.outputNames) { "the $role has no output $output" }
                } catch (e: IllegalArgumentException) {
                    return result.put("failed_model", role)
                        .put("request_error", BenchSession.describe(e, folder))
                }
            }
            measure(context, env, plan, sessions, folder, result)
        } finally {
            sessions.values.forEach { runCatching { it.close() } }
        }
        return result
    }

    /**
     * The timed items, with the phone's state before the first and after the last. Every call is
     * read into one buffer of the plan's `buffer_bytes`, the same size for every arm, so dropped
     * inputs leave no garbage behind; the heap is collected once before the first item, outside
     * any timed window, and the memory at that point is recorded.
     */
    private fun measure(
        context: Context,
        env: OrtEnvironment,
        plan: TimingPlan,
        sessions: Map<String, OrtSession>,
        folder: File,
        result: JSONObject,
    ) {
        val wanted = plan.models.mapValues { setOf(it.value.output) }
        val times = Array(plan.items.size) { LongArray(plan.items[it].calls.size) }
        val arena = BenchSession.arena(plan.bufferBytes)
        var done = 0
        System.gc()
        System.runFinalization()
        System.gc()
        val before = DeviceState.memoryKb()
        result.put("vm_hwm_kb_before_items", before[DeviceState.VM_HWM] ?: JSONObject.NULL)
            .put("vm_rss_kb_before_items", before[DeviceState.VM_RSS] ?: JSONObject.NULL)
        result.put("interactive_start", DeviceState.interactive(context))
            .put("thermal_start", DeviceState.thermal(context))
            .put("battery_start", DeviceState.battery(context))

        items@ for ((i, item) in plan.items.withIndex()) {
            for ((j, call) in item.calls.withIndex()) {
                try {
                    times[i][j] = timeCall(
                        env,
                        sessions.getValue(call.model),
                        wanted.getValue(call.model),
                        folder,
                        call.tensor,
                        arena,
                    )
                } catch (e: UnreadableInput) {
                    val error = BenchSession.describe(e.cause ?: e, folder)
                    result.put("request_error", at(error, i, j, call))
                    break@items
                } catch (e: Throwable) {
                    val error = BenchSession.describe(e, folder)
                    result.put("status", "run_failed")
                        .put("failed_model", call.model)
                        .put("run_error", at(error, i, j, call))
                    break@items
                }
            }
            done = i + 1
        }

        val memory = DeviceState.memoryKb()
        result.put("items_done", done)
            .put("vm_hwm_kb", memory[DeviceState.VM_HWM] ?: JSONObject.NULL)
            .put("vm_rss_kb", memory[DeviceState.VM_RSS] ?: JSONObject.NULL)
            .put("interactive_end", DeviceState.interactive(context))
            .put("thermal_end", DeviceState.thermal(context))
            .put("battery_end", DeviceState.battery(context))
        if (done == plan.items.size) result.put("status", "ok")
        result.put("items", records(plan, times, done))
    }

    /**
     * One call. Its inputs are read and wrapped first; only `session.run` lies between the two
     * time stamps, and the output and the input tensors are closed straight after the second.
     */
    private fun timeCall(
        env: OrtEnvironment,
        session: OrtSession,
        wanted: Set<String>,
        folder: File,
        name: String,
        arena: ByteBuffer,
    ): Long {
        val file = try {
            BenchFiles.resolve(folder, name)
        } catch (e: IllegalArgumentException) {
            throw UnreadableInput(e)
        }
        val tensors = BenchSession.readInputsInto(file, arena)
        val feeds = LinkedHashMap<String, OnnxTensor>()
        try {
            for (t in tensors) feeds[t.name] = BenchSession.tensor(env, t)
            val start = System.nanoTime()
            val outputs = session.run(feeds, wanted)
            val end = System.nanoTime()
            outputs.close()
            return end - start
        } finally {
            feeds.values.forEach { it.close() }
        }
    }

    /** Each finished item: its calls' times in plan order and their sums, in milliseconds. */
    private fun records(plan: TimingPlan, times: Array<LongArray>, done: Int): JSONArray {
        val out = JSONArray()
        for (i in 0 until done) {
            val item = plan.items[i]
            var detector = 0L
            var recognizer = 0L
            val calls = JSONArray()
            item.calls.forEachIndexed { j, call ->
                val ns = times[i][j]
                if (call.model == Plans.DETECTOR) detector += ns else recognizer += ns
                calls.put(ms(ns))
            }
            out.put(
                JSONObject()
                    .put("kind", item.kind)
                    .put("key", item.key)
                    .put("warmup", item.warmup)
                    .put("network_ms", ms(detector + recognizer))
                    .put("detector_ms", ms(detector))
                    .put("recognizer_ms", ms(recognizer))
                    .put("calls_ms", calls),
            )
        }
        return out
    }

    private fun ms(ns: Long): Double = ns / 1e6

    private fun at(error: JSONObject, item: Int, call: Int, planCall: PlanCall): JSONObject =
        error.put("item", item).put("call", call).put("tensor", planCall.tensor)

    private fun names(
        sessions: Map<String, OrtSession>,
        of: (OrtSession) -> Set<String>,
    ): JSONObject {
        val out = JSONObject()
        for ((role, session) in sessions) out.put(role, JSONArray(of(session).toList()))
        return out
    }
}
