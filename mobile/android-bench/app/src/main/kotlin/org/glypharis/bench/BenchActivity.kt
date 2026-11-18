package org.glypharis.bench

import android.app.Activity
import android.os.Bundle
import android.view.WindowManager
import android.widget.TextView
import java.io.File
import kotlin.concurrent.thread
import org.json.JSONObject

/**
 * The bench's one screen, started by adb (README.md). `mode` picks the work: `ready` creates the
 * app's external files folder and marks it, `operator_check` is GW-EFF-015 part a. A later mode is
 * one more branch in `dispatch`. The work runs on a worker thread and the activity closes itself
 * when it is done.
 */
class BenchActivity : Activity() {
    private lateinit var label: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        label = TextView(this).apply {
            text = "Glypharis bench: working"
            setPadding(48, 48, 48, 48)
        }
        setContentView(label)
        if (savedInstanceState != null) return // the work already running is not started twice
        val folder = getExternalFilesDir(null)
        if (folder == null) {
            label.text = "Glypharis bench: the external files folder is not available"
            return
        }
        val mode = intent.getStringExtra(EXTRA_MODE) ?: MODE_READY
        thread(name = "bench") {
            val status = dispatch(folder, mode)
            runOnUiThread {
                label.text = "Glypharis bench: $status"
                finishAndRemoveTask()
            }
        }
    }

    private fun dispatch(folder: File, mode: String): String {
        if (mode == MODE_READY) {
            write(folder, READY_FILE, JSONObject().put("schema", RESULT_SCHEMA))
            return MODE_READY
        }
        val result = try {
            when (mode) {
                OperatorCheck.MODE -> OperatorCheck.run(folder, operatorRequest())
                else -> failure(mode, "unknown mode: $mode")
            }
        } catch (e: Exception) {
            failure(mode, e.message ?: e.javaClass.name)
        }
        val name = intent.getStringExtra(EXTRA_RESULT) ?: DEFAULT_RESULT
        return try {
            write(folder, name, result)
            result.getString("status")
        } catch (e: Exception) {
            write(folder, DEFAULT_RESULT, failure(mode, "cannot write $name: ${e.message}"))
            STATUS_BAD_REQUEST
        }
    }

    private fun operatorRequest() = OperatorRequest(
        model = required(EXTRA_MODEL),
        input = required(EXTRA_INPUT),
        output = required(EXTRA_OUTPUT),
        threads = intent.getIntExtra(EXTRA_THREADS, 0),
    )

    private fun required(name: String): String =
        requireNotNull(intent.getStringExtra(name)) { "missing extra: $name" }

    private fun failure(mode: String, message: String): JSONObject =
        JSONObject()
            .put("schema", RESULT_SCHEMA)
            .put("mode", mode)
            .put("status", STATUS_BAD_REQUEST)
            .put("session_created", false)
            .put("request_error", JSONObject().put("message", message))

    private fun write(folder: File, name: String, doc: JSONObject) {
        BenchFiles.writeAtomic(BenchFiles.resolve(folder, name), doc.toString(1).toByteArray())
    }

    companion object {
        const val RESULT_SCHEMA = "glypharis-android-bench-1"
        private const val STATUS_BAD_REQUEST = "bad_request"
        private const val MODE_READY = "ready"
        private const val READY_FILE = "ready.json"
        private const val DEFAULT_RESULT = "result.json"
        private const val EXTRA_MODE = "mode"
        private const val EXTRA_MODEL = "model"
        private const val EXTRA_INPUT = "input"
        private const val EXTRA_OUTPUT = "output"
        private const val EXTRA_RESULT = "result"
        private const val EXTRA_THREADS = "threads"
    }
}
