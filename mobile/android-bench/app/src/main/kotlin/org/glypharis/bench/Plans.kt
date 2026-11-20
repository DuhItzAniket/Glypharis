package org.glypharis.bench

import java.io.File
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** A model a plan names: its file, the SHA-256 the file must have and the output to ask for. */
internal class PlanModel(val file: String, val sha256: String, val output: String)

/** A tensor file a plan names, with the SHA-256 and size it must have. */
internal class PlanTensor(val sha256: String, val bytes: Long)

/**
 * One `session.run` call of a timing item: the model it goes to (always `Plans.DETECTOR` or
 * `Plans.RECOGNIZER` itself, so a call holds no string of its own for it) and its inputs' GWT1 file.
 */
internal class PlanCall(val model: String, val tensor: String)

/** One item of a timing run: a page (its detector call, then its recognizer calls) or a line. */
internal class PlanItem(
    val kind: String,
    val key: String,
    val warmup: Boolean,
    val calls: List<PlanCall>,
)

/**
 * A timing plan, `glypharis-phone-plan-1` (README.md). Its tensor list is returned beside it
 * (`ParsedTiming`), so that it can be dropped once the files are checked and does not stay live
 * through the timed items.
 */
internal class TimingPlan(
    val experiment: String,
    val arm: String,
    val threads: Int,
    val models: Map<String, PlanModel>,
    val bufferBytes: Int,
    val warmupPages: Int,
    val timedPages: Int,
    val lines: Int,
    val items: List<PlanItem>,
)

/** A timing plan and the tensor files it lists, with their hashes and sizes. */
internal class ParsedTiming(val plan: TimingPlan, val tensors: Map<String, PlanTensor>)

/** One call of a check plan: its inputs' file, the file its logits go to, and its origin. */
internal class CheckCall(val tensor: String, val output: String, val source: String)

/** A check plan, `glypharis-phone-check-1` (README.md). */
internal class CheckPlan(
    val experiment: String,
    val arm: String,
    val threads: Int,
    val model: PlanModel,
    val calls: List<CheckCall>,
    val tensors: Map<String, PlanTensor>,
)

/**
 * The plans `scripts/phase22_android.py` writes, read strictly: a field of the wrong type, an item
 * out of the registered order or a tensor file not listed refuses the whole plan. The files a plan
 * names are checked here too, before any session exists.
 */
internal object Plans {
    const val TIMING_SCHEMA = "glypharis-phone-plan-1"
    const val CHECK_SCHEMA = "glypharis-phone-check-1"
    const val DETECTOR = "detector"
    const val RECOGNIZER = "recognizer"
    private const val PAGE = "page"
    private const val LINE = "line"
    private val ROLES = listOf(DETECTOR, RECOGNIZER)
    private val SHA256 = Regex("[0-9a-f]{64}")
    private const val MESSAGE_CHARS = 200

    /** The plan file's JSON object; a parse error is reported without the plan's text. */
    fun parse(bytes: ByteArray): JSONObject =
        try {
            JSONObject(String(bytes, Charsets.UTF_8))
        } catch (e: JSONException) {
            throw IllegalArgumentException(
                "the plan is not a JSON object: ${e.message?.take(MESSAGE_CHARS)}",
            )
        }

    /**
     * A timing plan. Its items run as `warmup_pages` warm-up pages, `timed_pages` timed pages and
     * then `lines` lines. A page's calls are its detector call and then its recognizer calls; a
     * line's is one recognizer call. `buffer_bytes` is the size of the one input buffer every call
     * is read into, the same for every arm.
     */
    fun timing(doc: JSONObject): ParsedTiming {
        require(doc.string("schema", "") == TIMING_SCHEMA) { "the plan is not $TIMING_SCHEMA" }
        val models = doc.obj("models", "")
        val roles = models.keys().asSequence().toSet()
        require(roles == ROLES.toSet()) {
            "models must be $DETECTOR and $RECOGNIZER, the plan has ${roles.sorted()}"
        }
        val tensors = tensors(doc.obj("tensors", ""))
        val list = doc.list("items", "")
        val items = List(list.length()) { i ->
            item(element(list, i, "items"), "items[$i].", tensors)
        }
        val warmup = doc.count("warmup_pages", "")
        val timed = doc.count("timed_pages", "")
        val lines = doc.count("lines", "")
        val due = warmup.toLong() + timed + lines
        require(items.isNotEmpty()) { "the plan has no items" }
        require(items.size.toLong() == due) {
            "the plan has ${items.size} items, its counts add up to $due"
        }
        items.forEachIndexed { i, item -> checkOrder(i, item, warmup.toLong(), timed.toLong()) }
        val buffer = doc.count("buffer_bytes", "")
        require(buffer > 0) { "buffer_bytes must be positive" }
        val plan = TimingPlan(
            experiment = doc.string("experiment", ""),
            arm = doc.string("arm", ""),
            threads = doc.count("threads", ""),
            models = ROLES.associateWith { model(models.obj(it, "models."), "models.$it.") },
            bufferBytes = buffer,
            warmupPages = warmup,
            timedPages = timed,
            lines = lines,
            items = items,
        )
        return ParsedTiming(plan, tensors)
    }

    /** A check plan: the recognizer and its calls, in order. */
    fun check(doc: JSONObject): CheckPlan {
        require(doc.string("schema", "") == CHECK_SCHEMA) { "the plan is not $CHECK_SCHEMA" }
        val tensors = tensors(doc.obj("tensors", ""))
        val list = doc.list("calls", "")
        require(list.length() > 0) { "the plan has no calls" }
        val calls = List(list.length()) { i ->
            val at = "calls[$i]."
            val call = element(list, i, "calls")
            CheckCall(
                tensor = listed(call.string("tensor", at), tensors, at),
                output = call.string("output", at),
                source = call.string("source", at),
            )
        }
        return CheckPlan(
            experiment = doc.string("experiment", ""),
            arm = doc.string("arm", ""),
            threads = doc.count("threads", ""),
            model = model(doc.obj("model", ""), "model."),
            calls = calls,
            tensors = tensors,
        )
    }

    /**
     * The model files, each present and with the plan's SHA-256. The hashes go under
     * `model_sha256`, and the model being checked is named under `failed_model` until it passes,
     * so a refusal says which model it was.
     */
    fun checkModels(
        folder: File,
        models: Map<String, PlanModel>,
        result: JSONObject,
    ): Map<String, File> {
        val hashes = JSONObject()
        result.put("model_sha256", hashes)
        val files = LinkedHashMap<String, File>()
        for ((role, model) in models) {
            result.put("failed_model", role)
            val file = BenchFiles.readable(folder, model.file)
            val got = BenchFiles.sha256(file)
            hashes.put(role, got)
            require(got == model.sha256) {
                "$role: ${model.file} has SHA-256 $got, the plan says ${model.sha256}"
            }
            files[role] = file
        }
        result.put("failed_model", JSONObject.NULL)
        return files
    }

    /**
     * Every tensor file `names` lists, each present with the size and SHA-256 the plan gives.
     * One read block serves every file. Returns the files by name.
     */
    fun checkTensors(
        folder: File,
        tensors: Map<String, PlanTensor>,
        names: List<String>,
    ): Map<String, File> {
        val block = ByteArray(1 shl 20)
        val files = LinkedHashMap<String, File>()
        for (name in names) {
            if (name in files) continue
            val want = tensors.getValue(name)
            val file = BenchFiles.readable(folder, name)
            val size = file.length()
            require(size == want.bytes) { "$name has $size bytes, the plan says ${want.bytes}" }
            val got = BenchFiles.sha256(file, block)
            require(got == want.sha256) { "$name has SHA-256 $got, the plan says ${want.sha256}" }
            files[name] = file
        }
        return files
    }

    /** The check's output files, one plain name per call; none may be a file the check reads. */
    fun checkOutputs(folder: File, plan: CheckPlan, planName: String): List<File> {
        val read = plan.tensors.keys + plan.model.file + planName
        val seen = HashSet<String>()
        return plan.calls.mapIndexed { i, call ->
            require(seen.add(call.output)) { "calls[$i].output ${call.output} is named twice" }
            require(call.output !in read) { "calls[$i].output ${call.output} is an input" }
            BenchFiles.resolve(folder, call.output)
        }
    }

    private fun checkOrder(i: Int, item: PlanItem, warmup: Long, timed: Long) {
        val page = i < warmup + timed
        val warm = i < warmup
        val kind = if (page) PAGE else LINE
        require(item.kind == kind && item.warmup == warm) {
            val got = label(item.kind, item.warmup)
            "items[$i] is a $got where a ${label(kind, warm)} is due: " +
                "items run as warm-up pages, then timed pages, then lines"
        }
        val models = item.calls.map { it.model }
        if (page) {
            require(models.first() == DETECTOR && models.drop(1).all { it == RECOGNIZER }) {
                "items[$i]: a page's calls are its detector call, then its recognizer calls"
            }
        } else {
            require(models == listOf(RECOGNIZER)) { "items[$i]: a line is one recognizer call" }
        }
    }

    private fun label(kind: String, warmup: Boolean): String =
        if (warmup) "warm-up $kind" else "timed $kind"

    private fun item(doc: JSONObject, where: String, tensors: Map<String, PlanTensor>): PlanItem {
        val list = doc.list("calls", where)
        require(list.length() > 0) { "${where}calls is empty" }
        val calls = List(list.length()) { j ->
            val at = "${where}calls[$j]."
            val call = element(list, j, "${where}calls")
            PlanCall(
                role(call.string("model", at), at),
                listed(call.string("tensor", at), tensors, at),
            )
        }
        return PlanItem(
            kind = doc.string("kind", where),
            key = doc.string("key", where),
            warmup = doc.bool("warmup", where),
            calls = calls,
        )
    }

    private fun model(doc: JSONObject, where: String): PlanModel =
        PlanModel(
            file = doc.string("file", where),
            sha256 = sha256(doc.string("sha256", where), where),
            output = doc.string("output", where),
        )

    private fun tensors(doc: JSONObject): Map<String, PlanTensor> {
        val out = LinkedHashMap<String, PlanTensor>()
        for (name in doc.keys()) {
            val at = "tensors.$name."
            val entry = doc.obj(name, "tensors.")
            val bytes = entry.whole("bytes", at)
            require(bytes > 0) { "${at}bytes must be positive" }
            out[name] = PlanTensor(sha256(entry.string("sha256", at), at), bytes)
        }
        return out
    }

    /** `name` as the shared constant of its role; any other model name refuses the plan. */
    private fun role(name: String, where: String): String =
        ROLES.firstOrNull { it == name }
            ?: throw IllegalArgumentException("${where}model $name is not $DETECTOR or $RECOGNIZER")

    private fun listed(name: String, tensors: Map<String, PlanTensor>, where: String): String {
        require(name in tensors) { "${where}tensor $name is not listed under tensors" }
        return name
    }

    private fun sha256(value: String, where: String): String {
        require(SHA256.matches(value)) { "${where}sha256 is not 64 lower-case hex digits" }
        return value
    }

    private fun element(list: JSONArray, i: Int, where: String): JSONObject =
        list.opt(i) as? JSONObject ?: throw IllegalArgumentException("$where[$i] must be an object")
}

private fun JSONObject.string(key: String, where: String): String =
    opt(key) as? String ?: throw IllegalArgumentException("$where$key must be a string")

private fun JSONObject.bool(key: String, where: String): Boolean =
    opt(key) as? Boolean ?: throw IllegalArgumentException("$where$key must be true or false")

private fun JSONObject.whole(key: String, where: String): Long =
    when (val value = opt(key)) {
        is Int -> value.toLong()
        is Long -> value
        else -> throw IllegalArgumentException("$where$key must be a whole number")
    }

private fun JSONObject.count(key: String, where: String): Int {
    val value = whole(key, where)
    require(value in 0L..Int.MAX_VALUE.toLong()) { "$where$key must be a count" }
    return value.toInt()
}

private fun JSONObject.obj(key: String, where: String): JSONObject =
    opt(key) as? JSONObject ?: throw IllegalArgumentException("$where$key must be an object")

private fun JSONObject.list(key: String, where: String): JSONArray =
    opt(key) as? JSONArray ?: throw IllegalArgumentException("$where$key must be a list")
