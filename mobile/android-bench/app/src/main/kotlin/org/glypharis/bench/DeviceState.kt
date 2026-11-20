package org.glypharis.bench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File
import org.json.JSONObject

/**
 * The phone's state around a timing run, read with no permission: the process's peak and current
 * resident memory, the thermal state, the battery, and whether the screen is on. A value the phone
 * does not give is null; a reading that fails says why under `error`.
 */
internal object DeviceState {
    const val VM_HWM = "VmHWM"
    const val VM_RSS = "VmRSS"
    private val MEMORY_LINE = Regex("^(VmHWM|VmRSS):\\s+(\\d+)\\s+kB")
    private val THERMAL_NAMES = mapOf(
        PowerManager.THERMAL_STATUS_NONE to "none",
        PowerManager.THERMAL_STATUS_LIGHT to "light",
        PowerManager.THERMAL_STATUS_MODERATE to "moderate",
        PowerManager.THERMAL_STATUS_SEVERE to "severe",
        PowerManager.THERMAL_STATUS_CRITICAL to "critical",
        PowerManager.THERMAL_STATUS_EMERGENCY to "emergency",
        PowerManager.THERMAL_STATUS_SHUTDOWN to "shutdown",
    )

    /** `VmHWM` and `VmRSS` of this process from `/proc/self/status`, in kB; empty if unreadable. */
    fun memoryKb(): Map<String, Long> =
        try {
            val out = HashMap<String, Long>()
            File("/proc/self/status").forEachLine { line ->
                MEMORY_LINE.find(line)?.let { out[it.groupValues[1]] = it.groupValues[2].toLong() }
            }
            out
        } catch (e: Exception) {
            emptyMap()
        }

    /**
     * `PowerManager.currentThermalStatus` as `status` and `status_name` (API 29 and later), and
     * `getThermalHeadroom(0)` as `headroom` (API 30 and later; null when the phone gives NaN).
     */
    fun thermal(context: Context): JSONObject {
        val out = JSONObject()
            .put("status", JSONObject.NULL)
            .put("status_name", JSONObject.NULL)
            .put("headroom", JSONObject.NULL)
            .put("error", JSONObject.NULL)
        try {
            val power = context.getSystemService(PowerManager::class.java)
                ?: return out.put("error", "no power service")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val status = power.currentThermalStatus
                out.put("status", status).put("status_name", THERMAL_NAMES[status] ?: "unknown")
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val headroom = power.getThermalHeadroom(0)
                if (headroom.isFinite()) out.put("headroom", headroom.toDouble())
            }
        } catch (e: Exception) {
            out.put("error", "${e.javaClass.name}: ${e.message}")
        }
        return out
    }

    /**
     * The battery from the sticky `ACTION_BATTERY_CHANGED` intent, which needs no permission:
     * temperature in degrees Celsius, level in percent, and the `plugged` and `status` codes of
     * `BatteryManager`.
     */
    fun battery(context: Context): JSONObject {
        val out = JSONObject()
            .put("temperature_c", JSONObject.NULL)
            .put("level_percent", JSONObject.NULL)
            .put("plugged", JSONObject.NULL)
            .put("status", JSONObject.NULL)
            .put("error", JSONObject.NULL)
        try {
            val sticky = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                ?: return out.put("error", "no battery state")
            val tenths = sticky.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            if (tenths != Int.MIN_VALUE) out.put("temperature_c", tenths / 10.0)
            val level = sticky.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = sticky.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) out.put("level_percent", level * 100.0 / scale)
            val plugged = sticky.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1)
            if (plugged >= 0) out.put("plugged", plugged)
            val status = sticky.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            if (status >= 0) out.put("status", status)
        } catch (e: Exception) {
            out.put("error", "${e.javaClass.name}: ${e.message}")
        }
        return out
    }

    /** `PowerManager.isInteractive`, or null when it cannot be read. */
    fun interactive(context: Context): Any =
        try {
            context.getSystemService(PowerManager::class.java)?.isInteractive ?: JSONObject.NULL
        } catch (e: Exception) {
            JSONObject.NULL
        }
}
