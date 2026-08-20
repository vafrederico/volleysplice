package com.volleycut.nativeanalysis

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import org.json.JSONArray
import org.json.JSONObject
import java.text.DateFormat
import java.util.Date

internal enum class MediaProcessingOperation(val label: String) {
    INFERENCE("project inference"),
    EXPORT("video export"),
}

internal data class ProcessingTimeoutEvent(
    val sequence: Int,
    val operation: MediaProcessingOperation,
    val projectId: String?,
    val sourceName: String?,
    val occurredAtMs: Long,
    val detail: String,
) {
    fun displayTime(): String = DateFormat.getDateTimeInstance().format(Date(occurredAtMs))

    fun toJson(): JSONObject = JSONObject().apply {
        put("sequence", sequence)
        put("operation", operation.name)
        put("projectId", projectId ?: JSONObject.NULL)
        put("sourceName", sourceName ?: JSONObject.NULL)
        put("occurredAtMs", occurredAtMs)
        put("detail", detail)
    }

    companion object {
        fun fromJson(value: JSONObject): ProcessingTimeoutEvent? {
            val operation = value.optString("operation").let { wire ->
                MediaProcessingOperation.entries.firstOrNull { it.name == wire }
            } ?: return null
            val sequence = value.optInt("sequence", 0)
            val occurredAtMs = value.optLong("occurredAtMs", 0)
            if (sequence <= 0 || occurredAtMs <= 0) return null
            return ProcessingTimeoutEvent(
                sequence = sequence,
                operation = operation,
                projectId = value.optString("projectId").takeUnless { it.isBlank() || it == "null" },
                sourceName = value.optString("sourceName").takeUnless { it.isBlank() || it == "null" },
                occurredAtMs = occurredAtMs,
                detail = value.optString("detail"),
            )
        }
    }
}

internal data class ProcessingTimeoutSnapshot(
    val totalCount: Int,
    val events: List<ProcessingTimeoutEvent>,
    val acknowledgedThrough: Int,
) {
    val unacknowledged: List<ProcessingTimeoutEvent>
        get() = events.filter { it.sequence > acknowledgedThrough }

    val latest: ProcessingTimeoutEvent?
        get() = events.maxByOrNull { it.sequence }
}

/** Persists Android media-processing timeout evidence and notifies an open editor. */
internal object ProcessingTimeoutTracker {
    private const val PREFERENCES = "volleycut-processing-timeouts"
    private const val KEY_TOTAL_COUNT = "totalCount"
    private const val KEY_EVENTS = "events"
    private const val KEY_ACKNOWLEDGED_THROUGH = "acknowledgedThrough"
    private const val MAX_EVENTS = 20
    private const val CHANNEL_ID = "volleycut_processing_interruptions"
    private const val CHANNEL_NAME = "Processing interruptions"
    private const val NOTIFICATION_ID_BASE = 4_500

    const val ACTION_UPDATED = "com.volleycut.nativeanalysis.PROCESSING_TIMEOUT_UPDATED"

    @Synchronized
    fun snapshot(context: Context): ProcessingTimeoutSnapshot {
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        return ProcessingTimeoutSnapshot(
            totalCount = preferences.getInt(KEY_TOTAL_COUNT, 0),
            events = decodeHistory(preferences.getString(KEY_EVENTS, null)),
            acknowledgedThrough = preferences.getInt(KEY_ACKNOWLEDGED_THROUGH, 0),
        )
    }

    @Synchronized
    fun record(
        context: Context,
        operation: MediaProcessingOperation,
        projectId: String?,
        sourceName: String?,
        detail: String,
    ): ProcessingTimeoutEvent {
        val current = snapshot(context)
        val event = ProcessingTimeoutEvent(
            sequence = current.totalCount + 1,
            operation = operation,
            projectId = projectId,
            sourceName = sourceName,
            occurredAtMs = System.currentTimeMillis(),
            detail = detail,
        )
        val history = (current.events + event).takeLast(MAX_EVENTS)
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putInt(KEY_TOTAL_COUNT, event.sequence)
            .putString(KEY_EVENTS, encodeHistory(history))
            .commit()
        context.sendBroadcast(Intent(ACTION_UPDATED).setPackage(context.packageName))
        postNotification(context, event)
        return event
    }

    @Synchronized
    fun acknowledgeThrough(context: Context, sequence: Int) {
        val current = snapshot(context)
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putInt(KEY_ACKNOWLEDGED_THROUGH, maxOf(current.acknowledgedThrough, sequence))
            .commit()
        context.sendBroadcast(Intent(ACTION_UPDATED).setPackage(context.packageName))
    }

    internal fun encodeHistory(events: List<ProcessingTimeoutEvent>): String =
        JSONArray().apply { events.forEach { put(it.toJson()) } }.toString()

    internal fun decodeHistory(encoded: String?): List<ProcessingTimeoutEvent> {
        if (encoded.isNullOrBlank()) return emptyList()
        return runCatching {
            val values = JSONArray(encoded)
            (0 until values.length()).mapNotNull { index ->
                values.optJSONObject(index)?.let(ProcessingTimeoutEvent::fromJson)
            }.takeLast(MAX_EVENTS)
        }.getOrDefault(emptyList())
    }

    private fun postNotification(context: Context, event: ProcessingTimeoutEvent) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT),
        )
        val pending = PendingIntent.getActivity(
            context,
            NOTIFICATION_ID_BASE + event.sequence,
            Intent(context, EditorActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val project = event.sourceName?.let { " for $it" }.orEmpty()
        val message = "Android stopped ${event.operation.label}$project after its background time limit."
        runCatching {
            manager.notify(
                NOTIFICATION_ID_BASE + event.sequence,
                NotificationCompat.Builder(context, CHANNEL_ID)
                    .setSmallIcon(android.R.drawable.stat_notify_error)
                    .setContentTitle("VolleyCut processing stopped")
                    .setContentText(message)
                    .setStyle(NotificationCompat.BigTextStyle().bigText("$message ${event.detail}"))
                    .setContentIntent(pending)
                    .setAutoCancel(true)
                    .setCategory(NotificationCompat.CATEGORY_ERROR)
                    .build(),
            )
        }
    }
}
