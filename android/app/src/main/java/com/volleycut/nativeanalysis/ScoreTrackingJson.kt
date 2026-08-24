package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject

internal object ScoreTrackingJson {
    fun decodeWire(json: JSONObject, durationMs: Long): ScoreTracking? = runCatching {
        val sourceVersion = json.getInt("version")
        require(sourceVersion in 1..SCORE_TRACKING_SCHEMA_VERSION)
        val serves = json.getJSONArray("serveMarkers")
        val switches = json.getJSONArray("sideSwitchMarkers")
        val removed = json.optJSONArray("removedModelMarkerIds") ?: JSONArray()
        val removedIds = removed.uniqueStrings()
        ScoreTracking(
            version = SCORE_TRACKING_SCHEMA_VERSION,
            enabled = json.getBoolean("enabled"),
            team1Name = json.getString("team1Name"),
            team2Name = json.getString("team2Name"),
            serveMarkers = List(serves.length()) { index -> serves.getJSONObject(index).let {
                ServeMarker(
                    it.getString("id"), secondsToMs(it.getDouble("timestamp")),
                    ServingSide.fromWireName(it.getString("side")) ?: error("Invalid side"),
                    ServeMarkerOrigin.fromWireName(it.getString("origin")) ?: error("Invalid origin"),
                    it.optServingSide("modelSide"),
                    it.getBoolean("ignorePreviousPoint"), it.optNullableString("rallyId"),
                )
            } },
            sideSwitchMarkers = List(switches.length()) { index -> switches.getJSONObject(index).let {
                SideSwitchMarker(
                    it.getString("id"), secondsToMs(it.getDouble("timestamp")),
                    ServeMarkerOrigin.fromWireName(it.optString("origin", "manual"))
                        ?: error("Invalid origin"),
                    it.optNullableDouble("modelConfidence"),
                    it.optNullableString("modelEventId"),
                    it.optJSONArray("rallyIds")?.uniqueStrings().orEmpty(),
                )
            } },
            removedModelMarkerIds = removedIds.toSet(),
        ).takeIf { validate(it, durationMs) }
    }.getOrNull()

    fun encodeWire(value: ScoreTracking) = JSONObject().apply {
        put("version", value.version)
        put("enabled", value.enabled)
        put("team1Name", value.team1Name)
        put("team2Name", value.team2Name)
        put("serveMarkers", JSONArray().apply {
            value.serveMarkers.forEach { marker -> put(JSONObject().apply {
                put("id", marker.id)
                put("timestamp", marker.timestampMs / 1_000.0)
                put("side", marker.side.wireName)
                put("origin", marker.origin.wireName)
                marker.modelSide?.let { put("modelSide", it.wireName) }
                put("ignorePreviousPoint", marker.ignorePreviousPoint)
                marker.rallyId?.let { put("rallyId", it) }
            }) }
        })
        put("sideSwitchMarkers", JSONArray().apply {
            value.sideSwitchMarkers.forEach { marker -> put(JSONObject().apply {
                put("id", marker.id)
                put("timestamp", marker.timestampMs / 1_000.0)
                put("origin", marker.origin.wireName)
                marker.modelConfidence?.let { put("modelConfidence", it) }
                marker.modelEventId?.let { put("modelEventId", it) }
                if (marker.rallyIds.isNotEmpty()) put("rallyIds", JSONArray(marker.rallyIds))
            }) }
        })
        put("removedModelMarkerIds", JSONArray(value.removedModelMarkerIds.sorted()))
    }

    fun encode(value: ScoreTracking) = JSONObject().apply {
        put("version", value.version)
        put("enabled", value.enabled)
        put("team1Name", value.team1Name)
        put("team2Name", value.team2Name)
        put("serveMarkers", JSONArray().apply {
            value.serveMarkers.forEach { marker -> put(JSONObject().apply {
                put("id", marker.id)
                put("timestampMs", marker.timestampMs)
                put("side", marker.side.wireName)
                put("origin", marker.origin.wireName)
                put("modelSide", marker.modelSide?.wireName ?: JSONObject.NULL)
                put("ignorePreviousPoint", marker.ignorePreviousPoint)
                put("rallyId", marker.rallyId ?: JSONObject.NULL)
            }) }
        })
        put("sideSwitchMarkers", JSONArray().apply {
            value.sideSwitchMarkers.forEach { marker -> put(JSONObject().apply {
                put("id", marker.id)
                put("timestampMs", marker.timestampMs)
                put("origin", marker.origin.wireName)
                put("modelConfidence", marker.modelConfidence ?: JSONObject.NULL)
                put("modelEventId", marker.modelEventId ?: JSONObject.NULL)
                put("rallyIds", JSONArray(marker.rallyIds))
            }) }
        })
        put("removedModelMarkerIds", JSONArray(value.removedModelMarkerIds.sorted()))
    }

    fun decode(json: JSONObject, durationMs: Long): ScoreTracking? = runCatching {
        val version = json.optInt("version", 1)
        if (version !in 1..SCORE_TRACKING_SCHEMA_VERSION) return null
        val servesJson = json.optJSONArray("serveMarkers") ?: JSONArray()
        val switchesJson = json.optJSONArray("sideSwitchMarkers") ?: JSONArray()
        val removedJson = if (version >= 2) {
            json.optJSONArray("removedModelMarkerIds") ?: JSONArray()
        } else JSONArray()
        val removedIds = removedJson.uniqueStrings()
        val value = ScoreTracking(
            enabled = json.optBoolean("enabled", true),
            team1Name = json.optString("team1Name", "Team 1"),
            team2Name = json.optString("team2Name", "Team 2"),
            serveMarkers = buildList {
                repeat(servesJson.length()) { index ->
                    val item = servesJson.getJSONObject(index)
                    add(ServeMarker(
                        item.getString("id"),
                        item.getLong("timestampMs"),
                        ServingSide.fromWireName(item.getString("side")) ?: return null,
                        ServeMarkerOrigin.fromWireName(item.getString("origin")) ?: return null,
                        item.optServingSide("modelSide"),
                        item.optBoolean("ignorePreviousPoint"),
                        item.optNullableString("rallyId"),
                    ))
                }
            },
            sideSwitchMarkers = buildList {
                repeat(switchesJson.length()) { index ->
                    val item = switchesJson.getJSONObject(index)
                    add(SideSwitchMarker(
                        item.getString("id"), item.getLong("timestampMs"),
                        if (version >= 3) {
                            ServeMarkerOrigin.fromWireName(item.optString("origin")) ?: return null
                        } else ServeMarkerOrigin.MANUAL,
                        if (version >= 3) item.optNullableDouble("modelConfidence") else null,
                        if (version >= 3) item.optNullableString("modelEventId") else null,
                        if (version >= 3) item.optJSONArray("rallyIds")?.uniqueStrings().orEmpty()
                        else emptyList(),
                    ))
                }
            },
            removedModelMarkerIds = removedIds.toSet(),
        )
        value.takeIf { validate(it, durationMs) }
    }.getOrNull()

    fun validate(value: ScoreTracking, durationMs: Long): Boolean {
        if (value.version != SCORE_TRACKING_SCHEMA_VERSION) {
            return false
        }
        if (value.team1Name.isBlank() || value.team2Name.isBlank()) return false
        if (value.serveMarkers.any {
            it.id.isBlank() || it.timestampMs !in 0..durationMs ||
                it.rallyId?.isBlank() == true
        } || value.sideSwitchMarkers.any {
            it.id.isBlank() || it.timestampMs !in 0..durationMs ||
                (it.modelConfidence != null &&
                    (!it.modelConfidence.isFinite() || it.modelConfidence !in 0.0..1.0)) ||
                it.modelEventId?.isBlank() == true || it.rallyIds.any(String::isBlank) ||
                it.rallyIds.distinct().size != it.rallyIds.size
        } || value.removedModelMarkerIds.any(String::isBlank)) return false
        val ids = value.serveMarkers.map { it.id } + value.sideSwitchMarkers.map { it.id }
        return ids.distinct().size == ids.size && ids.none(value.removedModelMarkerIds::contains)
    }

    private fun JSONObject.optNullableString(key: String): String? =
        if (!has(key) || isNull(key)) null else getString(key)

    private fun JSONObject.optNullableDouble(key: String): Double? =
        if (!has(key) || isNull(key)) null else getDouble(key)

    private fun JSONObject.optServingSide(key: String): ServingSide? {
        val value = optNullableString(key) ?: return null
        return ServingSide.fromWireName(value) ?: error("Invalid serving side")
    }

    private fun JSONArray.uniqueStrings(): List<String> =
        List(length()) { getString(it) }.also { require(it.distinct().size == it.size) }
}
