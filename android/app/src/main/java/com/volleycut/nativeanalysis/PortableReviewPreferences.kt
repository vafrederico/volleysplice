package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject

/** Bounded transport extension. Keep it as immutable JSON in drafts, separate from active settings. */
internal object PortableReviewPreferences {
    private val booleanKeys = listOf("cutPreviewEnabled", "renderScoreOnPreviewVideo",
        "renderScoreOnFinalVideo", "renderScoreTimeline", "fadeScoreOverlay")

    fun validate(value: Any, cuts: List<EditableCut>, checkIdentities: Boolean = true): JSONObject {
        var nodes = 0
        fun visit(item: Any, depth: Int) {
            require(++nodes <= 4096 && depth <= 16) { "corrections.ui exceeds JSON limits" }
            when (item) {
                JSONObject.NULL, is Boolean -> Unit
                is Number -> require(item.toDouble().isFinite()) { "corrections.ui contains a nonfinite number" }
                is String -> require(item.length <= 16_384) { "corrections.ui string is too long" }
                is JSONArray -> repeat(item.length()) { visit(item.get(it), depth + 1) }
                is JSONObject -> item.keys().forEach { key ->
                    require(key !in setOf("__proto__", "constructor", "prototype")) { "corrections.ui contains an unsafe key" }
                    visit(item.get(key), depth + 1)
                }
                else -> error("corrections.ui must contain JSON values only")
            }
        }
        visit(value, 0)
        require(value is JSONObject) { "corrections.ui must be an object" }
        require(value.toString().toByteArray(Charsets.UTF_8).size <= 65_536) { "corrections.ui exceeds 64 KiB" }
        if (value.has("schemaVersion")) {
            val version = value.get("schemaVersion")
            require(version is Number && version.toDouble().let {
                it >= 1 && it <= 9_007_199_254_740_991.0 && it % 1.0 == 0.0
            }) { "corrections.ui.schemaVersion must be a positive safe integer" }
        }
        val preferences = if (value.has("preferences")) value.get("preferences") else JSONObject()
        require(preferences is JSONObject) { "corrections.ui.preferences must be an object" }
        booleanKeys.forEach { key ->
            require(!preferences.has(key) || preferences.get(key) is Boolean) { "Invalid UI boolean: $key" }
        }
        listOf(Triple("playbackRate", .25, 8.0), Triple("confidenceReviewThreshold", 0.0, 1.0)).forEach { (key, min, max) ->
            if (preferences.has(key)) {
                val number = preferences.get(key)
                require(number is Number && number.toDouble() in min..max) { "Invalid UI number: $key" }
            }
        }
        if (preferences.has("reviewedCutIds")) {
            val ids = preferences.get("reviewedCutIds")
            require(ids is JSONArray && ids.length() <= 2048) { "Invalid reviewed cut IDs" }
            val seen = mutableSetOf<String>()
            repeat(ids.length()) {
                val id = ids.get(it)
                require(id is String && id.isNotEmpty() && id.length <= 256 && seen.add(id) &&
                    (!checkIdentities || cuts.any { cut -> cut.id == id && cut.origin == CutOrigin.INFERRED })) { "Invalid reviewed cut identity" }
            }
        }
        return value
    }

    fun isFuture(ui: JSONObject): Boolean = ui.has("schemaVersion") && ui.getDouble("schemaVersion") > 1

    fun import(draft: EditorDraft, corrections: JSONObject, bundle: JSONObject): EditorDraft {
        val alias = if (bundle.has("editorPreferences")) {
            val preferences = bundle.get("editorPreferences")
            require(preferences is JSONObject) { "editorPreferences must be an object" }
            if (preferences.has("renderScoreOnFinalVideo")) {
                val final = preferences.get("renderScoreOnFinalVideo")
                require(final is Boolean) { "Invalid final score compatibility preference" }
                final
            } else null
        } else null
        if (!corrections.has("ui")) return draft.copy(renderScoreOverlay = alias ?: draft.renderScoreOverlay)
        val ui = validate(corrections.get("ui"), draft.cuts)
        val retained = draft.copy(portableUi = ui.toString())
        if (isFuture(ui)) return retained.copy(renderScoreOverlay = alias ?: draft.renderScoreOverlay)
        val prefs = ui.optJSONObject("preferences") ?: JSONObject()
        val v1 = ui.has("schemaVersion")
        return retained.copy(
            finalPreviewEnabled = prefs.optBoolean("cutPreviewEnabled", draft.finalPreviewEnabled),
            renderScoreOverlay = prefs.optBoolean("renderScoreOnFinalVideo", alias ?: if (v1) true else draft.renderScoreOverlay),
            renderScoreTimeline = prefs.optBoolean("renderScoreTimeline", if (v1) false else draft.renderScoreTimeline),
            fadeScoreOverlay = prefs.optBoolean("fadeScoreOverlay", if (v1) false else draft.fadeScoreOverlay),
            fadeScoreOverlayExplicit = v1 || prefs.has("fadeScoreOverlay"),
            playbackRate = prefs.optDouble("playbackRate", 1.0),
            confidenceReviewThreshold = prefs.optDouble("confidenceReviewThreshold", .7).toFloat(),
            reviewedCutIds = prefs.optJSONArray("reviewedCutIds")?.let { ids ->
                (0 until ids.length()).mapTo(linkedSetOf()) { ids.getString(it) }
            }.orEmpty(),
        )
    }

    fun export(draft: EditorDraft): JSONObject {
        val ui = draft.portableUi?.let { validate(JSONObject(it), draft.cuts, false) } ?: JSONObject()
        if (isFuture(ui)) return ui
        val prefs = ui.optJSONObject("preferences") ?: JSONObject()
        // The confidence slider uses floats. Retain its exact imported JSON number until changed.
        fun wireNumber(key: String, active: Float): Double =
            (prefs.opt(key) as? Number)?.toDouble()?.takeIf { it.toFloat() == active } ?: active.toString().toDouble()
        val threshold = wireNumber("confidenceReviewThreshold", draft.confidenceReviewThreshold)
        prefs.put("cutPreviewEnabled", draft.finalPreviewEnabled)
        // Android has no independent web preview control. Its wire value stays opaque.
        if (!prefs.has("renderScoreOnPreviewVideo")) prefs.put("renderScoreOnPreviewVideo", true)
        prefs.put("renderScoreOnFinalVideo", draft.renderScoreOverlay)
        prefs.put("renderScoreTimeline", draft.renderScoreTimeline)
        prefs.put("fadeScoreOverlay", draft.fadeScoreOverlay)
        prefs.put("playbackRate", draft.playbackRate)
        prefs.put("confidenceReviewThreshold", threshold)
        prefs.put("reviewedCutIds", JSONArray(draft.reviewedCutIds.sorted()))
        ui.put("schemaVersion", 1)
        ui.put("preferences", prefs)
        return validate(ui, draft.cuts)
    }

    fun onOpen(draft: EditorDraft, deviceFade: Boolean): EditorDraft = draft.copy(
        fadeScoreOverlay = if (draft.fadeScoreOverlayExplicit) draft.fadeScoreOverlay else deviceFade,
        fadeScoreOverlayExplicit = true,
    )

    fun reset(seed: EditorSeed, draft: EditorDraft): EditorDraft {
        val ui = draft.portableUi?.let { JSONObject(it) }
        if (ui != null && !isFuture(ui)) {
            // Clear supported transport values before applying reset defaults, including numbers
            // which happen to round to the same native float. The web-only preview flag survives.
            val prefs = ui.optJSONObject("preferences")
            (booleanKeys - "renderScoreOnPreviewVideo" + listOf("playbackRate", "confidenceReviewThreshold", "reviewedCutIds"))
                .forEach { prefs?.remove(it) }
        }
        return ExplicitRallyGroups.reset(seed, draft).copy(portableUi = ui?.toString(), fadeScoreOverlayExplicit = true)
    }

    fun notice(draft: EditorDraft): String = if (draft.portableUi?.let { isFuture(JSONObject(it)) } == true) {
        "This project contains a newer preferences version. Those settings are preserved but not applied; changes here stay local."
    } else ""

    fun playbackRates(active: Double): List<Double> = (listOf(1.0, 1.5, 2.0, 4.0, 8.0) + active).distinct().sorted()
    fun playbackRateLabel(rate: Double): String = "${rate.toString().removeSuffix(".0")}x"
}
