package com.volleycut.nativeanalysis

import org.json.JSONObject

/** Additive feedback-v3 corrected-range metadata, deliberately outside the opaque UI envelope. */
internal object ExplicitRallyGroups {
    private val groupPattern = Regex("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

    fun decode(cut: JSONObject): String? {
        if (!cut.has("editableGroupId")) return null
        val value = cut.get("editableGroupId")
        require(value is String && groupPattern.matches(value)) { "Invalid editableGroupId" }
        return value
    }

    fun validate(cuts: List<EditableCut>) {
        require(cuts.map { it.id }.distinct().size == cuts.size) { "Duplicate cut identity" }
        cuts.filter { it.editableGroupId != null }.forEach {
            require(it.id.isNotBlank() && it.id.length <= 256 && groupPattern.matches(it.editableGroupId!!)) {
                "Invalid explicit rally identity"
            }
        }
        require(cuts.count { it.editableGroupId != null } <= 2048) { "Too many explicit rally members" }
    }

    /** Reset choices while retaining explicit split topology, including manual parts and cut IDs. */
    fun reset(seed: EditorSeed, draft: EditorDraft): EditorDraft {
        val defaults = EditorMath.newDraft(seed)
        val retained = draft.cuts.filter { it.editableGroupId != null }
        if (retained.isEmpty()) return defaults
        val ids = retained.mapTo(hashSetOf()) { it.id }
        val cuts = defaults.cuts.filterNot { it.id in ids } + retained.map { cut ->
            cut.copy(included = true,
                keepStartMs = if (cut.origin == CutOrigin.MANUAL) cut.coreStartMs
                    else (cut.coreStartMs - defaults.beforePaddingMs).coerceAtLeast(seed.gameStartMs),
                keepEndMs = if (cut.origin == CutOrigin.MANUAL) cut.coreEndMs
                    else (cut.coreEndMs + defaults.afterPaddingMs).coerceAtMost(seed.gameEndMs))
        }
        validate(cuts)
        return EditorMath.reconcileTouchedCuts(defaults.copy(cuts = cuts), seed)
    }
}
