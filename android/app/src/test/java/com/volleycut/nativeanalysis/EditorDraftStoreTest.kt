package com.volleycut.nativeanalysis

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class EditorDraftStoreTest {
    @get:Rule val temporary = TemporaryFolder()
    private val seed = EditorSeed(
        "content://synthetic/draft-test", "synthetic.mp4", 12_000, 320, 240, 0,
        listOf(SeedRange(3_000, 6_000, .9f, ProductionEnsemble.BOTH_MODELS)),
    )
    private val target get() = File(temporary.root, "editor-drafts/${seed.sourceRevision}.json")
    private fun store() = EditorDraftStore(temporary.root, seed)

    private fun goodBytes(): ByteArray {
        store().save(EditorMath.newDraft(seed).copy(beforePaddingMs = 1_000, playbackRate = 2f))
        return target.readBytes()
    }

    private fun assertProtected(bytes: ByteArray) {
        checkNotNull(target.parentFile).mkdirs()
        target.writeBytes(bytes)
        val opened = store()
        assertEquals(EditorDraftLoadResult.Unreadable, opened.load())
        // Reproduce the old default-fallback autosave, including a fresh writer
        // such as import. Both must refuse to replace the exact original bytes.
        listOf(opened, store()).forEach { writer ->
            assertThrows(UnreadableEditorDraftException::class.java) {
                writer.save(EditorMath.newDraft(seed))
            }
            assertArrayEquals(bytes, target.readBytes())
        }
    }

    @Test fun missingDraftCanSaveAndGoodDraftRestoresAndSavesEdits() {
        assertEquals(EditorDraftLoadResult.Missing, store().load())
        goodBytes()
        val restored = (store().load() as EditorDraftLoadResult.Restored).draft
        assertEquals(1_000L, restored.beforePaddingMs)
        assertEquals(2f, restored.playbackRate)
        store().save(restored.copy(beforePaddingMs = 500))
        assertEquals(500L, (store().load() as EditorDraftLoadResult.Restored).draft.beforePaddingMs)
    }

    @Test fun truncatedJsonSurvivesFallbackAndNewWriter() =
        assertProtected("{\"version\":8,\"cuts\":[".toByteArray())

    @Test fun undecodableBytesSurviveExactly() =
        assertProtected(byteArrayOf(123, 34, -1, -2, 0))

    @Test fun futureVersionSurvivesFallback() {
        val json = JSONObject(String(goodBytes())).put("version", EDITOR_DRAFT_VERSION + 1)
        assertProtected(json.toString().toByteArray())
    }

    @Test fun wrongSourceRevisionIsProtected() {
        val json = JSONObject(String(goodBytes())).put("sourceRevision", "different-synthetic-revision")
        assertProtected(json.toString().toByteArray())
    }

    @Test fun invalidDraftValuesAreProtected() {
        val json = JSONObject(String(goodBytes())).put("beforePaddingMs", -1)
        assertProtected(json.toString().toByteArray())
    }

    @Test fun unreadablePathAndFailedResetStayProtected() {
        target.mkdirs()
        val retained = File(target, "synthetic-retained-bytes").apply { writeText("unchanged") }
        assertEquals(EditorDraftLoadResult.Unreadable, store().load())
        assertThrows(UnreadableEditorDraftException::class.java) { store().save(EditorMath.newDraft(seed)) }
        assertEquals(EditorDraftLoadResult.Unreadable, store().resetUnreadable())
        assertEquals("unchanged", retained.readText())
    }

    @Test fun nonDirectoryParentIsNotMissing() {
        File(temporary.root, "editor-drafts").writeText("unchanged")
        assertEquals(EditorDraftLoadResult.Unreadable, store().load())
        assertThrows(UnreadableEditorDraftException::class.java) { store().save(EditorMath.newDraft(seed)) }
    }

    @Test fun explicitResetAllowsNewDraft() {
        assertProtected("{\"version\":8".toByteArray())
        assertEquals(EditorDraftLoadResult.Missing, store().resetUnreadable())
        assertFalse(target.exists())
        store().save(EditorMath.newDraft(seed))
        assertTrue(store().load() is EditorDraftLoadResult.Restored)
    }

    @Test fun retryCanRestoreRepairedBytesAndResetDoesNotDiscardNowReadableEdits() {
        val valid = goodBytes()
        assertProtected("{\"version\":8".toByteArray())
        target.writeBytes(valid)
        assertTrue(store().load() is EditorDraftLoadResult.Restored)
        assertTrue(store().resetUnreadable() is EditorDraftLoadResult.Restored)
        assertArrayEquals(valid, target.readBytes())
    }

    @Test fun relinkRefusesUnreadableSourceAndPreservesUnreadableDestination() {
        val valid = goodBytes()
        val replacement = seed.copy(sourceUri = "content://synthetic/relinked")
        val newTarget = File(target.parentFile, "${replacement.sourceRevision}.json")
        val invalid = "{\"version\":8".toByteArray()
        assertProtected(invalid)
        assertThrows(IllegalStateException::class.java) {
            EditorDraftStore.relink(temporary.root, seed, replacement)
        }
        assertFalse(newTarget.exists())
        assertArrayEquals(invalid, target.readBytes())
        target.writeBytes(valid)
        newTarget.writeBytes(invalid)
        assertThrows(UnreadableEditorDraftException::class.java) {
            EditorDraftStore.relink(temporary.root, seed, replacement)
        }
        assertArrayEquals(invalid, newTarget.readBytes())
        assertArrayEquals(valid, target.readBytes())
    }

    @Test fun relinkCopiesGoodEditsToNewIdentity() {
        goodBytes()
        val replacement = seed.copy(sourceUri = "content://synthetic/relinked")
        EditorDraftStore.relink(temporary.root, seed, replacement)
        val restored = (EditorDraftStore(temporary.root, replacement).load() as EditorDraftLoadResult.Restored).draft
        assertEquals(replacement.sourceRevision, restored.sourceRevision)
        assertEquals(1_000L, restored.beforePaddingMs)
    }
}
