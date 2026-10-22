package com.volleycut.nativeanalysis

import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.util.Base64
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Mounts the production editor with synthetic local media and saved draft bytes. */
class EditorDraftRecoveryInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun truncatedDraftSurvivesMountRecreationAndCancelledReset() = withProject { store, target ->
        val original = "{\"version\":8,\"cuts\":[".toByteArray()
        target.writeBytes(original)
        launchEditor { scenario ->
            assertRecoveryPreserves(target, original)
            scenario.recreate()
            assertRecoveryPreserves(target, original)
            compose.onNodeWithTag("draft-recovery-reset").performScrollTo().performClick()
            compose.onNodeWithText("Keep saved edits").performClick()
            assertRecoveryPreserves(target, original)
            assertEquals(EditorDraftLoadResult.Unreadable, store.load())
        }
    }

    @Test fun futureVersionSurvivesUntilConfirmedResetThenNewDraftSaves() = withProject { store, target ->
        val original = JSONObject(target.readText()).put("version", EDITOR_DRAFT_VERSION + 1)
            .toString().toByteArray()
        target.writeBytes(original)
        launchEditor { scenario ->
            assertRecoveryPreserves(target, original)
            compose.onNodeWithTag("draft-recovery-reset").performScrollTo().performClick()
            assertArrayEquals(original, target.readBytes())
            compose.onNodeWithTag("draft-recovery-confirm-reset").performClick()
            compose.waitUntil(10_000) { store.load() is EditorDraftLoadResult.Restored }
            compose.onNodeWithTag("draft-recovery-retry").assertDoesNotExist()
            scenario.recreate()
            compose.waitForIdle()
            assertTrue(store.load() is EditorDraftLoadResult.Restored)
        }
    }

    @Test fun retryRestoresRepairedDraftAndNormalAutosaveKeepsEdits() = withProject { store, target ->
        val valid = target.readBytes()
        target.writeText("{\"version\":8")
        launchEditor { scenario ->
            compose.onNodeWithTag("draft-recovery-retry").assertIsDisplayed()
            target.writeBytes(valid)
            compose.onNodeWithTag("draft-recovery-retry").performClick()
            compose.waitForIdle()
            SystemClock.sleep(600) // Allow the production 200 ms autosave to execute.
            compose.onNodeWithTag("draft-recovery-retry").assertDoesNotExist()
            assertEquals(0L, (store.load() as EditorDraftLoadResult.Restored).draft.beforePaddingMs)
            scenario.recreate()
            compose.waitForIdle()
            assertEquals(0L, (store.load() as EditorDraftLoadResult.Restored).draft.beforePaddingMs)
        }
    }

    private fun assertRecoveryPreserves(target: File, original: ByteArray) {
        compose.waitForIdle()
        compose.onNodeWithTag("draft-recovery-retry").assertIsDisplayed()
        compose.onNodeWithTag("draft-recovery-import").assertExists()
        SystemClock.sleep(600) // Opening must stay safe beyond the old autosave deadline.
        assertArrayEquals(original, target.readBytes())
    }

    private fun launchEditor(block: (ActivityScenario<EditorActivity>) -> Unit) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        ActivityScenario.launch<EditorActivity>(Intent(context, EditorActivity::class.java)).use(block)
    }

    private fun withProject(block: (EditorDraftStore, File) -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val previousSelection = NativeProjectStore.selectedId(context)
        val video = File(context.filesDir, "draft-recovery-synthetic.mp4")
        video.writeBytes(Base64.decode(instrumentation.context.assets.open("overlay-fixture.mp4.b64")
            .bufferedReader().use { it.readText() }, Base64.DEFAULT))
        val input = EditorSeed(Uri.fromFile(video).toString(), video.name, 1_000, 320, 240, 0,
            listOf(SeedRange(250, 750, .9f, ProductionEnsemble.BOTH_MODELS)))
        val project = NativeProjectStore.fromSeed(context, input)
        NativeProjectStore.save(context, project)
        NativeProjectStore.setSelectedId(context, project.id)
        GuidedTourStore.write(context, "done")
        val seed = checkNotNull(project.editorSeed())
        val store = EditorDraftStore(context, seed)
        val target = File(context.filesDir, "editor-drafts/${seed.sourceRevision}.json")
        target.delete() // Only this test's synthetic draft.
        store.save(EditorMath.newDraft(seed).copy(beforePaddingMs = 0))
        try {
            block(store, target)
        } finally {
            NativeProjectStore.delete(context, project)
            NativeProjectStore.setSelectedId(context, previousSelection)
            video.delete()
        }
    }
}
