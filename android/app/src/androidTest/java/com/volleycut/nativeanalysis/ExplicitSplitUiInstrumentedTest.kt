package com.volleycut.nativeanalysis

import android.content.Intent
import android.net.Uri
import android.util.Base64
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.test.click
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTouchInput
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Real production editor split/select/trim/remove, backed by a synthetic one-second MP4. */
class ExplicitSplitUiInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()
    @Test fun highConfidenceSplitSelectTrimRemoveAndReopen() = exercise(false)
    @Test fun manualSplitSelectTrimRemoveAndReopen() = exercise(true)

    private fun exercise(manual: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val kind = if (manual) "manual" else "inferred"
        val video = File(context.filesDir, "synthetic-split-$kind.mp4")
        video.writeBytes(Base64.decode(instrumentation.context.assets.open("overlay-fixture.mp4.b64")
            .bufferedReader().use { it.readText() }, Base64.DEFAULT))
        val input = EditorSeed(Uri.fromFile(video).toString(), video.name, 1_000, 320, 240, 0,
            listOf(SeedRange(200, 900, .95f, ProductionEnsemble.BOTH_MODELS)))
        val project = NativeProjectStore.fromSeed(context, input)
        NativeProjectStore.save(context, project)
        NativeProjectStore.setSelectedId(context, project.id)
        GuidedTourStore.write(context, "done")
        val seed = requireNotNull(project.editorSeed())
        val store = EditorDraftStore(context, seed)
        var initial = EditorMath.applyPadding(EditorMath.newDraft(seed), 0, 0, 1_000)
        if (manual) initial = initial.copy(cuts = initial.cuts.map {
            it.copy(id = "M001", origin = CutOrigin.MANUAL)
        })
        store.save(initial)
        val leftId = initial.cuts.single().id
        val device = UiDevice.getInstance(instrumentation)
        device.wakeUp()
        val evidence = File(context.getExternalFilesDir(null), "explicit-split-$kind").apply { mkdirs() }
        fun saved() = requireNotNull(store.load())
        fun seek(fraction: Float) {
            // The phone overview has one canvas per half of the source timeline.
            val half = if (fraction >= .5f) 1 else 0
            compose.onAllNodesWithContentDescription("Game timeline")[half].performScrollTo().performTouchInput {
                click(Offset(width * (fraction * 2 - half), height - 2f))
            }
        }
        ActivityScenario.launch<EditorActivity>(Intent(context, EditorActivity::class.java)).use { scenario ->
            compose.waitForIdle()
            seek(.55f)
            compose.onNodeWithText("Split at playhead").performScrollTo().performClick()
            compose.waitUntil(10_000) { saved().cuts.size == 2 }
            val split = saved()
            val right = split.cuts.single { it.id != leftId }
            assertEquals(2, EditorMath.editableRallyGroups(split.cuts, EditorMath.finalIntervals(split), emptyList()).size)
            compose.onNodeWithTag("rally-row-$leftId").performScrollTo().assertExists()
            compose.onNodeWithTag("rally-row-${right.id}").performScrollTo().performClick()
            seek(.75f)
            compose.onNodeWithText("Set rally start here").performScrollTo().performClick()
            compose.waitUntil(10_000) { saved().cuts.single { it.id == right.id }.coreStartMs > right.coreStartMs }
            val trimmed = saved()
            assertEquals(split.cuts.single { it.id == leftId }, trimmed.cuts.single { it.id == leftId })
            assertEquals(right.editableGroupId, trimmed.cuts.single { it.id == right.id }.editableGroupId)
            compose.onNodeWithText("Previous").performScrollTo().performClick()
            compose.onNodeWithText("1 / 2").assertExists()
            compose.onNodeWithText("Next").performScrollTo().performClick()
            compose.onNodeWithText("2 / 2").assertExists()
            compose.onNodeWithText("Remove").performScrollTo().performClick()
            try {
                compose.waitUntil(10_000) { !saved().cuts.single { it.id == right.id }.included }
            } catch (error: Throwable) {
                File(evidence, "removal-failure.json").writeText(editListJson(seed, saved(), EditorMath.finalIntervals(saved())).toString(2))
                device.takeScreenshot(File(evidence, "removal-failure.png"))
                throw error
            }
            val removed = saved()
            assertTrue(removed.cuts.single { it.id == leftId }.included)
            assertEquals(trimmed.cuts.single { it.id == leftId }, removed.cuts.single { it.id == leftId })
            assertEquals(trimmed.cuts.single { it.id == right.id }.copy(included = false), removed.cuts.single { it.id == right.id })
            assertEquals(listOf(leftId), EditorMath.finalIntervals(removed).single().cutIds)
            device.takeScreenshot(File(evidence, "split-trim-remove.png"))
            scenario.recreate()
            compose.waitForIdle()
            compose.onNodeWithTag("rally-row-${right.id}").performScrollTo().performClick()
            assertEquals(removed.cuts, saved().cuts)
            val archive = ModelFeedbackExporter.createBundle(project, saved(), EditorMath.finalIntervals(saved()), null, null)
            File(evidence, "feedback.json").writeText(archive.toString(2))
            assertEquals(removed.cuts, ModelFeedbackImporter.parse(archive.toString()).draft.cuts)
            device.takeScreenshot(File(evidence, "reopened.png"))
        }
    }
}
