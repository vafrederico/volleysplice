package com.volleycut.nativeanalysis

import android.content.Intent
import android.app.Activity
import android.app.Instrumentation
import android.net.Uri
import android.os.SystemClock
import android.util.Base64
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.printToString
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File
import java.util.concurrent.atomic.AtomicReference

/** Mounts the production editor with synthetic local media and saved draft bytes. */
class EditorDraftRecoveryInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun failedAutosaveShowsRetryAndPreservesLatestEditAfterRecovery() = withProject { store, target, _ ->
        val original = target.readBytes()
        val blocker = File(target.parentFile, "${target.name}.tmp").apply { mkdirs() }
        try {
            launchEditor { scenario ->
                compose.waitUntil(10_000) { compose.onAllNodesWithTag("draft-save-error").fetchSemanticsNodes().isNotEmpty() }
                compose.onNodeWithTag("draft-save-error").assertIsDisplayed()
                compose.onNodeWithTag("draft-save-retry").assertIsEnabled()
                compose.onNodeWithTag("draft-save-backup").assertIsEnabled()
                // Export tests can leave a Play rating prompt pending in this
                // shared debug install; dismiss it through the normal UI.
                if (compose.onAllNodesWithText("Not now").fetchSemanticsNodes().isNotEmpty()) {
                    compose.onNodeWithText("Not now").performClick()
                }
                compose.onNodeWithText("Remove").performScrollTo().performClick()
                compose.waitForIdle()
                compose.onNodeWithText("✓ Remove").assertExists()
                SystemClock.sleep(500)
                assertArrayEquals(original, target.readBytes())
                assertTrue(blocker.delete())
                compose.onNodeWithTag("draft-save-retry").performClick()
                compose.waitForIdle()
                compose.waitUntil(10_000) {
                    (store.load() as? EditorDraftLoadResult.Restored)?.draft?.cuts?.firstOrNull()?.included == false
                }
                compose.onNodeWithTag("draft-save-error").assertDoesNotExist()
                scenario.recreate()
                compose.waitForIdle()
                assertFalse((store.load() as EditorDraftLoadResult.Restored).draft.cuts.first().included)
            }
        } finally { blocker.delete() }
    }

    @Test fun failedImportedDraftWriteDoesNotPublishOrSelectProject() = withProject { store, target, project ->
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val draft = (store.load() as EditorDraftLoadResult.Restored).draft
        val text = ModelFeedbackExporter.createBundle(project, draft, EditorMath.finalIntervals(draft), null, null).toString()
        val beforeIds = NativeProjectStore.list(context).map { it.id }.toSet()
        val selection = NativeProjectStore.selectedId(context)
        val directory = checkNotNull(target.parentFile)
        val mode = android.system.Os.stat(directory.absolutePath).st_mode and 0xFFF
        try {
            android.system.Os.chmod(directory.absolutePath, mode and android.system.OsConstants.S_IWUSR.inv())
            assertThrows(EditorDraftSaveException::class.java) { ModelFeedbackImporter.import(context, text) }
            assertEquals(beforeIds, NativeProjectStore.list(context).map { it.id }.toSet())
            assertEquals(selection, NativeProjectStore.selectedId(context))
        } finally { android.system.Os.chmod(directory.absolutePath, mode) }
    }

    @Test fun truncatedDraftSurvivesMountRecreationAndCancelledReset() = withProject { store, target, _ ->
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

    @Test fun futureVersionSurvivesUntilConfirmedResetThenNewDraftSaves() = withProject { store, target, _ ->
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

    @Test fun retryRestoresRepairedDraftAndNormalAutosaveKeepsEdits() = withProject { store, target, _ ->
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

    @Test fun backupImportFailuresProgressAndCancellationAreVisibleAndRetryable() = withProject { _, target, _ ->
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = "content://${instrumentation.context.packageName}.draft-recovery-fixtures"
        instrumentation.targetContext.contentResolver.call(Uri.parse(provider), "reset", null, null)
        val original = "{\"version\":8,\"cuts\":[".toByteArray()
        target.writeBytes(original)
        withPickerResult { result ->
            launchEditor {
                for (failure in listOf("invalid", "unsupported", "unreadable")) {
                    result.set(Instrumentation.ActivityResult(Activity.RESULT_OK,
                        Intent().setData(Uri.parse("$provider/$failure"))))
                    compose.onNodeWithTag("draft-recovery-import").performScrollTo().performClick()
                    compose.waitUntil(10_000) {
                        compose.onAllNodesWithTag("draft-recovery-import-error").fetchSemanticsNodes().isNotEmpty()
                    }
                    compose.onNodeWithTag("draft-recovery-import-error").performScrollTo().assertIsDisplayed()
                    compose.onNodeWithTag("draft-recovery-import").assertIsEnabled()
                    assertArrayEquals(original, target.readBytes())
                }
                screenshot("import-error")
                result.set(Instrumentation.ActivityResult(Activity.RESULT_CANCELED, null))
                compose.onNodeWithTag("draft-recovery-import").performScrollTo().performClick()
                compose.onNodeWithTag("draft-recovery-import-error").assertDoesNotExist()
                compose.onNodeWithTag("draft-recovery-import-status").assertExists()

                result.set(Instrumentation.ActivityResult(Activity.RESULT_OK,
                    Intent().setData(Uri.parse("$provider/slow"))))
                compose.onNodeWithTag("draft-recovery-import").performScrollTo().performClick()
                compose.waitUntil(10_000) {
                    compose.onAllNodesWithTag("draft-recovery-import-progress").fetchSemanticsNodes().isNotEmpty()
                }
                compose.onNodeWithTag("draft-recovery-import-status").performScrollTo().assertIsDisplayed()
                screenshot("import-progress")
                compose.onNodeWithTag("draft-recovery-cancel-import").performScrollTo().performClick()
                instrumentation.targetContext.contentResolver.call(Uri.parse(provider), "release", null, null)
                SystemClock.sleep(600) // Late provider output must not turn cancellation into an error.
                compose.onNodeWithTag("draft-recovery-import-error").assertDoesNotExist()
                compose.onNodeWithTag("draft-recovery-import").assertIsEnabled()
                assertArrayEquals(original, target.readBytes())
            }
        }
    }

    @Test fun recoveredBackupRelinksOriginalVideoAndReopensWithoutDiscardingUnreadableDraft() =
        recoverBackup(legacyImport = false)

    @Test fun legacyImportedBackupCanStillRecoverToOriginalVideo() = recoverBackup(legacyImport = true)

    @Test fun silentVideoBackupRelinksOriginalVideoAndPreservesReview() =
        recoverBackup(legacyImport = false, silentVideo = true)

    private fun recoverBackup(legacyImport: Boolean, silentVideo: Boolean = false) =
        withProject(silentVideo) { store, target, project ->
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val savedDraft = (store.load() as EditorDraftLoadResult.Restored).draft
            val backup = File(context.filesDir, "draft-recovery-backup.json").apply {
                writeText(ModelFeedbackExporter.createBundle(
                    project, savedDraft, EditorMath.finalIntervals(savedDraft), null, null,
                ).toString())
            }
            val original = "{\"version\":8,\"cuts\":[".toByteArray()
            target.writeBytes(original)
            var recovered: NativeProject? = null
            val recoverySeeds = mutableListOf<EditorSeed>()
            try {
                withPickerResult { result ->
                    result.set(Instrumentation.ActivityResult(Activity.RESULT_OK, Intent().setData(Uri.fromFile(backup))))
                    launchEditor { scenario ->
                        compose.onNodeWithTag("draft-recovery-import").performScrollTo().performClick()
                        compose.waitUntil(10_000) {
                            NativeProjectStore.selectedId(context)?.startsWith("import-") == true
                        }
                        recovered = checkNotNull(NativeProjectStore.get(context, checkNotNull(NativeProjectStore.selectedId(context))))
                        recoverySeeds.add(checkNotNull(checkNotNull(recovered).editorSeed()))
                        if (legacyImport) {
                            // Emulate an imported project persisted before namespaces existed.
                            val importedSeed = checkNotNull(checkNotNull(recovered).editorSeed())
                            val draft = (EditorDraftStore(context, importedSeed).load() as EditorDraftLoadResult.Restored).draft
                            recovered = checkNotNull(recovered).copy(draftNamespace = null)
                            val legacySeed = checkNotNull(checkNotNull(recovered).editorSeed())
                            recoverySeeds.add(legacySeed)
                            EditorDraftStore(context, legacySeed).save(draft.copy(sourceRevision = legacySeed.sourceRevision))
                            NativeProjectStore.save(context, checkNotNull(recovered))
                            scenario.recreate()
                            compose.waitForIdle()
                        }
                        result.set(Instrumentation.ActivityResult(Activity.RESULT_OK, Intent().setData(Uri.parse(project.source.uri))))
                        compose.waitUntil(10_000) {
                            compose.onAllNodesWithText("Re-link video").fetchSemanticsNodes().isNotEmpty()
                        }
                        compose.onNodeWithText("Re-link video").performScrollTo().performClick()
                        try {
                            compose.waitUntil(10_000) {
                                NativeProjectStore.get(context, checkNotNull(recovered).id)?.source?.uri == project.source.uri
                            }
                        } catch (failure: Throwable) {
                            screenshot("relink-failure")
                            throw AssertionError(compose.onRoot().printToString(), failure)
                        }
                        val relinked = checkNotNull(NativeProjectStore.get(context, checkNotNull(recovered).id))
                        recovered = relinked
                        val relinkedSeed = checkNotNull(relinked.editorSeed())
                        recoverySeeds.add(relinkedSeed)
                        assertEquals(checkNotNull(project.editorSeed()).sourceRevision,
                            relinkedSeed.copy(draftNamespace = null).sourceRevision)
                        assertNotEquals(checkNotNull(project.editorSeed()).sourceRevision, relinkedSeed.sourceRevision)
                        scenario.recreate()
                        compose.waitForIdle()
                        compose.onNodeWithTag("draft-recovery-retry").assertDoesNotExist()
                        compose.onNodeWithText("Re-link video").assertDoesNotExist()
                        compose.onNodeWithContentDescription("Play video").performScrollTo().assertIsDisplayed()
                        val restored = (EditorDraftStore(context, relinkedSeed).load() as EditorDraftLoadResult.Restored).draft
                        assertEquals(savedDraft.cuts, restored.cuts)
                        assertEquals(savedDraft.beforePaddingMs, restored.beforePaddingMs)
                        val reexported = ModelFeedbackExporter.createBundle(
                            relinked, restored, EditorMath.finalIntervals(restored), null, null,
                        )
                        assertEquals(savedDraft.cuts, ModelFeedbackImporter.parse(reexported.toString()).draft.cuts)
                        assertArrayEquals(original, target.readBytes())
                    }
                }
            } finally {
                recovered?.let { NativeProjectStore.delete(context, it) }
                recoverySeeds.forEach { EditorDraftStore(context, it).clear() }
                backup.delete()
            }
        }

    private fun withPickerResult(block: (AtomicReference<Instrumentation.ActivityResult>) -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val result = AtomicReference(Instrumentation.ActivityResult(Activity.RESULT_CANCELED, null))
        val monitor = object : Instrumentation.ActivityMonitor() {
            override fun onStartActivity(intent: Intent): Instrumentation.ActivityResult? =
                if (intent.action == Intent.ACTION_OPEN_DOCUMENT) result.get() else null
        }
        instrumentation.addMonitor(monitor)
        try { block(result) } finally { instrumentation.removeMonitor(monitor) }
    }

    private fun screenshot(name: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.getExternalFilesDir(null), "draft-recovery-validation")
            .apply { mkdirs() }
        UiDevice.getInstance(instrumentation).takeScreenshot(File(directory, "$name.png"))
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

    private fun withProject(silentVideo: Boolean = false, block: (EditorDraftStore, File, NativeProject) -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val previousSelection = NativeProjectStore.selectedId(context)
        val video = File(context.filesDir, "draft-recovery-synthetic.mp4")
        // Exercise both synthetic AAC and video-only source metadata through
        // persistence, feedback import and the actual source-picker relink.
        val fixture = if (silentVideo) "overlay-fixture.mp4.b64" else "draft-recovery-fixture.mp4.b64"
        video.writeBytes(Base64.decode(instrumentation.context.assets.open(fixture)
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
            block(store, target, project)
        } finally {
            NativeProjectStore.delete(context, project)
            NativeProjectStore.setSelectedId(context, previousSelection)
            video.delete()
        }
    }
}
