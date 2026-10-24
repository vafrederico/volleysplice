package com.volleycut.nativeanalysis

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Base64
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.util.UUID

/** Real activity, real local video and the same persisted draft across both versions. */
@Suppress("DEPRECATION")
class RallyReviewUiInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    private class Fixture(
        val store: EditorDraftStore,
        val scenario: ActivityScenario<EditorActivity>,
        val context: Context,
    )

    private fun withProject(many: Boolean = false, removedRallyIds: Set<String> = emptySet(),
        sourceName: String = "Courtside test", legacyEditorPreference: Boolean = false,
        paddingMs: Long = 0, configureDraft: (EditorDraft) -> EditorDraft = { it }, block: (Fixture) -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val preferences = context.getSharedPreferences("desktop_editor_layout", Context.MODE_PRIVATE)
        val previousLayout = preferences.getBoolean("rally_review_layout", true)
        val previousProject = NativeProjectStore.selectedId(context)
        val previousTour = GuidedTourStore.read(context)
        val exportPreferences = context.getSharedPreferences("video-export-settings", Context.MODE_PRIVATE)
        val hadFadePreference = exportPreferences.contains("fadeScoreOverlay")
        val previousFadePreference = exportPreferences.getBoolean("fadeScoreOverlay", false)
        val source = context.filesDir.resolve("courtside-test-" + UUID.randomUUID() + ".mp4")
        source.writeBytes(Base64.decode(instrumentation.context.assets.open("rally-seek-fixture.mp4.b64")
            .bufferedReader().use { it.readText() }.trim(), Base64.DEFAULT))
        val ranges = if (many) (1..12).map {
            SeedRange(it * 1_500L + 200, it * 1_500L + 800, .95f, id = "R" + it.toString().padStart(3, '0'))
        } else listOf(SeedRange(2_000, 5_000, .4f, id = "R001"), SeedRange(12_000, 15_000, .95f, id = "R002"))
        val project = NativeProjectStore.fromSeed(context,
            EditorSeed(Uri.fromFile(source).toString(), sourceName, 20_000, 320, 240, 0, ranges,
                scoreTrackingInitiallyEnabled = true))
        val seed = requireNotNull(project.editorSeed())
        val store = EditorDraftStore(context, seed)
        val serves = if (many) ranges.mapIndexed { index, r ->
            ServeMarker("S$index", r.startMs, ServingSide.NEAR, ServeMarkerOrigin.MANUAL, rallyId = r.id)
        } else listOf(
            ServeMarker("S1", 2_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL, rallyId = "R001"),
            ServeMarker("S2", 12_000, ServingSide.REVIEW, ServeMarkerOrigin.MANUAL, rallyId = "R002"))
        val initialDraft = EditorMath.applyPadding(EditorMath.newDraft(seed), paddingMs, paddingMs, seed.durationMs)
        store.save(configureDraft(initialDraft.copy(
            cuts = initialDraft.cuts.map { it.copy(included = it.id !in removedRallyIds) },
            joinGapMs = 0, scoreTracking = ScoreTracking(team1Name = "Cedar", team2Name = "Harbor", serveMarkers = serves))))
        NativeProjectStore.save(context, project)
        NativeProjectStore.setSelectedId(context, project.id)
        GuidedTourStore.write(context, "dismissed")
        preferences.edit().putBoolean("rally_review_layout", !legacyEditorPreference).commit()
        val scenario = ActivityScenario.launch<EditorActivity>(Intent(context, EditorActivity::class.java))
        // Only the temporary synthetic project is visible; the fixture activity
        // can render while a physical test device is asleep or locked.
        scenario.onActivity {
            it.setTurnScreenOn(true); it.setShowWhenLocked(true)
            it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        try {
            compose.waitUntil(10_000) {
                context.resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_PORTRAIT
            }
            compose.waitUntil(10_000) { compose.onAllNodesWithTag("rally-review-workspace")
                .fetchSemanticsNodes(atLeastOneRootRequired = false).isNotEmpty() }
            block(Fixture(store, scenario, context))
        } finally {
            scenario.close()
            NativeProjectStore.delete(context, project)
            NativeProjectStore.setSelectedId(context, previousProject)
            GuidedTourStore.write(context, previousTour ?: "dismissed")
            preferences.edit().putBoolean("rally_review_layout", previousLayout).commit()
            exportPreferences.edit().apply {
                if (hadFadePreference) putBoolean("fadeScoreOverlay", previousFadePreference)
                else remove("fadeScoreOverlay")
            }.commit()
            source.delete()
        }
    }

    private fun awaitDraft(f: Fixture, predicate: (EditorDraft) -> Boolean) {
        try {
            compose.waitUntil(8_000) { f.store.load()?.let(predicate) == true }
        } catch (error: androidx.compose.ui.test.ComposeTimeoutException) {
            throw AssertionError("Fixture cut bounds: " + f.store.load()?.cuts?.map {
                "${it.coreStartMs}..${it.coreEndMs} retained ${it.keepStartMs}..${it.keepEndMs}"
            }, error)
        }
    }
    private fun pill(text: String) = compose.onNode(hasText(text) and hasClickAction())
    private fun detailScale(f: Fixture, cutId: String = "R001"): RallyTimelineScale {
        val draft = requireNotNull(f.store.load())
        val rallies = rallyGroupsForDraft(draft)
        val selected = rallies.first { cutId in it.cutIds }
        val effectiveIds = EditorMath.finalIntervals(draft).flatMapTo(mutableSetOf()) { it.cutIds }
        return rallyTimelineScale(rallyTimelineContext(rallies, selected, effectiveIds, 0, 20_000), 0, 20_000)
    }
    private fun capture(f: Fixture, name: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        compose.waitForIdle()
        val drawn = java.util.concurrent.CountDownLatch(1)
        f.scenario.onActivity { activity ->
            val view = activity.window.decorView
            view.viewTreeObserver.registerFrameCommitCallback {
                view.postOnAnimation { drawn.countDown() }
            }
            view.invalidate()
        }
        assertTrue(drawn.await(2, java.util.concurrent.TimeUnit.SECONDS))
        instrumentation.waitForIdleSync()
        val device = UiDevice.getInstance(instrumentation)
        device.waitForIdle()
        device.takeScreenshot(f.context.getExternalFilesDir(null)!!.resolve(name))
    }

    /** Check the actual TextureView pixels, not the UI playhead or a mocked Player. */
    private fun awaitVideoFrame(expected: Color) {
        var last = Color.Transparent
        try {
            compose.waitUntil(8_000) {
                val bounds = compose.onNodeWithTag("rally-review-player").fetchSemanticsNode().boundsInRoot
                val screenshot = requireNotNull(InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot())
                val pixel = screenshot.getPixel(bounds.center.x.toInt(), bounds.center.y.toInt())
                screenshot.recycle()
                last = Color(pixel)
                kotlin.math.abs(last.red - expected.red) < .15f &&
                    kotlin.math.abs(last.green - expected.green) < .15f &&
                    kotlin.math.abs(last.blue - expected.blue) < .15f
            }
        } catch (failure: Throwable) {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            UiDevice.getInstance(instrumentation).takeScreenshot(
                instrumentation.targetContext.getExternalFilesDir(null)!!.resolve("courtside-seek-failure.png"))
            throw AssertionError("Expected rendered frame $expected; saw $last", failure)
        }
    }

    private fun awaitPlaying(playing: Boolean) {
        val description = if (playing) "Pause" else "Play"
        compose.waitUntil(8_000) {
            compose.onAllNodesWithContentDescription(description).fetchSemanticsNodes().isNotEmpty()
        }
    }

    @Test
    fun rallyNavigationPreservesPlayingAndPausedState() = withProject { _ ->
        awaitVideoFrame(Color.Red)
        compose.onNodeWithContentDescription("Play").performClick()
        awaitPlaying(true)
        repeat(3) {
            compose.onNodeWithTag("next-rally").performScrollTo().performClick()
            compose.onNodeWithText("Rally 2").assertExists()
            awaitPlaying(true)
            awaitVideoFrame(Color.Blue)
            compose.onNodeWithTag("previous-rally").performScrollTo().performClick()
            compose.onNodeWithText("Rally 1").assertExists()
            awaitPlaying(true)
            awaitVideoFrame(Color.Red)
        }
        compose.onNodeWithContentDescription("Pause").performClick()
        awaitPlaying(false)
        compose.onNodeWithTag("next-rally").performScrollTo().performClick()
        awaitVideoFrame(Color.Blue)
        awaitPlaying(false)
        compose.onNodeWithTag("previous-rally").performScrollTo().performClick()
        awaitVideoFrame(Color.Red)
        awaitPlaying(false)

        compose.onNodeWithText("Keep").performScrollTo().performClick()
        compose.onNodeWithContentDescription("Play").performClick()
        awaitPlaying(true)
        compose.onNodeWithTag("rally-next-review").performScrollTo().performClick()
        compose.onNodeWithText("Rally 2").assertExists()
        awaitPlaying(true)
        awaitVideoFrame(Color.Blue)
        // The previous rally's stop boundary must not pause the new rally.
        compose.waitUntil(5_000) {
            compose.onNodeWithTag("rally-bounds-timeline").fetchSemanticsNode()
                .config[SemanticsProperties.ProgressBarRangeInfo].current > 12_500f
        }
        awaitPlaying(true)
    }

    @Test
    fun repeatedScrubbingAndLayoutSwitchesKeepRendering() = withProject { f ->
        compose.onNodeWithTag("rally-browse-all").performClick()
        repeat(6) { cycle ->
            val timeline = compose.onNodeWithTag("rally-vertical-timeline")
            timeline.performTouchInput { down(Offset(center.x, height * .125f)) }
            awaitVideoFrame(Color.Red)
            timeline.performTouchInput { moveTo(Offset(center.x, height * .375f)) }
            awaitVideoFrame(Color.Green)
            timeline.performTouchInput { moveTo(Offset(center.x, height * .625f)) }
            awaitVideoFrame(Color.Blue)
            timeline.performTouchInput { moveTo(Offset(center.x, height * .875f)) }
            awaitVideoFrame(Color.Yellow)
            timeline.performTouchInput { up() }
            awaitPlaying(false)
            pill("Editor").performClick()
            pill("Rallies").performClick()
            awaitVideoFrame(Color.Yellow)
            if (cycle % 2 == 1) {
                f.scenario.recreate()
                compose.waitUntil(10_000) {
                    compose.onAllNodesWithTag("rally-vertical-timeline").fetchSemanticsNodes().isNotEmpty()
                }
                awaitVideoFrame(Color.Yellow)
            }
        }
        compose.onNodeWithContentDescription("Play").performClick()
        awaitPlaying(true)
    }

    @Test
    fun speedAndTeamNamesShareTheEditorDraftAndSurviveRecreation() = withProject { f ->
        compose.onNodeWithTag("rally-playback-speed").performClick()
        compose.onNodeWithText("2x").performClick()
        awaitDraft(f) { it.playbackRate == 2f }
        compose.onNodeWithContentDescription("Edit Team 1 name").performClick()
        compose.onNodeWithTag("rally-team-1-input").performTextReplacement("Larch")
        compose.onNodeWithTag("rally-team-2-input").performTextReplacement("Bay")
        compose.onNodeWithTag("rally-team-names-save").performClick()
        awaitDraft(f) { it.scoreTracking.team1Name == "Larch" && it.scoreTracking.team2Name == "Bay" }
        compose.onNodeWithContentDescription("Play").assertIsDisplayed()
        pill("Editor").performClick()
        compose.onNodeWithText("Speed · 2x").performScrollTo().performClick()
        compose.onNodeWithText("4x").performClick()
        compose.onNode(hasSetTextAction() and hasText("Larch")).performScrollTo().performTextReplacement("Alder")
        awaitDraft(f) { it.playbackRate == 4f && it.scoreTracking.team1Name == "Alder" }
        pill("Rallies").performClick()
        compose.onNodeWithText("Alder").assertIsDisplayed()
        compose.onNodeWithContentDescription("Edit Team 2 name").performClick()
        compose.onNodeWithTag("rally-team-2-input").performTextReplacement("Harbor Lights")
        compose.onNodeWithTag("rally-team-names-save").performClick()
        awaitDraft(f) { it.scoreTracking.team2Name == "Harbor Lights" }
        f.scenario.recreate()
        compose.waitUntil(10_000) { compose.onAllNodesWithTag("rally-playback-speed").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithTag("rally-playback-speed").assert(
            SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, "4x"))
        compose.onNodeWithText("Alder").assertIsDisplayed()
        compose.onNodeWithText("Harbor Lights").assertIsDisplayed()
        capture(f, "courtside-shared-controls.png")
    }

    @Test
    fun headerOpensProjectPickerExportAndTheExistingSettingsModal() = withProject { f ->
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        compose.onNodeWithTag("editor-project-selector").assertIsDisplayed().performClick()
        compose.onNodeWithText("Start a new video…").assertIsDisplayed()
        compose.onNode(hasText("Courtside test") and hasAnyAncestor(isPopup())).performClick()
        compose.onNodeWithTag("rally-review-workspace").assertIsDisplayed()
        compose.onNodeWithTag("editor-settings-button").performClick()
        compose.onNodeWithText("Settings").assertIsDisplayed()
        device.pressBack()
        compose.onNodeWithTag("workspace-export-tab").performClick()
        compose.onNodeWithTag("workspace-export-tab").assertIsSelected()
        compose.onNodeWithText("Save MP4").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("YouTube chapters").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Save project").performScrollTo().assertIsDisplayed()
        f.scenario.recreate()
        compose.waitUntil(10_000) { compose.onAllNodesWithTag("editor-export-workspace").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithTag("workspace-export-tab").assertIsSelected()
        compose.onNodeWithText("Save highlights").performScrollTo()
        capture(f, "courtside-export.png")
        compose.onNodeWithTag("workspace-rallies-tab").performClick()
        compose.onNodeWithTag("rally-review-workspace").assertIsDisplayed()
    }

    @Test
    fun portraitEditorUsesTopBarAndExportOwnsScoreSettings() = withProject { f ->
        pill("Editor").performClick()
        compose.onAllNodesWithTag("editor-project-selector").assertCountEquals(1)
        compose.onNodeWithTag("editor-delete-project").assertIsDisplayed()
        compose.onNodeWithTag("open-settings").assertDoesNotExist()
        listOf("More", "Save final video", "YouTube chapters", "Project options",
            "Add scores to the final video", "Score visibility", "Show the point history").forEach {
            compose.onNodeWithText(it).assertDoesNotExist()
        }
        capture(f, "courtside-portrait-editor.png")
        compose.onNodeWithTag("editor-project-selector").performClick()
        compose.onNode(hasText("Courtside test") and hasText("ready to review") and hasAnyAncestor(isPopup()))
            .assertIsDisplayed()
        capture(f, "courtside-project-menu.png")
        compose.onNode(hasText("Courtside test") and hasAnyAncestor(isPopup())).performClick()
        compose.onNodeWithTag("editor-delete-project").performClick()
        compose.onNodeWithText("Delete Courtside test?").assertIsDisplayed()
        compose.onNodeWithText("Cancel").performClick()
        compose.onNodeWithTag("editor-settings-button").performClick()
        compose.onNodeWithTag("settings-project-options").performClick()
        compose.onNodeWithText("Start this review over").assertIsDisplayed()
        compose.onNodeWithText("Close").performClick()
        compose.onNodeWithTag("editor-settings-button").performClick()
        compose.onNodeWithTag("settings-app-version").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("settings-reset-tutorial").performScrollTo().assertIsDisplayed()
        capture(f, "courtside-settings.png")
        compose.onNodeWithText("Done").performClick()

        compose.onNodeWithTag("workspace-export-tab").performClick()
        compose.onNodeWithText("Add scores to the final video").assertIsDisplayed()
        compose.onNodeWithTag("export-render-scores").performClick()
        awaitDraft(f) { !it.renderScoreOverlay && !it.renderScoreTimeline }
        compose.onNodeWithText("Score visibility").assertDoesNotExist()
        compose.onNodeWithTag("export-render-scores").performClick()
        compose.onNodeWithText("Always visible").performScrollTo().performClick()
        awaitDraft(f) { it.renderScoreOverlay && !it.fadeScoreOverlay }
        compose.onNodeWithText("Fade in and out").performClick()
        awaitDraft(f) { it.fadeScoreOverlay }
        compose.onNodeWithTag("export-render-point-history").performScrollTo().assertIsOff().performClick()
        awaitDraft(f) { it.renderScoreTimeline }
        f.scenario.recreate()
        compose.waitUntil(10_000) {
            compose.onAllNodesWithTag("editor-export-workspace").fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithTag("export-render-scores").assertIsOn()
        compose.onNodeWithText("Fade in and out").assertIsSelected()
        compose.onNodeWithTag("export-render-point-history").assertIsOn()
        compose.onNodeWithText("Save highlights").performScrollTo()
        capture(f, "courtside-export-options.png")
        pill("Editor").performClick()
        compose.onNodeWithText("Add scores to the final video").assertDoesNotExist()
        compose.onNodeWithTag("editor-settings-button").performClick()
        compose.onNodeWithTag("settings-reset-tutorial").performScrollTo().performClick()
        compose.waitUntil(5_000) { GuidedTourStore.read(f.context) != "dismissed" }
        compose.onNodeWithTag("workspace-rallies-tab").assertIsSelected()
        compose.onNodeWithText(GuidedTourStep.RALLIES_REVIEW.title).assertIsDisplayed()
    }

    @Test
    fun defaultRalliesKeepsProjectControlsTogether() = withProject(sourceName = "Match", legacyEditorPreference = true) { f ->
        compose.onNodeWithTag("workspace-rallies-tab").assertIsSelected()
        val rallyTab = compose.onNodeWithTag("workspace-rallies-tab").fetchSemanticsNode().boundsInRoot
        val editorTab = compose.onNodeWithTag("workspace-editor-tab").fetchSemanticsNode().boundsInRoot
        assertTrue(rallyTab.left < editorTab.left)
        val selector = compose.onNodeWithTag("editor-project-selector").fetchSemanticsNode().boundsInRoot
        val label = compose.onNodeWithTag("editor-project-label", useUnmergedTree = true).fetchSemanticsNode().boundsInRoot
        val arrow = compose.onNodeWithTag("editor-project-chevron", useUnmergedTree = true).fetchSemanticsNode().boundsInRoot
        val trash = compose.onNodeWithTag("editor-delete-project").fetchSemanticsNode().boundsInRoot
        val density = compose.onNodeWithTag("editor-project-selector").fetchSemanticsNode().layoutInfo.density.density
        val textLayouts = mutableListOf<androidx.compose.ui.text.TextLayoutResult>()
        compose.onNodeWithText("Match", useUnmergedTree = true).performSemanticsAction(SemanticsActions.GetTextLayoutResult) {
            it(textLayouts)
        }
        capture(f, "courtside-project-sized.png")
        val nameLayout = textLayouts.single()
        assertFalse("A short project name must remain readable: size=${nameLayout.size}, " +
            "constraints=${nameLayout.layoutInput.constraints}, widthOverflow=${nameLayout.didOverflowWidth}, " +
            "heightOverflow=${nameLayout.didOverflowHeight}", nameLayout.isLineEllipsized(0))
        assertTrue("Arrow must remain beside the text", arrow.left - label.right in 0f..12f * density)
        assertTrue("Arrow must remain within the selector", arrow.right <= selector.right)
        val trashGap = trash.left - selector.right
        capture(f, "courtside-project-sized.png")
        assertTrue("Trash must follow the selector; gap=$trashGap px", trashGap in -density..8f * density)
        assertTrue("Selector should wrap its content", selector.width < label.width + 50f * density)
        pill("Editor").performClick()
        f.scenario.recreate()
        compose.waitUntil(10_000) {
            compose.onAllNodesWithTag("workspace-editor-tab").fetchSemanticsNodes().firstOrNull()
                ?.config?.getOrNull(SemanticsProperties.Selected) == true
        }
        compose.onNodeWithTag("workspace-editor-tab").assertIsSelected()
    }

    @Test
    fun landscapeHasOneSafeTopBarWithCenteredTabsAndReviewSummary() = withProject(sourceName = "Match") { f ->
        fun checkHeader() {
            compose.waitUntil(5_000) { compose.onAllNodesWithText("MATCH EVENTS").fetchSemanticsNodes().isNotEmpty() }
            compose.onNodeWithText("MATCH EVENTS").assertIsDisplayed()
            compose.onAllNodesWithTag("editor-top-bar").assertCountEquals(1)
            compose.onNodeWithTag("open-settings").assertDoesNotExist()
            compose.onNodeWithText("More").assertDoesNotExist()
            val header = compose.onNodeWithTag("editor-top-bar").fetchSemanticsNode().boundsInRoot
            val tabs = compose.onNodeWithTag("editor-version-pill").fetchSemanticsNode().boundsInRoot
            val brand = compose.onNodeWithTag("editor-brand").fetchSemanticsNode().boundsInRoot
            val cog = compose.onNodeWithTag("editor-settings-button").fetchSemanticsNode().boundsInRoot
            val density = compose.onNodeWithTag("editor-top-bar").fetchSemanticsNode().layoutInfo.density.density
            assertEquals(header.center.x, tabs.center.x, density)
            var windowWidth = 0
            var leftInset = 0
            var rightInset = 0
            f.scenario.onActivity { activity ->
                windowWidth = activity.window.decorView.width
                val insets = androidx.core.view.ViewCompat.getRootWindowInsets(activity.window.decorView)
                val safe = insets?.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars() or
                    androidx.core.view.WindowInsetsCompat.Type.displayCutout())
                val waterfall = insets?.displayCutout?.waterfallInsets
                leftInset = maxOf(safe?.left ?: 0, waterfall?.left ?: 0)
                rightInset = maxOf(safe?.right ?: 0, waterfall?.right ?: 0)
            }
            assertTrue("Logo must clear the safe left edge", brand.left >= leftInset + 8 * density)
            assertTrue("Cog must clear the safe right edge", cog.right <= windowWidth - rightInset - 8 * density)
            compose.onNodeWithTag("header-review-cleanup").assertIsDisplayed()
            compose.onNodeWithTag("header-review-clips").assertIsDisplayed()
            compose.onNodeWithTag("header-review-serves").assertIsDisplayed()
            compose.onNodeWithTag("header-included").assertIsDisplayed()
            compose.onNodeWithTag("header-total-time").assertIsDisplayed()
            val summary = compose.onNodeWithTag("header-review-summary").fetchSemanticsNode().boundsInRoot
            assertTrue("Review summary belongs right of the tabs", summary.left > tabs.right)
            assertTrue("Settings stays beyond the summary", cog.left >= summary.right)
            val time = compose.onNodeWithTag("header-total-time").fetchSemanticsNode().boundsInRoot
            val cleanup = compose.onNodeWithTag("header-review-cleanup").fetchSemanticsNode().boundsInRoot
            val serves = compose.onNodeWithTag("header-review-serves").fetchSemanticsNode().boundsInRoot
            assertTrue("Review buttons belong after the clip/time summary", cleanup.left >= time.right)
            assertTrue("Review buttons must sit immediately before Settings", cog.left - serves.right in -density..4 * density)
            compose.onNodeWithTag("header-review-cleanup").assertTextContains("Review\ncleanup")
            compose.onNodeWithTag("header-review-clips").assertTextContains("Review\nclips")
            compose.onNodeWithTag("header-review-serves").assertTextContains("Review\nserves")
        }
        try {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
            compose.waitUntil(10_000) {
                f.context.resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
            }
            pill("Editor").performClick()
            compose.waitUntil(5_000) { compose.onAllNodesWithTag("header-review-summary").fetchSemanticsNodes().isNotEmpty() }
            checkHeader()
            capture(f, "courtside-single-landscape-bar.png")
            var oldRotation = 0
            f.scenario.onActivity { oldRotation = requireNotNull(it.display).rotation }
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE }
            compose.waitUntil(10_000) {
                var rotated = false
                f.scenario.onActivity { rotated = requireNotNull(it.display).rotation != oldRotation }
                rotated
            }
            checkHeader()
            capture(f, "courtside-reverse-landscape-bar.png")
        } finally {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED }
        }
    }

    @Test
    fun rallyTutorialNavigatesRealViewsAndDoesNotEditTheProject() = withProject { f ->
        val before = requireNotNull(f.store.load())
        // An unfinished pre-Rallies tutorial migrates to the new first step.
        f.context.getSharedPreferences("volleycut-guided-tour-v2", Context.MODE_PRIVATE).edit()
            .putString("state", "EDITOR_FOCUS").putInt("flowVersion", 2).commit()
        assertEquals(GuidedTourStep.RALLIES_REVIEW.name, GuidedTourStore.read(f.context))
        f.scenario.recreate()
        val steps = guidedTourSteps.filter { it.stage == GuidedTourStage.EDITOR }
        steps.forEach { step ->
            compose.waitUntil(10_000) {
                compose.onAllNodesWithText(step.title).fetchSemanticsNodes().isNotEmpty()
            }
            compose.onNodeWithText(step.title).assertIsDisplayed()
            val tab = when (step.workspace) {
                GuidedTourWorkspace.EDITOR -> "workspace-editor-tab"
                GuidedTourWorkspace.EXPORT -> "workspace-export-tab"
                else -> "workspace-rallies-tab"
            }
            compose.waitUntil(5_000) {
                compose.onNodeWithTag(tab).fetchSemanticsNode().config.getOrNull(SemanticsProperties.Selected) == true
            }
            when (step.workspace) {
                GuidedTourWorkspace.RALLY -> compose.onNodeWithTag("rally-review-workspace").assertExists()
                GuidedTourWorkspace.RALLY_LIST -> compose.onNodeWithTag("rally-card-list").assertExists()
                GuidedTourWorkspace.ADD_RALLY -> {
                    compose.onNodeWithTag("whole-timeline").performTouchInput { click(Offset(width * .4f, center.y)) }
                    compose.onNodeWithTag("add-missing-rally-confirm").assertIsEnabled()
                }
                GuidedTourWorkspace.EDITOR -> compose.waitUntil(5_000) {
                    compose.onAllNodesWithTag("whole-timeline").onFirst().isDisplayed()
                }
                GuidedTourWorkspace.EXPORT -> compose.onNodeWithTag("editor-export-workspace").assertExists()
            }
            if (step == GuidedTourStep.RALLIES_SOURCE) {
                f.scenario.recreate()
                compose.waitUntil(10_000) { compose.onAllNodesWithText(step.title).fetchSemanticsNodes().isNotEmpty() }
                compose.onNodeWithTag("rally-vertical-timeline").assertExists()
                assertEquals(step.name, GuidedTourStore.read(f.context))
            }
            if (step in setOf(GuidedTourStep.RALLIES_TAG, GuidedTourStep.RALLIES_LIST,
                    GuidedTourStep.RALLIES_ADD, GuidedTourStep.EDITOR_OVERVIEW, GuidedTourStep.EDITOR_EXPORT)) {
                capture(f, "courtside-tour-" + step.name.lowercase() + ".png")
            }
            compose.onNodeWithTag("guided-tour-next").performScrollTo().performClick()
        }
        compose.onNodeWithTag("guided-tour-card").assertDoesNotExist()
        assertEquals("done", GuidedTourStore.read(f.context))
        val after = requireNotNull(f.store.load())
        assertEquals(before.cuts, after.cuts)
        assertEquals(before.reviewedCutIds, after.reviewedCutIds)
        assertEquals(before.scoreTracking.serveMarkers, after.scoreTracking.serveMarkers)
        // Reset is reachable from the new default view; closing remains persistent.
        compose.onNodeWithTag("editor-settings-button").performClick()
        compose.onNodeWithTag("settings-reset-tutorial").performScrollTo().performClick()
        compose.waitUntil(5_000) { compose.onAllNodesWithText(GuidedTourStep.RALLIES_REVIEW.title).fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithTag("guided-tour-close").performClick()
        f.scenario.recreate()
        compose.onNodeWithTag("guided-tour-card").assertDoesNotExist()
        assertEquals("dismissed", GuidedTourStore.read(f.context))
    }

    @Test
    fun mergingEitherNeighbourRetainsTheViewedFrame() {
        listOf(false, true).forEach { previous ->
            withProject(paddingMs = 2_000, configureDraft = { draft ->
                draft.copy(joinGapMs = 3_000, reviewedCutIds = draft.cuts.mapTo(mutableSetOf()) { it.id },
                    scoreTracking = draft.scoreTracking.copy(serveMarkers = emptyList()))
            }) { f ->
                compose.onNodeWithText("Browse rallies").performClick()
                compose.onNodeWithTag(if (previous) "rally-card-R002" else "rally-card-R001").performScrollTo().performClick()
                compose.onNodeWithTag("edit-selected-rally").performClick()
                val timeline = compose.onNodeWithTag("rally-bounds-timeline")
                val position = if (previous) 6_000f else 11_000f
                timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(position) }
                compose.onNodeWithContentDescription(if (previous) "Start earlier by 1 second" else "End later by 1 second").performClick()
                awaitDraft(f) { rallyGroupsForDraft(it).size == 1 }
                compose.waitUntil(5_000) {
                    timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current == position
                }
                assertEquals(setOf("R001", "R002"), rallyGroupsForDraft(requireNotNull(f.store.load())).single().cutIds)
            }
        }
    }

    private fun checkShortCoreHandles(duration: Long, padding: Long) = withProject(configureDraft = { draft ->
        draft.copy(cuts = draft.cuts.take(1).map { it.copy(coreStartMs = 10_000, coreEndMs = 10_000 + duration,
            keepStartMs = (10_000 - padding).coerceAtLeast(0), keepEndMs = (10_000 + duration + padding).coerceAtMost(20_000)) },
            scoreTracking = draft.scoreTracking.copy(enabled = false, serveMarkers = emptyList()))
    }) { f ->
        val timeline = compose.onNodeWithTag("rally-bounds-timeline")
        val node = timeline.fetchSemanticsNode()
        val density = node.layoutInfo.density.density
        val range = node.config[SemanticsProperties.ProgressBarRangeInfo].range
        val separationDp = (node.boundsInRoot.width / density - 44f) * duration / (range.endInclusive - range.start)
        assertTrue("Short-core handles need independent touch regions", separationDp >= 48f)
        compose.onNodeWithTag("rally-padding-outside-view").assertIsDisplayed()
        capture(f, "courtside-short-core-$duration.png")
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(10_000f + duration / 2f) }
        fun drag(from: Float, to: Float) {
            val scale = detailScale(f)
            val inset = 22f * density
            timeline.performTouchInput {
                fun x(value: Float) = inset + (width - inset * 2) * scale.fraction(value.toLong())
                down(Offset(x(from), center.y)); moveTo(Offset(x(to), center.y)); up()
            }
        }
        drag(10_000f + duration, 10_000f + duration * 1.5f)
        awaitDraft(f) { it.cuts.first().coreEndMs >= 10_000 + maxOf(RALLY_MIN_DURATION_MS, duration + duration / 2 - 10) }
        assertEquals(10_000L, requireNotNull(f.store.load()).cuts.first().coreStartMs)
        val end = requireNotNull(f.store.load()).cuts.first().coreEndMs
        drag(10_000f, 9_900f)
        awaitDraft(f) { it.cuts.first().coreStartMs in 9_880..9_920 }
        assertEquals(end, requireNotNull(f.store.load()).cuts.first().coreEndMs)
    }

    @Test
    fun halfSecondCoreKeepsSeparateHandlesWithMaximumPadding() = checkShortCoreHandles(500, 10_000)

    @Test
    fun importedTenthSecondCoreKeepsSeparateHandlesWithDefaultPadding() = checkShortCoreHandles(100, 2_000)

    @Test
    fun paddedTrimmingRetainsTheViewedFrameForNudgesAndDragging() = withProject(paddingMs = 2_000) { f ->
        val timeline = compose.onNodeWithTag("rally-bounds-timeline")
        fun position(value: Float) {
            compose.waitUntil(5_000) {
                val config = timeline.fetchSemanticsNode().config
                val progress = config[SemanticsProperties.ProgressBarRangeInfo]
                if (value in progress.range) progress.current == value else
                    config[SemanticsProperties.StateDescription].contains("video at ${preciseTime(value.toLong())}")
            }
        }
        fun seek(value: Float) {
            timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(value) }
            position(value)
        }
        fun drag(from: Float, to: Float) {
            val scale = detailScale(f)
            val inset = 22f * timeline.fetchSemanticsNode().layoutInfo.density.density
            timeline.performTouchInput {
                fun x(value: Float) = inset + (width - inset * 2) * scale.fraction(value.toLong())
                down(Offset(x(from), center.y))
                moveTo(Offset(x(to), center.y))
                up()
            }
        }
        seek(1_000f) // In retained start padding, before the core start at 2s.
        compose.onNodeWithContentDescription("End later by 1 second").performClick()
        awaitDraft(f) { it.cuts.first().coreEndMs == 6_000L }
        position(1_000f)
        drag(6_000f, 5_500f)
        awaitDraft(f) { it.cuts.first().coreEndMs in 5_450..5_550 }
        position(1_000f)
        seek(7_000f) // In retained end padding.
        compose.onNodeWithContentDescription("Start later by 1 second").performClick()
        awaitDraft(f) { it.cuts.first().coreStartMs == 3_000L }
        position(7_000f)
        drag(3_000f, 4_000f)
        awaitDraft(f) { it.cuts.first().coreStartMs in 3_950..4_050 }
        position(7_000f)
        val priorCut = requireNotNull(f.store.load()).cuts.first()
        val expectedEnd = maxOf(priorCut.coreEndMs - 1_000, priorCut.coreStartMs + RALLY_MIN_DURATION_MS)
        compose.onNodeWithContentDescription("End earlier by 1 second").performClick()
        awaitDraft(f) { it.cuts.first().coreEndMs == expectedEnd }
        val retainedEnd = requireNotNull(f.store.load()).cuts.first().keepEndMs
        position(retainedEnd.toFloat())
        assertTrue("Only the new retained edge may exclude the viewed frame", retainedEnd in 6_450..6_550)
    }

    @Test
    fun addMissingTimelineCanSeekAGapThroughAccessibilityWithoutAdding() = withProject { f ->
        val original = requireNotNull(f.store.load()).cuts
        compose.onNodeWithTag("rally-browse-all").performClick()
        compose.onNodeWithTag("add-missing-rally").performClick()
        compose.onNodeWithTag("add-missing-rally-confirm").assertIsNotEnabled()
        val timeline = compose.onNodeWithTag("whole-timeline")
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(8_000f) }
        compose.waitUntil(5_000) {
            timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current == 8_000f
        }
        compose.onNodeWithTag("add-missing-rally-confirm").assertIsEnabled()
        assertEquals(original, requireNotNull(f.store.load()).cuts)
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(3_000f) }
        compose.onNodeWithTag("add-missing-rally-confirm").assertIsNotEnabled()
        assertEquals(original, requireNotNull(f.store.load()).cuts)
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { assertFalse(it(Float.NaN)) }
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(8_000f) }
        compose.onNodeWithTag("add-missing-rally-confirm").performScrollTo().performClick()
        awaitDraft(f) { it.cuts.size == original.size + 1 }
    }

    @Test
    fun trimmingKeepsTheViewedFrameUntilTheBoundaryPassesIt() = withProject { f ->
        val timeline = compose.onNodeWithTag("rally-bounds-timeline")
        timeline.performSemanticsAction(SemanticsActions.SetProgress) { it(4_000f) }
        compose.waitUntil(5_000) {
            timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current == 4_000f
        }
        val window = timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].range
        val scale = detailScale(f)
        val inset = 22f * timeline.fetchSemanticsNode().layoutInfo.density.density
        timeline.performTouchInput {
            fun x(time: Float) = inset + (width - inset * 2) * scale.fraction(time.toLong())
            down(Offset(x(2_000f), center.y))
            moveTo(Offset(x(3_500f), center.y))
        }
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreStartMs in 3_450..3_550 }
        assertEquals(4_000f, timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current, .1f)
        assertEquals("The focused window must stay fixed while a handle is held", window,
            timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].range)
        timeline.performTouchInput {
            val x = inset + (width - inset * 2) * scale.fraction(4_200)
            moveTo(Offset(x, center.y))
        }
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreStartMs in 4_150..4_250 }
        val start = requireNotNull(f.store.load()).cuts.first { it.id == "R001" }.coreStartMs
        assertEquals(start.toFloat(), timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current, .1f)
        timeline.performTouchInput { up() }
        compose.onNodeWithContentDescription("Start earlier by 1 second").performClick()
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreStartMs == start - 1_000 }
        assertEquals(start.toFloat(), timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current, .1f)
        compose.onNodeWithContentDescription("End earlier by 1 second").performClick()
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreEndMs == 4_000L }
        assertEquals(4_000f, timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current, .1f)
    }

    @Test
    fun timelinesRenderVideoAndSelectedRallyAlwaysOpensForEditing() = withProject { f ->
        awaitVideoFrame(Color.Red)
        val scale = detailScale(f)
        compose.onNodeWithTag("rally-bounds-timeline").performTouchInput {
            // Read the displayed context, which now includes the next retained rally.
            val inset = 22f * compose.onNodeWithTag("rally-bounds-timeline").fetchSemanticsNode().layoutInfo.density.density
            click(Offset(inset + (width - inset * 2) * scale.fraction(8_000), center.y))
        }
        awaitVideoFrame(Color.Green)
        compose.onNodeWithText("All 2  →").performClick()
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput {
            down(Offset(center.x, height * .625f))
        }
        awaitVideoFrame(Color.Blue)
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput {
            moveTo(Offset(center.x, height * .875f))
        }
        awaitVideoFrame(Color.Yellow)
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput { up() }
        compose.onNodeWithTag("add-missing-rally").performClick()
        compose.onNodeWithTag("whole-timeline").performTouchInput {
            click(Offset(width * .15f, center.y))
        }
        awaitVideoFrame(Color.Red)
        compose.onNodeWithText("Cancel").performClick()

        // Edit is always present even with outstanding reviews. It edits the
        // highlighted selection, independently of the source gap or list center.
        compose.onNodeWithTag("rally-card-R002").performScrollTo().performClick()
        compose.onNodeWithTag("rally-card-R002").assertIsSelected()
        compose.onNodeWithTag("rally-browser-review").assertIsDisplayed()
        compose.onNodeWithTag("edit-selected-rally").assertIsEnabled().performClick()
        compose.onNodeWithText("Rally 2").assertIsDisplayed()
        awaitVideoFrame(Color.Blue)
        compose.onNodeWithText("Far").performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.first { m -> m.id == "S2" }.side == ServingSide.FAR }
        compose.onNodeWithContentDescription("Start earlier by 1 second").performClick()
        awaitDraft(f) { it.cuts.first { c -> c.id == "R002" }.coreStartMs == 11_000L }
        compose.onNodeWithTag("rally-browse-all").performClick()
        compose.onNodeWithTag("edit-selected-rally").assertIsDisplayed().assertIsEnabled()
        capture(f, "courtside-selected-edit.png")
    }

    @Test
    fun editsShareBothDirectionsAndOnlyExplicitMissingActionAdds() = withProject { f ->
        compose.onNodeWithText("Rally 1").assertIsDisplayed()
        compose.onNodeWithTag("rally-review-player").assertIsDisplayed()
        capture(f, "courtside-review.png")
        compose.onNodeWithText("Keep").performClick()
        awaitDraft(f) { "R001" in it.reviewedCutIds }
        compose.onNodeWithContentDescription("Start earlier by 1 second").performClick()
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreStartMs == 1_000L }
        compose.onNodeWithTag("rally-start-readout").assertTextEquals("Start 0:01.0")
        val density = compose.onNodeWithTag("rally-bounds-timeline").fetchSemanticsNode().layoutInfo.density.density
        val scale = detailScale(f)
        compose.onNodeWithTag("rally-bounds-timeline").performTouchInput {
            val startX = 22f * density + (width - 44f * density) * scale.fraction(1_000)
            swipe(Offset(startX, center.y), Offset(startX + 40f, center.y), 400)
        }
        awaitDraft(f) { it.cuts.first { c -> c.id == "R001" }.coreStartMs > 1_000L }
        val trimmedStart = requireNotNull(f.store.load()).cuts.first { it.id == "R001" }.coreStartMs

        // Rallies -> existing Editor. Its controls mutate the identical persisted cut.
        pill("Editor").performClick()
        compose.onNodeWithTag("rally-review-workspace").assertDoesNotExist()
        compose.onNodeWithTag("editor-rally-remove").performScrollTo().performClick()
        awaitDraft(f) { !it.cuts.first { c -> c.id == "R001" }.included }
        pill("Rallies").performClick()
        compose.onNodeWithTag("rally-start-readout").assertTextEquals("Start " + preciseTime(trimmedStart))
        compose.onNodeWithText("Remove").assertIsSelected()
        compose.onNodeWithTag("rally-next-review").performClick()
        compose.onNodeWithText("Rally 2").assertIsDisplayed()
        compose.onNodeWithText("Far").performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.first { m -> m.id == "S2" }.side == ServingSide.FAR }
        compose.onNodeWithTag("rally-next-review").performClick()
        compose.onNodeWithText("All checked.").assertIsDisplayed()
        compose.onNodeWithText("Browse rallies").performClick()
        compose.onNodeWithTag("rally-card-R002").performScrollTo().assertIsDisplayed()
        capture(f, "courtside-browse.png")
        compose.onNodeWithText("Hide timeline").performClick()
        compose.onNodeWithTag("rally-vertical-timeline").assertDoesNotExist()
        compose.onNodeWithText("Show timeline").performClick()

        // A source gap seeks in place. It never switches to Editor or creates a cut.
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput {
            click(Offset(center.x, height * .45f))
        }
        compose.onNodeWithTag("rally-review-workspace").assertIsDisplayed()
        compose.onNodeWithTag("rally-source-status").assertTextContains("No rally at", substring = true)
        assertEquals(2, requireNotNull(f.store.load()).cuts.size)
        compose.onNodeWithTag("add-missing-rally").performClick()
        compose.onNodeWithText("Add missing rally").assertIsDisplayed()
        compose.onNodeWithText("Cancel").performClick()
        assertEquals(2, requireNotNull(f.store.load()).cuts.size)

        // Explicit add stays in Rallies and immediately appears in the existing Editor.
        compose.onNodeWithTag("add-missing-rally").performClick()
        compose.onNodeWithTag("add-missing-rally-confirm").performScrollTo().performClick()
        awaitDraft(f) { it.cuts.size == 3 && it.cuts.any { c -> c.origin == CutOrigin.MANUAL && c.coreStartMs <= 9_000 && c.coreEndMs >= 9_000 } }
        compose.onNodeWithTag("rally-browser").assertIsDisplayed()
        pill("Editor").performClick()
        compose.onNodeWithTag("rally-review-workspace").assertDoesNotExist()
        assertEquals(3, requireNotNull(f.store.load()).cuts.size)
        f.scenario.recreate()
        compose.waitUntil(10_000) { compose.onAllNodesWithTag("editor-version-pill").fetchSemanticsNodes().isNotEmpty() }
        pill("Rallies").performClick()
        compose.onNodeWithTag("rally-review-workspace").assertIsDisplayed()
        assertTrue(requireNotNull(f.store.load()).cuts.any { it.origin == CutOrigin.MANUAL })
        assertEquals(ServingSide.FAR, requireNotNull(f.store.load()).scoreTracking.serveMarkers.first { it.id == "S2" }.side)
    }

    @Test
    fun notAServeIsVisibleForAssignedAndUnresolvedServesAndPersists() = withProject { f ->
        compose.onNodeWithText("Not a serve").assertIsDisplayed().performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.none { m -> m.id == "S1" } }
        compose.onNodeWithText("Not a serve").assertIsSelected()
        pill("Editor").performClick()
        pill("Rallies").performClick()
        compose.onNodeWithText("Not a serve").assertIsSelected()
        compose.onNodeWithText("Near").performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.any { m -> m.rallyId == "R001" && m.side == ServingSide.NEAR } }
        compose.onNodeWithText("Not a serve").performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.none { m -> m.rallyId == "R001" } }
        compose.onNodeWithText("Keep").performClick()
        compose.onNodeWithTag("rally-next-review").performClick()
        compose.onNodeWithText("Rally 2").assertIsDisplayed()
        compose.onNodeWithText("Not a serve").assertIsDisplayed().performClick()
        awaitDraft(f) { it.scoreTracking.serveMarkers.isEmpty() }
        compose.onNodeWithContentDescription("Serving side options").assertDoesNotExist()
        compose.onNodeWithText("Not a serve").assertIsSelected()
        compose.onNodeWithTag("rally-next-review").assertIsEnabled()
        f.scenario.recreate()
        compose.waitUntil(10_000) { compose.onAllNodesWithText("Not a serve").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText("Not a serve").assertIsSelected()
    }

    @Test
    fun focusedTimelineShowsTheNearestRetainedRalliesAfterRemoval() =
        withProject(many = true, removedRallyIds = setOf("R002", "R004", "R006")) { f ->
            compose.onNodeWithText("Browse rallies").performClick()
            compose.onNodeWithTag("rally-card-list").performScrollToNode(hasTestTag("rally-card-R003"))
            compose.onNodeWithTag("rally-card-R003").performClick()
            compose.onNodeWithTag("edit-selected-rally").performClick()
            compose.onNodeWithTag("rally-bounds-timeline").assert(
                SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, "Previous rally 1; next rally 5"))
            capture(f, "courtside-context.png")
            compose.onNodeWithText("Remove").performScrollTo().performClick()
            awaitDraft(f) { !it.cuts.first { c -> c.id == "R003" }.included }
            compose.onNodeWithTag("next-rally").performScrollTo().performClick()
            compose.onNodeWithText("Rally 5").assertExists()
            compose.onNodeWithTag("rally-bounds-timeline").assert(
                SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, "Previous rally 1; next rally 7"))
            compose.onNodeWithTag("previous-rally").performScrollTo().performClick()
            compose.onNodeWithText("Rally 1").assertExists()
            compose.onNodeWithTag("previous-rally").assertIsNotEnabled()
            compose.onNodeWithTag("next-rally").performScrollTo().performClick()
            compose.onNodeWithText("Rally 5").assertExists()
        }

    @Test
    fun neighbourBarsRemainVisibleAndSeekVideoBesideAShortCore() = withProject(many = true, configureDraft = { draft ->
        val ranges = listOf(500L to 1_500L, 10_000L to 10_500L, 18_000L to 19_000L)
        draft.copy(cuts = draft.cuts.take(3).mapIndexed { index, cut ->
            val (start, end) = ranges[index]
            cut.copy(coreStartMs = start, coreEndMs = end, keepStartMs = start, keepEndMs = end)
        }, scoreTracking = draft.scoreTracking.copy(enabled = false, serveMarkers = emptyList()))
    }) { f ->
        compose.onNodeWithText("Browse rallies").performClick()
        compose.onNodeWithTag("rally-card-R002").performClick()
        compose.onNodeWithTag("edit-selected-rally").performClick()
        val timeline = compose.onNodeWithTag("rally-bounds-timeline")
        fun assertVisibleBars() {
            val node = timeline.fetchSemanticsNode()
            val density = node.layoutInfo.density.density
            val pixels = timeline.captureToImage().toPixelMap()
            val row = (40 * density).toInt()
            fun barWidth(from: Float, to: Float): Int = ((pixels.width * from).toInt() until (pixels.width * to).toInt()).count { x ->
                val color = pixels[x, row]
                color.green - color.red > .07f && color.green - color.blue > .02f
            }
            assertTrue("Previous rally needs a rendered bar, not only an edge arrow or label", barWidth(0f, .30f) / density >= 20f)
            assertTrue("Next rally needs a rendered bar, not only an edge arrow or label", barWidth(.70f, 1f) / density >= 20f)
        }
        assertVisibleBars()
        capture(f, "courtside-neighbour-context-portrait.png")
        val original = requireNotNull(f.store.load()).cuts
        fun seek(time: Long) {
            val scale = detailScale(f, "R002")
            val inset = 22f * timeline.fetchSemanticsNode().layoutInfo.density.density
            timeline.performTouchInput { click(Offset(inset + (width - inset * 2) * scale.fraction(time), center.y)) }
            compose.waitUntil(5_000) {
                kotlin.math.abs(timeline.fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current - time) <= 5
            }
        }
        seek(900); awaitVideoFrame(Color.Red)
        seek(18_500); awaitVideoFrame(Color.Yellow)
        compose.onNodeWithTag("rally-start-readout").assertTextEquals("Start 0:10.0")
        assertEquals("Seeking context must not edit it or retarget the trimmed rally", original, requireNotNull(f.store.load()).cuts)
        seek(10_250)
        val scale = detailScale(f, "R002")
        val inset = 22f * timeline.fetchSemanticsNode().layoutInfo.density.density
        timeline.performTouchInput {
            fun x(time: Long) = inset + (width - inset * 2) * scale.fraction(time)
            down(Offset(x(10_500), center.y)); moveTo(Offset(x(10_700), center.y)); up()
        }
        awaitDraft(f) { it.cuts[1].coreEndMs in 10_680..10_720 }
        assertEquals(10_000L, requireNotNull(f.store.load()).cuts[1].coreStartMs)
        assertVisibleBars()
        try {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
            compose.waitUntil(10_000) { f.context.resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE }
            assertVisibleBars()
            capture(f, "courtside-neighbour-context-landscape.png")
        } finally {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED }
        }
    }

    @Test
    fun manualListScrollMovesOnlyItsMarkerAndSurvivesThePill() = withProject(many = true) { f ->
        compose.onNodeWithText("Browse rallies").performClick()
        val initial = compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString()
        val before = compose.onNodeWithTag("rally-vertical-timeline").fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current
        compose.onNodeWithTag("rally-card-list").performTouchInput { swipeUp(durationMillis = 500) }
        capture(f, "courtside-list-position.png")
        compose.waitUntil(5_000) {
            compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString() != initial
        }
        val center = compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString()
        val after = compose.onNodeWithTag("rally-vertical-timeline").fetchSemanticsNode().config[SemanticsProperties.ProgressBarRangeInfo].current
        assertEquals(before, after, .1f)
        pill("Editor").performClick()
        pill("Rallies").performClick()
        assertEquals(center, compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString())
        // Seeking resumes list follow immediately, including while the finger is still down.
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput {
            down(Offset(width / 2f, height * .325f))
        }
        compose.waitUntil(5_000) {
            compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString().contains("4")
        }
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput {
            moveTo(Offset(width / 2f, height * .775f))
        }
        compose.waitUntil(5_000) {
            compose.onNodeWithTag("rally-list-center").fetchSemanticsNode().config[SemanticsProperties.Text].joinToString().contains("10")
        }
        compose.onNodeWithTag("rally-vertical-timeline").performTouchInput { up() }
        compose.onNodeWithTag("rally-review-player").assertIsDisplayed()
    }

    @Test
    fun recreationAndLandscapeRetainTheRallyBrowser() = withProject(many = true) { f ->
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        compose.onNodeWithText("Browse rallies").performClick()
        // Stay away from the list end, where a viewport resize legitimately clamps scroll.
        compose.onNodeWithTag("rally-card-list").performTouchInput {
            swipe(Offset(center.x, height * .6f), Offset(center.x, height * .45f), durationMillis = 600)
        }
        compose.waitForIdle()
        device.waitForIdle()
        val scroll = compose.onNodeWithTag("rally-card-list").fetchSemanticsNode()
            .config[SemanticsProperties.VerticalScrollAxisRange].value()
        f.scenario.recreate()
        compose.waitUntil(10_000) { compose.onAllNodesWithTag("rally-browser").fetchSemanticsNodes().isNotEmpty() }
        compose.waitUntil(5_000) {
            kotlin.math.abs(scroll - compose.onNodeWithTag("rally-card-list").fetchSemanticsNode()
                .config[SemanticsProperties.VerticalScrollAxisRange].value()) < .01f
        }
        assertEquals(scroll, compose.onNodeWithTag("rally-card-list").fetchSemanticsNode()
            .config[SemanticsProperties.VerticalScrollAxisRange].value(), .01f)
        try {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
            compose.waitUntil(10_000) {
                f.context.resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
            }
            compose.waitUntil(10_000) { compose.onAllNodesWithTag("rally-browser").fetchSemanticsNodes().isNotEmpty() }
            compose.onNodeWithTag("rally-review-player").assertIsDisplayed()
            compose.onNodeWithTag("rally-browser").assertIsDisplayed()
            compose.onNodeWithTag("add-missing-rally").assertIsDisplayed()
            capture(f, "courtside-landscape.png")
        } finally {
            f.scenario.onActivity { it.requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED }
        }
    }
}
