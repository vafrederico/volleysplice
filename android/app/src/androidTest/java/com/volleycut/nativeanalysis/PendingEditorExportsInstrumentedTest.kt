package com.volleycut.nativeanalysis

import android.net.Uri
import android.os.Parcel
import android.os.SystemClock
import androidx.lifecycle.SavedStateHandle
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

class PendingEditorExportsInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Test fun savedActivityStateRestoresLargeRequestedPayloadWithoutPuttingItInTheBundle() = withStore { store ->
        val text = "{\"synthetic\":\"" + "large-review".repeat(200_000) + "\"}"
        val request = PendingEditorExport.Text(EditorExportKind.PROJECT, "synthetic-project", "review.json", text)
        val state = SavedStateHandle()
        val original = onMain { PendingEditorExports(state) }
        onMain { original.prepare(request.kind, request.projectId, store) { request } }
        await { original.ready.value != null }
        val ready = checkNotNull(original.ready.value)
        onMain { original.launched(ready) }
        val saved = onMain { state.savedStateProvider().saveState() }
        val parcel = Parcel.obtain()
        try {
            parcel.writeBundle(saved)
            assertTrue("Large review leaked into saved activity state", parcel.dataSize() < 16_384)
        } finally { parcel.recycle() }
        val restored = onMain { PendingEditorExports(SavedStateHandle.createHandle(saved, null)) }
        val written = AtomicReference<PendingEditorExport?>()
        onMain { restored.consume(request.kind, "different-current-project", Uri.parse("content://synthetic/destination"), store) { payload, _ -> written.set(payload) } }
        await { restored.outcome.value != null && restored.activeWrites.value == 0 }
        assertEquals(request, written.get())
        assertEquals(request.projectId, restored.outcome.value?.projectId)
        assertThrows(java.io.IOException::class.java) { store.load(ready.id) }
    }

    @Test fun restoredChaptersCancelAndMissingRequestNeverWriteTheDestination() = withStore { store ->
        val request = PendingEditorExport.Text(EditorExportKind.CHAPTERS, "synthetic", "chapters.txt", "0:00 Start\n0:01 Rally\n")
        val state = SavedStateHandle()
        val original = onMain { PendingEditorExports(state) }
        onMain { original.prepare(request.kind, request.projectId, store) { request } }
        await { original.ready.value != null }
        val ready = checkNotNull(original.ready.value)
        onMain { original.launched(ready) }
        val restored = onMain { PendingEditorExports(SavedStateHandle.createHandle(state.savedStateProvider().saveState(), null)) }
        onMain { restored.consume(request.kind, request.projectId, null, store) { _, _ -> fail("Cancelled picker wrote a destination") } }
        await { restored.activeWrites.value == 0 }
        assertNull(restored.outcome.value)
        assertThrows(java.io.IOException::class.java) { store.load(ready.id) }
        onMain { restored.consume(request.kind, request.projectId, Uri.parse("content://synthetic/destination"), store) { _, _ -> fail("Missing request wrote a destination") } }
        assertTrue(restored.outcome.value?.message.orEmpty().contains("without an export request"))
    }

    @Test fun restoredVideoRequestKeepsItsOriginalSourceIntervalsAndScoreOptions() = withStore { store ->
        val snapshot = ScoreExportSnapshotJson.encode(ScoreExportSnapshot(
            render = true, renderPointTimeline = false, fadeScoreOverlay = true,
            scoreTracking = ScoreTracking(enabled = true), ignoredIntervals = emptyList(),
            excludedRallyIds = setOf("R002"), rallyRanges = emptyList(),
            mergedRanges = listOf(ScoreMergedRange(2_000, 5_000)),
        ))
        val request = PendingEditorExport.Video("original-job", "original-project", "review.mp4",
            "content://synthetic/original-source", "synthetic.mp4", 10_000,
            listOf(2_000), listOf(5_000), snapshot)
        val state = SavedStateHandle()
        val original = onMain { PendingEditorExports(state) }
        onMain { original.prepare(request.kind, request.projectId, store) { request } }
        await { original.ready.value != null }
        onMain { original.launched(checkNotNull(original.ready.value)) }
        val restored = onMain { PendingEditorExports(SavedStateHandle.createHandle(state.savedStateProvider().saveState(), null)) }
        val written = AtomicReference<PendingEditorExport?>()
        onMain { restored.consume(request.kind, "different-current-project", Uri.parse("content://synthetic/destination"), store) { payload, _ -> written.set(payload) } }
        await { restored.outcome.value != null && restored.activeWrites.value == 0 }
        assertEquals(request, written.get())
        assertEquals(request.jobId, restored.outcome.value?.jobId)
        assertEquals(request.projectId, restored.outcome.value?.projectId)
    }

    @Test fun unreadablePreparedRequestIsReportedBeforeOpeningTheDestination() = withStore { store ->
        val state = SavedStateHandle()
        val requests = onMain { PendingEditorExports(state) }
        onMain { requests.prepare(EditorExportKind.PROJECT, "synthetic", store) {
            PendingEditorExport.Text(EditorExportKind.PROJECT, "synthetic", "review.json", "{}")
        } }
        await { requests.ready.value != null }
        val ready = checkNotNull(requests.ready.value)
        onMain { requests.launched(ready) }
        store.remove(ready.id)
        onMain { requests.consume(EditorExportKind.PROJECT, "synthetic", Uri.parse("content://synthetic/preserved"), store) { _, _ -> fail("Unreadable request opened its destination") } }
        await { requests.outcome.value != null }
        assertTrue(requests.outcome.value?.message.orEmpty().contains("No export was started"))
    }

    private fun <T> onMain(block: () -> T): T {
        val result = AtomicReference<T>()
        instrumentation.runOnMainSync { result.set(block()) }
        return result.get()
    }

    private fun await(condition: () -> Boolean) {
        val deadline = SystemClock.uptimeMillis() + 10_000
        while (!onMain(condition)) {
            check(SystemClock.uptimeMillis() < deadline) { "Pending export did not settle" }
            SystemClock.sleep(20)
        }
    }

    private fun withStore(block: (PendingEditorExportStore) -> Unit) {
        val directory = File(instrumentation.targetContext.cacheDir, "pending-export-test-${UUID.randomUUID()}").apply { mkdirs() }
        try { block(PendingEditorExportStore(directory)) } finally { directory.deleteRecursively() }
    }
}
