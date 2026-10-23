package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException

class PendingEditorExportStoreTest {
    @get:Rule val temporary = TemporaryFolder()
    private fun store() = PendingEditorExportStore(temporary.root)

    @Test fun freshStoreRestoresCompleteVideoRequestIncludingScoreAndSource() {
        val score = ScoreExportSnapshotJson.encode(ScoreExportSnapshot(
            render = true, renderPointTimeline = false, fadeScoreOverlay = true,
            scoreTracking = ScoreTracking(enabled = true, team1Name = "Synthetic A", team2Name = "Synthetic B"),
            ignoredIntervals = emptyList(), excludedRallyIds = setOf("R002"),
            rallyRanges = listOf(ScoreRallyRange(2_000, 6_000, 1_000, 7_000)),
            mergedRanges = listOf(ScoreMergedRange(1_000, 7_000)),
        ))
        val request = PendingEditorExport.Video("synthetic-job", "synthetic-project", "highlights.mp4",
            "content://synthetic/original", "synthetic.mp4", 20_000,
            listOf(1_000, 12_000), listOf(7_000, 16_000), score)
        val id = store().save(request)
        assertEquals(request, store().load(id))
        assertEquals(score, (store().load(id) as PendingEditorExport.Video).scoreSnapshot)
        assertTrue(store().remove(id))
        assertThrows(IOException::class.java) { store().load(id) }
    }

    @Test fun largeProjectPayloadUsesFileAndRestoresExactBytes() {
        val text = "{\"synthetic\":\"" + "abc123".repeat(350_000) + "\"}"
        val request = PendingEditorExport.Text(EditorExportKind.PROJECT, "synthetic-project", "review.json", text)
        val id = store().save(request)
        assertEquals(36, id.length)
        assertEquals(request, store().load(id))
        assertTrue(File(temporary.root, "pending-editor-exports/$id.json").length() > 2_000_000)
    }

    @Test fun chaptersAndProjectSnapshotsRemainIndependentAndPreserveNewlines() {
        val chapters = PendingEditorExport.Text(EditorExportKind.CHAPTERS, "synthetic-A", "chapters.txt", "0:00 Start\n0:10 Rally 2\n")
        val project = PendingEditorExport.Text(EditorExportKind.PROJECT, "synthetic-B", "review.json", "{\"saved\":true}")
        val chapterId = store().save(chapters)
        val projectId = store().save(project)
        assertNotEquals(chapterId, projectId)
        assertTrue(store().remove(chapterId))
        assertEquals(project, store().load(projectId))
        assertThrows(IOException::class.java) { store().load(chapterId) }
    }

    @Test fun unreadableAndFuturePayloadsFailBeforeTheyCanStartAnExport() {
        val id = store().save(PendingEditorExport.Text(EditorExportKind.CHAPTERS, "synthetic", "chapters.txt", "0:00 Start\n"))
        val target = File(temporary.root, "pending-editor-exports/$id.json")
        for (text in listOf("{\"version\":", "{\"version\":2,\"kind\":\"CHAPTERS\"}", "{\"version\":1,\"kind\":\"UNKNOWN\"}")) {
            target.writeText(text)
            val failure = assertThrows(IOException::class.java) { store().load(id) }
            assertTrue(failure.message!!.contains("No export was started"))
            assertEquals(text, target.readText())
        }
    }

    @Test fun invalidRequestIdsCannotReadOrRemoveOtherFiles() {
        val other = temporary.newFile("preserved.txt").apply { writeText("preserved") }
        for (id in listOf("../preserved", "", "wrong-id", "../../preserved.txt")) {
            assertThrows(IOException::class.java) { store().load(id) }
            assertThrows(IllegalArgumentException::class.java) { store().remove(id) }
        }
        assertEquals("preserved", other.readText())
    }

    @Test fun failedPreparationDoesNotClaimAStoredRequest() {
        File(temporary.root, "pending-editor-exports").writeText("synthetic blocked directory")
        assertThrows(IllegalStateException::class.java) {
            store().save(PendingEditorExport.Text(EditorExportKind.PROJECT, "synthetic", "review.json", "{}"))
        }
    }

    @Test fun invalidVideoIntervalsAreRejected() {
        for ((starts, ends) in listOf(listOf<Long>() to listOf(), listOf(2_000L) to listOf(1_000L),
            listOf(2_000L) to listOf(25_000L), listOf(2_000L, 8_000L) to listOf(5_000L))) {
            assertThrows(IllegalArgumentException::class.java) {
                PendingEditorExport.Video("job", "synthetic", "video.mp4", "content://synthetic", "video.mp4", 20_000, starts, ends, "{}")
            }
        }
    }

    @Test fun malformedScoreSnapshotFailsBeforeAnyVideoCanBeQueued() {
        assertThrows(IllegalArgumentException::class.java) {
            PendingEditorExport.Video("job", "synthetic", "video.mp4", "content://synthetic", "video.mp4", 20_000,
                listOf(2_000L), listOf(5_000L), "{\"version\":999}")
        }
    }
}
