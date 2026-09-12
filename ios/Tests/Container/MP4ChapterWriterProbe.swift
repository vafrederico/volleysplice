// Standalone macOS/iOS container test; NOT part of the app or core Swift package.
// Decode android/app/src/androidTest/assets/overlay-fixture.mp4.b64 to a scratch file
// (base64 decoding only), then compile with App/MP4ChapterWriter.swift:
// xcrun swiftc -parse-as-library MP4ChapterWriter.swift MP4ChapterWriterProbe.swift -o chapter-checks
// ./chapter-checks input.mp4
// Uses AVAsset metadata reads only. Its >4GiB fixture is sparse and removed.
@preconcurrency import AVFoundation
import Foundation
struct YouTubeChapter { var outputMs: Int64; var title: String }
enum ProjectError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case let .invalid(s) = self { return s }; return nil }
}
@main struct ContainerChecks {
    @MainActor static func main() async {
        do {
            let input = URL(fileURLWithPath: CommandLine.arguments[1])
            let seconds = try await AVURLAsset(url: input).load(.duration).seconds
            guard abs(seconds - 1) < 0.002 else { throw ProjectError.invalid("Container tests require the one-second Android overlay fixture") }
            let original = try Data(contentsOf: input)
            let folder = input.deletingLastPathComponent().appendingPathComponent("chapter-checks-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: folder) }
            func file(_ name: String, bytes: Data? = nil) throws -> URL {
                let url = folder.appendingPathComponent(name + ".mp4")
                try (bytes ?? original).write(to: url, options: .withoutOverwriting); return url
            }
            func require(_ yes: Bool, _ message: String) throws { if !yes { throw ProjectError.invalid(message) } }
            let chapters = [YouTubeChapter(outputMs: 0, title: "Rally 1 🏐"), .init(outputMs: 500, title: "Équipe 2 • 1–0")]
            let normal = try file("normal")
            try await MP4ChapterWriter.embed(in: normal, chapters: chapters, durationMs: 1000)
            let result = try Data(contentsOf: normal)
            let changed = zip(original, result.prefix(original.count)).enumerated().filter { $0.element.0 != $0.element.1 }.map(\.offset)
            try require(changed.count == 4, "Writer changed more than the old moov type in original bytes")
            try require(String(decoding: changed.map { original[$0] }, as: UTF8.self) == "moov", "Original changed bytes were not moov")
            try require(String(decoding: changed.map { result[$0] }, as: UTF8.self) == "free", "Replacement changed bytes were not free")
            print("PASS Unicode chapters, native readback, original media bytes/offsets unchanged")
            var duplicateRejected = false
            do {
                try await MP4ChapterWriter.embed(in: normal, chapters: chapters, durationMs: 1000)
            } catch { duplicateRejected = true }
            try require(duplicateRejected && (try Data(contentsOf: normal)) == result, "Duplicate attempt accepted or modified file")
            print("PASS duplicate chapter track rejected without mutation")
            let delayed = try file("delayed")
            try await MP4ChapterWriter.embed(in: delayed, chapters: [.init(outputMs: 250, title: "First"), .init(outputMs: 750, title: "Last")], durationMs: 1000)
            print("PASS nonzero first chapter edit list and exact final duration")
            let badCases: [(String, [YouTubeChapter], Data?)] = [
                ("duplicate-times", [.init(outputMs: 0, title: "A"), .init(outputMs: 0, title: "B")], nil),
                ("end-bound", [.init(outputMs: 1000, title: "A")], nil),
                ("oversize-title", [.init(outputMs: 0, title: String(repeating: "a", count: 65536))], nil),
                ("short-header", chapters, Data([0,0,0,8,109,111,111,118])),
                ("oversize-atom", chapters, Data([255,255,255,255,109,111,111,118])),
                ("short-extended-header", chapters, Data([0,0,0,1,109,111,111,118]))]
            for (name, labels, data) in badCases {
                let url = try file(name, bytes: data), before = try Data(contentsOf: url)
                var rejected = false
                do { try await MP4ChapterWriter.embed(in: url, chapters: labels, durationMs: 1000) } catch { rejected = true }
                try require(rejected && (try Data(contentsOf: url)) == before, "Invalid input mutated or accepted: " + name)
            }
            print("PASS invalid titles, timing, truncated and oversize atoms leave input unchanged")
            let cancelled = try file("cancelled")
            let job = Task { @MainActor in try await MP4ChapterWriter.embed(in: cancelled, chapters: chapters, durationMs: 1000) }
            job.cancel()
            do { _ = try await job.value; throw ProjectError.invalid("Cancelled chapter write completed") } catch is CancellationError {}
            try require(try Data(contentsOf: cancelled) == original, "Cancellation modified original")
            print("PASS pre-cancelled writer leaves input unchanged")
            let sparse = try file("sparse64")
            let handle = try FileHandle(forUpdating: sparse)
            let first = try handle.seekToEnd(), hole: UInt64 = 0x1_0000_0010
            var header = Data([0,0,0,1,102,114,101,101])
            header += Data((0..<8).reversed().map { UInt8((hole >> ($0 * 8)) & 255) })
            try handle.write(contentsOf: header); try handle.truncate(atOffset: first + hole); try handle.close()
            try await MP4ChapterWriter.embed(in: sparse, chapters: chapters, durationMs: 1000)
            let check = try FileHandle(forReadingFrom: sparse); defer { try? check.close() }
            let size = try check.seekToEnd(); try check.seek(toOffset: size - 2048)
            let tail = try check.read(upToCount: 2048)!
            guard let marker = tail.range(of: Data("co64".utf8), options: .backwards) else { throw ProjectError.invalid("Missing chapter co64") }
            let value = tail[(marker.upperBound + 8)..<(marker.upperBound + 16)].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
            try require(value == first + hole + 8 && value > UInt64(UInt32.max), "Chapter offset was truncated to 32 bits")
            print("PASS sparse >4GiB file and 64-bit chapter offset with native readback")
            print("ALL CONTAINER CHECKS PASSED (no video/audio decoding)")
        } catch { print("FAIL \(error)"); exit(1) }
    }
}
