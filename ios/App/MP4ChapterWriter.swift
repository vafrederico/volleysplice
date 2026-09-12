@preconcurrency import AVFoundation
import Foundation

/// Apple TN2174 chapter track authoring. AVMutableMovie cannot emit MP4 headers
/// on the tested OS (-16343), so preserve the existing media offsets and append
/// a disabled tx3g track plus a replacement moov. No media decoding/reencoding.
/// https://developer.apple.com/library/archive/technotes/tn2174/_index.html
@MainActor enum MP4ChapterWriter {
    private static let maximumMoovSize = 64 * 1024 * 1024
    private static let maximumTextSize = 16 * 1024 * 1024
    private struct Atom {
        let type: String
        let start: Int
        let size: Int
        let header: Int
        var payload: Int { start + header }
        var end: Int { start + size }
    }
    @discardableResult
    static func embed(in stagedMP4: URL, chapters: [YouTubeChapter], durationMs: Int64) async throws -> Int {
        try Task.checkCancellation()
        guard !chapters.isEmpty else { return 0 }
        guard stagedMP4.isFileURL, stagedMP4.pathExtension.lowercased() == "mp4", durationMs > 0 else {
            throw invalid("Chapter writing requires a completed MP4 export")
        }
        var text = Data(), sizes: [UInt32] = [], durations: [UInt32] = []
        for (index, chapter) in chapters.enumerated() {
            let end = index + 1 < chapters.count ? chapters[index + 1].outputMs : durationMs
            guard chapter.outputMs >= 0, end > chapter.outputMs, end <= durationMs,
                  end - chapter.outputMs <= Int64(UInt32.max) else { throw invalid("Invalid embedded chapter timing") }
            let utf8 = Data(chapter.title.utf8)
            guard !utf8.isEmpty, utf8.count <= Int(UInt16.max), text.count + utf8.count + 2 <= maximumTextSize else {
                throw invalid("Chapter titles exceed the MP4 chapter metadata limit")
            }
            text += be16(UInt16(utf8.count)); text += utf8
            sizes.append(UInt32(utf8.count + 2)); durations.append(UInt32(end - chapter.outputMs))
        }
        let file = try FileHandle(forUpdating: stagedMP4)
        defer { try? file.close() }
        let originalSize = try file.seekToEnd()
        var position: UInt64 = 0, moovPosition: UInt64?, moov = Data(), foundFTYP = false
        // Read only top-level headers and the bounded moov. mdat can exceed 4GiB.
        while position < originalSize {
            try Task.checkCancellation()
            guard originalSize - position >= 8 else { throw invalid("Truncated MP4 atom header") }
            try file.seek(toOffset: position)
            let header = try read(file, count: Int(min(16, originalSize - position)))
            let shortSize = try integer(header, at: 0, bytes: 4)
            let type = String(decoding: header[4..<8], as: UTF8.self)
            guard shortSize != 0 else { throw invalid("Open-ended MP4 atoms cannot receive embedded chapters") }
            let length = shortSize == 1 ? try integer(header, at: 8, bytes: 8) : shortSize
            guard length >= (shortSize == 1 ? 16 : 8), length <= originalSize - position else { throw invalid("Invalid MP4 atom size") }
            if type == "moof" { throw invalid("Fragmented MP4 chapter writing is unsupported") }
            if type == "ftyp" {
                guard length >= 16, length <= 4096 else { throw invalid("Invalid MP4 file type atom") }
                try file.seek(toOffset: position)
                let ftyp = try read(file, count: Int(length))
                let offset = shortSize == 1 ? 16 : 8
                guard ftyp.count >= offset + 8 else { throw invalid("Truncated MP4 file type") }
                let brand = String(decoding: ftyp[offset..<(offset + 4)], as: UTF8.self)
                guard brand != "qt  " else { throw invalid("A QuickTime MOV cannot be written as MP4") }
                foundFTYP = true
            }
            if type == "moov" {
                guard moovPosition == nil, length <= UInt64(maximumMoovSize) else { throw invalid("Unsupported MP4 movie header") }
                moovPosition = position; try file.seek(toOffset: position); moov = try read(file, count: Int(length))
            }
            position += length
        }
        guard foundFTYP, let moovPosition else { throw invalid("Missing MP4 file type or movie header") }
        let root = try atoms(moov, start: 0, end: moov.count)
        guard root.count == 1, root[0].type == "moov" else { throw invalid("Invalid movie header") }
        let children = try atoms(moov, start: root[0].payload, end: root[0].end)
        guard let mvhd = children.first(where: { $0.type == "mvhd" }) else { throw invalid("Missing movie timescale") }
        guard mvhd.size >= mvhd.header + 4 else { throw invalid("Truncated movie header") }
        let version = moov[mvhd.payload]
        guard version <= 1, mvhd.size >= mvhd.header + (version == 1 ? 112 : 100) else { throw invalid("Unsupported movie header version or size") }
        let timescale = try integer(moov, at: mvhd.payload + (version == 1 ? 20 : 12), bytes: 4)
        guard timescale > 0 else { throw invalid("Invalid movie timescale") }
        var largestID: UInt64 = 0, videoIndex: Int?
        for (index, child) in children.enumerated() where child.type == "trak" {
            let parts = try atoms(moov, start: child.payload, end: child.end)
            guard let tkhd = parts.first(where: { $0.type == "tkhd" }), tkhd.size >= tkhd.header + 4,
                  moov[tkhd.payload] <= 1, tkhd.size >= tkhd.header + (moov[tkhd.payload] == 1 ? 96 : 84) else { throw invalid("Invalid MP4 track header") }
            largestID = max(largestID, try integer(moov, at: tkhd.payload + (moov[tkhd.payload] == 1 ? 20 : 12), bytes: 4))
            if let reference = parts.first(where: { $0.type == "tref" }) {
                guard !(try atoms(moov, start: reference.payload, end: reference.end)).contains(where: { $0.type == "chap" }) else {
                    throw invalid("The staged export already contains a chapter track")
                }
            }
            if let mdia = parts.first(where: { $0.type == "mdia" }),
               let handler = try atoms(moov, start: mdia.payload, end: mdia.end).first(where: { $0.type == "hdlr" }) {
                guard handler.size >= handler.header + 12 else { throw invalid("Invalid MP4 handler") }
                let type = String(decoding: moov[(handler.payload + 8)..<(handler.payload + 12)], as: UTF8.self)
                let enabled = try integer(moov, at: tkhd.payload, bytes: 4) & 1 != 0
                if type == "vide" && enabled && videoIndex == nil { videoIndex = index }
            }
        }
        guard let videoIndex, largestID < UInt64(UInt32.max - 1), originalSize <= UInt64.max - UInt64(text.count + 8) else {
            throw invalid("Cannot associate MP4 chapters with an enabled video track")
        }
        let chapterID = UInt32(largestID + 1)
        let ticks = UInt64(durationMs).multipliedReportingOverflow(by: timescale)
        let firstTicks = UInt64(chapters[0].outputMs).multipliedReportingOverflow(by: timescale)
        guard !ticks.overflow, !firstTicks.overflow else { throw invalid("Chapter duration exceeds MP4 timing capacity") }
        let totalTicks = ticks.partialValue / 1000, startTicks = firstTicks.partialValue / 1000
        let newTrack = try chapterTrack(id: chapterID, durationMs: UInt64(durationMs), startMs: UInt64(chapters[0].outputMs),
            totalTicks: totalTicks, startTicks: startTicks, sizes: sizes, durations: durations, offset: originalSize + 8)
        var moviePayload = Data()
        for (index, child) in children.enumerated() {
            if child.type == "mvhd" {
                var header = Data(moov[child.start..<child.end])
                guard header.count >= 4 else { throw invalid("Invalid movie header") }
                header.replaceSubrange((header.count - 4)..<header.count, with: be32(chapterID + 1))
                moviePayload += header
            } else if index == videoIndex {
                var parts = Data(), associated = false
                for part in try atoms(moov, start: child.payload, end: child.end) {
                    if part.type == "tref" {
                        parts += try atom("tref", Data(moov[part.payload..<part.end]) + atom("chap", be32(chapterID)))
                        associated = true
                    } else { parts += moov[part.start..<part.end] }
                }
                if !associated { parts += try atom("tref", atom("chap", be32(chapterID))) }
                moviePayload += try atom("trak", parts)
            } else { moviePayload += moov[child.start..<child.end] }
        }
        moviePayload += newTrack
        let newMoov = try atom("moov", moviePayload)
        guard newMoov.count <= maximumMoovSize else { throw invalid("Embedded chapter header exceeds metadata limit") }
        // Append first, flush, and only then retire the old moov. Existing stco/
        // co64 offsets remain valid because no pre-existing media bytes move.
        do {
            try Task.checkCancellation()
            try file.seek(toOffset: originalSize)
            try file.write(contentsOf: atom("mdat", text)); try file.write(contentsOf: newMoov)
            try file.synchronize(); try Task.checkCancellation()
            try file.seek(toOffset: moovPosition + 4); try file.write(contentsOf: Data("free".utf8)); try file.synchronize()
            try Task.checkCancellation()
            try await verify(in: stagedMP4, chapters: chapters, durationMs: durationMs)
        } catch {
            // The caller only passes its private staging file. Restore the
            // original container on any failure before it could be published.
            try? file.seek(toOffset: moovPosition + 4); try? file.write(contentsOf: Data("moov".utf8))
            try? file.truncate(atOffset: originalSize); try? file.synchronize()
            throw error
        }
        return chapters.count
    }

    private static func chapterTrack(id: UInt32, durationMs: UInt64, startMs: UInt64, totalTicks: UInt64,
                                      startTicks: UInt64, sizes: [UInt32], durations: [UInt32], offset: UInt64) throws -> Data {
        let identity = [UInt32(0x10000),0,0,0,0x10000,0,0,0,0x40000000].reduce(into: Data()) { $0 += be32($1) }
        let tkhd = try atom("tkhd", concat(full(1, 6), be64(0), be64(0), be32(id), be32(0), be64(totalTicks),
            Data(repeating: 0, count: 16), identity, be32(0), be32(0)))
        var edit = Data()
        if startMs > 0 {
            let entries = concat(be32(2), be64(startTicks), be64(UInt64.max), be16(1), be16(0),
                be64(totalTicks - startTicks), be64(0), be16(1), be16(0))
            edit = try atom("edts", atom("elst", full(1) + entries))
        }
        let mdhd = try atom("mdhd", concat(full(1), be64(0), be64(0), be32(1000), be64(durationMs - startMs), be16(0x55c4), be16(0)))
        let hdlr = try atom("hdlr", concat(full(), be32(0), Data("text".utf8), Data(repeating: 0, count: 12), Data("Chapter titles\0".utf8)))
        let fonts = try atom("ftab", be16(1) + be16(1) + Data([9]) + Data("Helvetica".utf8))
        let sampleEntry = try atom("tx3g", concat(Data(repeating: 0, count: 6), be16(1), be32(0),
            Data(repeating: 0, count: 14), be16(0), be16(0), be16(1), Data([0,12,255,255,255,255]), fonts))
        let stsd = try atom("stsd", full() + be32(1) + sampleEntry)
        var runs: [(UInt32, UInt32)] = []
        for duration in durations {
            if runs.last?.1 == duration { runs[runs.count - 1].0 += 1 } else { runs.append((1, duration)) }
        }
        var timing = full() + be32(UInt32(runs.count))
        for run in runs { timing += be32(run.0) + be32(run.1) }
        let stts = try atom("stts", timing)
        let stsc = try atom("stsc", full() + be32(1) + be32(1) + be32(UInt32(sizes.count)) + be32(1))
        let stsz = try atom("stsz", full() + be32(0) + be32(UInt32(sizes.count)) + sizes.reduce(into: Data()) { $0 += be32($1) })
        let co64 = try atom("co64", full() + be32(1) + be64(offset))
        let stbl = try atom("stbl", stsd + stts + stsc + stsz + co64)
        let dinf = try atom("dinf", atom("dref", full() + be32(1) + atom("url ", full(0, 1))))
        let minf = try atom("minf", atom("nmhd", full()) + dinf + stbl)
        return try atom("trak", tkhd + edit + atom("mdia", mdhd + hdlr + minf))
    }
    private static func verify(in url: URL, chapters: [YouTubeChapter], durationMs: Int64) async throws {
        let groups = try await AVURLAsset(url: url).loadChapterMetadataGroups(bestMatchingPreferredLanguages: ["und", "en"])
        guard groups.count == chapters.count else { throw invalid("Embedded MP4 chapter count failed verification (\(groups.count)/\(chapters.count))") }
        for (index, group) in groups.enumerated() {
            try Task.checkCancellation()
            let end = index + 1 < chapters.count ? chapters[index + 1].outputMs : durationMs
            guard abs(group.timeRange.start.seconds - Double(chapters[index].outputMs) / 1000) < 0.002,
                  abs(CMTimeRangeGetEnd(group.timeRange).seconds - Double(end) / 1000) < 0.002,
                  let title = group.items.first(where: { $0.commonKey == .commonKeyTitle }),
                  try await title.load(.stringValue) == chapters[index].title else { throw invalid("Embedded MP4 chapter title or timing failed verification") }
        }
    }
    private static func atoms(_ data: Data, start: Int, end: Int) throws -> [Atom] {
        guard start >= 0, end <= data.count, start <= end else { throw invalid("Invalid MP4 child range") }
        var output: [Atom] = [], cursor = start
        while cursor < end {
            guard end - cursor >= 8 else { throw invalid("Truncated MP4 child atom") }
            let short = try integer(data, at: cursor, bytes: 4), header = short == 1 ? 16 : 8
            let size = short == 1 ? try integer(data, at: cursor + 8, bytes: 8) : short
            guard size >= UInt64(header), size <= UInt64(end - cursor) else { throw invalid("Invalid MP4 child size") }
            output.append(Atom(type: String(decoding: data[(cursor + 4)..<(cursor + 8)], as: UTF8.self), start: cursor, size: Int(size), header: header))
            cursor += Int(size)
        }
        return output
    }
    private static func integer(_ data: Data, at offset: Int, bytes: Int) throws -> UInt64 {
        guard offset >= 0, offset <= data.count - bytes else { throw invalid("Truncated MP4 integer") }
        return data[offset..<(offset + bytes)].reduce(0) { ($0 << 8) | UInt64($1) }
    }
    private static func read(_ file: FileHandle, count: Int) throws -> Data {
        guard let data = try file.read(upToCount: count), data.count == count else { throw invalid("Truncated MP4 file") }; return data
    }
    private static func atom(_ type: String, _ payload: Data) throws -> Data {
        guard type.utf8.count == 4, payload.count <= Int(UInt32.max) - 8 else { throw invalid("Oversize MP4 atom") }
        return be32(UInt32(payload.count + 8)) + Data(type.utf8) + payload
    }
    private static func concat(_ pieces: Data...) -> Data { pieces.reduce(into: Data()) { $0.append($1) } }
    private static func full(_ version: UInt8 = 0, _ flags: UInt32 = 0) -> Data { Data([version, UInt8((flags >> 16) & 255), UInt8((flags >> 8) & 255), UInt8(flags & 255)]) }
    private static func be16(_ n: UInt16) -> Data { Data([UInt8(n >> 8), UInt8(n & 255)]) }
    private static func be32(_ n: UInt32) -> Data { Data((0..<4).reversed().map { UInt8((n >> ($0 * 8)) & 255) }) }
    private static func be64(_ n: UInt64) -> Data { Data((0..<8).reversed().map { UInt8((n >> ($0 * 8)) & 255) }) }
    private static func invalid(_ text: String) -> ProjectError { .invalid(text) }
}
