import Foundation

/// Small atomic ledger and separate immutable input snapshots; feature caches remain reusable.
@MainActor final class ProcessingJobStore {
    let folder: URL
    init(folder: URL) throws {
        self.folder = folder
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: folder.path)
    }
    func load() throws -> ProcessingJobLedger {
        let file = folder.appendingPathComponent("queue.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return ProcessingJobLedger() }
        let ledger = try JSONDecoder().decode(ProcessingJobLedger.self, from: Data(contentsOf: file))
        try ledger.validate(); return ledger
    }
    func save(_ ledger: ProcessingJobLedger) throws {
        try ledger.validate()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(ledger).write(to: folder.appendingPathComponent("queue.json"), options: .atomic)
    }
    func saveSnapshot(_ project: ProjectDocument, for job: ProcessingJob) throws {
        try job.validate(); try project.save(to: snapshotURL(job))
    }
    func snapshot(_ job: ProcessingJob) throws -> ProjectDocument {
        try job.validate(); return try ProjectDocument.load(from: snapshotURL(job))
    }
    /// Remove only queue-owned immutable snapshots, never source videos, projects or exports.
    func discardUnreferencedSnapshots(in ledger: ProcessingJobLedger) throws {
        let retained = Set(ledger.jobs.map { $0.id + ".project.json" })
        for url in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) {
            let name = url.lastPathComponent
            guard name.hasSuffix(".project.json"), UUID(uuidString: String(name.dropLast(".project.json".count))) != nil,
                  !retained.contains(name) else { continue }
            try FileManager.default.removeItem(at: url)
        }
    }
    private func snapshotURL(_ job: ProcessingJob) -> URL { folder.appendingPathComponent(job.id + ".project.json") }
}
