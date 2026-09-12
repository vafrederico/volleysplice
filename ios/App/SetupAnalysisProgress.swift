import SwiftUI

/// Only this panel observes frequent queue progress; the editor/setup shell does not.
struct SetupAnalysisProgress: View {
    @ObservedObject var queue: ProcessingQueue
    let sourceName: String?
    let sourceFingerprint: String?
    @AppStorage("show_analysis_measurements") private var showMeasurements = false
    private var job: ProcessingJob? { queue.jobs.last { $0.kind == .analysis && sourceFingerprint != nil && $0.sourceFingerprint == sourceFingerprint } }
    private func steps(_ job: ProcessingJob) -> [AnalysisStepMeasurement] {
        if let measured = job.analysisSteps { return measured }
        var ids: [AnalysisStep] = [.video, .audio, .rally]
        if job.analysis?.prepareScores == true {
            ids.append(.servingSide)
            if job.analysis?.generateSideSwitchMarkers ?? true { ids.append(.sideSwitch) }
        }
        return ids.map { id in
            var row = AnalysisStepMeasurement(id: id)
            if job.state == .completed { row.status = .complete; row.fraction = 1 }
            return row
        }
    }
    var body: some View {
        if let job {
            VStack(alignment: .leading, spacing: 12) {
                Text("GETTING YOUR CLIPS READY").font(.system(size: 11, weight: .bold)).foregroundStyle(SetupPalette.orange)
                Text(job.sourceName).font(.headline).fixedSize(horizontal: false, vertical: true)
                Text(job.state == .queued ? "Waiting to start" : job.state == .completed ? "Your suggested clips are ready" : job.state == .failed ? "VolleySplice could not finish this video" : "VolleySplice is finding the rallies on this device")
                    .font(.system(size: 13)).foregroundStyle(SetupPalette.muted)
                HStack { ProgressView(value: job.progress); Text("\(Int(job.progress * 100))%").font(.system(.caption, design: .monospaced)) }
                ForEach(steps(job)) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: row.status == .complete ? "checkmark.circle.fill" : row.status == .running ? "circle.lefthalf.filled" : "circle")
                                .foregroundStyle(row.status == .complete || row.status == .running ? SetupPalette.green : SetupPalette.muted)
                            Text(row.id.label).font(.system(size: 12, weight: row.status == .running ? .bold : .regular))
                            Spacer()
                            Text(row.status == .complete ? "Ready" : row.status == .pending ? "Waiting" : row.status == .stopped ? "Stopped" : "\(Int(row.fraction * 100))%")
                                .font(.system(size: 11, design: .monospaced))
                        }
                        if row.status != .pending {
                            Text(row.metrics).font(.system(size: 11, design: .monospaced)).foregroundStyle(SetupPalette.muted)
                                .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("analysisMetrics-" + row.id.rawValue)
                        }
                    }
                }
                Text(job.detail).font(.system(size: 12)).foregroundStyle(job.state == .failed ? .red : SetupPalette.muted)
                    .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("analysisStageDetail")
                if showMeasurements, let seconds = job.stageSeconds {
                    ForEach(seconds.keys.sorted(), id: \.self) { key in
                        HStack { Text(key); Spacer(); Text("\(seconds[key] ?? 0, specifier: "%.2f") s") }.font(.caption.monospaced())
                    }
                }
                HStack {
                    if job.state.isActive || job.state == .queued { Button("Cancel analysis") { queue.cancel(job.id) }.accessibilityIdentifier("cancelAnalysis") }
                    if job.state.canResume { Button("Retry") { queue.resume(job.id) }.disabled(!queue.canRun) }
                }.buttonStyle(.bordered)
                Text("Your video stays on this device.").font(.system(size: 11)).foregroundStyle(SetupPalette.muted)
            }.padding(16).overlay(RoundedRectangle(cornerRadius: 10).stroke(SetupPalette.rail)).accessibilityIdentifier("analysisProgressPanel")
        }
    }
}
