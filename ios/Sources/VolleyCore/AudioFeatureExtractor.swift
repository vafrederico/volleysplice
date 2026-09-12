import Foundation

/// Streaming mono PCM DSP matching Android's AudioFeatureExtractor. Feed source
/// presentation timestamps (including negative priming), then pool once decoding ends.
/// This mutable extractor belongs to one decoding task and is not thread safe.
public final class AudioFeatureExtractor {
    public static let targetSampleRate = 16_000
    public static let featureNames = [
        "audio_available", "audio_rms", "audio_peak", "audio_peak_to_rms",
        "audio_noise_floor", "audio_snr", "audio_spectral_flux", "audio_rms_novelty",
        "audio_onset_strength", "audio_contact_like_transient", "audio_onset_cadence",
        "audio_cadence_collapse", "audio_seconds_since_transient",
        "audio_noise_removed_broadband", "audio_noise_normalized_flux",
        "audio_band_80_250_snr", "audio_band_80_250_snr_flux",
        "audio_band_250_500_snr", "audio_band_250_500_snr_flux",
        "audio_band_500_1000_snr", "audio_band_500_1000_snr_flux",
        "audio_band_1000_2000_snr", "audio_band_1000_2000_snr_flux",
        "audio_band_2000_4000_snr", "audio_band_2000_4000_snr_flux",
        "audio_band_4000_7800_snr", "audio_band_4000_7800_snr_flux"
    ]
    private static let frameSamples = 800
    private static let frameSeconds = 0.05
    private static let fftSize = 1024
    private static let bandBounds = [(80, 250), (250, 500), (500, 1000),
                                     (1000, 2000), (2000, 4000), (4000, 7800)]
    private static let meanPooled: Set<Int> = [1, 4, 5, 10, 11, 12]
    private let fft = Radix2FFT(size: AudioFeatureExtractor.fftSize)
    private let window: [Float]
    private let bandIndexes: [[Int]]
    private var rms: [Float] = [], peak: [Float] = [], spectralFlux: [Float] = []
    private var bandPower = [[Float]](repeating: [], count: 6)
    private var previousSpectrum: [Float]?
    private var sourceRate = 0
    private var nextSourceFrame: Int64 = 0
    private var sourceBuffer: [Float] = []
    private var sourcePosition = 0.0
    private var outputFrame = [Float](repeating: 0, count: AudioFeatureExtractor.frameSamples)
    private var outputFrameLength = 0
    private var finished = false
    public private(set) var resampledOutputSamples: Int64 = 0
    public private(set) var performanceMilliseconds: [String: Double] = [:]
    public var audioFeatureFrameCount: Int { rms.count }

    public init() {
        window = (0..<Self.frameSamples).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(Self.frameSamples - 1)))
        }
        bandIndexes = Self.bandBounds.map { lower, upper in
            (0...Self.fftSize / 2).filter { bin in
                let frequency = Double(bin * Self.targetSampleRate) / Double(Self.fftSize)
                return frequency >= Double(lower) && frequency < Double(upper)
            }
        }
    }

    public func push(_ mono: [Float], timestampSeconds: Double, sampleRate: Int) throws {
        guard !finished else { throw AnalysisError.invalid("Cannot push PCM after finishing audio") }
        guard sampleRate > 0, sampleRate <= 768_000, timestampSeconds.isFinite,
              abs(timestampSeconds) < Double(Int64.max / Int64(sampleRate)) - 1,
              mono.allSatisfy({ $0.isFinite && abs($0) <= 1 }) else {
            throw AnalysisError.invalid("Expected normalized finite PCM, a valid timestamp and sample rate")
        }
        guard sourceRate == 0 || sourceRate == sampleRate else {
            throw AnalysisError.invalid("Audio sample rate changed within the source track")
        }
        let started = Date.timeIntervalSinceReferenceDate
        defer { record("timeline_and_resample", started) }
        sourceRate = sampleRate
        // Java Math.round is floor(x + 0.5), including negative half ties.
        let startFrame = Int64(floor(timestampSeconds * Double(sampleRate) + 0.5))
        let primingFrames = Int(min(Int64(mono.count), max(0, -startFrame)))
        let presentedFrames = mono.count - primingFrames
        if presentedFrames == 0 { return }
        let presentedStart = startFrame + Int64(primingFrames)
        let gapFrames = max(0, presentedStart - nextSourceFrame)
        let overlapFrames = Int(min(Int64(presentedFrames), max(0, nextSourceFrame - presentedStart)))
        let trimFrames = primingFrames + overlapFrames
        if gapFrames > 0 { appendSilence(gapFrames) }
        if trimFrames < mono.count { appendSource(Array(mono[trimFrames...])) }
        nextSourceFrame = max(nextSourceFrame, presentedStart + Int64(presentedFrames))
    }

    /// Returns row-major raw audio features in `featureNames` order. Pooling uses
    /// inclusive +/- 0.125-second windows at the caller's 4 Hz video timestamps.
    public func finishAndPool(_ analysisTimes: [Double], progress: (Double, String) -> Void = { _, _ in }) throws -> [Float] {
        guard analysisTimes.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw AnalysisError.invalid("Invalid audio pooling timestamps")
        }
        let started = Date.timeIntervalSinceReferenceDate
        defer { record("finish_and_pool", started) }
        progress(0.82, "Flushing resampler + final FFT frames")
        if !finished {
            if !sourceBuffer.isEmpty {
                while sourcePosition < Double(sourceBuffer.count) {
                    // Android flush intentionally uses the final source sample
                    // without the interpolation/quantization of regular output.
                    pushOutput(sourceBuffer[min(sourceBuffer.count - 1, Int(floor(sourcePosition)))])
                    sourcePosition += Double(sourceRate) / Double(Self.targetSampleRate)
                }
            }
            if outputFrameLength > 0 {
                for i in outputFrameLength..<Self.frameSamples { outputFrame[i] = 0 }
                processFrame(outputFrame)
                outputFrameLength = 0
            }
            sourceBuffer.removeAll()
            finished = true
        }
        let columns = Self.featureNames.count
        var output = [Float](repeating: 0, count: analysisTimes.count * columns)
        if rms.isEmpty {
            progress(0.99, "No decoded audio frames to pool")
            return output
        }
        progress(0.88, "Computing audio ranks + rolling noise floors")
        let reductionStarted = Date.timeIntervalSinceReferenceDate
        let sources = try frameFeatureSources()
        record("whole_recording_audio_reductions", reductionStarted)
        let audioTimes = rms.indices.map { (Double($0) + 0.5) * Self.frameSeconds }
        progress(0.95, "Pooling audio features to 4 Hz video timestamps")
        let poolingStarted = Date.timeIntervalSinceReferenceDate
        for row in analysisTimes.indices {
            output[row * columns] = 1
            var left = Self.bound(audioTimes, analysisTimes[row] - 0.125, upper: false)
            var right = Self.bound(audioTimes, analysisTimes[row] + 0.125, upper: true)
            if right <= left {
                let position = floor((analysisTimes[row] / Self.frameSeconds - 0.5) + 0.5)
                left = Int(min(Double(audioTimes.count - 1), max(0, position)))
                right = left + 1
            }
            for column in 1..<columns {
                if Self.meanPooled.contains(column) {
                    var sum = 0.0
                    for i in left..<right { sum += Double(sources[column][i]) }
                    output[row * columns + column] = Float(sum / Double(right - left))
                } else {
                    var maximum = -Float.greatestFiniteMagnitude
                    for i in left..<right { maximum = max(maximum, sources[column][i]) }
                    output[row * columns + column] = maximum
                }
            }
        }
        record("pool_to_analysis_timestamps", poolingStarted)
        progress(0.99, "Audio DSP complete; preparing feature cache")
        return output
    }

    private func appendSource(_ mono: [Float]) {
        sourceBuffer.append(contentsOf: mono)
        let step = Double(sourceRate) / Double(Self.targetSampleRate)
        while sourcePosition + 1 < Double(sourceBuffer.count) {
            let lower = Int(floor(sourcePosition)), fraction = sourcePosition - floor(sourcePosition)
            let value = Double(sourceBuffer[lower]) * (1 - fraction) + Double(sourceBuffer[lower + 1]) * fraction
            let quantized = floor(value * 32768 + 0.5)
            pushOutput(Float(max(-32768, min(32767, quantized))) / 32768)
            sourcePosition += step
        }
        let consumed = min(sourceBuffer.count, Int(floor(sourcePosition)))
        if consumed > 0 {
            sourceBuffer.removeFirst(consumed)
            sourcePosition -= Double(consumed)
        }
    }

    private func appendSilence(_ frameCount: Int64) {
        let blockSize = Int(min(frameCount, Int64(max(1, sourceRate))))
        let silence = [Float](repeating: 0, count: blockSize)
        var remaining = frameCount
        while remaining > 0 {
            let length = Int(min(remaining, Int64(blockSize)))
            appendSource(length == blockSize ? silence : [Float](repeating: 0, count: length))
            remaining -= Int64(length)
        }
    }

    private func pushOutput(_ value: Float) {
        resampledOutputSamples += 1
        outputFrame[outputFrameLength] = value
        outputFrameLength += 1
        if outputFrameLength == Self.frameSamples {
            processFrame(outputFrame)
            outputFrameLength = 0
        }
    }

    private func processFrame(_ frame: [Float]) {
        let started = Date.timeIntervalSinceReferenceDate
        defer { record("fft_and_frame_features", started) }
        var squareTotal = 0.0, maximum: Float = 0
        var fftInput = [Float](repeating: 0, count: Self.fftSize)
        for i in 0..<Self.frameSamples {
            let value = frame[i]
            squareTotal += Double(value * value)
            maximum = max(maximum, abs(value))
            fftInput[i] = value * window[i]
        }
        rms.append(Float(sqrt(squareTotal / Double(Self.frameSamples))))
        peak.append(maximum)
        var spectrum = fft.magnitudes(fftInput)
        var spectrumTotal = 0.0
        for value in spectrum { spectrumTotal += Double(value) }
        for band in bandIndexes.indices {
            var power = 0.0
            for bin in bandIndexes[band] { power += Double(spectrum[bin] * spectrum[bin]) }
            bandPower[band].append(Float(power))
        }
        let divisor = max(spectrumTotal, 1e-8)
        for bin in spectrum.indices { spectrum[bin] = Float(Double(spectrum[bin]) / divisor) }
        var fluxSquares = 0.0
        if let previousSpectrum {
            for bin in spectrum.indices {
                let difference = Double(max(spectrum[bin] - previousSpectrum[bin], 0))
                fluxSquares += difference * difference
            }
        }
        spectralFlux.append(Float(sqrt(fluxSquares)))
        previousSpectrum = spectrum
    }

    private func frameFeatureSources() throws -> [[Float]] {
        let count = rms.count
        var novelty = [Float](repeating: 0, count: count)
        for i in 1..<count { novelty[i] = max(rms[i] - rms[i - 1], 0) }
        let peakRank = try FeatureMath.ranks(peak, rows: count, columns: 1)
        let noveltyRank = try FeatureMath.ranks(novelty, rows: count, columns: 1)
        let fluxRank = try FeatureMath.ranks(spectralFlux, rows: count, columns: 1)
        var onset = [Float](repeating: 0, count: count), contact = onset
        for i in 0..<count {
            onset[i] = 0.45 * fluxRank[i] + 0.35 * noveltyRank[i] + 0.2 * peakRank[i]
            contact[i] = onset[i] * Float(sqrt(Double(peakRank[i])))
        }
        let strongThreshold = max(Float(0.72), FeatureMath.quantile(contact, 0.85))
        let cadence = FeatureMath.rollingMean(contact, window: 40, future: false)
        let futureCadence = FeatureMath.rollingMean(contact, window: 20, future: true)
        var collapse = [Float](repeating: 0, count: count)
        var elapsed = [Float](repeating: 10, count: count)
        var lastTime: Double?
        for i in 0..<count {
            collapse[i] = max(cadence[i] - futureCadence[i], 0)
            let timestamp = (Double(i) + 0.5) * Self.frameSeconds
            if contact[i] >= strongThreshold && Double(peakRank[i]) >= 0.6 {
                lastTime = timestamp
                elapsed[i] = 0
            } else if let lastTime { elapsed[i] = Float(min(10, timestamp - lastTime)) }
        }
        let noiseFloor = Self.rollingPercentile(rms, window: 200, percentile: 0.2)
        var snr = [Float](repeating: 0, count: count), peakToRMS = snr
        for i in 0..<count {
            snr[i] = Float(log1p(Double(max(rms[i] - noiseFloor[i], 0)) / (Double(noiseFloor[i]) + 1e-4)))
            peakToRMS[i] = min(30, max(0, peak[i] / (rms[i] + Float(1e-5))))
        }
        var sources = [[Float]](repeating: [], count: Self.featureNames.count)
        sources[1] = rms; sources[2] = peak; sources[3] = peakToRMS
        sources[4] = noiseFloor; sources[5] = snr; sources[6] = spectralFlux
        sources[7] = novelty; sources[8] = onset; sources[9] = contact
        sources[10] = cadence; sources[11] = collapse; sources[12] = elapsed
        var numerator = [Float](repeating: 0, count: count), denominator = numerator, fluxSquares = numerator
        for band in bandPower.indices {
            let power = bandPower[band]
            let floor = Self.rollingPercentile(power, window: 200, percentile: 0.2)
            var bandSNR = [Float](repeating: 0, count: count), bandFlux = bandSNR
            for i in 0..<count {
                let excess = max(power[i] - floor[i], 0)
                bandSNR[i] = Float(log1p(Double(excess) / (Double(floor[i]) + 1e-8)))
                bandFlux[i] = max(bandSNR[i] - bandSNR[max(0, i - 1)], 0)
                numerator[i] += excess
                denominator[i] += floor[i]
                fluxSquares[i] += bandFlux[i] * bandFlux[i]
            }
            sources[15 + 2 * band] = bandSNR
            sources[16 + 2 * band] = bandFlux
        }
        sources[13] = (0..<count).map { Float(log1p(Double(numerator[$0]) / (Double(denominator[$0]) + 1e-8))) }
        sources[14] = fluxSquares.map { Float(sqrt(Double($0))) }
        return sources
    }

    /// Preserve Android's integral-position zero behavior for feature parity.
    /// Correcting that behavior requires a coordinated model/Android revision.
    static func rollingPercentile(_ values: [Float], window: Int, percentile: Double) -> [Float] {
        var sorted: [Float] = [], output = [Float](repeating: 0, count: values.count)
        for i in values.indices {
            func insertion(_ value: Float) -> Int {
                var left = 0, right = sorted.count
                while left < right {
                    let mid = (left + right) / 2
                    if sorted[mid] < value { left = mid + 1 } else { right = mid }
                }
                return left
            }
            sorted.insert(values[i], at: insertion(values[i]))
            if i >= window { sorted.remove(at: insertion(values[i - window])) }
            let position = Double(sorted.count - 1) * percentile
            let lower = Int(floor(position)), upper = Int(ceil(position))
            output[i] = Float(Double(sorted[lower]) * (Double(upper) - position)
                              + Double(sorted[upper]) * (position - Double(lower)))
        }
        return output
    }

    private static func bound(_ values: [Double], _ target: Double, upper: Bool) -> Int {
        var left = 0, right = values.count
        while left < right {
            let mid = (left + right) / 2
            if upper ? values[mid] <= target : values[mid] < target { left = mid + 1 } else { right = mid }
        }
        return left
    }

    private func record(_ name: String, _ started: Double) {
        performanceMilliseconds[name, default: 0] += (Date.timeIntervalSinceReferenceDate - started) * 1000
    }
}

/// Float input/output with Double butterfly intermediates, as on Android.
struct Radix2FFT {
    let size: Int
    private let levels: Int
    private let cosines: [Double], sines: [Double]
    init(size: Int) {
        precondition(size > 0 && size.nonzeroBitCount == 1, "FFT size must be a power of two")
        self.size = size
        levels = size.trailingZeroBitCount
        cosines = (0..<size / 2).map { cos(2 * Double.pi * Double($0) / Double(size)) }
        sines = (0..<size / 2).map { sin(2 * Double.pi * Double($0) / Double(size)) }
    }

    func magnitudes(_ input: [Float]) -> [Float] {
        precondition(input.count == size, "FFT input size mismatch")
        var real = [Double](repeating: 0, count: size), imaginary = real
        for i in 0..<size {
            var reversed = 0, bits = i
            for _ in 0..<levels { reversed = (reversed << 1) | (bits & 1); bits >>= 1 }
            real[reversed] = Double(input[i])
        }
        var block = 2
        while block <= size {
            let half = block >> 1, tableStep = size / block
            for start in stride(from: 0, to: size, by: block) {
                for j in 0..<half {
                    let table = j * tableStep, even = start + j, odd = even + half
                    let oddReal = real[odd] * cosines[table] + imaginary[odd] * sines[table]
                    let oddImaginary = -real[odd] * sines[table] + imaginary[odd] * cosines[table]
                    real[odd] = real[even] - oddReal
                    imaginary[odd] = imaginary[even] - oddImaginary
                    real[even] += oddReal
                    imaginary[even] += oddImaginary
                }
            }
            block <<= 1
        }
        return (0...size / 2).map { Float(hypot(real[$0], imaginary[$0])) }
    }
}
