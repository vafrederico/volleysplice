import Foundation

public enum TemporalFeatures {
    public static func generate(visual: [Float], rows: Int) throws -> [Float] {
        guard rows >= 0, visual.count == rows * 73 else { throw AnalysisError.invalid("Visual matrix shape mismatch") }
        func column(_ name: String) -> [Float] {
            let index = FeatureSchema.frame.firstIndex(of: name)!
            return (0..<rows).map { visual[$0 * 73 + index] }
        }
        let motion = column("player_motion_mean"), zones = column("player_motion_active_zone_fraction")
        let past = FeatureMath.rollingMean(motion, window: 4, future: false)
        let future = FeatureMath.rollingMean(motion, window: 2, future: true)
        let pastZones = FeatureMath.rollingMean(zones, window: 4, future: false)
        let futureZones = FeatureMath.rollingMean(zones, window: 2, future: true)
        let changes = ["player_motion_centroid_x", "player_motion_centroid_y", "player_motion_spread_x", "player_motion_spread_y"].map { name in
            let values = column(name)
            return zip(FeatureMath.rollingMean(values, window: 4, future: true), FeatureMath.rollingMean(values, window: 4, future: false)).map(-)
        }
        var result: [Float] = []
        for row in 0..<rows {
            let collapse = max(past[row] - future[row], 0)
            let squares = changes.reduce(0.0) { $0 + Double($1[row] * $1[row]) }
            result += [max(future[row] - past[row], 0), collapse,
                collapse * max(pastZones[row] - futureZones[row], 0),
                Float(sqrt(squares)) * min(1, (past[row] + future[row]) / 0.015)]
        }
        return result
    }
}
