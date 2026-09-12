// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VolleySplice",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [.library(name: "VolleyCore", targets: ["VolleyCore"]),
               .executable(name: "VolleyProjectCheck", targets: ["VolleyProjectCheck"])],
    targets: [
        .target(name: "VolleyCore"),
        .executableTarget(name: "VolleyProjectCheck", dependencies: ["VolleyCore"]),
        .testTarget(name: "VolleyCoreTests", dependencies: ["VolleyCore"])
    ]
)
