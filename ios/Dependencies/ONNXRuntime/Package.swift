// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v15), .macOS(.v14)],
    products: [.library(name: "onnxruntime", type: .static, targets: ["OnnxRuntimeBindings"])],
    targets: [
        .target(
            name: "OnnxRuntimeBindings",
            dependencies: ["onnxruntime"],
            path: "Artifacts/objectivec",
            exclude: ["test"],
            cxxSettings: [.define("SPM_BUILD")]
        ),
        .binaryTarget(name: "onnxruntime", path: "Artifacts/onnxruntime.zip"),
    ],
    cxxLanguageStandard: .cxx17
)
