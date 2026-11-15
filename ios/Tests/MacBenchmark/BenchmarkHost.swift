import AppKit

/// Headless XCTest host. Analysis sources and bundled models are shared with iOS.
@main
struct BenchmarkHost {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.prohibited)
        app.run()
    }
}
