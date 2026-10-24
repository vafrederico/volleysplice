import TokamakStaticHTML
import TokamakCore
import Foundation

// StaticHTML lacks SwiftUI's convenience modifier. This sets the environment
// consumed by the real button style; it does not establish interactive behavior.
extension View {
    func disabled(_ disabled: Bool) -> some View { environment(\.isEnabled, !disabled) }
}

/// Synthetic composition around the app's exact layout-only control source.
/// Static HTML has no AVPlayer, UIKit, native gestures, or live SwiftUI state.
struct Preview: View {
    let width: Double
    let landscape: Bool
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("VolleySplice").font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("Rallies   Editor   Export").font(.system(size: 12))
            }.padding(12).frame(width: width)
            HStack { Text("2 need review"); Spacer(); Text("0 / 2 checked") }
                .font(.system(size: 13)).foregroundColor(CourtsidePalette.muted).padding(12).frame(width: width)
            Rectangle().fill(CourtsidePalette.control).frame(height: 3)
            if landscape {
                HStack(alignment: .top, spacing: 10) {
                    media.frame(width: width * 0.53, height: 270)
                    controls.frame(width: width * 0.47 - 30).padding(10)
                }
            } else {
                media.frame(width: width, height: 230)
                controls.frame(width: width - 24).padding(12)
            }
        }.frame(width: width).background(CourtsidePalette.ground).foregroundColor(CourtsidePalette.ink)
    }
    var media: some View {
        VStack(spacing: 12) {
            HStack { Text("Cedar"); Spacer(); Text("0 : 0"); Spacer(); Text("Harbor") }.padding(12)
            Spacer()
            Text("Synthetic video placeholder").font(.system(size: 13))
            Text("AVPlayer is not rendered here").font(.system(size: 11))
            Spacer()
            HStack { Text("▶   0:09.0"); Spacer(); Text("5.0 s     1×") }.padding(12)
        }.foregroundColor(.white).background(Color.black)
    }
    var controls: some View {
        VStack(spacing: 10) {
            HStack {
                CourtsideTrimNudges(start: true, onNudge: { _ in })
                Spacer()
                Text("Rally 2").font(.system(size: 15, weight: .semibold))
                Spacer()
                CourtsideTrimNudges(start: false, onNudge: { _ in })
            }
            VStack(spacing: 6) {
                HStack { Text("‹ Previous · 0:01"); Spacer(); Text("Next · 60:00 ›") }.font(.system(size: 11))
                HStack(spacing: 3) {
                    Spacer()
                    Rectangle().fill(CourtsidePalette.accent).frame(width: 7, height: 35)
                    Rectangle().fill(Color(red: 0.87, green: 0.47, blue: 0.35)).frame(width: 84, height: 30)
                    Rectangle().fill(CourtsidePalette.accent).frame(width: 7, height: 35)
                    Spacer()
                }
            }.padding(8).background(CourtsidePalette.control.opacity(0.3))
            HStack { Text("Start 0:10.0"); Spacer(); Text("End 0:15.0") }.font(.system(size: 12)).foregroundColor(CourtsidePalette.muted)
            CourtsideDecisionRow(selected: nil, onDecision: { _ in })
            CourtsideServeRow(selection: 0, canAdd: true, hasMarker: true, onSelect: { _ in })
            CourtsideReviewNavigation(previous: true, pending: true, next: true, onNeighbor: { _ in }, onReview: {})
        }
    }
}

let width = Double(CommandLine.arguments.dropFirst().first ?? "390") ?? 390
let html = StaticHTMLRenderer(Preview(width: width, landscape: width >= 800)).render()
let notice = "<p style=\"max-width:700px;padding:12px;font:13px system-ui;color:#131416\">Locally compiled by Tokamak StaticHTML. Real app control source; synthetic surrounding composition. No Apple SwiftUI runtime, AVPlayer, live state or gesture validation.</p>"
print(html.replacingOccurrences(of: "<head>", with: "<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><style>body{display:block!important;height:auto!important;overflow:auto!important}body>p{white-space:normal}</style>")
    .replacingOccurrences(of: "</body>", with: notice + "</body>"))
