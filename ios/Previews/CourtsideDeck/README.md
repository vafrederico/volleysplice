# Local compatible SwiftUI control probe

This harness compiles the app's exact `CourtsideReviewControls.swift` with
[Tokamak StaticHTML](https://github.com/TokamakUI/Tokamak). It is an offline/local
alternative for a supported SwiftUI subset. Swift compiles locally; repository
source is never submitted to a hosted playground. Public dependencies may be
downloaded normally.

Validated renderer revision: `e0d8e9db462938610337327ade58161aa3b698ac`, Swift
6.4.1 on Linux, Swift 5 language mode. The renderer repository is archived; pin
the validated revision rather than assuming compatibility with other revisions.

Set `VOLLEYCUT_TOKAMAK_ROOT` to a local checkout of that revision, load the lab's
Swift environment, then run from a Linux/WSL shell:

```sh
python3 ios/Previews/CourtsideDeck/render.py \
  --work-dir "$VOLLEYCUT_PREVIEW_BUILD_ROOT" \
  --output "$VOLLEYCUT_PREVIEW_OUTPUT/portrait.html" --width 390
```

Use an external E-backed work/cache directory in the Windows lab. Repeat with
widths 768 and 844 for the tablet and landscape control probes. `render.py` copies
the real control source, package manifest and synthetic `Preview.swift` into that
directory. It never modifies the source checkout or native app. The host adapter
supplies StaticHTML's missing `disabled` convenience modifier by setting its
environment and places the generated document in a normal block-flow body.
It also ignores `contentShape`, since StaticHTML has no native hit testing;
interaction bounds require device checks.

The [portrait](rendered/portrait.html), [tablet](rendered/tablet.html) and
[landscape](rendered/landscape.html) outputs contain the locally compiled shared
controls. They were inspected with the HTML preview tool at their declared widths;
all three produced rendered pages with no console errors. T3's collaborative
browser snapshot operation timed out twice, so those snapshots are not evidence.

The synthetic shell, teams, score, timeline and media slot are placeholders.
StaticHTML provides no live state/gesture/media validation. In these probes its
intrinsic-width/grid behavior differs from native SwiftUI and does not preserve
native frame expansion. They are **control compatibility probes, not native UI
screenshots or final layout approval**. The native app must still be validated
on an Apple runtime; see [the implementation record](../../COURTSIDE-DECK.md).

Bounded alternatives research used primary documentation:

- [Coddy](https://coddy.tech/playground/swiftui) documents server-side compilation
  against its own implementation; it does not establish a local install or public
  renderer API. No code was submitted there.
- [SwiftWebUI](https://github.com/SwiftWebUI/SwiftWebUI) describes its demo browser
  implementation; it was not installed or used here.
- [Tokamak's static renderer](https://github.com/TokamakUI/Tokamak/blob/main/Sources/TokamakStaticHTML/StaticHTMLRenderer.swift)
  supplies the local source-reuse probe. Native Apple framework fidelity is
  outside its scope.

The checked-in HTML probes and `verification.json` describe the earlier portable
preview pass. Current native verification, including UIKit footer targets, is in
[the device receipt](device-verification.json). Raw device captures remain local.
