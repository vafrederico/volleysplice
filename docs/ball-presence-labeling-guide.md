# Ball-presence labeling workstation

The `/label/ball` route reviews the 15 fps, development-only ball-presence
sample. It streams the exact SHA-256-pinned PNGs from the pilot workspace and
writes role-aware human labels back to the NAS. It opens in Sol-assisted mode
when a prepared Sol artifact exists; assisted source exposure is recorded per
frame and can be disabled before opening an unreviewed frame.

## Start the workstation

```bash
npm run dev -- --hostname 0.0.0.0
hostname -I
```

Open `http://<lan-ip>:3000/label/ball`. The default pilot root is:

```text
/mnt/freenas/volleycut/ball-presence-v1/round-01
```

Override it with `VOLLEYCUT_BALL_PILOT_ROOT`. A Sol directory can be overridden
with `VOLLEYCUT_BALL_SOL_LABELS_ROOT`; otherwise it is `<pilot-root>/sol-labels`.

The server keeps each source structurally separate:

```text
round-01/
├── tasks/          pristine, unreviewed tasks
├── images/         immutable sampled PNGs
├── reviews/        human drafts and exposure-audit sidecars
├── sol-labels/     detector-blind Sol reviews
└── suggestions/    role-free detector proposals
```

The ordinary task and image APIs read only `tasks/`, `images/`, and `reviews/`.
They never open `sol-labels/` or `suggestions/`. Comparison artifacts are read
only through a frame-scoped POST after a saved human decision, or after an
explicit assisted-mode opt-in.

## Blind review

1. Choose a task. The annotator name is optional.
2. Use the six numbered window tabs. The sampling stratum and rally reference
   are intentionally hidden in the editor.
3. Step through all 45 frames in each window. Playback is only a context aid;
   pause on the exact frame before labeling it.
4. Draw exactly one primary-court box when the primary ball is localizable.
   Add other-court and unknown-role volleyballs separately when visible.
5. Otherwise choose **Out of frame**, **Fully occluded**, or **Indeterminate**.
6. Save happens automatically while a review is partial. **Save now** performs
   the same atomic NAS write. Once every frame is reviewed, use **Complete
   review** to persist the final frame and lock the task.

The browser receives a review-only document without a `suggestions` property.
The server rejects extra proposal fields, immutable changes, changed frame
sets, invalid boxes, role/state inconsistencies, and any attempt to erase a
recorded assisted exposure. The native JSON written under `reviews/` always
contains an empty `suggestions` object for compatibility with the project
schema.

## Comparison and assisted review

Human, Sol, and detector boxes are visually distinct and independently
toggleable. Human boxes are solid, Sol boxes are purple dashed, and detector
boxes are cyan dotted.

- **Reveal after human decision** first saves the current human frame, then
  requests only that frame's comparison layers. This is post-decision review;
  `proposalExposure` stays `not_shown`, so the human decision remains eligible
  for blind metrics only while that saved decision remains unchanged. The
  server binds the reveal to a canonical SHA-256 of that exact human frame. If
  the label, boxes, state, or notes are revised after viewing a proposal, the
  server irreversibly changes the frame to
  `shown_before_label_finalized` and excludes it from blind metrics.
- **Assisted pre-label mode** fetches the Sol comparison as each unreviewed
  frame opens. Before returning an available proposal, the
  server irreversibly persists
  `proposalExposure: "shown_before_label_finalized"` on that human frame.
  Each Sol object has **Correct · accept** and **Draw better** actions. Accept
  preserves the exact Sol geometry as a human-verified label; Draw better
  removes the corresponding proposal and activates the matching draw role.
  **Accept entire Sol label** remains available for a completely correct frame.
- Detector proposals are loaded separately on demand. **Use best detector
  box** is an explicit copy action. Detector boxes are role-free, so the active
  draw role determines how the selected box is copied; the default is
  primary-court.

Assisted frames are not discarded. They remain in the headline
human-verified workflow metrics, and each frame records `proposalSources` so
the evaluator can also report detector-independent and Sol-independent
subsets. Only detector-independent frames may select or freeze a detector
threshold. The inclusive Sol comparison is useful for measuring the verified
workflow but is explicitly circular when Sol supplied the pre-label; the
separate Sol-independent subset is the quality estimate.

Every comparison request also updates
`reviews/<recording-id>.proposal-exposure.json`. That sidecar records which
sources were shown before a decision, which were revealed afterward, and the
SHA-256 of the human annotation at the first post-decision reveal. Review and
exposure mutations are serialized per task so an overlapping autosave cannot
erase this state. The self-contained frame fields are authoritative for metric
populations; the sidecar retains source, timing, and revision history.

Sol overlays are accepted only when the artifact has complete annotations,
empty detector suggestions, exact immutable task identity, and validated
`solReviewProvenance` binding it to the pristine task SHA-256. Its SHA-pinned
preparation receipt must remain beside the Sol task and bind the original
source, reviewer identity, active pilot index, output path, and preparation
code hash. The completed Sol annotator must match the receipt reviewer and its
review time cannot precede preparation. Detector overlays must likewise retain
the pristine source-task SHA-256 and contain role-free normalized proposals.

## Controls

| Key | Action |
| --- | --- |
| `←` / `→` | Previous or next frame in the current window |
| `Space` | Play or pause the current window |
| `Enter` | Next frame |
| `P` | Draw primary-court ball |
| `A` | Draw other-court ball |
| `U` | Draw unknown-role ball |
| `O` | Mark primary ball out of frame and advance |
| `C` | Mark primary ball fully occluded and advance |
| `I` | Mark primary-ball state indeterminate and advance |
| `V` | Copy the previous human frame label and track IDs |
| `Delete` / `Backspace` | Delete the selected human box |
| `Esc` | Cancel the active draw tool |

Use 2× or 4× zoom for tiny balls. A copied prior-frame box remains a human
label and can be deleted and redrawn; no proposal is ever copied implicitly.
