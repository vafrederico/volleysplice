# VolleySplice Editor Design Language

This document defines the production web editor’s visual and interaction language. New UI should feel like part of one compact, dependable match-review console—not a collection of cards.

## Principles

1. **Keep the match primary.** Video, score, timeline, current rally, and review tasks outrank explanation or decoration.
2. **Show real state.** Controls, counts, highlights, and exports must reflect the current project and playhead; do not use placeholder review data.
3. **Be dense, not cramped.** Prefer compact controls and short labels, while preserving readable type, contrast, and touch targets.
4. **Use structure before decoration.** Separate major areas with background changes and thin rules. Avoid nested cards, excessive rounding, shadows, or weak gray text.
5. **Make the next action obvious.** Current playback state, unresolved review work, and destructive actions need distinct labels and colors.

## Layout

- The desktop editor is a three-part workspace:
  - left: match events and scrollable clip register;
  - center: video and seekable game timeline;
  - right: current-rally controls and range tools.
- Columns use solid light surfaces and `1px` separators. The center and right areas are not wrapped in large cards.
- Keep project selection, Review/Export navigation, review counts, settings, and Export in the top bar to preserve vertical space.
- Fine-tuning is secondary and belongs in the settings modal opened by the cogwheel beside Export.
- Setup and Analyze use the same shell, project selector, grid, ruled surfaces, type scale, and action colors as the editor. Setup may use one concise orientation heading; analysis progress stays flat and operational rather than becoming a separate marketing page.
- On mobile, place Serves & sides above the video and the Clip register after the editor controls. Lists remain bounded and scrollable.

## Color roles

| Role | Value | Use |
| --- | --- | --- |
| Workspace | `#edf0ec` | Page grid/background |
| Primary surface | `#f8faf6` | Columns, controls, modal |
| Secondary surface | `#e2e8e2` | Hover, selected regions, grouped controls |
| Primary ink | `#17201d` | Headings and important values |
| Muted ink | `#5b6863` | Supporting text only |
| Rule | `#c8d0cb` | Dividers and control outlines |
| Action green | `#8ccf45` | Primary actions, active retained footage |
| Alert orange | `#df7048` | Review-needed and excluded-footage cues |
| Team 1 red | `#d9342b` | Team 1 score and timeline |
| Team 2 blue | `#2367c9` | Team 2 score and timeline |
| Remove red | `#b42318` | Explicit rally removal/destructive state |
| Review amber | `#f3b43f` | Clips or serves that require review |
| Cleanup yellow | `#f4dc78` | Lower-priority cleanup review |

Color always has a second cue: text, icon, border, pattern, or state label. Buttons must have a visible background or border; never rely on colored text alone.

## Typography

- UI family: `Manrope`, then `Avenir Next`, then system sans-serif.
- Headings and primary numbers are heavy (`800–850`) with slightly tight tracking.
- Use monospace for timestamps, rally IDs, counts, scores, and compact uppercase kickers.
- Keep labels short and direct: “Keep rally”, “Remove rally”, “Review clips”, “Export”.
- Recommended minimums:
  - headings: `18–25px`;
  - controls/body: `10–12px`;
  - metadata: `9px`;
  - kickers: `10px`, uppercase, spaced.
- Avoid large weak headings and tiny low-contrast explanatory copy.

## Shape and spacing

- Use a `4px` base rhythm; common gaps are `8`, `12`, `14`, `18`, and `22px`.
- Column padding is generally `14–22px`.
- Controls are at least `34px` high; primary actions and mobile targets should approach `40–42px`.
- Default control radius is `6px`; panels may use `10px`. Large pill shapes are not part of the editor language.
- Shadows are reserved for overlays such as the settings modal. In-workspace hierarchy uses rules and surface color.

## Components and states

- **Top bar:** transparent VolleySplice mark, current-project selector, Review/Export stages, review counters, settings, and Export.
- **Timelines:** use green for retained rally core, pale green for padding, red/blue for teams, volleyball icons for serves, and arrows for side switches. Every marker is clickable and seekable.
- **Scrollable registers:** automatically reveal and highlight the event or rally at the playhead. Do not leave a separate persistent selection that makes controls stale.
- **Current rally:** follows the rally at the playhead, or the latest preceding rally when between rallies. Keep/Remove are joined equal-width buttons; green means kept, red means explicitly removed, and pale red means removed only by automatic cleanup.
- **Review CTAs:** amber for required clip/serve review; yellow for optional cleanup review. Clicking a review CTA cycles through outstanding items.
- **Score:** Team 1 and Team 2 names are editable inline. Team colors must remain consistent in the score, video overlay, point timeline, and exports.
- **Modal:** use a dark translucent backdrop, one focused light panel, a clear title, close control, and primary Done action. Escape and backdrop click close it.
- **Destructive actions:** label them explicitly, use red treatment, and confirm project deletion before removing stored edits and analysis.

## Interaction rules

- Clicking the video toggles play/pause.
- Clicking any rally, serve, side switch, or timeline position seeks the video immediately.
- Editing follows the current playhead context; do not require a separate selected rally or serve.
- Final-cut playback and Keep/Remove state must use the same effective suppression rules as export.
- Ready projects expose only Review and Export. New projects begin in Setup, then Analyze, and enter the editor when inference completes.
- Preserve draft changes automatically and keep source reconnection local to the browser.

## Accessibility and responsive behavior

- Maintain visible keyboard focus (`3px` outline), semantic labels, and keyboard-closeable dialogs.
- Disabled controls must remain legible. Do not use opacity alone for meaningful state.
- Preserve red/blue team meaning with labels and score values for color-blind users.
- At narrower widths, rearrange sections instead of shrinking type below the minimums.
- Horizontal or vertical scrolling is appropriate for dense timelines and registers; the page itself should not gain accidental horizontal overflow.

## Avoid

- Large card shells around whole columns or the center workspace.
- Decorative gradients, oversized hero typography, or marketing copy inside the editor.
- Mock data, simulated exports, or controls that do not affect production state.
- Hidden review status, ambiguous icon-only destructive actions, or low-contrast outline buttons.
- Duplicate controls for the same setting in multiple columns.
