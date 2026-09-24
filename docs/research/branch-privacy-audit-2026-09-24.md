# Neural research branch privacy audit

The files have been sanitized. The initial history scan flagged 226 earlier
blobs; one was a dependency-version false positive corrected before the final
publication gate. The user approved replacing the research
branch history with a sanitized squash and updating its Forgejo branch. The
replacement snapshot must pass both the current-content and historical-object
checks before it is pushed. A private NAS backup preserves the original history.

Scope: files changed relative to the main-branch merge base, plus modified and
untracked non-ignored files. This includes the pending native benchmark tooling.
It is not a new privacy certification of inherited main-branch content.

## Changes

- Reused the existing source alias ledger and added stable indexes for newer
  inventory recordings, source groups and artifact locations in a private ledger
  extension. The benchmark recording is `recording-044`.
- Removed identifiable recording names, workstation/media paths, private report
  URLs and network addresses from checked-in research text and script constants.
  Renamed the recording-specific review document to its recording index.
- Runtime resolvers recover exact inputs from the external ledger. Existing
  source files, labels, catalog IDs, model outputs and immutable receipts were
  not renamed or rewritten.
- The editor obtains its default recording from its configured catalog. No
  private identity is embedded as its default in the client component.
- Benchmark summary generation now requires recording and artifact indexes;
  the generated public summary does not include its local source directory.
- Added a repeatable branch privacy check covering known ledger identities,
  common camera filenames, private paths, network addresses and common credential
  formats. Reports omit matched text. This is not an exhaustive secret detector.

## Validation

The current-file scan passes. The initial history scan flagged the 226 older
blobs; no flagged commit messages were found. Commit author metadata remains
unchanged. Existing ignored environment files, media and private receipts are
intentionally outside the publication scan. The sanitized squash preserves the
latest committed lab work; the unfinished native benchmark remains uncommitted.

All 437 existing Python files in scope parsed successfully. The substituted
Python literals were compared with their original runtime values; the intended
differences are sanitized report prose, synthetic test paths, repository-relative
lookup and import ordering. A frozen-source test checks its archived original
snapshot rather than falsely expecting the sanitized file to retain its old hash.

The focused research suite passed 57 tests; five additional privacy-detector
tests passed. The focused UI suite passed 29 tests, including frozen catalog
fixtures. TypeScript checking passed. The three research plot PNGs were visually
checked and contain aggregate metrics, without recording names or private paths.
The replacement commit `291e206f` passed a committed-snapshot and history scan:
620 committed files checked, zero findings in files or branch-specific historical
objects. Forgejo was updated with an explicit lease and its remote tip verified.
Pending benchmark files were byte-checked and preserved through the rewrite.
The optimized Next build passed in an isolated NAS directory. It emitted export
warnings in untouched model-feedback modules; the privacy changes did not edit
those modules. No development or production server was started for this check.

See [private research references](./private-research-ledger.md) for configuration
and the publication check. Replacing the branch removes the earlier research
commits from that branch's ancestry. It does not erase private backups, reflogs,
other references, or objects retained by the hosting server pending collection.
No public remote is updated by this operation.
