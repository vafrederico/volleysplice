# Private research references

Checked-in research files use source indexes from the existing private source
ledger. New recordings and artifact locations use indexes from its private neural
extension. The mappings, original media names, workstation paths and raw execution
receipts stay outside Git. Indexes are stable references, not filesystem paths or
working labeling URLs.

Set `VOLLEYCUT_PRIVATE_LEDGER` to the external JSON ledger when running research
tools. Optional `_WINDOWS` and `_POSIX` suffixes select a platform-specific ledger
location. Keep these settings in ignored local configuration, never a public
environment variable. The ledger's `privateValues` object maps indexes to exact
runtime strings. Missing entries fail closed rather than creating output on the
system drive. Existing media, annotations, catalog IDs and frozen artifacts are
unchanged; only checked-in references are replaced.

Publication reports must use recording/source indexes and artifact indexes.
Resolve indexes privately before executing historical command examples. The
editor uses the configured catalog instead of embedding a recording filename in
the client bundle. Local catalog entries can still display their original names.

Run the publication check before committing generated results:

```sh
python scripts/audit-branch-privacy.py --base origin/main \
  --ledger "$VOLLEYCUT_PRIVATE_LEDGER" --output "$PRIVATE_AUDIT_OUTPUT"
```

Add `--history` to inspect earlier branch blobs and commit messages. A clean
working tree check does **not** erase earlier commits. Existing commit author
metadata also remains. Do not merge or publish this research history on the
strength of a working tree pass alone. A sanitized squash or explicit history
rewrite is needed before publication.

Frozen source-hash audits continue to refer to their archived original source
snapshots. Sanitized source files have different hashes; do not change historical
receipts or claim the sanitized files are byte-identical to those snapshots.
