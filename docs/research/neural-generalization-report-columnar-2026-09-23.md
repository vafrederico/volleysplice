# Scalable report transport, version 2

`scripts/render-neural-generalization-report-v2.py` preserves the frozen report's
metric values, row order, missing fields, nulls, nested values and complete source
download. It changes transport and browser access only. No rows are aggregated,
selected, rounded or removed. The original renderer remains unchanged and is
hash-bound as the UI source.

The browser stores columns, not one object per saved row. Low-cardinality values
use dictionaries and Uint32 codes; continuous numeric columns use Float64 buffers
with explicit missing/null states. Filters read reusable row views. Only rows in
the active chart/table selection become ordinary objects. Dropdown construction
also streams column values rather than retaining a full intermediate row array.
The encoded column package is gzip-compressed and embedded in the standalone HTML.
Opening it requires the browser's `DecompressionStream('gzip')` API; no CDN, extra
file, server fetch or third-party JavaScript is needed.

A separate embedded gzip stream contains the **exact original input JSON bytes**.
The download is plainly labeled `JSON.gz`; decompression restores those bytes,
including their whitespace. Downloading does not parse the canonical JSON or
materialize its rows. The footer and rendering receipt retain the original file
SHA-256 and `reportContentSha256` over the audited numerical payload. For real
results, rendering checks the referenced audit's file hash, passed status and
payload digest, in addition to the report's audit flag.

The active panel warning distinguishes all-video descriptions from unseen-source
tests. In particular, production exposure filtering does not exclude neural
training or calibration, and outside-original sources may enter expanded variants.
The historical-context warning remains visible. The precision selector is labeled
“DINO embedding precision”: FP16/INT8 affect embeddings, while temporal heads and
the FP32 calibration selections remain frozen without recalibration.
Immediately above the curves, the active label policy spells out the duration
numerators and denominators: exact recall measures retained rally-core seconds,
draft recall measures retained reviewed-live seconds, and export recall measures
retained fixed human-export seconds. These are distinguished from semantic
rally-count recall, event F1 and completely lost rallies.
Export-proxy scenarios also disclose whether approximate cores enter training
alone or training and calibration. The selected evaluation gold tier remains
unchanged. Fixed epoch counts are distinguished from fixed optimizer-update
budgets: four-head proxy supervision changes sampling, exact-tier scaling,
update counts and student tier order, so these are not isolated label-only effects.
The task-membership selector also shows whether that logical task performed
training or inherited all four fixed checkpoints, names its physical owner and
states that calibration/selection remain task-local. Shared owners are explicitly
described as correlated views rather than independent training replications.

## Qualification so far

Three focused transport tests passed: exact reconstruction of numeric,
categorical, missing/null, boolean and nested fields; byte-identical gzip download
and canonical-payload digest; rejection of unrepresentable integers and unaudited
real input. The separate final-report auditor also now retains only the small
already-verified task-display fields from completed evaluation documents, avoiding
accumulation of hundreds of full raw operating-point payloads.

Initial Chrome 153 QA used 2,645 invented fixture rows, including numeric columns,
missing/null fields, nested values and HTML-like text. The initial view
materialized only 44 rows. The fixture's original JSON was 2,764,650 bytes; its
column package was 404,392 bytes before gzip and 16,650 bytes after gzip. Including
the separate exact original JSON gzip and all interface code, HTML was 172,418
bytes. These are **small synthetic-fixture measurements**, not real-study
compression or mobile-memory results.

The new QA script checks every fixture cell against canonical input, exact gzip
download bytes, matching-only production overlay, filters, infeasible gaps,
context/label/precision/panel warnings, responsive layout, and absence of page
errors or external requests. The completed modest-fixture QA receipts and inspected
desktop/mobile screenshots live on NAS under
`2026-09-23-recall-sweep-generalization/html-columnar-qa-v3`.
HTML SHA-256 is
`d1ccb00afe184d1227fe2ad9314db3076b326927fa8cc9856863d3479794f781`.
The initial view opened in 162 ms with 2,149,280 bytes of JavaScript heap reported
by Chrome's performance API. This excludes total browser-process memory and is
not an extrapolation to the final report.

The ownership-display addition passed a second full modest-fixture QA under
`html-columnar-qa-v4`: both physical/reused states, owner identity and the four
checkpoint epochs are asserted. The unchanged 2,645-row transport still
materializes 44 initial rows; the richer fixture produces 173,589 bytes of HTML,
opens in 191 ms and reports 2,141,376 bytes of initial JavaScript heap. Both
desktop and 390px screenshots were inspected with no overflow; every transport
cell and downloaded source byte still matches. HTML SHA-256:
`4f64f431f2202031f85d7ee1effdbab54ae833729b15720ee4b956150d17302b`.

The full real report still needs measured payload size, browser memory/startup and
filter latency after heavy computation ends. Gzip reduces transfer size but does
not by itself establish an acceptable memory footprint. Desktop responsive QA is
not physical-phone qualification. No synthetic or unfinished report is published.

## Final rendering and publication

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/render-neural-generalization-report-v2.py \
  --input private-reference-0144 \
  --output private-reference-0145 \
  --receipt private-reference-0146
```

All outputs use exclusive creation. After the final real artifact passes local QA,
publish through the shared
Publish Review Artifacts skill (ledger `private-reference-0147`):

```sh
private-reference-0148 \
  --server private-reference-0199 --json upload \
  --title 'VolleyCut recall and generalization results' \
  private-reference-0145
```

Retain returned JSON on NAS, including artifact ID and `viewUrl`. This LAN endpoint
was resolved to `private-reference-0200` and its read-only CLI list succeeded. The service
advertises a 1,024 MB upload maximum, but that limit has not been stress-tested.
