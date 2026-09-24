# Source-derived QA for the final report

`scripts/qa-neural-generalization-report-real.mjs` is a separate entrypoint for the final audited v3 HTML. The frozen v3 synthetic test retains its invented missing-variant and mismatch cases; this new test never inserts, changes or fabricates real report rows.

The test chooses existing controls from source membership fields, without consulting metric values or feasibility to choose views. It covers every saved model, scenario, panel, label policy, production filter, padding, draw and embedding precision through a deterministic bounded set of views. Each active row must equal the corresponding source row, including its scope. Every displayed table cell, chart point coordinate and label, and infeasible gap is checked against those saved rows. A separate variant view and the 100% floor use whatever statuses are actually present.

The complete columnar transport is compared to the source in 512-row batches. The downloaded gzip must restore the exact source SHA-256 and byte count. Source, HTML, render receipt and passing audit identities are checked; the report's transport digest must match the numerical audit. Browser requests to external HTTP(S) addresses are blocked and fail QA, as do console/page errors and page-level horizontal overflow. Desktop and 390px screenshots still require visual inspection before publication.

The receipt records startup/filter timings, Chrome renderer JavaScript heap snapshots, and Node RSS. These are not whole-browser process-tree peak memory measurements. Browser profiles, temp/cache, screenshots and downloads remain on the NAS. The sole opt-in `--synthetic-self-test true` mode marks its receipt as synthetic and cannot qualify a real report.

Run only after the global selection freeze, numerical result audit and final render:

```text
node scripts/qa-neural-generalization-report-real.mjs --harness EXISTING_HARNESS --html NAS_REPORT.html --source-json NAS_REPORT.json --render-receipt NAS_RENDER_RECEIPT.json --output-root NAS_NEW_QA_DIRECTORY
```

The source and render receipt are read directly, and the audit referenced by source metadata is rehashed. No model training, inference, calibration, metric pooling or operating-point selection runs in this tool.

The entrypoint passed four focused helper tests and a 2,211-row synthetic self-test with 17 dynamically chosen views, all packed cells, exact table values, chart points and line segments, export download, and desktop/390px screenshots. Startup was about 136ms in that small fixture; this does not predict the final report's startup or memory. The 390px screenshot was visually inspected. Actual audited-data QA remains required.
