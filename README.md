# VolleyCut

AI-assisted volleyball video trimming, beginning with conservative rally suggestions and a fast human review workflow.

## Run locally

```bash
npm install
npm run dev
```

Open the URL printed by Next.js. The initial screen is a front-end prototype: adjust the pre/post-roll settings, select rallies, and inspect the resulting edit decision list.

Open `/label` for the local gold-label workstation. It loads a task JSON and matching proxy through browser file pickers, supports precise rally/ignored/hard-negative intervals, and exports resumable or completed labels without uploading video. See [`docs/labeling-guide.md`](docs/labeling-guide.md).

## Rally-analysis baseline

The worktree now includes a CPU-only v0 that can normalize and validate recordings, train a temporal rally classifier, infer rally intervals into `analysis.json`, and evaluate an untouched test split. It can be exercised before real video arrives with:

```bash
npm run analysis:setup
npm run test:analysis
.venv/bin/python -m analysis smoke
```

See [`analysis/README.md`](analysis/README.md) for the annotation contract and end-to-end commands. Research, licensing, camera-fit findings, and adoption decisions are indexed under [`docs/research/`](docs/research/).

## Project layout

- `app/` — Next.js application.
- `components/` — interactive review editor.
- `lib/` — edit decision list types and interval calculations.
- `data/videos/` — local, untracked source recordings and a future YouTube intake area.
- `docs/research/` — feasibility and implementation research retained for reference.
- `docs/labeling-guide.md` — exact boundary policy, keyboard workflow, and label validation/import.
- `analysis/` — local video normalization, feature extraction, training, inference, and evaluation.

Large videos, proxies, and exports are deliberately excluded from Git. The eventual production design will upload originals directly to private object storage.

## Initial scope

- Stationary end-line footage.
- Conservative suggested rally boundaries.
- Configurable pre-roll and post-roll without re-analysis.
- Human review before export.
- No player statistics in the initial critical path.
