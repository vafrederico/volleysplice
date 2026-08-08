# VolleyCut

AI-assisted volleyball video trimming, beginning with conservative rally suggestions and a fast human review workflow.

## Run locally

```bash
npm install
npm run dev
```

Open the URL printed by Next.js. The initial screen is a front-end prototype: adjust the pre/post-roll settings, select rallies, and inspect the resulting edit decision list.

## Project layout

- `app/` — Next.js application.
- `components/` — interactive review editor.
- `lib/` — edit decision list types and interval calculations.
- `data/videos/` — local, untracked source recordings and a future YouTube intake area.
- `docs/research/` — feasibility and implementation research retained for reference.

Large videos, proxies, and exports are deliberately excluded from Git. The eventual production design will upload originals directly to private object storage.

## Initial scope

- Stationary end-line footage.
- Conservative suggested rally boundaries.
- Configurable pre-roll and post-roll without re-analysis.
- Human review before export.
- No player statistics in the initial critical path.
