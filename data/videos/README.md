# Source video intake

Put local source recordings here while building the feasibility dataset. The directory is intended for the owner's indoor, grass, and beach volleyball videos, including files downloaded from their YouTube channel later.

Video files are intentionally ignored by Git because they are large and may contain private or youth footage. Only this guide and `.gitkeep` are tracked.

Suggested organization once videos arrive:

```text
data/videos/
  indoor/
  grass/
  beach/
  metadata.csv
```

Keep originals unchanged. Future ingestion should create proxies in `data/proxies/` and record provenance, consent-for-training status, capture conditions, and annotations separately.

The implemented v0 writes a constant-frame-rate analysis master and SHA-256 provenance sidecar with `python -m analysis normalize`. Rally annotations and leakage-safe recording groups live in a manifest; start with `analysis/examples/dataset.example.json` or run `python -m analysis init-manifest`. Do not annotate the variable-frame-rate original and then silently train against a transcoded proxy.
