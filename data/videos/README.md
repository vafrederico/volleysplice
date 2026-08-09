# Source video intake

This is the repository-local fallback when `VOLLEYCUT_DATA_ROOT` is unset. The durable dataset should live outside Git under `$VOLLEYCUT_DATA_ROOT/raw/`; this machine uses `/mnt/freenas/volleycut/raw/`.

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
