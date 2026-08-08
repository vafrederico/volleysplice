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
