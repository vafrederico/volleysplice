#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.dual_serve_fusion_experiment import (
    EXPERIMENT_ID,
    FUSION_VARIANT_DESCRIPTION,
    FUSION_VARIANT_LABEL,
    infer_dual_serve_fusion_dataset,
)


def main() -> int:
    data_root = Path(
        os.environ.get("VOLLEYCUT_DATA_ROOT", "data")
    ).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the frozen validation-selected v4+v5 boundary fusion "
            "for every recording in a manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument(
        "--v4-rally-model",
        type=Path,
        default=workspace / "models" / "full-audiovisual-v2-final",
    )
    parser.add_argument(
        "--v4-serve-model",
        type=Path,
        default=workspace / "models" / "serve-specialist-audiovisual-v4",
    )
    parser.add_argument(
        "--v4-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    parser.add_argument(
        "--v5-rally-model",
        type=Path,
        default=workspace / "models" / "full-audiovisual-audio-normalized-v3",
    )
    parser.add_argument(
        "--v5-serve-model",
        type=Path,
        default=workspace / "models" / "serve-specialist-audio-normalized-v5",
    )
    parser.add_argument(
        "--v5-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-audio-normalized-v3",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=workspace
        / "reports"
        / "dual-serve-v4-v5-fusion-v1-validation.json",
    )
    parser.add_argument("--output-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--model-version", default=EXPERIMENT_ID)
    parser.add_argument("--variant-label", default=FUSION_VARIANT_LABEL)
    parser.add_argument(
        "--variant-description", default=FUSION_VARIANT_DESCRIPTION
    )
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    result = infer_dual_serve_fusion_dataset(
        arguments.manifest,
        arguments.v4_rally_model,
        arguments.v4_serve_model,
        arguments.v4_cache_dir,
        arguments.v5_rally_model,
        arguments.v5_serve_model,
        arguments.v5_cache_dir,
        arguments.decision,
        arguments.output_root,
        model_version=arguments.model_version,
        variant_label=arguments.variant_label,
        variant_description=arguments.variant_description,
        limit=arguments.limit,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
