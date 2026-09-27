"""Create independent synthetic goldens for the distilled Large browser pipeline.

The existing Python training extractor owns expected resize, rounding, quality,
normalization and spatial-pool values. No recording or labels are needed. All
generated arrays remain in a caller-supplied external artifact directory. Frozen
encoder checks are optional so fixture preparation need not compete with timed
device benchmarks. They use ledger-resolved graphs and do not train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import mobile_visual_features as mobile
from analysis.private_ledger import private_value


KIND = "distilled-large-browser-image-fixtures-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_array(root: Path, name: str, values: np.ndarray) -> dict:
    values = np.ascontiguousarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Fixture contains nonfinite values")
    path = root / name
    with path.open("xb") as handle:
        handle.write(values.tobytes())
    return {"file": name, "dtype": values.dtype.name, "shape": list(values.shape),
            "sizeBytes": path.stat().st_size, "sha256": sha256(path)}


def pattern(width: int, height: int) -> np.ndarray:
    """Deterministic high-frequency RGB; includes clipping and channel edges."""
    y, x = np.indices((height, width), dtype=np.int32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (x * 37 + y * 17 + (x // 7) * 29) % 256
    rgb[:, :, 1] = (x * 11 + y * 43 + (y // 5) * 53) % 256
    rgb[:, :, 2] = ((x // 3 + y // 2) % 2) * 255
    rgb[:max(1, height // 8), :max(1, width // 8)] = 0
    rgb[-max(1, height // 8):, -max(1, width // 8):] = 255
    return rgb


def specifications() -> list[dict]:
    full = dict(x=0., y=0., width=1., height=1.)
    return [
        dict(id="landscape-1080", width=1920, height=1080, roi=full, offset=0.),
        dict(id="portrait-odd", width=361, height=641, roi=full, offset=.004),
        dict(id="fractional-roi", width=617, height=347,
             roi=dict(x=.073, y=.119, width=.817, height=.763), offset=-.008),
        dict(id="ties-even-roi", width=640, height=360,
             roi=dict(x=2.5 / 640, y=4.5 / 360, width=512. / 640, height=280. / 360), offset=0.),
        dict(id="small-upsample", width=7, height=3, roi=full, offset=.125),
        dict(id="black", width=224, height=224, roi=full, offset=0., solid=0),
        dict(id="white", width=224, height=224, roi=full, offset=0., solid=255),
        dict(id="square-high-frequency", width=257, height=257, roi=full, offset=-.125),
    ]


def prepare_case(root: Path, spec: dict) -> dict:
    rgb = pattern(spec["width"], spec["height"])
    if "solid" in spec:
        rgb.fill(spec["solid"])
    # The training helper consumes display-oriented BGR. The browser fixture
    # instead receives explicit RGBA to expose accidental channel swaps.
    pixels, box, quality = mobile.preprocess_frame(
        np.ascontiguousarray(rgb[:, :, ::-1]), spec["roi"], input_size=224)
    quality[5] = spec["offset"]
    weights = mobile.regional_pool_weights(box[None], 7, 7)
    np.testing.assert_allclose(weights.sum((2, 3)), 1., atol=1e-7, rtol=0)
    if np.any(weights < 0):
        raise ValueError("Spatial pools must have nonnegative weights")
    rgba = np.concatenate((rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=2)
    prefix = spec["id"]
    return {
        "id": prefix, "width": spec["width"], "height": spec["height"],
        "roi": spec["roi"], "rotationDegrees": 0,
        "selectedPtsOffsetSeconds": spec["offset"],
        "inputCoordinates": "display-oriented; rotation is already applied",
        "rgba": write_array(root, prefix + "-rgba.u8", rgba),
        "expected": {
            "pixels": write_array(root, prefix + "-pixels.f32", pixels[None].astype("<f4")),
            "quality": write_array(root, prefix + "-quality.f32", quality.astype("<f4")),
            "poolWeights": write_array(root, prefix + "-pool.f32", weights.astype("<f4")),
            "contentBox": write_array(root, prefix + "-box.f64", box.astype("<f8")),
        },
    }


def read_array(root: Path, entry: dict) -> np.ndarray:
    path = root / entry["file"]
    if sha256(path) != entry["sha256"] or path.stat().st_size != entry["sizeBytes"]:
        raise ValueError("Fixture identity changed")
    return np.fromfile(path, dtype=np.dtype(entry["dtype"]).newbyteorder("<")).reshape(entry["shape"])


def add_encoder_expected(root: Path, cases: list[dict], graph_root: Path, variant: str) -> dict:
    import onnxruntime as ort

    folder = graph_root / ("graphs-" + variant)
    contract = json.loads((folder / "input-contract.json").read_text(encoding="utf-8"))
    name = "mobile-large-encoder-fp32.onnx"
    graph = folder / name
    if sha256(graph) != contract["hashes"][name]:
        raise ValueError("Frozen encoder identity changed")
    config = ort.SessionOptions()
    config.intra_op_num_threads = 1
    config.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=config, providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["image", "pool_weights"]:
        raise ValueError("Unexpected frozen encoder inputs")
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("modelIdentity") != "dino-distilled-mobilenet-v3-large-tcn" \
            or metadata.get("selectionMode") != variant:
        raise ValueError("Frozen student metadata differs")
    for case in cases:
        raw = session.run(None, {
            "image": read_array(root, case["expected"]["pixels"]),
            "pool_weights": read_array(root, case["expected"]["poolWeights"]),
        })[0]
        if raw.shape != (1, 4, 960) or not np.isfinite(raw).all():
            raise ValueError("Invalid frozen encoder output")
        rounded = raw.astype(np.float16).astype(np.float32)
        prefix = case["id"] + "-" + variant
        case.setdefault("encoderExpected", {})[variant] = {
            "rawTokens": write_array(root, prefix + "-raw-tokens.f32", raw.astype("<f4")),
            "roundedTokens": write_array(root, prefix + "-rounded-tokens.f32", rounded.astype("<f4")),
        }
    return {"variant": variant, "graphSha256": sha256(graph), "graphBytes": graph.stat().st_size,
            "runtime": ort.__version__, "providers": session.get_providers(), "threads": 1,
            "tolerance": {"atol": 2e-5, "rtol": 2e-4},
            "tokenCache": "float32 -> IEEE float16 ties-to-even -> float32"}


def temporal_contract_fixtures(root: Path, graph_root: Path) -> dict:
    from analysis.neural_development import decode
    from analysis.features import ABSOLUTE_FEATURE_NAMES, feature_names, percentile_rank_values
    from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET

    configs = {mode: json.loads((graph_root / ("graphs-" + mode) / "mobile-large-pipeline.json")
                              .read_text(encoding="utf-8")) for mode in ("recall", "f1")}
    sequences = {
        "exact-high-threshold": [.9] * 9,
        "below-high-threshold": [float(np.nextafter(np.float32(.9), np.float32(0)))] * 9,
        "exact-low-threshold": [.2] * 9,
        "exit-threshold": [1.] * 7 + [.1] * 5 + [0.] * 4,
        "two-tick-bridge": [0.] * 5 + [1.] * 7 + [0.] * 2 + [1.] * 7 + [0.] * 5,
        "longer-gap": [0.] * 5 + [1.] * 7 + [0.] * 6 + [1.] * 7 + [0.] * 5,
        "boundary-equal-peaks": [0.] * 6 + [.95] * 9 + [0.] * 6,
        "single-tick": [.9],
        "three-ticks": [.9, .9, .9],
        "short-peak": [0.] * 4 + [.9] + [0.] * 4,
    }
    decoder = []
    for mode, config in configs.items():
        for name, live in sequences.items():
            n = len(live)
            times = np.arange(n, dtype=np.float64) / 4
            probabilities = np.zeros((n, 4), np.float32)
            probabilities[:, 0] = live
            if name == "boundary-equal-peaks":
                probabilities[3:8, 1] = .65
                probabilities[13:18, 2] = .65
            duration = n / 4
            expected = decode(SimpleNamespace(times=times, valid=np.ones(n, bool), duration=duration),
                              probabilities, config["decoder"])
            decoder.append(dict(id=mode + "-" + name, config=config["decoder"], times=times.tolist(),
                                probabilities=probabilities.reshape(-1).tolist(), duration=duration,
                                expected=[dict(start=r.start, end=r.end) for r in expected]))

    rng = np.random.default_rng(8421)
    n, count = 17, 9
    times = np.arange(n, dtype=np.float64) / 4
    embedding_times = np.arange(count, dtype=np.float64) / 2
    names = list(feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)))
    av = rng.integers(0, 9, size=(n, 104)).astype(np.float32) / np.float32(8)
    ranked = percentile_rank_values(av)
    for column, name in enumerate(names):
        if name in ABSOLUTE_FEATURE_NAMES:
            ranked[:, column] = av[:, column]
    raw_tokens = rng.uniform(-2, 2, (count, 4, 960)).astype(np.float32)
    tokens = raw_tokens.astype(np.float16).astype(np.float32)
    quality = rng.uniform(0, 1, (count, 6)).astype(np.float32)
    quality[:, 5] = np.linspace(-.002, .002, count, dtype=np.float32)
    cache = mobile.MobileVisualCache(Path("unused"), embedding_times, tokens, quality,
                                    embedding_times + quality[:, 5], {})
    aligned = mobile.align_mobile_features(cache, times)
    values = np.concatenate((ranked, aligned["tokens"].reshape(n, -1), aligned["quality"],
                             aligned["feature_age_seconds"][:, None], aligned["available"][:, None]), axis=1).astype(np.float32)
    inputs = {
        "times": write_array(root, "fusion-times.f64", times.astype("<f8")),
        "embeddingTimes": write_array(root, "fusion-embedding-times.f64", embedding_times.astype("<f8")),
        "rawAv": write_array(root, "fusion-raw-av.f32", av.astype("<f4")),
        "rankedAv": write_array(root, "fusion-ranked-av.f32", ranked.astype("<f4")),
        "rawTokens": write_array(root, "fusion-raw-tokens.f32", raw_tokens.astype("<f4")),
        "tokens": write_array(root, "fusion-rounded-tokens.f32", tokens.astype("<f4")),
        "quality": write_array(root, "fusion-quality.f32", quality.astype("<f4")),
    }
    fusion = []
    for mode, config in configs.items():
        mean, scale = (np.asarray(config[key], np.float32) for key in ("mean", "scale"))
        expected = values.copy()
        expected[:, :104] = np.clip((expected[:, :104] - mean[:104]) / scale[:104], -10, 10)
        expected[:, -8:] = np.clip((expected[:, -8:] - mean[104:]) / scale[104:], -10, 10)
        fusion.append(dict(id=mode, config=dict(mean=mean.tolist(), scale=scale.tolist()),
                           expected=write_array(root, "fusion-" + mode + "-expected.f32", expected.astype("<f4"))))
    return dict(decoder=decoder, fusion=dict(names=names, rows=n, sampleCount=count,
                                            inputs=inputs, variants=fusion),
                note="NumPy float32 scalar comparisons cast hysteresis and boundary thresholds to float32; short-peak comparison casts the score to Python float.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graphs-index", default="private-reference-0223")
    parser.add_argument("--with-encoder", action="store_true")
    parser.add_argument("--with-temporal", action="store_true")
    parser.add_argument("--variant", choices=("recall", "f1", "both"), default="both")
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError("Generated fixture tensors must remain outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    cases = [prepare_case(output, spec) for spec in specifications()]
    encoders = []
    temporal = None
    if args.with_temporal:
        temporal = temporal_contract_fixtures(output, Path(private_value(args.graphs_index)))
    if args.with_encoder:
        graphs = Path(private_value(args.graphs_index))
        for variant in ("recall", "f1") if args.variant == "both" else (args.variant,):
            encoders.append(add_encoder_expected(output, cases, graphs, variant))
    result = {
        "schemaVersion": 1, "kind": KIND, "labelsUsed": False, "trainingPerformed": False,
        "opencvVersion": cv2.__version__, "numpyVersion": np.__version__,
        "trainingExtractorSha256": sha256(Path(mobile.__file__)),
        "preparationScriptSha256": sha256(Path(__file__)),
        "encoderGraphLedgerIndex": args.graphs_index if args.with_encoder or args.with_temporal else None,
        "normalization": {"mean": mobile.RGB_MEAN.tolist(), "std": mobile.RGB_STD.tolist()},
        "qualityNames": list(mobile.QUALITY_NAMES), "regionNames": list(mobile.REGION_NAMES),
        "qualityTolerance": {"atol": 2e-6, "rtol": 2e-6},
        "pixelsTolerance": {"atol": 0., "rtol": 0.},
        "poolWeightsTolerance": {"atol": 1e-7, "rtol": 1e-7},
        "limitations": ["Synthetic RGB inputs do not qualify video decoding, color metadata or rotation.",
                        "Encoder results compare the frozen exported graph across runtimes, not model accuracy."],
        "cases": cases, "encoders": encoders, "temporalContract": temporal,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"kind": KIND, "cases": len(cases), "encoderVariants": len(encoders)}))


if __name__ == "__main__":
    main()
