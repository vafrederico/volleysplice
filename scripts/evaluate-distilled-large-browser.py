"""Independently replay and score frozen distilled Large browser observations.

All recording locations and labels resolve at runtime; generated reports contain
only indexed provenance, hashes, aggregate diagnostics and source-time ranges.
This is a one-recording runtime validation, not training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import ABSOLUTE_FEATURE_NAMES, feature_names, percentile_rank_values
from analysis import mobile_visual_features as mobile
from analysis.private_ledger import private_value


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def native_report():
    spec = importlib.util.spec_from_file_location(
        "distilled_native_report", Path(__file__).with_name("report-distilled-mobile-large-native.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_graph_identity(graphs, canonical, choice, helper):
    identity = helper.graph_identity(graphs, choice)
    if graphs.resolve() == canonical.resolve():
        return identity
    import onnx
    contract = read(graphs / "input-contract.json")
    original = read(canonical / "input-contract.json")
    provenance = contract.get("graphRewrite", {})
    encoder = "mobile-large-encoder-fp32.onnx"
    if provenance.get("kind") != "einsum-to-matmul-regional-pool-v1" \
            or provenance.get("trainingPerformed") is not False or provenance.get("precision") != "fp32" \
            or provenance.get("originalEncoderSha256") != digest(canonical / encoder) \
            or provenance.get("originalInputContractSha256") != digest(canonical / "input-contract.json") \
            or provenance.get("derivedEncoderSha256") != digest(graphs / encoder):
        raise ValueError("Derived encoder is not bound to the canonical selected encoder")
    for name, expected in original["hashes"].items():
        if name != encoder and digest(graphs / name) != expected:
            raise ValueError("Portable pooling changed a non-encoder component")
    # Reconstruct the allowed transformation from immutable original bytes. This
    # catches altered learned tensors or unrelated graph changes even if a new
    # input-contract hash was supplied alongside them.
    spec = importlib.util.spec_from_file_location("portable_pooling_preparer",
        Path(__file__).with_name("prepare-distilled-browser-portable-pooling.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_model = onnx.load(canonical / encoder)
    proof = module.rewrite(expected_model)
    if hashlib.sha256(expected_model.SerializeToString()).hexdigest() != digest(graphs / encoder):
        raise ValueError("Derived encoder differs from the allowed pooling-only rewrite")
    if any(provenance.get(key) != value for key, value in proof.items()):
        raise ValueError("Preserved-initializer proof differs")
    identity["graphRewrite"] = {key: provenance[key] for key in
        ("kind", "originalEncoderSha256", "derivedEncoderSha256", "originalInputContractSha256",
         "preservedInitializerCount", "addedShapeInitializers", "preservedInitializerManifestSha256",
         "trainingPerformed", "precision")}
    identity["originalSnapshotReplayEncoderSha256"] = digest(canonical / encoder)
    return identity


def compare(actual: np.ndarray, expected: np.ndarray, *, atol: float = 0., rtol: float = 0.) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Comparison requires matching finite arrays")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=True, shape=list(actual.shape), exact=bool(np.array_equal(actual, expected)),
                maxAbsoluteError=float(difference.max(initial=0)),
                meanAbsoluteError=float(difference.mean()) if difference.size else 0., atol=atol, rtol=rtol)


class Artifacts:
    def __init__(self, folder: Path, entries: dict):
        self.folder, self.entries = folder.resolve(), entries
        self.checked = []
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Registered browser artifacts are required")
        for name, entry in entries.items():
            path = self.path(name)
            if path.stat().st_size != entry.get("sizeBytes") or digest(path) != entry.get("sha256"):
                raise ValueError("Browser artifact identity differs: " + name)
            self.checked.append(dict(file=name, sizeBytes=path.stat().st_size, sha256=entry["sha256"]))

    def path(self, name: str) -> Path:
        if not isinstance(name, str) or Path(name).name != name or "\\" in name or "/" in name:
            raise ValueError("Artifact filename must be local")
        path = (self.folder / name).resolve()
        if path.parent != self.folder or name not in self.entries:
            raise ValueError("Artifact escaped its registered directory")
        return path

    def array(self, name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        path = self.path(name)
        expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError("Artifact tensor size differs: " + name)
        value = np.fromfile(path, dtype=dtype).reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite tensor: " + name)
        return value


def rank_av(raw: np.ndarray, names: list[str]) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] != 104 or len(names) != 104:
        raise ValueError("AV104 dimensions differ")
    values = percentile_rank_values(raw)
    for column, name in enumerate(names):
        if name in ABSOLUTE_FEATURE_NAMES:
            values[:, column] = raw[:, column]
    return values


def fused_inputs(times, ranked, embedding_times, tokens, quality, config):
    cache = mobile.MobileVisualCache(Path("unused"), embedding_times, tokens.reshape(-1, 4, 960),
                                    quality, embedding_times + quality[:, 5], {})
    aligned = mobile.align_mobile_features(cache, times)
    if not np.all(aligned["available"] == 1):
        raise ValueError("Incomplete encoder coverage on AV grid")
    values = np.concatenate((ranked, aligned["tokens"].reshape(len(times), -1), aligned["quality"],
                             aligned["feature_age_seconds"][:, None], aligned["available"][:, None]), axis=1).astype(np.float32)
    mean, scale = (np.asarray(config[name], np.float32) for name in ("mean", "scale"))
    if mean.shape != (112,) or scale.shape != (112,) or np.any(scale <= 0):
        raise ValueError("Frozen scaler dimensions differ")
    values[:, :104] = np.clip((values[:, :104] - mean[:104]) / scale[:104], -10, 10)
    values[:, -8:] = np.clip((values[:, -8:] - mean[104:]) / scale[104:], -10, 10)
    return values


def cpu_session(path: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def replay_temporal(session, fused):
    expected = np.empty((len(fused), 4), np.float32)
    for core in range(0, len(fused), 128):
        end = min(len(fused), core + 128)
        left, right = max(0, core - 62), min(len(fused), end + 62)
        logits = session.run(None, {"features": fused[None, left:right]})[0][0]
        expected[core:end] = (1 / (1 + np.exp(-logits)))[core - left:end - left]
    return expected


def validate_snapshots(report, artifacts, tokens_raw, quality, embedding_times, pool, graphs):
    import cv2
    cv2.setNumThreads(1)
    snapshots = report.get("snapshots", [])
    if len(snapshots) < min(3, len(embedding_times)):
        raise ValueError("Representative browser preprocessing snapshots are required")
    session, checks = cpu_session(graphs / "mobile-large-encoder-fp32.onnx"), []
    for item in snapshots:
        index, width, height = item["sampleIndex"], item["width"], item["height"]
        if not 0 <= index < len(embedding_times):
            raise ValueError("Snapshot index out of bounds")
        rgba = artifacts.array(item["rgbaFile"], "u1", (height, width, 4))
        expected, box, expected_quality = mobile.preprocess_frame(np.ascontiguousarray(rgba[:, :, 2::-1]))
        expected_quality[5] = item["pts"] - embedding_times[index]
        image = artifacts.array(item["imageFile"], "<f4", (3, 224, 224))
        expected_pool = mobile.regional_pool_weights(box[None], 7, 7)
        value = dict(sampleIndex=index, timeSeconds=float(embedding_times[index]),
                     pixels=compare(image, expected),
                     contentBox=compare(np.asarray(item["contentBox"]), box),
                     poolWeights=compare(pool, expected_pool, atol=1e-7, rtol=1e-7),
                     quality=compare(quality[index], expected_quality, atol=2e-6, rtol=2e-6))
        compare(np.asarray(item["quality"], np.float32), quality[index])
        expected_tokens = session.run(None, {"image": image[None], "pool_weights": pool})[0].reshape(-1)
        value["encoderCpuReplay"] = compare(tokens_raw[index], expected_tokens, atol=2e-4, rtol=2e-4)
        checks.append(value)
    return checks


def check_source(report, root, offset, duration):
    contract = read(root / "benchmark-contract.json")
    if contract.get("recordingIndex") != "recording-044":
        raise ValueError("Unexpected benchmark recording index")
    if abs(offset) < 1e-9 and abs(duration - contract["fullSourceSeconds"]) < 1e-6:
        scope = "full"
    elif abs(offset - contract["pilotSourceOffsetSeconds"]) < 1e-9 \
            and abs(duration - contract["pilotAnalysisSeconds"]) < 1e-6:
        scope = "pilot"
    else:
        raise ValueError("Analysis duration/offset is outside the registered benchmark scopes")
    source = report.get("source", {})
    expected = contract["sourceIdentities"][scope]
    if source.get("recordingIndex") != "recording-044" or source.get("sha256") != expected["sha256"] \
            or source.get("sizeBytes") != expected["sizeBytes"]:
        raise ValueError("Browser source identity differs from the registered media bytes")
    if source.get("sourceOffsetSeconds", offset) != offset:
        raise ValueError("Browser source offset differs")
    transport = source.get("transport", "url")
    if transport not in ("file", "url"):
        raise ValueError("Unknown browser media transport")
    return dict(scope=scope, recordingIndex="recording-044", sha256=expected["sha256"],
                sizeBytes=expected["sizeBytes"], sourceOffsetSeconds=offset,
                transport=transport,
                analysisDurationSeconds=duration,
                identityBasis=source.get("sha256Basis", "registered caller-provided source receipt"),
                accuracyAlignment="exact source time" if scope == "full" else
                "requested stream-copy offset; approximate excerpt-label alignment")


def score_summary(case, startup):
    scores = case.get("scores")
    if scores is None:
        return None
    result = dict(modelLoadMs=float(scores["modelLoadMs"]))
    for name in ("neural", "production"):
        row = scores[name]
        output = row["output"]
        result[name] = {key: row[key] for key in ("pipelineMs", "allReadyAdditionalMs", "rallyCount")}
        result[name]["outputs"] = {}
        for specialist in ("servingSide", "sideSwitch"):
            value = output[specialist]
            result[name]["outputs"][specialist] = dict(candidateCount=len(value["candidates"]),
                featureRows=value["features"]["rows"], featureColumns=value["features"]["columns"])
    timing = case["timings"]
    result["neuralAllReadyMs"] = (timing["neuralRalliesReadyMs"] + timing["productionMs"]
                                   + result["neural"]["allReadyAdditionalMs"])
    result["productionAllReadyMs"] = (startup["sourceProbeMs"] + timing["avMs"] + timing["productionMs"]
                                       + result["production"]["allReadyAdditionalMs"])
    for key, expected in (("neuralAllReadyMs", result["neuralAllReadyMs"]),
                          ("productionAllReadySharedPassMs", result["productionAllReadyMs"])):
        if abs(timing[key] - expected) > 1e-6:
            raise ValueError("Reported all-ready timing differs from independently summed stages")
    result["timingScope"] = ("Independent proposals with shared score-model-load estimate; neural total includes "
                             "production inference needed by the score specialists. Second proposal pass may use warmer caches.")
    return result


def timing_integrity(report):
    if "timingIntegrity" not in report:
        return dict(clean=None, status="Source-request timing diagnostics were not recorded by this harness revision.")
    source = report["sourceIo"]
    transport = report.get("source", {}).get("transport", "url")
    if report["timingIntegrity"].get("sourceTransport", "url") != transport:
        raise ValueError("Timing and source transport disagree")
    if transport == "file" and (source["requests"] != 0 or source.get("sourceBytesRead") is not None):
        raise ValueError("File transport must not claim HTTP media reads")
    retries = sum("Retrying failed fetch" in item["message"] for item in report["consoleDiagnostics"])
    non_abort = sum(item["error"] != "net::ERR_ABORTED" for item in source["failedBrowserRequests"])
    clean = not (source["streamErrors"] or source["rejectedRanges"] or retries or non_abort)
    if report["timingIntegrity"]["clean"] != clean:
        raise ValueError("Timing-integrity status differs from the recorded source diagnostics")
    return dict(clean=clean, sourceTransport=transport,
                sourceCachePolicy=report["timingIntegrity"]["sourceCachePolicy"],
                retryCount=retries, streamErrorCount=len(source["streamErrors"]),
                rejectedRangeCount=source["rejectedRanges"], nonAbortFetchFailureCount=non_abort,
                intentionalReadAheadCancellationCount=len(source["failedBrowserRequests"]) - non_abort)


def source_code_integrity(report, scope):
    guard = report.get("sourceGuard")
    if guard is None:
        if scope == "full":
            raise ValueError("Full benchmark requires unchanged-code guard")
        return dict(passed=None, status="This earlier pilot did not record a source-code guard.")
    hashes = guard.get("hashes", {})
    if guard.get("passed") is not True or guard.get("fileCount") != len(hashes) or not hashes:
        raise ValueError("Invalid browser source-code guard")
    for name, value in report["code"].items():
        if hashes.get("scripts/" + name) != value:
            raise ValueError("Browser runner hash differs from unchanged-code guard")
    for name, value in report["productionCode"].items():
        if hashes.get("prod/src/lib/on-device/" + name) != value:
            raise ValueError("Production feature code differs from unchanged-code guard")
    return dict(passed=True, fileCount=len(hashes),
                hashManifestSha256=hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def evaluate_result(result_path: Path, root: Path, labels_path: Path, source_offset: float, graphs_override=None):
    report = read(result_path)
    if report.get("kind") != "distilled-large-real-video-browser-v1" or report.get("schemaVersion") != 1 \
            or report.get("labelsUsed") is not False or report.get("trainingPerformed") is not False:
        raise ValueError("Expected completed, label-blind browser pipeline result")
    if report.get("browserErrors"):
        raise ValueError("Browser errors prevent qualification")
    helper = native_report()
    if digest(labels_path) != helper.GOLD_SHA256:
        raise ValueError("Saved human gold revision differs")
    duration = float(report["analysisDurationSeconds"])
    source = check_source(report, root, source_offset, duration)
    if report["analysisWindow"] != dict(start=0, end=duration):
        raise ValueError("Unexpected browser analysis window")
    choice = report["model"]["selectionMode"]
    if choice not in ("f1", "recall"):
        raise ValueError("Unexpected frozen selection")
    canonical = root / ("graphs-" + choice)
    # Always validate the immutable original as well as an optional graph-only
    # portability derivative. The selected checkpoint/scaler/decoder cannot vary.
    helper.graph_identity(canonical, choice)
    graphs = Path(graphs_override) if graphs_override is not None else canonical
    identity = validated_graph_identity(graphs, canonical, choice, helper)
    expected_assets = {row["component"]: dict(sizeBytes=row["bytes"], sha256=row["sha256"])
                       for row in identity["graphFiles"]}
    if report.get("graphAssets") != expected_assets:
        raise ValueError("Served browser graph receipts differ from frozen graph identity")
    config = read(graphs / "mobile-large-pipeline.json")
    if report["model"] != config:
        raise ValueError("Browser used a different pipeline/scaler/decoder")
    artifacts = Artifacts(result_path.parent, report["artifacts"])
    embedding = report["embedding"]
    count = math.ceil(duration * 2 - 1e-9)
    if embedding["sampleCount"] != count or embedding["tokenDimension"] != 3840:
        raise ValueError("Embedding row count/dimension differs")
    times = artifacts.array(embedding["timesFile"], "<f8", (count,))
    compare(times, np.arange(count, dtype=np.float64) / 2)
    pts = artifacts.array(embedding["selectedPtsFile"], "<f8", (count,))
    if np.any(np.diff(pts) < 0) or np.max(abs(pts - times)) > .25 + 1e-9:
        raise ValueError("Selected PTS violates nearest half-tick coverage")
    quality = artifacts.array(embedding["qualityFile"], "<f4", (count, 6))
    compare(quality[:, 5], (pts - times).astype(np.float32))
    raw = artifacts.array(embedding["tokensUnroundedFile"], "<f4", (count, 3840))
    tokens = artifacts.array(embedding["tokensFile"], "<f4", (count, 3840))
    rounding = compare(tokens, raw.astype(np.float16).astype(np.float32))
    pool = artifacts.array(embedding["poolWeightsFile"], "<f4", (1, 4, 7, 7))
    snapshot_checks = validate_snapshots(report, artifacts, raw, quality, times, pool, canonical)
    labels = helper.shifted_labels(read(labels_path), source_offset, duration)
    expected_names = list(feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)))
    cases, ids = [], set()
    session = cpu_session(canonical / "mobile-large-tcn-dynamic-fp32.onnx")
    from analysis.neural_development import decode
    for case in report["avCases"]:
        if case["id"] in ids or case["names"] != expected_names or case["columns"] != 104:
            raise ValueError("AV case identity/schema differs")
        ids.add(case["id"])
        n = case["rows"]
        av_times = artifacts.array(case["timesFile"], "<f8", (n,))
        if n < 1 or np.any(np.diff(av_times) <= 0) or av_times[0] < 0 or av_times[-1] >= duration:
            raise ValueError("AV presentation grid is invalid")
        if case["id"] == "corrected":
            compare(av_times, np.arange(math.ceil(duration * 4 - 1e-9), dtype=np.float64) / 4)
        raw_av = artifacts.array(case["rawAvFile"], "<f4", (n, 104))
        ranked = artifacts.array(case["rankedAvFile"], "<f4", (n, 104))
        rank_check = compare(ranked, rank_av(raw_av, expected_names))
        fused = artifacts.array(case["fusedFile"], "<f4", (n, 3952))
        fusion_check = compare(fused, fused_inputs(av_times, ranked, times, tokens, quality, config))
        probabilities = artifacts.array(case["probabilitiesFile"], "<f4", (n, 4))
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Invalid browser probabilities")
        replayed_probabilities = replay_temporal(session, fused)
        temporal_check = compare(probabilities, replayed_probabilities, atol=3e-5, rtol=3e-4)
        example = SimpleNamespace(times=av_times, valid=np.ones(n, bool), duration=duration)
        decoded = decode(example, probabilities, config["decoder"])
        bounds = np.asarray([(r.start, r.end) for r in decoded], np.float64).reshape(-1, 2)
        actual = np.asarray([(r["start"], r["end"]) for r in case["rallies"]], np.float64).reshape(-1, 2)
        decoder_check = compare(actual, bounds, atol=1e-8)
        cpu_decoded = decode(example, replayed_probabilities, config["decoder"])
        cpu_bounds = np.asarray([(r.start, r.end) for r in cpu_decoded], np.float64).reshape(-1, 2)
        cpu_boundary_check = dict(exact=bool(np.array_equal(cpu_bounds, bounds)),
                                 browserRallyCount=len(bounds), cpuReplayRallyCount=len(cpu_bounds),
                                 maxBoundaryDifferenceSeconds=(float(np.max(np.abs(cpu_bounds - bounds), initial=0))
                                                               if cpu_bounds.shape == bounds.shape else None))
        entry = dict(id=case["id"], rows=n, rankParity=rank_check, fusionParity=fusion_check,
                     alignment=case.get("alignment"),
                     temporalCpuReplay=temporal_check, decoderParity=decoder_check,
                     cpuReplayDecoderParity=cpu_boundary_check,
                     neuralEvaluation=helper.evaluate(case, labels, duration),
                     timings=case["timings"], storedTensorBytes=n * (104 + 104 + 3952 + 4) * 4 + n * 8)
        production = case.get("production", {})
        if isinstance(production, dict) and isinstance(production.get("intervals"), list):
            entry["productionEvaluation"] = helper.evaluate(dict(rallies=[row for row in production["intervals"]
                                                             if row.get("included", True)]), labels, duration)
        scores = score_summary(case, report["timings"])
        if scores is not None:
            entry["scores"] = scores
        if not cpu_boundary_check["exact"]:
            entry["cpuReplayEvaluation"] = helper.evaluate(
                dict(rallies=[dict(start=r.start, end=r.end) for r in cpu_decoded]), labels, duration)
        cases.append(entry)
    if not cases:
        raise ValueError("No completed AV variants")
    return dict(schemaVersion=1, kind="distilled-large-browser-independent-evaluation-v1", passed=True,
                artifactIndex="private-reference-0223", source=source, goldSha256=helper.GOLD_SHA256,
                resultSha256=digest(result_path), model=identity, runtime=report["runtime"],
                host={key: report.get("host", {}).get(key) for key in
                      ("browserVersion", "nodeVersion", "platform", "architecture", "cpu")},
                memory={key: report.get("memory", {}).get(key) for key in
                        ("metric", "maximumSampledJsHeapUsedBytes")},
                timingIntegrity=timing_integrity(report),
                sourceCodeIntegrity=source_code_integrity(report, source["scope"]),
                targetPaddingSeconds=2, joinGapSeconds=3, tokenRoundingParity=rounding,
                snapshotChecks=snapshot_checks, avCases=cases, embeddingTimings=embedding["timings"],
                timings=report["timings"], artifactChecks=artifacts.checked,
                storage=dict(embeddingRawFloat32Bytes=raw.nbytes, embeddingRoundedFloat32Bytes=tokens.nbytes,
                             embeddingPackedFloat16BytesCalculated=tokens.size * 2,
                             embeddingQualityBytes=quality.nbytes, sampleCount=count),
                limitations=["One-recording runtime validation; no training, calibration or model selection.",
                             "Browser decode/color accuracy relative to original desktop frames is separate from snapshot preprocessing parity.",
                             "Temporal replay uses the browser's actual tensors; agreement alone does not prove model accuracy.",
                             "Desktop browser timing is not a physical-phone measurement."])


def sanitized_summary(result):
    output = {key: result[key] for key in ("schemaVersion", "kind", "passed", "artifactIndex", "source",
              "goldSha256", "resultSha256", "model", "runtime", "host", "memory", "timingIntegrity", "sourceCodeIntegrity", "targetPaddingSeconds", "joinGapSeconds",
              "tokenRoundingParity", "snapshotChecks", "avCases", "embeddingTimings", "timings", "storage", "limitations")}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root-index", default="private-reference-0223")
    parser.add_argument("--source-offset", type=float, required=True)
    parser.add_argument("--graphs", type=Path, help="Optional pooling-only derived graph directory for this selection")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    labels = args.labels or Path(private_value("private-reference-0218")).parent / "recording-044-label-snapshot.private.json"
    result = evaluate_result(args.result, Path(private_value(args.root_index)), labels, args.source_offset, args.graphs)
    repository = Path(__file__).resolve().parents[1]
    if args.output.resolve() == repository or repository in args.output.resolve().parents:
        raise ValueError("Detailed runtime evaluation must remain outside the repository")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(sanitized_summary(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, scope=result["source"]["scope"], selection=result["model"]["selection"],
                         cases=[dict(id=case["id"], metrics=case["neuralEvaluation"]["primary"],
                                     whollyMissed=case["neuralEvaluation"]["whollyMissedSavedHumanRallies"])
                                for case in result["avCases"]])))


if __name__ == "__main__":
    main()
