"""Cross frozen browser AV values and time grids without decoding or retraining.

This is a counterfactual input diagnostic, never a deployable pipeline or timing
benchmark. Media and gold locations resolve through the private ledger.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "independent_browser_evaluation", Path(__file__).with_name("evaluate-distilled-large-browser.py"))
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def evaluate_cross(result_path, root, labels_path):
    report = evaluation.read(result_path)
    prior_path = result_path.with_name("independent-summary.json")
    prior = evaluation.read(prior_path)
    if prior.get("passed") is not True or prior.get("resultSha256") != evaluation.digest(result_path):
        raise ValueError("A matching independent validation receipt is required")
    if prior["source"]["scope"] != "full" or prior["source"].get("transport") != "file" \
            or prior["timingIntegrity"].get("clean") is not True:
        raise ValueError("Counterfactual inputs must come from a clean full file benchmark")
    helper = evaluation.native_report()
    if evaluation.digest(labels_path) != helper.GOLD_SHA256 or prior["goldSha256"] != helper.GOLD_SHA256:
        raise ValueError("Frozen gold revision differs")
    duration = float(report["analysisDurationSeconds"])
    source = evaluation.check_source(report, root, 0, duration)
    evaluation.source_code_integrity(report, "full")
    choice = report["model"]["selectionMode"]
    graphs = root / ("graphs-" + choice)
    identity = helper.graph_identity(graphs, choice)
    config = evaluation.read(graphs / "mobile-large-pipeline.json")
    if report["model"] != config:
        raise ValueError("Frozen pipeline identity differs")
    artifacts = evaluation.Artifacts(result_path.parent, report["artifacts"])
    embedding = report["embedding"]
    count = math.ceil(duration * 2 - 1e-9)
    embedding_times = artifacts.array(embedding["timesFile"], "<f8", (count,))
    tokens = artifacts.array(embedding["tokensFile"], "<f4", (count, 3840))
    quality = artifacts.array(embedding["qualityFile"], "<f4", (count, 6))
    observed = {item["id"]: item for item in report["avCases"]}
    if set(observed) != {"legacy", "corrected"}:
        raise ValueError("Exactly the legacy and corrected observed inputs are required")
    n = math.ceil(duration * 4 - 1e-9)
    if any(item["rows"] != n or item["columns"] != 104 for item in observed.values()):
        raise ValueError("AV row sequences are incompatible; interpolation is forbidden")
    times = {key: artifacts.array(item["timesFile"], "<f8", (n,)) for key, item in observed.items()}
    evaluation.compare(times["corrected"], np.arange(n, dtype=np.float64) / 4)
    displacement = np.abs(times["legacy"] - times["corrected"])
    if np.any(np.diff(times["legacy"]) <= 0) or np.max(displacement) > .125 + 1e-6:
        raise ValueError("Actual PTS rows do not correspond to the same nominal AV ticks")
    token_indexes = {key: np.searchsorted(embedding_times, value, side="right") - 1
                     for key, value in times.items()}
    if any(np.any(value < 0) for value in token_indexes.values()):
        raise ValueError("Counterfactual would begin before the first available embedding")
    token_time_difference = embedding_times[token_indexes["corrected"]] - embedding_times[token_indexes["legacy"]]
    ranked = {key: artifacts.array(item["rankedAvFile"], "<f4", (n, 104))
              for key, item in observed.items()}
    for key, item in observed.items():
        raw = artifacts.array(item["rawAvFile"], "<f4", (n, 104))
        evaluation.compare(ranked[key], evaluation.rank_av(raw, item["names"]))
    labels = helper.shifted_labels(evaluation.read(labels_path), 0, duration)
    from analysis.neural_development import decode
    session = evaluation.cpu_session(graphs / "mobile-large-tcn-dynamic-fp32.onnx")
    cases = []
    for values_source in ("legacy", "corrected"):
        for timeline_source in ("legacy", "corrected"):
            fused = evaluation.fused_inputs(times[timeline_source], ranked[values_source],
                                             embedding_times, tokens, quality, config)
            probabilities = evaluation.replay_temporal(session, fused)
            example = SimpleNamespace(times=times[timeline_source], valid=np.ones(n, bool), duration=duration)
            rallies = [dict(start=r.start, end=r.end) for r in decode(example, probabilities, config["decoder"])]
            diagonal = values_source == timeline_source
            row = dict(avValues=values_source, fusionAndDecoderTimeline=timeline_source,
                       counterfactual=not diagonal, rallyCount=len(rallies),
                       evaluation=helper.evaluate(dict(rallies=rallies), labels, duration))
            if diagonal:
                baseline = observed[values_source]
                row["savedFusionParity"] = evaluation.compare(fused, artifacts.array(
                    baseline["fusedFile"], "<f4", (n, 3952)))
                row["savedProbabilityParity"] = evaluation.compare(probabilities, artifacts.array(
                    baseline["probabilitiesFile"], "<f4", (n, 4)), atol=3e-5, rtol=3e-4)
                row["savedDecoderParity"] = evaluation.compare(
                    np.asarray([(r["start"], r["end"]) for r in rallies]).reshape(-1, 2),
                    np.asarray([(r["start"], r["end"]) for r in baseline["rallies"]]).reshape(-1, 2), atol=1e-8)
            cases.append(row)
    return dict(selection=choice, source=source, modelIdentity=identity,
                resultSha256=evaluation.digest(result_path), independentReceiptSha256=evaluation.digest(prior_path),
                avRows=n, embeddingRows=count, maximumTimestampDisplacementSeconds=float(displacement.max()),
                causalEmbeddingSelection=dict(rowsUsingDifferentTokens=int(np.count_nonzero(token_time_difference)),
                    rowsUsingOlderTokensInLegacy=int(np.count_nonzero(token_time_difference > 0)),
                    maximumTokenTimestampDifferenceSeconds=float(np.max(np.abs(token_time_difference)))),
                rankedAvExact=bool(np.array_equal(ranked["legacy"], ranked["corrected"])),
                cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--root-index", default="private-reference-0223")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.output.resolve() == repository or repository in args.output.resolve().parents:
        raise ValueError("Counterfactual diagnostic must remain outside the repository")
    root = Path(evaluation.private_value(args.root_index))
    labels = Path(evaluation.private_value("private-reference-0218")).parent / "recording-044-label-snapshot.private.json"
    results = [evaluate_cross(path, root, labels) for path in args.result]
    if len({item["selection"] for item in results}) != len(results):
        raise ValueError("Duplicate selected model")
    output = dict(schemaVersion=1, kind="distilled-large-browser-av-timeline-counterfactual-v1",
                  artifactIndex=args.root_index, recordingIndex="recording-044",
                  goldSha256=evaluation.digest(labels), targetPaddingSeconds=2, joinGapSeconds=3,
                  trainingPerformed=False, videoDecoded=False, interpolationPerformed=False,
                  benchmark=False, executionProvider="ONNX Runtime CPU, one thread", selections=results,
                  interpretation="AV-value changes bundle selected video frames, resize, video features, and audio-window timing because audio extraction uses each case's time grid. They are not a video-pixel-only effect. Timeline changes jointly affect causal embedding hold, quality age, and decoder boundaries. Embeddings and fitted scaler/decoder remain fixed per selection.",
                  limitations=["Counterfactual inputs are not an implemented or timed product pipeline.",
                               "This single recording does not establish generalization or select a new model."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, selections=[dict(selection=item["selection"], cases=[
        dict(avValues=case["avValues"], timeline=case["fusionAndDecoderTimeline"],
             primary=case["evaluation"]["primary"], whollyMissed=case["evaluation"]["whollyMissedSavedHumanRallies"])
        for case in item["cases"]]) for item in results])))


if __name__ == "__main__":
    main()
