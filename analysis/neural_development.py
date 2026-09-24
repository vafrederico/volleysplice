"""Preregistered nested development study for compact rally networks.

Historical semantic studies remain immutable. This runner selects exclusively on
the canonical padded-precision/core-recall metric, never on an outer fold.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DecoderConfig, FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from .crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
from .decoder import decode_probabilities
from .features import ABSOLUTE_FEATURE_NAMES, feature_names, percentile_rank_values
from .schema import Interval, labels_for_times, load_manifest, mask_for_times


SEEDS = (3407, 1729, 20260918)
CHECKPOINT_EPOCHS = (5, 10, 20, 30)
KINDS = ("linear", "mlp", "tcn", "dino_tcn")
PROTECTED_GROUPS = frozenset({private_value('source-group-008')})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Example:
    id: str
    group: str
    duration: float
    times: np.ndarray
    values: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    truth: tuple[Interval, ...]
    ignored: tuple[Interval, ...]
    environment: str

    def row(self, predictions: list[Interval]) -> dict[str, Any]:
        return {"id": self.id, "sourceGroup": self.group,
                "durationSeconds": self.duration, "rallies": self.truth,
                "ignoredIntervals": self.ignored, "predictions": predictions}


def boundary_targets(times: np.ndarray, boundaries: list[float]) -> np.ndarray:
    result = np.zeros(len(times), dtype=np.float32)
    for boundary in boundaries:
        distance = np.abs(times - boundary)
        pulse = np.exp(-0.5 * (distance / 0.35) ** 2)
        pulse[distance > 1.0] = 0
        result = np.maximum(result, pulse.astype(np.float32))
    return result


def segments(valid: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def supervision_mask(example: Example) -> np.ndarray:
    """Live validity plus a one-second uncertainty halo for boundary targets.

    Boundary pulses have radius one second. Censor both boundary heads around
    ignored spans so an unobservable event cannot leak a positive pulse (or a
    misleading negative target) into the neighboring evaluable segment.
    """
    result = np.repeat(example.valid[:, None], 3, axis=1).astype(np.float32)
    for interval in example.ignored:
        near_ignored = ((example.times >= interval.start - 1.0)
                        & (example.times <= interval.end + 1.0))
        result[near_ignored, 1:] = 0.0
    return result


def load_examples(manifest_path: Path, with_dino: bool) -> list[Example]:
    manifest = load_manifest(manifest_path)
    result = []
    for record in manifest.recordings:
        if record.split not in {"train", "validation"} or record.source_group in PROTECTED_GROUPS:
            raise ValueError(f"protected/nondevelopment row refused: {record.id}")
        if record.consent.get("train") is not True:
            raise ValueError(f"missing explicit training consent: {record.id}")
        cache = record.raw["featureCaches"]["audiovisual"]
        path = Path(cache["path"])
        if file_sha256(path) != cache["sha256"]:
            raise ValueError(f"changed feature cache: {record.id}")
        with np.load(path, allow_pickle=False) as data:
            times = data["times"].astype(np.float64)
            raw_values = data["values"].astype(np.float32)
            names = tuple(str(item) for item in data["names"])
            metadata = json.loads(str(data["metadata_json"].item()))
        expected_names = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
        if names != expected_names or raw_values.shape != (len(times), 104) or not np.isfinite(raw_values).all():
            raise ValueError(f"expected finite current 104-signal cache: {record.id}")
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError(f"invalid feature timestamps: {record.id}")
        if np.any(np.diff(times) <= 0) or np.any(np.abs(np.diff(times) - 0.25) > 0.10):
            raise ValueError(f"irregular 4Hz timeline: {record.id}")
        values = percentile_rank_values(raw_values)
        for index, name in enumerate(names):
            if name in ABSOLUTE_FEATURE_NAMES:
                values[:, index] = raw_values[:, index]
        if with_dino:
            entry = record.raw["featureCaches"]["dino"]
            dino_path = Path(entry["path"])
            if file_sha256(dino_path) != entry["sha256"]:
                raise ValueError(f"changed DINO cache: {record.id}")
            with np.load(dino_path, allow_pickle=False) as data:
                dino_times = data["timestamps"].astype(np.float64)
                tokens = data["tokens"].astype(np.float32)
            if (dino_times.ndim != 1 or not len(dino_times) or not np.isfinite(dino_times).all()
                    or np.any(np.diff(dino_times) <= 0) or tokens.shape != (len(dino_times), 10, 384)
                    or not np.isfinite(tokens).all()):
                raise ValueError(f"invalid DINO feature tensors: {record.id}")
            right = np.clip(np.searchsorted(dino_times, times), 0, len(dino_times) - 1)
            left = np.maximum(0, right - 1)
            nearest = np.where(np.abs(dino_times[left] - times) <= np.abs(dino_times[right] - times), left, right)
            if np.max(np.abs(dino_times[nearest] - times)) > 0.125 + 1e-8:
                raise ValueError(f"DINO timestamp mismatch: {record.id}")
            values = np.concatenate([values, tokens[nearest].reshape(len(times), -1)], axis=1)
        valid = mask_for_times(times, record.ignored_intervals)
        targets = np.stack([labels_for_times(times, record.rallies),
                            boundary_targets(times, [row.start for row in record.rallies]),
                            boundary_targets(times, [row.end for row in record.rallies])], axis=1)
        result.append(Example(record.id, record.source_group, float(metadata["duration"]), times,
                              values, targets, valid, record.rallies, record.ignored_intervals,
                              record.environment))
    if len({row.group for row in result}) < 3:
        raise ValueError("nested study needs at least three independent development groups")
    return result


def fit_scaler(examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    if not examples or not any(np.any(row.valid) for row in examples):
        raise ValueError("scaler requires valid training samples")
    values = np.concatenate([row.values[row.valid, :104] for row in examples])
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(values.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    return mean, scale


def standardized(example: Example, mean: np.ndarray, scale: np.ndarray, kind: str) -> np.ndarray:
    av = np.clip((example.values[:, :104] - mean) / scale, -10.0, 10.0)
    return np.concatenate([av, example.values[:, 104:]], axis=1) if kind == "dino_tcn" else av


def model_for(kind: str):
    from .compact_temporal_model import CompactTemporalConfig, CompactTemporalNetwork, DinoFusionNetwork
    return DinoFusionNetwork() if kind == "dino_tcn" else CompactTemporalNetwork(CompactTemporalConfig(kind=kind))


def predict(model, example: Example, mean: np.ndarray, scale: np.ndarray, kind: str, device: str) -> np.ndarray:
    import torch
    model.eval()
    result = np.zeros((len(example.times), 3), dtype=np.float32)
    values = standardized(example, mean, scale, kind)
    with torch.inference_mode():
        # Reset at ignored spans; only score interiors of overlapping chunks.
        for start, finish in segments(example.valid):
            for core_start in range(start, finish, 128):
                core_end = min(finish, core_start + 128)
                left, right = max(start, core_start - 62), min(finish, core_end + 62)
                inputs = torch.from_numpy(values[left:right][None]).to(device)
                scores = torch.sigmoid(model(inputs))[0].cpu().numpy()
                result[core_start:core_end] = scores[core_start-left:core_end-left]
    return result


def decoder_candidates() -> list[dict[str, Any]]:
    return [{"smoothing": smoothing, "enter": enter, "minimum": minimum, "boundary": boundary}
            for smoothing in (0.5, 1.0) for enter in (0.35, 0.5, 0.65)
            for minimum in (0.25, 1.0) for boundary in (False, True)]


def decode(example: Example, scores: np.ndarray, settings: dict[str, Any]) -> list[Interval]:
    config = DecoderConfig(smoothing_seconds=settings["smoothing"], enter_threshold=settings["enter"],
                           exit_threshold=settings["enter"]-0.1, min_live_seconds=settings["minimum"],
                           bridge_gap_seconds=0.5, short_event_min_seconds=0.25, short_event_threshold=0.9)
    result = []
    for left, right in segments(example.valid):
        times = example.times[left:right]
        rows, _ = decode_probabilities(times, scores[left:right, 0], example.duration, config, 4.0)
        segment_start = max(0.0, float(times[0]) - 0.125)
        segment_end = min(example.duration, float(times[-1]) + 0.125)
        for row in rows:
            start, end = max(segment_start, row.start), min(segment_end, row.end)
            if settings["boundary"]:
                for channel, boundary in ((1, start), (2, end)):
                    indices = np.flatnonzero((np.abs(times-boundary) <= 0.75) & (scores[left:right, channel] >= 0.65))
                    if len(indices):
                        peak = indices[int(np.argmax(scores[left:right, channel][indices]))]
                        value = float(times[peak])
                        if channel == 1:
                            start = value
                        else:
                            end = value
            if end > start:
                result.append(Interval(start, end))
    return result


def primary_score(examples: list[Example], probabilities: dict[str, np.ndarray], settings: dict[str, Any]) -> float:
    rows = [RecordingIntervals(row.id, "development", row.duration, row.truth,
                               tuple(decode(row, probabilities[row.id], settings)), row.ignored) for row in examples]
    return float(evaluate_f1_pad_p_core_r(rows, [2.0], 3.0)[0]["F1_padP_coreR"])


def choose_settings(examples: list[Example], checkpoint_predictions: dict[int, dict[str, np.ndarray]]) -> tuple[int, dict, float]:
    best = None
    for epoch, predictions in sorted(checkpoint_predictions.items()):
        for settings in decoder_candidates():
            score = primary_score(examples, predictions, settings)
            # Stable ties prefer the first/smaller epoch and simpler decoder.
            if best is None or score > best[2] + 1e-12:
                best = (epoch, settings, score)
    assert best is not None
    return best


def training_chunks(examples: list[Example], mean: np.ndarray, scale: np.ndarray, kind: str):
    """Central128 ticks +62 halos, each scored tick occurs once per epoch pool."""
    rows = []
    groups = {row.group for row in examples}
    group_recordings = {group: sum(row.group == group for row in examples) for group in groups}
    for example in examples:
        values = standardized(example, mean, scale, kind)
        head_valid = supervision_mask(example)
        one = []
        for start, finish in segments(example.valid):
            for core_start in range(start, finish, 128):
                core_end = min(finish, core_start + 128)
                indexes = np.arange(core_start - 62, core_start + 128 + 62)
                inside = (indexes >= start) & (indexes < finish)
                safe = np.clip(indexes, start, finish - 1)
                x = np.zeros((252, values.shape[1]), dtype=np.float32)
                x[inside] = values[safe[inside]]
                y = example.targets[safe].copy()
                mask = np.zeros((252, 3), dtype=np.float32)
                core_slice = slice(62, 62 + core_end-core_start)
                mask[core_slice] = head_valid[safe[core_slice]]
                # Actual clip edges need the same boundary behavior as inference.
                # Slice shorter chunks later rather than supplying extra temporal zeros.
                first, last = np.flatnonzero(inside)[[0, -1]]
                one.append((x[first:last+1], y[first:last+1], mask[first:last+1]))
        for x, y, mask in one:
            weight = 1.0 / (group_recordings[example.group] * max(1, len(one)))
            rows.append((x, y, mask, weight))
    return rows


def fit_model(train: list[Example], validation: list[Example], kind: str, seed: int,
              epochs: tuple[int, ...], destination: Path, device: str, contract_hash: str) -> dict[int, dict[str, np.ndarray]]:
    import torch
    if not train or not validation or {row.group for row in train} & {row.group for row in validation}:
        raise ValueError("fitting and validation must be nonempty disjoint source groups")
    completed = destination / "completed.json"
    if completed.exists():
        meta = json.loads(completed.read_text())
        if meta["contractSha256"] != contract_hash:
            raise ValueError(f"resume contract changed: {destination}")
        if meta["trainIds"] != [row.id for row in train] or meta["validationIds"] != [row.id for row in validation]:
            raise ValueError("resume fold identity changed")
        for name, digest in meta["artifacts"].items():
            if file_sha256(destination / name) != digest:
                raise ValueError(f"changed checkpoint artifact: {destination / name}")
        output = {}
        for epoch in epochs:
            with np.load(destination / f"predictions-{epoch}.npz", allow_pickle=False) as data:
                output[epoch] = {row.id: data[row.id] for row in validation}
        return output
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    mean, scale = fit_scaler(train)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = model_for(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_rows = np.concatenate([row.targets for row in train])
    target_masks = np.concatenate([supervision_mask(row) for row in train])
    positive = (target_rows * target_masks).sum(axis=0)
    valid_counts = target_masks.sum(axis=0)
    positive_weight = np.minimum(20.0, np.sqrt((valid_counts-positive) / np.maximum(positive, 1.0)))
    positive_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    head_weight = torch.tensor([1.0, 0.5, 0.5], device=device)
    chunks = training_chunks(train, mean, scale, kind)
    weights = np.array([row[3] for row in chunks], dtype=np.float64)
    weights /= weights.sum()
    history = []
    outputs = {}
    started = time.perf_counter()
    for epoch in range(1, max(epochs)+1):
        model.train()
        losses = []
        chosen = rng.choice(len(chunks), size=len(chunks), replace=True, p=weights)
        # Group equal lengths to avoid padded positions leaking through biases.
        buckets: dict[int, list[int]] = {}
        for index in chosen:
            buckets.setdefault(len(chunks[index][0]), []).append(int(index))
        for indexes in buckets.values():
            for start in range(0, len(indexes), 16):
                batch = [chunks[index] for index in indexes[start:start+16]]
                x = torch.from_numpy(np.stack([row[0] for row in batch])).to(device)
                y = torch.from_numpy(np.stack([row[1] for row in batch])).to(device)
                mask = torch.from_numpy(np.stack([row[2] for row in batch])).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss_elements = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight, reduction="none")
                loss = (loss_elements * head_weight * mask).sum() / mask[..., 0].sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        if epoch in epochs:
            outputs[epoch] = {row.id: predict(model, row, mean, scale, kind, device) for row in validation}
            np.savez_compressed(destination / f"predictions-{epoch}.npz", **outputs[epoch])
            # Tensor-only, non-pickle checkpoint plus explicit transform metadata.
            state = {f"model::{key}": value.detach().cpu().numpy() for key, value in model.state_dict().items()}
            np.savez_compressed(destination / f"weights-{epoch}.npz", mean=mean, scale=scale, **state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    write_json(completed, {"contractSha256": contract_hash, "kind": kind, "seed": seed,
                          "trainIds": [row.id for row in train], "validationIds": [row.id for row in validation],
                          "parameters": sum(value.numel() for value in model.parameters()), "epochs": epochs,
                          "history": history, "wallSeconds": time.perf_counter()-started,
                          "artifacts": {f"{stem}-{epoch}.npz": file_sha256(destination / f"{stem}-{epoch}.npz")
                                        for epoch in epochs for stem in ("predictions", "weights")},
                          "peakAllocatedCudaBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None})
    return outputs


def run_study(manifest_path: Path, output: Path, kinds: tuple[str, ...], seeds: tuple[int, ...],
              checkpoints: tuple[int, ...], device: str) -> dict[str, Any]:
    import torch
    from .neural_evaluation import evaluate_predictions
    if (output / "report.json").exists():
        raise FileExistsError("completed study exists; use a new output directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    examples = load_examples(manifest_path, "dino_tcn" in kinds)
    groups = sorted({row.group for row in examples})
    code_paths = [Path(__file__).with_name(name) for name in
                  ("neural_development.py", "compact_temporal_model.py", "neural_evaluation.py",
                   "crop_evaluation.py", "decoder.py", "features.py", "config.py", "schema.py")]
    contract = {"schemaVersion": 1, "manifestSha256": file_sha256(manifest_path),
                "kinds": kinds, "seeds": seeds, "checkpointEpochs": checkpoints, "groups": groups,
                "primaryMetric": "F1_padP_coreR", "targetPaddingSeconds": 2, "joinGapSeconds": 3,
                "paddingSweep": [0, 1, 2, 3], "decoderCandidates": decoder_candidates(),
                "training": {"optimizer": "AdamW", "learningRate": 0.001, "weightDecay": 0.0001,
                             "batchSize": 16, "sourceGroupRecordingBalancedSampling": True, "tf32": False,
                             "positiveWeight": "sqrt negative/positive capped20", "headWeights": [1, .5, .5],
                             "boundarySigmaSeconds": .35, "boundaryRadiusSeconds": 1, "boundaryMaskIgnoredHaloSeconds": 1,
                             "coreTicks": 128, "haloTicks": 62, "normalization": "recording percentiles then fit-only scaler"},
                "protectedTestOpened": False, "productionPromotionAllowed": False,
                "code": {path.name: file_sha256(path) for path in code_paths},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "device": device,
                                "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None}}
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if (output / "preregistration.json").exists():
        existing = json.loads((output / "preregistration.json").read_text())
        if existing["sha256"] != contract_hash:
            raise ValueError("existing preregistration differs; refuse adaptive overwrite")
    else:
        write_json(output / "preregistration.json", {"sha256": contract_hash, "contract": contract})
    all_results = []
    for kind in kinds:
        for seed in seeds:
            outer_rows = []
            selections = []
            for outer_index, outer in enumerate(groups):
                fitting = [row for row in examples if row.group != outer]
                held = [row for row in examples if row.group == outer]
                checkpoint_predictions = {epoch: {} for epoch in checkpoints}
                for inner_index, inner in enumerate(group for group in groups if group != outer):
                    training = [row for row in fitting if row.group != inner]
                    validating = [row for row in fitting if row.group == inner]
                    location = output / "fits" / kind / str(seed) / f"outer-{outer_index}" / f"inner-{inner_index}"
                    print(f"FIT {kind} seed={seed} outer={outer} inner={inner}", flush=True)
                    predictions = fit_model(training, validating, kind, seed, checkpoints, location, device, contract_hash)
                    for epoch in checkpoints:
                        checkpoint_predictions[epoch].update(predictions[epoch])
                epoch, settings, inner_score = choose_settings(fitting, checkpoint_predictions)
                location = output / "fits" / kind / str(seed) / f"outer-{outer_index}" / "refit"
                print(f"REFIT {kind} seed={seed} outer={outer} epochs={epoch} innerF1={inner_score:.6f}", flush=True)
                probabilities = fit_model(fitting, held, kind, seed, (epoch,), location, device, contract_hash)[epoch]
                for row in held:
                    outer_rows.append(row.row(decode(row, probabilities[row.id], settings)))
                selections.append({"heldSourceGroup": outer, "epoch": epoch, "decoder": settings, "innerF1_padP_coreR": inner_score})
            report = evaluate_predictions(outer_rows)
            result = {"kind": kind, "seed": seed, "selections": selections, "evaluation": report,
                      "predictions": [{**row, "rallies": [item.to_dict() for item in row["rallies"]],
                                       "ignoredIntervals": [item.to_dict() for item in row["ignoredIntervals"]],
                                       "predictions": [item.to_dict() for item in row["predictions"]]} for row in outer_rows]}
            write_json(output / f"result-{kind}-{seed}.json", result)
            all_results.append(result)
            print(f"RESULT {kind} seed={seed} F1_padP_coreR={report['objective']:.6f}", flush=True)
    report = {"schemaVersion": 1, "contractSha256": contract_hash, "status": "completed-development-screen",
              "protectedTestOpened": False, "productionPromotionAllowed": False,
              "records": len(examples), "sourceGroups": groups, "results": all_results,
              "ranking": sorted([{"kind": kind,
                                   "meanSeedF1_padP_coreR": float(np.mean([row["evaluation"]["objective"] for row in all_results if row["kind"] == kind])),
                                   "minSeedF1_padP_coreR": min(row["evaluation"]["objective"] for row in all_results if row["kind"] == kind)} for kind in kinds],
                                key=lambda row: -row["meanSeedF1_padP_coreR"])}
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--checkpoint-epoch", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoints = tuple(sorted(set(args.checkpoint_epoch or CHECKPOINT_EPOCHS)))
    if not checkpoints or min(checkpoints) < 1:
        parser.error("checkpoint epochs must be positive")
    run_study(args.manifest.resolve(), args.output.resolve(), tuple(args.kind or KINDS),
              tuple(args.seed or SEEDS), checkpoints, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
