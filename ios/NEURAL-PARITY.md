# iOS selected neural pipeline

The iOS app adds the same product choices as web and Android: **Maximum coverage ·
BETA** (new-analysis default), **Balanced · BETA**, and **Legacy model**. Both neural
choices use the frozen distilled MobileNetV3-Large + TCN bundles selected previously;
this port does not retrain, recalibrate or select on device test results.

## Runtime and progress contract

The selected bundle includes its matching encoder, temporal head, scalers and
decoder. Switching variants runs only the requested encoder and head, reusing
compatible base features and any cache for that exact bundle. Embeddings from one
variant must not feed the other. The portable ONNX graphs and pipeline configs are
size/hash pinned in [the iOS manifest](../models/distilled-large/ios-manifest.json).

The initial path is FP32 ONNX Runtime on CPU. GPU/Core ML/Neural Engine providers
are not the default and remain unqualified. A simulator on the Mac lab can check
execution, numerical contracts and short video handling; its elapsed
times do not predict real iPhone/iPad throughput. Full-video performance, thermal
behavior and physical-device numerical parity remain hardware validation work.

New analyses use the full frame. Existing projects keep their recorded analysis
region. The UI distinguishes source preparation, video scanning, audio features,
selected encoder embeddings, rally inference and score specialists. Live stage
detail remains visible while work is running. Serving-side inference considers
every selected neural rally; score-support legacy heads do not compose or veto
the neural rally list. The optional team-switch model uses those same rallies.

## Projects and model changes

The selected model is captured when work is queued and survives interruption.
Old jobs without that field decode as Legacy. New defaults never rewrite saved
projects. More → Change model or analyze again returns to setup; the new analysis
creates a separate project, preserving previous edits.

Shared `volleycut-model-feedback` v3 exports retain:

- The selected model ID, display label, selection key and source analysis ID.
- Immutable rally ranges, serving-side results, score corrections and base features.
- `initialInference.neuralScores`: timestamps and a row-major float32 matrix with
  four columns in `live`, `serve`, `end`, `keep` order.
- Explicit provenance for auxiliary rally/serve/dead traces. Native legacy heads
  support scoring; they must not be mistaken for the selected neural detector.

Embeddings and source media are excluded. Imported scores are validated for model
identity, head order, shape, finite probabilities, increasing timestamps and the
analyzed window. Edits and reexport preserve the retained traces. Missing neural
scores in an older export do not erase its saved model identity or rally ranges.
Receiving a project still requires reconnecting its original recording.

The historical [project interoperability audit](PROJECT-INTEROPERABILITY.md) and
[Android/iPad diagnostic](PARITY.md) remain frozen evidence. New options do not
establish lossless exchanges for unrelated preferences or verify the old audit's
remaining gaps. Current contract coverage is implemented in `ProcessingJobTests`,
`ProjectContractTests`, `AnalysisProgressTests` and `EditorProjectBindingTests`.
See the [validation report](../docs/research/ios-neural-port-validation.md) for
executed numerical, project-interchange and short-video checks.

## Packaging

Only the two pinned model bundles and required license notices are allowed in
Release. The archive/IPA resource audit checks each graph/config hash and size,
the neural manifest, license payloads and app/SDK privacy declarations. Diagnostic
fixtures and unrecognized model files are rejected. The teacher is DINOv2 ViT-S/14;
teacher weights and its training-only projection are not packaged. License terms
and attribution remain in [third-party notices](../THIRD_PARTY_NOTICES.md).
