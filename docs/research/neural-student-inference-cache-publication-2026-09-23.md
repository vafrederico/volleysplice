# Publication gates for student inference hash reuse

The student inference cache changes repeated file hashing, not numerical replay,
training, selection or accuracy formulas. The registered v3 queue explicitly
classifies every one of the 27 student tasks. Audits completed by the stopped v2
queue retain their original bytes and completion receipts. Every subsequently
created numerical audit requires its registered execution companion.

`audit-neural-student-inference-cache.py` runs independently of the warm cache.
It imports only the previously pinned cold-hashing helper and physically hashes
the union of all touched files, named numerical evidence, cold/warm qualification
proofs, execution companions and source/plan bindings. It verifies the complete
27-task classification, all 42 recordings and four checkpoint replays, and every
companion's exact task, panel, command, output and disclosed hash bindings. File
metadata is checked before and after hashing and again at the end. It never
imports a persisted hash cache or recursively expands arbitrary historical JSON.

Publication requires that cold gate, rehashed small gate/source documents, a
complete evidence inventory, and exact immutable metadata for every cold-verified
file. Missing companions, changed files and omitted evidence prevent publication.

`finalize-neural-generalization-v3.py` wraps the unchanged v2 finalizer. It requires
the student cold gate both before and after the original numerical report audit.
The delegated final audit remains intact, with every field and report-content
digest checked. Report construction calls the original v2 report path into a NAS
intermediate, proves the content digest unchanged, and adds only metadata binding
the new finalization and cold evidence. Production evaluation still uses v2 and
the global freeze of all 162 calibration selections. No evaluation formula,
source split, decoder, recall floor or numerical tolerance changes.
