"""VolleyCut's local rally-analysis baseline."""

from .version import __version__

ANALYSIS_SCHEMA_VERSION = 1
HEURISTIC_METHOD = "court-motion-audio-heuristic-v2"

# Compatibility aliases for the no-model dataset scripts merged from main.
SCHEMA_VERSION = ANALYSIS_SCHEMA_VERSION
METHOD = HEURISTIC_METHOD

__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "HEURISTIC_METHOD",
    "METHOD",
    "SCHEMA_VERSION",
    "__version__",
]
