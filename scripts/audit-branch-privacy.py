"""Check branch files (optionally historical blobs) without printing matched data.

Use the private source ledger for known identities as well as generic detectors.
This is a publication gate, not a claim to detect every possible credential.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


PATTERNS = {
    "camera-filename": re.compile(r"\b(?:PXL|IMG|VID)[_ -]\d{6,}", re.I),
    "private-mount": re.compile(r"/" r"mnt/(?![a-z]/(?:Windows|Program Files)/)[^/\s]+/", re.I),
    "user-home": re.compile(r"(?:/" r"home/[^/\s]+/|[a-z]:[\\/]+Users[\\/]+[^\\/\s]+[\\/])", re.I),
    "private-drive": re.compile(r"\b[z]:[\\/]", re.I),
    "lan-address": re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"),
    "credential": re.compile(r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bAKIA[A-Z0-9]{16}\b)"),
}

# These two established public app links lead to LICENSE and THIRD_PARTY_NOTICES.
# Only exact URL tokens in this source file may bypass ledger-identity matching;
# generic privacy detectors still inspect the original text. Hashes keep the
# public repository owner out of the audit's own source and do not exempt that
# identity elsewhere, arbitrary GitHub URLs, filenames, or commit messages.
PUBLIC_LEGAL_URL_SHA256 = {
    "android/app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt": {
        "77657dab312c536d06976783e15351ce6fec445651a7773fddc727acfb024425",
        "63d4d0eb3326310a17c5d02bb51151ede47a3956dcd90e04c3e26d77d1b4a728",
    },
}
HTTPS_TOKEN = re.compile(r"https://[^\s\"'<>]+")

# Only the four public, manifest-pinned graphs use protobuf-aware scanning.
# Random float bytes can resemble a short drive prefix. All textual fields,
# unknown fields, and external-data references remain in the scanned material.
PUBLIC_ONNX_PATH = re.compile(
    r"prod/public/runtime/rally-models/(?:f1|recall)/(?:encoder|temporal)\.onnx"
)
ONNX_CHILDREN = {
    "model": {7: "graph", 8: "opset", 14: "entry", 20: "training", 25: "function"},
    "graph": {1: "node", 5: "tensor", 11: "value", 12: "value", 13: "value",
              15: "sparse", 16: "entry"},
    "node": {5: "attribute", 9: "entry"},
    "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph",
                  22: "sparse", 23: "sparse"},
    "tensor": {13: "entry", 16: "entry"},
    "sparse": {1: "tensor", 2: "tensor"},
    "training": {1: "graph", 2: "graph", 3: "entry", 4: "entry"},
    "function": {7: "node", 9: "opset", 11: "attribute", 12: "value", 14: "entry"},
    "value": {4: "entry"},
}
# Byte widths of established numeric ONNX tensor types. STRING is deliberately
# absent. Unknown/new types receive the ordinary byte scan, not an exemption.
ONNX_NUMERIC_WIDTHS = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8,
                       9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 14: 8, 15: 16, 16: 2}


def protobuf_varint(data, start, end):
    value = 0
    for shift in range(0, 70, 7):
        if start >= end:
            raise ValueError("Truncated protobuf varint")
        byte = data[start]
        start += 1
        if shift == 63 and byte > 1:
            raise ValueError("Oversized protobuf varint")
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, start
    raise ValueError("Oversized protobuf varint")


def protobuf_fields(data, start, end):
    fields = []
    while start < end:
        tag, start = protobuf_varint(data, start, end)
        number, wire = tag >> 3, tag & 7
        if not 0 < number < (1 << 29):
            raise ValueError("Invalid protobuf field")
        if wire == 0:
            value, stop = protobuf_varint(data, start, end)
        elif wire == 2:
            size, start = protobuf_varint(data, start, end)
            value, stop = None, start + size
        elif wire in (1, 5):
            value, stop = None, start + (8 if wire == 1 else 4)
        else:
            raise ValueError("Unsupported protobuf wire type")
        if stop > end:
            raise ValueError("Truncated protobuf field")
        fields.append((number, wire, start, stop, value))
        start = stop
    return fields


def public_onnx_text(data):
    """Retain all bytes except validated numeric tensor payloads.

    This is a privacy parser, not an ONNX execution/shape validator. Preparation
    and build checks independently verify the pinned graph hashes. Protobuf
    framing, the model envelope, and excluded tensor lengths fail closed here.
    """
    spans, leaves = [], []

    def visit(kind, start, end, depth=0):
        if depth > 64:
            raise ValueError("Excessive ONNX nesting")
        fields = protobuf_fields(data, start, end)
        if kind == "model":
            if (sum(n == 7 and w == 2 for n, w, *_ in fields) != 1
                    or not any(n == 1 and w == 0 and v > 0 for n, w, _, _, v in fields)
                    or not any(n == 8 and w == 2 for n, w, *_ in fields)):
                raise ValueError("Missing ONNX model envelope")
        masked = set()
        if kind == "tensor":
            types = [v for n, w, _, _, v in fields if n == 2 and w == 0]
            raw = [(a, b) for n, w, a, b, _ in fields if n == 9 and w == 2]
            if raw and (len(raw) != 1 or len(types) != 1):
                raise ValueError("Ambiguous ONNX tensor storage")
            if raw and types[0] in ONNX_NUMERIC_WIDTHS:
                dimensions = []
                for n, w, a, b, v in fields:
                    if n != 1:
                        continue
                    if w == 0:
                        dimensions.append(v)
                    elif w == 2:
                        while a < b:
                            v, a = protobuf_varint(data, a, b)
                            dimensions.append(v)
                    else:
                        raise ValueError("Invalid ONNX tensor dimensions")
                count = 1
                for dimension in dimensions:
                    if dimension >= (1 << 63):
                        raise ValueError("Negative ONNX tensor dimension")
                    count *= dimension
                a, b = raw[0]
                if count * ONNX_NUMERIC_WIDTHS[types[0]] != b - a:
                    raise ValueError("ONNX tensor payload length mismatch")
                if any(n in (6, 13) or (n == 14 and v != 0) for n, _, _, _, v in fields):
                    raise ValueError("Conflicting ONNX tensor storage")
                spans.append((a, b))
                masked.add((a, b))
        for number, wire, a, b, _ in fields:
            child = ONNX_CHILDREN.get(kind, {}).get(number)
            if child:
                if wire != 2:
                    raise ValueError("Invalid ONNX nested message")
                visit(child, a, b, depth + 1)
            elif wire == 2 and (a, b) not in masked:
                # Also scan leaves independently: protobuf length/tag bytes must
                # not hide a text prefix by changing regex word boundaries.
                leaves.append(data[a:b])

    visit("model", 0, len(data))
    pieces, previous = [], 0
    for start, end in sorted(spans):
        if start < previous:
            raise ValueError("Overlapping ONNX tensor storage")
        pieces.extend((data[previous:start], b"\n"))
        previous = end
    pieces.append(data[previous:])
    return b"\n".join(pieces + leaves).decode("utf8", errors="replace")


def audit_text(data, source_path):
    if PUBLIC_ONNX_PATH.fullmatch(source_path or ""):
        try:
            return public_onnx_text(data), []
        except (ValueError, RecursionError):
            return data.decode("utf8", errors="replace"), ["invalid-public-onnx"]
    return data.decode("utf8", errors="replace"), []


def git(*args):
    return subprocess.check_output(["git", *args])


def categories(text, identities, source_path=None):
    # Apply the same version exception to individual lines and historical blobs.
    text = "\n".join(line for line in text.splitlines()
                     if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", line.strip()))
    result = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    approved = PUBLIC_LEGAL_URL_SHA256.get(source_path, set())
    ledger_text = HTTPS_TOKEN.sub(
        lambda match: "" if hashlib.sha256(match.group().encode("utf8")).hexdigest() in approved
        else match.group(), text,
    ) if approved else text
    if any(value in ledger_text for value in identities):
        result.append("ledger-identity")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--revision", help="Inspect a committed snapshot instead of current working files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf8"))
    identities = {key for key in ledger.get("originalToAlias", {}) if len(key) > 8}
    identities.update(ledger.get("denyTokens", []))
    # Private values include path prefixes and complete private URLs, but can also
    # include generic synthetic test paths. The generic rules cover both.
    identities.update(v for v in ledger.get("privateValues", {}).values()
                      if len(v) > 12 and ("/" in v or "\\" in v))
    revision = args.revision or "HEAD"
    paths = set(git("diff", "--name-only", f"{args.base}...{revision}").decode().splitlines())
    if not args.revision:
        paths.update(git("ls-files", "--modified", "--others", "--exclude-standard").decode().splitlines())
    findings = []
    inspected = 0
    for path in sorted(paths):
        if args.revision:
            result = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True)
            if result.returncode:
                continue  # Deleted path.
            data = result.stdout
        else:
            source = Path(path)
            if not source.is_file():
                continue
            data = source.read_bytes()
        inspected += 1
        scan_text, parse_findings = audit_text(data, path)
        if parse_findings:
            findings.append({"file": path, "line": 0, "categories": parse_findings})
        for line, text in enumerate(scan_text.splitlines(), 1):
            kinds = categories(text, identities, path)
            if kinds:
                findings.append({"file": path if not categories(path, identities) else "private-filename",
                                 "line": line, "categories": kinds})
        if categories(path, identities):
            findings.append({"file": "private-filename", "line": 0, "categories": ["filename"]})
    historical = []
    if args.history:
        objects = git("rev-list", "--objects", f"{args.base}..{revision}").decode().splitlines()
        process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for entry in objects:
                oid = entry.split(" ", 1)[0]
                object_path = entry.split(" ", 1)[1] if " " in entry else ""
                process.stdin.write((oid + "\n").encode()); process.stdin.flush()
                header = process.stdout.readline().decode().split()
                data = process.stdout.read(int(header[2])); process.stdout.read(1)
                # Commit metadata includes author identities; count separately.
                if header[1] == "blob":
                    scan_text, parse_findings = audit_text(data, object_path)
                    kinds = sorted(set(parse_findings + categories(scan_text, identities, object_path)
                                       + categories(object_path, identities)))
                    if kinds: historical.append({"object": oid, "categories": kinds})
                elif header[1] == "commit":
                    message = data.decode("utf8", errors="replace").split("\n\n", 1)[-1]
                    kinds = categories(message, identities)
                    if kinds: historical.append({"object": oid, "categories": kinds, "kind": "commit-message"})
        finally:
            process.stdin.close(); process.wait()
    result = {"baseCommit": git("rev-parse", args.base).decode().strip(),
              "headCommit": git("rev-parse", revision).decode().strip(),
              "scope": "committed-snapshot" if args.revision else "working-files",
              "filesInspected": inspected, "workingTreeFindings": findings,
              "historicalObjectsWithFindings": historical,
              "historyChecked": args.history,
              "commitAuthorMetadata": "Not rewritten; publication requires a separate history decision.",
              "status": "fail" if findings or historical else "pass"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"filesInspected": inspected, "workingTreeFindings": len(findings),
                      "historicalObjectsWithFindings": len(historical), "status": result["status"]}))
    raise SystemExit(1 if findings or historical else 0)


if __name__ == "__main__":
    main()
