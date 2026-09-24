"""Check branch files (optionally historical blobs) without printing matched data.

Use the private source ledger for known identities as well as generic detectors.
This is a publication gate, not a claim to detect every possible credential.
"""
import argparse
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


def git(*args):
    return subprocess.check_output(["git", *args])


def categories(text, identities):
    # Apply the same version exception to individual lines and historical blobs.
    text = "\n".join(line for line in text.splitlines()
                     if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", line.strip()))
    result = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    if any(value in text for value in identities):
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
        for line, text in enumerate(data.decode("utf8", errors="replace").splitlines(), 1):
            kinds = categories(text, identities)
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
                    kinds = sorted(set(categories(data.decode("utf8", errors="replace"), identities)
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
