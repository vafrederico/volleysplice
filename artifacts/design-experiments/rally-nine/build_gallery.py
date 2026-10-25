"""Package the nine prototypes without restyling their UI (fonttools + brotli)."""
from __future__ import annotations

import json
import re
import base64
import gzip
import io
from html import unescape
from pathlib import Path
from fontTools import subset
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parent
CONDITIONS = (
    ("no-skills", "No design skills", "A"),
    ("frontend-design", "Frontend Design", "B"),
    ("taste", "Taste", "C"),
)


def build() -> None:
    designs = []
    receipts = []
    assets = []
    asset_ids = {}
    font_cache = {}
    characters = set(range(32, 127))
    for condition, _, _ in CONDITIONS:
        for source in (ROOT / condition).glob("variant-*.html"):
            text = source.read_text(encoding="utf-8-sig")
            characters.update(ord(char) for char in unescape(text))
            characters.update(int(code, 16) for code in re.findall(r"\\u([0-9a-fA-F]{4})", text))

    def pack_asset(match: re.Match) -> str:
        uri = match.group(0)
        if uri.startswith("data:font/"):
            if uri not in font_cache:
                font = TTFont(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])), recalcTimestamp=False)
                options = subset.Options()
                options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
                worker = subset.Subsetter(options=options)
                worker.populate(unicodes=characters)
                worker.subset(font)
                font.flavor = "woff2"
                buffer = io.BytesIO()
                font.save(buffer)
                font_cache[uri] = "data:font/woff2;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            uri = font_cache[uri]
        if uri not in asset_ids:
            asset_ids[uri] = len(assets)
            assets.append(uri)
        return f"@@VS_ASSET_{asset_ids[uri]}@@"
    for condition, label, prefix in CONDITIONS:
        directory = ROOT / condition
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
        entries = manifest.get("entries", manifest.get("variants", manifest.get("designs", [])))
        if len(entries) != 3:
            raise ValueError(f"Expected three variants in {condition}")
        for number, entry in enumerate(entries, 1):
            filename = entry.get("file", f"variant-{number}.html")
            source = directory / Path(filename).name
            html = source.read_text(encoding="utf-8-sig")
            if "<html" not in html.lower() or "<script" not in html.lower():
                raise ValueError(f"Missing interactive document: {source.name}")
            html = re.sub(r"data:[\w.+/-]+(?:;charset=[\w-]+)?;base64,[A-Za-z0-9+/=]+", pack_asset, html)
            html = re.sub(r"format\((['\"])(?:truetype|opentype)\1\)", "format('woff2')", html)
            designs.append({
                "id": f"{prefix}{number}", "condition": condition,
                "method": label, "name": entry.get("name", f"Design {number}"),
                "concept": entry.get("concept", ""), "html": html,
            })
        receipts.append({"condition": condition, "manifest": manifest})

    template = (ROOT / "gallery.template.html").read_text(encoding="utf-8")
    raw = json.dumps({"designs": designs, "assets": assets}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
    result = template.replace("__DESIGNS_JSON__", payload)
    result = result.replace("__INFLATE_JS__", (ROOT / "inflate.min.js").read_text(encoding="utf-8"))
    (ROOT / "index.html").write_text(result, encoding="utf-8")
    (ROOT / "comparison.json").write_text(json.dumps({
        "purpose": "Nine Android rally-editor design prototypes; three per skill condition.",
        "generation": {
            "agents": 3, "variantsPerAgent": 3,
            "provider": "codex", "model": "gpt-6.1-sol",
            "options": {"reasoningEffort": "xhigh", "serviceTier": "priority"},
            "parentHistoryCopied": False, "separateWorktrees": True,
            "identicalFixture": "fixture.json", "identicalFrame": "frame.svg",
            "otherConditionsVisibleDuringGeneration": False,
        },
        "conditions": receipts,
        "limits": [
            "Synthetic UI prototypes, not Android APK implementations or real video analysis.",
            "The comparison gives each embedded prototype fresh isolated simulated storage; standalone originals retain their authored behavior.",
            "The comparison losslessly packages font outlines as WOFF2 subsets containing the prototypes' characters; original prototype files are preserved.",
            "Three outputs from one independent agent per condition; this is not a statistical skill evaluation.",
            "Context isolation uses fresh tasks and separate worktrees; skill/file read limits are instructions documented by agent receipts, not filesystem access controls.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Packaged {len(designs)} designs into index.html ({len(result):,} characters)")


if __name__ == "__main__":
    build()
