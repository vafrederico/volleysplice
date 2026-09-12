#!/usr/bin/env python3
"""Verify chapter embedding preserved encoded A/V, timestamps, and offsets.

Usage: python verify-chapter-container.py before.mp4 after.mp4 --report report.json
Requires ffprobe on PATH. Does not decode audio or video.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_packets", "-show_data_hash", "sha256",
        "-show_entries", "packet=codec_type,pts,dts,duration,size,pos,data_hash:chapter:"
        "format=duration:format_tags=major_brand,compatible_brands", "-of", "json", str(path)
    ]))


def verify(before, after):
    original, embedded = probe(before), probe(after)
    keep = lambda p: p.get("codec_type") in ("audio", "video")
    a = list(filter(keep, original["packets"]))
    b = list(filter(keep, embedded["packets"]))
    if not a or a != b:
        raise ValueError("Encoded samples, timestamps or file offsets changed")
    if not embedded.get("chapters"):
        raise ValueError("No embedded chapters recognized by ffprobe")
    if original["format"] != embedded["format"]:
        raise ValueError("Duration or MP4 file-type brands changed")
    # Check the pre-existing container byte for byte in bounded chunks.
    changes, old, new = [], bytearray(), bytearray()
    with before.open("rb") as source, after.open("rb") as destination:
        offset = 0
        while block := source.read(1024 * 1024):
            compared = destination.read(len(block))
            if len(compared) != len(block):
                raise ValueError("Chapter output is shorter than input")
            if block != compared:
                for index, (x, y) in enumerate(zip(block, compared)):
                    if x != y:
                        changes.append(offset + index); old.append(x); new.append(y)
                        if len(changes) > 4:
                            raise ValueError("Writer modified pre-existing bytes beyond the moov type")
            offset += len(block)
    if len(changes) != 4 or old != b"moov" or new != b"free" or changes != list(range(changes[0], changes[0] + 4)):
        raise ValueError("Original container mutation was not exactly moov -> free")
    return {
        "source": str(before), "output": str(after), "encodedPacketsEqual": True,
        "packetCounts": {kind: sum(p["codec_type"] == kind for p in a) for kind in ("video", "audio")},
        "packetMetadataAndDataHashSHA256": hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest(),
        "changedOriginalByteOffsets": changes, "chapters": embedded["chapters"],
        "format": embedded["format"], "noMediaDecode": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.before, args.after)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
