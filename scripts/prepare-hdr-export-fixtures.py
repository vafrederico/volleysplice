"""Create indexed SDR, HLG and PQ export probes without altering the source.

The HDR files encode the SDR source in BT.2020 with 203-nit diffuse white;
they do not reconstruct highlights absent from the source recording.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from private_ledger import private_value


def inspect(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,pix_fmt,"
        "color_range,color_space,color_transfer,color_primaries,r_frame_rate",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--profile", choices=["sdr", "hlg", "pq"], required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--height", type=int)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Optional compatible FFmpeg executable")
    parser.add_argument("--conversion", choices=["zscale", "libplacebo"], default="zscale")
    parser.add_argument("--encoder", choices=["hevc_nvenc", "libx265"], default="hevc_nvenc")
    parser.add_argument("--name", required=True, help="Non-identifying artifact basename, without extension")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.name):
        parser.error("Use a lower-case, non-identifying artifact basename")
    if args.start < 0 or args.duration is not None and args.duration <= 0:
        parser.error("The sample range must be positive")
    if args.height is not None and (args.height < 2 or args.height % 2):
        parser.error("The output height must be a positive even number")
    source = Path(private_value(args.source_index))
    root = Path(private_value(args.output_index))
    if not source.is_absolute() or not source.is_file() or not root.is_absolute():
        raise ValueError("Ledger entries must resolve to an existing source and absolute output directory")
    source_info = inspect(source)
    video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    if any(video.get(k) != "bt709" for k in ["color_space", "color_transfer", "color_primaries"]):
        raise ValueError("This fixture recipe requires explicitly tagged BT.709 SDR input")
    if video.get("color_range") != "tv":
        raise ValueError("This fixture recipe requires limited-range SDR input")
    root.mkdir(parents=True, exist_ok=True)
    output = root / (args.name + ".mp4")
    partial = root / (args.name + ".partial.mp4")
    receipt = root / (args.name + ".json")
    if output.exists() or partial.exists() or receipt.exists():
        raise FileExistsError("The indexed artifact already exists; choose a new basename")
    ffmpeg = str(Path(args.ffmpeg).resolve()) if Path(args.ffmpeg).is_file() else args.ffmpeg
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-n",
               "-threads", "4", "-filter_threads", "4"]
    if args.profile != "sdr" and args.conversion == "libplacebo":
        command += ["-init_hw_device", "vulkan=exportvk:0", "-filter_hw_device", "exportvk"]
    if args.start:
        command += ["-ss", str(args.start)]
    command += ["-i", str(source)]
    if args.duration:
        command += ["-t", str(args.duration)]
    command += ["-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "-1"]
    resize = f"zscale=w=-2:h={args.height}," if args.height else ""
    transfer = {"sdr": "bt709", "hlg": "arib-std-b67", "pq": "smpte2084"}[args.profile]
    if args.profile == "sdr":
        command += ["-vf", resize + "format=yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
        primaries, matrix = "bt709", "bt709"
    else:
        # agamma=0 selects the luminance-aware HLG OOTF in zimg. npl=203
        # maps normalized SDR white to HDR reference white, rather than peak white.
        filters = (resize + "zscale=transferin=709:primariesin=709:matrixin=709:rangein=limited:"
                   "transfer=linear:agamma=0,format=gbrpf32le,zscale=primaries=2020:agamma=0,"
                   f"zscale=transfer={transfer}:matrix=2020_ncl:range=limited:npl=203:agamma=0,"
                   "format=yuv420p10le")
        if args.conversion == "libplacebo":
            filters = (resize + "hwupload,libplacebo=colorspace=bt2020nc:color_primaries=bt2020:"
                       f"color_trc={transfer}:range=tv:format=yuv420p10le:peak_detect=0:"
                       "inverse_tonemapping=0:tonemapping=clip:contrast_recovery=0:"
                       "gamut_mode=clip:deband=0,hwdownload,format=yuv420p10le")
        command += ["-vf", filters, "-c:v", args.encoder, "-profile:v", "main10"]
        if args.encoder == "hevc_nvenc":
            command += ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0", "-pix_fmt", "p010le"]
        else:
            command += ["-preset", "fast", "-crf", "20", "-x265-params", "pools=4:frame-threads=2", "-pix_fmt", "yuv420p10le"]
        command += ["-tag:v", "hvc1"]
        primaries, matrix = "bt2020", "bt2020nc"
    command += ["-color_primaries", primaries, "-colorspace", matrix, "-color_trc", transfer,
                "-color_range", "tv", "-c:a", "copy", "-movflags", "+faststart", str(partial)]
    print(f"Preparing {args.name} ({args.profile})", flush=True)
    try:
        process = subprocess.run(command)
        if process.returncode:
            raise RuntimeError(f"Fixture encoding failed with exit code {process.returncode}")
        result = inspect(partial)
        output_video = next(s for s in result["streams"] if s["codec_type"] == "video")
        expected = {"color_primaries": primaries, "color_space": matrix, "color_transfer": transfer, "color_range": "tv"}
        if any(output_video.get(k) != v for k, v in expected.items()):
            raise ValueError("Encoded output color properties differ from the fixture contract")
        if args.profile != "sdr" and (output_video.get("codec_name") != "hevc" or "10" not in output_video.get("pix_fmt", "")):
            raise ValueError("HDR fixture must contain 10-bit HEVC")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(output)
    record = {"schemaVersion": 1, "sourceIndex": args.source_index, "artifactIndex": args.output_index,
              "name": args.name, "profile": args.profile, "syntheticHdrFromSdr": args.profile != "sdr",
              "conversion": args.conversion if args.profile != "sdr" else "none",
              "diffuseWhiteNits": 203 if args.profile != "sdr" else None,
              "startSeconds": args.start, "requestedDurationSeconds": args.duration,
              "sha256": sha256(output), "media": result}
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
