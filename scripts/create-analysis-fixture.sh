#!/usr/bin/env bash
set -euo pipefail

fixture_path="${1:-data/videos/synthetic-two-bursts.mp4}"
mkdir -p "$(dirname "$fixture_path")"

ffmpeg -nostdin -hide_banner -loglevel error -y \
  -f lavfi -i 'color=c=0x24535b:s=640x360:d=5:r=30' \
  -f lavfi -i 'testsrc2=s=640x360:d=7:r=30' \
  -f lavfi -i 'color=c=0x24535b:s=640x360:d=4:r=30' \
  -f lavfi -i 'testsrc2=s=640x360:d=5:r=30' \
  -f lavfi -i 'color=c=0x24535b:s=640x360:d=3:r=30' \
  -f lavfi -i 'sine=frequency=440:sample_rate=48000:duration=24' \
  -filter_complex '[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1:a=0,drawgrid=w=160:h=90:t=2:c=white@0.65[v]' \
  -map '[v]' -map 5:a \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  "$fixture_path"

printf 'Synthetic fixture ready: %s\n' "$fixture_path"
