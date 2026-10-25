Synthetic 320×240 baseline H.264 video, 10 fps, 20 seconds, with solid red, green,
blue and yellow sections of 5 seconds each. A long keyframe interval exercises
paused exact seeking across GOPs. No original recording or private media is used.

Generated with FFmpeg, then Base64 encoded as `rally-seek-fixture.mp4.b64`:

```sh
ffmpeg -f lavfi -i color=c=red:s=320x240:r=10:d=5 \
  -f lavfi -i color=c=lime:s=320x240:r=10:d=5 \
  -f lavfi -i color=c=blue:s=320x240:r=10:d=5 \
  -f lavfi -i color=c=yellow:s=320x240:r=10:d=5 \
  -filter_complex '[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]' \
  -map '[v]' -c:v libx264 -profile:v baseline -bf 0 -pix_fmt yuv420p -g 100 -sc_threshold 0 \
  -crf 18 -movflags +faststart rally-seek-fixture.mp4
```
