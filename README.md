# How to run

## Line crossing

Horizontal in/out line
```
python process_video.py videos/highway/yHuR-GMbRNKnZLBvDBtHs_minimax-h3_topaz_upscale.mp4 \
  --line 0 700 1892 700 --in-text "EASTBOUND →" --out-text "← WESTBOUND" \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25
```

Vertical line
```
python process_video.py videos/conveyor/upscaled/ttC_MBiNQpWETFACh7yoA_minimax-h3_upscaled.mp4 \
  --line 940 0 940 1080 --in-text "COUNT" --out-text "" \
  --classes apple \
  --model large \
  --hud-title "CONVEYOR BELT MONITORING" \
  --hud-rows "APPLES=apple" \
  --min-size 0 0 \
  --threshold 0.05
```

## Total count instead of line crossing

python process_video.py videos/shipping_container/7RswKdWQhN167Z_MV8IgP_minimax-h3_upscale.mp4 \
  --count-mode total \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25


Counts every distinct tracked object seen in the video. No line is drawn.

## Detectors

`--detector rfdetr|cloud|locate-anything` (default `rfdetr`). All of them run
the same way: detection first (a pre-pass over the video), then tracking,
counting and drawing. RF-DETR runs on every frame by default; add
`--sample-rate N` to run it only N times per second (the last result is held
 between samples).

### Smooth boxes between samples (`--interpolate`, on by default)

When detection is sampled (fewer detections than video frames), a box that is
simply held until the next sample would jump. By default the boxes move
linearly between samples instead. Use `--no-interpolate` to hold them:

    python process_video.py input.mp4 --sample-rate 10
    python process_video.py input.mp4 --sample-rate 10 --no-interpolate

- Tracking, counting and the size filter run over all samples first, then the
  video is rendered by interpolating between consecutive samples. Objects are
  matched by track id; an object that is gone at the next sample stays where it
  was, and a new one appears when its sample is reached.
- The HUD and in/out counts show the values as of the previous sample, so they
  stay in sync with the boxes.
- Traces (`--trace-length`) are drawn too, since every frame now has a position.
- Works best at about 5 samples/s or more. At very low rates (the `cloud` and
  `locate-anything` defaults are 2 and 1 per second) fast or non-linear motion
  and wrong track matches show up as boxes sliding across the screen; consider
  `--no-interpolate` there.
- No effect when detecting on every frame. Works with every detector and with
  `--load-detections`.

## Cloud model instead of RF-DETR

    # put ANTHROPIC_API_KEY=... in .env (loaded automatically)
    python process_video.py videos/shipping_container/7RswKdWQhN167Z_MV8IgP_minimax-h3_upscale.mp4 \
      --detector cloud \
      --cloud-model claude-sonnet-5-5 --sample-rate 4 \
      --count-mode total \
      --classes "shipping container" \
      --hud-title "TRAFFIC MONITORING" \
      --hud-rows "SHIPPING_CONTAINERS=shipping container" \
      --min-size 0 0 \
      --cloud-concurrency 10 \
      --no-labels \
      --min-track-frames 3

With `--detector cloud`, `--classes` is free-form text and is not limited to
the COCO classes (quote multi-word names, or separate with commas):

    python process_video.py input.mp4 --detector cloud --count-mode total \
      --classes "traffic cone" "person wearing a helmet" \
      --hud-rows "CONES=traffic cone" "HELMETS=person wearing a helmet"

`--hud-rows` and `--colors` refer to the same names you passed in `--classes`.

Cloud calls run in parallel: all sampled frames are sent first, at most
`--cloud-concurrency N` at a time (default 8, `1` = sequential), then the
video is rendered with tracking/counting replayed in frame order. Lower N if
you hit API rate limits (429s are retried automatically).

`--sample-rate` is model calls per second of video (a 10 s clip at 2/s makes
20 calls, whatever the FPS). Between samples the last result stays on screen.
Works with both `--count-mode crossing` and `--count-mode total`.

## locate-anything.cpp instead of RF-DETR

Uses [locate-anything.cpp](https://github.com/mudler/locate-anything.cpp)
(`locate-anything-cli` must be in PATH). Like the cloud detector it runs on
sampled frames only (`--sample-rate`, default 1/s) and `--classes` is
free-form text. Each class becomes part of the prompt
(`Locate all the instances that matches the following description: apple.`).

    python process_video.py videos/conveyor/upscaled/ttC_MBiNQpWETFACh7yoA_minimax-h3_upscaled.mp4 \
      --detector locate-anything \
      --locate-model /Users/jpb/Downloads/locate-anything-q8_0.gguf \
      --sample-rate 1 --infer-resolution 720 \
      --count-mode total --classes apple \
      --hud-rows "APPLES=apple" --min-size 0 0

`--locate-model` can also come from the `LOCATE_ANYTHING_MODEL` env var (or
`.env`). Extra options: `--locate-mode hybrid|slow|fast`, `--locate-threads N`.
The CLI gives no confidence scores (all detections get 1.0, so `--threshold`
has no effect) and reloads the model on every call.

## Faster inference on big videos

`--infer-resolution 720` runs detection on a copy of each frame whose
shorter side is 720 px (a 1080p video becomes 1280x720; smaller videos are
never upscaled). Boxes are mapped back, so the output video keeps its
original resolution and the counting line / `--min-size` still use original
pixels. Works with every detector. Very small objects may be missed at lower
resolutions. RF-DETR already resizes to its own input size internally, so the
gain there is mainly less preprocessing; it matters most for `locate-anything`.

## Change the look without re-running detection

Detection is the slow part. Save the raw detections once, then re-render from
the JSON as many times as you want:

    # 1) detect + render, and keep the detections
    python process_video.py input.mp4 --infer-resolution 720 \
      --save-detections dets.json

    # 2) no detector runs, only tracking + counting + drawing
    python process_video.py input.mp4 --load-detections dets.json \
      --hud-title "NEW TITLE" --colors "car=#FF0000" --no-labels

With `--load-detections` you can change anything that happens after
detection: colors, HUD, labels, counting line, `--count-mode`, size filter,
tracker settings, `--classes` (any class that was detected), and raise
`--threshold`. The detector is taken from the file (`--detector`,
`--sample-rate` and `--infer-resolution` are ignored), so cloud / locate-anything
files need no API key or model. Use the same source video. Detections are
saved before tracking, so the file holds every class above the threshold used
when it was made (lower `--threshold` when saving if you might want to raise it
later).
