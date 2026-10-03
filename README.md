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
linearly between samples instead. Use `--no-interpolate` to hold them. It also
fills in the frames an object was missed on, with every detector and when
detecting on every frame (see `--hold-frames` below):

    python process_video.py input.mp4 --sample-rate 10
    python process_video.py input.mp4 --sample-rate 10 --no-interpolate

- Tracking, counting and the size filter run over all samples first, then the
  video is rendered by interpolating each tracked object (matched by track id)
  between its detections. An object a sample missed is interpolated between the
  samples before and after it, see `--hold-frames` below. An object that is not
  found again stays where it was until the next sample, and a new one appears
  when its sample is reached.
- The HUD and in/out counts show the values as of the previous sample, so they
  stay in sync with the boxes.
- Traces (`--trace-length`) are drawn too, since every frame now has a position.
- Works best at about 5 samples/s or more. At very low rates (the `cloud` and
  `locate-anything` defaults are 2 and 1 per second) fast or non-linear motion
  and wrong track matches show up as boxes sliding across the screen; consider
  `--no-interpolate` there.
- When detecting on every frame there is nothing to smooth, so it only matters
  for `--hold-frames`. Works with every detector and with `--load-detections`.

### Boxes blinking off and on (`--hold-frames`)

Generative detectors (`locate-anything`, `cloud`) have no per-box confidence
and sometimes return only some of the objects for a frame.

`--hold-frames N` keeps drawing an object the detector misses in up to N
detections in a row (frames, or samples when detection is sampled; default 2
for `cloud` and `locate-anything`, off for `rfdetr`; `0` disables). It only
affects what is drawn, not the counts.

- With `--interpolate` (the default) the box of a missed object moves from
  where it was last found to where it is found again: an object found at N,
  missed at N+1 and N+2 and found at N+3 is drawn on the line between its
  positions at N and N+3. If it is missed in more than N samples in a row it is
  taken to be gone.
- With `--no-interpolate` the box stays where it was last found.
- The tracker keeps a track's id across those N missed detections, which also
  means that at low sample rates it remembers lost objects longer (at 1 sample/s
  with N=2, 2 s instead of 1 s) and may re-use the id of an object that is found
  again instead of counting it twice. The shipping container video counts 112
  with `--hold-frames 0` and 109 with the default.
- N counts detections, not time: at 60 samples/s 2 is 1/30 s, at 1 sample/s it
  is 2 s. Raise it if boxes still blink when the detector misses an object for
  longer.

Counts are a separate matter: at one detection per frame (`--sample-rate`
equal to the FPS) a container the model finds only now and then is given new
track ids, which inflates `--count-mode total`. The shipping container video
counts 154 at 60 samples/s against 112 (109 with the default `--hold-frames`) at
1 sample/s. A few samples per second
with interpolation gives much steadier boxes and counts than every frame.

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
(`locate-anything-cli` in PATH, or the shared library, see below). Like the
cloud detector it runs on sampled frames only (`--sample-rate`, default 1/s)
and `--classes` is free-form text. Each class becomes part of the prompt
(`Locate all the instances that matches the following description: apple.`).

    python process_video.py videos/conveyor/upscaled/ttC_MBiNQpWETFACh7yoA_minimax-h3_upscaled.mp4 \
      --detector locate-anything \
      --locate-model /Users/jpb/Downloads/locate-anything-q8_0.gguf \
      --sample-rate 1 --infer-resolution 720 \
      --count-mode total --classes apple \
      --hud-rows "APPLES=apple" --min-size 0 0

`--locate-model` can also come from the `LOCATE_ANYTHING_MODEL` env var (or
`.env`). Extra options: `--locate-mode hybrid|slow|fast`, `--locate-threads N`.
There are no confidence scores (all detections get 1.0, so `--threshold`
has no effect).

### Load the model once (`--locate-lib`)

`locate-anything-cli` is a one-shot tool: it loads the model on every call and
has no batch mode. The engine also ships as a C library, which this project can
drive through a small worker process (`locate_anything_worker.py`) that loads the model
once and then serves every sampled frame. Build it once (add
`-DLA_GGML_CUDA=ON` / `-DLA_GGML_VULKAN=ON` instead of Metal on other
hardware):

    cmake -S path/to/locate-anything.cpp -B path/to/locate-anything.cpp/build-shared \
      -DLA_SHARED=ON -DLA_GGML_METAL=ON -DLA_BUILD_CLI=OFF
    cmake --build path/to/locate-anything.cpp/build-shared -j

and point to it with `--locate-lib .../build-shared/liblocate_anything.dylib`
(or `LOCATE_ANYTHING_LIB` in `.env`). Without it the CLI is used, as before.

- If the GPU backend crashes on a frame (it has been seen to fault with Metal
  page faults), only the worker dies: that sample is skipped, like a failed CLI
  call, and a new worker is started for the next one.
- Don't expect a big speedup from this alone. Measured on an M5 Max (q8_0
  model, 1260x720 frame, interleaved runs) a call takes ~5.3 s with the CLI
  and ~5.1 s with the library: the reload costs about 0.2 s while the model
  file is in the OS cache, because inference dominates. It helps more when the
  model file is not cached (slow disk, memory pressure). Results match: same
  box counts on every sampled frame; coordinates differed by at most 0.72 px
  from the separately built CLI that was installed here.
- What does make it faster is the input size, see `--infer-resolution` below.
  Same frame, resident engine: 720p 4.95 s, 540p 3.33 s (1 of 43 boxes lost),
  360p 2.80 s. `--locate-mode fast` was no quicker than `hybrid` (4.94 s) and
  `slow` was slower (6.62 s). Check small objects on your own footage before
  going low.

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
