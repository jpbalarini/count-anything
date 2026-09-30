# How to run

## Line crossing

python process_video.py highway/yHuR-GMbRNKnZLBvDBtHs_minimax-h3_topaz_upscale.mp4 \
  --line 0 700 1892 700 --in-text "EASTBOUND →" --out-text "← WESTBOUND" \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25


## Total count instead of line crossing

python process_video.py shipping_container/7RswKdWQhN167Z_MV8IgP_minimax-h3_upscale.mp4 \
  --count-mode total \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25


Counts every distinct tracked object seen in the video. No line is drawn.

## Cloud model instead of RF-DETR

    # put ANTHROPIC_API_KEY=... in .env (loaded automatically)
    python process_video.py shipping_container/7RswKdWQhN167Z_MV8IgP_minimax-h3_upscale.mp4 \
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
