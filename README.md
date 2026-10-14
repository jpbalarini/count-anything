# How to run

## Showcase mode
```
python process_video.py conveyor.mp4 \
    --decisions decisions.example.yaml \
    --line 940 0 940 1080 \
    --classes apple \
    --hud-title "CONVEYOR BELT MONITORING" \
    --min-size 0 0 \
    --mm-per-px 0.5 \
    --size-grades S=95 M=105 L=115 XL
```

## Classic mode line crossing

Horizontal in/out line
```
python process_video.py videos/highway/yHuR-GMbRNKnZLBvDBtHs_minimax-h3_topaz_upscale.mp4 \
  --line 0 700 1892 700 --in-text "EASTBOUND →" --out-text "← WESTBOUND" \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25 \
  --style classic
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
  --threshold 0.05 \
  --style classic
```

## Classic mode total count instead of line crossing

python process_video.py videos/shipping_container/7RswKdWQhN167Z_MV8IgP_minimax-h3_upscale.mp4 \
  --count-mode total \
  --classes car truck \
  --hud-title "TRAFFIC MONITORING" \
  --hud-rows "VEHICLES=all" "CARS=car" "TRUCKS=truck" --min-size 35 25 \
  --style classic


Counts every distinct tracked object seen in the video. No line is drawn.

## Detectors

`--detector rfdetr|cloud|locate-anything` (default `rfdetr`), or
`--weights models/NAME` for an RF-DETR fine-tuned on your own detections (see
[Fine-tune RF-DETR on your detections](#fine-tune-rf-detr-on-your-detections)). All of them run
the same way: detection first (a pre-pass over the video), then tracking,
counting and drawing. `--classes` says what to detect and is required (there
is no default), except with `--weights` (the model's classes) and when
loading a cloud / locate-anything / `--weights` file. RF-DETR runs on every frame by default; add
`--sample-rate N` to run it only N times per second (the last result is held
 between samples).

### Smooth boxes between samples (`--interpolate`, on by default)

When detection is sampled (fewer detections than video frames), a box that is
simply held until the next sample would jump. By default the boxes move
linearly between samples instead. Use `--no-interpolate` to hold them. It also
fills in the frames an object was missed on, with every detector and when
detecting on every frame (see `--hold-frames` below):

    python process_video.py input.mp4 --classes car --sample-rate 10
    python process_video.py input.mp4 --classes car --sample-rate 10 --no-interpolate

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
and `--classes` is free-form text. Each class is asked for separately
(`Locate all the instances that matches the following description: apple.`),
so a frame takes one call per class. The model also accepts several classes
in one prompt (`apple</c>box`), but then labels every box with the first
class (the crates came back as "apple") and repeats boxes until it runs out
of tokens.

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
the JSON as many times as you want (add `--no-render` to only detect and
save, without writing a video):

    # 1) detect + render, and keep the detections
    python process_video.py input.mp4 --classes car --infer-resolution 720 \
      --save-detections dets.json

    # 2) no detector runs, only tracking + counting + drawing
    python process_video.py input.mp4 --load-detections dets.json \
      --classes car --hud-title "NEW TITLE" --colors "car=#FF0000" --no-labels

With `--load-detections` you can change anything that happens after
detection: colors, HUD, labels, counting line, `--count-mode`, size filter,
tracker settings, `--classes` (any class that was detected; required for
`rfdetr` files, which hold every COCO class found; for cloud,
locate-anything and `--weights` files the default is every class in the
file), and raise `--threshold`. The detector is taken from the file (`--detector`,
`--sample-rate` and `--infer-resolution` are ignored), so cloud / locate-anything
files need no API key or model. Use the same source video. Detections are
saved before tracking, so the file holds every class above the threshold used
when it was made (lower `--threshold` when saving if you might want to raise it
later).

### Detections file format

A COCO-style JSON where the video is a sequence of images: each annotation
belongs to an image (a frame), each image to the video. The images are
virtual, nothing is written to disk for them.

```json
{
  "info": {"description": "rfdetr detections of apple, orange in input.mp4",
           "url": "", "version": "1.0", "year": 2026, "contributor": "",
           "date_created": "2026-10-04T17:34:25-03:00",
           "format": "coco-video", "format_version": 2,
           "detector": "rfdetr", "sample_step": 6.0, "infer_resolution": null,
           "threshold": 0.2, "model": "medium"},
  "categories": [{"id": 53, "name": "apple"}, {"id": 55, "name": "orange"}],
  "videos": [{"id": 1, "file_name": "input.mp4", "fps": 24.0,
              "width": 1344, "height": 768, "total_frames": 243}],
  "images": [{"id": 1, "video_id": 1, "frame_id": 0,
              "file_name": "input/000000.jpg", "width": 1344, "height": 768}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 53,
                   "bbox": [75.41, 265.22, 91.66, 133.9], "area": 12273.27,
                   "iscrowd": 0, "segmentation": [], "score": 0.6341}]
}
```

- `bbox` is COCO's `[x, y, width, height]` in video pixels; `score` is the
  detector confidence (1.0 for detectors without one, and for boxes drawn by
  hand).
- `info` holds the detection settings; the video's properties are only in
  `videos`. Set `description` / `contributor` / `url` with the flags of the
  same name when saving (the description is generated otherwise).
- One image per frame the detector ran on. An image without annotations is a
  frame where nothing was found; `"detection_failed": true` marks a frame whose
  detection call failed, `"edited": true` one changed in the annotator.
- Category ids are COCO's for `rfdetr` files (1–90, with gaps) and 1..N for
  free-form classes. Loading matches categories by name.
- No `track_id`: detections are saved before tracking, which runs when
  rendering.
- The file opens in standard COCO tools (e.g. `pycocotools`).

## Showcase overlay: measurements, throughput, classification

`--style showcase` (the default) draws corner brackets with an ID chip on
every object, a dashed count line that pulses on each count, and one side
panel with:

- the total count and how many were counted in the last `--throughput-window`
  seconds (default 10);
- next to it, the last item counted: a thumbnail of its best view, with its id,
  class from `--decisions` and size. It changes at most every half second, so
  it can be read on a busy line;
- a row of stats: throughput per hour, over that same window and
  recalculated every `--throughput-update` seconds (default 2), and the
  average size (with `--mm-per-px`);
- with `--decisions`, a donut of the classes from the YAML (e.g. ripe /
  unripe / defect) with the share of "good" ones in the middle;
- with `--size-grades`, a bar per size grade.

`--style classic` gives the original boxes, labels and corner counter.

    python process_video.py conveyor.mp4 \
      --decisions decisions.example.yaml \
      --line 940 0 940 1080 \
      --classes apple \
      --hud-title "CONVEYOR BELT MONITORING" \
      --min-size 0 0 \
      --mm-per-px 0.5 \
      --size-grades S=95 M=105 L=115 XL \
      --load-detections dets.json \
      --load-decisions decisions.json \

Each object's **best view** is the box, among those fully inside the picture,
that is closest to its usual width and height. That skips boxes the detector
merged with a neighbour or cut short. The best view is what goes to the
Decisions API and what the thumbnail shows.

**Size.** For now the size is the **diagonal of the box**. Each object gets one
size: the median over the frames where its box is fully inside the picture
(boxes cut off at the edge are skipped). `--mm-per-px` turns pixels into mm.
Measure something of known size in the video: a 600 mm crate that is 540 px
wide gives `--mm-per-px 1.11`. The grades are in mm. Each `NAME=MAX` grade
holds sizes below MAX, and the last grade takes no MAX, so
`S=95 M=105 L=115 XL` means S < 95 ≤ M < 105 ≤ L < 115 ≤ XL. The grades apply
to the diagonal, so a box diagonal is about 1.41 × the fruit's diameter.

**Classification (OpenAI Decisions API).** This is optional and lives in
`decisions.py`. With `--decisions config.yaml`, every tracked object is
cropped once, from its largest fully visible box. The crop is sent to
`client.decisions.create` with one choice question, and the brackets and chips
are colored by the answer. Everything video-specific goes in the YAML, so the
same code works for other produce or defects:

- the context text and the question;
- the choices, with a description and a color for each;
- which choices count as `good` (the panel's headline %);
- which choices are `flag`ged (always written on the chip);
- `min_confidence`, below which an object is shown as "unverified";
- crop padding and size, and concurrency.

See `decisions.example.yaml`. It needs `OPENAI_API_KEY` in the environment or
in `.env`. The calls run after tracking and before rendering.

Every answer is saved next to the output video as `<output>_decisions.json`
(or wherever `--save-decisions FILE` says). Each saved answer has the track
id, the frame and box the crop came from, the choice, confidence,
probabilities and status (`ok`, `low_confidence`, `refused`, `failed`). The
file also stores the config the answers were made with. To re-render without
calling the API again, pass it back:

    python process_video.py ... --decisions decisions.example.yaml \
      --load-decisions ttC_showcase_decisions.json

Answers are matched by crop (same frame, box overlap ≥ 90%), so changing
the look, the line or the size options still reuses them. Only objects
without a matching answer (and failed calls) go to the API, and the key is
only needed then. If the model, question, choices or crop settings in the
YAML changed, the file is ignored. `min_confidence` is applied again on
load, so you can change it without new calls.

Other options:

- `--hud-position left|right` puts the panel on that side.
- `--hud-font Regular.ttf,Medium.ttf,Bold.ttf` sets the font. The default is
  Manrope if it is installed, else Helvetica Neue on macOS, else DejaVu Sans.
- `--no-labels` hides the chips.
- `--no-hud` hides the panel.
- `--trace-length N` turns traces back on.

## Projects: upload, label, train (AI Studio)

    python annotate_server.py            # http://127.0.0.1:8000

The home page is a workspace of **folders** and **projects** (like
Roboflow's). A folder holds folders and projects; drag a card onto a folder
(or onto the breadcrumb above) to move it, or use its ⋮ menu (rename, move,
delete). The search box searches every folder. A project is a dataset of
images and the models trained on it:

- **Upload**: drop pictures, videos or whole folders. Pictures go in as one
  batch; each video is its own batch, cut into frames at the rate you pick
  (1 frame every 2 s up to every frame; the page shows how many images that
  makes). Frames are extracted in the background. Everything arrives
  **unlabeled**.
- **Annotate**: the batches, with how many images are labeled,
  **pre-labeled** (boxes from a model, not checked yet) and unlabeled. A
  batch opens in the same annotator as detections files
  (`/annotate?project=ID&batch=ID`, see below), starting at the first image
  still to label. Saving any change to an image marks it labeled; `M` (or
  "Mark as labeled & next") accepts an image as it is, which is how
  pre-labels are reviewed. The batch's ⋮ menu can accept all its pre-labels
  at once.
- **Auto label** (on a batch): runs one of Label Assist's detectors
  (locate-anything, Claude, the stock RF-DETR, or a trained model) on every
  unlabeled image of the batch, or also on the pre-labeled ones, in the
  background. What it finds, above a minimum confidence (default: the
  detector's own threshold), is saved as pre-labels to review. Images you
  labeled are never changed, even ones labeled while it runs. The batch
  shows its progress and a **Stop** button (it stops after the current
  image; what was found so far is kept).
- **Tags**: free-text notes on an image ("one hand", "blurry", "odd
  angle"), added in the annotator's Frame panel (`G` to type one, Enter to
  add). Spelling differences in case or `_`/`-` count as the same tag.
  Tagging doesn't change whether an image is labeled, nor what is trained
  on. The annotator's images filter lists them too.
- **Dataset**: every image with its boxes and tags, filtered by status,
  batch, class, tag (or "No tags") or file name. Click an image to open it
  in the annotator, a tag to show the images that have it, ⌘-click (or the
  corner check) to select several and mark them labeled / unlabeled, add
  or remove a tag, or delete them.
- **Classes**: box counts per class; rename a class (into an existing one to
  merge them) or delete it with its boxes.
- **Train**: fine-tunes RF-DETR on the project (`train_rfdetr.py --project`),
  on its labeled images only, or also on the pre-labeled ones (their boxes of
  at least a minimum confidence). Unlabeled images are never used.
- **Models**: the project's models with their scores (mAP per epoch, F1 at
  the chosen threshold), and the training in progress. Label Assist's
  **Trained** tab offers them first, to pre-label the next images.
- **Detections**: run one of the project's models on a video (any video
  under `--root`, or one you upload) to get a detections file
  (`process_video.py --weights ... --no-render`, saved next to the video).
  The project's detections files are listed (made with its models, or
  imported into it): **Open** reviews one in the annotator (below), and
  **Render command** copies the `process_video.py --load-detections`
  command that renders the video from it.

Everything is plain files in the **projects location**, a folder outside
the code: `~/.count-anything` until you pick another one with the folder
button on the home page (saved in `~/.config/ai-studio/config.json`). Picking
one either moves your folders and projects there or only switches to it
(showing what is already there). `--workspace` sets it for one run instead.
Projects made before this lived in `<root>/workspace`; that folder keeps
being used, with a notice on the home page, until you pick a location.

A folder is a directory with `folder.json`; a project is a directory with
`project.json`, `annotations.json` (COCO, plus each image's `batch_id` and
`status`), `images/`, `thumbs/` and `models/`.

Train from the command line on a project:

    python train_rfdetr.py --project ~/.count-anything/highway --only-edited

## Fix the detections by hand (annotation UI)

A web UI (FastAPI) to review and correct a detections file before
re-rendering: move, resize, relabel, add and delete boxes, frame by frame.

    python annotate_server.py                      # choose the files in the UI
    python annotate_server.py dets.json            # video found from the file
    python annotate_server.py input.mp4 dets.json
    python process_video.py input.mp4 --load-detections dets.json ...

A project's **Detections** page opens its files here; any other file opens
at http://127.0.0.1:8000/annotate. Its file picker (folder button, or `⌘O`) lists the detections files under
`--root` (default: the current directory; the workspace isn't searched),
newest first, each matched to the video named in its `videos[0].file_name`.
Pick one, or type the paths (relative to `--root` or absolute) when the
video isn't found. Each tab edits the files it opened; when many are opened
the oldest tab's files are closed, and that tab stops saving instead of
writing into the wrong file.

- Every change is saved to `dets.json` right away (the first change keeps the
  untouched file as `dets.orig.json`). Re-render with `--load-detections`.
- Only the frames the detector ran on (the ones in the file) are shown.
- Box tool (`B`): drag anywhere to draw, even over other boxes; click to
  select, drag the selected box to move it, drag a handle to resize. Clicking
  the selected box again cycles through the boxes under the cursor.
- `R` (or "Fill in from previous frame") adds the previous frame's boxes that
  have no overlapping box of the same class here: the objects the detector
  missed on this frame. `O` shows the previous frame's boxes as dashed ghosts.
- The timeline under the image shows the box count per frame: dips are frames
  where the detector likely missed objects. Click or drag it to jump. The
  frames panel can be filtered to edited / not edited / empty / failed frames.
- `⌘C` / `⌘V` copy boxes across frames, `1`–`9` set the class, `⌘Z` undoes
  (across frames). Press `?` for every shortcut.
- With an `rfdetr` file the class editor suggests COCO names (the labels must
  be COCO classes to load back), and a slider hides / deletes low-confidence
  boxes.
- Classes added in the UI become categories (new COCO classes get their COCO
  id in `rfdetr` files), and changed frames get `"edited": true` on their
  image.

Options: `--root`, `--workspace`, `--port`, `--host`, `--no-browser`.

### Label Assist: find objects with a model

The wand button in the toolbar (or `I`) runs a detector on the frame you are
looking at and shows what it found as dashed proposals to review:

- **Cloud**: Claude Sonnet 5.5 (default) or Opus 5.5, like
  `--detector cloud` (needs `ANTHROPIC_API_KEY` in `.env`). Any class name,
  each with an optional description ("only the ripe ones") added to the
  prompt.
- **Local**: locate-anything (any class name; uses `LOCATE_ANYTHING_MODEL` /
  `LOCATE_ANYTHING_LIB` from `.env`, or the model recorded in a
  locate-anything file), or the stock RF-DETR (COCO classes only).
- **Trained**: an RF-DETR fine-tuned here (see below), with its own classes.

Pick the classes to find (the file's classes are chosen by default; add
others in the panel), then **Find objects** (`Enter`). Then:

- Click a proposal to leave it out (click again to keep it).
- For scored models (RF-DETR, cloud), the confidence slider hides weak
  proposals. It starts at the model's threshold, and moving it doesn't
  detect again.
- **Add the new ones** skips proposals that overlap a box of the same class
  already on the frame. **Replace** removes this frame's boxes of the
  chosen classes and adds the proposals instead.
- **Save** applies it as one change (`⌘Z` undoes it). Changing frames
  discards unsaved proposals; the panel stays open, so you can go to the
  next frame and press `Enter` again.

Drag the panel by its title to see what's under it (double-click the title
to put it back). Local models stay loaded in the annotator between runs, so only the first
run pays for loading them. Nothing runs or is saved until you press the
buttons.

## Fine-tune RF-DETR on your detections

The big detectors (`cloud`, `locate-anything`, RF-DETR `large`) are slow or
paid. Use one of them on some frames, fix the boxes in the annotator if
needed, then train a small RF-DETR on the result and run that on every
frame of the same or similar videos:

    # 1) label frames with a big model; --no-render only detects and saves
    python process_video.py input.mp4 --detector locate-anything \
      --classes apple --sample-rate 4 --save-detections dets.json --no-render

    # 2) (optional) review / fix the boxes
    python annotate_server.py dets.json

    # 3) fine-tune (dets.json's video is found under the current folder)
    python train_rfdetr.py dets.json -o models/apples

    # 4) use it, on every frame, on this or other videos
    python process_video.py other.mp4 --weights models/apples \
      --count-mode total --hud-rows "APPLES=apple"

Measured on an M5 Max, apples on a conveyor: locate-anything detections of
605 frames, every 2nd frame used (243 train + 60 held out), RF-DETR
`small`, 10 epochs, ~23 s per epoch. It scores mAP50 0.99 on the held-out
frames and then runs at ~50 frames/s on a conveyor video it was not trained
on, finding the same number of apples per frame as locate-anything (which
takes ~5 s per frame).

What `train_rfdetr.py` does:

- Cuts every frame the detections files have out of their videos (frames
  whose detection failed are skipped; frames with no boxes are kept, they
  teach the model what is not an object) and writes them, with their boxes,
  as a COCO dataset in `<output>/dataset/`.
- Holds out `--val-fraction` (default 20%) of each video to score the model,
  in contiguous stretches: neighbouring frames are nearly identical, so a
  random split would mostly measure how well the model remembers them.
- Fine-tunes RF-DETR (`--model`, default `small`) from its COCO weights on
  the GPU (CUDA or Apple's), printing the scores of every epoch.
- Picks the confidence threshold with the best F1 on the held-out frames. A
  fine-tuned model's scores depend on how long it trained (a short run can
  score every object under 0.1), so no fixed threshold fits every model.
  `process_video.py --weights` uses it unless `--threshold` is given.
- Writes `<output>/model.json` (classes, threshold, scores per epoch, what it
  was trained on) and keeps one checkpoint, `checkpoint_best_total.pth` (the
  epoch with the best mAP). RF-DETR's other checkpoints, only needed to
  resume, are deleted unless `--keep-checkpoints`.
- Ctrl-C stops training and keeps the best checkpoint so far.

Options worth knowing (`python train_rfdetr.py --help` for all of them):

- Several files: `train_rfdetr.py a.json b.json`. Each is paired with the
  video named in it, found under `--root`, or pass `--videos a.mp4 b.mp4`.
- `--only-edited` trains only on the frames you changed in the annotator.
  Without it, unreviewed boxes are learned as they are, mistakes included.
- `--classes apple` keeps only some classes (default: every class with
  boxes). `--min-score` (default 0.5) leaves out low-confidence boxes of
  scored detectors (RF-DETR); cloud / locate-anything / hand-drawn boxes are
  all 1.0.
- `--every N` uses every N-th frame: when detecting on every frame,
  neighbouring frames add little and slow training down.
- `--epochs` (default 20), `--batch-size` (default 4, lower it if you run
  out of memory), `--patience N` to stop when the score stops improving.
- `--model nano|small|medium|large`: bigger is more accurate and slower.

With `--weights`, `--classes` are the model's classes (default: all of
them; `--list-classes` prints them) and `--model` is ignored. Detections
saved with `--weights` record the model in `info.weights`; they load back
with `--load-detections` like a cloud file (classes matched by name), so they
can be reviewed in the annotator and used to train the next model.

### From the annotator

The chip button in the annotator's top bar (or `T`) opens the same
training:

- **New model**: tick the detections files to learn from (the open one is
  ticked; files whose video isn't found can't be used), the classes, which
  frames (all, or only the ones fixed by hand, every N-th), the model size and
  epochs. "Same thing from the command line" shows the equivalent
  `train_rfdetr.py` command.
- Training runs in the background, so you can keep annotating; the top bar
  shows its progress. **Models & jobs** shows each epoch's scores, the output
  (`train_rfdetr.py`'s log), and **Stop**, which ends training after scoring
  the model once more and keeps the best checkpoint so far.
- Every trained model under `--root` is listed. **Run on a video…** runs it
  on every frame of a video (`process_video.py --weights ... --no-render`)
  and writes `<video>_<model>_dets.json` next to the video. Pick any video
  under `--root`, or **Upload…** one from your computer (it is saved in
  `videos/uploads/`). With "Open the results in the annotator when done"
  ticked, the detections open by themselves when the job finishes; otherwise
  **Open the detections** opens them, to review what the model found (press
  play, or `P`, to step through the frames) or fix it and train again.
  **Copy command** gives the `process_video.py` command to render with it.
- The server keeps running the Python code it was started with, while the
  page picks up a new UI on reload. After updating the code, the page says
  so (`annotate_server.py changed since it was started`): restart the
  server.
- One job runs at a time. Jobs are kept only in the server's memory: closing
  the page doesn't stop them, stopping the server does, and the list starts
  empty when the server starts. The × on a finished job (or **Clear
  finished**) takes it off the list; the model or detections it made stay.
- **Delete…** on a model deletes its folder: the weights, the frames it was
  trained on and the logs (after asking). Files you put in the folder
  yourself are kept, and so are detections made with the model. A model a
  running job uses can't be deleted.
