# CourtVision AI

An AI-powered basketball video analytics project built with deep learning and computer vision.

This repository is being developed incrementally. **V1 — Player Tracking & Motion Analytics** extends the working V0 detector with ByteTrack IDs, recent trajectories, pixel movement summaries, and an image-coordinate heatmap. The model detects people; it does not distinguish players from other people.

## V1 Demo

**Player ID → Bounding Box → Trajectory**, with a movement heatmap and per-track JSON analytics.

| Tracking preview | Movement heatmap |
| --- | --- |
| ![ByteTrack IDs, bounding boxes, and recent trajectories](docs/assets/v1-demo.gif) | ![Tracked center activity in image coordinates](docs/assets/v1-heatmap.png) |

The public basketball sample keeps ByteTrack IDs **1** and **2** across all **141 frames**, at **320 × 240** and **29.97 FPS**, using CPU inference. The GIF is a lightweight preview sampled at approximately 6 FPS; the annotated MP4 preserves the source FPS. The heatmap represents activity in image coordinates.

[View the generated JSON analytics](docs/assets/v1-analytics.json), including frames seen, cumulative pixel displacement, average/max pixel speed, and normalized displacement for each ID. Distance is measured in **pixels** and speed in **px/s**; court calibration is required for physical measurements.

## Roadmap

- [x] V0 — Player Detection
- [x] V1 — Player Tracking & Motion Analytics
- [ ] V2 — Basketball Event Analytics
- [ ] V3 — Custom Model Fine-Tuning
- [ ] V4 — Pose & Advanced Analytics

## Features

- Reads local video sequentially with OpenCV and loads pretrained COCO YOLO weights once.
- Filters for COCO `person`, with a configurable confidence threshold.
- Preserves the source resolution and FPS in an annotated MP4.
- Selects CUDA when available, otherwise CPU, and prints progress.
- Validates inputs, creates output directories, and releases video resources after errors.

### V1 Features

- Uses the official Ultralytics ByteTrack implementation and `bytetrack.yaml` defaults.
- Displays temporary persistent track IDs and confidence scores.
- Draws bounded trajectories with deterministic colors for each ID.
- Reports cumulative pixel displacement and speed in **px/s**, with optional speed labels.
- Saves per-track JSON analytics and a PNG movement heatmap automatically.
- Handles missing IDs, gaps, and videos with no visible people.

## V1 Architecture

Basketball Video → OpenCV → YOLO → Person Detection → ByteTrack → Persistent IDs → Track History → Motion Analytics → Heatmap → Annotated Video

| Module | Responsibility |
| --- | --- |
| `detector.py` | Load YOLO, select the device, and retain the V0 person detection API. |
| `tracker.py` | Run person inference and pass boxes to the official ByteTrack implementation; extract valid IDs and centers. |
| `track_history.py` | Store a bounded deque of recent centers for each ID; expire stale trails. |
| `analytics.py` | Calculate pixel movement and speed, and export running summaries as JSON. |
| `heatmap.py` | Accumulate center visits in a fixed image grid and export a colored PNG. |
| `utils.py` | Validate paths and draw boxes, labels, and trails. |
| `video_processor.py` | Read frames, coordinate the modules, and write the outputs. |
| `main.py` | Parse CLI options, report progress, and handle normal user errors. |

The original `PlayerDetector.detect(frame)` and `VideoProcessor(detector)` detection-only APIs remain available. Existing CLI arguments continue to work; the CLI now enables V1 tracking by default.

## Installation

Use Python 3.11 or newer. From the `courtvision-ai` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics installs PyTorch as a dependency. `lap` supplies the assignment solver used by ByteTrack and is installed explicitly to avoid a surprise installation during tracking. If you need a particular CUDA build, install PyTorch first using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this project's requirements.

The first run downloads the small pretrained `yolo26n.pt` weights unless they are already available locally. No training or fine-tuning occurs.

## Usage

Place your basketball video at `data/input/basketball.mp4`. Run:

```bash
python main.py --input data/input/basketball.mp4 --output data/output/basketball_tracked.mp4
```

This creates:

```text
data/output/basketball_tracked.mp4
data/output/basketball_tracked_tracking.json
data/output/basketball_tracked_heatmap.png
```

To configure confidence, trail length, speed labels, and export filenames:

```bash
python main.py \
  --input data/input/basketball.mp4 \
  --output data/output/basketball_tracked.mp4 \
  --confidence 0.5 \
  --model yolo26n.pt \
  --tracker bytetrack.yaml \
  --trajectory-length 40 \
  --show-speed \
  --analytics-output data/output/basketball_tracking.json \
  --heatmap-output data/output/basketball_heatmap.png
```

Use `python main.py --help` for all options. A custom `--tracker` path must point to a local YAML file whose `tracker_type` is `bytetrack`. Start with the packaged defaults.

Output directories are created automatically. Video, analytics, and heatmap paths must be distinct and end in `.mp4`, `.json`, and `.png` respectively. Processing can be slower than playback; the saved video still uses the source FPS.

Files are staged before publication. Processing or export failures remove temporary files and preserve existing outputs. If a final rename fails, the processor attempts to restore the previous output set. Filesystem failures that prevent restoration report the location of recoverable backups. If deleting an old backup fails after successful publication, processing remains successful and a warning identifies the leftover backup; other backups are still cleaned up. Publishing three files is not a single atomic filesystem operation, and a power loss during publication cannot be rolled back automatically.

### Run the public basketball demo

The demo uses `v_Basketball_g01_c01.avi` from [tiny-ucf101](https://github.com/bryanyzhu/tiny-ucf101), the clip used in the [GluonCV UCF101 tutorial](https://cv.gluon.ai/build/examples_action_recognition/demo_tsn_ucf101.html). Videos, generated outputs, and model weights are excluded from Git.

On macOS or Linux, download the sample if it is not already present:

```bash
curl --fail --location --output data/input/basketball_sample.avi https://raw.githubusercontent.com/bryanyzhu/tiny-ucf101/master/v_Basketball_g01_c01.avi
```

Run the V1 acceptance command:

```bash
python main.py --input data/input/basketball_sample.avi --output data/output/basketball_sample_tracked.mp4
```

Open `data/output/basketball_sample_tracked.mp4` in your video player. The corresponding files are `basketball_sample_tracked_tracking.json` and `basketball_sample_tracked_heatmap.png` in the same folder. Try your own game footage next with different input and output paths.

### Verified V1 Result

The full public sample was processed on **CPU** with `yolo26n.pt`. Every output frame was decoded, and an annotated frame and the heatmap were inspected visually.

| Property | Input | Output |
| --- | --- | --- |
| Resolution | 320 × 240 | 320 × 240 |
| FPS | 29.97003 | 29.97000 |
| Decoded frames | 141 | 141 |

Both track IDs, `1` and `2`, appeared in all 141 frames. There were 280 trajectory drawing calls after the first observations. The JSON contains two valid track summaries, and the heatmap is a readable 320×240 PNG with activity around the observed centers.

The small FPS difference comes from codec rounding. This short clip verifies the pipeline; it is not a tracking accuracy benchmark or a count of players in a full game.

Validated environment: Python 3.13, Ultralytics 8.4.160, PyTorch 2.14.0, OpenCV 5.0.0, NumPy 2.5.3, and LAP 0.5.13. CUDA selection remains covered by unit tests; actual GPU inference has not been tested on this Mac.

## Understanding ByteTrack

YOLO answers **“Where are the people in this frame?”** It predicts a box, class, and confidence score from the image. COCO class `0` is `person` for the supplied detection weights; model loading checks that mapping.

ByteTrack answers **“Which detections likely belong to the same person across frames?”** It predicts track motion with a Kalman filter and associates boxes between frames. It first matches stronger detections, then uses weaker detections to recover existing tracks. An ID such as `7` is a temporary tracker identity within one video, not a name, jersey number, or biological identity. Occlusion, missed detections, overlapping people, and large movement can cause an ID switch or a new ID.

`PlayerTracker` uses Ultralytics' documented `BYTETracker` directly, with its packaged configuration. This keeps tracker state separate from YOLO and the V0 detection API. It calls `model.predict()` once per frame, inside `torch.inference_mode()`, requesting only people. The inference threshold is the lower of the user threshold and ByteTrack's low threshold, so weak detections can reach association. The user threshold is then applied to tracked outputs used for visualization and analytics. It still controls which people appear in the output.

ByteTrack may need another frame to confirm a newly appearing person. Untracked boxes or invalid IDs are skipped. Empty detection frames still update ByteTrack, allowing lost tracks to age out. A fresh tracker is created for each video; ID numbers can start at `1` again in another video.

## Motion Analytics

Boxes use `[x1, y1, x2, y2]` image pixel coordinates: top-left followed by bottom-right. The tracked position is the **bounding-box center**:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

Float centers are retained for calculations before box coordinates are converted to integers for drawing. For consecutive observations of the same ID:

```text
displacement_px = sqrt((cx2 - cx1)^2 + (cy2 - cy1)^2)
speed_px_per_second = displacement_px * video_fps
```

For example, movement of 3 pixels horizontally and 4 vertically gives 5 pixels of displacement. At 30 FPS, that interval's speed is 150 px/s.

Cumulative distance adds consecutive interval displacements. A missing frame or a hidden low-confidence observation breaks the interval: no distance is added across that gap, and the displayed trail and speed window restart on return. This avoids treating an unknown path as a measured jump.

Speed labels average the last **five valid consecutive intervals** (or fewer during startup). JSON `max_pixel_speed` is the maximum of these smoothed values. `average_pixel_speed` is total measured displacement divided by the time represented by valid consecutive intervals; it excludes missing intervals. A one-frame track has zero measured distance and speed.

`normalized_distance` divides cumulative pixel distance by `sqrt(width² + height²)`, the image diagonal. A value of `1` means the accumulated image displacement equals one frame diagonal. It removes uniform resolution scaling; it does not correct perspective or camera motion.

### Analytics JSON

The report includes source video dimensions, FPS, frames processed, coordinate units, a scientific warning, and one summary per visible ID:

```json
{
  "track_id": 7,
  "frames_seen": 3,
  "cumulative_pixel_distance": 10.0,
  "average_pixel_speed": 150.0,
  "max_pixel_speed": 150.0,
  "normalized_distance": 0.025
}
```

This illustrative example uses two 5-pixel intervals at 30 FPS and a 400-pixel image diagonal. `frames_seen` counts observations meeting the user's confidence threshold; unique tracks count temporary IDs, not unique real players.

### Important Limitation

**Tracking coordinates are image coordinates. Pixel-space distance and speed are NOT real-world distance or physical speed.** Perspective, camera movement, zoom, and box jitter affect the values. Court calibration/homography is required before reporting physical movement in meters or km/h. V1 reports only pixels and px/s.

### Movement Heatmap

A fixed `height × width` float grid counts one visit at each visible tracked center per frame. Visits outside the image are ignored. At export, the grid is blurred with an 8-pixel Gaussian standard deviation, normalized to its own maximum, and colored using OpenCV's Turbo colormap. Inactive pixels remain black. With fixed FPS, more visits correspond to more observed time at that image position.

This is an **image-coordinate heatmap**, not a calibrated court map or shot chart. Colors show relative activity within one video and should not be compared as absolute scales across videos. A video with no visible tracks produces an empty track list and a valid black heatmap.

### Memory and Track Lifecycle

Trails use `deque(maxlen=40)` by default, configurable with `--trajectory-length`. Histories absent for more than 60 frames are removed, and paths never join across missing observations. The heatmap uses a fixed-size grid rather than storing video frames.

Analytics retain small running summaries and at most five speed samples for each unique ID so departed tracks can appear in the final report. Memory therefore scales with recent IDs × trail length, image size, and the number of unique IDs. Complete trajectories and all video frames are never stored. Many ID switches in a very long video can still increase summary memory.

## Tests

Run the fast suite:

```bash
python -m pytest -q
```

**82 tests pass**, including all original V0 tests. Tests use synthetic boxes, tiny generated videos, and mocked predictions without downloading weights. One test runs the real ByteTrack association algorithm with synthetic detections.

Coverage includes confidence/device selection, valid IDs and person filtering, weak-detection association, tracker resets, centers, bounded histories and expiry, motion/smoothing/gaps, zero displacement, heatmaps, JSON, no-person videos, FPS/resolution preservation, invalid configuration/paths, output cleanup, successful processing despite backup-cleanup warnings, and restoring previous files after failures.

## Seven Parts to Understand Before Putting V1 on Your CV

1. **Detection and device selection — `PlayerDetector`:** YOLO loads once; class `0` is checked; CUDA is selected only when available; inference avoids autograd bookkeeping.
2. **Association and ID extraction — `PlayerTracker.track()` / `extract_tracked_people()`:** weak detections support ByteTrack, user confidence filters outputs, and only genuine tracker IDs are returned.
3. **Bounded trajectories — `TrackHistory.update()`:** deques retain recent centers, gaps restart trails, and old histories expire.
4. **Movement math — `MotionAnalytics._update_track()` / `speed()`:** Euclidean displacement, FPS-based timing, gap handling, and five-interval speed smoothing.
5. **Summary semantics — `MotionAnalytics.to_dict()` / `save_json()`:** valid observation time, maximum smoothed speed, diagonal normalization, scientific units, and safe JSON export.
6. **Heatmaps — `MovementHeatmap.update()` / `render()`:** fixed grid accumulation, blur, normalization, and image-coordinate limitations.
7. **Video orchestration — `VideoProcessor.process()` and drawing helpers:** frame sequencing, consistent colors, IDs/trails, original metadata, staged outputs, and resource cleanup.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Input video does not exist | Check the path and working directory. Quote paths containing spaces. |
| Could not load YOLO model | Allow the initial weight download or use a local COCO checkpoint with `--model`. |
| Invalid ByteTrack configuration | Use `--tracker bytetrack.yaml`, or check your local YAML values and tracker type. |
| Could not open input video | Try a file that plays locally; the source or codec may be unsupported. |
| Invalid FPS | Re-export the source with a valid fixed frame rate. |
| Even width and height required | Re-export with even dimensions to avoid silent codec cropping. |
| Decoding stopped early | Check for corruption or incorrect frame-count metadata. |
| IDs flicker or change | Try clearer footage and check detector confidence; occlusion and overlapping people can break association. |
| Large px/s values | Check camera motion, box jitter, and ID switches; values are not physical speed. |
| Slow processing on CPU | Start with a short clip. Processing duration does not change output playback FPS. |

## Project Structure

```text
courtvision-ai/
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── tracker.py
│   ├── track_history.py
│   ├── analytics.py
│   ├── heatmap.py
│   ├── video_processor.py
│   └── utils.py
├── data/
│   ├── input/
│   └── output/
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_track_history.py
│   ├── test_analytics.py
│   ├── test_heatmap.py
│   ├── test_v1_pipeline.py
│   └── test_video_processor.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The data folders are kept in Git; videos, `.pt` weights, and generated JSON/PNG files in `data/output` are ignored.

## Current Limitations

- Pretrained COCO detection may include spectators and referees.
- IDs can switch during occlusion; one real player may receive several IDs. A track ID is not actual player identity.
- Tracking quality depends on detector accuracy, camera angle, lighting, resolution, and hardware.
- Pixel motion is perspective-dependent and includes camera movement and bounding-box jitter.
- The heatmap uses image coordinates; no court calibration/homography is implemented.
- No basketball detection/tracking, shot detection, team recognition, or pose estimation.
- Source audio is not copied; OpenCV writes annotated frames only.
- OpenCV's `mp4v` playback depends on local codecs, and width/height must be even.
- Source FPS is treated as fixed; variable-frame-rate timestamps are not preserved.
- Unknown or inaccurate frame-count metadata can limit detection of decoding failures.

## Future Work

V2 will explore basketball event analytics, beginning with appropriately evaluated ball and event detection. Court calibration is needed before introducing real-world movement measurements. V1 deliberately contains no V2 features.

## References

- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/)
- [Official Ultralytics ByteTrack API](https://docs.ultralytics.com/reference/trackers/byte_tracker/)
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics COCO class list](https://docs.ultralytics.com/datasets/detect/coco/)
