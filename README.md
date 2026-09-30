# CourtVision AI

An AI-powered basketball video analytics project built with deep learning and computer vision.

This repository is being developed incrementally. **V0** detects people in a local basketball video and saves an annotated copy. It does not identify which people are players.

## Roadmap

- [x] V0 — Player Detection
- [ ] V1 — Player Tracking
- [ ] V2 — Basketball Event Analytics
- [ ] V3 — Custom Model Fine-Tuning
- [ ] V4 — Pose & Advanced Analytics

## Features

- Reads a local video frame by frame with OpenCV.
- Runs pretrained, COCO based YOLO detection on each frame.
- Keeps only `person` detections that meet a configurable confidence threshold.
- Draws a `Player` label and confidence score on each box.
- Saves an MP4 at the source video's resolution and FPS.
- Selects CUDA when available, otherwise CPU, and prints progress.

## Architecture

Basketball Video → OpenCV → YOLO → Person Detection → Bounding Boxes → Annotated Video

`PlayerDetector` loads the model once and converts frame predictions to simple detection dictionaries. `VideoProcessor` handles video input, frame processing, and output. `utils.py` contains validation and drawing helpers. `main.py` is the command-line entry point.

## Installation

Use Python 3.11 or newer. From the `courtvision-ai` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics installs PyTorch as a dependency. If you need a particular CUDA build, install PyTorch first using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this project's requirements. The first run downloads the small pretrained `yolo26n.pt` weights unless the file is already available locally.

## Usage

Place your basketball video at `data/input/basketball.mp4`. From the `courtvision-ai` directory, run:

```bash
python main.py --input data/input/basketball.mp4 --output data/output/basketball_detected.mp4
```

To change the minimum confidence or use another COCO detection checkpoint:

```bash
python main.py --input data/input/basketball.mp4 --output data/output/basketball_detected.mp4 --confidence 0.6 --model yolo26n.pt
```

The output folder is created automatically. The output filename must end in `.mp4`. Model inference may take longer than playback, but the saved file uses the original FPS.

Output is first written to a temporary file in the destination folder. On success, the completed video replaces the requested output. If processing fails, the temporary file is removed and any existing output is preserved.

### Run the public basketball demo

The local demo uses `v_Basketball_g01_c01.avi` from [tiny-ucf101](https://github.com/bryanyzhu/tiny-ucf101), the clip used in the [GluonCV UCF101 tutorial](https://cv.gluon.ai/build/examples_action_recognition/demo_tsn_ucf101.html). The downloaded video and model weights are excluded from Git.

On macOS or Linux, download the sample if it is not already in your input folder:

```bash
curl --fail --location --output data/input/basketball_sample.avi https://raw.githubusercontent.com/bryanyzhu/tiny-ucf101/master/v_Basketball_g01_c01.avi
```

Process the demo with:

```bash
python main.py --input data/input/basketball_sample.avi --output data/output/basketball_sample_detected.mp4 --confidence 0.5
```

Open `data/output/basketball_sample_detected.mp4` in your video player. Try your own game footage next using the same command with different input and output paths.

### Verified V0 result

The full public sample was processed on CPU with `yolo26n.pt`. Every output frame was decoded to check that the result is readable. A frame was also inspected visually for boxes and confidence labels.

| Property | Input | Output |
| --- | --- | --- |
| Resolution | 320 × 240 | 320 × 240 |
| FPS | 29.97003 | 29.97000 |
| Decoded frames | 141 | 141 |

The tiny FPS difference comes from the output codec's rounding. This demonstration verifies the video pipeline; it is not an accuracy benchmark for basketball players.

Validated environment: Python 3.13, Ultralytics 8.4.160, PyTorch 2.14.0, OpenCV 5.0.0, and NumPy 2.5.3. CUDA selection is covered by unit tests; actual GPU inference still needs a compatible NVIDIA system.

## Tests

Run the fast tests with:

```bash
python -m pytest -q
```

Tests use small generated videos and mocked predictions, so they do not download weights or require a full game. They cover person filtering, confidence validation, device selection, FPS/resolution preservation, visible annotation, invalid paths, decoding failures, resource cleanup, and preservation of existing outputs when processing fails.

## Understanding the Code

1. **Command-line options — `main.py`:** `argparse` reads the file paths, confidence threshold, and model name. The entry point creates one detector and passes it to the video processor.
2. **Model loading — `src/detector.py`:** the pretrained weights are loaded once. CUDA is selected when available; otherwise inference uses CPU. No training occurs.
3. **Inference and filtering — `src/detector.py`:** each frame is passed to `model.predict()` inside `torch.inference_mode()`. COCO class `0` is `person`. Only person detections that meet the confidence threshold are returned as dictionaries.
4. **Video processing — `src/video_processor.py`:** OpenCV reads one frame at a time. The writer uses the input width, height, and FPS so the output keeps the same frame size and playback timing. Resources are released in `finally`, including after errors.
5. **Drawing — `src/utils.py`:** boxes use `[x1, y1, x2, y2]` pixel coordinates: top-left followed by bottom-right. `cv2.rectangle()` draws a box; `cv2.putText()` adds a label such as `Player 0.92`.

Confidence is a model score, not a guarantee of correctness. Raising the threshold removes more uncertain detections but can also hide real people. Each frame is analyzed independently; a person in consecutive frames does not receive a persistent identity.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Input video does not exist | Check the path and run from the project directory. Quote paths containing spaces. |
| Could not load YOLO model | Allow the initial weight download or pass an existing COCO checkpoint with `--model`. |
| Could not open input video | Try a video that plays locally; the file or codec may be unsupported. |
| Invalid FPS | Re-export the source with a valid fixed frame rate. |
| Even width and height required | Re-export the source with even dimensions before processing. |
| Decoding stopped early | Check the source for corruption or incorrect frame-count metadata. |
| Slow processing on CPU | Start with a short clip. Processing time does not change the saved video's playback FPS. |

## Project Structure

```text
courtvision-ai/
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── video_processor.py
│   └── utils.py
├── data/
│   ├── input/
│   └── output/
├── tests/
│   ├── test_detector.py
│   └── test_video_processor.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The data folders are kept in Git, while videos and `.pt` weights are ignored.

## Current Limitations

- Uses a pretrained COCO `person` detector, so spectators and referees may also be detected.
- No persistent player tracking: the same person is detected independently in every frame.
- The basketball is not specifically detected in V0.
- Accuracy and speed depend on camera angle, lighting, resolution, and hardware.
- OpenCV writes the annotated video frames only; source audio is not copied.
- The MP4 writer uses OpenCV's `mp4v` codec; playback depends on the codecs available on your system.
- Width and height must both be even for this codec. Odd dimensions are rejected before writing to avoid silent cropping.
- Processing reports an error if decoding stops before the reported frame count. Frame-count metadata can be inaccurate, and an unknown count limits the ability to distinguish a decoding failure from the end of a file.

## Future Work

V1 will add tracking: Player Detection → ByteTrack → Persistent Player IDs → Trajectories.

## References

- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics COCO class list](https://docs.ultralytics.com/datasets/detect/coco/)
