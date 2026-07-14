# video_gaussian_blur.py

## What this code is for

`video_gaussian_blur.py` is a command-line Python script that processes a video and adds small Gaussian-blurred spots around detected feature points in every frame.

In practical terms, it reads an input video, finds visually important corner-like points in each frame, softly blurs small circular areas around those points, and writes the processed frames to a new video file.

## How it works

The script uses OpenCV and NumPy.

1. It opens the input video with `cv2.VideoCapture`.
2. For each video frame, it creates a fully Gaussian-blurred copy of the frame.
3. It converts the original frame to grayscale.
4. It detects strong corner/feature points using `cv2.goodFeaturesToTrack`.
5. Around each detected point, it builds a circular fade mask.
6. It blends the original frame with the blurred frame only inside those small masked regions.
7. It writes each processed frame into an output video with `cv2.VideoWriter`.

The result is not a full-frame blur. Only small regions around detected feature points are blurred.

## Main functions

### `add_gaussian_keypoints(frame, num_points=50, blur_kernel=21, point_radius=8)`

Processes one frame.

- `frame`: the input image frame in OpenCV BGR format.
- `num_points`: maximum number of feature points to detect.
- `blur_kernel`: Gaussian blur kernel size. If an even number is provided, the script automatically increases it by 1 because OpenCV requires an odd kernel size.
- `point_radius`: radius of each blurred spot.

This function returns a copy of the frame with blurred spots added.

### `process_video(input_path, output_path, num_points=50, blur_kernel=21, point_radius=8)`

Processes the whole video.

It reads video metadata such as FPS, width, height, and total frame count, then applies `add_gaussian_keypoints` to every frame. Progress is printed every 30 frames.

### `main()`

Defines the command-line interface.

It accepts an input video path, optional output path, and optional tuning parameters for the number of points, blur strength, and point size.

## Usage

Basic usage:

```bash
python video_gaussian_blur.py input.mp4
```

This creates an output file named:

```text
input_blurred.mp4
```

Specify an output file:

```bash
python video_gaussian_blur.py input.mp4 -o output.mp4
```

Change the visual effect:

```bash
python video_gaussian_blur.py input.mp4 -n 100 -k 31 -r 12
```

Options:

- `-o`, `--output`: output video path.
- `-n`, `--num-points`: number of feature points to blur per frame. Default: `50`.
- `-k`, `--blur-kernel`: Gaussian blur kernel size. Default: `21`.
- `-r`, `--point-radius`: radius of each blurred point. Default: `8`.

## Dependencies

The script requires:

- `opencv-python`
- `numpy`

Install them with:

```bash
pip install opencv-python numpy
```

## Notes and limitations

- The output video is written with the `mp4v` codec.
- The script processes video frames only; it does not preserve or copy the original audio track.
- Feature points are detected independently on every frame, so the blurred spots may move or flicker depending on the video content.
- If the input video cannot be opened, the script prints an error and exits.
