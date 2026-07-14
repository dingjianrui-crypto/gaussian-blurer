import argparse
import os
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def mux_original_audio(input_path, video_only_path, output_path):
    """Use ffmpeg to attach the original audio stream to the processed video."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Warning: ffmpeg was not found; output will not include original audio.")
        os.replace(video_only_path, output_path)
        return

    command = [
        ffmpeg,
        "-y",
        "-i", video_only_path,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        print("Warning: audio muxing failed; output will not include original audio.")
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        if stderr:
            print(stderr)
        os.replace(video_only_path, output_path)
    else:
        os.remove(video_only_path)


def make_odd(value):
    """OpenCV Gaussian kernels must have odd dimensions."""
    return value if value % 2 == 1 else value + 1


def cv_type_for_frame(frame):
    if frame.dtype != np.uint8:
        raise ValueError("Only uint8 video frames are supported.")

    if len(frame.shape) == 2:
        return cv2.CV_8UC1

    channels = frame.shape[2]
    if channels == 3:
        return cv2.CV_8UC3
    if channels == 4:
        return cv2.CV_8UC4

    raise ValueError(f"Unsupported channel count: {channels}")


def cuda_available(device_id=0):
    if not hasattr(cv2, "cuda"):
        return False, "cv2.cuda is not available in this OpenCV build."

    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
    except cv2.error as exc:
        return False, f"Could not query CUDA devices: {exc}"

    if device_count <= 0:
        return False, "No CUDA-enabled GPU was detected by OpenCV."

    if device_id < 0 or device_id >= device_count:
        return False, f"CUDA device {device_id} is invalid; found {device_count} device(s)."

    try:
        cv2.cuda.setDevice(device_id)
    except cv2.error as exc:
        return False, f"Could not select CUDA device {device_id}: {exc}"

    return True, f"Using CUDA device {device_id} of {device_count}."


class CudaFrameProcessor:
    """GPU helper for full-frame blur and BGR-to-gray conversion."""

    def __init__(self, blur_kernel):
        self.blur_kernel = make_odd(blur_kernel)
        self._filter = None
        self._filter_type = None

    def _get_gaussian_filter(self, src_type):
        if self._filter is None or self._filter_type != src_type:
            self._filter = cv2.cuda.createGaussianFilter(
                src_type,
                src_type,
                (self.blur_kernel, self.blur_kernel),
                0,
            )
            self._filter_type = src_type
        return self._filter

    def blur_and_gray(self, frame):
        src_type = cv_type_for_frame(frame)
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        gaussian_filter = self._get_gaussian_filter(src_type)
        blurred_gpu = gaussian_filter.apply(gpu_frame)

        if len(frame.shape) == 2:
            gray_gpu = gpu_frame
        else:
            gray_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        return blurred_gpu.download(), gray_gpu.download()


def cpu_blur_and_gray(frame, blur_kernel):
    blur_kernel = make_odd(blur_kernel)
    blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return blurred, gray


def add_gaussian_keypoints(
    frame,
    num_points=50,
    blur_kernel=21,
    point_radius=8,
    cuda_processor=None,
):
    """
    Add soft Gaussian-blurred spots around detected feature points.

    The CUDA path accelerates the full-frame blur and grayscale conversion.
    Feature detection and per-point blending still run on CPU.
    """
    if cuda_processor is not None:
        try:
            blurred, gray = cuda_processor.blur_and_gray(frame)
        except (cv2.error, ValueError) as exc:
            print(f"CUDA frame processing failed; falling back to CPU. Detail: {exc}")
            blurred, gray = cpu_blur_and_gray(frame, blur_kernel)
    else:
        blurred, gray = cpu_blur_and_gray(frame, blur_kernel)

    corners = cv2.goodFeaturesToTrack(gray, num_points, 0.01, 10)
    result = frame.copy()

    if corners is None:
        return result

    corners = corners.astype(np.intp)

    for corner in corners:
        x, y = corner.ravel()

        y_min = max(0, y - point_radius)
        y_max = min(frame.shape[0], y + point_radius)
        x_min = max(0, x - point_radius)
        x_max = min(frame.shape[1], x + point_radius)

        roi_blur = blurred[y_min:y_max, x_min:x_max]

        h, w = y_max - y_min, x_max - x_min
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = y - y_min, x - x_min
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = np.clip(1.0 - dist / point_radius, 0, 1).astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=2)

        roi_original = result[y_min:y_max, x_min:x_max].astype(np.float32)
        roi_blur = roi_blur.astype(np.float32)

        blended = roi_original * (1 - mask_3ch) + roi_blur * mask_3ch
        result[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)

    return result


def process_video(
    input_path,
    output_path,
    num_points=50,
    blur_kernel=21,
    point_radius=8,
    device_id=0,
    force_cpu=False,
):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        print("Output file cannot be the same as input file because original audio must be preserved.")
        return False

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Could not open video file: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    output_ext = os.path.splitext(output_path)[1] or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(
        prefix=".gaussian_blur_video_only_",
        suffix=output_ext,
        dir=output_dir,
        delete=False,
    )
    temp_video_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        os.remove(temp_video_path)
        print(f"Could not create output video file: {output_path}")
        return False

    cuda_processor = None
    if force_cpu:
        print("CUDA disabled by --force-cpu; using CPU processing.")
    else:
        ok, message = cuda_available(device_id)
        print(message)
        if ok:
            cuda_processor = CudaFrameProcessor(blur_kernel)
        else:
            print("Using CPU processing instead.")

    frame_count = 0
    print(f"Start processing video, total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = add_gaussian_keypoints(
            frame,
            num_points=num_points,
            blur_kernel=blur_kernel,
            point_radius=point_radius,
            cuda_processor=cuda_processor,
        )
        out.write(processed)

        frame_count += 1
        if frame_count % 30 == 0:
            if total_frames > 0:
                progress = frame_count / total_frames * 100
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
            else:
                print(f"Progress: {frame_count} frames processed")

    cap.release()
    out.release()
    mux_original_audio(input_path, temp_video_path, output_path)
    print(f"Done. Output file: {output_path}")
    return True


def list_video_files(input_folder):
    videos = []
    for filename in sorted(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path):
            continue

        _, ext = os.path.splitext(filename)
        if ext.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            videos.append(input_path)

    return videos


def process_folder(
    input_folder,
    output_folder,
    num_points=50,
    blur_kernel=21,
    point_radius=8,
    device_id=0,
    force_cpu=False,
):
    if not os.path.isdir(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    video_files = list_video_files(input_folder)
    if not video_files:
        print(f"No supported video files found in input folder: {input_folder}")
        return

    durations = []
    print(f"Found {len(video_files)} video file(s). Starting batch processing...")

    for index, input_path in enumerate(video_files, start=1):
        filename = os.path.basename(input_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{base}_cuda_blurred{ext}")

        print(f"\n[{index}/{len(video_files)}] Processing: {filename}")
        start_time = time.perf_counter()
        ok = process_video(
            input_path=input_path,
            output_path=output_path,
            num_points=num_points,
            blur_kernel=blur_kernel,
            point_radius=point_radius,
            device_id=device_id,
            force_cpu=force_cpu,
        )
        elapsed = time.perf_counter() - start_time

        if ok:
            durations.append(elapsed)
            print(f"File processing time: {elapsed:.2f} seconds")
        else:
            print(f"File processing failed after {elapsed:.2f} seconds")

    if durations:
        average = sum(durations) / len(durations)
        print(f"\nBatch complete: {len(durations)}/{len(video_files)} file(s) succeeded")
        print(f"Average processing time: {average:.2f} seconds/file")
    else:
        print("\nBatch complete: no files were processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Add Gaussian-blurred feature-point spots to videos, with single-file and folder batch modes."
    )
    parser.add_argument("input", help="Input video file path or input video folder path")
    parser.add_argument("output", nargs="?", help="Output video file path or output video folder path")
    parser.add_argument("-o", "--output-file", help="Single-file output video path, kept for compatibility")
    parser.add_argument("-n", "--num-points", type=int, default=50, help="Feature-point count. Default: 50")
    parser.add_argument("-k", "--blur-kernel", type=int, default=21, help="Gaussian blur kernel size. Default: 21")
    parser.add_argument("-r", "--point-radius", type=int, default=8, help="Blurred point radius in pixels. Default: 8")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id. Default: 0")
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA and run the CPU path")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        if not args.output:
            parser.error("folder batch mode requires an output folder path")

        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            num_points=args.num_points,
            blur_kernel=args.blur_kernel,
            point_radius=args.point_radius,
            device_id=args.device,
            force_cpu=args.force_cpu,
        )
        return

    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}")
        return

    output_path = args.output_file or args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_cuda_blurred{ext}"

    start_time = time.perf_counter()
    ok = process_video(
        input_path=args.input,
        output_path=output_path,
        num_points=args.num_points,
        blur_kernel=args.blur_kernel,
        point_radius=args.point_radius,
        device_id=args.device,
        force_cpu=args.force_cpu,
    )
    elapsed = time.perf_counter() - start_time

    if ok:
        print(f"File processing time: {elapsed:.2f} seconds")
        print(f"Average processing time: {elapsed:.2f} seconds/file")
    else:
        print(f"File processing failed after {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
