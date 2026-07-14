import cv2
import numpy as np
import argparse
import os
import time


SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def add_gaussian_keypoints(frame, num_points=50, blur_kernel=21, point_radius=8):
    """
    给单帧图像添加高斯模糊的特征点

    Args:
        frame: 输入图像 (BGR格式)
        num_points: 特征点数量
        blur_kernel: 高斯模糊核大小 (必须是奇数)
        point_radius: 特征点半径

    Returns:
        添加了高斯模糊特征点的图像
    """
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, num_points, 0.01, 10)

    result = frame.copy()

    if corners is not None:
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


def process_video(input_path, output_path, num_points=50, blur_kernel=21, point_radius=8):
    """
    处理视频，给每一帧添加高斯模糊特征点

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        num_points: 特征点数量
        blur_kernel: 高斯模糊核大小
        point_radius: 特征点半径
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        print(f"无法创建输出视频文件: {output_path}")
        return False

    frame_count = 0
    print(f"开始处理视频，共 {total_frames} 帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = add_gaussian_keypoints(frame, num_points, blur_kernel, point_radius)
        out.write(processed)

        frame_count += 1
        if frame_count % 30 == 0:
            if total_frames > 0:
                progress = frame_count / total_frames * 100
                print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
            else:
                print(f"处理进度: 已处理 {frame_count} 帧")

    cap.release()
    out.release()
    print(f"处理完成! 输出文件: {output_path}")
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


def process_folder(input_folder, output_folder, num_points=50, blur_kernel=21, point_radius=8):
    """
    批量处理输入文件夹中的视频，并将结果输出到另一个文件夹。
    """
    if not os.path.isdir(input_folder):
        print(f"输入文件夹不存在: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    video_files = list_video_files(input_folder)
    if not video_files:
        print(f"输入文件夹中没有找到支持的视频文件: {input_folder}")
        return

    durations = []
    print(f"找到 {len(video_files)} 个视频文件，开始批量处理...")

    for index, input_path in enumerate(video_files, start=1):
        filename = os.path.basename(input_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{base}_blurred{ext}")

        print(f"\n[{index}/{len(video_files)}] 开始处理: {filename}")
        start_time = time.perf_counter()
        ok = process_video(
            input_path=input_path,
            output_path=output_path,
            num_points=num_points,
            blur_kernel=blur_kernel,
            point_radius=point_radius
        )
        elapsed = time.perf_counter() - start_time

        if ok:
            durations.append(elapsed)
            print(f"文件处理耗时: {elapsed:.2f} 秒")
        else:
            print(f"文件处理失败，耗时: {elapsed:.2f} 秒")

    if durations:
        average = sum(durations) / len(durations)
        print(f"\n批量处理完成: 成功 {len(durations)}/{len(video_files)} 个文件")
        print(f"平均处理耗时: {average:.2f} 秒/文件")
    else:
        print("\n批量处理完成: 没有文件处理成功")


def main():
    parser = argparse.ArgumentParser(description='给视频每一帧添加高斯模糊的特征点，支持单文件或文件夹批量处理')
    parser.add_argument('input', help='输入视频文件路径或输入视频文件夹路径')
    parser.add_argument('output', nargs='?', help='输出视频文件路径或输出视频文件夹路径')
    parser.add_argument('-o', '--output-file', help='单文件输出视频路径，兼容旧版命令格式')
    parser.add_argument('-n', '--num-points', type=int, default=50, help='特征点数量 (默认: 50)')
    parser.add_argument('-k', '--blur-kernel', type=int, default=21, help='高斯模糊核大小，必须是奇数 (默认: 21)')
    parser.add_argument('-r', '--point-radius', type=int, default=8, help='特征点半径 (默认: 8)')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        if not args.output:
            parser.error('批量处理时必须提供输出文件夹路径')

        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            num_points=args.num_points,
            blur_kernel=args.blur_kernel,
            point_radius=args.point_radius
        )
        return

    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return

    output_path = args.output_file or args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_blurred{ext}"

    start_time = time.perf_counter()
    ok = process_video(
        input_path=args.input,
        output_path=output_path,
        num_points=args.num_points,
        blur_kernel=args.blur_kernel,
        point_radius=args.point_radius
    )
    elapsed = time.perf_counter() - start_time

    if ok:
        print(f"文件处理耗时: {elapsed:.2f} 秒")
        print(f"平均处理耗时: {elapsed:.2f} 秒/文件")
    else:
        print(f"文件处理失败，耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
