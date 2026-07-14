# Ubuntu server setup for video Gaussian blur tests

This README explains how to manage a Python virtual environment on an Ubuntu server for these two scripts:

- `video_gaussian_blur.py`
- `video_gaussian_blur_cuda.py`

The first script uses normal CPU OpenCV. The second script tries to use `cv2.cuda` when a CUDA-enabled OpenCV build and NVIDIA GPU are available.

## 1. Check system tools

On the Ubuntu server, check Python:

```bash
python3 --version
```

If you want to test CUDA/NVIDIA acceleration, also check:

```bash
nvidia-smi
```

If `nvidia-smi` does not work, the server either does not have an NVIDIA GPU or the NVIDIA driver is not installed correctly.

## 2. Install uv

If `uv` is already installed:

```bash
uv --version
```

If it is not installed, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart the shell, or run:

```bash
source ~/.bashrc
```

Check again:

```bash
uv --version
```

## 3. Create the virtual environment

Go to the folder that contains the Python files:

```bash
cd /path/to/watermark
```

Create a virtual environment:

```bash
uv venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, your shell prompt should usually show `(.venv)`.

## 4. Install Python dependencies

For the CPU script and basic testing:

```bash
uv pip install numpy opencv-python
```

For the Python backend and browser upload frontend:

```bash
uv pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import cv2, numpy as np; print('cv2', cv2.__version__); print('numpy', np.__version__)"
```

## 5. Important CUDA note

The normal PyPI package `opencv-python` usually does **not** include real CUDA acceleration.

That means this command:

```bash
uv pip install opencv-python
```

is enough for:

```bash
python video_gaussian_blur.py ...
```

but may not be enough for actual GPU acceleration in:

```bash
python video_gaussian_blur_cuda.py ...
```

Check whether OpenCV can see CUDA devices:

```bash
python -c "import cv2; print('has cuda:', hasattr(cv2, 'cuda')); print('cuda devices:', cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 'n/a')"
```

Expected result for real CUDA usage:

```text
has cuda: True
cuda devices: 1
```

or another number greater than `0`.

If it prints `cuda devices: 0`, the CUDA script will fall back to CPU processing.

To get real `cv2.cuda` acceleration, you usually need a custom OpenCV build compiled with CUDA support on the Ubuntu server.

## 6. Run the CPU batch script

Prepare folders:

```bash
mkdir -p input_videos output_cpu
```

Put test videos into `input_videos/`.

Run:

```bash
python video_gaussian_blur.py input_videos output_cpu
```

With custom parameters:

```bash
python video_gaussian_blur.py input_videos output_cpu -n 50 -k 21 -r 8
```

Supported video extensions:

```text
.mp4, .mov, .avi, .mkv, .webm, .m4v
```

The script prints:

- progress for each video;
- time used for each file;
- average processing time at the end.

## 7. Run the CUDA batch script

Prepare an output folder:

```bash
mkdir -p output_cuda
```

Run with CUDA auto-detection:

```bash
python video_gaussian_blur_cuda.py input_videos output_cuda --device 0
```

Force CPU mode for comparison:

```bash
python video_gaussian_blur_cuda.py input_videos output_cuda --force-cpu
```

Use the same visual parameters:

```bash
python video_gaussian_blur_cuda.py input_videos output_cuda -n 50 -k 21 -r 8 --device 0
```

The CUDA script prints whether it is using CUDA or falling back to CPU.

## 8. Run single-file mode

Both scripts still support single-file processing.

CPU:

```bash
python video_gaussian_blur.py input.mp4 -o output.mp4
```

CUDA:

```bash
python video_gaussian_blur_cuda.py input.mp4 -o output_cuda.mp4 --device 0
```

## 9. Deactivate the environment

When finished:

```bash
deactivate
```

## 10. Re-enter the environment later

Return to the project folder:

```bash
cd /path/to/watermark
```

Activate:

```bash
source .venv/bin/activate
```

Run scripts as normal.

## 11. Update dependencies

To upgrade installed packages:

```bash
uv pip install --upgrade numpy opencv-python
```

To see installed packages:

```bash
uv pip freeze
```

To save dependencies:

```bash
uv pip freeze > requirements.txt
```

To reinstall from saved dependencies:

```bash
uv pip install -r requirements.txt
```

## 12. Recreate the environment

If the environment becomes broken, remove it and create a new one:

```bash
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv pip install numpy opencv-python
```

Be careful with `rm -rf .venv`: only run it from the project folder and only when you are sure you want to delete the virtual environment.

## 13. Suggested test procedure

1. Put the same test videos in `input_videos/`.
2. Run the CPU script:

```bash
python video_gaussian_blur.py input_videos output_cpu
```

3. Run the CUDA script:

```bash
python video_gaussian_blur_cuda.py input_videos output_cuda --device 0
```

4. Compare the printed average processing times.
5. Check the output videos visually.
6. If testing watermark robustness, run your watermark extraction tool on both `output_cpu/` and `output_cuda/`.

## 14. Build OpenCV with real CUDA support

The normal `opencv-python` package from PyPI is convenient, but it usually does not provide real CUDA acceleration. For real `cv2.cuda` support, build OpenCV from source on the Ubuntu server.

The rough flow is:

1. Install NVIDIA driver and CUDA Toolkit.
2. Create and activate the project `.venv`.
3. Install Python build dependencies into `.venv`.
4. Download matching OpenCV and `opencv_contrib` source code.
5. Configure OpenCV with `WITH_CUDA=ON`.
6. Build and install the generated Python binding into `.venv`.
7. Verify that `cv2.cuda.getCudaEnabledDeviceCount()` returns a number greater than `0`.

### 14.1 Check NVIDIA driver and CUDA Toolkit

Check that the server can see the NVIDIA GPU:

```bash
lspci | grep -i nvidia
nvidia-smi
```

Check CUDA compiler:

```bash
nvcc --version
```

If `nvcc` is missing, install the CUDA Toolkit using NVIDIA's official instructions for your Ubuntu version.

For many Ubuntu servers, after the NVIDIA CUDA apt repository is configured, the Toolkit install step is:

```bash
sudo apt update
sudo apt install cuda-toolkit
```

Then confirm:

```bash
nvcc --version
```

### 14.2 Install system build dependencies

Install build tools and common video/image dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config \
  ffmpeg \
  libavcodec-dev \
  libavformat-dev \
  libavutil-dev \
  libswscale-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libwebp-dev \
  python3-dev
```

### 14.3 Prepare the uv environment

From this project folder:

```bash
cd /path/to/watermark
uv venv .venv
source .venv/bin/activate
uv pip install numpy
```

Remove the PyPI OpenCV wheel if it was already installed:

```bash
uv pip uninstall opencv-python opencv-contrib-python opencv-python-headless
```

It is okay if those packages were not installed.

### 14.4 Download OpenCV source

Use matching versions for `opencv` and `opencv_contrib`.

Example using OpenCV `4.x`:

```bash
mkdir -p ~/src/opencv-cuda
cd ~/src/opencv-cuda
git clone --branch 4.x --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.x --depth 1 https://github.com/opencv/opencv_contrib.git
```

### 14.5 Configure the CUDA build

Stay inside the activated `.venv`, then run:

```bash
cd ~/src/opencv-cuda
mkdir -p build
cd build
```

Configure with CMake:

```bash
cmake -G Ninja ../opencv \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python2=OFF \
  -D PYTHON3_EXECUTABLE="$(which python)" \
  -D PYTHON3_PACKAGES_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
  -D PYTHON3_NUMPY_INCLUDE_DIRS="$(python -c 'import numpy; print(numpy.get_include())')" \
  -D WITH_CUDA=ON \
  -D WITH_CUBLAS=ON \
  -D CUDA_FAST_MATH=ON \
  -D ENABLE_FAST_MATH=ON \
  -D WITH_FFMPEG=ON
```

During configuration, look for lines showing CUDA is enabled. If CMake says CUDA is unavailable, stop and fix the CUDA Toolkit / driver setup before building.

Optional: set your GPU architecture explicitly for faster builds and smaller binaries:

```bash
-D CUDA_ARCH_BIN=8.6
```

Use the compute capability that matches your GPU. For example, some common values are `7.5`, `8.0`, `8.6`, `8.9`, or `9.0`.

### 14.6 Build and install

Build:

```bash
ninja
```

This may take a long time.

Install into the active `.venv`:

```bash
ninja install
```

If import fails because shared libraries are not found, add the virtual environment library path:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
```

To make that persistent for this environment, add it to your shell profile or to a small activation helper script.

### 14.7 Verify CUDA-enabled cv2

From the project folder with `.venv` activated:

```bash
python - <<'PY'
import cv2

print("cv2 version:", cv2.__version__)
print("has cuda:", hasattr(cv2, "cuda"))
print("cuda devices:", cv2.cuda.getCudaEnabledDeviceCount())
print()
print(cv2.getBuildInformation())
PY
```

The important check is:

```text
cuda devices: 1
```

or any number greater than `0`.

You can also search the build info:

```bash
python - <<'PY'
import cv2
info = cv2.getBuildInformation()
for line in info.splitlines():
    if "CUDA" in line or "NVIDIA" in line or "cuDNN" in line:
        print(line)
PY
```

### 14.8 Run the CUDA script

Now run:

```bash
cd /path/to/watermark
source .venv/bin/activate
python video_gaussian_blur_cuda.py input_videos output_cuda --device 0
```

If it prints that CUDA is being used, the OpenCV CUDA build is working.

### 14.9 Common problems

- `cuda devices: 0`: OpenCV was not built with working CUDA support, the NVIDIA driver is missing, or the server cannot access the GPU.
- `nvcc: command not found`: CUDA Toolkit is not installed or not on `PATH`.
- `ImportError: libopencv_*.so`: add `$VIRTUAL_ENV/lib` to `LD_LIBRARY_PATH`.
- CMake cannot find Python: make sure `.venv` is activated and `PYTHON3_EXECUTABLE="$(which python)"` points to `.venv/bin/python`.
- CMake cannot find NumPy: run `uv pip install numpy` inside `.venv`.
- Build runs out of memory: reduce parallelism with `ninja -j2`.

## 15. Run the Python backend and upload frontend

The project also includes a separate Python backend and static frontend:

- `backend.py`
- `frontend/index.html`

The two original Python processing scripts are still kept for command-line testing.

### 15.1 Install backend dependencies

From the project folder:

```bash
cd /path/to/watermark
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you built a custom CUDA-enabled OpenCV into `.venv`, do not reinstall `opencv-python` from PyPI afterward because that can replace your CUDA-enabled build.

### 15.2 Start the server

Run:

```bash
python -m uvicorn backend:app --host 0.0.0.0 --port 8787
```

Open the frontend in a browser:

```text
http://SERVER_IP:8787/app/
```

Health check:

```text
http://SERVER_IP:8787/api/health
```

### 15.3 What the frontend supports

The browser UI supports:

- uploading one or more video files;
- setting feature point count;
- setting blur kernel size;
- setting blurred point radius;
- optionally trying the CUDA processor;
- downloading processed output files;
- viewing per-file processing time and average processing time.

### 15.4 Backend API

Process videos:

```text
POST /api/process
```

Multipart fields:

- `files`: one or more video files.
- `num_points`: default `50`.
- `blur_kernel`: default `21`.
- `point_radius`: default `8`.
- `use_cuda`: `true` or `false`.
- `force_cpu`: `true` or `false`.
- `device_id`: CUDA device id, default `0`.

Download processed file:

```text
GET /api/download/{filename}
```

### 15.5 Backend storage

The backend stores temporary uploads and outputs under:

```text
backend_data/
```

Temporary uploaded input files are deleted after processing. Processed output files remain in:

```text
backend_data/outputs/
```

Delete old outputs manually when you no longer need them.

### 15.6 Concerns for server testing

- Processing is currently synchronous, so a request stays open until the video batch finishes.
- Large videos may take a long time and can tie up the backend worker.
- There is no authentication yet, so do not expose this directly to the public internet.
- Output files are retained until manually deleted.
- CUDA mode only accelerates if OpenCV was actually built with CUDA and can see the NVIDIA GPU.
- If you install `opencv-python` from PyPI after building CUDA OpenCV, it may overwrite the CUDA-enabled Python binding.
