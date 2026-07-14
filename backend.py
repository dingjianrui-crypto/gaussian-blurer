import re
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from video_gaussian_blur import SUPPORTED_VIDEO_EXTENSIONS, process_video as process_video_cpu
from video_gaussian_blur_cuda import process_video as process_video_cuda


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "backend_data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

for folder in (UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Gaussian Blur Video Processor")


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "video.mp4").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "video.mp4"


def validate_params(num_points: int, blur_kernel: int, point_radius: int) -> None:
    if not 1 <= num_points <= 1000:
        raise HTTPException(status_code=400, detail="num_points must be between 1 and 1000.")
    if not 3 <= blur_kernel <= 151:
        raise HTTPException(status_code=400, detail="blur_kernel must be between 3 and 151.")
    if not 1 <= point_radius <= 200:
        raise HTTPException(status_code=400, detail="point_radius must be between 1 and 200.")


def validate_video_file(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported video extension: {ext}. Supported: {supported}")


def save_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    with destination.open("wb") as output:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            output.write(chunk)
    return total


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "supportedExtensions": sorted(SUPPORTED_VIDEO_EXTENSIONS),
        "maxUploadBytes": MAX_UPLOAD_BYTES,
    }


@app.post("/api/process")
def process_uploads(
    files: Annotated[list[UploadFile], File(description="One or more video files")],
    num_points: Annotated[int, Form()] = 50,
    blur_kernel: Annotated[int, Form()] = 21,
    point_radius: Annotated[int, Form()] = 8,
    use_cuda: Annotated[bool, Form()] = False,
    force_cpu: Annotated[bool, Form()] = False,
    device_id: Annotated[int, Form()] = 0,
):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one video file.")

    validate_params(num_points, blur_kernel, point_radius)

    processor = process_video_cuda if use_cuda else process_video_cpu
    results = []
    durations = []
    batch_id = uuid.uuid4().hex[:12]

    for index, upload in enumerate(files, start=1):
        safe_name = sanitize_filename(upload.filename or f"video_{index}.mp4")
        validate_video_file(safe_name)

        source_ext = Path(safe_name).suffix.lower()
        source_stem = Path(safe_name).stem
        input_path = UPLOAD_DIR / f"{batch_id}_{index}_{safe_name}"
        output_filename = f"{batch_id}_{index}_{source_stem}_blurred{source_ext}"
        output_path = OUTPUT_DIR / output_filename

        start_time = time.perf_counter()
        file_result = {
            "originalFilename": safe_name,
            "outputFilename": output_filename,
            "downloadUrl": None,
            "seconds": None,
            "ok": False,
            "error": None,
        }

        try:
            size = save_upload(upload, input_path)
            if size == 0:
                raise HTTPException(status_code=400, detail=f"{safe_name} is empty.")

            if use_cuda:
                ok = processor(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    num_points=num_points,
                    blur_kernel=blur_kernel,
                    point_radius=point_radius,
                    device_id=device_id,
                    force_cpu=force_cpu,
                )
            else:
                ok = processor(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    num_points=num_points,
                    blur_kernel=blur_kernel,
                    point_radius=point_radius,
                )

            elapsed = time.perf_counter() - start_time
            file_result["seconds"] = round(elapsed, 3)

            if ok and output_path.exists():
                durations.append(elapsed)
                file_result["ok"] = True
                file_result["downloadUrl"] = f"/api/download/{output_filename}"
            else:
                file_result["error"] = "Processing failed. Check backend logs for details."
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            file_result["seconds"] = round(elapsed, 3)
            file_result["error"] = str(exc)
        finally:
            upload.file.close()
            input_path.unlink(missing_ok=True)

        results.append(file_result)

    average_seconds = round(sum(durations) / len(durations), 3) if durations else None

    return {
        "ok": any(item["ok"] for item in results),
        "batchId": batch_id,
        "useCuda": use_cuda,
        "forceCpu": force_cpu,
        "deviceId": device_id,
        "results": results,
        "averageSeconds": average_seconds,
        "successCount": len(durations),
        "totalCount": len(results),
    }


@app.get("/api/download/{filename}")
def download_file(filename: str):
    safe_name = sanitize_filename(filename)
    file_path = OUTPUT_DIR / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path=file_path, filename=safe_name, media_type="application/octet-stream")


@app.head("/api/download/{filename}")
def download_file_head(filename: str):
    safe_name = sanitize_filename(filename)
    file_path = OUTPUT_DIR / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return Response(
        headers={
            "Content-Length": str(file_path.stat().st_size),
            "Content-Disposition": f'attachment; filename="{safe_name}"',
        },
        media_type="application/octet-stream",
    )


@app.get("/")
def root():
    return RedirectResponse(url="/app/")


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
