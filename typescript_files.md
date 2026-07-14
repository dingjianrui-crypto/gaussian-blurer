# TypeScript files overview

This folder contains three TypeScript/TSX files that form a small web application around the video Gaussian blur script.

At a high level, the app lets a user upload a video in a browser, choose Gaussian blur parameters, send the video to a backend API, run the Python processor, and download the processed video.

## Main purpose

The TypeScript code provides a user interface and server API for `video_gaussian_blur.py`.

The Python file does the actual video processing. The TypeScript files make it easier to use by adding:

- a React page for uploading and previewing videos;
- sliders for changing blur settings;
- a backend route that receives the uploaded video;
- server-side code that runs the Python script;
- a download endpoint for the generated output video.

There is also some unrelated/demo Seedance2.0 video-generation API code in the server and routing code.

## File summary

## `App.tsx`

`App.tsx` defines the frontend route table for the React application.

It uses `react-router-dom` to map URL paths to React pages:

- `/` renders `Seed2Lab`.
- `/landing` renders `Home`.
- `/seed2` renders `Seed2Lab`.
- `/gaussian-blur` renders `GaussianBlur`.
- `/other` renders a simple placeholder page.

The most relevant route for this project is:

```tsx
<Route path="/gaussian-blur" element={<GaussianBlur />} />
```

That route opens the video Gaussian blur tool.

Important note: this file imports pages from paths such as `@/pages/GaussianBlur`. In this folder, `GaussianBlur.tsx` exists at the top level, so the actual project may normally expect this file to live under a `pages` directory or use a build alias.

## `GaussianBlur.tsx`

`GaussianBlur.tsx` is the main React user interface for the Gaussian blur video tool.

It lets the user:

- select or drag-and-drop a video file;
- preview the selected video in the browser;
- adjust the number of detected feature points;
- adjust the Gaussian blur kernel size;
- adjust the radius of each blurred point;
- submit the video for processing;
- download the processed video after the backend finishes.

### State values

The component stores UI state with React `useState`:

- `file`: the selected video file.
- `previewUrl`: a local browser preview URL for the uploaded file.
- `numPoints`: how many feature points to blur per frame.
- `blurKernel`: how strong the Gaussian blur should be.
- `pointRadius`: how large each blurred point should be.
- `status`: current processing state: `idle`, `loading`, `success`, or `error`.
- `statusText`: message shown to the user.
- `downloadUrl`: backend URL for downloading the processed video.
- `outputFilename`: suggested filename for the processed video.

### Upload behavior

The page supports both file picker upload and drag-and-drop upload.

When a video is selected, the component creates a preview URL using:

```ts
URL.createObjectURL(selectedFile)
```

That preview URL is shown inside a `<video controls>` element.

### Processing behavior

When the user clicks the process button, `handleProcess` creates a `FormData` request containing:

- the uploaded video file as `video`;
- `numPoints`;
- `blurKernel`;
- `pointRadius`.

It sends this request to:

```text
POST /api/gaussian-blur/process
```

If the backend returns success, the component stores the returned download URL and displays a download button.

### Visual design

The page uses:

- `framer-motion` for simple entrance animations;
- `lucide-react` icons;
- Tailwind-style utility classes for layout and styling.

The interface text is mostly Chinese. It presents the feature as a video effects tool that detects feature points and applies a soft Gaussian blur effect.

## `server.ts`

`server.ts` is an Express backend server.

It provides API routes for:

- health checking;
- Seedance2.0 demo/video-generation requests;
- Gaussian blur video upload, processing, and download.

The Gaussian blur routes are the parts that connect directly to the Python script.

### Server setup

The server imports and configures:

- `express` for API routing;
- `cors` for cross-origin requests;
- `dotenv` for environment variables;
- `multer` for video file uploads;
- Node `child_process.spawn` for running the Python script;
- Node `fs` and `path` for file management.

Uploaded videos are stored in:

```text
uploads/
```

Processed videos are stored in:

```text
outputs/
```

The upload limit is:

```text
200 MB
```

The server listens on:

```text
API_PORT
```

or defaults to:

```text
8787
```

### Gaussian blur processing endpoint

The main processing route is:

```text
POST /api/gaussian-blur/process
```

It expects a multipart form upload with a video file field named:

```text
video
```

It also accepts these parameters:

- `numPoints`;
- `blurKernel`;
- `pointRadius`.

After receiving the file, the server builds this command:

```bash
python video_gaussian_blur.py uploaded_file -o output_file -n numPoints -k blurKernel -r pointRadius
```

The server waits for the Python process to finish.

If processing succeeds, it returns JSON like:

```json
{
  "ok": true,
  "downloadUrl": "/api/gaussian-blur/download/example_blurred.mp4",
  "filename": "example_blurred.mp4"
}
```

If processing fails, it returns a `500` response with error details from the Python process.

The uploaded temporary input file is deleted after processing.

### Gaussian blur download endpoint

The download route is:

```text
GET /api/gaussian-blur/download/:filename
```

It looks for the generated file in the output directory and sends it as a download.

If the file does not exist, it returns:

```json
{
  "error": "文件不存在"
}
```

### Seedance2.0 routes

The server also includes routes that are separate from the Gaussian blur tool:

- `GET /api/health`
- `POST /api/seed2/examples/:exampleId`
- `POST /api/videos/generate`
- `GET /api/videos/:jobId`

These appear to support a Seedance2.0 demo or video-generation workflow. They are not required for the Gaussian blur video processing feature.

## End-to-end flow

The Gaussian blur feature works like this:

1. The user opens the React app at `/gaussian-blur`.
2. The user uploads a video.
3. `GaussianBlur.tsx` previews the video locally in the browser.
4. The user adjusts blur parameters with sliders.
5. The user clicks the process button.
6. The frontend sends the video and parameters to `POST /api/gaussian-blur/process`.
7. `server.ts` saves the uploaded video to `uploads/`.
8. `server.ts` runs `video_gaussian_blur.py` with the selected parameters.
9. The Python script writes a processed MP4 file to `outputs/`.
10. The backend returns a download URL.
11. The frontend shows a download button.
12. The user downloads the processed video from `/api/gaussian-blur/download/:filename`.

## What this app is for

This code is for turning the Python video processor into a browser-based tool.

Instead of running the Python script manually in a terminal, a user can upload a video, adjust parameters visually, process the video through the backend, and download the result from the browser.

It is best understood as a video effect / visual obfuscation demo, not as a true invisible watermarking system.

## Important limitations

- The backend depends on Python, OpenCV, and NumPy being installed.
- The server calls `python`, so the command must resolve correctly on the machine running the backend.
- The processed output does not preserve audio unless the Python script is changed to handle audio.
- Uploaded files are deleted after processing, but output files remain in `outputs/`.
- There is no authentication or permission system.
- The download endpoint accepts a filename parameter, so production code should add stricter filename validation.
- The frontend route imports assume a larger project structure with `@/pages/...` aliases.
- Some imported files such as `Home`, `Seed2Lab`, and `../shared/seed2Demo` are referenced but are not present in this folder.
