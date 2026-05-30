"""
OpenShorts - Modified app.py
Changes:
  1. S3 upload removed (no auto-publish)
  2. Upload-Post publish endpoint removed
  3. Gaming/Podcast video type passed through pipeline
  4. Clip results include video_type metadata
"""

import os
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 2048
JOB_RETENTION_SECONDS = 3600
DISABLE_YOUTUBE_URL = os.environ.get("DISABLE_YOUTUBE_URL", "false").lower() in ("1", "true", "yes")

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
thumbnail_sessions: Dict[str, Dict] = {}
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)
        for clip_path in glob.glob(os.path.join(root, f"{base_name}_clip_*.mp4")):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)
        return True
    except Exception:
        return False


async def cleanup_jobs():
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300)
            now = time.time()
            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                        os.remove(file_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


async def process_queue():
    print(f"🚀 Job Queue Worker started ({MAX_CONCURRENT_JOBS} slots).")
    while True:
        try:
            job_id = await job_queue.get()
            await concurrency_semaphore.acquire()
            asyncio.create_task(run_job_wrapper(job_id))
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)


async def run_job_wrapper(job_id):
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
        print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        concurrency_semaphore.release()
        job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_queue())
    asyncio.create_task(cleanup_jobs())
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")


def enqueue_output(out, job_id):
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()


async def run_job(job_id, job_data):
    """Executes the clip generation subprocess. NO S3/upload-post calls."""
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']

    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append("Job started.")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()

        while process.poll() is None:
            await asyncio.sleep(2)
            # Poll for partial clip results
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files and os.path.getsize(json_files[0]) > 0:
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                    base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
                    clips = data.get('shorts', [])
                    cost_analysis = data.get('cost_analysis')
                    video_type = data.get('video_type', 'PODCAST')
                    ready_clips = []
                    for i, clip in enumerate(clips):
                        clip_filename = f"{base_name}_clip_{i+1}.mp4"
                        clip_path = os.path.join(output_dir, clip_filename)
                        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                            clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                            ready_clips.append(clip)
                    if ready_clips:
                        jobs[job_id]['result'] = {
                            'clips': ready_clips,
                            'cost_analysis': cost_analysis,
                            'video_type': video_type
                        }
            except Exception:
                pass

        returncode = process.returncode

        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")
            # NOTE: No S3 upload. Clips stay local.

            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')
                video_type = data.get('video_type', 'PODCAST')
                for i, clip in enumerate(clips):
                    clip_filename = f"{base_name}_clip_{i+1}.mp4"
                    clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                jobs[job_id]['result'] = {
                    'clips': clips,
                    'cost_analysis': cost_analysis,
                    'video_type': video_type
                }
            else:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['logs'].append("No metadata file generated.")
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(f"Process failed (exit code {returncode})")

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(f"Execution error: {str(e)}")


@app.get("/api/config")
async def get_config():
    return {"youtubeUrlEnabled": not DISABLE_YOUTUBE_URL}


@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    acknowledged: Optional[str] = Form(None),
    force_type: Optional[str] = Form(None),  # 'podcast' | 'gaming' | None
):
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    ack_flag = str(acknowledged).lower() in ("1", "true", "yes")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        ack_flag = bool(body.get("acknowledged"))
        force_type = body.get("force_type")

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")
    if not ack_flag:
        raise HTTPException(status_code=400, detail="You must confirm you own the content.")
    if url and DISABLE_YOUTUBE_URL:
        raise HTTPException(status_code=403, detail="YouTube URL ingest is disabled.")

    client_ip = request.client.host if request.client else "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        client_ip = fwd.split(",")[0].strip()

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    cmd = ["python", "-u", "main.py"]
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key

    if url:
        cmd.extend(["-u", url])
    else:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)
        cmd.extend(["-i", input_path])

    cmd.extend(["-o", job_output_dir])

    # Pass force_type if provided
    if force_type in ("podcast", "gaming"):
        cmd.extend(["--force-type", force_type])

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
    }
    await job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result')
    }


# ── Editor, Subtitles, Hooks (local-only, no upload) ──────────────
from editor import VideoEditor
from subtitles import generate_srt, burn_subtitles, generate_srt_from_video
from hooks import add_hook_to_video
from translate import translate_video, get_supported_languages
from thumbnail import analyze_video_for_titles, refine_titles, generate_thumbnail, generate_youtube_description


class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None


@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    final_api_key = req.api_key or x_gemini_key or os.environ.get("GEMINI_API_KEY")
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    try:
        if req.input_filename:
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        edited_filename = f"edited_{filename}"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)

        def run_edit():
            editor = VideoEditor(api_key=final_api_key)
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            shutil.copy(input_path, safe_input_path)
            try:
                vid_file = editor.upload_video(safe_input_path)
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                        transcript = data.get('transcript')
                except Exception:
                    pass
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps,
                                                        width=width, height=height, transcript=transcript)
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                return filter_data
            finally:
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)

        return {
            "success": True,
            "new_video_url": f"/videos/{req.job_id}/{edited_filename}",
            "edit_plan": plan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom"
    font_size: int = 16
    font_name: str = "Verdana"
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 2
    bg_color: str = "#000000"
    bg_opacity: float = 0.0
    input_filename: Optional[str] = None


@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found.")
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
            base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
            filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {input_path}")

    srt_path = os.path.join(output_dir, f"subs_{req.clip_index}_{int(time.time())}.srt")
    output_filename = f"subtitled_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        is_dubbed = filename.startswith("translated_")
        if is_dubbed:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, generate_srt_from_video, input_path, srt_path)
        else:
            success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)

        if not success:
            raise HTTPException(status_code=400, detail="No words found for this clip.")

        def run_burn():
            burn_subtitles(input_path, srt_path, output_path,
                           alignment=req.position, fontsize=req.font_size,
                           font_name=req.font_name, font_color=req.font_color,
                           border_color=req.border_color, border_width=req.border_width,
                           bg_color=req.bg_color, bg_opacity=req.bg_opacity)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    return {"success": True, "new_video_url": f"/videos/{req.job_id}/{output_filename}"}


class HookRequest(BaseModel):
    job_id: str
    clip_index: int
    text: str
    input_filename: Optional[str] = None
    position: Optional[str] = "top"
    size: Optional[str] = "M"


@app.post("/api/hook")
async def add_hook(req: HookRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip_data = clips[req.clip_index]

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
            base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
            filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {input_path}")

    output_filename = f"hook_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    size_map = {"S": 0.8, "M": 1.0, "L": 1.3}
    font_scale = size_map.get(req.size, 1.0)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, add_hook_to_video, input_path, req.text, output_path, req.position, font_scale
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    return {"success": True, "new_video_url": f"/videos/{req.job_id}/{output_filename}"}


class TranslateRequest(BaseModel):
    job_id: str
    clip_index: int
    target_language: str
    source_language: Optional[str] = None
    input_filename: Optional[str] = None


@app.get("/api/translate/languages")
async def get_languages():
    return {"languages": get_supported_languages()}


@app.post("/api/translate")
async def translate_clip(
    req: TranslateRequest,
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key")
):
    if not x_elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing X-ElevenLabs-Key header")
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip_data = clips[req.clip_index]

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {input_path}")

    output_filename = f"translated_{req.target_language}_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, translate_video,
            input_path, output_path, req.target_language, req.source_language, x_elevenlabs_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "new_video_url": f"/videos/{req.job_id}/{output_filename}"}


# Thumbnail endpoints (unchanged, no upload)
class ThumbnailAnalyzeRequest(BaseModel):
    job_id: str
    clip_index: int
    input_filename: Optional[str] = None


@app.post("/api/thumbnail/analyze")
async def analyze_clip_for_thumbnail(
    req: ThumbnailAnalyzeRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    final_api_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        clip = job['result']['clips'][req.clip_index]
        filename = clip['video_url'].split('/')[-1]

    input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")

    session_id = str(uuid.uuid4())

    def run_analyze():
        result = analyze_video_for_titles(input_path, final_api_key)
        thumbnail_sessions[session_id] = {
            "video_path": input_path,
            "api_key": final_api_key,
            "titles": result.get("titles", []),
            "scene_description": result.get("scene_description", ""),
            "chat_history": []
        }
        return result

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_analyze)
    return {"session_id": session_id, **result}


class ThumbnailGenerateRequest(BaseModel):
    session_id: str
    selected_title: str
    face_image_b64: Optional[str] = None
    background_prompt: Optional[str] = None


@app.post("/api/thumbnail/generate")
async def generate_thumbnail_endpoint(req: ThumbnailGenerateRequest):
    if req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = thumbnail_sessions[req.session_id]

    def run_generate():
        return generate_thumbnail(
            video_path=session["video_path"],
            api_key=session["api_key"],
            selected_title=req.selected_title,
            scene_description=session.get("scene_description", ""),
            face_image_b64=req.face_image_b64,
            background_prompt=req.background_prompt
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_generate)
    if result.get("thumbnail_b64"):
        thumb_filename = f"thumb_{req.session_id}.jpg"
        thumb_path = os.path.join(THUMBNAILS_DIR, thumb_filename)
        import base64
        with open(thumb_path, 'wb') as f:
            f.write(base64.b64decode(result["thumbnail_b64"]))
        result["thumbnail_url"] = f"/thumbnails/{thumb_filename}"
    return result


class DescriptionRequest(BaseModel):
    job_id: str
    clip_index: int


@app.post("/api/youtube/description")
async def generate_description(
    req: DescriptionRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    final_api_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    transcript = None
    clip_data = {}
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        transcript = data.get('transcript')
        clips = data.get('shorts', [])
        if req.clip_index < len(clips):
            clip_data = clips[req.clip_index]

    def run_desc():
        return generate_youtube_description(
            transcript=transcript,
            clip_start=clip_data.get('start', 0),
            clip_end=clip_data.get('end', 0),
            title=clip_data.get('video_title_for_youtube_short', ''),
            api_key=final_api_key
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_desc)
    return result


@app.get("/api/clip/{job_id}/{clip_index}/transcript")
async def get_clip_transcript(job_id: str, clip_index: int):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found")
    clips = data.get('shorts', [])
    if clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip_data = clips[clip_index]
    clip_start = clip_data.get('start', 0)
    clip_end = clip_data.get('end', 0)

    captions = []
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                captions.append({
                    "text": word_info.get('word', '').strip(),
                    "startMs": int((max(0, word_info['start'] - clip_start)) * 1000),
                    "endMs": int((max(0, word_info['end'] - clip_start)) * 1000),
                })

    return {
        "captions": captions,
        "durationSec": clip_end - clip_start,
        "language": transcript.get('language', 'en'),
    }


# Remotion render proxy
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL", "http://renderer:3100")


@app.post("/api/render")
async def proxy_render(request: Request):
    import httpx
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{RENDER_SERVICE_URL}/render", json=body)
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")


@app.get("/api/render/{render_id}")
async def proxy_render_status(render_id: str):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RENDER_SERVICE_URL}/render/{render_id}")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")
