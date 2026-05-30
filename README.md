# OpenShorts.app

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

**Free & open source AI video platform** that turns long-form videos into viral-ready 9:16 shorts. Self-hosted with Docker. No watermarks, no limits, no auto-upload — **your clips stay local**.

---

## What's Different in This Fork?

This is a modified version of OpenShorts with key changes:

- **❌ No S3 Upload** — Clips are stored locally only, no cloud auto-publish
- **❌ No Upload-Post** — Social media auto-posting removed
- **✅ Podcast vs Gaming Detection** — AI auto-classifies video type (Podcast or Gaming)
- **✅ Gaming Mode** — Detects facecam region, composites 30% facecam + 70% gameplay vertically
- **✅ Face-Centered Tracking** — Camera always keeps detected faces in the center of frame
- **✅ Video Type Metadata** — Clip results include `video_type` field for downstream use

---

## Video Type Detection

### Podcast Mode
For talking-head content: podcasts, interviews, vlogs, tutorials, reactions.
- MediaPipe face detection + YOLOv8 fallback
- Face always centered horizontally in the 9:16 frame
- Smooth camera tracking with "CenteredCameraman" stabilization

### Gaming Mode
For gameplay content with facecam overlay.
- **Auto-detects facecam corner** (top-left, top-right, bottom-left, bottom-right)
- **Frame composite**: Top 30% = facecam (zoomed & centered on face), Bottom 70% = gameplay
- Face zoomed to fill ~60% of facecam panel height
- Blurred background fill when facecam is too small

### Detection Pipeline
1. **Heuristic** (fast) — Analyzes 8 frames: aspect ratio, corner face detection, color variance
2. **Gemini** (optional) — If heuristic confidence < 75%, sends 3 thumbnail frames + transcript snippet to Gemini 2.5 Flash for confirmation

---

## Features

### Clip Generator
- **Viral Moment Detection**: Google Gemini analyzes transcripts and scene boundaries to detect 3-15 high-potential moments
- **Smart 9:16 Cropping**: Dual-mode AI reframing — TRACK mode (face tracking) and GENERAL mode (blurred background)
- **Auto Subtitles**: faster-whisper with word-level timestamps, styled and burned into clips
- **AI Voice Dubbing**: ElevenLabs integration for 30+ languages with voice cloning
- **Hook Text Overlays**: AI-generated attention-grabbing text overlays
- **AI Video Effects**: Gemini-generated FFmpeg filters for professional effects

---

## Requirements

- **Docker & Docker Compose**
- **Google Gemini API Key** ([Free — get it here](https://aistudio.google.com/app/apikey)) — required
- **ElevenLabs API Key** ([Free tier](https://elevenlabs.io)) — optional, for voice dubbing

---

## Getting Started

### 1. Clone
```bash
git clone https://github.com/aldiboncel49-lgtm/openshorts.git
cd openshorts
```

### 2. Launch
```bash
docker compose up --build
```

### 3. Open Dashboard
Navigate to **`http://localhost:5175`**

1. Go to **Settings** and enter your Gemini API key
2. Upload a long-form video to generate viral shorts
3. Select video type: **Podcast** or **Gaming** (or let AI auto-detect)

---

## API Usage

### Process a video
```bash
curl -X POST http://localhost:8000/api/process \
  -H "X-Gemini-Key: YOUR_GEMINI_API_KEY" \
  -F "url=https://youtube.com/watch?v=XXXXX" \
  -F "acknowledged=true" \
  -F "force_type=gaming"
```

### Force video type
- `force_type=podcast` — Force podcast mode
- `force_type=gaming` — Force gaming mode
- Omit — Auto-detect via heuristics + Gemini

### Check status
```bash
curl http://localhost:8000/api/status/{job_id}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, google-genai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg |
| Frontend | React 18, Vite 4, Tailwind CSS 3.4 |
| AI APIs | Google Gemini 2.5 Flash |
| Infrastructure | Docker + Docker Compose |

---

## Technical Pipeline

1. **Ingest** — YouTube download (yt-dlp) or local upload
2. **Transcribe** — faster-whisper with word-level timestamps
3. **Detect** — PySceneDetect for scene boundaries
4. **Classify** — Podcast vs Gaming (heuristic + optional Gemini confirmation)
5. **Analyze** — Gemini identifies 3-15 viral moments (15-60s each)
6. **Extract** — FFmpeg precise clip cutting
7. **Reframe** — AI vertical cropping:
   - Podcast: face-centered tracking
   - Gaming: 30% facecam + 70% gameplay composite
8. **Effects** — Subtitles, hooks, AI video effects

---

## License

MIT License. Fork it, use it, modify it.
