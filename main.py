"""
OpenShorts - Modified Version
Changes from original:
  1. NO auto-upload (S3 / Upload-Post removed entirely)
  2. Face detection centered (face always in the middle of frame)
  3. Video type detection: PODCAST vs GAMING via Gemini + heuristics
  4. Gaming mode: detects facecam region, composites 30% facecam + 70% gameplay vertically
"""

import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
from google import genai
from dotenv import load_dotenv
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video.
- Only NUMBERS with decimal point, up to 3 decimals.
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "shorts": [
    {{
      "start": <number in seconds>,
      "end": <number in seconds>,
      "video_description_for_tiktok": "<TikTok description>",
      "video_description_for_instagram": "<Instagram description>",
      "video_title_for_youtube_short": "<YouTube Short title, 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words). SAME LANGUAGE AS TRANSCRIPT.>"
    }}
  ]
}}
"""

VIDEO_TYPE_PROMPT = """
You are a video content classifier. Analyze the video thumbnail frames and transcript to determine if this is:
1. PODCAST - People talking, interview, discussion, commentary, reaction, vlog, tutorial where the person is visible
2. GAMING - Gameplay footage (game UI, HUD elements, game graphics visible), possibly with a facecam in a corner

Look for these GAMING signals:
- Game UI elements (health bars, minimaps, ammo counters, score displays)
- Game graphics / 3D environments
- Small camera overlay in corner (facecam)
- Game-related vocabulary in transcript

Look for these PODCAST signals:
- People sitting/standing talking directly to camera
- Studio/room background
- Interview-style setup
- Multiple speakers facing each other

Return ONLY valid JSON:
{{
  "video_type": "PODCAST" or "GAMING",
  "confidence": 0.0 to 1.0,
  "reasoning": "<brief explanation>"
}}
"""

# Load models once
model = YOLO('yolov8n.pt')
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# ─────────────────────────────────────────────────────────────────
# VIDEO TYPE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_video_type_heuristic(video_path: str) -> dict:
    """
    Heuristic-based video type detection without Gemini (fast fallback).
    Samples several frames and looks for gaming HUD indicators.
    Returns {"video_type": "PODCAST"|"GAMING", "confidence": float}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_type": "PODCAST", "confidence": 0.5}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample 8 frames spread across the video
    sample_indices = [int(total_frames * i / 8) for i in range(1, 8)]

    gaming_score = 0
    podcast_score = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Signal 1: Check if video is landscape (wide) — gaming is usually 16:9
        aspect = width / height if height > 0 else 1.0
        if aspect > 1.3:  # landscape
            gaming_score += 1

        # Signal 2: Look for faces ONLY in corners → facecam indicator
        corner_size = int(min(width, height) * 0.3)
        corners = [
            frame[:corner_size, :corner_size],          # top-left
            frame[:corner_size, -corner_size:],         # top-right
            frame[-corner_size:, :corner_size],         # bottom-left
            frame[-corner_size:, -corner_size:],        # bottom-right
        ]
        full_frame_faces = detect_face_candidates(frame)
        corner_faces = 0
        for corner in corners:
            if corner.shape[0] > 0 and corner.shape[1] > 0:
                corner_faces += len(detect_face_candidates(corner))

        if full_frame_faces:
            if corner_faces > 0 and len(full_frame_faces) <= 2:
                # Face mostly in corner → gaming facecam
                gaming_score += 2
            else:
                # Face fills the frame → podcast
                podcast_score += 2

        # Signal 3: High color variance in non-face areas (game graphics are colorful)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = gray.var()
        if variance > 1500:
            gaming_score += 1
        else:
            podcast_score += 1

    cap.release()

    total = gaming_score + podcast_score
    if total == 0:
        return {"video_type": "PODCAST", "confidence": 0.5}

    gaming_ratio = gaming_score / total
    if gaming_ratio > 0.55:
        return {"video_type": "GAMING", "confidence": gaming_ratio}
    else:
        return {"video_type": "PODCAST", "confidence": 1.0 - gaming_ratio}


def detect_video_type(video_path: str, transcript: dict = None, api_key: str = None) -> dict:
    """
    Detect whether video is PODCAST or GAMING.
    Uses heuristics first, optionally confirms with Gemini if api_key provided.
    """
    print("🎮 Detecting video type (Podcast vs Gaming)...")
    result = detect_video_type_heuristic(video_path)
    print(f"  Heuristic result: {result['video_type']} (confidence: {result['confidence']:.2f})")

    # If heuristic is not confident enough and we have an API key, use Gemini
    if api_key and result["confidence"] < 0.75:
        try:
            client = genai.Client(api_key=api_key)
            # Sample 3 frames as base64 for Gemini
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_b64 = []
            for i in [0.1, 0.5, 0.8]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i))
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    import base64
                    frames_b64.append(base64.b64encode(buf).decode())
            cap.release()

            transcript_snippet = ""
            if transcript:
                transcript_snippet = transcript.get("text", "")[:500]

            prompt = VIDEO_TYPE_PROMPT
            if transcript_snippet:
                prompt += f"\n\nTRANSCRIPT SNIPPET: {transcript_snippet}"

            contents = []
            for b64 in frames_b64:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64
                    }
                })
            contents.append({"text": prompt})

            from google.genai import types as gtypes
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=gtypes.GenerateContentConfig(response_mime_type="application/json")
            )
            text = response.text.strip().strip("```json").strip("```").strip()
            gemini_result = json.loads(text)
            result = {
                "video_type": gemini_result.get("video_type", result["video_type"]),
                "confidence": gemini_result.get("confidence", result["confidence"]),
                "reasoning": gemini_result.get("reasoning", "")
            }
            print(f"  Gemini refined: {result['video_type']} (confidence: {result['confidence']:.2f})")
            if result.get("reasoning"):
                print(f"  Reason: {result['reasoning']}")
        except Exception as e:
            print(f"  ⚠️ Gemini type detection failed, using heuristic: {e}")

    return result


# ─────────────────────────────────────────────────────────────────
# GAMING: FACECAM DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_facecam_region(video_path: str) -> dict:
    """
    Detect where the facecam overlay is located in a gaming video.
    Returns {"corner": "top-left"|"top-right"|"bottom-left"|"bottom-right"|"none",
             "rel_w": float, "rel_h": float}  (relative to frame dimensions)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    corner_counts = {"top-left": 0, "top-right": 0, "bottom-left": 0, "bottom-right": 0}
    sample_size = int(min(width, height) * 0.35)  # check 35% of smallest dimension

    for i in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 1) / 11))
        ret, frame = cap.read()
        if not ret:
            continue

        # Check each corner for face presence
        corners_map = {
            "top-left":     frame[:sample_size, :sample_size],
            "top-right":    frame[:sample_size, width - sample_size:],
            "bottom-left":  frame[height - sample_size:, :sample_size],
            "bottom-right": frame[height - sample_size:, width - sample_size:],
        }
        for corner_name, patch in corners_map.items():
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                faces = detect_face_candidates(patch)
                if faces:
                    corner_counts[corner_name] += 1

    cap.release()

    best_corner = max(corner_counts, key=corner_counts.get)
    max_count = corner_counts[best_corner]

    if max_count < 3:  # not enough frames had a face in any corner
        return {"corner": "none", "rel_w": 0.3, "rel_h": 0.3}

    # Estimate facecam size: sample a frame and find the face bounding box
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.5))
    ret, frame = cap.read()
    cap.release()

    rel_w = 0.3  # default 30% of width
    rel_h = 0.3  # default 30% of height

    if ret:
        patch = None
        if best_corner == "top-left":
            patch = frame[:sample_size, :sample_size]
        elif best_corner == "top-right":
            patch = frame[:sample_size, width - sample_size:]
        elif best_corner == "bottom-left":
            patch = frame[height - sample_size:, :sample_size]
        elif best_corner == "bottom-right":
            patch = frame[height - sample_size:, width - sample_size:]

        if patch is not None:
            faces = detect_face_candidates(patch)
            if faces:
                fx, fy, fw, fh = faces[0]['box']
                # Estimate: facecam region is roughly 2x the face size
                fc_w = min(fw * 2.5, sample_size)
                fc_h = min(fh * 2.5, sample_size)
                rel_w = fc_w / width
                rel_h = fc_h / height

    print(f"  📹 Facecam detected at: {best_corner} (~{rel_w:.0%}w x {rel_h:.0%}h)")
    return {"corner": best_corner, "rel_w": rel_w, "rel_h": rel_h}


# ─────────────────────────────────────────────────────────────────
# GAMING: COMPOSITE FRAME (30% facecam top + 70% gameplay bottom)
# ─────────────────────────────────────────────────────────────────

def create_gaming_frame(frame, output_width, output_height, facecam_info):
    """
    Creates vertical gaming frame:
    - TOP 30%: Facecam (face centered, blurred background fill)
    - BOTTOM 70%: Gameplay (cropped center of remaining frame)

    facecam_info = {"corner": str, "rel_w": float, "rel_h": float}
    """
    orig_h, orig_w = frame.shape[:2]

    facecam_h = int(output_height * 0.30)
    gameplay_h = output_height - facecam_h

    corner = facecam_info.get("corner", "none")

    # ── Facecam panel ──────────────────────────────────────────────
    if corner == "none":
        # No facecam found → use general blurred fill for top panel
        facecam_panel = create_general_frame(frame, output_width, facecam_h)
    else:
        fc_rel_w = facecam_info.get("rel_w", 0.30)
        fc_rel_h = facecam_info.get("rel_h", 0.30)

        fc_px_w = int(orig_w * fc_rel_w)
        fc_px_h = int(orig_h * fc_rel_h)
        fc_px_w = max(fc_px_w, 60)
        fc_px_h = max(fc_px_h, 60)

        # Extract facecam region
        if corner == "top-left":
            fc_crop = frame[:fc_px_h, :fc_px_w]
        elif corner == "top-right":
            fc_crop = frame[:fc_px_h, orig_w - fc_px_w:]
        elif corner == "bottom-left":
            fc_crop = frame[orig_h - fc_px_h:, :fc_px_w]
        else:  # bottom-right
            fc_crop = frame[orig_h - fc_px_h:, orig_w - fc_px_w:]

        if fc_crop.shape[0] == 0 or fc_crop.shape[1] == 0:
            facecam_panel = create_general_frame(frame, output_width, facecam_h)
        else:
            facecam_panel = _center_face_in_panel(fc_crop, output_width, facecam_h)

    # ── Gameplay panel ─────────────────────────────────────────────
    # Remove facecam region from gameplay crop (use center of frame)
    gameplay_crop = _extract_gameplay_region(frame, corner, facecam_info)
    gameplay_panel = _scale_to_fill(gameplay_crop, output_width, gameplay_h)

    # ── Stack vertically ───────────────────────────────────────────
    combined = np.vstack([facecam_panel, gameplay_panel])
    return combined


def _center_face_in_panel(face_crop, panel_w, panel_h):
    """Scale the facecam crop to fill the panel, centering it."""
    fc_h, fc_w = face_crop.shape[:2]

    # Find faces and center on the face
    faces = detect_face_candidates(face_crop)
    if faces:
        fx, fy, fw, fh = faces[0]['box']
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        # Compute zoom so face takes up ~60% of panel height
        target_face_h = panel_h * 0.6
        zoom = target_face_h / max(fh, 1)
        zoom = min(zoom, 4.0)  # cap zoom

        new_w = int(fc_w * zoom)
        new_h = int(fc_h * zoom)
        scaled = cv2.resize(face_crop, (new_w, new_h))

        # Center on face
        new_face_cx = int(face_cx * zoom)
        new_face_cy = int(face_cy * zoom)

        x_start = max(0, new_face_cx - panel_w // 2)
        y_start = max(0, new_face_cy - panel_h // 2)
        x_end = x_start + panel_w
        y_end = y_start + panel_h

        # Clamp
        if x_end > new_w:
            x_start = max(0, new_w - panel_w)
            x_end = new_w
        if y_end > new_h:
            y_start = max(0, new_h - panel_h)
            y_end = new_h

        cropped = scaled[y_start:y_end, x_start:x_end]

        # Pad if needed
        if cropped.shape[1] < panel_w or cropped.shape[0] < panel_h:
            canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cy = (panel_h - cropped.shape[0]) // 2
            cx = (panel_w - cropped.shape[1]) // 2
            canvas[cy:cy + cropped.shape[0], cx:cx + cropped.shape[1]] = cropped

            # Blur background
            bg = cv2.GaussianBlur(cv2.resize(face_crop, (panel_w, panel_h)), (51, 51), 0)
            mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
            mask[cy:cy + cropped.shape[0], cx:cx + cropped.shape[1]] = 255
            mask_3 = cv2.merge([mask, mask, mask])
            result = np.where(mask_3 > 0, canvas, bg)
            return result

        return cv2.resize(cropped, (panel_w, panel_h))

    # No face found: just scale to fill
    return _scale_to_fill(face_crop, panel_w, panel_h)


def _extract_gameplay_region(frame, facecam_corner, facecam_info):
    """Extract the gameplay portion (center of frame, avoiding facecam corner)."""
    orig_h, orig_w = frame.shape[:2]

    # Use full frame as source but focus center
    # The gameplay content is usually in the center, so just use the whole frame
    # and let _scale_to_fill crop the center
    return frame


def _scale_to_fill(frame, target_w, target_h):
    """Scale frame to fill target dimensions (center crop, no letterboxing)."""
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale_w = target_w / src_w
    scale_h = target_h / src_h
    scale = max(scale_w, scale_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    x_off = (new_w - target_w) // 2
    y_off = (new_h - target_h) // 2
    cropped = resized[y_off:y_off + target_h, x_off:x_off + target_w]

    # Ensure exact size
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(cropped, (target_w, target_h))

    return cropped


# ─────────────────────────────────────────────────────────────────
# PODCAST: CENTERED FACE CAMERAMAN
# ─────────────────────────────────────────────────────────────────

class CenteredCameraman:
    """
    Smooth camera that always tries to keep the detected face
    in the horizontal CENTER of the output frame.
    Unlike the original which had a 'safe zone' that allowed drift,
    this version always pulls toward center.
    """

    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height

        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Crop dimensions
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
            self.crop_width = video_width
            self.crop_height = int(self.crop_width / ASPECT_RATIO)

        self.smooth_factor = 0.08  # lerp speed (lower = smoother/slower)

    def update_target(self, face_box):
        if face_box:
            x, y, w, h = face_box
            # Target: center of the face horizontally
            self.target_center_x = x + w / 2

    def get_crop_box(self, force_snap=False):
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            # Smooth lerp — always move toward target (no dead zone)
            diff = self.target_center_x - self.current_center_x
            self.current_center_x += diff * self.smooth_factor

        # Clamp
        half_crop = self.crop_width / 2
        self.current_center_x = max(half_crop, min(self.video_width - half_crop, self.current_center_x))

        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)

        return x1, 0, x2, self.video_height


class SpeakerTracker:
    """Tracks speakers to prevent rapid switching."""

    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.stabilization_threshold = stabilization_frames
        self.switch_cooldown = cooldown_frames
        self.last_switch_frame = -1000
        self.next_id = 0
        self.known_faces = []

    def get_target(self, face_candidates, frame_number, width):
        current_candidates = []
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            best_match_id = -1
            min_dist = width * 0.15

            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30:
                    continue
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']

            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1

            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            current_candidates.append({'id': best_match_id, 'box': face['box'], 'score': face['score']})

        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand['id']
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        if not current_candidates:
            return None

        best_candidate = None
        max_score = -1
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            if pid == self.active_speaker_id:
                total_score *= 3.0
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate:
            target_id = best_candidate['id']
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']

            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']

        return None


# ─────────────────────────────────────────────────────────────────
# FACE DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────

def detect_face_candidates(frame):
    """Returns list of all detected faces using MediaPipe FaceDetection."""
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        return []
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    candidates = []
    if not results.detections:
        return []
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        candidates.append({'box': [x, y, w, h], 'score': w * h})
    return candidates


def detect_person_yolo(frame):
    """Fallback: Detect largest person using YOLO."""
    results = model(frame, verbose=False, classes=[0])
    if not results:
        return None
    best_box = None
    max_area = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area > max_area:
                max_area = area
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
    return best_box


def create_general_frame(frame, output_width, output_height):
    """Creates a 'General Shot' frame: blurred background + fit-width foreground."""
    orig_h, orig_w = frame.shape[:2]

    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height))
    start_x = max(0, (bg_w - output_width) // 2)
    background = bg_resized[:, start_x:start_x + output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
    background = cv2.GaussianBlur(background, (51, 51), 0)

    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h))

    y_offset = (output_height - fg_h) // 2
    final_frame = background.copy()
    final_frame[y_offset:y_offset + fg_h, :] = foreground
    return final_frame


# ─────────────────────────────────────────────────────────────────
# SCENE ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_scenes_strategy(video_path, scenes):
    """Analyze each scene → TRACK or GENERAL."""
    cap = cv2.VideoCapture(video_path)
    strategies = []
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    for start, end in tqdm(scenes, desc=" Analyzing Scenes"):
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5,
        ]
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            face_counts.append(len(detect_face_candidates(frame)))

        avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0

        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()
    return strategies


def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


# ─────────────────────────────────────────────────────────────────
# PODCAST VERTICAL PROCESSING
# ─────────────────────────────────────────────────────────────────

def process_video_to_vertical_podcast(input_video, final_output_video):
    """
    Podcast mode: standard face-tracked vertical crop.
    Face is always centered horizontally using CenteredCameraman.
    """
    script_start_time = time.time()

    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    for p in [temp_video_output, temp_audio_output, final_output_video]:
        if os.path.exists(p):
            os.remove(p)

    print(f"🎬 [PODCAST] Processing: {input_video}")
    print(" Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)

    if not scenes:
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f" ✅ Found {len(scenes)} scenes.")

    original_width, original_height = get_video_resolution(input_video)
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    cameraman = CenteredCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height)

    print(" 🤖 Step 3: Analyzing scenes...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)

    print(" ✂️ Step 4: Processing frames...")
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '23', '-an', temp_video_output
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    current_scene_index = 0

    scene_boundaries = [(s.get_frames(), e.get_frames()) for s, e in scenes]
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc=" Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scene_boundaries):
                _, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1

            current_strategy = (scene_strategies[current_scene_index]
                                if current_scene_index < len(scene_strategies) else 'TRACK')

            if current_strategy == 'GENERAL':
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
            else:
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)

    ffmpeg_process.stdin.close()
    ffmpeg_process.stderr.read()
    ffmpeg_process.wait()
    cap.release()

    return _merge_audio(input_video, temp_video_output, temp_audio_output, final_output_video)


# ─────────────────────────────────────────────────────────────────
# GAMING VERTICAL PROCESSING
# ─────────────────────────────────────────────────────────────────

def process_video_to_vertical_gaming(input_video, final_output_video, facecam_info):
    """
    Gaming mode: 30% facecam (top) + 70% gameplay (bottom).
    Facecam face is centered. Gameplay is center-cropped.
    """
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    for p in [temp_video_output, temp_audio_output, final_output_video]:
        if os.path.exists(p):
            os.remove(p)

    print(f"🎮 [GAMING] Processing: {input_video}")
    print(f"   Facecam at: {facecam_info['corner']}")

    original_width, original_height = get_video_resolution(input_video)
    _, fps = detect_scenes(input_video)  # reuse scene detector just for fps
    cap_fps = cv2.VideoCapture(input_video)
    fps = cap_fps.get(cv2.CAP_PROP_FPS) or 30
    cap_fps.release()

    # Output: 9:16 vertical
    OUTPUT_HEIGHT = max(original_height, 1920)
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    print(f" ✂️ Processing frames ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})...")
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '23', '-an', temp_video_output
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=" Gaming frames", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = create_gaming_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT, facecam_info)
            ffmpeg_process.stdin.write(output_frame.tobytes())
            pbar.update(1)

    ffmpeg_process.stdin.close()
    ffmpeg_process.stderr.read()
    ffmpeg_process.wait()
    cap.release()

    return _merge_audio(input_video, temp_video_output, temp_audio_output, final_output_video)


# ─────────────────────────────────────────────────────────────────
# SHARED AUDIO MERGE
# ─────────────────────────────────────────────────────────────────

def _merge_audio(input_video, temp_video_output, temp_audio_output, final_output_video):
    print("\n 🔊 Extracting audio...")
    audio_cmd = ['ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output]
    try:
        subprocess.run(audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("  ❌ Audio extraction failed (no audio?). Proceeding without audio.")

    print(" ✨ Merging video + audio...")
    if os.path.exists(temp_audio_output):
        merge_cmd = ['ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
                     '-c:v', 'copy', '-c:a', 'copy', final_output_video]
    else:
        merge_cmd = ['ffmpeg', '-y', '-i', temp_video_output, '-c:v', 'copy', final_output_video]

    try:
        subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f" ✅ Saved: {final_output_video}")
    except subprocess.CalledProcessError as e:
        print(f" ❌ Merge failed: {e.stderr.decode()}")
        return False

    for p in [temp_video_output, temp_audio_output]:
        if os.path.exists(p):
            os.remove(p)

    return True


# ─────────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def process_video_to_vertical(input_video, final_output_video,
                               video_type_result=None, api_key=None):
    """
    Master dispatcher: detects video type then routes to appropriate processor.
    """
    if video_type_result is None:
        video_type_result = detect_video_type(input_video, api_key=api_key)

    vtype = video_type_result.get("video_type", "PODCAST")
    print(f"\n🎬 Video type: {vtype}")

    if vtype == "GAMING":
        facecam_info = detect_facecam_region(input_video)
        return process_video_to_vertical_gaming(input_video, final_output_video, facecam_info)
    else:
        return process_video_to_vertical_podcast(input_video, final_output_video)


# ─────────────────────────────────────────────────────────────────
# TRANSCRIPTION & GEMINI CLIP DETECTION
# ─────────────────────────────────────────────────────────────────

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:100]


def download_youtube_video(url, output_dir="."):
    print(f"📥 Downloading from YouTube...")
    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
        except Exception:
            cookies_path = None
    else:
        cookies_path = None

    _COMMON_YDL_OPTS = {
        'quiet': False, 'verbose': True, 'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
        'nocheckcertificate': True, 'cachedir': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['tv_embed', 'android', 'mweb', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        },
    }

    with yt_dlp.YoutubeDL(_COMMON_YDL_OPTS) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info.get('title', 'youtube_video')
        sanitized_title = sanitize_filename(video_title)

    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    ydl_opts = {
        **_COMMON_YDL_OPTS,
        'format': 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'overwrites': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    print(f"✅ Downloaded: {downloaded_file}")
    return downloaded_file, sanitized_title


def transcribe_video(video_path):
    print("🎙️ Transcribing with Faster-Whisper...")
    from faster_whisper import WhisperModel
    model_w = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model_w.transcribe(video_path, word_timestamps=True)
    print(f"  Language: {info.language} ({info.language_probability:.2f})")

    transcript_segments = []
    full_text = ""
    for segment in segments:
        print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        seg_dict = {
            'text': segment.text, 'start': segment.start, 'end': segment.end, 'words': []
        }
        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word, 'start': word.start,
                    'end': word.end, 'probability': word.probability
                })
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "

    return {'text': full_text.strip(), 'segments': transcript_segments, 'language': info.language}


def get_viral_clips(transcript_result, video_duration):
    print("🤖 Analyzing with Gemini for viral clips...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'

    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        transcript_text=json.dumps(transcript_result['text']),
        words_json=json.dumps(words)
    )

    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        result_json = json.loads(text)

        # Cost info (informational only, no upload)
        try:
            usage = response.usage_metadata
            if usage:
                in_cost = (usage.prompt_token_count / 1_000_000) * 0.10
                out_cost = (usage.candidates_token_count / 1_000_000) * 0.40
                result_json['cost_analysis'] = {
                    "input_tokens": usage.prompt_token_count,
                    "output_tokens": usage.candidates_token_count,
                    "total_cost": in_cost + out_cost,
                    "model": model_name
                }
                print(f"💰 Estimated cost: ${in_cost + out_cost:.6f}")
        except Exception:
            pass

        return result_json
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenShorts - Clip Generator (no auto-upload).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    parser.add_argument('-o', '--output', type=str, help="Output directory or file.")
    parser.add_argument('--keep-original', action='store_true', help="Keep downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis, convert whole video.")
    parser.add_argument('--force-type', choices=['podcast', 'gaming'], default=None,
                        help="Force video type instead of auto-detecting.")
    args = parser.parse_args()

    def _ensure_dir(path):
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    api_key = os.getenv("GEMINI_API_KEY")

    # 1. Get Input
    if args.url:
        output_dir = _ensure_dir(args.output) if args.output else "."
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        if args.output:
            if os.path.isdir(args.output):
                output_dir = args.output
            else:
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video) or "."
        else:
            output_dir = os.path.dirname(input_video) or "."
        _ensure_dir(output_dir)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Detect video type
    if args.force_type:
        video_type_result = {"video_type": args.force_type.upper(), "confidence": 1.0}
    else:
        video_type_result = detect_video_type(input_video, api_key=api_key)

    print(f"\n🎯 Video type determined: {video_type_result['video_type']}")

    # 3. If skip-analysis: process whole video
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if (args.output and not os.path.isdir(args.output)) \
            else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        process_video_to_vertical(input_video, output_file, video_type_result, api_key)
    else:
        # 4. Transcribe
        transcript = transcribe_video(input_video)

        # 5. Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 6. Gemini clip detection
        clips_data = get_viral_clips(transcript, duration)

        if not clips_data or 'shorts' not in clips_data:
            print("❌ Failed to identify clips. Converting whole video as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
            process_video_to_vertical(input_video, output_file, video_type_result, api_key)
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")

            clips_data['transcript'] = transcript
            clips_data['video_type'] = video_type_result['video_type']

            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f" Saved metadata: {metadata_file}")

            # Pre-detect facecam if gaming (once, reused for all clips)
            facecam_info = None
            if video_type_result['video_type'] == "GAMING":
                facecam_info = detect_facecam_region(input_video)

            # 7. Process each clip
            for i, clip in enumerate(clips_data['shorts']):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Clip {i+1}/{len(clips_data['shorts'])}: {start}s → {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'N/A')}")

                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)

                # Cut clip
                cut_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start), '-to', str(end),
                    '-i', input_video,
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    '-c:a', 'aac',
                    clip_temp_path
                ]
                subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                if not os.path.exists(clip_temp_path):
                    print(f"  ❌ Failed to cut clip {i+1}")
                    continue

                # Process to vertical
                if video_type_result['video_type'] == "GAMING" and facecam_info:
                    # For gaming clips, use the pre-detected facecam info
                    success = process_video_to_vertical_gaming(clip_temp_path, clip_final_path, facecam_info)
                else:
                    success = process_video_to_vertical_podcast(clip_temp_path, clip_final_path)

                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

                if success:
                    clip['output_file'] = clip_final_path
                    print(f"  ✅ Clip saved: {clip_final_path}")
                else:
                    print(f"  ❌ Failed to process clip {i+1}")

            # Update metadata with output paths
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)

            print(f"\n✅ All done! Clips saved to: {output_dir}")
            print(f"   Metadata: {metadata_file}")
            # NOTE: No auto-upload. Files are local only.
