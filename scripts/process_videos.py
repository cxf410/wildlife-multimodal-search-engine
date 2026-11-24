"""Split wildlife videos into short searchable segments and thumbnails.

Put this file inside the project's scripts folder, then run:
    python scripts/process_videos.py

Dependency (already installed for the audio step):
    python -m pip install imageio-ffmpeg

This script does not modify the original videos in data/videos.
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
import sys
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:
    print("Missing dependency. Run: python -m pip install imageio-ffmpeg")
    sys.exit(1)

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
VIDEOS_FOLDER = PROJECT_FOLDER / "data" / "videos"
PROCESSED_FOLDER = PROJECT_FOLDER / "data" / "processed"
CLIPS_FOLDER = PROCESSED_FOLDER / "video_segments"
THUMBNAILS_FOLDER = PROCESSED_FOLDER / "video_frames"
MANIFEST_FILE = PROCESSED_FOLDER / "video_manifest.csv"

MAX_SEGMENT_SECONDS = 30
SUPPORTED_EXTENSIONS = {".webm", ".mp4", ".mov", ".mkv", ".avi", ".ogv"}

if not VIDEOS_FOLDER.is_dir():
    print(f"ERROR: Video folder not found: {VIDEOS_FOLDER}")
    sys.exit(1)

CLIPS_FOLDER.mkdir(parents=True, exist_ok=True)
THUMBNAILS_FOLDER.mkdir(parents=True, exist_ok=True)

try:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as exc:
    print(f"ERROR: FFmpeg could not be initialized: {exc}")
    sys.exit(1)


def run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *arguments],
        capture_output=True, text=True,
    )


def get_duration_seconds(source: Path) -> float:
    # FFmpeg can display the duration without decoding or modifying the video.
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-i", str(source)],
        capture_output=True, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError("FFmpeg could not read video duration")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        raise RuntimeError("Video has no measurable duration")
    return duration


videos = sorted(
    file for file in VIDEOS_FOLDER.iterdir()
    if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
)

rows = []
processed = 0
errors = 0

for video in videos:
    print(f"[VIDEO] {video.name}", flush=True)
    source_rows = []
    try:
        duration = get_duration_seconds(video)
        count = math.ceil(duration / MAX_SEGMENT_SECONDS)
        species = video.stem.removesuffix("_video")

        for index in range(count):
            start = index * MAX_SEGMENT_SECONDS
            segment_duration = min(MAX_SEGMENT_SECONDS, duration - start)
            if segment_duration <= 0.01:
                continue
            clip_name = f"{video.stem}_segment_{index + 1:03d}.mp4"
            frame_name = f"{video.stem}_segment_{index + 1:03d}.jpg"
            clip_path = CLIPS_FOLDER / clip_name
            frame_path = THUMBNAILS_FOLDER / frame_name

            if not clip_path.is_file() or clip_path.stat().st_size == 0:
                result = run_ffmpeg([
                    "-y", "-ss", f"{start:.3f}", "-i", str(video),
                    "-t", f"{segment_duration:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", "scale=640:360:force_original_aspect_ratio=decrease",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "29",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                    "-movflags", "+faststart", str(clip_path),
                ])
                if result.returncode != 0 or not clip_path.is_file() or clip_path.stat().st_size == 0:
                    clip_path.unlink(missing_ok=True)
                    raise RuntimeError(result.stderr.strip() or "FFmpeg could not create video segment")

            if not frame_path.is_file() or frame_path.stat().st_size == 0:
                frame_time = start + min(segment_duration / 2, max(segment_duration - 0.1, 0))
                result = run_ffmpeg([
                    "-y", "-ss", f"{frame_time:.3f}", "-i", str(video),
                    "-frames:v", "1",
                    "-vf", "scale=640:360:force_original_aspect_ratio=decrease",
                    "-q:v", "3", str(frame_path),
                ])
                if result.returncode != 0 or not frame_path.is_file() or frame_path.stat().st_size == 0:
                    frame_path.unlink(missing_ok=True)
                    raise RuntimeError(result.stderr.strip() or "FFmpeg could not extract a frame")

            source_rows.append({
                "segment_id": clip_path.stem,
                "species": species,
                "source_file": video.relative_to(PROJECT_FOLDER).as_posix(),
                "segment_file": clip_path.relative_to(PROJECT_FOLDER).as_posix(),
                "frame_file": frame_path.relative_to(PROJECT_FOLDER).as_posix(),
                "start_seconds": f"{start:.2f}",
                "end_seconds": f"{min(start + segment_duration, duration):.2f}",
                "duration_seconds": f"{segment_duration:.2f}",
            })

        rows.extend(source_rows)
        processed += 1
        print(f"  OK: {duration:.1f}s, {len(source_rows)} segment(s)", flush=True)
    except Exception as exc:
        errors += 1
        print(f"  ERROR: {exc}", flush=True)

columns = [
    "segment_id", "species", "source_file", "segment_file", "frame_file",
    "start_seconds", "end_seconds", "duration_seconds",
]
with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\nSUMMARY")
print(f"Video files found: {len(videos)}")
print(f"Video files processed: {processed}")
print(f"Segments created: {len(rows)}")
print(f"Frames extracted: {len(rows)}")
print(f"Errors: {errors}")
print(f"Manifest saved to: {MANIFEST_FILE}")

if errors:
    sys.exit(1)
