"""Prepare bird recordings for multimodal search.

Run from the project folder:
    python .\scripts\process_audio.py

Dependency:
    python -m pip install imageio-ffmpeg

The original recordings in data/audio are never modified.
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:
    print("Missing dependency. Run: python -m pip install imageio-ffmpeg")
    sys.exit(1)

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
AUDIO_FOLDER = PROJECT_FOLDER / "data" / "audio"
SEGMENTS_FOLDER = PROJECT_FOLDER / "data" / "processed" / "audio_segments"
MANIFEST_FILE = PROJECT_FOLDER / "data" / "processed" / "audio_manifest.csv"

SAMPLE_RATE = 16000
MAX_SEGMENT_SECONDS = 30
SUPPORTED_EXTENSIONS = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".flac", ".aac"}

SEGMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

if not AUDIO_FOLDER.is_dir():
    print(f"ERROR: Audio folder not found: {AUDIO_FOLDER}")
    sys.exit(1)

recordings = sorted(
    p for p in AUDIO_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
)

try:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as exc:
    print(f"ERROR: FFmpeg could not be initialized: {exc}")
    sys.exit(1)

rows = []
processed = 0
errors = 0

for recording in recordings:
    print(f"[AUDIO] {recording.name}")
    try:
        with tempfile.TemporaryDirectory(prefix="bird_audio_") as temp_dir:
            converted = Path(temp_dir) / "decoded.wav"
            command = [
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(recording), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le", str(converted),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "FFmpeg could not decode this file")

            with wave.open(str(converted), "rb") as source:
                frame_count = source.getnframes()
                if frame_count <= 0:
                    raise RuntimeError("The audio file has no samples")
                rate = source.getframerate()
                channels = source.getnchannels()
                width = source.getsampwidth()

                num_segments = max(1, math.ceil(frame_count / (rate * MAX_SEGMENT_SECONDS)))
                species = recording.stem.removesuffix("_audio")
                # Equal-length segments avoid an extremely short final fragment.
                segment_base = frame_count // num_segments
                extra_frames = frame_count % num_segments
                position = 0

                # Clear older segments from previous runs of this source only.
                for old_file in SEGMENTS_FOLDER.glob(f"{recording.stem}_segment_*.wav"):
                    old_file.unlink()

                for index in range(num_segments):
                    segment_frames = segment_base + (1 if index < extra_frames else 0)
                    name = f"{recording.stem}_segment_{index + 1:03d}.wav"
                    output = SEGMENTS_FOLDER / name
                    chunk = source.readframes(segment_frames)
                    with wave.open(str(output), "wb") as dest:
                        dest.setnchannels(channels)
                        dest.setsampwidth(width)
                        dest.setframerate(rate)
                        dest.writeframes(chunk)

                    end = position + segment_frames
                    rows.append({
                        "segment_id": output.stem,
                        "species": species,
                        "source_file": recording.relative_to(PROJECT_FOLDER).as_posix(),
                        "segment_file": output.relative_to(PROJECT_FOLDER).as_posix(),
                        "start_seconds": f"{position / rate:.2f}",
                        "end_seconds": f"{end / rate:.2f}",
                        "duration_seconds": f"{segment_frames / rate:.2f}",
                        "sample_rate_hz": rate,
                        "channels": channels,
                    })
                    position = end
                print(f"  OK: {frame_count / rate:.1f}s, {num_segments} segment(s)")
                processed += 1
    except Exception as exc:
        errors += 1
        print(f"  ERROR: {exc}")

columns = [
    "segment_id", "species", "source_file", "segment_file",
    "start_seconds", "end_seconds", "duration_seconds", "sample_rate_hz", "channels",
]
with MANIFEST_FILE.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\nSUMMARY")
print(f"Audio files found: {len(recordings)}")
print(f"Audio files processed: {processed}")
print(f"Segments created: {len(rows)}")
print(f"Errors: {errors}")
print(f"Manifest saved to: {MANIFEST_FILE}")

if errors:
    sys.exit(1)
