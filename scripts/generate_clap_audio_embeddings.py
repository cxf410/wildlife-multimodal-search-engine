r"""Generate CLAP embeddings for the bird sound segments (CSDS 305).

Requirements (installed for the previous steps, unless missing):
    python -m pip install "transformers==4.50.3" numpy imageio-ffmpeg

Run from the Wildlife Bird Search project directory:
    python .\scripts\generate_clap_audio_embeddings.py

Input:
    data/processed/audio_manifest.csv
    data/processed/audio_segments/*.wav

Output:
    data/processed/clap_audio_embeddings.npz
    data/processed/clap_audio_embeddings_manifest.csv

CLAP vectors are in CLAP's AUDIO-TEXT space, NOT CLIP's IMAGE-TEXT space.
Do not compare or combine CLAP and CLIP vectors as if they used one space.
This script preserves all original files and existing CLIP embeddings.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

try:
    import imageio_ffmpeg
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, ClapModel
except ImportError as error:
    print(f"Missing dependency: {error}")
    print('Install: python -m pip install "transformers==4.50.3" numpy imageio-ffmpeg')
    sys.exit(1)

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
SOURCE_MANIFEST = PROCESSED / "audio_manifest.csv"
VECTORS_FILE = PROCESSED / "clap_audio_embeddings.npz"
OUTPUT_MANIFEST = PROCESSED / "clap_audio_embeddings_manifest.csv"
MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48_000  # CLAP expects 48 kHz mono audio.
WINDOW_SECONDS = 10   # CLAP uses approximately 10-second audio inputs.


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run process_audio.py first.")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def decode_mono_float32(file_path: Path, ffmpeg: str) -> np.ndarray:
    """Decode any segment into raw mono, 48 kHz, 32-bit floating samples."""
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(file_path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "FFmpeg audio decoding failed")
    sound = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(sound) == 0:
        raise ValueError("Audio segment has no samples")
    return sound


def windows_of(sound: np.ndarray):
    step = SAMPLE_RATE * WINDOW_SECONDS
    for start in range(0, len(sound), step):
        yield sound[start:start + step]


def embed_recording(audio: np.ndarray, extractor, model) -> np.ndarray:
    """Embed 10s windows separately and average normalized window embeddings."""
    weighted = []
    weights = []
    for window in windows_of(audio):
        inputs = extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_audio_features(**inputs)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        weighted.append(features.squeeze(0).cpu().numpy())
        weights.append(len(window))
    final = np.average(np.stack(weighted), axis=0, weights=weights).astype("float32")
    magnitude = np.linalg.norm(final)
    if not np.isfinite(magnitude) or magnitude < 1e-12:
        raise ValueError("The resulting embedding is invalid")
    return final / magnitude


def main() -> None:
    rows = read_rows(SOURCE_MANIFEST)
    if not rows:
        raise RuntimeError("The audio manifest is empty")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    for row in rows:
        relative = row["segment_file"].replace("/", os.sep)
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Audio segment not found: {path}")

    print("Loading CLAP model. The first run downloads model files...", flush=True)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    vectors = []
    metadata = []
    errors = 0
    for index, row in enumerate(rows, start=1):
        path = PROJECT / Path(row["segment_file"].replace("/", os.sep))
        print(f"[AUDIO] {index}/{len(rows)}: {path.name}", flush=True)
        try:
            samples = decode_mono_float32(path, ffmpeg)
            vector = embed_recording(samples, extractor, model)
            vectors.append(vector)
            metadata.append({
                "item_id": row["segment_id"],
                "modality": "audio",
                "species": row["species"],
                "file_path": row["segment_file"],
                "source_file": row["source_file"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "description": f"Bird call or song from {row['species'].replace('_', ' ')}",
                "model": MODEL_NAME,
                "embedding_space": "clap_audio_text",
            })
            print(f"  OK: {len(samples)/SAMPLE_RATE:.1f}s, {len(vector)} dimensions", flush=True)
        except Exception as error:
            errors += 1
            print(f"  ERROR: {error}", flush=True)

    if not vectors:
        raise RuntimeError("No valid audio embeddings were generated")

    array = np.stack(vectors).astype("float32")
    np.savez_compressed(VECTORS_FILE, embeddings=array)
    fields = [
        "item_id", "modality", "species", "file_path", "source_file",
        "start_seconds", "end_seconds", "description", "model", "embedding_space",
    ]
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    print("\nSUMMARY", flush=True)
    print(f"Audio segments found: {len(rows)}")
    print(f"Audio embeddings generated: {len(vectors)}")
    print(f"Vector dimensions: {array.shape[1]}")
    print(f"Errors: {errors}")
    print(f"Vectors saved to: {VECTORS_FILE}")
    print(f"Metadata saved to: {OUTPUT_MANIFEST}")
    print("Note: CLAP audio vectors are NOT in the CLIP vector space.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
