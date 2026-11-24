r"""Generate cross-modal CLIP embeddings for text, photos and video frames.

Project: Wildlife Multimodal Search Engine (CSDS 305)

Install (Windows PowerShell, in the project folder):
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install "transformers==4.50.3" numpy

Run:
    python .\scripts\generate_clip_embeddings.py

The first run downloads the pretrained CLIP model from Hugging Face.
Original data is never modified. Audio is intentionally NOT included here:
CLIP cannot directly embed bird sounds in the same vector space.
"""

from __future__ import annotations

import csv
from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 4  # Small batches reduce RAM use on student laptops.

try:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError as error:
    print(f"Missing package: {error}")
    print('Install: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu')
    print('Then:    python -m pip install "transformers==4.50.3" numpy')
    sys.exit(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. Run the preprocessing script first.")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def resolve_file(relative: str) -> Path:
    file = PROJECT / Path(relative.replace("/", os.sep))
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")
    return file


texts = read_csv(PROCESSED / "text_chunks.csv")
photos = read_csv(PROCESSED / "image_manifest.csv")
videos = read_csv(PROCESSED / "video_manifest.csv")

items: list[dict[str, str]] = []

for row in texts:
    items.append({
        "item_id": row["chunk_id"],
        "modality": "text",
        "species": row["species"],
        "file_path": f'data/text/{row["source_file"]}',
        "source_file": row["source_file"],
        "start_seconds": "",
        "end_seconds": "",
        "description": row["text"],
        # CLIP accepts at most 77 tokens. Use the opening context for this baseline.
        "embed_text": f'{row["species"].replace("_", " ")}: ' + " ".join(row["text"].split()[:55]),
    })

for index, row in enumerate(photos, start=1):
    items.append({
        "item_id": f'image_{row["species"]}_{index:03d}',
        "modality": "image",
        "species": row["species"],
        "file_path": row["processed_file"],
        "source_file": row["original_file"],
        "start_seconds": "",
        "end_seconds": "",
        "description": f'Photograph of {row["species"].replace("_", " ")}',
        "embed_text": "",
    })

for row in videos:
    items.append({
        "item_id": row["segment_id"],
        "modality": "video",
        "species": row["species"],
        "file_path": row["frame_file"],
        "source_file": row["segment_file"],
        "start_seconds": row["start_seconds"],
        "end_seconds": row["end_seconds"],
        "description": f'Representative frame from {row["species"].replace("_", " ")} video segment',
        "embed_text": "",
    })

if not items:
    raise SystemExit("No processed items found.")

for item in items:
    if item["modality"] != "text":
        resolve_file(item["file_path"])

print("Loading pretrained CLIP model (requires Internet on first run)...", flush=True)
try:
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
except Exception as error:
    print(f"Could not load model: {error}")
    print("Check your Internet connection and try the command again.")
    sys.exit(1)

model.eval()  # Inference only, no training needed.
model.to("cpu")  # Works on Windows PCs without a dedicated GPU.

embeddings: list[np.ndarray] = []

def batches(group: list[dict[str, str]]):
    for start in range(0, len(group), BATCH_SIZE):
        yield group[start:start+BATCH_SIZE]

for modality in ("text", "image", "video"):
    group = [x for x in items if x["modality"] == modality]
    print(f"Embedding {modality}: {len(group)} items", flush=True)
    for batch in batches(group):
        with torch.inference_mode():
            if modality == "text":
                model_inputs = processor(
                    text=[x["embed_text"] for x in batch],
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=77,
                )
                result = model.get_text_features(**model_inputs)
            else:
                images = []
                try:
                    for item in batch:
                        with Image.open(resolve_file(item["file_path"])) as image:
                            images.append(image.convert("RGB"))
                    model_inputs = processor(images=images, return_tensors="pt")
                    result = model.get_image_features(**model_inputs)
                finally:
                    for image in images:
                        image.close()
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("Unexpected CLIP result type. Use transformers==4.50.3.")
        result = torch.nn.functional.normalize(result.float(), p=2, dim=-1)
        embeddings.extend(result.cpu().numpy())

# The vectors were created by modality, so reorder metadata the same way.
ordered_items = [x for kind in ("text", "image", "video") for x in items if x["modality"] == kind]
array = np.stack(embeddings).astype("float32")
if array.shape[0] != len(ordered_items):
    raise RuntimeError("Vector count does not match the manifest")

npz_file = PROCESSED / "clip_embeddings.npz"
metadata_file = PROCESSED / "clip_embeddings_manifest.csv"
np.savez_compressed(npz_file, embeddings=array)
fields = ["item_id", "modality", "species", "file_path", "source_file",
          "start_seconds", "end_seconds", "description"]
with metadata_file.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for item in ordered_items:
        writer.writerow({key: item[key] for key in fields})

print("\nSUMMARY")
print("Text embeddings:", len(texts))
print("Image embeddings:", len(photos))
print("Video-frame embeddings:", len(videos))
print("Total embeddings:", len(ordered_items))
print("Vector dimensions:", array.shape[1])
print("Errors: 0")
print("Vectors saved to:", npz_file)
print("Metadata saved to:", metadata_file)
print("Audio remains a separate step: CLIP does not directly process sound.")
