"""Prepare the bird image dataset for embedding generation.

Run from anywhere:
    python -m pip install Pillow
    python scripts/process_images.py

This program leaves the original photos unchanged, prepares resized copies,
and creates a CSV file with basic image metadata.
"""
from pathlib import Path
import csv

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:
    raise SystemExit("Pillow is missing. Install it using: python -m pip install Pillow")

project_folder = Path(__file__).resolve().parent.parent
images_folder = project_folder / "data" / "images"
output_folder = project_folder / "data" / "processed" / "images"
metadata_folder = project_folder / "data" / "processed"

output_folder.mkdir(parents=True, exist_ok=True)
metadata_folder.mkdir(parents=True, exist_ok=True)

supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
image_files = sorted(
    p for p in images_folder.iterdir()
    if p.is_file() and p.suffix.lower() in supported_extensions
) if images_folder.exists() else []

records = []
errors = []

for image_file in image_files:
    try:
        with Image.open(image_file) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            original_width, original_height = image.size
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            target = output_folder / (image_file.stem + ".jpg")
            image.save(target, "JPEG", quality=90, optimize=True)
            records.append({
                "species": image_file.stem.removesuffix("_image"),
                "original_file": image_file.relative_to(project_folder).as_posix(),
                "processed_file": target.relative_to(project_folder).as_posix(),
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": image.width,
                "processed_height": image.height,
            })
            print("OK:", image_file.name, "->", target.name)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        print("FAILED:", image_file.name, error)
        errors.append(image_file.name)

manifest = metadata_folder / "image_manifest.csv"
with manifest.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "species", "original_file", "processed_file",
        "original_width", "original_height", "processed_width", "processed_height"
    ])
    writer.writeheader()
    writer.writerows(records)

print("\nSUMMARY")
print("Images found:", len(image_files))
print("Images processed:", len(records))
print("Errors:", len(errors))
print("Manifest saved to:", manifest)
