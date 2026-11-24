from pathlib import Path
import csv

# Find the project folder from the location of this script.
project_folder = Path(__file__).resolve().parent.parent
text_folder = project_folder / "data" / "text"
output_folder = project_folder / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# Each chunk contains at most 100 words, with 20 words of overlap.
chunk_size = 100
overlap = 20
step = chunk_size - overlap

text_files = sorted(text_folder.glob("*.txt"))
if not text_files:
    raise SystemExit(f"ERROR: No .txt files found in {text_folder}")

chunks = []
for file in text_files:
    words = file.read_text(encoding="utf-8").split()
    species = file.stem.replace("_identification", "")
    species = species.replace("_behaviour_and_sound", "")

    for start in range(0, len(words), step):
        part = words[start:start + chunk_size]
        if not part:
            break
        chunks.append({
            "chunk_id": f"{file.stem}_{start}",
            "species": species,
            "source_file": file.name,
            "text": " ".join(part),
        })
        if start + chunk_size >= len(words):
            break

output_file = output_folder / "text_chunks.csv"
with output_file.open("w", newline="", encoding="utf-8-sig") as result:
    writer = csv.DictWriter(result, fieldnames=[
        "chunk_id", "species", "source_file", "text"
    ])
    writer.writeheader()
    writer.writerows(chunks)

print("Text files processed:", len(text_files))
print("Chunks created:", len(chunks))
print("Saved to:", output_file)
