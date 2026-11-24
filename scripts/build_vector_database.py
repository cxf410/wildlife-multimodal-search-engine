r"""Build a local ChromaDB vector database for the Wildlife Bird Search project.

CSDS 305 - Wildlife Multimodal Semantic Search Engine

Run from anywhere (after installation):
    python -m pip install chromadb
    python .\scripts\build_vector_database.py

Inputs (already created in the embedding phase):
    data/processed/clip_embeddings.npz
    data/processed/clip_embeddings_manifest.csv
    data/processed/clap_audio_embeddings.npz
    data/processed/clap_audio_embeddings_manifest.csv

Creates TWO independent collections using HNSW and cosine distance:
    wildlife_clip   -> text, images, video frames (CLIP space)
    wildlife_audio  -> audio clips (CLAP space)

CLIP and CLAP embedding vectors cannot be compared directly even if they
have the same 512 dimensions. They are stored in separate collections.

The script is safe to rerun: records are upserted with stable IDs.
Original multimedia files and embedding files are not modified.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

try:
    import chromadb
    import numpy as np
except ImportError as error:
    print(f"Missing library: {error}")
    print("Install it with: python -m pip install chromadb")
    sys.exit(1)


PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
DATABASE_DIR = PROJECT / "data" / "vector_db"

COLLECTIONS = [
    (
        "wildlife_clip",
        "clip_embeddings.npz",
        "clip_embeddings_manifest.csv",
        "clip_text_image_video",
        43,
    ),
    (
        "wildlife_audio",
        "clap_audio_embeddings.npz",
        "clap_audio_embeddings_manifest.csv",
        "clap_audio_text",
        9,
    ),
]


def load_data(vector_filename: str, manifest_filename: str):
    vector_path = PROCESSED / vector_filename
    manifest_path = PROCESSED / manifest_filename
    if not vector_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing: {vector_path.name} or {manifest_path.name}. "
            "Generate embeddings first."
        )

    with np.load(vector_path, allow_pickle=False) as archive:
        if "embeddings" not in archive.files:
            raise ValueError(f"'{vector_filename}' has no 'embeddings' array")
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if vectors.ndim != 2 or vectors.shape[1] != 512:
        raise ValueError(f"Unexpected vector shape for {vector_filename}: {vectors.shape}")
    if not rows or len(rows) != len(vectors):
        raise ValueError(
            f"Manifest/vector mismatch: {len(rows)} rows vs {len(vectors)} vectors "
            f"in {vector_filename}"
        )
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"Invalid (non-finite) vector in {vector_filename}")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError(f"Zero-length vector in {vector_filename}")
    # Safe normalization for cosine distance, even if input is already normalized.
    vectors = vectors / lengths[:, None]

    identifiers = [row.get("item_id", "").strip() for row in rows]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Missing or repeated item_id values in {manifest_filename}")

    return vectors, rows, identifiers


def make_collection(client, name: str):
    # Current Chroma versions configure HNSW via `configuration`.
    # Older versions accepted `metadata={"hnsw:space": "cosine"}`.
    try:
        return client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def make_metadata(row: dict[str, str], embedding_space: str) -> dict[str, str]:
    # Chroma metadata values must be simple types. Empty strings are allowed.
    result = {
        "species": row.get("species", ""),
        "modality": row.get("modality", ""),
        "file_path": row.get("file_path", ""),
        "source_file": row.get("source_file", ""),
        "start_seconds": row.get("start_seconds", ""),
        "end_seconds": row.get("end_seconds", ""),
        "embedding_space": embedding_space,
    }
    return result


def main():
    # Check ALL inputs before writing anything to the DB.
    packages = []
    for collection_name, vectors_file, manifest_file, space, expected in COLLECTIONS:
        vectors, rows, ids = load_data(vectors_file, manifest_file)
        packages.append((collection_name, vectors, rows, ids, space, expected))
        print(f"[CHECK] {collection_name}: {len(ids)} vectors, {vectors.shape[1]} dimensions")

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    print("\nSaving vectors to ChromaDB...")
    for name, vectors, rows, ids, space, expected in packages:
        collection = make_collection(client, name)
        if len(ids) != expected:
            print(f"  Note: expected {expected}, found {len(ids)} records")

        collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=[make_metadata(row, space) for row in rows],
            documents=[row.get("description", "") or row.get("species", "") for row in rows],
        )
        count = collection.count()
        print(f"  OK: {name} -> {count} records stored")

        # A small smoke test using an existing stored vector to confirm queries work.
        result = collection.query(
            query_embeddings=[vectors[0].tolist()],
            n_results=1,
            include=["distances"],
        )
        if not result.get("ids") or not result["ids"][0]:
            raise RuntimeError(f"Query test failed for {name}")
        print(f"  Search test OK: top result = {result['ids'][0][0]}")

    print("\nSUMMARY")
    print("CLIP collection records:", client.get_collection("wildlife_clip").count())
    print("CLAP collection records:", client.get_collection("wildlife_audio").count())
    print("Total records:", sum(client.get_collection(n).count() for n, *_ in COLLECTIONS))
    print("Vector index: HNSW (cosine distance)")
    print("Errors: 0")
    print("Database saved to:", DATABASE_DIR)
    print("Important: CLIP and CLAP are stored and searched separately.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
