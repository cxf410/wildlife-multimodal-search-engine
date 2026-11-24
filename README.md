# Wildlife Multimodal Semantic Search Engine

**Course:** CSDS 305 — Files, Indexes and Access Structures for Big Data  
**Project status:** Data collection, preprocessing, embedding generation, and vector indexing completed. Search interface not yet implemented.

## Project overview

This class project explores semantic search across text, photographs, bird sound recordings, and short wildlife videos. The initial dataset covers six bird species: European robin, common blackbird, Eurasian wren, common cuckoo, Eurasian blue tit, and red-winged blackbird.

## Current progress

| Stage | Result |
| --- | --- |
| Data collection | 30 original local files: 12 text files, 6 images, 6 audio files, 6 videos |
| Text preprocessing | 24 text chunks |
| Image preprocessing | 6 resized/standardized images |
| Audio preprocessing | 9 audio segments from 6 recordings |
| Video preprocessing | 13 video segments and 13 representative frames |
| CLIP embeddings | 43 vectors (24 text, 6 images, 13 frames), 512 dimensions |
| CLAP embeddings | 9 audio vectors, 512 dimensions |
| Vector database | 52 vectors in local ChromaDB, HNSW index, cosine distance |

The scripts were run on a Windows machine with Python 3.12 and returned zero processing/indexing errors for the completed stages. The database was tested by retrieving nearest neighbors using already stored vectors. **A user-facing semantic search interface and tests with unseen queries have not yet been implemented.**

## Technical approach

1. **Collection:** Download open-licensed media from Wikimedia Commons; retain the original source pages, licensing details and species labels in `metadata/`.
2. **Preprocessing:** Divide text into overlapping chunks, standardize images, split bird audio and video into up-to-30-second clips, and extract representative video frames.
3. **Embeddings:** Use the pretrained CLIP model for text/images/video frames and CLAP for bird audio.
4. **Indexing:** Store embeddings and metadata in two local ChromaDB collections: `wildlife_clip` (43 records) and `wildlife_audio` (9 records), with HNSW and cosine distance.

**Important limitation:** CLIP and CLAP vectors belong to *different* embedding spaces even when they have the same dimensionality. Cross-modal retrieval between audio and the other media is not yet implemented; it will require a compatible retrieval or ranking strategy.

## Code and files

- `scripts/download_media.py`: download selected multimedia files from Wikimedia Commons.
- `scripts/check_dataset.py`: verify local dataset files.
- `scripts/chunk_text.py`: create overlapping text chunks.
- `scripts/process_images.py`: standardize images.
- `scripts/process_audio.py`: create audio segments.
- `scripts/process_videos.py`: split videos and extract frames.
- `scripts/generate_clip_embeddings.py`: generate CLIP embeddings.
- `scripts/generate_clap_audio_embeddings.py`: generate CLAP audio embeddings.
- `scripts/build_vector_database.py`: index embeddings in ChromaDB.
- `Wildlife_Multimodal_Search_CSDS305.ipynb`: notebook summarizing the progress and offering an optional database demo.
- `metadata/`: file inventory, source links, licenses and species descriptions.

## Running the pipeline locally

From the project root with Python 3.12 on Windows:

```powershell
python -m pip install -r requirements.txt
python scripts/download_media.py
python scripts/check_dataset.py
python scripts/chunk_text.py
python scripts/process_images.py
python scripts/process_audio.py
python scripts/process_videos.py
python scripts/generate_clip_embeddings.py
python scripts/generate_clap_audio_embeddings.py
python scripts/build_vector_database.py
```

PyTorch can be installed separately using its official CPU wheel instructions if necessary. The first embedding run downloads pretrained model weights and requires a working Internet connection. Download availability and remote licensing rules can change.

### Data sharing

Raw photos, audio recordings, videos, processed media, model weights, vector files and the local ChromaDB directory are **not committed to GitHub** to keep this source-code repository lightweight. To rebuild the pipeline, download the media using the included script and rerun preprocessing and embedding generation. Attribution/source files in `metadata/` must be retained; check source-specific licenses before redistributing any originals.

## Remaining work

- Implement text/image/audio queries and top-K retrieval.
- Design and test cross-modal result fusion (especially CLIP versus CLAP).
- Create the user-facing application and evaluate retrieval quality.
- Submit the final PDF report and GitHub repository link.
