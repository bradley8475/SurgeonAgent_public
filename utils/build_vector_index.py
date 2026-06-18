import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_index import ClipFaissIndex, collect_items_from_manifest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VECTOR_INDEX_DIR = os.path.join(PROJECT_ROOT, "vector_index")
DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "splits", "train.txt")
DEFAULT_IMAGE_FILENAME = "panel_stitch.png"


def build(
    index_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    image_filename: str = DEFAULT_IMAGE_FILENAME,
) -> None:
    index_dir = index_dir or DEFAULT_VECTOR_INDEX_DIR
    manifest_path = manifest_path or DEFAULT_MANIFEST
    items = collect_items_from_manifest(manifest_path, image_filename=image_filename)
    if not items:
        raise SystemExit(f"No items collected from manifest: {manifest_path}")
    print(f"collected {len(items)} (image, json) pairs from {manifest_path}")
    print(f"image filename: {image_filename}")
    index = ClipFaissIndex(index_dir=index_dir)
    index.build(items)
    print(f"built index with {len(items)} items at {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP+FAISS vector index from a split manifest")
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None, help=f"path to manifest of pattern.json paths (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--image_filename", type=str, default=DEFAULT_IMAGE_FILENAME)
    args = parser.parse_args()
    os.makedirs(args.index_dir or DEFAULT_VECTOR_INDEX_DIR, exist_ok=True)
    build(index_dir=args.index_dir, manifest_path=args.manifest, image_filename=args.image_filename)
