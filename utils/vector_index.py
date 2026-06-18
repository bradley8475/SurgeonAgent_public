import os
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

try:
    import open_clip
except Exception as e:  # pragma: no cover
    open_clip = None

try:
    import faiss
except Exception as e:  # pragma: no cover
    faiss = None

# 默认配置
DEFAULT_VECTOR_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_index")


class ClipFaissIndex:
    """Encapsulates CLIP image embedder and FAISS index for nearest-neighbor search."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        index_dir: Optional[str] = None,
    ) -> None:
        if open_clip is None:
            raise RuntimeError("open-clip-torch is not installed")
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.index_dir = index_dir or DEFAULT_VECTOR_INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Normalize like CLIP expects
        self.transform = T.Compose([
            T.Resize(self.preprocess.transforms[0].size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.preprocess.transforms[1].size),
            T.ToTensor(),
            T.Normalize(mean=self.preprocess.transforms[-1].mean, std=self.preprocess.transforms[-1].std),
        ])

        self.index: Optional[faiss.IndexFlatIP] = None
        self.ids: List[str] = []
        self.meta_path = os.path.join(self.index_dir, "meta.json")
        self.index_path = os.path.join(self.index_dir, "index.faiss")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text prompt into the same vector space as images."""
        text_tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(text_tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    def build(self, items: List[Tuple[str, str]]) -> None:
        """
        Build the index from a list of (image_path, json_path).
        Stores metadata mapping and saves FAISS index to disk.
        """
        embeddings: List[np.ndarray] = []
        self.ids = []

        for image_path, json_path in items:
            if not os.path.exists(image_path) or not os.path.exists(json_path):
                continue
            try:
                with Image.open(image_path) as im:
                    emb = self.encode_image(im)  # (1, d)
                    embeddings.append(emb)
                    self.ids.append(json_path)
            except Exception:
                continue

        if not embeddings:
            raise RuntimeError("No valid embeddings were created from the dataset.")

        X = np.vstack(embeddings)
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X)

        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids}, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            raise FileNotFoundError("Vector index files not found. Build the index first.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids = meta.get("ids", [])

    def is_ready(self) -> bool:
        return self.index is not None and len(self.ids) == self.index.ntotal

    @torch.no_grad()
    def search(self, query_text: str, k: int = 5) -> List[Dict[str, object]]:
        """Search the index using text query.
        
        Args:
            query_text: Text prompt to search with
            k: Number of results to return
            
        Returns:
            List of search results with rank, score, json_path, and pattern_json
        """
        if not self.is_ready():
            self.load()
        assert self.index is not None

        # Encode text query
        q = self.encode_text(query_text)  # (1, d)

        sims, idxs = self.index.search(q, k)
        results: List[Dict[str, object]] = []
        for rank, (score, idx) in enumerate(zip(sims[0].tolist(), idxs[0].tolist())):
            if idx == -1 or idx >= len(self.ids):
                continue
            json_path = self.ids[idx]
            result = {"rank": rank, "score": float(score), "json_path": json_path}
            # Try to open and attach basic JSON content (optional)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result["pattern_json"] = data
            except Exception:
                pass
            results.append(result)
        return results


def collect_items_from_manifest(
    manifest_path: str,
    image_filename: str = "panel_stitch.png",
) -> List[Tuple[str, str]]:
    """Pair each pattern.json listed in the manifest with `image_filename` in the same dir."""
    pairs: List[Tuple[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            json_path = line.strip()
            if not json_path:
                continue
            image_path = os.path.join(os.path.dirname(json_path), image_filename)
            if (
                os.path.exists(image_path)
                and os.path.exists(json_path)
                and os.path.getsize(json_path) > 0
            ):
                pairs.append((image_path, json_path))
    return pairs


