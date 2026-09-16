"""CLIP Vision-Language Model Wrapper for Zero-Shot Cross-Modal Retrieval (Method A2).

Provides CPU-only image and text embedding extraction with mandatory L2 normalization.
Wraps Hugging Face transformers CLIPModel and CLIPProcessor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Loads a pretrained CLIP model and processor strictly forced onto the specified device.

    Args:
        model_name: HuggingFace model identifier.
        device: Target execution device. Must be 'cpu' for M4 research integrity.

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    if device.lower() != "cpu":
        logger.warning(
            f"Requested device '{device}' overrides CPU constraint. Forcing 'cpu' for M4 evaluation integrity."
        )
        device = "cpu"

    logger.info(f"Loading CLIP model '{model_name}' on Device: CPU")
    print(f"[CLIP Model Loader] Device: CPU (Model: {model_name})")

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor


class CLIPRetriever:
    """Encapsulates image and text encoding using CLIP ViT-B/32 on CPU."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        """Initializes CLIPRetriever.

        Args:
            model_name: Model identifier string.
            device: Execution device (forced to 'cpu').
            batch_size: Batch size for image and text forward passes.
        """
        self.device = "cpu"
        self.batch_size = batch_size
        self.model_name = model_name
        self.model, self.processor = load_clip_model(model_name, device=self.device)

    def encode_images(
        self,
        image_inputs: Sequence[Union[Path, str, Image.Image]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extracts L2-normalized image embeddings for a sequence of image paths or PIL Images.

        Args:
            image_inputs: Sequence of file paths or PIL Image objects.
            batch_size: Optional batch size override.

        Returns:
            Float32 array of shape (N, 512) where each row vector has unit L2 norm.
        """
        bs = batch_size or self.batch_size
        all_embeddings: List[np.ndarray] = []

        total_images = len(image_inputs)
        logger.info(f"Encoding {total_images} images in batches of {bs} on CPU...")

        for start_idx in range(0, total_images, bs):
            batch_items = image_inputs[start_idx : start_idx + bs]
            pil_images: List[Image.Image] = []

            for item in batch_items:
                if isinstance(item, (str, Path)):
                    with Image.open(item) as img:
                        pil_images.append(img.convert("RGB"))
                elif isinstance(item, Image.Image):
                    pil_images.append(item.convert("RGB"))
                else:
                    raise TypeError(f"Unsupported image input type: {type(item)}")

            # Preprocess and tokenize
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                # L2 normalize
                features = features / torch.norm(features, p=2, dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy().astype(np.float32))

        embeddings = np.vstack(all_embeddings)
        return embeddings

    def encode_text(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extracts L2-normalized text embeddings for a sequence of query strings.

        Args:
            texts: Sequence of natural-language text queries.
            batch_size: Optional batch size override.

        Returns:
            Float32 array of shape (N, 512) with unit L2 norm.
        """
        bs = batch_size or self.batch_size
        all_embeddings: List[np.ndarray] = []

        total_texts = len(texts)

        for start_idx in range(0, total_texts, bs):
            batch_texts = list(texts[start_idx : start_idx + bs])

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                # L2 normalize
                features = features / torch.norm(features, p=2, dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy().astype(np.float32))

        embeddings = np.vstack(all_embeddings)
        return embeddings

    def encode_single_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a normalized (1, 512) vector."""
        return self.encode_text([query], batch_size=1)
