"""
VLM Image Captioner — makes scanned pages, embedded figures, tables-as-images
and diagrams searchable by turning them into text.

WHERE IT SITS: called by IngestionOrchestrator right after a page is parsed.
If a page has < N characters of extractable text (i.e. it's a scan) OR
contains embedded raster images (figures, stamps, signatures, org charts),
each image is rendered/extracted and sent here. The returned caption is
stored as a normal Chunk (modality="image_caption") — it goes through the
SAME chunking -> embedding -> FAISS/BM25 -> GraphRAG entity-extraction path
as regular text, so an image caption can be retrieved and cited exactly
like a text passage.

Model: Qwen2.5-VL via Ollama (requires Ollama >= 0.7.0). The 3B variant is the
default — small enough to sit on a small GPU next to the 7B chat/judge models,
and Qwen2.5-VL is markedly stronger than tiny VLMs at document OCR, tables and
charts, which is exactly what scanned compliance pages need. Falls back to
qwen2.5vl:7b automatically if the 3B isn't pulled / fails, for richer captions.
Both tags are set in configs/models.yaml > vlm.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.llm.ollama_manager import OllamaManager

logger = LoggerManager.get_logger()

_CAPTION_PROMPT = (
    "You are extracting information from a page image of a compliance/legal "
    "document. Describe any figures, tables, diagrams, charts, stamps or "
    "handwritten annotations precisely. Transcribe any visible text verbatim. "
    "Be factual and literal — do not invent details you cannot see."
)


# Qwen-VL spends visual tokens in proportion to pixel count, so a huge embedded
# scan (e.g. 4000x3000) would cost several times a normal page render. A 150-dpi
# A4 page is ~1240x1754, so this cap leaves normal pages untouched while
# bounding worst-case latency/VRAM on oversized embedded images.
_MAX_IMAGE_SIDE = 1600


class ImageCaptioner:
    def __init__(self, ollama_manager: OllamaManager | None = None) -> None:
        cfg = ApplicationConfig().models["vlm"]
        self.model = cfg["model"]
        self.fallback_model = cfg.get("fallback_model")
        self.ollama = ollama_manager or OllamaManager(
            host=ApplicationConfig().models["ollama"]["host"]
        )

    def caption(self, image: Image.Image) -> str:
        b64 = self._to_base64(image)
        try:
            return self.ollama.generate_vision(
                model=self.model, prompt=_CAPTION_PROMPT, images=[b64]
            )
        except Exception as exc:
            logger.warning(f"VLM '{self.model}' failed ({exc}); trying fallback "
                            f"'{self.fallback_model}'.")
            if self.fallback_model:
                return self.ollama.generate_vision(
                    model=self.fallback_model, prompt=_CAPTION_PROMPT, images=[b64]
                )
            return ""

    @staticmethod
    def _to_base64(image: Image.Image) -> str:
        image = image.convert("RGB")
        if max(image.size) > _MAX_IMAGE_SIDE:
            image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
