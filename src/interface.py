"""Common interfaces so any model can be swapped without touching the pipeline.

Every candidate (VLM, TrOCR, PaddleOCR, ...) is wrapped as a Reader.
Every quality-gate method is wrapped as a QualityChecker.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Region types used in labels (confirm the final list with the manager).
REGION_TYPES = [
    "usn", "student_name", "semester",        # header fields
    "question_number", "answer_text",
    "diagram", "diagram_label", "formula",
    "margin_note", "crossed_out", "other",
]


@dataclass
class Region:
    bbox: tuple[int, int, int, int]            # x1, y1, x2, y2 in pixels
    type: str                                  # one of REGION_TYPES
    text: str = ""
    confidence: Optional[float] = None         # 0..1, None if the model gives none
    parent_question: Optional[str] = None      # e.g. "Q3a": links an answer to its question


@dataclass
class ReadResult:
    regions: list[Region] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0


@dataclass
class QualityResult:
    scores: dict[str, float]                   # sharpness, lighting, line_clutter, skew, ... each 0..1
    is_ocr_ready: bool
    shadow_regions: list[tuple[int, int, int, int]] = field(default_factory=list)


class Reader(Protocol):
    name: str

    def read(self, image_path: str) -> ReadResult: ...


class QualityChecker(Protocol):
    name: str

    def check(self, image_path: str) -> QualityResult: ...
