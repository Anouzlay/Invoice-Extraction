# tools/pdf_extractor_fitz.py
from __future__ import annotations

import io
from typing import Any, ClassVar, Dict, List, Tuple

import fitz
import numpy as np
from pydantic import BaseModel, Field

from crewai.tools import BaseTool

try:
    from PIL import Image
    _pil_available = True
except ImportError:
    Image = None  # type: ignore[assignment]
    _pil_available = False

try:
    from easyocr import Reader

    _easyocr_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    Reader = None  # type: ignore[assignment]
    _easyocr_import_error = exc


class PdfExtractorArgs(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")
    show_log: bool | None = Field(
        default=None,
        description="(Optional) Ignored flag kept for backward compatibility with older agents.",
    )


class PdfExtractorFitz(BaseTool):
    name: str = "pdf_extractor_fitz"
    description: str = "Extract full text from a PDF using PyMuPDF (fitz)."
    args_schema: type[BaseModel] = PdfExtractorArgs

    max_pages: int = 0  # 0 = all
    ocr_lang: str = "en,de,fr"
    ocr_use_angle_cls: bool = True  # kept for backwards compat/config parity
    ocr_dpi: int = 200
    ocr_use_gpu: bool = False

    _ocr_clients: ClassVar[dict[Tuple[Tuple[str, ...], bool], Any]] = {}

    def _run(self, pdf_path: str, show_log: bool | None = None) -> Dict:  # noqa: ARG002
        # STEP 1: Extract all text content as raw (from all pages)
        text = self._extract_text_with_fitz(pdf_path)
        has_text = bool(text.strip())
        
        # STEP 2: Run OCR based on whether text was found
        ocr_text = ""
        try:
            if has_text:
                # If text was found: OCR only first page (for vendor logos/images)
                ocr_text = self._extract_text_with_ocr_first_page_only(pdf_path)
            else:
                # If no text found: OCR all pages (scanned PDF fallback)
                ocr_text = self._extract_text_with_ocr_all_pages(pdf_path)
        except Exception:
            # If OCR fails, continue with text-only extraction
            pass
        
        # STEP 3: Combine both results into a single unified text layer
        # Merge text and OCR text naturally without separators
        combined_text_parts = []
        if text.strip():
            combined_text_parts.append(text)
        if ocr_text.strip():
            combined_text_parts.append(ocr_text)
        
        # Join all text parts with newlines to create unified text layer
        combined_text = "\n".join(combined_text_parts).strip()
        if combined_text:
            return {"pdf_path": pdf_path, "text": combined_text}
        
        # Fallback: if no text at all, return OCR result (even if empty)
        return {"pdf_path": pdf_path, "text": ocr_text if ocr_text else ""}

    def _extract_text_with_fitz(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        try:
            parts: List[str] = []
            for i, page in enumerate(doc):
                if self.max_pages and i >= self.max_pages:
                    break
                parts.append(page.get_text() or "")
            text = "\n".join(parts).strip()
            if self._is_low_quality_extraction(text):
                return ""
            return text
        finally:
            doc.close()

    def _extract_text_with_ocr_first_page_only(self, pdf_path: str) -> str:
        """
        Run OCR only on the first page of the PDF.
        This captures vendor logos and images that typically appear on the first page.
        Used when PDF has extractable text - we only need OCR for images on first page.
        """
        if Reader is None:
            missing_dep = "easyocr"
            if _easyocr_import_error is not None:
                missing_dep = f"{missing_dep} ({_easyocr_import_error})"
            raise RuntimeError(
                "EasyOCR is not available. Please install the optional OCR dependency "
                "easyocr>=1.7.1. "
                f"Original import issue: {missing_dep}"
            )

        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                return ""
            
            # Process only the first page (page 0)
            first_page = doc[0]
            langs = self._resolve_ocr_langs()
            ocr_client = self._get_ocr_client(langs, self.ocr_use_gpu)
            
            # Convert the entire first page to an image and run OCR on it
            # This captures all text from images, logos, and any rendered content
            pix = first_page.get_pixmap(dpi=self.ocr_dpi)
            data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                data = data[:, :, :3]
            
            ocr_results = self._perform_ocr(ocr_client, data)
            return "\n".join(ocr_results).strip() if ocr_results else ""
        finally:
            doc.close()
    
    def _extract_text_with_ocr_all_pages(self, pdf_path: str) -> str:
        """
        Run OCR on all pages of the PDF.
        Used when PDF has no extractable text (scanned PDFs) - need OCR on all pages.
        """
        if Reader is None:
            missing_dep = "easyocr"
            if _easyocr_import_error is not None:
                missing_dep = f"{missing_dep} ({_easyocr_import_error})"
            raise RuntimeError(
                "EasyOCR is not available. Please install the optional OCR dependency "
                "easyocr>=1.7.1. "
                f"Original import issue: {missing_dep}"
            )

        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                return ""
            
            page_texts: List[str] = []
            langs = self._resolve_ocr_langs()
            ocr_client = self._get_ocr_client(langs, self.ocr_use_gpu)
            
            # Process all pages for scanned PDFs
            for i, page in enumerate(doc):
                if self.max_pages and i >= self.max_pages:
                    break
                
                # Convert each page to an image and run OCR on it
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    data = data[:, :, :3]
                
                ocr_results = self._perform_ocr(ocr_client, data)
                if ocr_results:
                    page_texts.extend(ocr_results)
            
            return "\n".join(page_texts).strip()
        finally:
            doc.close()

    def _resolve_ocr_langs(self) -> List[str]:
        langs = [chunk.strip() for chunk in self.ocr_lang.split(",")]
        unique_langs: List[str] = []
        for lang in langs:
            if lang and lang not in unique_langs:
                unique_langs.append(lang)
        return unique_langs or ["en"]

    def _get_ocr_client(self, langs: List[str], use_gpu: bool) -> Any:
        if Reader is None:
            raise RuntimeError(
                "EasyOCR is not available. Install easyocr>=1.7.1 to enable OCR fallback."
            )

        key = (tuple(langs), use_gpu)
        if key not in self.__class__._ocr_clients:
            self.__class__._ocr_clients[key] = Reader(langs, gpu=use_gpu, verbose=False)
        return self.__class__._ocr_clients[key]

    def _perform_ocr(self, ocr_client: Any, image: Any) -> List[str]:
        """Run EasyOCR reader and return detected text lines."""
        results = ocr_client.readtext(image, detail=1, paragraph=False) or []
        return [text for _, text, _ in results]

    def _is_low_quality_extraction(self, text: str) -> bool:
        """Heuristic detection for unusable PyMuPDF output (e.g., single characters per line)."""
        stripped = text.strip()
        if not stripped:
            return True

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return True

        if len(stripped) >= 200:
            unique_chars = len(set(stripped))
            if unique_chars <= 5:
                return True

        short_lines = sum(1 for line in lines if len(line) <= 2)
        short_line_ratio = short_lines / len(lines)

        multi_word_lines = sum(1 for line in lines if len(line.split()) >= 2)

        return short_line_ratio > 0.8 and multi_word_lines == 0
