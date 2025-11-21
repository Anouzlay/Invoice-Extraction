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

        key = (tuple(sorted(langs)), use_gpu)  # Sort langs for consistent key matching
        if key not in self.__class__._ocr_clients:
            # Suppress warnings when creating new client (shouldn't happen if pre-init worked)
            import logging
            import warnings
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO
            
            # Suppress EasyOCR loggers
            for logger_name in ['easyocr', 'easyocr.easyocr']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    self.__class__._ocr_clients[key] = Reader(langs, gpu=use_gpu, verbose=False)
        return self.__class__._ocr_clients[key]

    @classmethod
    def pre_initialize_ocr(cls, ocr_lang: str = "en,de,fr", use_gpu: bool = False) -> None:
        """
        Pre-initialize EasyOCR client to download models at startup instead of during first use.
        This prevents model downloads during runtime, which is especially important for deployment.
        
        Args:
            ocr_lang: Comma-separated list of language codes (e.g., "en,de,fr")
            use_gpu: Whether to use GPU acceleration
        """
        if Reader is None:
            # EasyOCR not available, skip initialization
            return
        
        # Resolve languages using the EXACT same logic as _resolve_ocr_langs
        langs = [chunk.strip() for chunk in ocr_lang.split(",")]
        unique_langs: List[str] = []
        for lang in langs:
            if lang and lang not in unique_langs:
                unique_langs.append(lang)
        unique_langs = unique_langs or ["en"]
        
        # Initialize the client (this will download models if needed)
        # Sort langs for consistent key matching with _get_ocr_client
        key = (tuple(sorted(unique_langs)), use_gpu)
        if key not in cls._ocr_clients:
            # Completely suppress all EasyOCR output during initialization
            import warnings
            import logging
            import sys
            import os
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO
            
            # Set environment variable to suppress EasyOCR output
            os.environ['EASYOCR_MODULE_PATH'] = os.environ.get('EASYOCR_MODULE_PATH', '')
            
            # Suppress all EasyOCR loggers
            for logger_name in ['easyocr', 'easyocr.easyocr']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.disabled = True
            
            # Suppress root logger warnings
            root_logger = logging.getLogger()
            old_root_level = root_logger.level
            root_logger.setLevel(logging.CRITICAL)
            
            try:
                # Suppress all warnings and redirect both stdout and stderr
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Redirect both stdout and stderr to completely silence output
                    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                        # Create the Reader - this triggers model download if not already cached
                        reader = Reader(unique_langs, gpu=use_gpu, verbose=False)
                        cls._ocr_clients[key] = reader
                        
                        # CRITICAL: Force model download by running OCR on a dummy image
                        # This ensures models are fully downloaded and loaded into memory
                        try:
                            # Create a minimal 10x10 white image to trigger model loading
                            dummy_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
                            # Run readtext to force model download and initialization
                            # This is the key step that actually downloads the models
                            reader.readtext(dummy_image, detail=0, paragraph=False)
                        except Exception:
                            # If dummy OCR fails, that's OK - models are still initialized
                            # The actual OCR will work when called with real images
                            pass
            finally:
                # Restore logging
                root_logger.setLevel(old_root_level)
                for logger_name in ['easyocr', 'easyocr.easyocr']:
                    logger = logging.getLogger(logger_name)
                    logger.disabled = False
                    logger.setLevel(logging.WARNING)

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
