# tools/pdf_extractor_fitz.py
from __future__ import annotations

import io
import os
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
    import google.generativeai as genai
    _gemini_available = True
    _gemini_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    _gemini_available = False
    _gemini_import_error = exc


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
    ocr_lang: str = "en,de,fr"  # kept for backwards compat, not used with Gemini
    ocr_use_angle_cls: bool = True  # kept for backwards compat/config parity
    ocr_dpi: int = 200
    ocr_use_gpu: bool = False  # kept for backwards compat, not used with Gemini
    ocr_model: str = "gemini-1.5-pro"  # Gemini Vision model to use (supports vision)

    _ocr_client: ClassVar[Any] = None

    def _run(self, pdf_path: str, show_log: bool | None = None) -> Dict:  # noqa: ARG002
        # STEP 1: Extract all text content as raw (from all pages)
        text = self._extract_text_with_fitz(pdf_path)
        has_text = bool(text.strip())
        
        # STEP 2: Run OCR based on whether text was found
        ocr_text = ""
        try:
            if has_text:
                # If text was found: OCR only first page (for vendor logos/images)
                print(f"[PdfExtractorFitz] Text found, running OCR on first page only...")
                ocr_text = self._extract_text_with_ocr_first_page_only(pdf_path)
            else:
                # If no text found: OCR all pages (scanned PDF fallback)
                print(f"[PdfExtractorFitz] No text found, running OCR on all pages...")
                ocr_text = self._extract_text_with_ocr_all_pages(pdf_path)
            if ocr_text:
                print(f"[PdfExtractorFitz] OCR extracted {len(ocr_text)} characters")
        except Exception as e:
            # Log the error but continue with text-only extraction
            print(f"[PdfExtractorFitz] WARNING: OCR failed but continuing with text extraction: {type(e).__name__}: {e}")
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
        if not _gemini_available:
            missing_dep = "google-generativeai"
            if _gemini_import_error is not None:
                missing_dep = f"{missing_dep} ({_gemini_import_error})"
            raise RuntimeError(
                "Gemini Vision API is not available. Please install the optional OCR dependency "
                "google-generativeai>=0.3.0. "
                f"Original import issue: {missing_dep}"
            )

        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                return ""
            
            # Process only the first page (page 0)
            first_page = doc[0]
            ocr_client = self._get_ocr_client()
            
            # Convert the entire first page to an image and run OCR on it
            # This captures all text from images, logos, and any rendered content
            pix = first_page.get_pixmap(dpi=self.ocr_dpi)
            data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                data = data[:, :, :3]
            
            ocr_text = self._perform_ocr(ocr_client, data)
            return ocr_text.strip() if ocr_text else ""
        finally:
            doc.close()
    
    def _extract_text_with_ocr_all_pages(self, pdf_path: str) -> str:
        """
        Run OCR on all pages of the PDF.
        Used when PDF has no extractable text (scanned PDFs) - need OCR on all pages.
        """
        if not _gemini_available:
            missing_dep = "google-generativeai"
            if _gemini_import_error is not None:
                missing_dep = f"{missing_dep} ({_gemini_import_error})"
            raise RuntimeError(
                "Gemini Vision API is not available. Please install the optional OCR dependency "
                "google-generativeai>=0.3.0. "
                f"Original import issue: {missing_dep}"
            )

        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                return ""
            
            page_texts: List[str] = []
            ocr_client = self._get_ocr_client()
            
            # Process all pages for scanned PDFs
            for i, page in enumerate(doc):
                if self.max_pages and i >= self.max_pages:
                    break
                
                # Convert each page to an image and run OCR on it
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    data = data[:, :, :3]
                
                ocr_text = self._perform_ocr(ocr_client, data)
                if ocr_text and ocr_text.strip():
                    page_texts.append(ocr_text.strip())
            
            return "\n".join(page_texts).strip()
        finally:
            doc.close()

    def _resolve_ocr_langs(self) -> List[str]:
        """Kept for backwards compatibility, not used with Gemini Vision."""
        langs = [chunk.strip() for chunk in self.ocr_lang.split(",")]
        unique_langs: List[str] = []
        for lang in langs:
            if lang and lang not in unique_langs:
                unique_langs.append(lang)
        return unique_langs or ["en"]

    def _get_ocr_client(self) -> Any:
        """Initialize and return Gemini Vision client."""
        if not _gemini_available:
            error_msg = "Gemini Vision API is not available. Install google-generativeai>=0.3.0 to enable OCR fallback."
            if _gemini_import_error is not None:
                error_msg += f" Import error: {_gemini_import_error}"
            raise RuntimeError(error_msg)
        
        # Initialize client if not already done
        if self.__class__._ocr_client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("[PdfExtractorFitz] ERROR: GEMINI_API_KEY not found in environment")
                raise RuntimeError(
                    "GEMINI_API_KEY not found in environment. "
                    "Please set it in your environment variables or .env file."
                )
            
            print(f"[PdfExtractorFitz] Initializing Gemini Vision client with model: {self.ocr_model}")
            try:
                genai.configure(api_key=api_key)
                
                # First, try to list available models to see what's actually available
                try:
                    print(f"[PdfExtractorFitz] Checking available models...")
                    available_models = genai.list_models()
                    vision_models = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods and 'vision' in m.name.lower() or 'gemini' in m.name.lower()]
                    print(f"[PdfExtractorFitz] Available vision models: {vision_models[:5]}")  # Show first 5
                except Exception as list_error:
                    print(f"[PdfExtractorFitz] Could not list models (this is OK): {list_error}")
                    vision_models = []
                
                # Try the primary model first
                try:
                    self.__class__._ocr_client = genai.GenerativeModel(self.ocr_model)
                    print(f"[PdfExtractorFitz] ✓ Gemini Vision client initialized successfully with: {self.ocr_model}")
                except Exception as model_error:
                    # If primary model fails, try fallback models
                    print(f"[PdfExtractorFitz] Primary model '{self.ocr_model}' failed: {model_error}")
                    print(f"[PdfExtractorFitz] Trying fallback models...")
                    
                    # Try common model names in order of preference (newest first)
                    fallback_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-pro-vision", "gemini-pro", "gemini-1.0-pro-vision"]
                    for fallback_model in fallback_models:
                        if fallback_model == self.ocr_model:
                            continue  # Skip if it's the same as primary
                        try:
                            print(f"[PdfExtractorFitz] Trying fallback model: {fallback_model}")
                            self.__class__._ocr_client = genai.GenerativeModel(fallback_model)
                            print(f"[PdfExtractorFitz] ✓ SUCCESS! Initialized with fallback model: {fallback_model}")
                            # Update the model name for future reference
                            self.ocr_model = fallback_model
                            break
                        except Exception as fallback_error:
                            print(f"[PdfExtractorFitz] ✗ Fallback model '{fallback_model}' failed: {fallback_error}")
                            continue
                    else:
                        # All models failed - show helpful error
                        error_msg = (
                            f"Failed to initialize Gemini Vision client with any model.\n"
                            f"Primary model '{self.ocr_model}' failed.\n"
                            f"All fallback models also failed.\n"
                            f"Last error: {model_error}\n"
                            f"Available models (if listed): {vision_models}\n"
                            f"Please check your API key and model availability."
                        )
                        print(f"[PdfExtractorFitz] ERROR: {error_msg}")
                        raise RuntimeError(error_msg) from model_error
            except Exception as e:
                print(f"[PdfExtractorFitz] ERROR: Failed to initialize Gemini client: {e}")
                raise RuntimeError(f"Failed to initialize Gemini Vision client: {e}") from e
        
        return self.__class__._ocr_client

    def _perform_ocr(self, ocr_client: Any, image: np.ndarray) -> str:
        """Run Gemini Vision API and return extracted text.
        
        This is where the actual OCR API call happens on the Gemini Vision model.
        Line 241: response = ocr_client.generate_content([prompt, pil_image])
        """
        if not _pil_available or Image is None:
            raise RuntimeError("PIL/Pillow is required for image processing with Gemini Vision.")
        
        # Convert numpy array to PIL Image
        try:
            print(f"[PdfExtractorFitz] Converting image array to PIL Image...")
            print(f"[PdfExtractorFitz] Image shape: {image.shape}, dtype: {image.dtype}")
            pil_image = Image.fromarray(image)
            print(f"[PdfExtractorFitz] PIL Image created: size={pil_image.size}, mode={pil_image.mode}")
        except Exception as e:
            print(f"[PdfExtractorFitz] ERROR: Failed to convert image to PIL: {e}")
            return ""
        
        # Use Gemini Vision to extract text
        prompt = "Extract all text from this image. Preserve the structure and layout as much as possible. Return only the extracted text without any additional commentary."
        
        try:
            print(f"[PdfExtractorFitz] ===== CALLING GEMINI VISION API FOR OCR =====")
            print(f"[PdfExtractorFitz] Model: {self.ocr_model}")
            print(f"[PdfExtractorFitz] Prompt: {prompt[:50]}...")
            print(f"[PdfExtractorFitz] Image: {pil_image.size} pixels, mode: {pil_image.mode}")
            
            # THIS IS WHERE THE OCR API CALL HAPPENS ON THE MODEL
            # Try the API call with current model
            try:
                response = ocr_client.generate_content([prompt, pil_image])
            except Exception as api_error:
                # If API call fails with model not found, try to reinitialize with a different model
                error_str = str(api_error).lower()
                if "not found" in error_str or "404" in error_str or "not supported" in error_str:
                    print(f"[PdfExtractorFitz] Model '{self.ocr_model}' not available, trying fallback models...")
                    # Reset the client to try a different model
                    self.__class__._ocr_client = None
                    
                    # Try fallback models (newest first)
                    fallback_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-pro-vision", "gemini-pro"]
                    for fallback_model in fallback_models:
                        if fallback_model == self.ocr_model:
                            continue
                        try:
                            print(f"[PdfExtractorFitz] Trying fallback model: {fallback_model}")
                            fallback_client = genai.GenerativeModel(fallback_model)
                            response = fallback_client.generate_content([prompt, pil_image])
                            print(f"[PdfExtractorFitz] ✓ SUCCESS with fallback model: {fallback_model}")
                            # Update the model and client for future use
                            self.ocr_model = fallback_model
                            self.__class__._ocr_client = fallback_client
                            break
                        except Exception as fallback_error:
                            print(f"[PdfExtractorFitz] ✗ Fallback model '{fallback_model}' failed: {fallback_error}")
                            continue
                    else:
                        # All models failed
                        raise api_error  # Re-raise the original error
                else:
                    # Different error, re-raise it
                    raise api_error
            
            print(f"[PdfExtractorFitz] Response received: type={type(response)}")
            print(f"[PdfExtractorFitz] Response attributes: {dir(response)}")
            
            if not response:
                print(f"[PdfExtractorFitz] WARNING: Gemini API returned empty response")
                return ""
            
            # Handle different response formats
            if hasattr(response, 'text') and response.text:
                text = response.text
                print(f"[PdfExtractorFitz] ✓ OCR SUCCESS: Extracted {len(text)} characters")
                print(f"[PdfExtractorFitz] First 100 chars: {text[:100]}...")
                return text
            elif hasattr(response, 'candidates') and response.candidates:
                print(f"[PdfExtractorFitz] Response has {len(response.candidates)} candidates")
                # Try to extract text from candidates
                for idx, candidate in enumerate(response.candidates):
                    print(f"[PdfExtractorFitz] Processing candidate {idx+1}...")
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part_idx, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'text') and part.text:
                                text = part.text
                                print(f"[PdfExtractorFitz] ✓ OCR SUCCESS: Extracted {len(text)} characters from candidate {idx+1}, part {part_idx}")
                                return text
                    # Also try direct text access on candidate
                    if hasattr(candidate, 'text') and candidate.text:
                        text = candidate.text
                        print(f"[PdfExtractorFitz] ✓ OCR SUCCESS: Extracted {len(text)} characters from candidate {idx+1}")
                        return text
            
            # Debug: Print full response structure
            print(f"[PdfExtractorFitz] WARNING: Could not extract text from Gemini response")
            print(f"[PdfExtractorFitz] Response structure: {response}")
            if hasattr(response, '__dict__'):
                print(f"[PdfExtractorFitz] Response __dict__: {response.__dict__}")
            return ""
        except Exception as e:
            # Log the error so we can debug
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"[PdfExtractorFitz] ===== OCR API CALL FAILED =====")
            print(f"[PdfExtractorFitz] ERROR Type: {error_type}")
            print(f"[PdfExtractorFitz] ERROR Message: {error_msg}")
            print(f"[PdfExtractorFitz] ERROR Full details: {repr(e)}")
            
            # Check if it's an API key error
            if "api" in error_msg.lower() or "key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                print(f"[PdfExtractorFitz] ⚠️  This looks like an API key issue!")
                print(f"[PdfExtractorFitz] ⚠️  Please check GEMINI_API_KEY is set correctly in environment.")
            elif "quota" in error_msg.lower() or "429" in error_msg:
                print(f"[PdfExtractorFitz] ⚠️  This looks like a quota/rate limit issue!")
            elif "model" in error_msg.lower():
                print(f"[PdfExtractorFitz] ⚠️  This looks like a model name issue! Check if '{self.ocr_model}' is correct.")
            
            return ""

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
