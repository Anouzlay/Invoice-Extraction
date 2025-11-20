from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List

import fitz
import numpy as np

try:
    from paddleocr import PaddleOCR
    _paddleocr_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore[assignment]
    _paddleocr_import_error = exc

DEFAULT_PDF_PATH = Path(
    r"C:\Users\yanou\OneDrive\Bureau\Area51\WorkSpace\FreeLance\Invoice-Extraction\Invoice-Examples V20251117\Invoice-Examples\05_Rechnung 121620 Sigvaris.pdf"
)
DEFAULT_LANGS_STRING = "en,de,fr"
DEFAULT_DPI = 200
DEFAULT_USE_ANGLE_CLS = True

_ocr_clients: Dict[str, Any] = {}
_PADDLE_SUPPORTS_TEXTLINE_ORIENTATION = False

if PaddleOCR is not None:
    try:
        signature = inspect.signature(PaddleOCR.__init__)
    except (TypeError, ValueError):
        signature = None
    if signature:
        _PADDLE_SUPPORTS_TEXTLINE_ORIENTATION = "use_textline_orientation" in signature.parameters


def resolve_langs(lang_string: str | None) -> List[str]:
    if not lang_string:
        return ["en"]
    langs = [chunk.strip() for chunk in lang_string.split(",")]
    unique_langs: List[str] = []
    for lang in langs:
        if lang and lang not in unique_langs:
            unique_langs.append(lang)
    return unique_langs or ["en"]


def get_ocr_client(lang: str, use_angle_cls: bool) -> Any:
    """Initializes or retrieves a PaddleOCR client for a specific language."""
    if PaddleOCR is None:
        missing_dep = "paddleocr / paddlepaddle"
        if _paddleocr_import_error is not None:
            missing_dep = f"{missing_dep} ({_paddleocr_import_error})"
        raise RuntimeError(
            "PaddleOCR is not available. Install paddleocr>=2.7.0 and "
            "paddlepaddle>=2.5.1 to run this OCR test script. "
            f"Original import issue: {missing_dep}"
        )

    if lang not in _ocr_clients:
        init_kwargs: Dict[str, Any] = {
            "lang": lang,
            "show_log": False,
        }
        orientation_kwarg = (
            "use_textline_orientation"
            if _PADDLE_SUPPORTS_TEXTLINE_ORIENTATION
            else "use_angle_cls"
        )
        init_kwargs[orientation_kwarg] = use_angle_cls
        try:
            # This is the line that triggers model download/load the first time per language
            _ocr_clients[lang] = PaddleOCR(**init_kwargs)
        except ModuleNotFoundError as exc:
            if exc.name == "paddle":
                raise RuntimeError(
                    "PaddleOCR requires the 'paddlepaddle' package. "
                    "Add `paddlepaddle>=2.5.1` to your environment."
                ) from exc
            raise
        except (TypeError, ValueError) as exc:
            if "show_log" in str(exc):
                init_kwargs.pop("show_log", None)
                _ocr_clients[lang] = PaddleOCR(**init_kwargs)
            else:
                raise
    return _ocr_clients[lang]


def perform_ocr(ocr_client: Any, image: Any, use_angle_cls: bool) -> Any:
    """Performs OCR on the given image using the initialized client."""
    if hasattr(ocr_client, "predict"):
        try:
            return ocr_client.predict(image)
        except TypeError:
            pass

    candidate_kwargs: list[dict[str, Any]] = []
    if use_angle_cls:
        candidate_kwargs.extend(
            [
                {"cls": True},
                {"use_cls": True},
                {"use_angle_cls": True},
            ]
        )
    candidate_kwargs.append({})

    last_exc: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return ocr_client.ocr(image, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return None


def extract_pdf_with_ocr(pdf_path: Path, dpi: int, langs: List[str], use_angle_cls: bool) -> str:
    """
    Extracts text from PDF using OCR. 
    NOTE: Assumes all clients for 'langs' have been pre-initialized via a call to pre_initialize_ocr_clients().
    """
    doc = fitz.open(pdf_path)
    try:
        page_texts: List[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                data = data[:, :, :3]
            
            # --- OPTIMIZED LOOP: Retrieves clients from cache ---
            for lang in langs:
                # Client is retrieved here. If pre_initialize_ocr_clients was run,
                # this only takes milliseconds.
                ocr = get_ocr_client(lang, use_angle_cls) 
                result = perform_ocr(ocr, data, use_angle_cls) or []
                for page_result in result:
                    for line in page_result:
                        page_texts.append(line[1][0])
            # ----------------------------------------------------

        return "\n".join(page_texts).strip()
    finally:
        doc.close()

def pre_initialize_ocr_clients(langs: List[str], use_angle_cls: bool) -> None:
    """
    CRITICAL: This function initializes all required language models BEFORE 
    starting the main OCR task. This ensures the slow downloading/loading 
    is done once at the beginning, outside the time-sensitive execution loop.
    """
    print(f"Pre-initializing PaddleOCR clients for languages: {langs}...")
    for lang in langs:
        # Calls get_ocr_client, which triggers the model download/load if not in cache.
        get_ocr_client(lang, use_angle_cls)
    print("Pre-initialization complete. Subsequent OCR calls will be fast.")


def main() -> None:
    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    langs = resolve_langs(DEFAULT_LANGS_STRING)
    use_angle_cls = DEFAULT_USE_ANGLE_CLS
    dpi = DEFAULT_DPI
    print(f"OCR langs: {langs} | DPI: {dpi} | Angle CLS: {use_angle_cls}")
    print(f"Running OCR on: {pdf_path}")

    # --- CRITICAL CHANGE: Pre-initialize all clients here ---
    # This moves the long download/load step to the start.
    pre_initialize_ocr_clients(langs, use_angle_cls) 
    # --------------------------------------------------------

    text = extract_pdf_with_ocr(pdf_path, dpi, langs, use_angle_cls)
    if not text:
        print("No text extracted via OCR.")
        return

    print("\n=== OCR OUTPUT START ===\n")
    print(text)
    print("\n=== OCR OUTPUT END ===")


if __name__ == "__main__":
    main()