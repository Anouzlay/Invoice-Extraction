from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
import numpy as np

try:
    from easyocr import Reader

    _easyocr_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    Reader = None  # type: ignore[assignment]
    _easyocr_import_error = exc

DEFAULT_PDF_PATH = Path(
    r"C:\Users\yanou\OneDrive\Bureau\Area51\WorkSpace\FreeLance\Invoice-Extraction\Invoice-Examples V20251117\Invoice-Examples\09_Scan_Michael Rütsche_2025101616412"
)
DEFAULT_LANGS_STRING = "en,de,fr"
DEFAULT_DPI = 200
DEFAULT_USE_GPU = False

_easyocr_clients: Dict[Tuple[Tuple[str, ...], bool], Any] = {}


def resolve_langs(lang_string: str | None) -> List[str]:
    if not lang_string:
        return ["en"]
    langs = [chunk.strip() for chunk in lang_string.split(",")]
    unique_langs: List[str] = []
    for lang in langs:
        if lang and lang not in unique_langs:
            unique_langs.append(lang)
    return unique_langs or ["en"]


def get_easyocr_client(langs: List[str], use_gpu: bool) -> Any:
    """Initializes or retrieves an EasyOCR Reader for the provided languages."""
    if Reader is None:
        missing_dep = "easyocr"
        if _easyocr_import_error is not None:
            missing_dep = f"{missing_dep} ({_easyocr_import_error})"
        raise RuntimeError(
            "EasyOCR is not available. Install easyocr>=1.7.1 to run this OCR test script. "
            f"Original import issue: {missing_dep}"
        )

    key = (tuple(langs), use_gpu)
    if key not in _easyocr_clients:
        _easyocr_clients[key] = Reader(langs, gpu=use_gpu, verbose=False)
    return _easyocr_clients[key]


def perform_easyocr(ocr_client: Any, image: np.ndarray) -> List[str]:
    """Runs OCR on the image using EasyOCR and returns the detected lines."""
    results = ocr_client.readtext(image, detail=1, paragraph=False)
    return [text for _, text, _ in results]


def extract_pdf_with_easyocr(pdf_path: Path, dpi: int, langs: List[str], use_gpu: bool) -> str:
    """
    Extracts text from PDF using EasyOCR.
    NOTE: Assumes the EasyOCR reader has been pre-initialized.
    """
    doc = fitz.open(pdf_path)
    try:
        page_texts: List[str] = []
        ocr_client = get_easyocr_client(langs, use_gpu)

        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                data = data[:, :, :3]

            page_texts.extend(perform_easyocr(ocr_client, data))

        return "\n".join(page_texts).strip()
    finally:
        doc.close()


def pre_initialize_easyocr_client(langs: List[str], use_gpu: bool) -> None:
    """
    Initializes the EasyOCR reader before starting the main OCR task so that
    any heavy model downloads happen up front.
    """
    print(f"Pre-initializing EasyOCR reader for languages: {langs} (GPU={use_gpu})...")
    get_easyocr_client(langs, use_gpu)
    print("Pre-initialization complete. Subsequent OCR calls will be fast.")


def main() -> None:
    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    langs = resolve_langs(DEFAULT_LANGS_STRING)
    use_gpu = DEFAULT_USE_GPU
    dpi = DEFAULT_DPI
    print(f"OCR langs: {langs} | DPI: {dpi} | GPU: {use_gpu}")
    print(f"Running OCR on: {pdf_path}")

    pre_initialize_easyocr_client(langs, use_gpu)
    text = extract_pdf_with_easyocr(pdf_path, dpi, langs, use_gpu)

    if not text:
        print("No text extracted via OCR.")
        return

    print("\n=== OCR OUTPUT START ===\n")
    print(text)
    print("\n=== OCR OUTPUT END ===")


if __name__ == "__main__":
    main()


