# tools/pdf_extractor_fitz.py
from typing import Dict
import fitz
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class PdfExtractorArgs(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")

class PdfExtractorFitz(BaseTool):
    name: str = "pdf_extractor_fitz"
    description: str = "Extract full text from a PDF using PyMuPDF (fitz)."
    args_schema: type[BaseModel] = PdfExtractorArgs
    
    max_pages: int = 0  # 0 = all

    def _run(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        parts = []
        for i, page in enumerate(doc):
            if self.max_pages and i >= self.max_pages:
                break
            parts.append(page.get_text())
        return {"pdf_path": pdf_path, "text": "\n".join(parts)}
