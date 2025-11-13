# tools/vendor_master_lookup.py
from typing import Optional, Dict
from pathlib import Path
import re
import pandas as pd
from rapidfuzz import process, fuzz
from pydantic import BaseModel, Field
from crewai_tools import BaseTool

class VendorLookupArgs(BaseModel):
    vendor_name: str = Field(..., description="Vendor name to search in the master Excel")

class VendorMasterLookup(BaseTool):
    name: str = "vendor_master_lookup"
    description: str = "Lookup vendor_number by vendor_name in a fixed Excel master."
    args_schema: type[BaseModel] = VendorLookupArgs

    # ---- Base config (you asked for it): keep config on the tool instance
    master_path: str = "knowledge/Vendor-Master_20251030_total.xlsx"
    min_score: float = 0.82

    def _normalize_company(self, s: Optional[str]) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"( ag| gmbh| se| sa)\b", "", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _run(self, vendor_name: str) -> Dict:
        print(f"[VendorMasterLookup] Searching for vendor: '{vendor_name}'")
        master = Path(self.master_path)
        if not master.exists():
            error_msg = f"Master not found: {master}"
            print(f"[VendorMasterLookup] ERROR: {error_msg}")
            return {
                "vendor_number": None,
                "matched_vendor_name": None,
                "match_score": 0.0,
                "error": error_msg
            }

        df = pd.read_excel(master)
        
        # Use specific column names: ADDRESSDESCRIPTION for vendor name, VENDORACCOUNTNUMBER for vendor number
        name_col = "ADDRESSDESCRIPTION"
        id_col = "VENDORACCOUNTNUMBER"
        
        # Check if required columns exist
        if name_col not in df.columns:
            error_msg = f"Column '{name_col}' not found in Excel file. Available columns: {list(df.columns)}"
            print(f"[VendorMasterLookup] ERROR: {error_msg}")
            return {
                "vendor_number": None,
                "matched_vendor_name": None,
                "match_score": 0.0,
                "error": error_msg
            }
        if id_col not in df.columns:
            error_msg = f"Column '{id_col}' not found in Excel file. Available columns: {list(df.columns)}"
            print(f"[VendorMasterLookup] ERROR: {error_msg}")
            return {
                "vendor_number": None,
                "matched_vendor_name": None,
                "match_score": 0.0,
                "error": error_msg
            }

        df["_name_norm"] = df[name_col].astype(str).map(self._normalize_company)

        q = self._normalize_company(vendor_name)
        if not q or df.empty:
            print(f"[VendorMasterLookup] No query or empty dataframe")
            return {"vendor_number": None, "matched_vendor_name": None, "match_score": 0.0}

        print(f"[VendorMasterLookup] Loaded {len(df)} vendors from master file")
        best = process.extractOne(q, df["_name_norm"].tolist(), scorer=fuzz.WRatio)
        if not best:
            print(f"[VendorMasterLookup] No match found for '{vendor_name}'")
            return {"vendor_number": None, "matched_vendor_name": None, "match_score": 0.0}

        idx = best[2]
        score = float(best[1]) / 100.0
        row = df.iloc[idx]

        if score < self.min_score:
            result = {"vendor_number": None, "matched_vendor_name": str(row[name_col]), "match_score": score}
            print(f"[VendorMasterLookup] Match found but score too low: {score:.3f} < {self.min_score}")
            print(f"[VendorMasterLookup] Result: {result}")
            return result

        result = {
            "vendor_number": (str(row[id_col]) if id_col else None),
            "matched_vendor_name": str(row[name_col]),
            "match_score": score
        }
        print(f"[VendorMasterLookup] SUCCESS: Match found with score {score:.3f}")
        print(f"[VendorMasterLookup] Result: {result}")
        return result