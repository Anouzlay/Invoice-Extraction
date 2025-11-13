# tools/cost_category_lookup.py
from typing import Optional, Dict
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class CostCategoryLookupArgs(BaseModel):
    vendor_number: str = Field(..., description="Vendor number (VENDORACCOUNTNUMBER) to search in the cost category allocation Excel")

class CostCategoryLookup(BaseTool):
    name: str = "cost_category_lookup"
    description: str = "Lookup Cost Category by vendor_number in the cost category allocation Excel file."
    args_schema: type[BaseModel] = CostCategoryLookupArgs

    # Base config
    allocation_path: str = "knowledge/Vendor_Cost-Category-Allocation_V20251031.xlsx"

    def _clean_number(self, value) -> Optional[str]:
        """Convert number to clean string without .0 suffix"""
        if pd.isna(value) or value is None:
            return None
        # Convert to string first
        str_value = str(value).strip()
        # If empty string after stripping, return None
        if not str_value:
            return None
        # If it's a float representation (ends with .0), convert to int then string
        try:
            # Try to convert to float first
            float_val = float(str_value)
            # If it's a whole number (no decimal part), return as integer string
            if float_val.is_integer():
                return str(int(float_val))
            # Otherwise return as is (has decimal part)
            return str_value
        except (ValueError, TypeError):
            # If conversion fails, return as string
            return str_value

    def _run(self, vendor_number: str) -> Dict:
        print(f"[CostCategoryLookup] Searching for vendor_number: '{vendor_number}'")
        allocation_file = Path(self.allocation_path)
        if not allocation_file.exists():
            error_msg = f"Cost category allocation file not found: {allocation_file}"
            print(f"[CostCategoryLookup] ERROR: {error_msg}")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "error": error_msg
            }

        try:
            df = pd.read_excel(allocation_file)
            print(f"[CostCategoryLookup] Loaded Excel file. Available columns: {list(df.columns)}")
            print(f"[CostCategoryLookup] Total rows: {len(df)}")
        except Exception as e:
            error_msg = f"Error reading Excel file: {str(e)}"
            print(f"[CostCategoryLookup] ERROR: {error_msg}")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "error": error_msg
            }
        
        # Column names: VENDORACCOUNTNUMBER for vendor number, Account Number for account number
        vendor_col = "VENDORACCOUNTNUMBER"
        account_col = "Account Number"  # This is the Excel column name we read from
        
        # Check if required columns exist
        if vendor_col not in df.columns:
            error_msg = f"Column '{vendor_col}' not found in Excel file. Available columns: {list(df.columns)}"
            print(f"[CostCategoryLookup] ERROR: {error_msg}")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "error": error_msg
            }
        if account_col not in df.columns:
            error_msg = f"Column '{account_col}' not found in Excel file. Available columns: {list(df.columns)}"
            print(f"[CostCategoryLookup] ERROR: {error_msg}")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "error": error_msg
            }

        # Convert vendor_number to string for comparison (clean it first)
        vendor_number_str = self._clean_number(vendor_number)
        if not vendor_number_str:
            print(f"[CostCategoryLookup] Invalid vendor_number: '{vendor_number}'")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "matched": False
            }
        vendor_number_str = vendor_number_str.strip()
        
        # Convert the vendor column to string and clean numbers
        df[vendor_col] = df[vendor_col].apply(self._clean_number)
        df[vendor_col] = df[vendor_col].astype(str).str.strip()
        
        # Find exact match
        matches = df[df[vendor_col] == vendor_number_str]
        
        if matches.empty:
            print(f"[CostCategoryLookup] No match found for vendor_number '{vendor_number}'")
            return {
                "cost_category": None,
                "vendor_number": vendor_number,
                "matched": False
            }

        # Get the first match (should be unique, but take first if multiple)
        row = matches.iloc[0]
        
        # IMPORTANT: We read from "Account Number" column in Excel, but return as "cost_category" field
        # Debug: Print the raw value from Excel
        raw_account_value = row[account_col]  # Reading from "Account Number" column
        print(f"[CostCategoryLookup] Raw Account Number value from Excel: {raw_account_value} (type: {type(raw_account_value)})")
        
        # Extract and clean the cost category value from Account Number column
        # This will be returned as "cost_category" field (not "account_number")
        cost_category = self._clean_number(raw_account_value)
        
        # If cost_category is None but we found a match, it means the Account Number is empty/NaN
        if cost_category is None:
            print(f"[CostCategoryLookup] WARNING: Account Number column is empty/NaN for vendor_number '{vendor_number}'")
        
        result = {
            "cost_category": cost_category,
            "vendor_number": vendor_number,
            "matched": True
        }
        print(f"[CostCategoryLookup] SUCCESS: Found cost_category '{cost_category}' for vendor_number '{vendor_number}'")
        print(f"[CostCategoryLookup] Result: {result}")
        return result

