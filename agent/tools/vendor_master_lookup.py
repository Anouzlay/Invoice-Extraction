# tools/vendor_master_lookup.py
from typing import Optional, Dict
from pathlib import Path
import re
import pandas as pd
from rapidfuzz import fuzz
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class VendorLookupArgs(BaseModel):
    vendor_name: str = Field(..., description="Vendor name to search in the master Excel")

class VendorMasterLookup(BaseTool):
    name: str = "vendor_master_lookup"
    description: str = "Lookup vendor_number by vendor_name in a fixed Excel master."
    args_schema: type[BaseModel] = VendorLookupArgs

    # ---- Base config (you asked for it): keep config on the tool instance
    master_path: str = "knowledge/Vendor-Master_20251030_total.xlsx"
    min_score: float = 0.82
    single_token_min_score: float = 0.85
    required_token_overlap: int = 1
    _token_stop_words = {
        "switzerland",
        "suisse",
        "group",
        "holding",
        "international",
        "solutions",
        "services",
        "company",
        "co",
        "the",
        "sa",
        "se",
        "ag",
        "gmbh",
        "ltd",
        "inc",
        "medical",  # Common word that shouldn't drive matching
        "translation",  # Common word that shouldn't drive matching
        "translations",  # Common word that shouldn't drive matching
        "consulting",  # Common word that shouldn't drive matching
        "consultancy",  # Common word that shouldn't drive matching
        "technology",  # Common word that shouldn't drive matching
        "technologies",  # Common word that shouldn't drive matching
        "management",  # Common word that shouldn't drive matching
        "systems",  # Common word that shouldn't drive matching
        "system",  # Common word that shouldn't drive matching
    }

    def _normalize_company(self, s: Optional[str]) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"( ag| gmbh| se| sa)\b", "", s)
        # Preserve hyphens in identifiers (like "mt-g") by temporarily replacing them
        # Replace hyphens with a placeholder, then restore them after removing other special chars
        s = re.sub(r"([a-z0-9])-([a-z0-9])", r"\1_HYPHEN_\2", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"_HYPHEN_", "-", s)  # Restore hyphens
        s = re.sub(r"\s+", " ", s)
        return s

    def _tokenize(self, raw: Optional[str]) -> set[str]:
        norm = self._normalize_company(raw)
        if not norm:
            return set()
        tokens = set()
        for token in norm.split():
            # Include tokens that are >= 3 chars OR contain hyphens (unique identifiers like "mt-g")
            if token not in self._token_stop_words and (len(token) >= 3 or '-' in token):
                tokens.add(token)
        return tokens
    
    def _get_token_weight(self, token: str) -> float:
        """
        Calculate weight for a token based on its uniqueness.
        Shorter tokens, tokens with special chars (like hyphens), and unique identifiers get higher weight.
        """
        weight = 1.0
        
        # Shorter tokens are more unique (e.g., "mt-g" is more unique than "medical")
        if len(token) <= 5:
            weight *= 2.0  # Boost short tokens significantly
        elif len(token) <= 8:
            weight *= 1.5  # Moderate boost for medium tokens
        
        # Tokens with hyphens or special patterns are often unique identifiers
        if '-' in token or '_' in token:
            weight *= 2.5  # Strong boost for identifiers like "mt-g"
        
        # Tokens that are all lowercase letters (common words) get reduced weight
        if token.isalpha() and len(token) > 6:
            weight *= 0.7  # Reduce weight for long common words
        
        return weight

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
        name_col = "VENDORORGANIZATIONNAME"
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
        df["_tokens"] = df[name_col].astype(str).map(self._tokenize)

        q = self._normalize_company(vendor_name)
        if not q or df.empty:
            print(f"[VendorMasterLookup] No query or empty dataframe")
            return {"vendor_number": None, "matched_vendor_name": None, "match_score": 0.0}

        print(f"[VendorMasterLookup] Loaded {len(df)} vendors from master file")

        q_tokens_norm = q.split()
        q_first_token = q_tokens_norm[0] if q_tokens_norm else ""
        query_tokens = self._tokenize(vendor_name)

        best_idx = None
        best_rank = -1.0
        best_base_score = 0.0

        candidate_norms = df["_name_norm"].tolist()
        candidate_tokens_list = df["_tokens"].tolist()

        # Calculate weighted token overlap - prioritize unique identifiers
        ranked_candidates: list[tuple[int, float]] = []
        if query_tokens:
            for idx, tokens in enumerate(candidate_tokens_list):
                overlapping_tokens = tokens & query_tokens
                if overlapping_tokens:
                    # Calculate weighted overlap - unique tokens count more
                    weighted_overlap = sum(self._get_token_weight(token) for token in overlapping_tokens)
                    ranked_candidates.append((idx, weighted_overlap))

        # Fallback to the whole dataframe if nothing shares a strong token
        if ranked_candidates:
            # Sort by weighted overlap (descending) to prioritize matches with unique tokens
            ranked_candidates.sort(key=lambda x: x[1], reverse=True)
            candidate_space = ranked_candidates
        else:
            candidate_space = [(idx, 0.0) for idx in range(len(candidate_norms))]

        for idx, weighted_overlap in candidate_space:
            candidate_norm = candidate_norms[idx]
            base_score = fuzz.WRatio(q, candidate_norm) / 100.0

            prefix_bonus = 0.0
            candidate_tokens = candidate_norm.split()
            candidate_first_token = candidate_tokens[0] if candidate_tokens else ""
            if q_first_token and candidate_first_token == q_first_token:
                prefix_bonus += 0.12  # Increased bonus for identical leading tokens (often unique identifiers)

            # Weighted token bonus - unique tokens contribute more
            # Check for exact matches of unique tokens (like "mt-g")
            unique_token_bonus = 0.0
            candidate_token_set = candidate_tokens_list[idx] if idx < len(candidate_tokens_list) else set()
            for query_token in query_tokens:
                if query_token in candidate_token_set:
                    token_weight = self._get_token_weight(query_token)
                    unique_token_bonus += 0.08 * token_weight  # Weighted bonus based on token uniqueness
            
            # General token overlap bonus (reduced weight)
            token_bonus = 0.02 * weighted_overlap

            candidate_rank = base_score + prefix_bonus + unique_token_bonus + token_bonus
            if candidate_rank > best_rank:
                best_rank = candidate_rank
                best_idx = idx
                best_base_score = base_score

        if best_idx is None:
            print(f"[VendorMasterLookup] No match found for '{vendor_name}'")
            return {"vendor_number": None, "matched_vendor_name": None, "match_score": 0.0}

        score = best_base_score
        row = df.iloc[best_idx]

        match_tokens = self._tokenize(row[name_col])
        token_overlap = len(query_tokens & match_tokens)
        effective_min_score = self.min_score
        if token_overlap >= 1:
            score = max(score, self.single_token_min_score)
            effective_min_score = min(effective_min_score, self.single_token_min_score)

        is_confident = not (
            score < effective_min_score
            or (
                self.required_token_overlap > 0
                and query_tokens
                and match_tokens
                and token_overlap < self.required_token_overlap
            )
        )

        if not is_confident:
            reason = []
            if score < effective_min_score:
                reason.append(
                    f"score {score:.3f} < min_score {effective_min_score}"
                )
            if (
                self.required_token_overlap > 0
                and query_tokens
                and match_tokens
                and token_overlap < self.required_token_overlap
            ):
                reason.append(
                    "insufficient unique token overlap "
                    f"({token_overlap}/{self.required_token_overlap})"
                )

            debug_msg = (
                "[VendorMasterLookup] Match rejected: "
                + "; ".join(reason)
                + f". Candidate was '{row[name_col]}'"
            )
            print(debug_msg)

            return {
                "vendor_number": None,
                "matched_vendor_name": None,
                "match_score": score,
            }

        result = {
            "vendor_number": (str(row[id_col]) if id_col else None),
            "matched_vendor_name": str(row[name_col]),
            "match_score": score
        }
        print(f"[VendorMasterLookup] SUCCESS: Match found with score {score:.3f}")
        print(f"[VendorMasterLookup] Result: {result}")
        return result