# tools/vat_calculator.py
from typing import Optional, Dict, List, Union, Any
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import re
from crewai.tools import BaseTool

class VATCalculationItem(BaseModel):
    amount: float = Field(..., description="Amount value (excluding or including VAT depending on context)")
    vat_rate: float = Field(..., description="VAT rate as percentage (e.g., 8.1 for 8.1%)")
    is_amount_excluding_vat: bool = Field(default=True, description="True if amount is excluding VAT, False if including VAT")

class VATCalculatorArgs(BaseModel):
    amounts: Union[List[Dict], str] = Field(..., description="List of amount dictionaries with keys: amount (float), vat_rate (float), is_amount_excluding_vat (bool)")
    currency: Optional[str] = Field(None, description="Currency code (e.g., CHF, EUR, USD)")
    
    @model_validator(mode='before')
    @classmethod
    def parse_input(cls, data: Any) -> Any:
        """Parse input from various formats - handles JSON strings from CrewAI"""
        print(f"[VATCalculatorArgs] parse_input called with type: {type(data)}, value: {str(data)[:200]}")
        
        # Store original input string for later extraction if needed
        original_input_str = None
        if isinstance(data, str):
            original_input_str = data
        elif isinstance(data, dict):
            # If data is already a dict, try to extract from its string representation
            # This helps when CrewAI has already parsed the JSON but truncated the amounts string
            original_input_str = str(data)
        
        # If data is a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
                print(f"[VATCalculatorArgs] Successfully parsed JSON string to: {type(data)}")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[VATCalculatorArgs] JSON parse failed, trying bracket matching... Error: {e}")
                # If parsing fails, try to extract JSON from the string
                # Handle cases where the string might be wrapped or malformed
                try:
                    # Try to find JSON structure in the string
                    if '{' in data or '[' in data:
                        # Find the first { and try to parse from there
                        start_idx = data.find('{')
                        if start_idx >= 0:
                            # Try to find matching closing bracket
                            bracket_count = 0
                            end_idx = start_idx
                            for i in range(start_idx, len(data)):
                                if data[i] == '{':
                                    bracket_count += 1
                                elif data[i] == '}':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_idx = i + 1
                                        break
                            if end_idx > start_idx:
                                json_str = data[start_idx:end_idx]
                                print(f"[VATCalculatorArgs] Extracted JSON substring: {json_str[:100]}...")
                                data = json.loads(json_str)
                                print(f"[VATCalculatorArgs] Successfully parsed extracted JSON")
                except Exception as e2:
                    print(f"[VATCalculatorArgs] Failed to parse JSON string: {data[:100]}... Error: {e2}")
                    pass
        
        # If data is a dict, ensure it has the right structure
        if isinstance(data, dict):
            print(f"[VATCalculatorArgs] Processing dict with keys: {list(data.keys())}")
            
            # Check if the dict has been incorrectly split (e.g., amounts string is truncated)
            # This happens when CrewAI splits JSON strings incorrectly
            if 'amounts' in data and isinstance(data['amounts'], str):
                amounts_str = data['amounts']
                print(f"[VATCalculatorArgs] amounts is a string: {amounts_str[:100]}...")
                
                # Check if the string looks incomplete (starts with [ but doesn't end with ])
                # OR if we have vat_rate/is_amount_excluding_vat keys that suggest split input
                is_incomplete = amounts_str.strip().startswith('[') and not amounts_str.strip().endswith(']')
                has_split_keys = 'vat_rate' in data or 'is_amount_excluding_vat' in data
                
                if is_incomplete or (has_split_keys and amounts_str.startswith('[{"amount"')):
                    # Try to extract all fields from the truncated string using regex first
                    # Also try to extract from the original input string if available
                    amount_val = None
                    vat_rate_val = None
                    is_excl_val = None
                    
                    # First, try to extract from the truncated amounts_str
                    amount_match = re.search(r'"amount":\s*([0-9.]+)', amounts_str)
                    if amount_match:
                        amount_val = amount_match.group(1)
                    
                    vat_rate_match = re.search(r'"vat_rate":\s*([0-9.]+)', amounts_str)
                    if vat_rate_match:
                        try:
                            vat_rate_val = float(vat_rate_match.group(1))
                        except:
                            pass
                    
                    is_excl_match = re.search(r'"is_amount_excluding_vat":\s*(true|false)', amounts_str, re.IGNORECASE)
                    if is_excl_match:
                        is_excl_val = is_excl_match.group(1).lower() == 'true'
                    
                    # If we couldn't extract from truncated string, try the original input string
                    if original_input_str and (vat_rate_val is None or is_excl_val is None):
                        if vat_rate_val is None:
                            # Try to extract vat_rate from original input string
                            vat_rate_match = re.search(r'"vat_rate":\s*([0-9.]+)', original_input_str)
                            if vat_rate_match:
                                try:
                                    vat_rate_val = float(vat_rate_match.group(1))
                                    print(f"[VATCalculatorArgs] Extracted vat_rate from original input: {vat_rate_val}")
                                except:
                                    pass
                            # Also try without quotes (in case it's a dict string representation)
                            if vat_rate_val is None:
                                vat_rate_match = re.search(r"'vat_rate':\s*([0-9.]+)", original_input_str)
                                if vat_rate_match:
                                    try:
                                        vat_rate_val = float(vat_rate_match.group(1))
                                        print(f"[VATCalculatorArgs] Extracted vat_rate from dict string: {vat_rate_val}")
                                    except:
                                        pass
                        
                        if is_excl_val is None:
                            # Try to extract is_amount_excluding_vat from original input string
                            is_excl_match = re.search(r'"is_amount_excluding_vat":\s*(true|false)', original_input_str, re.IGNORECASE)
                            if is_excl_match:
                                is_excl_val = is_excl_match.group(1).lower() == 'true'
                                print(f"[VATCalculatorArgs] Extracted is_amount_excluding_vat from original input: {is_excl_val}")
                            # Also try without quotes (in case it's a dict string representation)
                            if is_excl_val is None:
                                is_excl_match = re.search(r"'is_amount_excluding_vat':\s*(True|False|true|false)", original_input_str, re.IGNORECASE)
                                if is_excl_match:
                                    is_excl_val = is_excl_match.group(1).lower() == 'true'
                                    print(f"[VATCalculatorArgs] Extracted is_amount_excluding_vat from dict string: {is_excl_val}")
                    
                    # If we found amount, try to reconstruct
                    if amount_val:
                        # Use extracted values or fall back to dict keys
                        vat_rate = vat_rate_val
                        if vat_rate is None and 'vat_rate' in data:
                            vat_rate_str = str(data.get('vat_rate', '0.0')).strip('"\'')
                            try:
                                vat_rate = float(vat_rate_str)
                            except:
                                vat_rate = 0.0
                        if vat_rate is None:
                            vat_rate = 0.0
                        
                        is_excl = is_excl_val
                        if is_excl is None and 'is_amount_excluding_vat' in data:
                            is_excl_str = str(data.get('is_amount_excluding_vat', 'true')).strip('"\'')
                            is_excl_str = is_excl_str.rstrip('}]').strip()
                            if is_excl_str.lower() in ('true', '1', 'yes'):
                                is_excl = True
                            elif is_excl_str.lower() in ('false', '0', 'no'):
                                is_excl = False
                            else:
                                is_excl = True
                        if is_excl is None:
                            is_excl = True
                        
                        # Build complete JSON
                        complete_json = f'[{{"amount": {amount_val}, "vat_rate": {vat_rate}, "is_amount_excluding_vat": {str(is_excl).lower()}}}]'
                        try:
                            data['amounts'] = json.loads(complete_json)
                            print(f"[VATCalculatorArgs] Successfully reconstructed from truncated string: {complete_json}")
                            
                            # Clean up used keys
                            if 'vat_rate' in data:
                                del data['vat_rate']
                            if 'is_amount_excluding_vat' in data:
                                del data['is_amount_excluding_vat']
                            
                            # Skip to currency cleanup
                            if 'currency' in data and isinstance(data['currency'], str):
                                currency = data['currency'].strip('"\'')
                                data['currency'] = currency if currency else None
                            return data
                        except Exception as e:
                            print(f"[VATCalculatorArgs] JSON reconstruction failed: {e}")
                    
                    # Fallback: Try to manually reconstruct from split keys
                    if amounts_str.startswith('[{"amount"') and has_split_keys:
                        try:
                            # Extract amount value
                            amount_match = re.search(r'"amount":\s*([0-9.]+)', amounts_str)
                            if amount_match:
                                amount_val = amount_match.group(1)
                                
                                # Get vat_rate
                                vat_rate = 0.0
                                if 'vat_rate' in data:
                                    vat_rate_str = str(data.get('vat_rate', '0.0')).strip('"\'')
                                    try:
                                        vat_rate = float(vat_rate_str)
                                    except:
                                        vat_rate = 0.0
                                
                                # Get is_amount_excluding_vat (may have closing brackets attached)
                                is_excl = True
                                if 'is_amount_excluding_vat' in data:
                                    is_excl_str = str(data.get('is_amount_excluding_vat', 'true')).strip('"\'')
                                    # Remove any closing brackets that might be attached
                                    is_excl_str = is_excl_str.rstrip('}]').strip()
                                    if is_excl_str.lower() in ('true', '1', 'yes'):
                                        is_excl = True
                                    elif is_excl_str.lower() in ('false', '0', 'no'):
                                        is_excl = False
                                
                                # Build complete JSON
                                complete_json = f'[{{"amount": {amount_val}, "vat_rate": {vat_rate}, "is_amount_excluding_vat": {str(is_excl).lower()}}}]'
                                data['amounts'] = json.loads(complete_json)
                                print(f"[VATCalculatorArgs] Successfully reconstructed from split keys: {complete_json}")
                                
                                # Clean up used keys
                                if 'vat_rate' in data:
                                    del data['vat_rate']
                                if 'is_amount_excluding_vat' in data:
                                    del data['is_amount_excluding_vat']
                                
                                # Skip to currency cleanup
                                if 'currency' in data and isinstance(data['currency'], str):
                                    currency = data['currency'].strip('"\'')
                                    data['currency'] = currency if currency else None
                                return data
                        except Exception as e:
                            print(f"[VATCalculatorArgs] Manual reconstruction from split keys failed: {e}")
                    
                    # Fallback: Try to reconstruct from other dict keys that might contain the rest
                    if is_incomplete:
                        reconstructed = amounts_str
                        remaining_keys = []
                        
                        # Check if there are other keys that might contain the rest
                        for key in data.keys():
                            if key not in ['amounts', 'currency']:
                                # This might be part of the split amounts string
                                remaining_keys.append((key, data[key]))
                        
                        # Try to reconstruct by appending remaining parts
                        if remaining_keys:
                            # Sort by key to maintain order (if keys are like 'vat_rate', 'is_amount_excluding_vat')
                            for key, value in sorted(remaining_keys):
                                if isinstance(value, str):
                                    # Remove closing brackets if attached
                                    clean_value = value.rstrip('}]').strip()
                                    # Try to append if it looks like JSON continuation
                                    if clean_value.strip().startswith(('"', "'", 'true', 'false', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                                        reconstructed += ', "' + key + '": ' + clean_value
                                    else:
                                        reconstructed += ', "' + key + '": "' + clean_value + '"'
                        
                        # Try to close the JSON structure
                        if not reconstructed.strip().endswith(']'):
                            # Try to find where we are in the structure
                            if '[' in reconstructed and reconstructed.count('[') > reconstructed.count(']'):
                                # We're inside a list
                                if '{' in reconstructed and reconstructed.count('{') > reconstructed.count('}'):
                                    # We're inside an object in a list
                                    reconstructed += '}]'
                                else:
                                    reconstructed += ']'
                        
                        print(f"[VATCalculatorArgs] Attempting to reconstruct: {reconstructed[:200]}...")
                        try:
                            parsed_amounts = json.loads(reconstructed)
                            if isinstance(parsed_amounts, list):
                                data['amounts'] = parsed_amounts
                                print(f"[VATCalculatorArgs] Successfully reconstructed amounts from split input")
                                # Remove the keys we used for reconstruction
                                for key, _ in remaining_keys:
                                    if key in data:
                                        del data[key]
                            else:
                                raise ValueError("Reconstructed string is not a list")
                        except Exception as e:
                            print(f"[VATCalculatorArgs] Reconstruction failed: {e}")
                            # Fall through to normal parsing
                
                # Normal parsing attempt
                try:
                    parsed_amounts = json.loads(amounts_str)
                    print(f"[VATCalculatorArgs] Successfully parsed amounts string to: {type(parsed_amounts)}")
                    data['amounts'] = parsed_amounts
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"[VATCalculatorArgs] Failed to parse amounts string: {amounts_str[:100]}... Error: {e}")
                    # Try bracket matching for the amounts string
                    if amounts_str.strip().startswith('['):
                        try:
                            bracket_count = 0
                            end_idx = -1
                            for i, char in enumerate(amounts_str):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_idx = i + 1
                                        break
                            if end_idx > 0:
                                extracted = amounts_str[:end_idx]
                                data['amounts'] = json.loads(extracted)
                                print(f"[VATCalculatorArgs] Successfully extracted amounts using bracket matching")
                        except Exception as e2:
                            print(f"[VATCalculatorArgs] Bracket matching also failed: {e2}")
                            pass
            
            # Clean up currency field if it's a string with quotes
            if 'currency' in data and isinstance(data['currency'], str):
                currency = data['currency'].strip('"\'')
                data['currency'] = currency if currency else None
            
            return data
        
        # If data is already a list, wrap it in a dict
        if isinstance(data, list):
            print(f"[VATCalculatorArgs] Wrapping list in dict")
            return {'amounts': data, 'currency': None}
        
        return data
    
    @field_validator('amounts', mode='before')
    @classmethod
    def parse_amounts(cls, v):
        """Parse amounts field - handles string inputs"""
        print(f"[VATCalculatorArgs] parse_amounts called with type: {type(v)}, value: {str(v)[:200]}")
        
        if isinstance(v, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(v)
                # If parsed is a list, return it
                if isinstance(parsed, list):
                    print(f"[VATCalculatorArgs] Parsed string to list: {len(parsed)} items")
                    return parsed
                # If parsed is a dict with 'amounts' key, extract it
                elif isinstance(parsed, dict) and 'amounts' in parsed:
                    print(f"[VATCalculatorArgs] Extracted amounts from dict")
                    return parsed['amounts']
                return parsed
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[VATCalculatorArgs] JSON parse failed, trying bracket matching... Error: {e}")
                # Try to extract list from string if it looks like a list
                if v.strip().startswith('['):
                    try:
                        # Find the matching closing bracket
                        bracket_count = 0
                        end_idx = -1
                        for i, char in enumerate(v):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > 0:
                            extracted = v[:end_idx]
                            print(f"[VATCalculatorArgs] Extracted JSON substring: {extracted[:100]}...")
                            return json.loads(extracted)
                    except Exception as e2:
                        print(f"[VATCalculatorArgs] Bracket matching failed: {e2}")
                        pass
                
                # If all parsing fails, try to find and extract the list manually
                # Look for pattern like: [{"amount": ...}]
                list_pattern = r'\[.*?\]'
                matches = re.findall(list_pattern, v, re.DOTALL)
                if matches:
                    try:
                        # Try the longest match first
                        for match in sorted(matches, key=len, reverse=True):
                            try:
                                parsed = json.loads(match)
                                if isinstance(parsed, list):
                                    print(f"[VATCalculatorArgs] Extracted list using regex: {len(parsed)} items")
                                    return parsed
                            except:
                                continue
                    except:
                        pass
                
                print(f"[VATCalculatorArgs] All parsing attempts failed for: {v[:200]}")
                raise ValueError(f"Failed to parse amounts from string: {str(e)}")
        
        if isinstance(v, list):
            # Ensure all items are dicts and have required fields
            result = []
            for item in v:
                if isinstance(item, dict):
                    # Check if item is missing required fields (incomplete dict from truncated JSON)
                    if 'amount' in item and ('vat_rate' not in item or 'is_amount_excluding_vat' not in item):
                        print(f"[VATCalculatorArgs] WARNING: Incomplete dict in list: {item}")
                        # Try to extract missing fields from the original string representation
                        item_str = str(item)
                        # Try to extract vat_rate if missing
                        if 'vat_rate' not in item:
                            vat_rate_match = re.search(r'"vat_rate":\s*([0-9.]+)', item_str)
                            if vat_rate_match:
                                try:
                                    item['vat_rate'] = float(vat_rate_match.group(1))
                                    print(f"[VATCalculatorArgs] Extracted vat_rate from string: {item['vat_rate']}")
                                except:
                                    item['vat_rate'] = 0.0
                            else:
                                item['vat_rate'] = 0.0
                                print(f"[VATCalculatorArgs] Using default vat_rate: 0.0")
                        
                        # Try to extract is_amount_excluding_vat if missing
                        if 'is_amount_excluding_vat' not in item:
                            is_excl_match = re.search(r'"is_amount_excluding_vat":\s*(true|false)', item_str, re.IGNORECASE)
                            if is_excl_match:
                                item['is_amount_excluding_vat'] = is_excl_match.group(1).lower() == 'true'
                                print(f"[VATCalculatorArgs] Extracted is_amount_excluding_vat from string: {item['is_amount_excluding_vat']}")
                            else:
                                item['is_amount_excluding_vat'] = True
                                print(f"[VATCalculatorArgs] Using default is_amount_excluding_vat: True")
                    
                    result.append(item)
                elif isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            result.append(parsed)
                    except:
                        pass
            print(f"[VATCalculatorArgs] Processed list: {len(result)} items")
            return result
        
        print(f"[VATCalculatorArgs] Returning value as-is: {type(v)}")
        return v

class VATCalculator(BaseTool):
    name: str = "vat_calculator"
    description: str = (
        "Calculate total amounts with VAT and without VAT from a list of amounts and VAT rates. "
        "This tool performs accurate financial calculations to avoid calculation errors. "
        "The agent should detect amounts and VAT rates from the invoice text and send them to this tool for calculation. "
        "Input format: amounts should be a list of dictionaries, each with 'amount' (float), 'vat_rate' (float as percentage, e.g., 8.1 for 8.1%), and 'is_amount_excluding_vat' (bool)."
    )
    args_schema: type[BaseModel] = VATCalculatorArgs

    def _round_decimal(self, value: Decimal, precision: int = 2) -> Decimal:
        """Round decimal to specified precision using banker's rounding"""
        quantize_str = f"0.{'0' * precision}"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _run(self, amounts: List[Dict], currency: Optional[str] = None) -> Dict:
        """
        Calculate total amounts with VAT and without VAT.
        
        Args:
            amounts: List of dictionaries with keys:
                - amount: float (the amount value)
                - vat_rate: float (VAT rate as percentage, e.g., 8.1 for 8.1%)
                - is_amount_excluding_vat: bool (True if amount excludes VAT, False if includes VAT)
            currency: Optional currency code
        
        Returns:
            Dictionary with:
                - total_amount_without_vat: float
                - total_amount_with_vat: float
                - total_vat_amount: float
                - currency: str or None
                - calculation_details: list of calculated items
        """
        # Fallback: Handle case where amounts might still be a string
        if isinstance(amounts, str):
            try:
                parsed = json.loads(amounts)
                if isinstance(parsed, list):
                    amounts = parsed
                elif isinstance(parsed, dict) and 'amounts' in parsed:
                    amounts = parsed['amounts']
                    if 'currency' in parsed and currency is None:
                        currency = parsed['currency']
            except (json.JSONDecodeError, TypeError, ValueError):
                print(f"[VATCalculator] ERROR: Failed to parse amounts string: {amounts}")
                return {
                    "total_amount_without_vat": 0.0,
                    "total_amount_with_vat": 0.0,
                    "total_vat_amount": 0.0,
                    "currency": currency,
                    "calculation_details": [],
                    "error": f"Failed to parse amounts: {amounts}"
                }
        
        # Ensure amounts is a list
        if not isinstance(amounts, list):
            print(f"[VATCalculator] ERROR: amounts is not a list: {type(amounts)}")
            return {
                "total_amount_without_vat": 0.0,
                "total_amount_with_vat": 0.0,
                "total_vat_amount": 0.0,
                "currency": currency,
                "calculation_details": [],
                "error": f"amounts must be a list, got {type(amounts)}"
            }
        
        print(f"[VATCalculator] Starting calculation with {len(amounts)} items")
        
        if not amounts:
            print("[VATCalculator] WARNING: No amounts provided")
            return {
                "total_amount_without_vat": 0.0,
                "total_amount_with_vat": 0.0,
                "total_vat_amount": 0.0,
                "currency": currency,
                "calculation_details": [],
                "error": "No amounts provided"
            }
        
        total_without_vat = Decimal('0')
        total_with_vat = Decimal('0')
        total_vat = Decimal('0')
        calculation_details = []
        
        for idx, item in enumerate(amounts):
            try:
                # Extract values from item dict
                amount = Decimal(str(item.get('amount', 0)))
                vat_rate = Decimal(str(item.get('vat_rate', 0)))
                is_excluding = item.get('is_amount_excluding_vat', True)
                
                if amount <= 0:
                    print(f"[VATCalculator] WARNING: Skipping item {idx+1} - amount is zero or negative")
                    continue
                
                # Convert VAT rate from percentage to decimal (e.g., 8.1% -> 0.081)
                vat_rate_decimal = vat_rate / Decimal('100')
                
                if is_excluding:
                    # Amount is excluding VAT
                    amount_without_vat = amount
                    vat_amount = amount * vat_rate_decimal
                    amount_with_vat = amount + vat_amount
                else:
                    # Amount is including VAT
                    amount_with_vat = amount
                    # Calculate amount without VAT: amount_with_vat / (1 + vat_rate)
                    amount_without_vat = amount / (Decimal('1') + vat_rate_decimal)
                    vat_amount = amount_with_vat - amount_without_vat
                
                # Round to 2 decimal places
                amount_without_vat = self._round_decimal(amount_without_vat, 2)
                amount_with_vat = self._round_decimal(amount_with_vat, 2)
                vat_amount = self._round_decimal(vat_amount, 2)
                
                # Add to totals
                total_without_vat += amount_without_vat
                total_with_vat += amount_with_vat
                total_vat += vat_amount
                
                calculation_details.append({
                    "item_index": idx + 1,
                    "amount": float(amount),
                    "vat_rate": float(vat_rate),
                    "is_amount_excluding_vat": is_excluding,
                    "amount_without_vat": float(amount_without_vat),
                    "amount_with_vat": float(amount_with_vat),
                    "vat_amount": float(vat_amount)
                })
                
                print(f"[VATCalculator] Item {idx+1}: Amount={float(amount)}, VAT={float(vat_rate)}%, "
                      f"Excl={is_excluding}, Result: Without={float(amount_without_vat)}, "
                      f"With={float(amount_with_vat)}, VAT={float(vat_amount)}")
                
            except Exception as e:
                print(f"[VATCalculator] ERROR processing item {idx+1}: {str(e)}")
                continue
        
        # Round final totals
        total_without_vat = self._round_decimal(total_without_vat, 2)
        total_with_vat = self._round_decimal(total_with_vat, 2)
        total_vat = self._round_decimal(total_vat, 2)
        
        result = {
            "total_amount_without_vat": float(total_without_vat),
            "total_amount_with_vat": float(total_with_vat),
            "total_vat_amount": float(total_vat),
            "currency": currency,
            "calculation_details": calculation_details,
            "items_processed": len(calculation_details)
        }
        
        print(f"[VATCalculator] SUCCESS: Total without VAT={float(total_without_vat)}, "
              f"Total with VAT={float(total_with_vat)}, Total VAT={float(total_vat)}")
        print(f"[VATCalculator] Result: {result}")
        
        return result

