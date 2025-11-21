from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Only load .env file if OPENAI_API_KEY is not already set
# On Streamlit Cloud, streamlit_app.py sets it from secrets before importing this module
# This ensures .env is NEVER used on Streamlit Cloud, only for local development
if "OPENAI_API_KEY" not in os.environ:
    load_dotenv()

# Also ensure GEMINI_API_KEY is loaded from .env if not already set
# On Streamlit Cloud, streamlit_app.py sets it from secrets before importing this module
if "GEMINI_API_KEY" not in os.environ:
    load_dotenv()  # This will load GEMINI_API_KEY if it exists in .env

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from agent.tools.cost_category_lookup import CostCategoryLookup
from agent.tools.pdf_extractor_fitz import PdfExtractorFitz
from agent.tools.vendor_master_lookup import VendorMasterLookup

# Configure LLM explicitly for CrewAI
# This ensures CrewAI can find and use the OpenAI provider
# The API key should already be set in os.environ by streamlit_app.py
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Please set it in Streamlit secrets or .env file."
    )

llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-3.5-turbo" for faster/cheaper
    temperature=0.7,
)

# Get base directory for deployment-safe paths
BASE_DIR = Path(__file__).resolve().parent

# Instantiate tools with their BaseTool "config" fields using absolute paths
pdf_tool = PdfExtractorFitz(max_pages=0)  
master_tool = VendorMasterLookup(master_path=str(BASE_DIR / "knowledge" / "Vendor-Master_20251030_total.xlsx"), min_score=0.82)
cost_category_tool = CostCategoryLookup(allocation_path=str(BASE_DIR / "knowledge" / "Vendor_Cost-Category-Allocation_V20251031.xlsx"))

# Load agents from YAML using absolute path
agent_config_path = BASE_DIR / "agent" / "config" / "agent.yaml"
with open(agent_config_path, "r", encoding="utf-8") as f:
    agents_data = yaml.safe_load(f)

agents = []
agents_by_name = {}  # Dictionary to map agent names to agent objects
for agent_data in agents_data["agents"]:
    # Assign tools based on agent name
    agent_name = agent_data["name"]
    if agent_name == "Vendor_Basic_Info_Agent":
        # Vendor_Basic_Info_Agent needs pdf, vendor master, and cost category tools
        agent_tools = [pdf_tool, master_tool, cost_category_tool]
    elif agent_name == "Invoice_Details_Agent":
        # Invoice_Details_Agent now performs all amount analysis via text only, so it only needs the pdf tool
        agent_tools = [pdf_tool]
    elif agent_name == "Result_Merger_Agent":
        # Result_Merger_Agent needs no tools (just merges JSON)
        agent_tools = []
    else:
        # Default: pdf + lookup tools
        agent_tools = [pdf_tool, master_tool, cost_category_tool]
    
    agent = Agent(
        name=agent_data["name"],
        role=agent_data["role"],
        goal=agent_data["goal"],
        backstory=agent_data["backstory"],
        allow_delegation=agent_data.get("allow_delegation", False),
        verbose=agent_data.get("verbose", True),
        memory=agent_data.get("memory", False),
        tools=agent_tools,
        llm=llm  # Explicitly set the LLM
    )
    agents.append(agent)
    agents_by_name[agent_data["name"]] = agent  # Store mapping

# Load tasks from YAML using absolute path
tasks_config_path = BASE_DIR / "agent" / "config" / "tasks.yaml"
with open(tasks_config_path, "r", encoding="utf-8") as f:
    tasks_data = yaml.safe_load(f)

tasks = []
task_objects = {}  # Dictionary to store task objects for dependencies

# First pass: Create all tasks
for task_name, task_data in tasks_data.items():
    # Find the agent by name using the dictionary
    agent = agents_by_name.get(task_data["agent"])
    if agent:
        task = Task(
            description=task_data["description"],
            expected_output=task_data["expected_output"],
            agent=agent
        )
        tasks.append(task)
        task_objects[task_name] = task

# Second pass: Set up context dependencies for merge_results_task
if "merge_results_task" in task_objects:
    vendor_task = task_objects.get("extract_vendor_basic_info_task")
    invoice_task = task_objects.get("extract_invoice_details_task")
    
    if vendor_task and invoice_task:
        # Update the merge task with context from previous tasks
        merge_task = task_objects["merge_results_task"]
        # In CrewAI, we need to recreate the task with context
        # Find the merge task in the tasks list and update it
        for i, task in enumerate(tasks):
            if task == merge_task:
                # Recreate the task with context
                agent = agents_by_name.get(tasks_data["merge_results_task"]["agent"])
                tasks[i] = Task(
                    description=tasks_data["merge_results_task"]["description"],
                    expected_output=tasks_data["merge_results_task"]["expected_output"],
                    agent=agent,
                    context=[vendor_task, invoice_task]
                )
                task_objects["merge_results_task"] = tasks[i]
                break

def _normalize_payment_chunk(chunk: str) -> str:
    """
    Normalize payment chunk by:
    1. Removing all whitespace
    2. Converting O/o to 0, I/l/L to 1
    3. Extracting ONLY digits (preserving all of them)
    """
    lookup = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "L": "1",
        }
    )
    # Remove all whitespace first
    collapsed = re.sub(r"\s+", "", chunk)
    # Translate letter substitutions
    translated = collapsed.translate(lookup)
    # Extract ALL digits - this preserves the exact count
    digits_only = "".join(ch for ch in translated if ch.isdigit())
    return digits_only


def _sanitize_payment_reference(value: Any) -> Any:
    """
    Sanitize payment_id by extracting all digits and normalizing letter substitutions.
    CRITICAL: This function preserves the EXACT number of digits found - no truncation, no padding.
    Handles cases where the value might have been partially processed or truncated during agent transfer.
    """
    # Handle None or empty values
    if value is None:
        return value
    
    # Convert to string if it's not already (handles numeric types that might lose leading zeros)
    if not isinstance(value, str):
        # If it's a number, convert to string - but this might have already lost leading zeros
        # This is a fallback, but ideally the value should come as a string
        value = str(value)
    
    # Preserve the original value for comparison
    original = value
    
    # Strip whitespace but preserve the content
    normalized = value.strip()
    if not normalized:
        return value

    # Replace newlines and other whitespace characters for consistent processing
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    
    # CRITICAL: Extract all digits from the string (normalizing O/o to 0, I/l/L to 1)
    # This preserves the EXACT count of digits found in the input
    # The normalization function handles all the letter-to-digit conversions
    digits_only = _normalize_payment_chunk(normalized)
    
    # Return the normalized digits - this preserves the exact count (25, 27, etc.)
    # No truncation, no padding, no modification beyond normalization
    if digits_only:
        return digits_only
    
    # If no digits were found after normalization, return original
    # This handles edge cases where the value might be malformed
    return original


def _format_amount_field(value: Any) -> Any:
    """
    Format amount fields to always show 2 decimal places.
    Preserves null values and formats floats/integers to 2 decimal places.
    Examples: 1300.0 -> 1300.00, 366.0 -> 366.00, 0.0 -> 0.00
    """
    if value is None:
        return value
    
    try:
        # Convert to float if it's a number
        if isinstance(value, (int, float)):
            # Format to 2 decimal places
            return float(f"{value:.2f}")
        elif isinstance(value, str):
            # Try to parse as float, then format
            try:
                float_val = float(value)
                return float(f"{float_val:.2f}")
            except (ValueError, TypeError):
                # If it's not a number, return as-is
                return value
        else:
            return value
    except (ValueError, TypeError):
        # If conversion fails, return original value
        return value


def _postprocess_payment_fields(payload: Any) -> Any:
    """
    Recursively post-process the payload to sanitize payment_id fields and format amount fields.
    This ensures payment_id values are normalized and amount fields always have 2 decimal places.
    """
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            key_lower = key.lower()
            if key_lower == "payment_id":
                # CRITICAL: Normalize payment_id to preserve ALL digits
                # This handles the case where the value might have been truncated during agent transfer
                normalized = _sanitize_payment_reference(value)
                result[key] = normalized
            elif key_lower in ("total_amount_without_vat", "total_amount_with_vat", "total_vat_amount"):
                # Format amount fields to always show 2 decimal places
                # Examples: 1300.0 -> 1300.00, 366.0 -> 366.00, 0.0 -> 0.00
                result[key] = _format_amount_field(value)
            else:
                # Recursively process nested structures
                result[key] = _postprocess_payment_fields(value)
        return result
    if isinstance(payload, list):
        return [_postprocess_payment_fields(item) for item in payload]
    return payload


class PostProcessingCrew(Crew):
    def kickoff(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        result = super().kickoff(*args, **kwargs)
        # CRITICAL: Post-process the result to normalize payment_id fields and format amount fields
        # This ensures payment_id values are properly normalized and amount fields always have 2 decimal places
        # Examples: 1300.0 -> 1300.00, 366.0 -> 366.00, 0.0 -> 0.00
        processed_result = _postprocess_payment_fields(result)
        return processed_result


crew = PostProcessingCrew(
    agents=agents,
    tasks=tasks,
    process=Process.sequential
)

def run(pdf_path: str):
    # kickoff executes tasks; agent will call tools by name
    result = crew.kickoff(inputs={"pdf_path": pdf_path})
    print(result)
    return result

if __name__ == "__main__":
    run("ExampleFiles/Attachments_OCR-Project (1)/CopyBlitz_re-2025-1535.pdf")
