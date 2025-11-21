from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

def _get_openai_api_key() -> str | None:

    # First, try to get from Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
#Method 1 : Attribute-style access (st.secrets.OPENAI_API_KEY) 
            try:
                api_key = getattr(st.secrets, "OPENAI_API_KEY", None)
                if api_key and str(api_key).strip():
                    return str(api_key).strip()
            except (AttributeError, TypeError):
                pass
            
           # Method 2 : Dictionary-style access (st.secrets['OPENAI_API_KEY']) this is the second most common way on the streamlit cloud 

            try:
                if hasattr(st.secrets, "get"):
                    api_key = st.secrets.get("OPENAI_API_KEY")
                else:
                    api_key = st.secrets["OPENAI_API_KEY"]
                if api_key and str(api_key).strip():
                    return str(api_key).strip()
            except (KeyError, TypeError, AttributeError):
                pass
            
            # Method 3: Try accessing via __getitem__ or direct attribute
            try:
                api_key = st.secrets.__getitem__("OPENAI_API_KEY")
                if api_key and str(api_key).strip():
                    return str(api_key).strip()
            except (KeyError, AttributeError, TypeError):
                pass
    except Exception:
        # Secrets not available or not configured - this is OK for local dev
        pass
    
    # Fallback to environment variable (for local development)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    
    return None


# Set OpenAI API key from Streamlit secrets before importing crew
# This ensures secrets are used on Streamlit Cloud instead of .env files
api_key = _get_openai_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

from crew import crew
from agent.tools.pdf_extractor_fitz import PdfExtractorFitz

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "ExampleFiles"
LOGO_PATH = BASE_DIR / "public" / "logo.png"
MAX_UPLOAD_FILES = 15


def _run_pipeline(pdf_path: Path) -> Any:
    """Execute the Crew AI pipeline with the provided PDF path."""
    result = crew.kickoff(inputs={"pdf_path": str(pdf_path)})
    return result


def _cleanup_temp_files(file_infos: Iterable[dict[str, Any]]) -> None:
    """Remove temporary files tracked in session state."""
    for info in file_infos:
        temp_path = info.get("path") if isinstance(info, dict) else None
        if not temp_path:
            continue
        try:
            path_obj = Path(temp_path)
        except (TypeError, ValueError):
            continue
        if path_obj.exists():
            try:
                os.remove(path_obj)
            except OSError:
                pass


def _normalize_to_records(result: Any) -> list[dict[str, Any]]:
    """Convert the Crew result into a list of dictionaries suitable for streamlit dataframe table display."""

    def _coerce(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, dict):
            return [obj]

        if isinstance(obj, list):
            dict_items = [item for item in obj if isinstance(item, dict)]
            if dict_items and len(dict_items) == len(obj):
                return dict_items
            return [{"value": item} if not isinstance(item, dict) else item for item in obj]

        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except json.JSONDecodeError:
                return [{"raw_result": obj}]
            return _coerce(parsed)

        try:
            parsed = json.loads(str(obj))
        except json.JSONDecodeError:
            return [{"raw_result": str(obj)}]
        return _coerce(parsed)

    records = _coerce(result)

    #Ensure every record is a dictionary , this is very important for the streamlit dataframe display
    sanitized: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, dict):
            sanitized.append(item)
        else:
            sanitized.append({"value": item})
    return sanitized


def _supports_streamlit_param(func: Any, param_name: str) -> bool:
    """Check if a Streamlit callable supports a given keyword argument."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return param_name in signature.parameters


def _display_result_history(records: Iterable[dict[str, Any]]) -> None:
    """Render the accumulated extraction results as a streamlit dataframe table."""
    if not records:
        return
    
    # Convert to list if it's not already
    records_list = list(records) if not isinstance(records, list) else records
    if not records_list:
        return
    
    data_frame = pd.DataFrame(records_list)
    if data_frame.empty:
        st.warning("No extraction data available to display.")
        return

    column_labels = {
        "vendor_name_extracted": "Vendor Name (Extracted)",
        "vendor_name_matched": "Vendor Name (Matched)",
        "vendor_number": "Vendor Number",
        "cost_category": "Cost Category",
        "invoice_date": "Invoice Date",
        "due_date": "Due Date",
        "invoice_number": "Invoice Number",
        "orderer_name": "Orderer Name",
        "currency": "Currency",
        "invoice_description": "Invoice Description",
        "po_number": "PO Number",
        "payment_id": "Payment Id",
        "total_amount_without_vat": "Total Amount Without Vat",
        "total_amount_with_vat": "Total Amount With Vat",
        "total_vat_amount": "Total Vat Amount",
        "source_file": "Source File",
    }
    humanized_headers = {
        column: column_labels.get(column, column.replace("_", " ").title())
        for column in data_frame.columns
    }
    display_df = data_frame.rename(columns=humanized_headers)
    
    # Format amount columns to always show 2 decimal places
    amount_columns = [
        "Total Amount Without Vat",
        "Total Amount With Vat", 
        "Total Vat Amount"
    ]
    for col in amount_columns:
        if col in display_df.columns:
            # Format numeric values to 2 decimal places (e.g., 1300.0 -> "1300.00", 366.0 -> "366.00")
            def format_amount(x):
                if pd.isna(x) or x is None:
                    return x
                try:
                    if isinstance(x, (int, float)):
                        return f"{float(x):.2f}"
                    # Try to convert string to float
                    float_val = float(str(x))
                    return f"{float_val:.2f}"
                except (ValueError, TypeError):
                    return x
            
            display_df[col] = display_df[col].apply(format_amount)

    keyword_targets = [
        ("Vendor", "vendor_name_matched"),
        ("Vendor", "vendor_name_extracted"),
        ("Vendor ID", "vendor_number"),
        ("Invoice #", "invoice_number"),
        ("Due Date", "due_date"),
        ("Cost Category", "cost_category"),
        ("Orderer", "orderer_name"),
    ]

    latest_entry = data_frame.iloc[-1]
    keyword_badges: list[str] = []
    for label, column in keyword_targets:
        if column not in data_frame.columns:
            continue
        raw_value = latest_entry.get(column)
        if pd.isna(raw_value) or str(raw_value).strip() == "":
            continue
        badge = (
            f'<span class="keyword-chip"><span class="keyword-chip__label">{label}</span>'
            f"{raw_value}</span>"
        )
        if badge not in keyword_badges:
            keyword_badges.append(badge)

    st.markdown('<div class="card results-card">', unsafe_allow_html=True)
    st.markdown("#### Extraction Results (current session)")
    # Show count of results
    result_count = len(data_frame)
    st.caption(f"Total records: {result_count}")
    if keyword_badges:
        st.markdown(
            f'<div class="keyword-panel">{"".join(keyword_badges)}</div>',
            unsafe_allow_html=True,
        )
    # Use st.dataframe for better scrolling support and display the dataframe table 
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        height=400,
    )
    
    # Add a javascript code to 
    st.markdown(
        """
        <script>
        (function() {
            function hideExportButtons() {
                // Hide ALL element toolbars (not just in results-card)
                const toolbars = document.querySelectorAll('[data-testid="stElementToolbar"]');
                toolbars.forEach(toolbar => {
                    // Check if it's near a dataframe
                    const nearbyDataframe = toolbar.closest('[data-testid="stDataFrame"]') || 
                                           toolbar.previousElementSibling?.querySelector('[data-testid="stDataFrame"]') ||
                                           toolbar.nextElementSibling?.querySelector('[data-testid="stDataFrame"]');
                    if (nearbyDataframe || toolbar.closest('.results-card')) {
                        toolbar.style.display = 'none';
                        toolbar.style.visibility = 'hidden';
                        toolbar.style.opacity = '0';
                        toolbar.style.height = '0';
                        toolbar.style.width = '0';
                        toolbar.style.pointerEvents = 'none';
                    }
                });
                
                // Hide buttons in dataframe
                const buttons = document.querySelectorAll('[data-testid="stDataFrame"] button');
                buttons.forEach(button => {
                    button.style.display = 'none';
                    button.style.visibility = 'hidden';
                    button.style.pointerEvents = 'none';
                });
                
                // Hide any buttons with download/search/fullscreen related attributes
                const allButtons = document.querySelectorAll('button');
                allButtons.forEach(button => {
                    const title = button.getAttribute('title') || '';
                    const ariaLabel = button.getAttribute('aria-label') || '';
                    if (title.toLowerCase().includes('download') || 
                        title.toLowerCase().includes('search') || 
                        title.toLowerCase().includes('fullscreen') ||
                        ariaLabel.toLowerCase().includes('download') ||
                        ariaLabel.toLowerCase().includes('search') ||
                        ariaLabel.toLowerCase().includes('fullscreen')) {
                        const nearDataframe = button.closest('[data-testid="stDataFrame"]') ||
                                            button.closest('.results-card');
                        if (nearDataframe) {
                            button.style.display = 'none';
                            button.style.visibility = 'hidden';
                            button.style.pointerEvents = 'none';
                        }
                    }
                });
            }
            
            // Run immediately
            hideExportButtons();
            
            // Run after a short delay to catch dynamically loaded elements
            setTimeout(hideExportButtons, 100);
            setTimeout(hideExportButtons, 500);
            setTimeout(hideExportButtons, 1000);
            
            // Use MutationObserver to catch dynamically added elements
            const observer = new MutationObserver(hideExportButtons);
            observer.observe(document.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("</div>", unsafe_allow_html=True)


def _render_progress_messages(messages: list[dict[str, Any]]) -> None:
    """Display human-readable progress updates for the current batch."""
    if not messages:
        return

    status_emojis = {
        "pending": "📄",
        "running": "⏳",
        "success": "✅",
        "error": "❌",
    }

    st.markdown('<div class="card progress-card">', unsafe_allow_html=True)
    st.markdown("#### Processing status")
    for entry in messages:
        label = f"{status_emojis.get(entry.get('status'), '📄')} `{entry.get('file', 'Unknown file')}`"
        detail = entry.get("detail") or ""
        status = entry.get("status")
        if status == "success":
            st.success(f"{label} — {detail}")
        elif status == "error":
            st.error(f"{label} — {detail}")
        elif status == "running":
            st.info(f"{label} — {detail}")
        else:
            st.write(f"{label} — {detail or 'Queued'}")
    st.markdown("</div>", unsafe_allow_html=True)


def _update_progress_status(entry_index: int, status: str, detail: str) -> None:
    """Helper to safely update progress messages in session state."""
    messages: list[dict[str, Any]] = st.session_state.get("progress_messages", [])
    if entry_index < 0 or entry_index >= len(messages):
        return
    messages[entry_index]["status"] = status
    messages[entry_index]["detail"] = detail
    st.session_state.progress_messages = messages


def _inject_ui_overrides() -> None:
    """Inject custom CSS to elevate the Streamlit UI."""
    st.markdown(
        """
        <style>
            :root {
                --sigvaris-navy: #002b49;
                --sigvaris-blue: #0076c0;
                --sigvaris-light: #f0f6fb;
            }

            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 3rem;
                max-width: 1100px;
            }

            .hero-text h1 {
                font-size: 2.2rem;
                margin-bottom: 0.4rem;
                color: var(--sigvaris-navy);
            }

            .hero-text p {
                font-size: 1.05rem;
                color: #30475e;
                margin-bottom: 0;
            }

            .info-banner {
                display: flex;
                gap: 1rem;
                padding: 1rem 1.5rem;
                margin: 1.5rem 0 0.2rem;
                border-radius: 12px;
                background: var(--sigvaris-light);
                border: 1px solid rgba(0,118,192,0.12);
                flex-wrap: wrap;
            }

            .info-banner .info-step {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                font-weight: 600;
                color: var(--sigvaris-navy);
            }

            .info-banner .label {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 26px;
                height: 26px;
                border-radius: 50%;
                background: var(--sigvaris-blue);
                color: white;
                font-size: 0.85rem;
            }

            .card {
                background: white;
                border-radius: 16px;
                padding: 1.8rem;
                margin-top: 1.5rem;
                box-shadow: 0 12px 24px rgba(0, 34, 74, 0.08);
                border: 1px solid rgba(0, 34, 74, 0.04);
            }

            .card h4, .card h3, .card h2, .card h1, .card h5, .card h6 {
                margin-top: 0;
                color: var(--sigvaris-navy);
            }

            .results-card {
                position: relative;
                overflow: visible;
            }

            .results-card .stDataFrame {
                overflow-x: auto;
                overflow-y: auto;
                max-width: 100%;
            }

            .results-card div[data-testid="stDataFrame"] {
                overflow-x: auto !important;
                overflow-y: auto !important;
            }

            .results-card div[data-testid="stDataFrame"] > div {
                overflow-x: auto !important;
                overflow-y: auto !important;
            }

            /* Ensure the dataframe wrapper allows scrolling */
            .results-card [data-testid="stDataFrame"] {
                width: 100%;
                overflow-x: auto;
            }

            .results-card [data-testid="stDataFrame"] > div {
                min-width: max-content;
            }

            /* Hide the Streamlit element toolbar , this is very important  */
            [data-testid="stElementToolbar"],
            .results-card [data-testid="stElementToolbar"],
            .results-card [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
            [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
            [data-testid="stDataFrame"] ~ [data-testid="stElementToolbar"],
            [data-testid="stDataFrame"] + [data-testid="stElementToolbar"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                width: 0 !important;
                overflow: hidden !important;
                pointer-events: none !important;
            }

            /* Hide export buttons (download, search, fullscreen) - multiple selectors */
            .results-card [data-testid="stDataFrame"] button,
            .results-card [data-testid="stDataFrame"] [title*="Download"],
            .results-card [data-testid="stDataFrame"] [title*="Search"],
            .results-card [data-testid="stDataFrame"] [title*="Fullscreen"],
            .results-card [data-testid="stDataFrame"] [aria-label*="Download"],
            .results-card [data-testid="stDataFrame"] [aria-label*="Search"],
            .results-card [data-testid="stDataFrame"] [aria-label*="Fullscreen"],
            .results-card [data-testid="stDataFrame"] [data-testid*="download"],
            .results-card [data-testid="stDataFrame"] [data-testid*="search"],
            .results-card [data-testid="stDataFrame"] [data-testid*="fullscreen"] {
                display: none !important;
                visibility: hidden !important;
            }

            /* Hide the toolbar that contains export buttons */
            .results-card [data-testid="stDataFrame"] > div:first-child > div:first-child,
            .results-card [data-testid="stDataFrame"] > div > div:first-child {
                display: none !important;
            }

            /* Alternative selector for export buttons toolbar */
            .results-card [data-testid="stDataFrame"] .stToolbar,
            .results-card [data-testid="stDataFrame"] [class*="toolbar"],
            .results-card [data-testid="stDataFrame"] [class*="Toolbar"],
            .results-card [data-testid="stDataFrame"] [class*="stDataFrameToolbar"],
            .results-card [data-testid="stDataFrame"] [class*="element-container"] > div:first-child {
                display: none !important;
            }

            /* Hide any element with download/search/fullscreen icons */
            .results-card [data-testid="stDataFrame"] svg[viewBox*="24"],
            .results-card [data-testid="stDataFrame"] [role="button"] {
                display: none !important;
            }

            /* Hide toolbar buttons more aggressively */
            .results-card button[kind="header"],
            .results-card [data-testid="stDataFrame"] ~ [data-testid="stElementToolbar"],
            .results-card [data-testid="stDataFrame"] + [data-testid="stElementToolbar"] {
                display: none !important;
            }

            .results-card h4 {
                margin-bottom: 1.1rem;
            }

            .keyword-panel {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-bottom: 1.2rem;
            }

            .keyword-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.45rem 0.9rem;
                border-radius: 999px;
                background: rgba(0, 118, 192, 0.08);
                color: var(--sigvaris-navy);
                font-weight: 600;
                font-size: 0.95rem;
                border: 1px solid rgba(0, 118, 192, 0.18);
            }

            .keyword-chip__label {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.1rem 0.5rem;
                border-radius: 999px;
                background: var(--sigvaris-blue);
                color: #ffffff;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .results-card div[data-testid="stDataFrame"] {
                border-radius: 14px;
                border: 1px solid rgba(0, 43, 73, 0.12);
                overflow-x: auto !important;
                overflow-y: auto !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
                max-width: 100%;
            }

            .results-card div[data-testid="stDataFrame"] > div {
                overflow-x: auto !important;
                min-width: max-content;
            }

            .progress-card {
                margin-top: 1rem;
            }

            .results-card div[data-testid="stDataFrame"] table {
                border-collapse: separate !important;
                border-spacing: 0;
            }

            .results-card div[data-testid="stDataFrame"] thead tr:first-child th {
                background: linear-gradient(90deg, var(--sigvaris-navy), #003a66);
                color: white;
                font-weight: 600;
                padding-top: 0.85rem;
                padding-bottom: 0.85rem;
                padding-left: 1rem;
                padding-right: 1rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                border-bottom: 1px solid rgba(3, 22, 42, 0.35);
                white-space: nowrap;
                min-width: 120px;
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .results-card div[data-testid="stDataFrame"] tbody td {
                padding: 0.75rem 1rem;
                border-bottom: 1px solid rgba(0, 43, 73, 0.08);
                background: white;
                font-size: 0.95rem;
                white-space: nowrap;
                min-width: 120px;
                max-width: 400px;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .results-card div[data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
                background: rgba(0, 118, 192, 0.02);
            }

            .results-card div[data-testid="stDataFrame"] tbody tr:hover td {
                background: rgba(0, 118, 192, 0.12);
                color: var(--sigvaris-navy);
            }

            .upload-card .stFileUploader {
                padding-top: 0.6rem;
            }

            .action-card .stButton button {
                height: 48px;
                font-size: 1.05rem;
                font-weight: 600;
                background: linear-gradient(90deg, var(--sigvaris-blue), #0094da);
                border: none;
            }

            .action-card .stButton button:hover {
                box-shadow: 0 10px 20px rgba(0, 118, 192, 0.25);
            }

            .upload-hint {
                text-align: center;
                margin-top: 0.6rem;
                font-weight: 600;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                display: block;
                font-size: 0.95rem;
            }

            .upload-hint--available {
                color: #0a6c31;
                background: rgba(10, 108, 49, 0.12);
                border: 1px solid rgba(10, 108, 49, 0.2);
            }

            .upload-hint--full {
                color: #8a2c0a;
                background: rgba(255, 145, 102, 0.15);
                border: 1px solid rgba(138, 44, 10, 0.3);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="SIGVARIS Invoice Extraction",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🧾",
        layout="wide",
    )

#Old Code used to check for API key and show error if not found (only on Streamlit Cloud)
    api_key = _get_openai_api_key()
    if not api_key:
        # Check if we're likely on Streamlit Cloud (secrets should be available)
        try:
            if hasattr(st, "secrets") and st.secrets is not None:
                # We're on Streamlit Cloud but key is missing
                st.error("⚠️ **OPENAI_API_KEY not found in Streamlit secrets**")
                
                # Debug: Show what secrets are available (without revealing values)
                try:
                    secrets_keys = list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else []
                    if secrets_keys:
                        st.info(f"🔍 Available secrets keys: {', '.join(secrets_keys)}")
                    else:
                        # Try to see if secrets object has any attributes
                        attrs = [attr for attr in dir(st.secrets) if not attr.startswith('_')]
                        if attrs:
                            st.info(f"🔍 Secrets object has attributes: {', '.join(attrs[:5])}")
                except Exception:
                    pass
                
                st.error("Please add it in your app settings:")
                st.markdown("1. Go to your Streamlit Cloud app dashboard")
                st.markdown("2. Navigate to **Settings** → **Secrets** (or **Advanced settings** → **Secrets**)")
                st.markdown("3. Add your secret in TOML format (make sure there are NO brackets or sections):")
                st.code('OPENAI_API_KEY = "sk-proj-your-key-here"', language="toml")
                st.warning("⚠️ **Important:** The format should be exactly as shown above - no `[secrets]` section header!")
                st.markdown("4. Click **Save** and wait 1-2 minutes for changes to propagate")
                st.markdown("5. **Restart your app** from the Streamlit Cloud dashboard")
                st.stop()
        except Exception as e:
            # Local development - will use .env file via crew.py
            # But if we're on Streamlit Cloud and there's an error, show it
            if "streamlit" in str(type(st)).lower():
                st.warning(f"Could not access secrets: {str(e)}")
            pass

    _inject_ui_overrides()

    with st.container():
        header_left, header_right = st.columns([1, 4])
        with header_left:
            if LOGO_PATH.exists():
                logo_kwargs = {"width": 110}
                if _supports_streamlit_param(st.image, "use_container_width"):
                    logo_kwargs["use_container_width"] = False
                st.image(str(LOGO_PATH), **logo_kwargs)
            else:
                st.markdown("### SIGVARIS GROUP")
        with header_right:
            st.markdown(
                """
                <div class="hero-text">
                    <h1>Intelligent Invoice Extraction</h1>
                    <p>
                        Upload supplier invoices to run them through the OCR Crew pipeline
                        and receive structured data in seconds.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="info-banner">
            <div class="info-step"><span class="label">1</span> Upload or select a sample PDF</div>
            <div class="info-step"><span class="label">2</span> Launch the extraction workflow</div>
            <div class="info-step"><span class="label">3</span> Review the structured output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info: list[dict[str, Any]] = []
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue: list[dict[str, Any]] = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "progress_messages" not in st.session_state:
        st.session_state.progress_messages: list[dict[str, Any]] = []

    with st.container():
        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown("#### Upload PDF files")
        st.caption(
            f"Supported format: PDF • Up to {MAX_UPLOAD_FILES} files per run • Max size defined by your Streamlit deployment"
        )
        uploaded_files = st.file_uploader(
            "Drag & drop or browse your invoices",
            type=["pdf"],
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        current_selection = len(uploaded_files) if uploaded_files else 0
        if current_selection == 0:
            st.markdown(
                f"""
                <div class="upload-hint upload-hint--available">
                    You still have {MAX_UPLOAD_FILES} upload slot(s) available for this batch.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif current_selection <= MAX_UPLOAD_FILES:
            remaining_slots = MAX_UPLOAD_FILES - current_selection
            if remaining_slots > 0:
                st.markdown(
                    f"""
                    <div class="upload-hint upload-hint--available">
                        {current_selection} file(s) staged • {remaining_slots} slot(s) still available.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="upload-hint upload-hint--full">
                        Maximum of 15 files staged for this run.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Please upload at most {MAX_UPLOAD_FILES} PDF files at a time.")
            _cleanup_temp_files(st.session_state.uploaded_files_info)
            st.session_state.uploaded_files_info = []
        else:
            _cleanup_temp_files(st.session_state.uploaded_files_info)
            prepared_files: list[dict[str, Any]] = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    prepared_files.append(
                        {"path": tmp_file.name, "name": uploaded_file.name}
                    )
            st.session_state.uploaded_files_info = prepared_files
            # Reset processing state when new files are uploaded
            if st.session_state.processing_complete:
                st.session_state.processing_complete = False
            st.success(f"{len(prepared_files)} file(s) ready for extraction.")
    else:
        _cleanup_temp_files(st.session_state.uploaded_files_info)
        st.session_state.uploaded_files_info = []

    if st.session_state.uploaded_files_info:
        staged_names = ", ".join(
            info["name"] for info in st.session_state.uploaded_files_info
        )
        st.info(f"Queued for processing: {staged_names}")

    # Show success message if processing is complete and not currently processing
    if st.session_state.processing_complete and not st.session_state.is_processing:
        st.success("All uploaded documents have been processed for this session.")

    progress_placeholder = st.empty()

    st.markdown('<div class="card action-card">', unsafe_allow_html=True)
    st.markdown("#### Run the extraction workflow")
    st.caption(
        "We will launch the Crew pipeline on all queued documents and display the parsed result below."
    )
    if st.button(
        "Run extraction",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_processing,
    ):
        staged_files = [
            {"name": info["name"], "path": Path(info["path"])}
            for info in st.session_state.uploaded_files_info
            if info.get("path")
        ]
        staged_files = [file for file in staged_files if file["path"].exists()]

        if not staged_files:
            st.error("Please upload at least one PDF before running the extraction.")
        else:
            st.session_state.processing_queue = [
                {"index": idx, **file_info} for idx, file_info in enumerate(staged_files)
            ]
            st.session_state.progress_messages = [
                {
                    "index": idx,
                    "file": file_info["name"],
                    "status": "pending",
                    "detail": "Queued for processing",
                }
                for idx, file_info in enumerate(staged_files)
            ]
            st.session_state.is_processing = True
            st.session_state.processing_complete = False
            st.session_state.uploaded_files_info = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    with progress_placeholder:
        _render_progress_messages(st.session_state.progress_messages)

    # Display extraction results table - always visible when results exist
    if st.session_state.extraction_results and len(st.session_state.extraction_results) > 0:
        st.divider()
        _display_result_history(st.session_state.extraction_results)
    elif st.session_state.processing_complete and not st.session_state.is_processing:
        # Show message if processing is complete but no results
        st.divider()
        st.info("Processing completed, but no extraction results were generated.")

    if st.session_state.is_processing and st.session_state.processing_queue:
        current_job = st.session_state.processing_queue[0]
        job_index = current_job["index"]
        pdf_name = current_job["name"]
        pdf_path = current_job["path"]
        _update_progress_status(
            job_index, "running", f"Running Crew pipeline for {pdf_name}..."
        )

        try:
            with st.spinner(f"Running Crew pipeline for {pdf_name}..."):
                result = _run_pipeline(pdf_path)
            normalized_records = _normalize_to_records(result)
            if normalized_records:
                for record in normalized_records:
                    record.setdefault("source_file", pdf_name)
                st.session_state.extraction_results.extend(normalized_records)
                _update_progress_status(
                    job_index, "success", f"Extraction completed for {pdf_name}."
                )
            else:
                _update_progress_status(
                    job_index, "error", f"No structured data returned for {pdf_name}."
                )
        except Exception as exc:  # pylint: disable=broad-except
            _update_progress_status(
                job_index,
                "error",
                f"An error occurred while processing {pdf_name}: {exc}",
            )
        finally:
            if pdf_path.exists() and pdf_path.parent == Path(tempfile.gettempdir()):
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

            st.session_state.processing_queue = st.session_state.processing_queue[1:]
            if st.session_state.processing_queue:
                st.rerun()
            else:
                st.session_state.is_processing = False
                if st.session_state.progress_messages:
                    st.session_state.processing_complete = True
                # Force rerun to display results
                st.rerun()


if __name__ == "__main__":
    main()

