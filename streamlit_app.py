from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


def _get_openai_api_key() -> str | None:
    """Get OpenAI API key from Streamlit secrets or environment.
    
    Returns the API key if found, None otherwise.
    This function handles both local development and Streamlit Cloud.
    """
    # First, try to get from Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            # Method 1: Try attribute-style access (st.secrets.OPENAI_API_KEY)
            # This is the most common way on Streamlit Cloud
            try:
                api_key = getattr(st.secrets, "OPENAI_API_KEY", None)
                if api_key and str(api_key).strip():
                    return str(api_key).strip()
            except (AttributeError, TypeError):
                pass
            
            # Method 2: Try dictionary-style access (st.secrets["OPENAI_API_KEY"])
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

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "ExampleFiles"
LOGO_PATH = BASE_DIR / "public" / "logo.png"


def _run_pipeline(pdf_path: Path) -> Any:
    """Execute the Crew pipeline with the provided PDF path."""
    result = crew.kickoff(inputs={"pdf_path": str(pdf_path)})
    return result


def _normalize_to_records(result: Any) -> list[dict[str, Any]]:
    """Convert the Crew result into a list of dictionaries suitable for tabular display."""

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

    # Ensure every record is a dictionary
    sanitized: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, dict):
            sanitized.append(item)
        else:
            sanitized.append({"value": item})
    return sanitized


def _display_result_history(records: Iterable[dict[str, Any]]) -> None:
    """Render the accumulated extraction results as a table."""
    data_frame = pd.DataFrame(records)
    if data_frame.empty:
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
    }
    humanized_headers = {
        column: column_labels.get(column, column.replace("_", " ").title())
        for column in data_frame.columns
    }
    display_df = data_frame.rename(columns=humanized_headers)

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
    st.markdown("#### Extraction history (current session)")
    if keyword_badges:
        st.markdown(
            f'<div class="keyword-panel">{"".join(keyword_badges)}</div>',
            unsafe_allow_html=True,
        )
    # Use st.dataframe for better scrolling support
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        height=400,
    )
    
    # Add JavaScript to forcefully hide export buttons as a fallback
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

            /* Hide the Streamlit element toolbar (contains export buttons) - Global and specific */
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

    # Check for API key and show error if not found (only on Streamlit Cloud)
    # On local dev, crew.py will use .env file
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
                st.image(str(LOGO_PATH), use_container_width=False, width=110)
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

    if "uploaded_pdf_path" not in st.session_state:
        st.session_state.uploaded_pdf_path = None
        st.session_state.uploaded_pdf_name = None
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = []

    with st.container():
        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown("#### Upload a PDF file")
        st.caption("Supported format: PDF • Max size defined by your Streamlit deployment")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse your invoice",
            type=["pdf"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    selected_pdf_path: Path | None = None

    if uploaded_file is not None:
        file_changed = uploaded_file.name != st.session_state.uploaded_pdf_name
        if file_changed:
            previous_path = st.session_state.uploaded_pdf_path
            if previous_path and Path(previous_path).exists():
                try:
                    os.remove(previous_path)
                except OSError:
                    pass

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                st.session_state.uploaded_pdf_path = tmp_file.name
                st.session_state.uploaded_pdf_name = uploaded_file.name

        if st.session_state.uploaded_pdf_path:
            selected_pdf_path = Path(st.session_state.uploaded_pdf_path)
            st.info(f"Temporary file ready at `{selected_pdf_path}`.")
    else:
        previous_path = st.session_state.uploaded_pdf_path
        if previous_path and Path(previous_path).exists():
            try:
                os.remove(previous_path)
            except OSError:
                pass
        st.session_state.uploaded_pdf_path = None
        st.session_state.uploaded_pdf_name = None

    st.markdown('<div class="card action-card">', unsafe_allow_html=True)
    st.markdown("#### Run the extraction workflow")
    st.caption(
        "We will launch the Crew pipeline on your document and display the parsed result below."
    )
    if st.button("Run extraction", type="primary", use_container_width=True):
        if selected_pdf_path is None:
            st.error("Please upload or select a PDF before running the extraction.")
        else:
            st.info(f"Processing `{selected_pdf_path.name}`...")
            try:
                with st.spinner("Running Crew pipeline..."):
                    result = _run_pipeline(selected_pdf_path)
                normalized_records = _normalize_to_records(result)
                if normalized_records:
                    st.session_state.extraction_results.extend(normalized_records)
                    st.success("Extraction completed.")
                else:
                    st.warning("No structured data returned by the pipeline.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"An error occurred: {exc}")
            finally:
                if selected_pdf_path is not None:
                    st.session_state.uploaded_pdf_path = None
                    st.session_state.uploaded_pdf_name = None
                    if (
                        selected_pdf_path.exists()
                        and selected_pdf_path.parent == Path(tempfile.gettempdir())
                    ):
                        try:
                            os.remove(selected_pdf_path)
                        except OSError:
                            pass
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.extraction_results:
        _display_result_history(st.session_state.extraction_results)


if __name__ == "__main__":
    main()

