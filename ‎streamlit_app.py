from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

# Set OpenAI API key from Streamlit secrets before importing crew
# This ensures secrets are used on Streamlit Cloud instead of .env files
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("⚠️ OPENAI_API_KEY not found in Streamlit secrets. Please add it in your app settings.")
        st.stop()
except (AttributeError, KeyError, FileNotFoundError):
    # Streamlit secrets not available - this should only happen in local dev
    # Will fall back to .env file in crew.py for local development
    pass

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

    column_config: dict[str, Any] = {}
    if "invoice_date" in data_frame.columns:
        column_config[humanized_headers["invoice_date"]] = st.column_config.DateColumn(
            "Invoice Date", format="YYYY-MM-DD"
        )
    if "due_date" in data_frame.columns:
        column_config[humanized_headers["due_date"]] = st.column_config.DateColumn(
            "Due Date", format="YYYY-MM-DD"
        )

    st.markdown('<div class="card results-card">', unsafe_allow_html=True)
    st.markdown("#### Extraction history (current session)")
    if keyword_badges:
        st.markdown(
            f'<div class="keyword-panel">{"".join(keyword_badges)}</div>',
            unsafe_allow_html=True,
        )
    table_height = max(300, min(600, 140 + len(display_df) * 48))
    st.dataframe(
        display_df,
        use_container_width=True,
        height=table_height,
        column_config=column_config,
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
                overflow: hidden;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
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
                text-transform: uppercase;
                letter-spacing: 0.05em;
                border-bottom: 1px solid rgba(3, 22, 42, 0.35);
            }

            .results-card div[data-testid="stDataFrame"] tbody td {
                padding: 0.75rem 1rem;
                border-bottom: 1px solid rgba(0, 43, 73, 0.08);
                background: white;
                font-size: 0.95rem;
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


