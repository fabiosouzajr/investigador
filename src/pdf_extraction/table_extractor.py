#!/usr/bin/env python3
"""
Table Extraction Module for PDF Extraction

Provides advanced table extraction capabilities:
- Structured table extraction using pdfplumber
- CSV format output
- Table boundary detection
- Multi-page table handling
- Table metadata preservation
"""

import csv
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def table_to_csv(table: List[List[Any]], delimiter: str = ",") -> str:
    """
    Convert a table (list of lists) to CSV format string.
    
    Args:
        table: List of rows, where each row is a list of cells
        delimiter: CSV delimiter (default: comma)
        
    Returns:
        CSV-formatted string
    """
    if not table:
        return ""
    
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    
    for row in table:
        # Convert None to empty string and clean cell values
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Convert to string and strip whitespace
                cell_str = str(cell).strip()
                cleaned_row.append(cell_str)
        writer.writerow(cleaned_row)
    
    return output.getvalue()


def extract_tables_from_page(
    page: Any,  # pdfplumber.Page
    table_settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using pdfplumber.
    
    Args:
        page: pdfplumber Page object
        table_settings: Optional settings for table extraction
        
    Returns:
        List of table dictionaries with structure and CSV data
    """
    if not PDFPLUMBER_AVAILABLE:
        return []
    
    if table_settings is None:
        # Try multiple strategies in order of preference
        # Start with "text" strategy (most flexible, works without explicit lines)
        # Fall back to "lines" if needed
        strategies = [
            {
                "vertical_strategy": "text",  # Most flexible - works without explicit lines
                "horizontal_strategy": "text",
                "min_words_vertical": 2,
                "min_words_horizontal": 1,
            },
            {
                "vertical_strategy": "lines",  # Fallback: use lines if available
                "horizontal_strategy": "lines",
                "min_words_vertical": 2,
            "min_words_horizontal": 1,
            },
        ]
        
        tables = []
        for strategy_settings in strategies:
            try:
                tables = page.extract_tables(strategy_settings)
                if tables:
                    LOGGER.debug("Found %d tables using %s strategy", len(tables), strategy_settings["vertical_strategy"])
                    break
            except Exception as e:
                LOGGER.debug("Strategy %s failed: %s", strategy_settings["vertical_strategy"], str(e))
                continue
    else:
        # Use provided settings
    try:
        tables = page.extract_tables(table_settings)
    except Exception as e:
        LOGGER.warning("Error extracting tables from page: %s", str(e))
        return []
    
    extracted_tables = []
    
    for table_idx, table in enumerate(tables):
        if not table:
            continue
        
        # Filter out false positives:
        # - Single-row tables (likely headers or false detections)
        # - Tables with only 1 column (not really a table)
        num_rows = len(table)
        if num_rows < 2:
            LOGGER.debug("Skipping single-row table (likely false positive)")
            continue
        
        num_cols = max(len(row) for row in table) if table else 0
        if num_cols < 2:
            LOGGER.debug("Skipping single-column table (likely false positive)")
            continue
        
        # Filter out tables that are just page numbers or headers
        # Check if table looks like a header (short text, few words)
        first_row_text = ' '.join([str(cell) for cell in table[0] if cell]).strip()
        if len(first_row_text) < 20 and num_rows <= 2:
            LOGGER.debug("Skipping table that looks like a header: %s", first_row_text[:50])
            continue
        
        # Convert table to CSV
        csv_data = table_to_csv(table)
        
        # Get table bounding box if available
        bbox = None
        try:
            # Try to get table bounding box from page
            table_bbox = page.find_tables(table_settings)
            if table_bbox and table_idx < len(table_bbox):
                bbox_obj = table_bbox[table_idx]
                bbox = [
                    bbox_obj.bbox[0],  # x0
                    bbox_obj.bbox[1],  # y0
                    bbox_obj.bbox[2],  # x1
                    bbox_obj.bbox[3],  # y1
                ]
        except Exception:
            pass
        
        extracted_tables.append({
            "table_index": table_idx,
            "csv_data": csv_data,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "bbox": bbox,
            "raw_table": table,  # Keep raw table for reference
        })
    
    return extracted_tables


def extract_tables_from_pdf(
    pdf_path: Any,  # Path or pdfplumber.PDF
    table_settings: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract tables from all pages of a PDF.
    
    Args:
        pdf_path: Path to PDF file or pdfplumber.PDF object
        table_settings: Optional settings for table extraction
        
    Returns:
        Dictionary mapping page numbers to lists of extracted tables
    """
    if not PDFPLUMBER_AVAILABLE:
        return {}
    
    all_tables = {}
    
    try:
        if isinstance(pdf_path, str) or hasattr(pdf_path, "open"):
            # It's a path
            pdf = pdfplumber.open(pdf_path)
        else:
            # Assume it's already a pdfplumber.PDF object
            pdf = pdf_path
        
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = extract_tables_from_page(page, table_settings)
            if tables:
                all_tables[page_num] = tables
        
        # Close if we opened it
        if hasattr(pdf, "close"):
            pdf.close()
            
    except Exception as e:
        LOGGER.error("Error extracting tables from PDF %s: %s", pdf_path, str(e))
    
    return all_tables


def format_table_for_text_output(
    table: Dict[str, Any],
    include_csv: bool = False,
) -> str:
    """
    Format a table for inclusion in text output.
    
    Args:
        table: Table dictionary with CSV data
        include_csv: Whether to include CSV data in output
        
    Returns:
        Formatted string representation of table
    """
    if include_csv and table.get("csv_data"):
        return f"[TABELA {table.get('table_index', 0) + 1}]\n{table['csv_data']}"
    else:
        # Simple text representation
        raw_table = table.get("raw_table", [])
        if not raw_table:
            return ""
        
        lines = []
        for row in raw_table:
            row_str = " | ".join(str(cell) if cell else "" for cell in row)
            lines.append(row_str)
        
        return "\n".join(lines)


def detect_table_blocks_pymupdf(
    page: Any,  # fitz.Page
) -> List[Dict[str, Any]]:
    """
    Detect table blocks using PyMuPDF (for pages where pdfplumber isn't used).
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        List of potential table block dictionaries
    """
    if not FITZ_AVAILABLE:
        return []
    
    table_blocks = []
    blocks = page.get_text("blocks")
    
    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        
        # Block type 5 is typically a table/image block
        if block_type == 5 and text and text.strip():
            table_blocks.append({
                "text": text.strip(),
                "bbox": [x0, y0, x1, y1],
                "block_no": block_no,
                "block_type": block_type,
            })
    
    return table_blocks


def merge_table_extraction_results(
    pymupdf_tables: List[Dict[str, Any]],
    pdfplumber_tables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge table extraction results from different methods.
    
    Args:
        pymupdf_tables: Tables detected by PyMuPDF
        pdfplumber_tables: Tables extracted by pdfplumber
        
    Returns:
        Merged list of tables
    """
    # Prefer pdfplumber tables (more structured)
    if pdfplumber_tables:
        return pdfplumber_tables
    
    # Fallback to PyMuPDF table blocks
    if pymupdf_tables:
        # Convert PyMuPDF blocks to table format
        result = []
        for idx, block in enumerate(pymupdf_tables):
            # Try to parse as CSV-like structure
            text = block.get("text", "")
            lines = text.split("\n")
            
            # Simple heuristic: if text has multiple lines with similar structure, treat as table
            if len(lines) > 2:
                # Convert to simple table structure
                table_rows = []
                for line in lines:
                    # Try to split by common delimiters
                    if "|" in line:
                        cells = [cell.strip() for cell in line.split("|")]
                    elif "\t" in line:
                        cells = [cell.strip() for cell in line.split("\t")]
                    else:
                        # Single cell
                        cells = [line.strip()]
                    table_rows.append(cells)
                
                if table_rows:
                    csv_data = table_to_csv(table_rows)
                    result.append({
                        "table_index": idx,
                        "csv_data": csv_data,
                        "num_rows": len(table_rows),
                        "num_cols": max(len(row) for row in table_rows) if table_rows else 0,
                        "bbox": block.get("bbox"),
                        "raw_table": table_rows,
                    })
        
        return result
    
    return []

