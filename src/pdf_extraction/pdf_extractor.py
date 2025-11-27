#!/usr/bin/env python3
"""
PDF to Doccano JSONL Extractor

Extracts text from PDF files in a format suitable for importing into Doccano
for NER annotation. Features:
- File validation (extension, integrity, corruption detection)
- Layout-aware text extraction (tables, multi-column documents)
- PyMuPDF with pdfplumber fallback
- Parallel processing with auto-detected CPU cores
- Chunking for long documents
- Preserves input file naming convention
"""

import importlib.util
import csv
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from tqdm import tqdm

# Bootstrap: Import config.py directly to setup Python path before importing from src
_config_path = Path(__file__).parent.parent / "utils" / "config.py"
spec = importlib.util.spec_from_file_location("config", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)  # This executes config.py and sets up the path

# Now we can import from src (path is set up by config.py)
from src.utils.config import EXTRACTED_DIR, RAW_PDF_DIR
from src.utils.logging import setup_logging
from src.pdf_extraction.layout_analyzer import (
    detect_headers_footers,
    remove_headers_footers,
    extract_block_font_info,
)
from src.pdf_extraction.paragraph_detector import (
    join_blocks_with_paragraph_detection,
    DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    DEFAULT_INDENTATION_THRESHOLD,
    DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
)
from src.pdf_extraction.table_extractor import (
    extract_tables_from_page,
    extract_tables_from_pdf,
    format_table_for_text_output,
    detect_table_blocks_pymupdf,
    merge_table_extraction_results,
)
from src.pdf_extraction.heading_detector import detect_headings
from src.pdf_extraction.semantic_chunker import (
    chunk_by_headings_page_bound,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_MIN_CHUNK_WORDS,
    DEFAULT_MAX_CHUNK_WORDS,
)
from src.pdf_extraction.text_cleaner import (
    clean_extracted_text_enhanced,
    get_cleaning_config,
)

# Try to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logging.warning("PyMuPDF (fitz) not available. Will use pdfplumber fallback.")

# Try to import pdfplumber
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    if not FITZ_AVAILABLE:
        raise ImportError(
            "Neither PyMuPDF nor pdfplumber is available. "
            "Please install at least one: pip install pymupdf pdfplumber"
        )

LOGGER = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Extract PDFs to Doccano JSONL format.")

# Default chunk size for splitting long documents (characters)
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200

# Default chunk sizes in words (for NER-optimized chunking)
# These are imported from semantic_chunker but kept here for backward compatibility


def clean_extracted_text(text: str, preserve_paragraphs: bool = True) -> str:
    """
    Clean and normalize extracted text from PDFs.
    
    Removes excessive newlines and whitespace while preserving meaningful
    paragraph breaks and structure.
    
    Args:
        text: Raw extracted text
        preserve_paragraphs: If True, preserves double newlines as paragraph breaks
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace non-breaking spaces and other special whitespace with regular spaces
    text = text.replace("\xa0", " ")  # Non-breaking space
    text = text.replace("\u2009", " ")  # Thin space
    text = text.replace("\u2008", " ")  # Punctuation space
    text = text.replace("\u2007", " ")  # Figure space
    text = text.replace("\u2006", " ")  # Six-per-em space
    text = text.replace("\u2005", " ")  # Four-per-em space
    text = text.replace("\u2004", " ")  # Three-per-em space
    text = text.replace("\u2003", " ")  # Em space
    text = text.replace("\u2002", " ")  # En space
    text = text.replace("\u2001", " ")  # Em quad
    text = text.replace("\u2000", " ")  # En quad
    text = text.replace("\t", " ")  # Tabs to spaces
    
    # Normalize line breaks: convert various line break types to \n
    text = text.replace("\r\n", "\n")  # Windows line breaks
    text = text.replace("\r", "\n")  # Old Mac line breaks
    
    if preserve_paragraphs:
        # Replace 3+ consecutive newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Remove single newlines that appear to be within paragraphs
        # (lines that don't end with sentence-ending punctuation)
        # But preserve double newlines (paragraph breaks)
        lines = text.split("\n")
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                # Empty line - preserve if it's part of a paragraph break
                if i > 0 and i < len(lines) - 1:
                    # Only preserve if surrounded by non-empty lines
                    if lines[i-1].strip() and lines[i+1].strip():
                        cleaned_lines.append("")
                continue
            
            # Check if this line should be merged with previous
            if cleaned_lines and cleaned_lines[-1]:
                prev_line = cleaned_lines[-1]
                # Don't merge if previous line is empty (paragraph break)
                if not prev_line:
                    cleaned_lines.append(line_stripped)
                    continue
                
                # Don't merge if previous line ends with sentence punctuation
                if re.search(r"[.!?]\s*$", prev_line):
                    cleaned_lines.append(line_stripped)
                    continue
                
                # Don't merge if current line starts with capital and is reasonably long
                # (likely a new sentence/paragraph)
                if line_stripped[0].isupper() and len(line_stripped) > 40:
                    cleaned_lines.append(line_stripped)
                    continue
                
                # Don't merge if current line looks like a header (all caps, short)
                if line_stripped.isupper() and len(line_stripped) < 100:
                    cleaned_lines.append(line_stripped)
                    continue
                
                # Merge short lines or lines that continue the previous sentence
                if len(line_stripped) < 100 or not line_stripped[0].isupper():
                    cleaned_lines[-1] = prev_line + " " + line_stripped
                    continue
            
            cleaned_lines.append(line_stripped)
        
        # Join lines, preserving double newlines for paragraphs
        text = "\n".join(cleaned_lines)
        
        # Final cleanup: normalize multiple spaces to single space (except in paragraph breaks)
        # Replace spaces around newlines
        text = re.sub(r" +", " ", text)  # Multiple spaces to single
        text = re.sub(r" *\n *", "\n", text)  # Spaces around single newlines
        text = re.sub(r" *\n\n *", "\n\n", text)  # Spaces around double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)  # More than 2 newlines to 2
    else:
        # More aggressive: remove all newlines and normalize whitespace
        text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_pdf_file(pdf_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate a PDF file before processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    if not pdf_path.suffix.lower() == ".pdf":
        return False, f"Invalid file extension: {pdf_path.suffix}"
    
    # Check if file exists
    if not pdf_path.exists():
        return False, "File does not exist"
    
    # Check if file is readable
    if not os.access(pdf_path, os.R_OK):
        return False, "File is not readable"
    
    # Check file size (not empty)
    if pdf_path.stat().st_size == 0:
        return False, "File is empty"
    
    # Try to open and validate PDF integrity
    if FITZ_AVAILABLE:
        try:
            with fitz.open(pdf_path) as doc:
                # Try to access page count to verify PDF is not corrupted
                _ = doc.page_count
                # Try to load first page if available
                if doc.page_count > 0:
                    _ = doc.load_page(0)
        except Exception as e:
            return False, f"PDF integrity check failed: {str(e)}"
    elif PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Try to access pages to verify PDF is not corrupted
                _ = len(pdf.pages)
                if len(pdf.pages) > 0:
                    _ = pdf.pages[0]
        except Exception as e:
            return False, f"PDF integrity check failed: {str(e)}"
    else:
        return False, "No PDF library available for validation"
    
    return True, None


def extract_text_with_pymupdf(
    pdf_path: Path,
    remove_headers: bool = True,
    remove_footers: bool = True,
    header_threshold: float = 0.1,
    footer_threshold: float = 0.1,
    min_header_footer_repetition: int = 3,
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    indentation_threshold: float = DEFAULT_INDENTATION_THRESHOLD,
    font_size_change_threshold: float = DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
    use_sentence_boundaries: bool = True,
    extract_tables_separately: bool = False,
    detect_headings_enabled: bool = True,
    min_font_size_increase: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF with layout-aware extraction.
    
    Args:
        pdf_path: Path to the PDF file
        remove_headers: Whether to detect and remove headers
        remove_footers: Whether to detect and remove footers
        header_threshold: Fraction of page height for header zone (default: 0.1 = top 10%)
        footer_threshold: Fraction of page height for footer zone (default: 0.1 = bottom 10%)
        min_header_footer_repetition: Minimum pages where header/footer must appear
        paragraph_gap_threshold: Minimum vertical gap for paragraph break (points)
        indentation_threshold: Minimum indentation change for paragraph (points)
        font_size_change_threshold: Minimum font size change for structure change (points)
        use_sentence_boundaries: Whether to use sentence boundary detection
        
    Returns:
        List of dictionaries with page content and metadata
    """
    pages_content = []
    pages_with_blocks = []  # For header/footer detection
    
    with fitz.open(pdf_path) as doc:
        # First pass: extract all pages with block information
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_rect = page.rect
            page_height = page_rect.height
            
            # Extract text blocks with layout information
            blocks = page.get_text("blocks")
            
            # Group blocks by content type and position
            text_blocks = []
            table_blocks = []
            all_blocks_with_metadata = []
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                
                if not text or not text.strip():
                    continue
                
                # Clean the block text immediately to remove internal excessive newlines
                cleaned_block_text = text.strip()
                
                # Extract font information
                font_info = extract_block_font_info(page, block_no)
                
                block_data = {
                    "text": cleaned_block_text,
                    "bbox": [x0, y0, x1, y1],
                    "block_no": block_no,
                    "block_type": block_type,
                    "font_info": font_info,
                }
                
                all_blocks_with_metadata.append(block_data)
                
                # Block type 5 is typically a table/image block
                if block_type == 5:
                    table_blocks.append(block_data)
                else:
                    text_blocks.append(block_data)
            
            # Detect headings if enabled (before sorting)
            if detect_headings_enabled:
                all_blocks_with_metadata = detect_headings(
                    all_blocks_with_metadata,
                    min_font_size_increase=min_font_size_increase,
                    require_bold=False,
                )
            
            # Combine text blocks, preserving order
            all_blocks = sorted(
                text_blocks + table_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])
            )
            
            # Store page data with blocks for header/footer detection
            pages_with_blocks.append({
                "page_number": page_num + 1,
                "page_height": page_height,
                "blocks": all_blocks_with_metadata,
            })
            
            # Extract full page text for fallback
            full_text = page.get_text("text")
            
            # Try to extract tables using pdfplumber if available (better table extraction)
            extracted_tables = []
            has_tables = False
            if PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        if page_num < len(pdf.pages):
                            pdf_page = pdf.pages[page_num]
                            extracted_tables = extract_tables_from_page(pdf_page)
                            has_tables = len(extracted_tables) > 0
                except Exception as e:
                    LOGGER.debug("Could not extract tables with pdfplumber for page %d: %s", page_num + 1, str(e))
            
            # If no tables found with pdfplumber, use PyMuPDF table blocks
            if not extracted_tables and table_blocks:
                pymupdf_table_blocks = detect_table_blocks_pymupdf(page)
                extracted_tables = merge_table_extraction_results(pymupdf_table_blocks, [])
                has_tables = len(extracted_tables) > 0
            
            # Build page content with enhanced paragraph detection
            if all_blocks:
                # Use enhanced paragraph detection with multiple signals
                page_text = join_blocks_with_paragraph_detection(
                    all_blocks,
                    paragraph_gap_threshold=paragraph_gap_threshold,
                    indentation_threshold=indentation_threshold,
                    font_size_change_threshold=font_size_change_threshold,
                    use_sentence_boundaries=use_sentence_boundaries,
                )
                # Final cleanup to normalize whitespace (but preserve paragraph breaks)
                # Only normalize spaces, don't merge lines aggressively
                page_text = re.sub(r" +", " ", page_text)  # Multiple spaces to single
                page_text = re.sub(r" *\n\n *", "\n\n", page_text)  # Clean spaces around paragraph breaks
                page_text = re.sub(r"\n{3,}", "\n\n", page_text)  # More than 2 newlines to 2
                page_text = page_text.strip()
            else:
                page_text = ""
            
            # If we have blocks, use them; otherwise use full text
            if page_text.strip():
                # Clean full_text as well for consistency
                full_text_cleaned = clean_extracted_text(full_text, preserve_paragraphs=True)
                # Add table information to text if not extracting separately
                if has_tables and not extract_tables_separately:
                    table_texts = []
                    for table in extracted_tables:
                        table_str = format_table_for_text_output(table, include_csv=False)
                        if table_str:
                            table_texts.append(table_str)
                    if table_texts:
                        page_text += "\n\n[TABELAS]\n\n" + "\n\n".join(table_texts)
                        # Re-clean after adding tables
                        page_text = re.sub(r" +", " ", page_text)
                        page_text = re.sub(r" *\n\n *", "\n\n", page_text)
                        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
                        page_text = page_text.strip()
                
                pages_content.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "full_text": full_text_cleaned,
                    "has_tables": has_tables,
                    "block_count": len(all_blocks),
                    "page_height": page_height,
                    "blocks": all_blocks,  # Store blocks for header/footer processing
                    "tables": extracted_tables if extract_tables_separately else [],
                })
                
                # If extracting tables separately, add table entries
                if extract_tables_separately and extracted_tables:
                    for table in extracted_tables:
                        table_entry = {
                            "page_number": page_num + 1,
                            "text": f"[TABELA {table['table_index'] + 1}]\n{table['csv_data']}",
                            "full_text": f"[TABELA {table['table_index'] + 1}]\n{table['csv_data']}",
                            "has_tables": True,
                            "block_count": 1,
                            "table_data": table,  # Full table metadata
                            "is_table_only": True,
                            "page_height": page_height,
                            "blocks": [],
                        }
                        pages_content.append(table_entry)
            elif full_text.strip():
                # Clean the fallback text
                full_text_cleaned = clean_extracted_text(full_text, preserve_paragraphs=True)
                pages_content.append({
                    "page_number": page_num + 1,
                    "text": full_text_cleaned,
                    "full_text": full_text_cleaned,
                    "has_tables": False,
                    "block_count": 0,
                    "page_height": page_height,
                    "blocks": [],
                })
        
        # Detect and remove headers/footers if requested
        if (remove_headers or remove_footers) and pages_with_blocks:
            detected = detect_headers_footers(
                pages_with_blocks,
                header_threshold=header_threshold,
                footer_threshold=footer_threshold,
                min_repetition=min_header_footer_repetition,
            )
            
            headers = detected.get("headers", []) if remove_headers else []
            footers = detected.get("footers", []) if remove_footers else []
            
            if headers or footers:
                LOGGER.info(
                    "Detected %d headers and %d footers in %s",
                    len(headers),
                    len(footers),
                    pdf_path.name,
                )
                
                # Remove headers/footers from pages
                cleaned_pages, removed_metadata = remove_headers_footers(
                    pages_with_blocks,
                    headers,
                    footers,
                    header_threshold=header_threshold,
                    footer_threshold=footer_threshold,
                )
                
                # Rebuild page text from cleaned blocks
                for i, page_data in enumerate(pages_content):
                    page_num = page_data["page_number"]
                    cleaned_page = next(
                        (p for p in cleaned_pages if p["page_number"] == page_num), None
                    )
                    
                    if cleaned_page and cleaned_page.get("blocks"):
                        # Rebuild text from cleaned blocks using enhanced paragraph detection
                        cleaned_blocks = sorted(
                            cleaned_page["blocks"], key=lambda b: (b["bbox"][1], b["bbox"][0])
                        )
                        page_text = join_blocks_with_paragraph_detection(
                            cleaned_blocks,
                            paragraph_gap_threshold=paragraph_gap_threshold,
                            indentation_threshold=indentation_threshold,
                            font_size_change_threshold=font_size_change_threshold,
                            use_sentence_boundaries=use_sentence_boundaries,
                        )
                        # Final cleanup to normalize whitespace (but preserve paragraph breaks)
                        # Only normalize spaces, don't merge lines aggressively
                        page_text = re.sub(r" +", " ", page_text)  # Multiple spaces to single
                        page_text = re.sub(r" *\n\n *", "\n\n", page_text)  # Clean spaces around paragraph breaks
                        page_text = re.sub(r"\n{3,}", "\n\n", page_text)  # More than 2 newlines to 2
                        page_text = page_text.strip()
                        page_data["text"] = page_text
                        page_data["full_text"] = page_text
                    
                    # Add header/footer metadata
                    if page_num in removed_metadata:
                        page_data["removed_headers"] = removed_metadata[page_num].get("headers", [])
                        page_data["removed_footers"] = removed_metadata[page_num].get("footers", [])
                    else:
                        page_data["removed_headers"] = []
                        page_data["removed_footers"] = []
                    
                    # Store detected headers/footers in metadata
                    page_data["detected_headers"] = headers
                    page_data["detected_footers"] = footers
    
    return pages_content


def extract_text_with_pdfplumber(
    pdf_path: Path,
    extract_tables_separately: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using pdfplumber (fallback method).
    
    Args:
        pdf_path: Path to the PDF file
        extract_tables_separately: If True, extract tables as separate entries with CSV data
        
    Returns:
        List of dictionaries with page content and metadata
    """
    pages_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text()
            
            # Extract tables using advanced extraction
            extracted_tables = extract_tables_from_page(page)
            has_tables = len(extracted_tables) > 0
            
            # Combine text and tables
            page_text = text or ""
            
            # Add table content
            if extracted_tables:
                if extract_tables_separately:
                    # Tables will be added as separate entries, just add placeholder
                    page_text += "\n\n[TABELAS DETECTADAS - ver entradas separadas]"
                else:
                    # Add table content inline as text
                    table_texts = []
                    for table in extracted_tables:
                        table_str = format_table_for_text_output(table, include_csv=False)
                        if table_str:
                            table_texts.append(table_str)
                    
                    if table_texts:
                        page_text += "\n\n[TABELAS]\n\n" + "\n\n".join(table_texts)
            
            if page_text.strip():
                # Clean the extracted text to remove excessive newlines
                page_text_cleaned = clean_extracted_text(page_text, preserve_paragraphs=True)
                page_data = {
                    "page_number": page_num,
                    "text": page_text_cleaned,
                    "full_text": page_text_cleaned,
                    "has_tables": has_tables,
                    "block_count": len(extracted_tables),
                    "tables": extracted_tables if extract_tables_separately else [],
                }
                pages_content.append(page_data)
                
                # If extracting tables separately, add table entries
                if extract_tables_separately and extracted_tables:
                    for table in extracted_tables:
                        table_entry = {
                            "page_number": page_num,
                            "text": f"[TABELA {table['table_index'] + 1}]\n{table['csv_data']}",
                            "full_text": f"[TABELA {table['table_index'] + 1}]\n{table['csv_data']}",
                            "has_tables": True,
                            "block_count": 1,
                            "table_data": table,  # Full table metadata
                            "is_table_only": True,
                        }
                        pages_content.append(table_entry)
    
    return pages_content


def extract_pdf_text(
    pdf_path: Path,
    remove_headers: bool = True,
    remove_footers: bool = True,
    header_threshold: float = 0.1,
    footer_threshold: float = 0.1,
    min_header_footer_repetition: int = 3,
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    indentation_threshold: float = DEFAULT_INDENTATION_THRESHOLD,
    font_size_change_threshold: float = DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
    use_sentence_boundaries: bool = True,
    extract_tables_separately: bool = False,
    detect_headings_enabled: bool = True,
    min_font_size_increase: float = 2.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract text from PDF using PyMuPDF with pdfplumber fallback.
    
    Args:
        pdf_path: Path to the PDF file
        remove_headers: Whether to detect and remove headers
        remove_footers: Whether to detect and remove footers
        header_threshold: Fraction of page height for header zone
        footer_threshold: Fraction of page height for footer zone
        min_header_footer_repetition: Minimum pages where header/footer must appear
        paragraph_gap_threshold: Minimum vertical gap for paragraph break (points)
        indentation_threshold: Minimum indentation change for paragraph (points)
        font_size_change_threshold: Minimum font size change for structure change (points)
        use_sentence_boundaries: Whether to use sentence boundary detection
        
    Returns:
        Tuple of (pages_content, extraction_method_used)
    """
    if FITZ_AVAILABLE:
        try:
            pages_content = extract_text_with_pymupdf(
                pdf_path,
                remove_headers=remove_headers,
                remove_footers=remove_footers,
                header_threshold=header_threshold,
                footer_threshold=footer_threshold,
                min_header_footer_repetition=min_header_footer_repetition,
                paragraph_gap_threshold=paragraph_gap_threshold,
                indentation_threshold=indentation_threshold,
                font_size_change_threshold=font_size_change_threshold,
                use_sentence_boundaries=use_sentence_boundaries,
                extract_tables_separately=extract_tables_separately,
                detect_headings_enabled=detect_headings_enabled,
                min_font_size_increase=min_font_size_increase,
            )
            return pages_content, "pymupdf"
        except Exception as e:
            LOGGER.warning(
                "PyMuPDF extraction failed for %s: %s. Trying pdfplumber fallback.",
                pdf_path.name,
                str(e),
            )
            if PDFPLUMBER_AVAILABLE:
                pages_content = extract_text_with_pdfplumber(
                    pdf_path, extract_tables_separately=extract_tables_separately
                )
                return pages_content, "pdfplumber"
            else:
                raise
    
    elif PDFPLUMBER_AVAILABLE:
        pages_content = extract_text_with_pdfplumber(
            pdf_path, extract_tables_separately=extract_tables_separately
        )
        return pages_content, "pdfplumber"
    else:
        raise RuntimeError("No PDF extraction library available")


def split_text_into_chunks(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split long text into smaller chunks with overlap.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            sentence_endings = re.finditer(r"[.!?]\s+", text[max(0, end - 200):end + 200])
            sentence_ends = list(sentence_endings)
            
            if sentence_ends:
                # Use the last sentence ending before the chunk boundary
                last_match = sentence_ends[-1]
                adjusted_end = max(0, end - 200) + last_match.end()
                if adjusted_end > start:
                    end = adjusted_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks if chunks else [text]


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_length: int = 50,
    output_format: str = "jsonl",
    remove_headers: bool = True,
    remove_footers: bool = True,
    header_threshold: float = 0.1,
    footer_threshold: float = 0.1,
    min_header_footer_repetition: int = 3,
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    indentation_threshold: float = DEFAULT_INDENTATION_THRESHOLD,
    font_size_change_threshold: float = DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
    use_sentence_boundaries: bool = True,
    extract_tables_separately: bool = False,
    chunking_strategy: str = "semantic_first",
    chunk_size_words: Optional[int] = None,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    cleaning_preset: Optional[str] = None,
    include_font_metadata: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single PDF file and generate output for Doccano (JSONL or CSV).
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output file
        chunk_size: Maximum chunk size for splitting long documents
        chunk_overlap: Overlap between chunks
        min_chunk_length: Minimum length for a chunk to be included
        output_format: Output format - 'jsonl' or 'csv'
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Validate PDF
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            return False, error_msg
        
        # Extract text (with heading detection enabled by default)
        pages_content, extraction_method = extract_pdf_text(
            pdf_path,
            remove_headers=remove_headers,
            remove_footers=remove_footers,
            header_threshold=header_threshold,
            footer_threshold=footer_threshold,
            min_header_footer_repetition=min_header_footer_repetition,
            paragraph_gap_threshold=paragraph_gap_threshold,
            indentation_threshold=indentation_threshold,
            font_size_change_threshold=font_size_change_threshold,
            use_sentence_boundaries=use_sentence_boundaries,
            extract_tables_separately=extract_tables_separately,
            detect_headings_enabled=True,  # Enable heading detection
            min_font_size_increase=2.0,  # Default threshold
        )
        
        if not pages_content:
            return False, "No text extracted from PDF"
        
        # Prepare output file (preserve naming convention)
        output_extension = ".jsonl" if output_format == "jsonl" else ".csv"
        output_filename = pdf_path.stem + output_extension
        output_path = output_dir / output_filename
        
        # Get cleaning configuration
        cleaning_config = get_cleaning_config(cleaning_preset)
        
        # Determine chunking strategy
        use_semantic_chunking = (
            chunking_strategy in ["semantic_first", "semantic"]
            and chunk_size_words is not None
        )
        
        # Generate entries
        entries = []
        
        if use_semantic_chunking:
            # Use semantic chunking (page-bound, word-based)
            # Ensure pages have blocks for heading detection
            pages_with_blocks = []
            for page_data in pages_content:
                if page_data.get("blocks"):
                    pages_with_blocks.append(page_data)
                else:
                    # If no blocks, create a minimal page entry
                    pages_with_blocks.append({
                        **page_data,
                        "blocks": [],
                    })
            
            semantic_chunks = chunk_by_headings_page_bound(
                pages_with_blocks,
                min_chunk_words=min_chunk_words,
                max_chunk_words=max_chunk_words,
                default_chunk_words=chunk_size_words or DEFAULT_CHUNK_SIZE_WORDS,
            )
            
            for chunk_idx, chunk_data in enumerate(semantic_chunks):
                chunk_text = chunk_data.get("text", "")
                chunk_word_count = chunk_data.get("word_count", 0)
                
                # Skip very short chunks
                if chunk_word_count < min_chunk_words or len(chunk_text.strip()) < min_chunk_length:
                    continue
                
                # Apply enhanced text cleaning
                cleaned_text = clean_extracted_text_enhanced(
                    chunk_text,
                    preserve_paragraphs=cleaning_config.get("preserve_paragraphs", True),
                    unicode_normalization=cleaning_config.get("unicode_normalization"),
                    remove_non_printing=cleaning_config.get("remove_non_printing", True),
                    aggressive_line_joining=cleaning_config.get("aggressive_line_joining", False),
                )
                
                # Get page data for metadata
                page_number = chunk_data.get("page_number", 0)
                page_data = next(
                    (p for p in pages_content if p.get("page_number") == page_number),
                    {},
                )
                
                # Build metadata
                metadata = {
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(semantic_chunks),
                    "extraction_method": extraction_method,
                    "chunk_type": chunk_data.get("chunk_type", "semantic"),
                    "word_count": chunk_word_count,
                    "has_tables": page_data.get("has_tables", False),
                    "block_count": page_data.get("block_count", 0),
                    "detected_headers": page_data.get("detected_headers", []),
                    "detected_footers": page_data.get("detected_footers", []),
                    "removed_headers": page_data.get("removed_headers", []),
                    "removed_footers": page_data.get("removed_footers", []),
                }
                
                # Add heading metadata if available
                if chunk_data.get("heading"):
                    metadata["heading"] = chunk_data.get("heading")
                    metadata["heading_level"] = chunk_data.get("heading_level")
                
                # Add font metadata if requested
                if include_font_metadata and page_data.get("blocks"):
                    # Collect font info from blocks in this chunk
                    font_metadata = []
                    for block in page_data.get("blocks", []):
                        if block.get("font_info"):
                            font_metadata.append({
                                "size": block["font_info"].get("size"),
                                "is_bold": block["font_info"].get("is_bold", False),
                                "is_italic": block["font_info"].get("is_italic", False),
                                "font": block["font_info"].get("font"),
                            })
                    if font_metadata:
                        metadata["font_metadata"] = font_metadata
                
                entry = {
                    "text": cleaned_text,
                    "labels": [],  # Empty labels for annotation in Doccano
                    "metadata": metadata,
                }
                
                entries.append(entry)
        else:
            # Use legacy character-based chunking
            for page_data in pages_content:
                page_text = page_data["text"]
                page_number = page_data["page_number"]
                
                # Apply enhanced text cleaning
                cleaned_page_text = clean_extracted_text_enhanced(
                    page_text,
                    preserve_paragraphs=cleaning_config.get("preserve_paragraphs", True),
                    unicode_normalization=cleaning_config.get("unicode_normalization"),
                    remove_non_printing=cleaning_config.get("remove_non_printing", True),
                    aggressive_line_joining=cleaning_config.get("aggressive_line_joining", False),
                )
                
                # Split into chunks if text is too long
                chunks = split_text_into_chunks(cleaned_page_text, chunk_size, chunk_overlap)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Skip very short chunks
                    if len(chunk.strip()) < min_chunk_length:
                        continue
                    
                    # Create entry
                    entry = {
                        "text": chunk,
                        "labels": [],  # Empty labels for annotation in Doccano
                        "metadata": {
                            "source_file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "page_number": page_number,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "extraction_method": extraction_method,
                            "chunk_type": "size_based",
                            "has_tables": page_data.get("has_tables", False),
                            "block_count": page_data.get("block_count", 0),
                            # Header/footer metadata (preserved as per user requirement)
                            "detected_headers": page_data.get("detected_headers", []),
                            "detected_footers": page_data.get("detected_footers", []),
                            "removed_headers": page_data.get("removed_headers", []),
                            "removed_footers": page_data.get("removed_footers", []),
                        },
                    }
                    
                    entries.append(entry)
        
        # Write output file
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_format == "jsonl":
            # Write JSONL file (one entry per line)
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:  # CSV format
            # Write CSV file
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "text",
                        "source_file",
                        "source_path",
                        "page_number",
                        "chunk_index",
                        "total_chunks",
                        "extraction_method",
                        "has_tables",
                        "block_count",
                    ],
                )
                writer.writeheader()
                
                for entry in entries:
                    row = {
                        "text": entry["text"],
                        "source_file": entry["metadata"]["source_file"],
                        "source_path": entry["metadata"]["source_path"],
                        "page_number": entry["metadata"]["page_number"],
                        "chunk_index": entry["metadata"]["chunk_index"],
                        "total_chunks": entry["metadata"]["total_chunks"],
                        "extraction_method": entry["metadata"]["extraction_method"],
                        "has_tables": entry["metadata"]["has_tables"],
                        "block_count": entry["metadata"]["block_count"],
                    }
                    writer.writerow(row)
        
        LOGGER.info(
            "Processed %s: %d entries written to %s",
            pdf_path.name,
            len(entries),
            output_path.name,
        )
        
        return True, None
    
    except Exception as e:
        error_msg = f"Error processing {pdf_path.name}: {str(e)}"
        LOGGER.error(error_msg, exc_info=True)
        return False, error_msg


def find_pdf_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Find all PDF files in the input directory.
    
    Args:
        input_dir: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of PDF file paths
    """
    if recursive:
        pdf_files = list(input_dir.rglob("*.pdf"))
    else:
        pdf_files = list(input_dir.glob("*.pdf"))
    
    return sorted(pdf_files)


def discover_subfolders_with_pdfs(input_dir: Path) -> Dict[Path, int]:
    """
    Discover subfolders in the input directory that contain PDF files.
    
    Args:
        input_dir: Base directory to search
        
    Returns:
        Dictionary mapping subfolder paths to PDF file counts
    """
    subfolders = {}
    
    # Check the root directory for PDFs
    root_pdfs = list(input_dir.glob("*.pdf"))
    if root_pdfs:
        subfolders[input_dir] = len(root_pdfs)
    
    # Check all immediate subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            pdf_files = list(subdir.glob("*.pdf"))
            if pdf_files:
                subfolders[subdir] = len(pdf_files)
    
    return subfolders


def prompt_folder_selection(subfolders: Dict[Path, int], input_dir: Path) -> List[Path]:
    """
    Prompt user to select which folders to process.
    
    Args:
        subfolders: Dictionary mapping folder paths to PDF counts
        input_dir: Base input directory
        
    Returns:
        List of selected folder paths to process
    """
    if len(subfolders) == 0:
        return []
    
    if len(subfolders) == 1:
        # Only one folder, no need to prompt
        return list(subfolders.keys())
    
    # Multiple folders found - prompt user
    print("\n" + "=" * 60)
    print("Multiple folders with PDF files found:")
    print("=" * 60)
    
    folder_list = sorted(subfolders.items(), key=lambda x: str(x[0]))
    for idx, (folder_path, count) in enumerate(folder_list, start=1):
        relative_path = folder_path.relative_to(input_dir) if folder_path != input_dir else Path(".")
        display_path = str(relative_path) if str(relative_path) != "." else "(root)"
        print(f"  {idx}. {display_path} - {count} PDF file(s)")
    
    print("=" * 60)
    print("Options:")
    print("  - Enter folder numbers (comma-separated, e.g., 1,2,3)")
    print("  - Enter 'all' to process all folders")
    print("  - Press Ctrl+C to cancel")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelect folders to process: ").strip().lower()
            
            if choice == "all":
                return list(subfolders.keys())
            
            # Parse comma-separated numbers
            selected_indices = [int(x.strip()) for x in choice.split(",")]
            
            # Validate indices
            valid_selections = []
            for idx in selected_indices:
                if 1 <= idx <= len(folder_list):
                    valid_selections.append(folder_list[idx - 1][0])
                else:
                    print(f"Warning: Invalid folder number {idx}, skipping...")
            
            if valid_selections:
                return valid_selections
            else:
                print("No valid folders selected. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or 'all'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)


def prompt_output_format() -> str:
    """
    Prompt user to select output format.
    
    Returns:
        Selected format: 'jsonl' or 'csv'
    """
    print("\n" + "=" * 60)
    print("Select output format:")
    print("=" * 60)
    print("  1. JSONL (Doccano-compatible)")
    print("  2. CSV (Excel-compatible)")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelect format (1 or 2): ").strip()
            
            if choice == "1":
                return "jsonl"
            elif choice == "2":
                return "csv"
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)


@app.command()
def extract(
    input_dir: Path = typer.Option(
        Path(RAW_PDF_DIR).resolve(),
        "--input-dir",
        "-i",
        help="Directory containing raw PDF files.",
    ),
    output_dir: Path = typer.Option(
        Path(EXTRACTED_DIR).resolve(),
        "--output-dir",
        "-o",
        help="Directory to save output files (JSONL or CSV).",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_CHUNK_SIZE,
        "--chunk-size",
        "-c",
        help="Maximum chunk size for splitting long documents (characters).",
    ),
    chunk_overlap: int = typer.Option(
        DEFAULT_CHUNK_OVERLAP,
        "--chunk-overlap",
        help="Overlap between chunks (characters).",
    ),
    min_chunk_length: int = typer.Option(
        50,
        "--min-chunk-length",
        help="Minimum chunk length to include (characters).",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        "-w",
        help="Maximum number of parallel workers (default: auto-detect CPU cores).",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search for PDFs recursively in subdirectories.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit the number of PDFs to process (for testing).",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip PDFs that already have corresponding output files.",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output-format",
        "-f",
        help="Output format: 'jsonl' or 'csv'. If not specified, will prompt for selection.",
    ),
    no_prompt: bool = typer.Option(
        False,
        "--no-prompt",
        help="Disable interactive prompts (requires --output-format and will process all folders).",
    ),
    remove_headers: bool = typer.Option(
        True,
        "--remove-headers/--keep-headers",
        help="Detect and remove headers from extracted text (preserved in metadata).",
    ),
    remove_footers: bool = typer.Option(
        True,
        "--remove-footers/--keep-footers",
        help="Detect and remove footers from extracted text (preserved in metadata).",
    ),
    header_threshold: float = typer.Option(
        0.1,
        "--header-threshold",
        help="Fraction of page height for header zone (default: 0.1 = top 10%%).",
    ),
    footer_threshold: float = typer.Option(
        0.1,
        "--footer-threshold",
        help="Fraction of page height for footer zone (default: 0.1 = bottom 10%%).",
    ),
    min_header_footer_repetition: int = typer.Option(
        3,
        "--min-header-footer-repetition",
        help="Minimum number of pages where header/footer must appear to be detected.",
    ),
    paragraph_gap_threshold: float = typer.Option(
        DEFAULT_PARAGRAPH_GAP_THRESHOLD,
        "--paragraph-gap-threshold",
        help="Minimum vertical gap for paragraph break (points, default: 20.0).",
    ),
    indentation_threshold: float = typer.Option(
        DEFAULT_INDENTATION_THRESHOLD,
        "--indentation-threshold",
        help="Minimum indentation change for paragraph detection (points, default: 10.0).",
    ),
    font_size_change_threshold: float = typer.Option(
        DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
        "--font-size-change-threshold",
        help="Minimum font size change for structure detection (points, default: 2.0).",
    ),
    use_sentence_boundaries: bool = typer.Option(
        True,
        "--use-sentence-boundaries/--no-sentence-boundaries",
        help="Use sentence boundary detection for paragraph detection.",
    ),
    extract_tables_separately: bool = typer.Option(
        False,
        "--extract-tables-separately",
        help="Extract tables as separate entries with CSV data (not critical for NER).",
    ),
    chunking_strategy: str = typer.Option(
        "semantic_first",
        "--chunking-strategy",
        help="Chunking strategy: 'semantic_first' (default), 'size_first', or 'page_only'. Requires --chunk-size-words.",
    ),
    chunk_size_words: Optional[int] = typer.Option(
        None,
        "--chunk-size-words",
        help="Target chunk size in words (default: 400, optimal for NER). If set, enables semantic chunking.",
    ),
    min_chunk_words: int = typer.Option(
        DEFAULT_MIN_CHUNK_WORDS,
        "--min-chunk-words",
        help="Minimum chunk size in words (default: 200).",
    ),
    max_chunk_words: int = typer.Option(
        DEFAULT_MAX_CHUNK_WORDS,
        "--max-chunk-words",
        help="Maximum chunk size in words (default: 1000).",
    ),
    cleaning_preset: Optional[str] = typer.Option(
        None,
        "--cleaning-preset",
        help="Text cleaning preset: 'default', 'aggressive', or 'preserve_structure' (default: 'default').",
    ),
    include_font_metadata: bool = typer.Option(
        False,
        "--include-font-metadata",
        help="Include font/position metadata in JSONL output (increases file size).",
    ),
):
    """
    Extract PDF files to Doccano-compatible JSONL or CSV format.
    """
    setup_logging()
    
    # Validate input directory
    if not input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        raise typer.Exit(1)
    
    if not input_dir.is_dir():
        LOGGER.error("Input path is not a directory: %s", input_dir)
        raise typer.Exit(1)
    
    # Check for PDF libraries
    if not FITZ_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        LOGGER.error(
            "No PDF extraction library available. Please install PyMuPDF or pdfplumber."
        )
        raise typer.Exit(1)
    
    # Discover subfolders with PDFs (if not recursive mode)
    pdf_files = []
    
    if recursive:
        # In recursive mode, process all PDFs found recursively (skip folder selection)
        pdf_files = find_pdf_files(input_dir, recursive=True)
        if not pdf_files:
            LOGGER.warning("No PDF files found in %s", input_dir)
            raise typer.Exit(0)
        LOGGER.info("Recursive mode: found %d PDF files in %s and subdirectories", len(pdf_files), input_dir)
    else:
        # Check folder structure
        subfolders = discover_subfolders_with_pdfs(input_dir)
        
        if not subfolders:
            LOGGER.warning("No PDF files found in %s", input_dir)
            raise typer.Exit(0)
        
        if len(subfolders) == 1:
            # Single folder, no prompting needed
            folders_to_process = list(subfolders.keys())
            folder_name = list(subfolders.keys())[0].name if list(subfolders.keys())[0] != input_dir else "(root)"
            file_count = list(subfolders.values())[0]
            LOGGER.info("Found 1 folder with PDFs: %s (%d files)", folder_name, file_count)
        else:
            # Multiple folders found
            if not no_prompt:
                folders_to_process = prompt_folder_selection(subfolders, input_dir)
            else:
                # No prompt mode - process all folders
                folders_to_process = list(subfolders.keys())
                LOGGER.info("Processing all %d folders (--no-prompt enabled)", len(subfolders))
        
        # Collect PDF files from selected folders
        for folder in folders_to_process:
            if folder == input_dir:
                # Root directory, non-recursive
                folder_files = list(folder.glob("*.pdf"))
            else:
                # Subfolder, non-recursive
                folder_files = list(folder.glob("*.pdf"))
            pdf_files.extend(folder_files)
    
    pdf_files = sorted(pdf_files)
    
    # Determine output format
    if output_format:
        output_format = output_format.lower()
        if output_format not in ["jsonl", "csv"]:
            LOGGER.error("Invalid output format: %s. Must be 'jsonl' or 'csv'", output_format)
            raise typer.Exit(1)
    elif not no_prompt:
        output_format = prompt_output_format()
    else:
        LOGGER.error("--output-format is required when --no-prompt is enabled")
        raise typer.Exit(1)
    
    LOGGER.info("Output format: %s", output_format.upper())
    
    if not pdf_files:
        LOGGER.warning("No PDF files found in selected folders")
        raise typer.Exit(0)
    
    # Apply limit if specified
    if limit:
        pdf_files = pdf_files[:limit]
        LOGGER.info("Processing limited to %d files", limit)
    
    # Filter out existing files if requested
    if skip_existing:
        original_count = len(pdf_files)
        output_extension = ".jsonl" if output_format == "jsonl" else ".csv"
        pdf_files = [
            pdf_file
            for pdf_file in pdf_files
            if not (output_dir / f"{pdf_file.stem}{output_extension}").exists()
        ]
        skipped_count = original_count - len(pdf_files)
        if skipped_count > 0:
            LOGGER.info("Skipping %d files that already have output", skipped_count)
    
    if not pdf_files:
        LOGGER.info("No files to process")
        raise typer.Exit(0)
    
    LOGGER.info("Found %d PDF files to process", len(pdf_files))
    
    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 1
        LOGGER.info("Auto-detected %d CPU cores, using %d workers", os.cpu_count(), max_workers)
    else:
        LOGGER.info("Using %d workers", max_workers)
    
    # Process PDFs in parallel
    success_count = 0
    error_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(
                process_single_pdf,
                pdf_path,
                output_dir,
                chunk_size,
                chunk_overlap,
                min_chunk_length,
                output_format,
                remove_headers,
                remove_footers,
                header_threshold,
                footer_threshold,
                min_header_footer_repetition,
                paragraph_gap_threshold,
                indentation_threshold,
                font_size_change_threshold,
                use_sentence_boundaries,
                extract_tables_separately,
                chunking_strategy,
                chunk_size_words,
                min_chunk_words,
                max_chunk_words,
                cleaning_preset,
                include_font_metadata,
            ): pdf_path
            for pdf_path in pdf_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append((pdf_path.name, error_msg))
                        LOGGER.warning("Failed to process %s: %s", pdf_path.name, error_msg)
                except Exception as e:
                    error_count += 1
                    error_msg = f"Unexpected error: {str(e)}"
                    errors.append((pdf_path.name, error_msg))
                    LOGGER.error("Unexpected error processing %s: %s", pdf_path.name, error_msg)
                
                pbar.update(1)
    
    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("Processing complete!")
    LOGGER.info("Successfully processed: %d files", success_count)
    LOGGER.info("Failed: %d files", error_count)
    
    if errors:
        LOGGER.warning("Errors encountered:")
        for filename, error_msg in errors[:10]:  # Show first 10 errors
            LOGGER.warning("  %s: %s", filename, error_msg)
        if len(errors) > 10:
            LOGGER.warning("  ... and %d more errors", len(errors) - 10)
    
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Output format: %s", output_format.upper())


if __name__ == "__main__":
    app()

