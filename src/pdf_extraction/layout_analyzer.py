#!/usr/bin/env python3
"""
Layout Analysis Module for PDF Extraction

Provides functions for detecting and handling layout elements such as:
- Headers and footers
- Paragraph boundaries
- Multi-column layouts
- Reading order
- Font-based structure detection
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Default thresholds for header/footer detection
DEFAULT_HEADER_THRESHOLD = 0.1  # Top 10% of page
DEFAULT_FOOTER_THRESHOLD = 0.15  # Bottom 15% of page (increased from 10% to catch more footers)
DEFAULT_MIN_REPETITION = 2  # Minimum pages for header/footer detection (reduced from 3 for better detection)
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Text similarity threshold for matching (reduced from 0.8 for better matching)


def estimate_font_size(page: fitz.Page, block_no: int) -> Optional[float]:
    """
    Estimate font size for a block.
    
    Args:
        page: PyMuPDF page object
        block_no: Block number
        
    Returns:
        Font size in points, or None if not available
    """
    try:
        page_dict = page.get_text("dict")
        if block_no < len(page_dict.get("blocks", [])):
            block = page_dict["blocks"][block_no]
            if "lines" in block and len(block["lines"]) > 0:
                if "spans" in block["lines"][0] and len(block["lines"][0]["spans"]) > 0:
                    return block["lines"][0]["spans"][0].get("size")
    except (IndexError, KeyError, TypeError, AttributeError):
        pass
    return None


def extract_block_font_info(page: fitz.Page, block_no: int) -> Dict[str, Any]:
    """
    Extract font information for a block.
    
    Args:
        page: PyMuPDF page object
        block_no: Block number
        
    Returns:
        Dictionary with font information (size, flags, font)
    """
    font_info = {
        "size": None,
        "flags": None,
        "font": None,
        "is_bold": False,
        "is_italic": False,
    }
    
    try:
        page_dict = page.get_text("dict")
        if block_no < len(page_dict.get("blocks", [])):
            block = page_dict["blocks"][block_no]
            if "lines" in block and len(block["lines"]) > 0:
                if "spans" in block["lines"][0] and len(block["lines"][0]["spans"]) > 0:
                    span = block["lines"][0]["spans"][0]
                    font_info["size"] = span.get("size")
                    font_info["flags"] = span.get("flags", 0)
                    font_info["font"] = span.get("font")
                    # Check if bold (flag 16) or italic (flag 1)
                    flags = font_info["flags"] or 0
                    font_info["is_bold"] = bool(flags & 16)
                    font_info["is_italic"] = bool(flags & 1)
    except (IndexError, KeyError, TypeError, AttributeError):
        pass
    
    return font_info


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison (remove extra whitespace, lowercase).
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return " ".join(text.strip().split()).lower()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts (simple word overlap).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(normalize_text_for_comparison(text1).split())
    words2 = set(normalize_text_for_comparison(text2).split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def is_in_header_zone(
    y0: float, y1: float, page_height: float, threshold: float = DEFAULT_HEADER_THRESHOLD
) -> bool:
    """
    Check if a block is in the header zone (top of page).
    
    Args:
        y0: Top Y coordinate of block
        y1: Bottom Y coordinate of block
        page_height: Total page height
        threshold: Fraction of page height for header zone (default: 0.1 = top 10%)
        
    Returns:
        True if block is in header zone
    """
    header_zone_height = page_height * threshold
    return y0 < header_zone_height


def is_in_footer_zone(
    y0: float, y1: float, page_height: float, threshold: float = DEFAULT_FOOTER_THRESHOLD
) -> bool:
    """
    Check if a block is in the footer zone (bottom of page).
    
    Args:
        y0: Top Y coordinate of block
        y1: Bottom Y coordinate of block
        page_height: Total page height
        threshold: Fraction of page height for footer zone (default: 0.1 = bottom 10%)
        
    Returns:
        True if block is in footer zone
    """
    footer_zone_start = page_height * (1 - threshold)
    return y1 > footer_zone_start


def detect_headers_footers(
    pages_content: List[Dict[str, Any]],
    header_threshold: float = DEFAULT_HEADER_THRESHOLD,
    footer_threshold: float = DEFAULT_FOOTER_THRESHOLD,
    min_repetition: int = DEFAULT_MIN_REPETITION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, List[str]]:
    """
    Detect headers and footers by analyzing repetitive patterns across pages.
    
    Args:
        pages_content: List of page content dictionaries with blocks and page info
        header_threshold: Fraction of page height for header zone
        footer_threshold: Fraction of page height for footer zone
        min_repetition: Minimum number of pages where header/footer must appear
        similarity_threshold: Minimum similarity score for matching
        
    Returns:
        Dictionary with 'headers' and 'footers' lists
    """
    headers = []
    footers = []
    
    if not pages_content:
        return {"headers": headers, "footers": footers}
    
    # Collect candidate headers and footers from each page
    header_candidates = []
    footer_candidates = []
    
    for page_data in pages_content:
        page_height = page_data.get("page_height")
        blocks = page_data.get("blocks", [])
        
        if not page_height or not blocks:
            continue
        
        for block in blocks:
            bbox = block.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x0, y0, x1, y1 = bbox
            text = block.get("text", "").strip()
            
            if not text:
                continue
            
            # Check if in header zone
            if is_in_header_zone(y0, y1, page_height, header_threshold):
                header_candidates.append({
                    "text": text,
                    "normalized": normalize_text_for_comparison(text),
                    "page": page_data.get("page_number", 0),
                })
            
            # Check if in footer zone
            if is_in_footer_zone(y0, y1, page_height, footer_threshold):
                footer_candidates.append({
                    "text": text,
                    "normalized": normalize_text_for_comparison(text),
                    "page": page_data.get("page_number", 0),
                })
    
    # Group similar headers
    if header_candidates:
        header_groups = _group_similar_texts(
            header_candidates, similarity_threshold, min_repetition
        )
        headers = [group["text"] for group in header_groups]
    
    # Group similar footers
    if footer_candidates:
        footer_groups = _group_similar_texts(
            footer_candidates, similarity_threshold, min_repetition
        )
        footers = [group["text"] for group in footer_groups]
    
    LOGGER.debug(
        "Detected %d headers and %d footers", len(headers), len(footers)
    )
    
    return {"headers": headers, "footers": footers}


def _group_similar_texts(
    candidates: List[Dict[str, Any]], similarity_threshold: float, min_repetition: int
) -> List[Dict[str, Any]]:
    """
    Group similar text candidates together.
    
    Args:
        candidates: List of candidate dictionaries with 'text', 'normalized', 'page'
        similarity_threshold: Minimum similarity for grouping
        min_repetition: Minimum occurrences to include in result
        
    Returns:
        List of grouped texts with occurrence counts
    """
    if not candidates:
        return []
    
    groups = []
    used = set()
    
    for i, candidate in enumerate(candidates):
        if i in used:
            continue
        
        # Find all similar candidates
        similar = [candidate]
        used.add(i)
        
        for j, other in enumerate(candidates[i + 1:], start=i + 1):
            if j in used:
                continue
            
            similarity = calculate_text_similarity(
                candidate["normalized"], other["normalized"]
            )
            
            if similarity >= similarity_threshold:
                similar.append(other)
                used.add(j)
        
        # Only include if appears on enough pages
        unique_pages = len(set(c["page"] for c in similar))
        if unique_pages >= min_repetition:
            # Use the most common text variant
            text_counter = Counter(c["text"] for c in similar)
            most_common_text = text_counter.most_common(1)[0][0]
            
            groups.append({
                "text": most_common_text,
                "occurrences": len(similar),
                "pages": unique_pages,
            })
    
    return groups


def remove_headers_footers(
    pages_content: List[Dict[str, Any]],
    headers: List[str],
    footers: List[str],
    header_threshold: float = DEFAULT_HEADER_THRESHOLD,
    footer_threshold: float = DEFAULT_FOOTER_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Remove detected headers and footers from page content.
    
    Args:
        pages_content: List of page content dictionaries
        headers: List of header texts to remove
        footers: List of footer texts to remove
        header_threshold: Fraction of page height for header zone
        footer_threshold: Fraction of page height for footer zone
        
    Returns:
        Tuple of (cleaned_pages_content, removed_content_metadata)
        where removed_content_metadata contains lists of removed headers/footers per page
    """
    cleaned_pages = []
    removed_metadata = {}  # page_number -> {"headers": [...], "footers": [...]}
    
    # Normalize headers and footers for matching
    normalized_headers = [normalize_text_for_comparison(h) for h in headers]
    normalized_footers = [normalize_text_for_comparison(f) for f in footers]
    
    for page_data in pages_content:
        page_number = page_data.get("page_number", 0)
        page_height = page_data.get("page_height")
        blocks = page_data.get("blocks", [])
        
        if not blocks:
            cleaned_pages.append(page_data)
            continue
        
        cleaned_blocks = []
        removed_headers = []
        removed_footers = []
        
        for block in blocks:
            bbox = block.get("bbox", [])
            if len(bbox) < 4:
                cleaned_blocks.append(block)
                continue
            
            x0, y0, x1, y1 = bbox
            text = block.get("text", "").strip()
            normalized_text = normalize_text_for_comparison(text)
            
            should_remove = False
            removal_reason = None
            
            # Check if matches a header
            if page_height and is_in_header_zone(y0, y1, page_height, header_threshold):
                for norm_header, orig_header in zip(normalized_headers, headers):
                    similarity = calculate_text_similarity(normalized_text, norm_header)
                    if similarity >= DEFAULT_SIMILARITY_THRESHOLD:
                        should_remove = True
                        removal_reason = "header"
                        removed_headers.append(orig_header)
                        break
            
            # Check if matches a footer
            if not should_remove and page_height and is_in_footer_zone(
                y0, y1, page_height, footer_threshold
            ):
                for norm_footer, orig_footer in zip(normalized_footers, footers):
                    similarity = calculate_text_similarity(normalized_text, norm_footer)
                    if similarity >= DEFAULT_SIMILARITY_THRESHOLD:
                        should_remove = True
                        removal_reason = "footer"
                        removed_footers.append(orig_footer)
                        break
            
            if not should_remove:
                cleaned_blocks.append(block)
        
        # Create cleaned page data
        cleaned_page = page_data.copy()
        cleaned_page["blocks"] = cleaned_blocks
        
        # Store removed content in metadata
        if removed_headers or removed_footers:
            removed_metadata[page_number] = {
                "headers": list(set(removed_headers)),  # Remove duplicates
                "footers": list(set(removed_footers)),
            }
        
        cleaned_pages.append(cleaned_page)
    
    return cleaned_pages, removed_metadata

