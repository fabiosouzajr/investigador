#!/usr/bin/env python3
"""
Heading Detection Module for PDF Extraction

Detects headings in PDF documents based on font characteristics.
Designed for governmental publications where headings typically use font size changes.
"""

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Default thresholds for heading detection
DEFAULT_MIN_FONT_SIZE_INCREASE = 2.0  # Points
DEFAULT_MAX_HEADING_LENGTH = 100  # Characters
DEFAULT_REQUIRE_BOLD = False  # Bold is optional for headings


def detect_headings(
    blocks: List[Dict[str, Any]],
    min_font_size_increase: float = DEFAULT_MIN_FONT_SIZE_INCREASE,
    require_bold: bool = DEFAULT_REQUIRE_BOLD,
    max_heading_length: int = DEFAULT_MAX_HEADING_LENGTH,
) -> List[Dict[str, Any]]:
    """
    Detect headings in document blocks based on font characteristics.
    
    Designed for governmental publications where headings use font size changes.
    
    Args:
        blocks: List of block dictionaries with 'text', 'font_info', and 'bbox'
        min_font_size_increase: Minimum font size increase (in points) to consider as heading
        require_bold: Whether bold formatting is required for headings
        max_heading_length: Maximum character length for a heading
        
    Returns:
        List of blocks with added 'is_heading' and 'heading_level' fields
    """
    if not blocks:
        return []
    
    # First pass: calculate average body text font size
    font_sizes = []
    for block in blocks:
        font_info = block.get("font_info", {})
        font_size = font_info.get("size")
        if font_size:
            font_sizes.append(font_size)
    
    if not font_sizes:
        LOGGER.debug("No font size information available for heading detection")
        return blocks
    
    # Calculate median font size (more robust than mean)
    sorted_sizes = sorted(font_sizes)
    median_font_size = sorted_sizes[len(sorted_sizes) // 2]
    
    # Detect headings based on font size and other characteristics
    heading_blocks = []
    for block in blocks:
        font_info = block.get("font_info", {})
        font_size = font_info.get("size")
        text = block.get("text", "").strip()
        
        if not text or not font_size:
            continue
        
        # Check if this block is a heading
        is_heading = False
        heading_level = None
        
        # Criterion 1: Font size significantly larger than body text
        font_size_increase = font_size - median_font_size
        if font_size_increase >= min_font_size_increase:
            is_heading = True
            
            # Determine heading level based on font size hierarchy
            # Level 1: Very large (>= 6 points larger)
            # Level 2: Large (>= 4 points larger)
            # Level 3: Medium (>= 2 points larger)
            if font_size_increase >= 6.0:
                heading_level = 1
            elif font_size_increase >= 4.0:
                heading_level = 2
            else:
                heading_level = 3
        
        # Criterion 2: Bold formatting (if required)
        if require_bold and is_heading:
            is_bold = font_info.get("is_bold", False)
            if not is_bold:
                is_heading = False
                heading_level = None
        
        # Criterion 3: Heading length (headings are typically shorter)
        if is_heading and len(text) > max_heading_length:
            # Might still be a heading if font size is very large
            if font_size_increase < 4.0:
                is_heading = False
                heading_level = None
        
        # Criterion 4: Position check (headings are typically at start of line)
        # This is handled by the block structure, so we'll trust font size
        
        if is_heading:
            block["is_heading"] = True
            block["heading_level"] = heading_level
            heading_blocks.append(block)
            LOGGER.debug(
                "Detected heading (level %d): %s (font size: %.1f, increase: %.1f)",
                heading_level,
                text[:50] + "..." if len(text) > 50 else text,
                font_size,
                font_size_increase,
            )
        else:
            block["is_heading"] = False
            block["heading_level"] = None
    
    LOGGER.info("Detected %d headings in document", len(heading_blocks))
    return blocks


def get_heading_hierarchy(blocks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Organize headings by level to understand document structure.
    
    Args:
        blocks: List of blocks with heading information
        
    Returns:
        Dictionary mapping heading level to list of heading blocks
    """
    hierarchy = {}
    
    for block in blocks:
        if block.get("is_heading", False):
            level = block.get("heading_level")
            if level:
                if level not in hierarchy:
                    hierarchy[level] = []
                hierarchy[level].append(block)
    
    return hierarchy


def find_next_heading(
    blocks: List[Dict[str, Any]], start_index: int
) -> Optional[int]:
    """
    Find the index of the next heading after start_index.
    
    Args:
        blocks: List of blocks with heading information
        start_index: Starting index to search from
        
    Returns:
        Index of next heading, or None if not found
    """
    for i in range(start_index + 1, len(blocks)):
        if blocks[i].get("is_heading", False):
            return i
    return None

