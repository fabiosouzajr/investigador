#!/usr/bin/env python3
"""
Paragraph Detection Module for PDF Extraction

Provides enhanced paragraph detection using multiple signals:
- Vertical spacing
- Indentation patterns
- Font size changes
- Line break patterns
- Sentence boundary detection
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Default thresholds
DEFAULT_PARAGRAPH_GAP_THRESHOLD = 20.0  # Points
DEFAULT_INDENTATION_THRESHOLD = 10.0  # Points
DEFAULT_FONT_SIZE_CHANGE_THRESHOLD = 2.0  # Points
DEFAULT_MIN_LINE_LENGTH = 40  # Characters


def detect_sentence_endings(text: str) -> List[int]:
    """
    Detect sentence endings in text using regex patterns.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of character positions where sentences end
    """
    sentence_endings = []
    
    # Pattern for sentence endings: . ! ? followed by whitespace or end of text
    # Also handle common abbreviations
    pattern = r"[.!?](?:\s+|$)"
    
    for match in re.finditer(pattern, text):
        pos = match.end()
        # Check if it's not an abbreviation (heuristic: if followed by lowercase, might be abbreviation)
        if pos < len(text):
            next_char = text[pos:pos+1].strip()
            if next_char and next_char.islower():
                # Might be abbreviation, but still consider it if followed by capital later
                continue
        sentence_endings.append(match.end())
    
    return sentence_endings


def calculate_indentation_difference(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the difference in left indentation between two blocks.
    
    Args:
        bbox1: Bounding box of first block [x0, y0, x1, y1]
        bbox2: Bounding box of second block [x0, y0, x1, y1]
        
    Returns:
        Difference in x0 coordinates (positive if bbox2 is more indented)
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    return abs(bbox2[0] - bbox1[0])


def calculate_vertical_gap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate vertical gap between two blocks.
    
    Args:
        bbox1: Bounding box of first block [x0, y0, x1, y1]
        bbox2: Bounding box of second block [x0, y0, x1, y1]
        
    Returns:
        Vertical gap in points (y0 of bbox2 - y1 of bbox1)
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    return bbox2[1] - bbox1[3]  # y0 of second - y1 of first


def detect_paragraph_boundaries(
    blocks: List[Dict[str, Any]],
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    indentation_threshold: float = DEFAULT_INDENTATION_THRESHOLD,
    font_size_change_threshold: float = DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
    use_sentence_boundaries: bool = True,
) -> List[int]:
    """
    Detect paragraph boundaries using multiple signals.
    
    Args:
        blocks: List of block dictionaries with 'bbox', 'text', and optional 'font_info'
        paragraph_gap_threshold: Minimum vertical gap to indicate paragraph break (points)
        indentation_threshold: Minimum indentation change to indicate paragraph (points)
        font_size_change_threshold: Minimum font size change to indicate structure change (points)
        use_sentence_boundaries: Whether to use sentence boundary detection
        
    Returns:
        List of block indices where paragraph breaks should occur (before this block)
    """
    if not blocks or len(blocks) < 2:
        return []
    
    paragraph_breaks = []
    
    for i in range(1, len(blocks)):
        prev_block = blocks[i - 1]
        curr_block = blocks[i]
        
        prev_bbox = prev_block.get("bbox", [])
        curr_bbox = curr_block.get("bbox", [])
        prev_text = prev_block.get("text", "")
        curr_text = curr_block.get("text", "")
        
        if not prev_bbox or not curr_bbox:
            continue
        
        # Signal 1: Vertical spacing
        vertical_gap = calculate_vertical_gap(prev_bbox, curr_bbox)
        has_vertical_break = vertical_gap > paragraph_gap_threshold
        
        # Signal 2: Indentation change
        indentation_diff = calculate_indentation_difference(prev_bbox, curr_bbox)
        has_indentation_change = indentation_diff > indentation_threshold
        
        # Signal 3: Font size change
        prev_font_info = prev_block.get("font_info", {})
        curr_font_info = curr_block.get("font_info", {})
        prev_font_size = prev_font_info.get("size")
        curr_font_size = curr_font_info.get("size")
        
        has_font_change = False
        if prev_font_size and curr_font_size:
            font_size_diff = abs(curr_font_size - prev_font_size)
            has_font_change = font_size_diff > font_size_change_threshold
            # Also check if font size increases significantly (might be heading)
            if curr_font_size > prev_font_size + font_size_change_threshold:
                has_font_change = True
        
        # Signal 4: Sentence boundary detection
        has_sentence_boundary = False
        if use_sentence_boundaries and prev_text:
            # Check if previous block ends with sentence punctuation
            prev_text_stripped = prev_text.strip()
            if prev_text_stripped:
                # Check last 50 characters for sentence endings
                last_chars = prev_text_stripped[-50:]
                if re.search(r"[.!?]\s*$", last_chars):
                    has_sentence_boundary = True
                
                # Also check if current block starts with capital (new sentence)
                curr_text_stripped = curr_text.strip()
                if curr_text_stripped and curr_text_stripped[0].isupper():
                    # Check if previous ended with sentence punctuation
                    if re.search(r"[.!?]\s*$", prev_text_stripped):
                        has_sentence_boundary = True
        
        # Signal 5: Line length and structure
        prev_text_len = len(prev_text.strip())
        curr_text_len = len(curr_text.strip())
        
        # Check if current block looks like a new paragraph start
        is_likely_new_paragraph = False
        
        # If current block starts with capital and is reasonably long
        if curr_text.strip() and curr_text.strip()[0].isupper():
            if curr_text_len > DEFAULT_MIN_LINE_LENGTH:
                is_likely_new_paragraph = True
        
        # If previous block is short and current is long, might be paragraph break
        if prev_text_len < 50 and curr_text_len > DEFAULT_MIN_LINE_LENGTH:
            is_likely_new_paragraph = True
        
        # Decision: Combine signals
        # Strong signals (any of these suggests paragraph break):
        strong_signals = [
            has_vertical_break and (has_sentence_boundary or is_likely_new_paragraph),
            has_font_change and (has_vertical_break or has_indentation_change),
            has_indentation_change and has_vertical_break and is_likely_new_paragraph,
        ]
        
        # Medium signals (combination suggests paragraph break):
        medium_signals = [
            has_vertical_break and vertical_gap > paragraph_gap_threshold * 1.5,
            has_sentence_boundary and has_vertical_break,
            has_indentation_change and is_likely_new_paragraph,
        ]
        
        if any(strong_signals) or any(medium_signals):
            paragraph_breaks.append(i)
    
    return paragraph_breaks


def join_blocks_with_paragraph_detection(
    blocks: List[Dict[str, Any]],
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    indentation_threshold: float = DEFAULT_INDENTATION_THRESHOLD,
    font_size_change_threshold: float = DEFAULT_FONT_SIZE_CHANGE_THRESHOLD,
    use_sentence_boundaries: bool = True,
) -> str:
    """
    Join blocks into text with intelligent paragraph detection.
    
    Args:
        blocks: List of block dictionaries
        paragraph_gap_threshold: Minimum vertical gap for paragraph break
        indentation_threshold: Minimum indentation change for paragraph
        font_size_change_threshold: Minimum font size change for structure change
        use_sentence_boundaries: Whether to use sentence boundary detection
        
    Returns:
        Joined text with paragraph breaks (\n\n) where detected
    """
    if not blocks:
        return ""
    
    # Sort blocks by position (top to bottom, left to right)
    sorted_blocks = sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
    
    # Detect paragraph boundaries
    paragraph_breaks = detect_paragraph_boundaries(
        sorted_blocks,
        paragraph_gap_threshold=paragraph_gap_threshold,
        indentation_threshold=indentation_threshold,
        font_size_change_threshold=font_size_change_threshold,
        use_sentence_boundaries=use_sentence_boundaries,
    )
    
    # Join blocks with appropriate spacing
    result_lines = []
    paragraph_break_set = set(paragraph_breaks)
    
    for i, block in enumerate(sorted_blocks):
        block_text = block.get("text", "").strip()
        
        if not block_text:
            continue
        
        # Add paragraph break if detected
        if i in paragraph_break_set:
            if result_lines and result_lines[-1]:  # Add paragraph break before this block
                result_lines.append("")  # Empty line for paragraph break
            result_lines.append(block_text)
        else:
            # Check if we should merge with previous or add as new line
            if result_lines and result_lines[-1]:
                prev_line = result_lines[-1]
                
                # Don't merge if previous line ends with sentence punctuation
                if re.search(r"[.!?]\s*$", prev_line):
                    result_lines.append(block_text)
                # Don't merge if current block is likely a new sentence/paragraph
                elif block_text and block_text[0].isupper() and len(block_text) > DEFAULT_MIN_LINE_LENGTH:
                    result_lines.append(block_text)
                else:
                    # Merge with previous line
                    result_lines[-1] = prev_line + " " + block_text
            else:
                result_lines.append(block_text)
    
    # Join with newlines, but ensure paragraph breaks are double newlines
    # Convert empty lines (paragraph breaks) to double newlines
    result = []
    for i, line in enumerate(result_lines):
        result.append(line)
        # If this is an empty line (paragraph break), add another newline
        # If next line exists and this line is empty, we want \n\n
        if not line and i < len(result_lines) - 1 and result_lines[i + 1]:
            result.append("")  # This will create \n\n when joined
    
    # Join and replace patterns to ensure proper paragraph breaks
    text = "\n".join(result)
    # Ensure paragraph breaks are double newlines
    text = re.sub(r"\n\n+", "\n\n", text)  # Normalize multiple newlines to double
    return text


def enhance_paragraph_detection_in_text(
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    paragraph_gap_threshold: float = DEFAULT_PARAGRAPH_GAP_THRESHOLD,
    use_sentence_boundaries: bool = True,
) -> str:
    """
    Enhance paragraph detection in already-joined text.
    
    This is a fallback when block-level information is not available.
    Uses sentence boundaries and line patterns to detect paragraphs.
    
    Args:
        text: Text to process
        blocks: Optional list of blocks for better detection
        paragraph_gap_threshold: Not used in text-only mode, kept for API consistency
        use_sentence_boundaries: Whether to use sentence boundary detection
        
    Returns:
        Text with improved paragraph breaks
    """
    if not text:
        return ""
    
    # If we have blocks, use block-based detection
    if blocks:
        return join_blocks_with_paragraph_detection(
            blocks,
            paragraph_gap_threshold=paragraph_gap_threshold,
            use_sentence_boundaries=use_sentence_boundaries,
        )
    
    # Otherwise, use text-based heuristics
    lines = text.split("\n")
    enhanced_lines = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if not line_stripped:
            # Empty line - preserve if it's a paragraph break
            if enhanced_lines and enhanced_lines[-1]:
                enhanced_lines.append("")
            continue
        
        if not enhanced_lines:
            enhanced_lines.append(line_stripped)
            continue
        
        prev_line = enhanced_lines[-1]
        
        # Check if this should be a new paragraph
        is_new_paragraph = False
        
        # Signal 1: Previous line ends with sentence punctuation
        if use_sentence_boundaries and re.search(r"[.!?]\s*$", prev_line):
            # Check if current line starts with capital (new sentence/paragraph)
            if line_stripped[0].isupper() and len(line_stripped) > DEFAULT_MIN_LINE_LENGTH:
                is_new_paragraph = True
        
        # Signal 2: Current line looks like a heading/title
        if line_stripped.isupper() and len(line_stripped) < 100:
            is_new_paragraph = True
        
        # Signal 3: Significant length difference
        if len(prev_line) < 50 and len(line_stripped) > DEFAULT_MIN_LINE_LENGTH:
            if line_stripped[0].isupper():
                is_new_paragraph = True
        
        if is_new_paragraph:
            if enhanced_lines[-1]:  # Don't add double break
                enhanced_lines.append("")
            enhanced_lines.append(line_stripped)
        else:
            # Merge with previous line
            if prev_line and not prev_line.endswith((".", "!", "?")):
                enhanced_lines[-1] = prev_line + " " + line_stripped
            else:
                enhanced_lines.append(line_stripped)
    
    return "\n".join(enhanced_lines)

