#!/usr/bin/env python3
"""
Semantic Chunking Module for PDF Extraction

Provides page-bound semantic chunking optimized for NER annotation.
Chunks are created based on headings and respect page boundaries.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Default chunk sizes (in words) optimized for NER
DEFAULT_CHUNK_SIZE_WORDS = 400  # Optimal for NER tasks
DEFAULT_MIN_CHUNK_WORDS = 200  # Minimum words per chunk
DEFAULT_MAX_CHUNK_WORDS = 1000  # Maximum words per chunk


def count_words(text: str) -> int:
    """
    Count words in text (simple whitespace-based counting).
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words
    """
    if not text:
        return 0
    # Split by whitespace and filter empty strings
    words = [w for w in text.split() if w.strip()]
    return len(words)


def split_text_into_words(text: str) -> List[str]:
    """
    Split text into words.
    
    Args:
        text: Text to split
        
    Returns:
        List of words
    """
    if not text:
        return []
    return [w for w in text.split() if w.strip()]


def chunk_by_headings_page_bound(
    pages_content: List[Dict[str, Any]],
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    default_chunk_words: int = DEFAULT_CHUNK_SIZE_WORDS,
) -> List[Dict[str, Any]]:
    """
    Chunk document by headings with page boundary constraints.
    
    Chunks are created based on headings but never cross page boundaries.
    Optimized for NER annotation with target size of 200-500 words (default: 400).
    
    Args:
        pages_content: List of page content dictionaries with blocks and heading info
        min_chunk_words: Minimum words per chunk (default: 200)
        max_chunk_words: Maximum words per chunk (default: 1000)
        default_chunk_words: Target chunk size in words (default: 400)
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    
    if not pages_content:
        return chunks
    
    # Process each page separately (page-bound constraint)
    for page_data in pages_content:
        page_number = page_data.get("page_number", 0)
        page_text = page_data.get("text", "")
        blocks = page_data.get("blocks", [])
        
        if not page_text.strip():
            continue
        
        # Check if page has headings
        has_headings = any(block.get("is_heading", False) for block in blocks)
        
        if has_headings:
            # Semantic chunking: use headings to create chunks
            page_chunks = _chunk_by_headings_on_page(
                page_data,
                min_chunk_words,
                max_chunk_words,
                default_chunk_words,
            )
            chunks.extend(page_chunks)
        else:
            # Fallback: size-based chunking within page
            page_chunks = _chunk_by_size_on_page(
                page_data,
                min_chunk_words,
                max_chunk_words,
                default_chunk_words,
            )
            chunks.extend(page_chunks)
    
    LOGGER.info(
        "Created %d chunks (page-bound, semantic-first strategy)", len(chunks)
    )
    return chunks


def _chunk_by_headings_on_page(
    page_data: Dict[str, Any],
    min_chunk_words: int,
    max_chunk_words: int,
    default_chunk_words: int,
) -> List[Dict[str, Any]]:
    """
    Chunk a single page by headings.
    
    Args:
        page_data: Page content dictionary
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        default_chunk_words: Target chunk size
        
    Returns:
        List of chunks for this page
    """
    chunks = []
    blocks = page_data.get("blocks", [])
    page_number = page_data.get("page_number", 0)
    
    if not blocks:
        return chunks
    
    # Find all heading indices
    heading_indices = [
        i for i, block in enumerate(blocks) if block.get("is_heading", False)
    ]
    
    if not heading_indices:
        # No headings on this page, use size-based chunking
        return _chunk_by_size_on_page(
            page_data, min_chunk_words, max_chunk_words, default_chunk_words
        )
    
    # Create chunks starting at each heading
    for heading_idx in heading_indices:
        heading_block = blocks[heading_idx]
        heading_text = heading_block.get("text", "").strip()
        heading_level = heading_block.get("heading_level", 1)
        
        # Find next heading or end of page
        next_heading_idx = None
        for i in range(heading_idx + 1, len(blocks)):
            if blocks[i].get("is_heading", False):
                next_heading_idx = i
                break
        
        # Collect text from heading to next heading (or end of page)
        chunk_blocks = blocks[heading_idx : next_heading_idx] if next_heading_idx else blocks[heading_idx:]
        
        # Build chunk text
        chunk_text_parts = []
        for block in chunk_blocks:
            text = block.get("text", "").strip()
            if text:
                chunk_text_parts.append(text)
        
        chunk_text = "\n\n".join(chunk_text_parts)
        chunk_word_count = count_words(chunk_text)
        
        # If chunk is too large, split it by size
        if chunk_word_count > max_chunk_words:
            # Split the chunk while preserving heading
            sub_chunks = _split_large_chunk(
                chunk_text,
                heading_text,
                heading_level,
                min_chunk_words,
                max_chunk_words,
                default_chunk_words,
            )
            for sub_chunk_text, sub_chunk_metadata in sub_chunks:
                chunks.append({
                    "text": sub_chunk_text,
                    "page_number": page_number,
                    "chunk_type": "semantic",
                    "heading": heading_text,
                    "heading_level": heading_level,
                    "word_count": count_words(sub_chunk_text),
                    "is_first_chunk": sub_chunk_metadata.get("is_first", False),
                })
        elif chunk_word_count < min_chunk_words:
            # Chunk is too small, try to merge with next chunk if available
            # For now, keep it as is (merging across headings is complex)
            chunks.append({
                "text": chunk_text,
                "page_number": page_number,
                "chunk_type": "semantic",
                "heading": heading_text,
                "heading_level": heading_level,
                "word_count": chunk_word_count,
                "is_first_chunk": True,
            })
        else:
            # Chunk size is good
            chunks.append({
                "text": chunk_text,
                "page_number": page_number,
                "chunk_type": "semantic",
                "heading": heading_text,
                "heading_level": heading_level,
                "word_count": chunk_word_count,
                "is_first_chunk": True,
            })
    
    return chunks


def _chunk_by_size_on_page(
    page_data: Dict[str, Any],
    min_chunk_words: int,
    max_chunk_words: int,
    default_chunk_words: int,
) -> List[Dict[str, Any]]:
    """
    Chunk a single page by size (fallback when no headings).
    
    Args:
        page_data: Page content dictionary
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        default_chunk_words: Target chunk size
        
    Returns:
        List of chunks for this page
    """
    chunks = []
    page_text = page_data.get("text", "")
    page_number = page_data.get("page_number", 0)
    
    if not page_text.strip():
        return chunks
    
    # Split text into words
    words = split_text_into_words(page_text)
    total_words = len(words)
    
    if total_words <= max_chunk_words:
        # Page fits in one chunk
        chunks.append({
            "text": page_text,
            "page_number": page_number,
            "chunk_type": "size_based",
            "word_count": total_words,
        })
        return chunks
    
    # Split page into chunks of target size
    start = 0
    while start < total_words:
        end = min(start + default_chunk_words, total_words)
        
        # Try to break at sentence boundary
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        # If we're not at the end, try to extend to sentence boundary
        if end < total_words:
            # Solution 2.1: Expanded search window (50→150 words lookback, 20→100 words lookahead)
            lookback_start = max(0, end - 150)  # Increased from 50
            lookback_words = words[lookback_start:end + 100]  # Increased from 20
            lookback_text = " ".join(lookback_words)
            
            # Solution 2.2: Enhanced Portuguese sentence detection
            # Pattern handles abbreviations (Dr., Prof., Sr., Sra., Exmo., Exma.)
            # and decimal numbers (3.14, R$ 1.234,56)
            sentence_pattern = r'(?<!\d)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Sra\.)(?<!Exmo\.)(?<!Exma\.)(?<!art\.)(?<!n\.)(?<!nº)(?<!N\.)(?<!Nº)[.!?](?:\s+|$)(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])'
            sentence_endings = re.finditer(sentence_pattern, lookback_text, re.IGNORECASE)
            sentence_ends = list(sentence_endings)
            
            if sentence_ends:
                last_match = sentence_ends[-1]
                # Calculate adjusted end position
                match_pos_in_lookback = last_match.end()
                adjusted_end = lookback_start + match_pos_in_lookback
                if adjusted_end > start:
                    end = min(adjusted_end, total_words)
                    chunk_words = words[start:end]
                    chunk_text = " ".join(chunk_words)
            else:
                # Solution 2.3: Fallback to semantic boundaries or extend chunk
                # Look for strong punctuation (colon, semicolon) or comma clusters
                semantic_pattern = r'[;:]\s+|,\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ]'
                semantic_ends = list(re.finditer(semantic_pattern, lookback_text))
                
                if semantic_ends:
                    # Use last semantic boundary
                    last_match = semantic_ends[-1]
                    match_pos_in_lookback = last_match.end()
                    adjusted_end = lookback_start + match_pos_in_lookback
                    if adjusted_end > start:
                        end = min(adjusted_end, total_words)
                        chunk_words = words[start:end]
                        chunk_text = " ".join(chunk_words)
                else:
                    # Last resort: extend chunk by up to 25% to find sentence boundary
                    max_extension = int(default_chunk_words * 0.25)  # 25% extension
                    extended_end = min(end + max_extension, total_words)
                    extended_words = words[start:extended_end]
                    extended_text = " ".join(extended_words)
                    
                    # Try again with extended text
                    sentence_endings = re.finditer(sentence_pattern, extended_text, re.IGNORECASE)
                    sentence_ends = list(sentence_endings)
                    
                    if sentence_ends:
                        last_match = sentence_ends[-1]
                        # Calculate actual position
                        text_before_match = extended_text[:last_match.end()]
                        word_count = len(text_before_match.split())
                        end = min(start + word_count, total_words)
                        chunk_words = words[start:end]
                        chunk_text = " ".join(chunk_words)
                    else:
                        # Still no sentence boundary: log for manual review
                        LOGGER.debug(
                            "No sentence boundary found even after extension for chunk starting at word %d (page %d)",
                            start, page_number
                        )
        
        chunk_word_count = len(chunk_words)
        
        # Ensure minimum size
        if chunk_word_count < min_chunk_words and end < total_words:
            # Extend to minimum size
            end = min(start + min_chunk_words, total_words)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunk_word_count = len(chunk_words)
        
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "page_number": page_number,
                "chunk_type": "size_based",
                "word_count": chunk_word_count,
            })
        
        start = end
    
    return chunks


def _split_large_chunk(
    chunk_text: str,
    heading_text: str,
    heading_level: int,
    min_chunk_words: int,
    max_chunk_words: int,
    default_chunk_words: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Split a large chunk into smaller chunks while preserving heading context.
    
    Args:
        chunk_text: Text to split
        heading_text: Heading text to include in first chunk
        heading_level: Heading level
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        default_chunk_words: Target chunk size
        
    Returns:
        List of (chunk_text, metadata) tuples
    """
    chunks = []
    words = split_text_into_words(chunk_text)
    total_words = len(words)
    
    # First chunk includes heading
    start = 0
    chunk_idx = 0
    
    while start < total_words:
        end = min(start + default_chunk_words, total_words)
        
        # Try to break at sentence boundary
        chunk_words = words[start:end]
        chunk_text_part = " ".join(chunk_words)
        
        # If we're not at the end, try to extend to sentence boundary
        if end < total_words:
            # Solution 2.1: Expanded search window (50→150 words lookback, 20→100 words lookahead)
            lookback_start = max(0, end - 150)  # Increased from 50
            lookback_words = words[lookback_start:end + 100]  # Increased from 20
            lookback_text = " ".join(lookback_words)
            
            # Solution 2.2: Enhanced Portuguese sentence detection
            # Pattern handles abbreviations (Dr., Prof., Sr., Sra., Exmo., Exma.)
            # and decimal numbers (3.14, R$ 1.234,56)
            sentence_pattern = r'(?<!\d)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Sra\.)(?<!Exmo\.)(?<!Exma\.)(?<!art\.)(?<!n\.)(?<!nº)(?<!N\.)(?<!Nº)[.!?](?:\s+|$)(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])'
            sentence_endings = re.finditer(sentence_pattern, lookback_text, re.IGNORECASE)
            sentence_ends = list(sentence_endings)
            
            if sentence_ends:
                last_match = sentence_ends[-1]
                match_pos_in_lookback = last_match.end()
                adjusted_end = lookback_start + match_pos_in_lookback
                if adjusted_end > start:
                    end = min(adjusted_end, total_words)
                    chunk_words = words[start:end]
                    chunk_text_part = " ".join(chunk_words)
            else:
                # Solution 2.3: Fallback to semantic boundaries or extend chunk
                # Look for strong punctuation (colon, semicolon) or comma clusters
                semantic_pattern = r'[;:]\s+|,\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ]'
                semantic_ends = list(re.finditer(semantic_pattern, lookback_text))
                
                if semantic_ends:
                    # Use last semantic boundary
                    last_match = semantic_ends[-1]
                    match_pos_in_lookback = last_match.end()
                    adjusted_end = lookback_start + match_pos_in_lookback
                    if adjusted_end > start:
                        end = min(adjusted_end, total_words)
                        chunk_words = words[start:end]
                        chunk_text_part = " ".join(chunk_words)
                else:
                    # Last resort: extend chunk by up to 25% to find sentence boundary
                    max_extension = int(default_chunk_words * 0.25)  # 25% extension
                    extended_end = min(end + max_extension, total_words)
                    extended_words = words[start:extended_end]
                    extended_text = " ".join(extended_words)
                    
                    # Try again with extended text
                    sentence_endings = re.finditer(sentence_pattern, extended_text, re.IGNORECASE)
                    sentence_ends = list(sentence_endings)
                    
                    if sentence_ends:
                        last_match = sentence_ends[-1]
                        # Calculate actual position
                        text_before_match = extended_text[:last_match.end()]
                        word_count = len(text_before_match.split())
                        end = min(start + word_count, total_words)
                        chunk_words = words[start:end]
                        chunk_text_part = " ".join(chunk_words)
                    else:
                        # Still no sentence boundary: log for manual review
                        LOGGER.debug(
                            "No sentence boundary found even after extension for chunk starting at word %d (heading: %s)",
                            start, heading_text[:50] if heading_text else "N/A"
                        )
        
        # Prepend heading to first chunk
        if chunk_idx == 0:
            chunk_text_part = f"{heading_text}\n\n{chunk_text_part}"
        
        metadata = {
            "is_first": chunk_idx == 0,
        }
        
        chunks.append((chunk_text_part.strip(), metadata))
        start = end
        chunk_idx += 1
    
    return chunks

