#!/usr/bin/env python3
"""
Enhanced Text Cleaning Module for PDF Extraction

Provides configurable text cleaning with Unicode normalization,
non-printing character removal, and improved line break handling.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Cleaning presets
CLEANING_PRESETS = {
    "default": {
        "remove_headers": True,
        "remove_footers": True,
        "unicode_normalization": "NFKC",
        "remove_non_printing": True,
        "aggressive_line_joining": False,
        "preserve_paragraphs": True,
    },
    "aggressive": {
        "remove_headers": True,
        "remove_footers": True,
        "unicode_normalization": "NFKC",
        "remove_non_printing": True,
        "aggressive_line_joining": True,
        "preserve_paragraphs": False,
    },
    "preserve_structure": {
        "remove_headers": True,
        "remove_footers": True,
        "unicode_normalization": "NFKC",
        "remove_non_printing": True,
        "aggressive_line_joining": False,
        "preserve_paragraphs": True,
    },
}


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """
    Normalize Unicode characters.
    
    Args:
        text: Text to normalize
        form: Normalization form (NFKC, NFC, NFD, NFKD)
            - NFKC: Compatibility decomposition + composition (recommended)
            - NFC: Canonical composition
            - NFD: Canonical decomposition
            - NFKD: Compatibility decomposition
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    try:
        return unicodedata.normalize(form, text)
    except Exception as e:
        LOGGER.warning("Unicode normalization failed: %s", str(e))
        return text


def remove_non_printing_chars(
    text: str, keep_newlines: bool = True, keep_tabs: bool = False
) -> str:
    """
    Remove non-printing characters from text.
    
    Args:
        text: Text to clean
        keep_newlines: Whether to keep newline characters
        keep_tabs: Whether to keep tab characters
        
    Returns:
        Text with non-printing characters removed
    """
    if not text:
        return ""
    
    result = []
    for char in text:
        # Keep printable characters
        if char.isprintable():
            result.append(char)
        # Keep newlines if requested
        elif char == "\n" and keep_newlines:
            result.append(char)
        # Keep tabs if requested
        elif char == "\t" and keep_tabs:
            result.append(char)
        # Remove other control characters
        elif unicodedata.category(char)[0] == "C":
            # Control character - skip
            continue
        # Remove zero-width characters
        elif char in ["\u200b", "\u200c", "\u200d"]:  # Zero-width space, non-joiner, joiner
            continue
        # Remove soft hyphens
        elif char == "\u00ad":  # Soft hyphen
            continue
        # Remove directional markers (optional - might want to keep for some languages)
        elif char in ["\u200e", "\u200f"]:  # Left-to-right mark, right-to-left mark
            continue
        else:
            # Keep other characters (might be special Unicode)
            result.append(char)
    
    return "".join(result)


def normalize_special_characters(text: str) -> str:
    """
    Normalize special characters (quotes, dashes, etc.) to standard forms.
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized special characters
    """
    if not text:
        return ""
    
    # Map special quotes to standard quotes
    text = text.replace(""", '"')  # Left double quotation mark
    text = text.replace(""", '"')  # Right double quotation mark
    text = text.replace("'", "'")  # Left single quotation mark
    text = text.replace("'", "'")  # Right single quotation mark
    
    # Map dashes to standard hyphen
    text = text.replace("—", "-")  # Em dash
    text = text.replace("–", "-")  # En dash
    
    # Map ellipsis
    text = text.replace("…", "...")  # Horizontal ellipsis
    
    return text


def clean_extracted_text_enhanced(
    text: str,
    preserve_paragraphs: bool = True,
    unicode_normalization: Optional[str] = "NFKC",
    remove_non_printing: bool = True,
    aggressive_line_joining: bool = False,
) -> str:
    """
    Enhanced text cleaning with Unicode normalization and non-printing character removal.
    
    Args:
        text: Raw extracted text
        preserve_paragraphs: If True, preserves double newlines as paragraph breaks
        unicode_normalization: Unicode normalization form (NFKC, NFC, NFD, NFKD, or None)
        remove_non_printing: Whether to remove non-printing characters
        aggressive_line_joining: Whether to use more aggressive line joining
        
    Returns:
        Cleaned text with normalized whitespace and Unicode
    """
    if not text:
        return ""
    
    # Step 1: Unicode normalization
    if unicode_normalization:
        text = normalize_unicode(text, unicode_normalization)
    
    # Step 2: Normalize special characters
    text = normalize_special_characters(text)
    
    # Step 3: Remove non-printing characters
    if remove_non_printing:
        text = remove_non_printing_chars(text, keep_newlines=True, keep_tabs=False)
    
    # Step 4: Replace non-breaking spaces and other special whitespace
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
    
    # Step 5: Normalize line breaks
    text = text.replace("\r\n", "\n")  # Windows line breaks
    text = text.replace("\r", "\n")  # Old Mac line breaks
    
    # Step 5.5: Context-aware removal of space+newline patterns
    # Preserve line breaks in tables/lists, remove in regular text
    # Check if text contains table markers
    has_table_markers = '[TABELAS]' in text or '[TABELA' in text
    
    if has_table_markers:
        # Split text into table and non-table sections
        # Pattern matches [TABELAS] or [TABELA N] followed by content until next paragraph break or end
        table_pattern = r'(\[TABELAS?[^\]]*\].*?)(?=\n\n|$)'
        parts = []
        last_end = 0
        
        for match in re.finditer(table_pattern, text, flags=re.DOTALL):
            # Add non-table text before this match
            if match.start() > last_end:
                non_table_text = text[last_end:match.start()]
                # Remove space+newline and mid-sentence newlines in non-table sections
                # Pattern 1: space/tab + newline (unless paragraph break)
                non_table_text = re.sub(r'[ \t]+\n(?!\n)', ' ', non_table_text)
                # Pattern 2: lowercase letter + newline + uppercase letter (mid-sentence break)
                non_table_text = re.sub(r'([a-záéíóúâêôãõç])\n([A-ZÁÉÍÓÚÂÊÔÃÕÇ])', r'\1 \2', non_table_text)
                if non_table_text:  # Only add if non-empty
                    parts.append(('text', non_table_text))
            
            # Add table section (preserve line breaks)
            parts.append(('table', match.group(0)))
            last_end = match.end()
        
        # Add remaining non-table text
        if last_end < len(text):
            non_table_text = text[last_end:]
            # Remove space+newline and mid-sentence newlines
            non_table_text = re.sub(r'[ \t]+\n(?!\n)', ' ', non_table_text)
            non_table_text = re.sub(r'([a-záéíóúâêôãõç])\n([A-ZÁÉÍÓÚÂÊÔÃÕÇ])', r'\1 \2', non_table_text)
            if non_table_text:  # Only add if non-empty
                parts.append(('text', non_table_text))
        
        # Reconstruct text
        if parts:
            text = ''.join([part[1] for part in parts])
        else:
            # Fallback: if pattern matching failed, just remove space+newline and mid-sentence breaks
            text = re.sub(r'[ \t]+\n(?!\n)', ' ', text)
            text = re.sub(r'([a-záéíóúâêôãõç])\n([A-ZÁÉÍÓÚÂÊÔÃÕÇ])', r'\1 \2', text)
    else:
        # No tables, safe to remove space+newline patterns and mid-sentence breaks
        # Pattern 1: space/tab + newline (unless paragraph break)
        text = re.sub(r'[ \t]+\n(?!\n)', ' ', text)
        # Pattern 2: lowercase letter + newline + uppercase letter (mid-sentence break)
        text = re.sub(r'([a-záéíóúâêôãõç])\n([A-ZÁÉÍÓÚÂÊÔÃÕÇ])', r'\1 \2', text)
    
    # Step 6: Clean whitespace and line breaks
    if preserve_paragraphs:
        # Replace 3+ consecutive newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        if aggressive_line_joining:
            # More aggressive: merge lines that don't end with sentence punctuation
            lines = text.split("\n")
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    # Empty line - preserve if it's part of a paragraph break
                    if i > 0 and i < len(lines) - 1:
                        if lines[i - 1].strip() and lines[i + 1].strip():
                            cleaned_lines.append("")
                    continue
                
                # Check if this line should be merged with previous
                if cleaned_lines and cleaned_lines[-1]:
                    prev_line = cleaned_lines[-1]
                    if not prev_line:
                        cleaned_lines.append(line_stripped)
                        continue
                    
                    # Don't merge if previous line ends with sentence punctuation
                    if re.search(r"[.!?]\s*$", prev_line):
                        cleaned_lines.append(line_stripped)
                        continue
                    
                    # Don't merge if current line starts with capital and is reasonably long
                    if line_stripped[0].isupper() and len(line_stripped) > 40:
                        cleaned_lines.append(line_stripped)
                        continue
                    
                    # Merge short lines or lines that continue the previous sentence
                    if len(line_stripped) < 100 or not line_stripped[0].isupper():
                        cleaned_lines[-1] = prev_line + " " + line_stripped
                        continue
                
                cleaned_lines.append(line_stripped)
            
            text = "\n".join(cleaned_lines)
        else:
            # Less aggressive: preserve more line breaks
            lines = text.split("\n")
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    if i > 0 and i < len(lines) - 1:
                        if lines[i - 1].strip() and lines[i + 1].strip():
                            cleaned_lines.append("")
                    continue
                
                if cleaned_lines and cleaned_lines[-1]:
                    prev_line = cleaned_lines[-1]
                    if not prev_line:
                        cleaned_lines.append(line_stripped)
                        continue
                    
                    # Only merge if previous line doesn't end with sentence punctuation
                    # and current line doesn't start with capital
                    if (
                        not re.search(r"[.!?]\s*$", prev_line)
                        and not line_stripped[0].isupper()
                        and len(line_stripped) < 80
                    ):
                        cleaned_lines[-1] = prev_line + " " + line_stripped
                    else:
                        cleaned_lines.append(line_stripped)
                else:
                    cleaned_lines.append(line_stripped)
            
            text = "\n".join(cleaned_lines)
        
        # Final cleanup: normalize multiple spaces to single space
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


def get_cleaning_config(preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Get cleaning configuration from preset or return default.
    
    Args:
        preset: Preset name ('default', 'aggressive', 'preserve_structure')
        
    Returns:
        Cleaning configuration dictionary
    """
    if preset and preset in CLEANING_PRESETS:
        return CLEANING_PRESETS[preset].copy()
    return CLEANING_PRESETS["default"].copy()

