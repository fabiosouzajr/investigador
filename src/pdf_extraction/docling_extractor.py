#!/usr/bin/env python3
"""
PDF Extraction using Docling - Optimized for NER Labeling

Extracts text from PDF files using the Docling library and saves results as JSONL files
optimized for Label Studio and Doccano annotation tools.

Features:
- Displays directory structure with PDF counts
- Interactive selection of files to process
- Uses Docling for advanced PDF parsing
- Chunking optimized for NER (400 words per chunk)
- Text normalization and hyphenation fixing
- Preserves document structure (headings, tables)
- Supports both Label Studio and Doccano formats
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available. Please install it with: pip install docling")


# Default directories
DEFAULT_PDF_DIR = Path(__file__).parent.parent.parent / "data" / "pdfs"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "extracted_texts"

# Chunking defaults (optimized for NER)
DEFAULT_CHUNK_SIZE_WORDS = 400  # Optimal for NER tasks
DEFAULT_MIN_CHUNK_WORDS = 200   # Minimum words per chunk
DEFAULT_MAX_CHUNK_WORDS = 1000  # Maximum words per chunk
DEFAULT_CHUNK_OVERLAP_WORDS = 50  # Overlap between chunks


def discover_pdf_structure(pdf_dir: Path) -> Tuple[Dict[Path, int], List[Path]]:
    """
    Discover the structure of PDF files in the directory.
    
    Args:
        pdf_dir: Directory to scan for PDFs
        
    Returns:
        Tuple of (subfolders_dict, root_pdfs_list)
        - subfolders_dict: Maps subfolder paths to PDF counts
        - root_pdfs_list: List of PDF files in root directory
    """
    subfolders = {}
    root_pdfs = []
    
    if not pdf_dir.exists():
        return subfolders, root_pdfs
    
    # Check root directory for PDFs
    root_pdfs = sorted([f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
    
    # Check all immediate subdirectories
    for subdir in sorted(pdf_dir.iterdir()):
        if subdir.is_dir():
            pdf_files = sorted([f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
            if pdf_files:
                subfolders[subdir] = len(pdf_files)
    
    return subfolders, root_pdfs


def display_structure(subfolders: Dict[Path, int], root_pdfs: List[Path], pdf_dir: Path):
    """
    Display the PDF directory structure.
    
    Args:
        subfolders: Dictionary mapping subfolder paths to PDF counts
        root_pdfs: List of PDF files in root directory
        pdf_dir: Base PDF directory
    """
    print("\n" + "=" * 70)
    print("PDF Directory Structure")
    print("=" * 70)
    
    if subfolders or root_pdfs:
        # Display root PDFs if any
        if root_pdfs:
            print(f"\n📁 (root) - {len(root_pdfs)} PDF file(s)")
        
        # Display subfolders
        if subfolders:
            print("\n📁 Subfolders:")
            for idx, (folder_path, count) in enumerate(sorted(subfolders.items(), key=lambda x: str(x[0])), start=1):
                relative_path = folder_path.relative_to(pdf_dir)
                print(f"  {idx}. {relative_path} - {count} PDF file(s)")
        else:
            print("\n(No subfolders found)")
    else:
        print("\n⚠️  No PDF files found in the directory!")
    
    print("=" * 70)


def collect_all_pdfs(subfolders: Dict[Path, int], root_pdfs: List[Path], pdf_dir: Path) -> List[Tuple[Path, str]]:
    """
    Collect all PDF files from root and subfolders with their display paths.
    
    Args:
        subfolders: Dictionary mapping subfolder paths to PDF counts
        root_pdfs: List of PDF files in root directory
        pdf_dir: Base PDF directory
        
    Returns:
        List of tuples (pdf_path, display_path) for all PDF files
    """
    all_pdfs = []
    
    # Add root PDFs
    for pdf_path in sorted(root_pdfs):
        display_path = pdf_path.name
        all_pdfs.append((pdf_path, display_path))
    
    # Add PDFs from subfolders
    for folder_path in sorted(subfolders.keys(), key=lambda x: str(x)):
        relative_folder = folder_path.relative_to(pdf_dir)
        pdf_files = sorted([f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
        for pdf_path in pdf_files:
            display_path = f"{relative_folder}/{pdf_path.name}"
            all_pdfs.append((pdf_path, display_path))
    
    return all_pdfs


def display_files(all_pdfs: List[Tuple[Path, str]]):
    """
    Display all PDF files in a numbered list.
    
    Args:
        all_pdfs: List of tuples (pdf_path, display_path) for all PDF files
    """
    print("\n" + "=" * 70)
    print(f"Available PDF Files ({len(all_pdfs)} total)")
    print("=" * 70)
    
    for idx, (pdf_path, display_path) in enumerate(all_pdfs, start=1):
        # Show file size for context
        try:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB"
        except:
            size_str = "unknown size"
        print(f"  {idx:3d}. {display_path:<50} ({size_str})")
    
    print("=" * 70)


def prompt_selection(all_pdfs: List[Tuple[Path, str]]) -> List[Path]:
    """
    Prompt user to select individual files for processing.
    
    Args:
        all_pdfs: List of tuples (pdf_path, display_path) for all PDF files
        
    Returns:
        List of selected PDF file paths to process
    """
    if not all_pdfs:
        print("\n⚠️  No PDF files to process!")
        return []
    
    # Display all files
    display_files(all_pdfs)
    
    # Display selection instructions
    print("\nSelection Options:")
    print("  - Enter a single number (e.g., 1) to process one file")
    print("  - Enter multiple numbers (comma-separated, e.g., 1,3,5) to process multiple files")
    print("  - Enter 'all' to process all files")
    print("  - Press Ctrl+C to cancel")
    print("=" * 70)
    
    while True:
        try:
            choice = input("\nSelect file(s) to process: ").strip().lower()
            
            if choice == "all":
                print(f"\n✓ Processing all {len(all_pdfs)} PDF file(s)")
                return [pdf_path for pdf_path, _ in all_pdfs]
            
            # Parse comma-separated numbers
            selected_indices = [int(x.strip()) for x in choice.split(",")]
            
            # Collect selected PDFs
            selected_pdfs = []
            invalid_indices = []
            
            for idx in selected_indices:
                if 1 <= idx <= len(all_pdfs):
                    selected_pdfs.append(all_pdfs[idx - 1][0])
                else:
                    invalid_indices.append(idx)
            
            if invalid_indices:
                print(f"Warning: Invalid file number(s) {invalid_indices}, skipping...")
            
            if selected_pdfs:
                if len(selected_pdfs) == 1:
                    print(f"\n✓ Processing 1 PDF file")
                else:
                    print(f"\n✓ Processing {len(selected_pdfs)} PDF file(s)")
                return selected_pdfs
            else:
                print("No valid files selected. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or 'all'.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            sys.exit(0)


def prompt_format_selection() -> str:
    """
    Prompt user to select output format (Label Studio or Doccano).
    
    Returns:
        Selected format: 'labelstudio' or 'doccano'
    """
    print("\n" + "=" * 70)
    print("Select Output Format")
    print("=" * 70)
    print("  1. Label Studio (https://labelstud.io/)")
    print("  2. Doccano (https://github.com/doccano/doccano)")
    print("=" * 70)
    
    while True:
        try:
            choice = input("\nSelect format (1 or 2): ").strip()
            
            if choice == "1":
                print("\n✓ Selected: Label Studio format")
                return "labelstudio"
            elif choice == "2":
                print("\n✓ Selected: Doccano format")
                return "doccano"
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            sys.exit(0)


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len([w for w in text.split() if w.strip()])


def fix_hyphenation(text: str) -> str:
    """
    Fix end-of-line word hyphenation.
    
    Removes hyphens at end of lines and joins with next word if it starts with lowercase.
    Example: "ex-\nample" -> "example"
    """
    # Pattern: word ending with hyphen, followed by newline, followed by word starting with lowercase
    # This handles cases like "ex-\nample" -> "example"
    text = re.sub(r'([a-zA-ZÀ-ÿ])-\s*\n\s*([a-zà-ÿ])', r'\1\2', text)
    
    # Also handle cases with spaces: "ex- \nample"
    text = re.sub(r'([a-zA-ZÀ-ÿ])-\s+\n\s+([a-zà-ÿ])', r'\1\2', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace and newlines.
    
    - Multiple spaces -> single space
    - Multiple newlines -> double newline (paragraph break)
    - Normalize line breaks
    """
    # Fix hyphenation first
    text = fix_hyphenation(text)
    
    # Normalize line breaks (CRLF -> LF, CR -> LF)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Join lines back, preserving paragraph breaks (empty lines)
    text = '\n'.join(lines)
    
    # Multiple spaces -> single space (but preserve intentional spacing)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Multiple newlines -> double newline max (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove spaces at start/end of lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Clean up any remaining issues
    text = re.sub(r' +', ' ', text)  # Multiple spaces
    text = re.sub(r'\n +', '\n', text)  # Spaces after newlines
    text = re.sub(r' +\n', '\n', text)  # Spaces before newlines
    
    return text.strip()


def extract_text_from_docling_doc(doc) -> Tuple[str, List[Dict]]:
    """
    Extract text and tables from Docling document, preserving structure.
    
    Args:
        doc: Docling document object
        
    Returns:
        Tuple of (full_text, tables_list)
    """
    try:
        # Get markdown representation (preserves structure like headings, tables)
        markdown_text = doc.export_to_markdown()
        
        # Extract tables separately for better handling
        tables = []
        
        # Try to extract structured table data
        if hasattr(doc, 'body') and hasattr(doc.body, 'children'):
            for item in doc.body.children:
                if hasattr(item, 'type') and item.type == 'table':
                    table_data = {
                        'rows': [],
                        'markdown': ''
                    }
                    # Try to get table as markdown
                    try:
                        # Extract table rows if available
                        if hasattr(item, 'rows'):
                            for row in item.rows:
                                if hasattr(row, 'cells'):
                                    table_data['rows'].append([cell.text if hasattr(cell, 'text') else '' for cell in row.cells])
                    except:
                        pass
                    tables.append(table_data)
        
        return markdown_text, tables
        
    except Exception as e:
        print(f"    Warning: Error extracting text: {e}")
        return "", []


def chunk_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """
    Chunk text into segments optimized for NER annotation.
    
    Chunking Strategy:
    - Target size: ~400 words (optimal for NER)
    - Preserves paragraph boundaries when possible
    - Uses overlap to avoid splitting entities
    - Respects sentence boundaries
    
    Args:
        text: Text to chunk
        chunk_size_words: Target chunk size in words
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        overlap_words: Number of words to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    
    # For very large texts, process in segments to save memory
    # Split into paragraphs first (preserve structure)
    # Use generator for memory efficiency if text is very large
    if len(text) > 50_000_000:  # 50MB threshold
        # Process in chunks of paragraphs
        paragraphs = text.split('\n\n')
        del text  # Free memory early
    else:
        paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_word_count = count_words(para)
        
        # If paragraph alone exceeds max, split it by sentences
        if para_word_count > max_chunk_words:
            # Save current chunk if it has content
            if current_chunk and current_word_count >= min_chunk_words:
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_content,
                    'word_count': current_word_count,
                    'type': 'paragraph'
                })
                current_chunk = []
                current_word_count = 0
            
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sent_word_count = count_words(sentence)
                
                if current_word_count + sent_word_count > max_chunk_words and current_chunk:
                    # Save current chunk
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_content,
                        'word_count': current_word_count,
                        'type': 'paragraph'
                    })
                    
                    # Start new chunk with overlap
                    if overlap_words > 0 and len(current_chunk) > 0:
                        # Get last few words for overlap
                        last_chunk = '\n\n'.join(current_chunk)
                        words = last_chunk.split()
                        overlap_text = ' '.join(words[-overlap_words:]) if len(words) >= overlap_words else last_chunk
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_word_count = count_words(overlap_text) + sent_word_count
                    else:
                        current_chunk = [sentence]
                        current_word_count = sent_word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += sent_word_count
        else:
            # Check if adding this paragraph would exceed max
            if current_word_count + para_word_count > max_chunk_words and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_content,
                    'word_count': current_word_count,
                    'type': 'paragraph'
                })
                
                # Start new chunk with overlap
                if overlap_words > 0 and len(current_chunk) > 0:
                    last_chunk = '\n\n'.join(current_chunk)
                    words = last_chunk.split()
                    overlap_text = ' '.join(words[-overlap_words:]) if len(words) >= overlap_words else last_chunk
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_word_count = count_words(overlap_text) + para_word_count
                else:
                    current_chunk = [para]
                    current_word_count = para_word_count
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_word_count += para_word_count
                
                # If we've reached target size, save chunk
                if current_word_count >= chunk_size_words:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_content,
                        'word_count': current_word_count,
                        'type': 'paragraph'
                    })
                    current_chunk = []
                    current_word_count = 0
    
    # Save remaining chunk if it meets minimum size
    if current_chunk and current_word_count >= min_chunk_words:
        chunk_content = '\n\n'.join(current_chunk)
        chunks.append({
            'text': chunk_content,
            'word_count': current_word_count,
            'type': 'paragraph'
        })
    
    return chunks


def format_for_label_studio(text: str, source_file: str) -> Dict:
    """
    Format chunk for Label Studio import.
    
    Label Studio format: {"data": {"text": "..."}}
    Reference: https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format
    
    Args:
        text: Text content
        source_file: Source PDF filename
        
    Returns:
        Dictionary in Label Studio format
    """
    return {
        "data": {
            "text": text
        },
        "meta": {
            "source_file": source_file
        }
    }


def format_for_doccano(text: str, source_file: str) -> Dict:
    """
    Format chunk for Doccano import.
    
    Doccano format: {"text": "...", "labels": []}
    
    Args:
        text: Text content
        source_file: Source PDF filename
        
    Returns:
        Dictionary in Doccano format
    """
    return {
        "text": text,
        "labels": []
    }


def extract_pdf_with_docling(
    pdf_path: Path,
    output_path: Path,
    output_format: str = "labelstudio",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> bool:
    """
    Extract text from a PDF file using Docling and save as JSONL.
    Memory-efficient streaming approach to prevent system freezing.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Path to save the output JSONL file
        output_format: 'labelstudio' or 'doccano'
        chunk_size_words: Target chunk size in words
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        overlap_words: Overlap between chunks
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        print("    [1/5] Initializing Docling converter...", end=" ", flush=True)
        # Initialize Docling converter with optimized settings
        converter = DocumentConverter()
        print("✓")
        
        print("    [2/5] Converting PDF (this may take a while)...", end=" ", flush=True)
        # Convert PDF to DoclingDocument
        result = converter.convert(str(pdf_path))
        doc = result.document
        print("✓")
        
        print("    [3/5] Extracting text and structure...", end=" ", flush=True)
        # Extract text and tables
        markdown_text, tables = extract_text_from_docling_doc(doc)
        
        # Clear doc from memory early
        del doc, result
        import gc
        gc.collect()
        
        if not markdown_text:
            print(f"\n    ⚠️  No text extracted from {pdf_path.name}")
            return False
        print(f"✓ ({len(markdown_text):,} chars)")
        
        print("    [4/5] Normalizing text and chunking...", end=" ", flush=True)
        # Normalize whitespace and fix hyphenation (process in chunks to save memory)
        # Process in smaller pieces to avoid memory issues
        if len(markdown_text) > 10_000_000:  # 10MB threshold
            # Process in segments
            segment_size = 1_000_000  # 1MB segments
            normalized_segments = []
            for i in range(0, len(markdown_text), segment_size):
                segment = markdown_text[i:i+segment_size]
                normalized_segments.append(normalize_whitespace(segment))
            normalized_text = ''.join(normalized_segments)
            del normalized_segments
        else:
            normalized_text = normalize_whitespace(markdown_text)
        
        # Clear original markdown from memory
        del markdown_text
        gc.collect()
        
        # Add tables to text if present (preserve structure)
        if tables:
            table_section = "\n\n[TABELAS]\n\n"
            for i, table in enumerate(tables):
                if table.get('markdown'):
                    table_section += table['markdown'] + "\n\n"
                elif table.get('rows'):
                    # Format table rows as text
                    for row in table['rows']:
                        table_section += " | ".join(str(cell) for cell in row) + "\n"
                    table_section += "\n"
            normalized_text += table_section
            normalized_text = normalize_whitespace(normalized_text)
        
        # Chunk the text
        chunks = chunk_text(
            normalized_text,
            chunk_size_words=chunk_size_words,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
        )
        
        # Clear normalized_text from memory
        del normalized_text
        gc.collect()
        
        if not chunks:
            print(f"\n    ⚠️  No chunks created from {pdf_path.name}")
            return False
        print(f"✓ ({len(chunks)} chunks)")
        
        print("    [5/5] Writing JSONL file...", end=" ", flush=True)
        # Write JSONL file incrementally (streaming) to save memory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_content = chunk_data['text']
                
                if output_format == "labelstudio":
                    entry = format_for_label_studio(chunk_content, pdf_path.name)
                else:  # doccano
                    entry = format_for_doccano(chunk_content, pdf_path.name)
                
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                chunk_count += 1
                
                # Clear entry from memory
                del entry, chunk_content
        
        # Clear chunks from memory
        del chunks
        gc.collect()
        
        print(f"✓ ({chunk_count} entries written)")
        return True
        
    except MemoryError:
        print(f"\n    ❌ Out of memory processing {pdf_path.name}")
        print("    💡 Try processing smaller files or increase system memory")
        return False
    except Exception as e:
        print(f"\n    ❌ Error processing {pdf_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_pdfs(
    pdf_files: List[Path],
    output_dir: Path,
    pdf_dir: Path,
    output_format: str = "labelstudio",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
):
    """
    Process selected PDF files and save results to JSONL.
    
    Args:
        pdf_files: List of PDF file paths to process
        output_dir: Directory to save output JSONL files
        pdf_dir: Base PDF directory (for preserving relative structure)
        output_format: 'labelstudio' or 'doccano'
        chunk_size_words: Target chunk size in words
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        overlap_words: Overlap between chunks
    """
    if not pdf_files:
        print("\n⚠️  No PDF files selected for processing!")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Processing {len(pdf_files)} PDF file(s)...")
    print(f"Format: {output_format.upper()}")
    print(f"Chunk size: {chunk_size_words} words (min: {min_chunk_words}, max: {max_chunk_words})")
    print(f"{'=' * 70}")
    print("⚠️  Note: Large PDFs may take several minutes to process.")
    print("   Progress indicators will show each step.\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    total_chunks = 0
    
    for pdf_path in pdf_files:
        # Determine output path, preserving relative structure if in subfolder
        try:
            relative_path = pdf_path.relative_to(pdf_dir)
            # If in subfolder, create corresponding subfolder in output
            if len(relative_path.parts) > 1:
                output_subdir = output_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{pdf_path.stem}.jsonl"
            else:
                output_path = output_dir / f"{pdf_path.stem}.jsonl"
        except ValueError:
            # PDF is not relative to pdf_dir, just use filename
            output_path = output_dir / f"{pdf_path.stem}.jsonl"
        
        # Skip if already exists
        if output_path.exists():
            print(f"⏭️  Skipping {pdf_path.name} (already extracted)")
            continue
        
        print(f"📄 Processing {pdf_path.name}...", end=" ", flush=True)
        
        if extract_pdf_with_docling(
            pdf_path,
            output_path,
            output_format=output_format,
            chunk_size_words=chunk_size_words,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
        ):
            # Count chunks in output file
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    chunk_count = sum(1 for _ in f)
                total_chunks += chunk_count
                print(f"✓ ({chunk_count} chunks)")
            except:
                print("✓")
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"  ✓ Successful: {successful}")
    print(f"  📊 Total chunks created: {total_chunks}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}")
    print(f"{'=' * 70}\n")


def main():
    """Main function to run the PDF extraction script."""
    if not DOCLING_AVAILABLE:
        print("ERROR: Docling is not installed. Please install it with:")
        print("  pip install docling")
        sys.exit(1)
    
    # Get PDF directory (default or from command line)
    if len(sys.argv) > 1:
        pdf_dir = Path(sys.argv[1]).resolve()
    else:
        pdf_dir = DEFAULT_PDF_DIR.resolve()
    
    # Get output directory (default or from command line)
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2]).resolve()
    else:
        output_dir = DEFAULT_OUTPUT_DIR.resolve()
    
    # Validate PDF directory
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory does not exist: {pdf_dir}")
        sys.exit(1)
    
    if not pdf_dir.is_dir():
        print(f"ERROR: PDF path is not a directory: {pdf_dir}")
        sys.exit(1)
    
    print(f"\n📂 PDF Directory: {pdf_dir}")
    print(f"📂 Output Directory: {output_dir}")
    
    # Discover PDF structure
    subfolders, root_pdfs = discover_pdf_structure(pdf_dir)
    
    # Display structure overview
    display_structure(subfolders, root_pdfs, pdf_dir)
    
    # Collect all PDFs from all locations
    all_pdfs = collect_all_pdfs(subfolders, root_pdfs, pdf_dir)
    
    # Prompt for file selection
    selected_pdfs = prompt_selection(all_pdfs)
    
    # Prompt for format selection
    output_format = prompt_format_selection()
    
    # Process selected PDFs
    if selected_pdfs:
        process_pdfs(
            selected_pdfs,
            output_dir,
            pdf_dir,
            output_format=output_format,
        )
    else:
        print("\n⚠️  No PDF files selected. Exiting.")


if __name__ == "__main__":
    main()
