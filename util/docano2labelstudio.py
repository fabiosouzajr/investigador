#!/usr/bin/env python3
"""
Convert Doccano JSONL files to Label Studio JSON format.

This script iterates through Doccano JSONL files in the extracted_texts directory
and converts them to Label Studio format, supporting both:
- Text Classification: labels as list of strings
- Named Entity Recognition (NER): labels as list of [start, end, label] tuples
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Union
import argparse


def is_ner_label(label: Any) -> bool:
    """Check if a label is in NER format [start, end, label_name]."""
    return (
        isinstance(label, list)
        and len(label) == 3
        and isinstance(label[0], int)
        and isinstance(label[1], int)
        and isinstance(label[2], str)
    )


def is_text_classification_label(label: Any) -> bool:
    """Check if a label is in text classification format (string)."""
    return isinstance(label, str)


def convert_doccano_to_labelstudio(
    doccano_entry: Dict[str, Any],
    annotation_id: int = 1,
    from_name: str = "label",
    to_name: str = "text",
) -> Dict[str, Any]:
    """
    Convert a single Doccano entry to Label Studio format.
    
    Args:
        doccano_entry: Doccano entry with 'text' and 'labels' fields
        annotation_id: ID for the annotation
        from_name: Name of the annotation field (for Label Studio config)
        to_name: Name of the data field being annotated
        
    Returns:
        Label Studio task dictionary
    """
    text = doccano_entry.get("text", "")
    labels = doccano_entry.get("labels", [])
    
    # Create base task structure
    task: Dict[str, Any] = {
        "data": {
            "text": text
        }
    }
    
    # Preserve metadata if present
    if "metadata" in doccano_entry:
        task["meta"] = doccano_entry["metadata"]
    
    # Convert annotations if labels exist
    if labels:
        results = []
        has_ner = False
        has_classification = False
        
        # Check what type of labels we have
        for label in labels:
            if is_ner_label(label):
                has_ner = True
            elif is_text_classification_label(label):
                has_classification = True
        
        # Handle NER labels (sequence labeling)
        if has_ner:
            for label in labels:
                if is_ner_label(label):
                    start, end, label_name = label
                    # Extract the text span
                    span_text = text[start:end]
                    
                    results.append({
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [label_name]
                        }
                    })
        
        # Handle text classification labels
        if has_classification:
            classification_labels = [
                label for label in labels 
                if is_text_classification_label(label)
            ]
            
            if classification_labels:
                results.append({
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "choices",
                    "value": {
                        "choices": classification_labels
                    }
                })
        
        # Add annotations if we have results
        if results:
            task["annotations"] = [
                {
                    "id": str(annotation_id),
                    "result": results
                }
            ]
    
    return task


def process_doccano_file(
    input_path: Path,
    output_path: Path,
    output_format: str = "jsonl",
    from_name: str = "label",
    to_name: str = "text",
) -> tuple[int, int]:
    """
    Process a single Doccano JSONL file and convert to Label Studio format.
    
    Args:
        input_path: Path to input Doccano JSONL file
        output_path: Path to output Label Studio file
        output_format: "json" (array) or "jsonl" (one per line)
        from_name: Name of the annotation field
        to_name: Name of the data field
        
    Returns:
        Tuple of (total_entries, entries_with_annotations)
    """
    tasks = []
    total_entries = 0
    entries_with_annotations = 0
    
    # Read and convert entries
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                doccano_entry = json.loads(line)
                total_entries += 1
                
                # Convert to Label Studio format
                annotation_id = entries_with_annotations + 1 if doccano_entry.get("labels") else None
                task = convert_doccano_to_labelstudio(
                    doccano_entry,
                    annotation_id=annotation_id or 1,
                    from_name=from_name,
                    to_name=to_name,
                )
                
                if task.get("annotations"):
                    entries_with_annotations += 1
                
                tasks.append(task)
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Warning: Skipping invalid JSON on line {line_num} of {input_path.name}: {e}")
                continue
            except Exception as e:
                print(f"  ⚠️  Warning: Error processing line {line_num} of {input_path.name}: {e}")
                continue
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == "json":
        # Write as JSON array
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    else:
        # Write as JSONL (one task per line)
        with open(output_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    
    return total_entries, entries_with_annotations


def find_doccano_files(directory: Path) -> List[Path]:
    """Find all JSONL files in the directory (recursively)."""
    jsonl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(Path(root) / file)
    return sorted(jsonl_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Doccano JSONL files to Label Studio format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/extracted_texts",
        help="Directory containing Doccano JSONL files (default: data/extracted_texts)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input, with _labelstudio suffix)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format: json (array, default) or jsonl (one per line)"
    )
    parser.add_argument(
        "--from-name",
        type=str,
        default="label",
        help="Name of annotation field in Label Studio config (default: label)"
    )
    parser.add_argument(
        "--to-name",
        type=str,
        default="text",
        help="Name of data field in Label Studio config (default: text)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_labelstudio",
        help="Suffix to add to output filenames (default: _labelstudio)"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return 1
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use same directory structure with suffix
        output_dir = input_dir.parent / f"{input_dir.name}{args.suffix}"
    
    # Find all JSONL files
    jsonl_files = find_doccano_files(input_dir)
    
    if not jsonl_files:
        print(f"⚠️  No JSONL files found in {input_dir}")
        return 0
    
    print(f"📁 Found {len(jsonl_files)} JSONL file(s) in {input_dir}")
    print(f"📤 Output directory: {output_dir}")
    print(f"📝 Output format: {args.format}\n")
    
    # Process each file
    total_files = 0
    total_entries = 0
    total_annotated = 0
    
    for jsonl_file in jsonl_files:
        # Calculate relative path to maintain directory structure
        rel_path = jsonl_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Change extension based on format
        if args.format == "json":
            output_file = output_file.with_suffix(".json")
        else:
            output_file = output_file.with_suffix(".jsonl")
        
        print(f"🔄 Processing: {rel_path}")
        
        try:
            entries, annotated = process_doccano_file(
                jsonl_file,
                output_file,
                output_format=args.format,
                from_name=args.from_name,
                to_name=args.to_name,
            )
            
            total_files += 1
            total_entries += entries
            total_annotated += annotated
            
            print(f"  ✓ Converted {entries} entries ({annotated} with annotations)")
            print(f"  📄 Saved to: {output_file.relative_to(output_dir.parent)}\n")
            
        except Exception as e:
            print(f"  ❌ Error processing {jsonl_file}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("=" * 60)
    print(f"✅ Conversion complete!")
    print(f"   Files processed: {total_files}/{len(jsonl_files)}")
    print(f"   Total entries: {total_entries}")
    print(f"   Entries with annotations: {total_annotated}")
    print(f"   Output directory: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

