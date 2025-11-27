# Text Chunking for NER Annotation - Detailed Explanation

## What is Chunking?

Chunking is the process of breaking down large documents into smaller, manageable pieces (chunks) that are optimal for annotation and processing. For Named Entity Recognition (NER) tasks, chunking is essential because:

1. **Annotation Efficiency**: Smaller chunks are easier and faster to annotate
2. **Model Training**: Most NER models work best with text segments of specific sizes
3. **Context Preservation**: Proper chunking maintains context while keeping chunks manageable
4. **Memory Management**: Prevents memory issues when processing large documents

## Chunking Strategy Used

### Target Size: 400 Words
- **Optimal for NER**: Research shows that 300-500 word chunks are ideal for NER annotation
- **Balance**: Large enough to preserve context, small enough to annotate efficiently
- **Entity Coverage**: Most entities (PER, ORG, DATE, LOC, MONEY) fit within this size

### Size Constraints
- **Minimum**: 200 words - Ensures chunks have enough context
- **Maximum**: 1000 words - Prevents chunks from becoming too large
- **Target**: 400 words - Optimal balance

### Overlap: 50 Words
- **Purpose**: Prevents splitting entities across chunk boundaries
- **How it works**: Last 50 words of one chunk become first 50 words of next chunk
- **Example**: If "John Smith" appears at the end of chunk 1, it will also appear at the start of chunk 2

## Chunking Algorithm

### Step 1: Paragraph Preservation
- First splits text by paragraph boundaries (`\n\n`)
- Preserves document structure (headings, sections, lists)
- Maintains readability and context

### Step 2: Smart Splitting
- If a paragraph is too large (>1000 words), splits by sentences
- Respects sentence boundaries to avoid breaking entity mentions
- Uses regex pattern: `(?<=[.!?])\s+` to detect sentence endings

### Step 3: Chunk Assembly
- Builds chunks by adding paragraphs/sentences until target size reached
- When approaching maximum size, saves current chunk and starts new one
- Applies overlap when starting new chunks

### Step 4: Quality Control
- Skips chunks smaller than minimum (200 words)
- Ensures all chunks meet quality standards
- Preserves document flow and structure

## Example

**Original Text** (800 words):
```
[Heading] Annual Report 2024

[Paragraph 1: 150 words] The company reported strong growth...

[Paragraph 2: 200 words] Revenue increased by 25%...

[Paragraph 3: 250 words] Key personnel changes included...

[Paragraph 4: 200 words] Financial highlights show...
```

**Chunking Result**:
- **Chunk 1** (400 words): Heading + Paragraph 1 + Paragraph 2 + first 50 words of Paragraph 3
- **Chunk 2** (400 words): Last 50 words of Paragraph 3 + Paragraph 4 + [next content]

## Benefits for NER

1. **Entity Completeness**: Overlap ensures entities aren't split
2. **Context Preservation**: Paragraph boundaries maintain semantic context
3. **Annotation Speed**: 400-word chunks are quick to annotate
4. **Model Performance**: Optimal size for training and inference
5. **Structure Awareness**: Preserves headings and document structure

## Configuration

You can adjust chunking parameters in the script:
- `DEFAULT_CHUNK_SIZE_WORDS = 400` - Target chunk size
- `DEFAULT_MIN_CHUNK_WORDS = 200` - Minimum chunk size
- `DEFAULT_MAX_CHUNK_WORDS = 1000` - Maximum chunk size
- `DEFAULT_CHUNK_OVERLAP_WORDS = 50` - Overlap between chunks

## Why Not Character-Based Chunking?

Character-based chunking (e.g., 1000 characters) can:
- Split words in the middle
- Break sentences awkwardly
- Lose semantic meaning
- Make annotation confusing

Word-based chunking (as used here) is superior because:
- Respects natural language boundaries
- Preserves semantic units
- Makes annotation more intuitive
- Better for NER model training

