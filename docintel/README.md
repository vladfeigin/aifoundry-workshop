# Document Intelligence PDF to Markdown Converter

This directory contains a script for converting PDF documents to Markdown format using Azure AI Document Intelligence. Markdown format is a recommended textual format when working with LLMs.

## Script

- `pdf-2-md.py` - Converts PDF files to Markdown using the `prebuilt-layout` model

## Usage

```bash
# Convert local PDF file
python -m docintel.pdf-2-md <pdf_path> [output.md]

# Convert from URL
python -m docintel.pdf-2-md https://example.com/document.pdf output.md

# Examples
python -m docintel.pdf-2-md ./docintel/data/document-intelligence-4.pdf ./docintel/data/document-intelligence-4.md
python -m docintel.pdf-2-md https://example.com/report.pdf ./reports/report.md
```

**Default Output**: If no output file is specified, saves to `out.md` in the current directory.

## Conversion Process

1. **Document Analysis**: Uses Azure Document Intelligence `prebuilt-layout` model to extract content and structure
2. **Page Processing**: Extracts text lines and tables from each page
3. **Markdown Generation**:
   - Adds page delimiters: `<!---- Page X ---------------------------------------------------------------------------------------------------------------------------------->`
   - Preserves text content line-by-line
   - Converts tables to Markdown table format with headers and dividers
   - Replaces newlines in table cells with spaces

## Table Conversion

Tables are converted to standard Markdown format:

- First row becomes the header
- Automatic divider row with `---` separators
- Cell content is cleaned (newlines replaced with spaces)
- Empty cells are preserved in the table structure

## Input Sources

- **Local Files**: Accepts any local PDF file path
- **URLs**: Supports HTTP/HTTPS URLs pointing to PDF files
- **Detection**: Automatically detects input type based on `http` prefix

## Environment Variables

Required environment variables in `.env`:

```env
AZURE_DOCINTEL_ENDPOINT=https://your-doc-intel-service.cognitiveservices.azure.com/
AZURE_DOCINTEL_KEY=your-api-key
```

## Output Format

The generated Markdown includes:

- Page delimiters for each page with specific format used by the search indexing system
- Original text content preserved line-by-line
- Tables formatted as Markdown tables
- UTF-8 encoding for proper character support

## Dependencies

- `azure-ai-documentintelligence` - Azure Document Intelligence client
- `azure-core` - Azure SDK core functionality
- `rich` - Progress bar display during processing
- `python-dotenv` - Environment variable loading

## Sample Data

The `data/` directory contains example files:

- `document-intelligence-4.pdf` - Sample PDF file
- `document-intelligence-4.md` - Converted Markdown output
- `GPT-4-Technical-Report.pdf` - Additional sample PDF
- `GPT-4-Technical-Report.md` - Converted Markdown output
