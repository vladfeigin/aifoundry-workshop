"""
pdf_to_markdown.py  –  Convert any PDF to Markdown with Azure AI Document Intelligence.
Adapted directly from the official 'prebuilt-layout' quick-start sample.

Prerequisites
-------------
pip install azure-ai-documentintelligence rich python-dotenv

# .env
AZURE_DOCINTEL_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
AZURE_DOCINTEL_KEY="your-key"

Usage
-----
python pdf_to_markdown.py <local_pdf_or_https_url> [output.md]
python ./docintel/pdf-2-md.py ./docintel/data/document-intelligence-4.pdf   ./docintel/data/document-intelligence-4.md
"""
import os, sys, textwrap
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from rich.progress import Progress

# --------------------------------------------------------------------- config
load_dotenv()
ENDPOINT = os.environ["AZURE_DOCINTEL_ENDPOINT"]
KEY      = os.environ["AZURE_DOCINTEL_KEY"]

SRC      = sys.argv[1]                                    # PDF path or URL
OUT_MD   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out.md")
MODEL_ID = "prebuilt-layout"

# --------------------------------------------------------------------- client
client = DocumentIntelligenceClient(ENDPOINT, AzureKeyCredential(KEY))

# ------------------------------------------------------------------ analyze
if SRC.startswith("http"):
    req = AnalyzeDocumentRequest(url_source=SRC)
else:
    req = AnalyzeDocumentRequest(bytes_source=Path(SRC).read_bytes())

with Progress() as progress:
    task = progress.add_task("[green]Analyzing…", total=None)
    poller  = client.begin_analyze_document(MODEL_ID, req)
    result  = poller.result()
    progress.update(task, completed=100)

# ----------------------------------------------------- markdown formatter
def as_md(di_result) -> str:
    md = []

    # Index tables by page number
    tables_by_page = {}
    for tbl in di_result.tables or []:
        pg = tbl.bounding_regions[0].page_number
        tables_by_page.setdefault(pg, []).append(tbl)

    for page in di_result.pages:
        md.append(f"\n<!---- Page {page.page_number} ---------------------------------------------------------------------------------------------------------------------------------->")  # noqa:E501

        # Plain lines
        for ln in page.lines:
            md.append(ln.content.strip())
        md.append("")

        # Any tables on this page
        for tbl in tables_by_page.get(page.page_number, []):
            rows = [["" for _ in range(tbl.column_count)] for _ in range(tbl.row_count)]
            for cell in tbl.cells:
                rows[cell.row_index][cell.column_index] = cell.content.replace("\n", " ")
            header  = "| " + " | ".join(rows[0]) + " |"
            divider = "| " + " | ".join(["---"] * tbl.column_count) + " |"
            md.extend([header, divider])
            for r in rows[1:]:
                md.append("| " + " | ".join(r) + " |")
            md.append("")

    return "\n".join(md)

markdown = as_md(result)
OUT_MD.write_text(markdown, encoding="utf-8")
print(f"✅  Markdown saved to {OUT_MD.resolve()}")