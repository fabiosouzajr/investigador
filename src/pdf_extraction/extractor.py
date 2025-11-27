# src/pdf_extraction/extractor.py
import os
import fitz  # PyMuPDF
import pdfminer.high_level
import json
from pathlib import Path
from tabula import read_pdf
from utils.config import RAW_PDF_DIR, EXTRACTED_DIR

def extract_single_pdf(pdf_path: str, output_path: str):
    # extrai texto página-por-página
    doc = fitz.open(pdf_path)
    extracted = {"file": os.path.basename(pdf_path), "sections": []}
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            extracted["sections"].append({"page": page_num, "text": text})
        # tentar extrair tabelas na página
        try:
            tables = read_pdf(pdf_path, pages=page_num, multiple_tables=True)
            if tables:
                for tbl in tables:
                    # converte para lista de listas ou similar
                    records = tbl.values.tolist()
                    extracted["sections"].append({"page": page_num, "table": records})
        except Exception as e:
            # pode haver erro se não há tabela naquela página
            pass

    # salva em JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

def run_extraction():
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    for fname in os.listdir(RAW_PDF_DIR):
        if fname.lower().endswith(".pdf"):
            inpath = os.path.join(RAW_PDF_DIR, fname)
            outpath = os.path.join(EXTRACTED_DIR, f"{Path(fname).stem}.json")
            if not os.path.exists(outpath):
                print(f"Extraindo {fname} → {outpath}")
                extract_single_pdf(inpath, outpath)

if __name__ == "__main__":
    run_extraction()
