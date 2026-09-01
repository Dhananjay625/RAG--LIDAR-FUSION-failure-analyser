# src/data_loader.py

import os
import pdfplumber
from typing import List, Dict

def load_papers_from_directory(papers_dir: str = "/Volumes/PortableSSD/Dissertation/data/Researchpapers") -> List[Dict]:
    """
    Load all PDFs from a directory and extract text.
    
    Returns:
        List of dicts: [{"filename": "paper.pdf", "text": "...", "pages": N}, ...]
    """
    documents = []
    
    # Your code here:
    # 1. Get list of all PDF files in papers_dir
    if not os.path.exists(papers_dir):
        print(f"Directory {papers_dir} does not exist.")
        return documents

    for filename in os.listdir(papers_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(papers_dir, filename)
    # 2. For each PDF:
    #    a. Open it with pdfplumber
            try:
        #    b. Extract all text from all pages
                with pdfplumber.open(file_path) as pdf:
                    page_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            page_text.append(text)

                    full_text = "\n".join(page_text)

        #    c. Store: filename, full text, page count
                    documents.append({
                        "filename": filename,
                        "text": full_text.strip(),
                        "pages": len(pdf.pages)
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # 3. Return documents list
    return documents
    
    # HINT: Use os.listdir(), pdfplumber.open(), loop through pdf.pages
    # HINT: Join all page text with "\n"
    


# Test it
if __name__ == "__main__":
    docs = load_papers_from_directory()
    print(f"Loaded {len(docs)} papers")
    for doc in docs[:2]:  # Print first 2
        print(f"\n{doc['filename']}: {doc['pages']} pages, {len(doc['text'])} characters")