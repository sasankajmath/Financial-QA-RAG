"""
PDF Loader - Loads 10-K PDF files and extracts text with metadata.

This file reads all PDF files from the Assignment/10-k_docs folder and
extracts text from each page along with important metadata like:
- Company (AMZN or UBER)
- Year (2019, 2020, or 2021)
- Page number
- Source filename

WHY: We need metadata to filter searches by company and year (Problem 1)
"""

import PyPDF2
import os
from typing import List, Dict
from tqdm import tqdm


def load_pdfs(pdf_directory: str) -> List[Dict]:
    """
    Load all PDFs from directory and extract text with metadata.
    
    Args:
        pdf_directory: Path to folder containing company subfolders with PDFs
        
    Returns:
        List of dictionaries, each containing:
        - text: extracted text from one page
        - company: 'AMZN' or 'UBER'
        - year: 2019, 2020, or 2021
        - page: page number (1-indexed)
        - source_file: original PDF filename
    
    Example:
        documents = load_pdfs("Assignment/10-k_docs")
        print(f"Loaded {len(documents)} pages")
    """
    documents = []
    
    print("📄 Loading PDF files...")
    
    # Process each company folder
    for company_folder in ['Amazon', 'Uber']:
        company_path = os.path.join(pdf_directory, company_folder)
        
        # Skip if folder doesn't exist
        if not os.path.exists(company_path):
            print(f"⚠️ Warning: {company_path} not found")
            continue
        
        # Convert folder name to ticker symbol
        company_ticker = 'AMZN' if company_folder == 'Amazon' else 'UBER'
        print(f"\n  Processing {company_ticker}...")
        
        # Process each PDF in the folder
        pdf_files = [f for f in os.listdir(company_path) if f.endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc=f"  {company_ticker} PDFs"):
            # Extract year from filename
            year = extract_year_from_filename(pdf_file)
            
            # Read the PDF file
            pdf_path = os.path.join(company_path, pdf_file)
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Extract text from each page
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        # Only store pages with actual text content
                        if text.strip():
                            documents.append({
                                'text': text,
                                'company': company_ticker,
                                'year': year,
                                'page': page_num + 1,  # 1-indexed page numbers
                                'source_file': pdf_file
                            })
            
            except Exception as e:
                print(f"\n⚠️ Error reading {pdf_file}: {e}")
                continue
    
    print(f"\n✅ Loaded {len(documents)} pages from {len(set([d['source_file'] for d in documents]))} PDF files")
    return documents


def extract_year_from_filename(filename: str) -> int:
    """
    Extract fiscal year from SEC filename.
    
    SEC filenames are formatted as: XXXXXXXXXX-YY-XXXXXX.pdf
    where YY is the year suffix (e.g., "20" means 2020)
    
    Args:
        filename: PDF filename (e.g., "0001018724-20-000004.pdf")
        
    Returns:
        Full year as integer (e.g., 2020)
    
    Example:
        year = extract_year_from_filename("0001018724-20-000004.pdf")
        # Returns: 2020
    """
    # Split by dash to get parts
    parts = filename.split('-')
    
    # The second part is the year suffix
    year_suffix = int(parts[1])  # "20", "21", or "22"
    
    # Convert to full year
    full_year = 2000 + year_suffix
    
    return full_year


def get_document_stats(documents: List[Dict]) -> Dict:
    """
    Get summary statistics about loaded documents.
    Useful for debugging and verification.
    
    Args:
        documents: List of document dictionaries from load_pdfs()
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_pages': len(documents),
        'companies': list(set([d['company'] for d in documents])),
        'years': sorted(list(set([d['year'] for d in documents]))),
        'files': set([d['source_file'] for d in documents])
    }
    
    # Count pages per company and year
    stats['by_company'] = {}
    for company in stats['companies']:
        company_docs = [d for d in documents if d['company'] == company]
        stats['by_company'][company] = {}
        for year in stats['years']:
            year_docs = [d for d in company_docs if d['year'] == year]
            stats['by_company'][company][year] = len(year_docs)
    
    return stats


if __name__ == "__main__":
    # Test the PDF loader
    docs = load_pdfs("../Assignment/10-k_docs")
    stats = get_document_stats(docs)
    
    print("\n📊 Document Statistics:")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Companies: {stats['companies']}")
    print(f"Years: {stats['years']}")
    print("\nPages by company and year:")
    for company in stats['by_company']:
        print(f"\n{company}:")
        for year, count in stats['by_company'][company].items():
            print(f"  {year}: {count} pages")
