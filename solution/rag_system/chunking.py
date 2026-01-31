"""
Chunking - Split documents into smaller pieces for better retrieval.

WHY: 
- PDF pages can be very long (thousands of words)
- LLMs have token limits
- Smaller chunks = more precise search results
- Overlapping chunks ensure we don't lose context at boundaries
"""

from typing import List, Dict


def chunk_documents(documents: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split documents into overlapping chunks.
    
    Each chunk preserves the metadata from its source document.
    Chunks overlap to avoid splitting important information at boundaries.
    
    Args:
        documents: List of document dicts from pdf_loader.load_pdfs()
        chunk_size: Number of characters per chunk (default: 1000)
        overlap: Number of characters that overlap between chunks (default: 200)
        
    Returns:
        List of chunk dictionaries with same metadata structure as input
    
    Example:
        chunks = chunk_documents(documents, chunk_size=1000, overlap=200)
        print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    """
    chunks = []
    
    print(f"✂️ Chunking documents (size={chunk_size}, overlap={overlap})...")
    
    for doc in documents:
        text = doc['text']
        
        # Skip empty or very short pages
        if not text.strip() or len(text) < 100:
            continue
        
        # Split this document's text into chunks
        start = 0
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create chunk with all metadata from source document
            chunk = {
                'text': chunk_text,
                'company': doc['company'],
                'year': doc['year'],
                'page': doc['page'],
                'source_file': doc['source_file']
            }
            chunks.append(chunk)
            
            # Move to next chunk
            # We subtract overlap so chunks share some content
            start += (chunk_size - overlap)
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """
    Get statistics about the chunks.
    Useful for verification and debugging.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with chunk statistics
    """
    stats = {
        'total_chunks': len(chunks),
        'avg_chunk_length': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0,
        'companies': list(set([c['company'] for c in chunks])),
        'years': sorted(list(set([c['year'] for c in chunks])))
    }
    
    # Count chunks per company and year
    stats['by_company_year'] = {}
    for company in stats['companies']:
        stats['by_company_year'][company] = {}
        for year in stats['years']:
            count = len([c for c in chunks if c['company'] == company and c['year'] == year])
            stats['by_company_year'][company][year] = count
    
    return stats


if __name__ == "__main__":
    # Test chunking
    from pdf_loader import load_pdfs
    
    docs = load_pdfs("../Assignment/10-k_docs")
    chunks = chunk_documents(docs)
    stats = get_chunk_stats(chunks)
    
    print("\n📊 Chunk Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Avg chunk length: {stats['avg_chunk_length']:.0f} characters")
    print("\nChunks by company and year:")
    for company in stats['by_company_year']:
        print(f"\n{company}:")
        for year, count in stats['by_company_year'][company].items():
            print(f"  {year}: {count} chunks")
