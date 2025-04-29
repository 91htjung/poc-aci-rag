import re
from PyPDF2 import PdfReader

def parse_pdf(file):
    """
    Extract all text from a PDF file.
    `file` can be a file path or a file-like object (e.g., from an upload).
    """
    try:
        reader = PdfReader(file)
    except Exception:
        # If file is bytes, wrap in BytesIO
        from io import BytesIO
        reader = PdfReader(BytesIO(file))
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except Exception:
            continue
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of roughly `chunk_size` words with `overlap` word overlap.
    Returns a list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    i = 0
    while i < len(words):
        end = i + chunk_size
        chunk_words = words[i:end]
        if not chunk_words:
            break
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        if end >= len(words):
            break
        i += step
    return chunks

def extract_year(filename):
    """
    Extract a four-digit year from a filename, if present.
    Returns the year as an integer, or None if no year found.
    """
    years = re.findall(r'\d{4}', filename)
    years = [int(y) for y in years if 1900 <= int(y) <= 2100]
    if not years:
        return None
    # If multiple year candidates found, return the first occurrence
    return years[0]
