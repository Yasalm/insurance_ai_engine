"""Utility functions for image processing, text extraction, rendering, and logging."""

import base64
import re
import logging
from io import BytesIO
from typing import List, Tuple
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.logging import RichHandler

console = Console()


def encode_image(image_input) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_input: Either a PIL Image object or file path to an image
        
    Returns:
        Base64 encoded string of the image
    """
    if isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[int, Image.Image]]:
    """
    Convert PDF bytes to a list of PIL Images (one per page).
    
    Args:
        pdf_bytes: PDF file content as bytes
        dpi: Resolution for PDF rendering (default: 200)
        
    Returns:
        List of tuples (page_number, PIL Image) for each page
        
    Raises:
        ImportError: If PyMuPDF is not installed
        Exception: If PDF processing fails
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF processing. "
            "Install it with: pip install pymupdf"
        )
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Render page to pixmap
            pix = page.get_pixmap(dpi=dpi)
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num + 1, img))  # Page numbers start at 1
        
        pdf_document.close()
        return images
        
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")


def detect_file_type(file_bytes: bytes, filename: str) -> str:
    """
    Detect file type from bytes and filename.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename
        
    Returns:
        File type: 'pdf', 'image', or 'unknown'
    """
    # Check by extension first
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return 'pdf'
    
    # Check by magic bytes
    if file_bytes.startswith(b'%PDF'):
        return 'pdf'
    
    # Check for image formats
    image_signatures = [
        (b'\xff\xd8\xff', 'image'),  # JPEG
        (b'\x89PNG\r\n\x1a\n', 'image'),  # PNG
        (b'GIF87a', 'image'),  # GIF
        (b'GIF89a', 'image'),  # GIF
        (b'BM', 'image'),  # BMP
        (b'RIFF', 'image'),  # WebP (starts with RIFF)
    ]
    
    for signature, file_type in image_signatures:
        if file_bytes.startswith(signature):
            return file_type
    
    # Try to open as image
    try:
        Image.open(BytesIO(file_bytes))
        return 'image'
    except:
        pass
    
    return 'unknown'


# Text processing utilities

def extract_ground_truth_text(ocr_data) -> str:
    """
    Extract ground truth text from OCR data structure.
    
    Args:
        ocr_data: List of dictionaries containing OCR data with 'text' keys
        
    Returns:
        Merged text string from all OCR data items
    """
    if isinstance(ocr_data, list):
        texts = [item.get("text", "") for item in ocr_data if isinstance(item, dict)]
        return " ".join(texts)
    return ""


def clean_html_markdown(text: str) -> str:
    """
    Remove HTML tags and markdown formatting from text.
    
    Args:
        text: Text that may contain HTML tags and markdown formatting
        
    Returns:
        Cleaned text with HTML tags and markdown removed
    """
    if not text:
        return ""
    
    # Fast path: if no HTML/markdown markers, return as-is
    if not any(c in text for c in '<*#`['):
        return text.strip()
    
    # Remove HTML tags (most common, do first)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown code blocks ```code``` (before inline code)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    
    # Remove markdown inline code `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown bold/italic (**text** or *text*) - handle bold first
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown lists (- or *)
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown horizontal rules (--- or ***)
    text = re.sub(r'^[\s]*[-*]{3,}[\s]*$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace (combine operations)
    text = re.sub(r'\s+', ' ', text)  # All whitespace to single space
    text = text.strip()
    
    return text


def clean_ocr_text(text: str, normalize_whitespace: bool = True, normalize_punctuation: bool = True) -> str:
    """
    Clean OCR output text for better metric comparison.
    Handles OCR-specific parsing issues like extra whitespace, punctuation differences, etc.
    
    Args:
        text: Raw OCR text to clean
        normalize_whitespace: Normalize all whitespace to single spaces
        normalize_punctuation: Normalize punctuation spacing
        
    Returns:
        Cleaned text ready for metric comparison
    """
    if not text:
        return ""
    
    # First clean HTML/markdown
    text = clean_html_markdown(text)
    
    # Normalize Unicode characters (common OCR errors)
    # Replace common Unicode variants with ASCII equivalents
    text = text.replace('\u2018', "'")  # Left single quotation mark
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201C', '"')   # Left double quotation mark
    text = text.replace('\u201D', '"')   # Right double quotation mark
    text = text.replace('\u2013', '-')   # En dash
    text = text.replace('\u2014', '--') # Em dash
    text = text.replace('\u2026', '...') # Ellipsis
    text = text.replace('\u00A0', ' ')  # Non-breaking space
    
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace all types of whitespace with single space
        text = re.sub(r'[\s\u00A0\u2000-\u200A\u202F\u205F]+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
    
    # Normalize punctuation spacing
    if normalize_punctuation:
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1\2', text)  # Fix double punctuation
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)  # Ensure single space after punctuation
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove excessive punctuation (more than 3 consecutive)
    text = re.sub(r'([.,!?;:])\1{3,}', r'\1\1\1', text)
    
    # Normalize multiple spaces to single space (final pass)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text


# UI rendering utilities

def render_ocr_response(text: str) -> None:
    """
    Render OCR response with markdown and HTML table support.
    
    Args:
        text: OCR response text that may contain HTML tables and markdown
    """
    table_pattern = r'<table>(.*?)</table>'
    tables = re.findall(table_pattern, text, re.DOTALL)
    
    # Replace tables with placeholders
    for i, table_html in enumerate(tables):
        text = text.replace(f'<table>{table_html}</table>', f'__TABLE_{i}__', 1)
    
    # Split text by table placeholders
    parts = re.split(r'__TABLE_(\d+)__', text)
    
    # Render parts
    for i, part in enumerate(parts):
        if part.isdigit() and int(part) < len(tables):
            render_html_table(tables[int(part)])
        else:
            clean_text = re.sub(r'<[^>]+>', '', part).strip()
            if clean_text:
                console.print(Markdown(clean_text))
                console.print()


def render_html_table(table_html: str) -> None:
    """
    Convert HTML table to rich Table and display it.
    
    Args:
        table_html: HTML table string to render
    """
    header_pattern = r'<t[hd]>(.*?)</t[hd]>'
    
    # Extract headers
    thead_match = re.search(r'<thead>(.*?)</thead>', table_html, re.DOTALL)
    if thead_match:
        header_row_match = re.search(r'<tr>(.*?)</tr>', thead_match.group(1), re.DOTALL)
        if header_row_match:
            headers = [h.strip() for h in re.findall(header_pattern, header_row_match.group(1), re.DOTALL)]
        else:
            return
    else:
        first_row_match = re.search(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
        if first_row_match:
            headers = [h.strip() for h in re.findall(header_pattern, first_row_match.group(1), re.DOTALL)]
        else:
            return
    
    # Create rich table
    rich_table = Table(show_header=True, header_style="bold cyan")
    for header in headers:
        rich_table.add_column(header)
    
    # Extract data rows
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL)
    if tbody_match:
        rows = re.findall(r'<tr>(.*?)</tr>', tbody_match.group(1), re.DOTALL)
    else:
        rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
        if not thead_match:
            rows = rows[1:]
    
    # Add data rows
    for row_html in rows:
        cells = [c.strip() for c in re.findall(header_pattern, row_html, re.DOTALL)]
        if cells:
            rich_table.add_row(*cells)
    
    console.print(rich_table)
    console.print()


# Logging utilities

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup rich logging configuration.
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)

