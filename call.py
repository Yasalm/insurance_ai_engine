from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import re

console = Console()

def render_ocr_response(text):
    """Render OCR response with markdown and HTML table support."""
    
    # Extract HTML tables
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
            # Render table
            table_html = tables[int(part)]
            render_html_table(table_html)
        else:
            # Render markdown (strip remaining HTML tags)
            clean_text = re.sub(r'<[^>]+>', '', part).strip()
            if clean_text:
                console.print(Markdown(clean_text))
                console.print()

def render_html_table(table_html):
    """Convert HTML table to rich Table."""
    header_pattern = r'<t[hd]>(.*?)</t[hd]>'
    
    # Check if there's a thead section
    thead_match = re.search(r'<thead>(.*?)</thead>', table_html, re.DOTALL)
    
    if thead_match:
        # Extract headers from thead
        thead_html = thead_match.group(1)
        header_row_match = re.search(r'<tr>(.*?)</tr>', thead_html, re.DOTALL)
        if header_row_match:
            headers = re.findall(header_pattern, header_row_match.group(1), re.DOTALL)
            headers = [h.strip() for h in headers]
        else:
            return
    else:
        # Extract headers from first row
        first_row_match = re.search(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
        if first_row_match:
            headers = re.findall(header_pattern, first_row_match.group(1), re.DOTALL)
            headers = [h.strip() for h in headers]
        else:
            return
    
    # Create rich table
    rich_table = Table(show_header=True, header_style="bold cyan")
    for header in headers:
        rich_table.add_column(header)
    
    # Extract data rows from tbody (or all rows if no tbody)
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL)
    if tbody_match:
        tbody_html = tbody_match.group(1)
        row_pattern = r'<tr>(.*?)</tr>'
        rows = re.findall(row_pattern, tbody_html, re.DOTALL)
    else:
        # No tbody, extract all rows but skip first if no thead
        row_pattern = r'<tr>(.*?)</tr>'
        rows = re.findall(row_pattern, table_html, re.DOTALL)
        if not thead_match:
            rows = rows[1:]  # Skip first row (used as headers)
    
    # Add data rows
    for row_html in rows:
        cells = re.findall(header_pattern, row_html, re.DOTALL)
        cells = [c.strip() for c in cells]
        if cells:
            rich_table.add_row(*cells)
    
    console.print(rich_table)
    console.print()

client = OpenAI(
    base_url="https://fpqr9yfck4x72w-8002.proxy.runpod.net/v1", 
    api_key="DUMMY_API_KEY",)

model = "nanonets/Nanonets-OCR2-3B"

def encode_image(image_input):
    if isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def infer(img_base64):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {
                            "type": "text",
                            "text": "Extract the text from the above document as if you were reading it naturally.",
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=15000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during inference: {type(e).__name__}: {e}")
        raise