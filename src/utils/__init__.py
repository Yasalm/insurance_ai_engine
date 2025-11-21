"""Utility functions - imports from parent utils.py module."""

# Import from parent utils.py file
import sys
import os

# Add parent directory to path to import utils.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from utils.py module
from utils import (
    encode_image,
    extract_ground_truth_text,
    render_ocr_response,
    render_html_table,
    setup_logging,
    pdf_to_images,
    detect_file_type
)

__all__ = [
    "encode_image",
    "extract_ground_truth_text",
    "render_ocr_response",
    "render_html_table",
    "setup_logging",
    "pdf_to_images",
    "detect_file_type"
]

