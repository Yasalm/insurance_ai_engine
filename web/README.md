# Insurance AI Engine - Web Demo

A simple web interface to demonstrate the OCR and Translation API capabilities.

## Features

- **OCR**: Upload images (PNG, JPEG, GIF, BMP, WebP) or PDF files to extract text
- **Translation**: Translate text between languages
- **Markdown Rendering**: Properly renders markdown content in OCR and translation results
- **Configurable API URL**: Set the API server URL via environment variable or UI

## Usage

### Option 1: Open directly in browser

Simply open `index.html` in your web browser. The API URL can be configured in the UI.

### Option 2: Serve with a local server

For better CORS handling and file serving:

```bash
# Using Python
python -m http.server 8080

# Using Node.js (if you have http-server installed)
npx http-server -p 8080

# Using PHP
php -S localhost:8080
```

Then open `http://localhost:8080` in your browser.

## Configuration

The API server URL defaults to `https://fpqr9yfck4x72w-8003.proxy.runpod.net` but can be changed:

1. **Via UI**: Enter the URL in the "API Server URL" field and click "Save"
2. **Via localStorage**: The URL is saved in browser localStorage for persistence

## API Endpoints Used

- `POST /ocr/infer` - Upload file for OCR processing
- `POST /translation/translate` - Translate text

## File Structure

- `index.html` - Main HTML structure
- `styles.css` - Styling and layout
- `app.js` - JavaScript logic for API calls and UI interactions


