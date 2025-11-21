
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
import base64
from PIL import Image
from io import BytesIO
from rich.console import Console

from src.config import load_config
from src.inference import infer, infer_batch
from src.inference.translation import infer as translate, infer_batch as translate_batch, preload_mbart_model
from src.utils import setup_logging, encode_image, pdf_to_images, detect_file_type

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]Starting server initialization...[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")
    
    logger.info(f"Found {len(config.translation_models)} active translation model(s)")
    
    transformer_models_loaded = 0
    transformer_models_failed = 0
    
    for model_config in config.translation_models:
        if model_config.type == "mbart":
            if preload_mbart_model(model_config):
                transformer_models_loaded += 1
            else:
                transformer_models_failed += 1
        elif model_config.type == "llm":
            logger.info(f"Skipping LLM model '{model_config.name}' (loaded via API, not transformers)")
        else:
            logger.warning(f"Unknown model type '{model_config.type}' for model '{model_config.name}', skipping")
    
    console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
    console.print(
        f"[bold green]Server initialization complete.[/bold green] "
        f"[cyan]Transformer models loaded:[/cyan] [bold]{transformer_models_loaded}[/bold], "
        f"[cyan]failed:[/cyan] [bold red]{transformer_models_failed}[/bold red]"
    )
    console.print("[bold green]" + "=" * 60 + "[/bold green]\n")
    
    yield


app = FastAPI(
    title="Insurance AI Engine - OCR & Translation API",
    description="API for OCR and Translation model evaluation and inference",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64ImageRequest(BaseModel):
    """Request model for base64 image inference."""
    image_base64: str
    model_name: Optional[str] = None
    max_retries: int = 3


class TranslationRequest(BaseModel):
    """Request model for translation inference."""
    text: str
    model_name: Optional[str] = None
    src_lang: Optional[str] = None
    target_lang: Optional[str] = None
    max_retries: int = 3


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation inference."""
    texts: List[str]
    model_name: Optional[str] = None
    src_lang: Optional[str] = None
    target_lang: Optional[str] = None
    max_workers: Optional[int] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Insurance AI Engine OCR & Translation API",
        "version": "0.1.0",
        "active_ocr_models": [model.name for model in config.ocr_models],
        "active_translation_models": [model.name for model in config.translation_models]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List all active models."""
    return {
        "ocr_models": [
            {
                "name": model.name,
                "url": model.url,
                "batch_size": model.batch_size,
                "max_workers": model.max_workers,
                "active": model.active
            }
            for model in config.ocr_models
        ],
        "translation_models": [
            {
                "name": model.name,
                "type": model.type,
                "url": model.url if model.type == "llm" else None,
                "model_path": model.model_path,
                "src_lang": model.src_lang,
                "target_lang": model.target_lang,
                "max_workers": model.max_workers,
                "active": model.active
            }
            for model in config.translation_models
        ]
    }


@app.post("/ocr/infer")
async def ocr_infer(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    max_retries: int = 3,
    pdf_dpi: int = 200
):
    """
    Run OCR inference on an uploaded image or PDF document.
    
    For PDFs, each page is converted to an image and processed separately.
    
    Args:
        file: Image file (PNG, JPEG, etc.) or PDF file to process
        model_name: Name of the model to use (defaults to first active model)
        max_retries: Maximum number of retry attempts per page/image
        pdf_dpi: DPI for PDF page rendering (default: 200)
        
    Returns:
        For images: Single OCR text result
        For PDFs: Dictionary with page-by-page results
    """
    try:
        model_config = None
        if model_name:
            for model in config.ocr_models:
                if model.name == model_name and model.active:
                    model_config = model
                    break
            if not model_config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found or not active"
                )
        else:
            if not config.ocr_models:
                raise HTTPException(
                    status_code=404,
                    detail="No active models found in configuration"
                )
            model_config = config.ocr_models[0]
        
        contents = await file.read()
        file_type = detect_file_type(contents, file.filename or "")
        
        if file_type == 'pdf':
            try:
                pages = pdf_to_images(contents, dpi=pdf_dpi)
                logger.info(f"Processing PDF with {len(pages)} pages")
                
                page_results = []
                img_base64_list = []
                page_numbers = []
                
                for page_num, page_image in pages:
                    img_base64 = encode_image(page_image)
                    img_base64_list.append(img_base64)
                    page_numbers.append(page_num)
                
                results = infer_batch(
                    img_base64_list, 
                    model_config, 
                    max_workers=model_config.max_workers
                )
                
                for page_num, result_text in zip(page_numbers, results):
                    page_results.append({
                        "page": page_num,
                        "text": result_text,
                        "status": "success" if result_text else "failed"
                    })
                
                return {
                    "model": model_config.name,
                    "file_type": "pdf",
                    "total_pages": len(pages),
                    "pages": page_results,
                    "status": "success"
                }
                
            except ImportError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF processing requires PyMuPDF: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process PDF: {str(e)}"
                )
        
        elif file_type == 'image':
            try:
                image = Image.open(BytesIO(contents))
                image = image.convert("RGB")
                img_base64 = encode_image(image)
                result = infer(img_base64, model_config, max_retries=max_retries)
                
                return {
                    "model": model_config.name,
                    "file_type": "image",
                    "text": result,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process image: {str(e)}"
                )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported formats: PDF, PNG, JPEG, GIF, BMP, WebP"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during OCR inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/infer-base64")
async def ocr_infer_base64(request: Base64ImageRequest):
    """
    Run OCR inference on a base64 encoded image.
    
    Args:
        request: Base64ImageRequest containing image_base64 and optional parameters
        
    Returns:
        OCR text result
    """
    try:
        model_config = None
        if request.model_name:
            for model in config.ocr_models:
                if model.name == request.model_name and model.active:
                    model_config = model
                    break
            if not model_config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model_name}' not found or not active"
                )
        else:
            if not config.ocr_models:
                raise HTTPException(
                    status_code=404,
                    detail="No active models found in configuration"
                )
            model_config = config.ocr_models[0]
        
        result = infer(request.image_base64, model_config, max_retries=request.max_retries)
        
        return {
            "model": model_config.name,
            "text": result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error during OCR inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/batch")
async def ocr_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = None,
    max_workers: Optional[int] = None,
    pdf_dpi: int = 200
):
    """
    Run OCR inference on multiple images/PDFs in parallel.
    
    For PDFs, all pages from all PDFs are processed together.
    
    Args:
        files: List of image or PDF files to process
        model_name: Name of the model to use (defaults to first active model)
        max_workers: Maximum number of parallel workers
        pdf_dpi: DPI for PDF page rendering (default: 200)
        
    Returns:
        Dictionary with results organized by file and page
    """
    try:
        model_config = None
        if model_name:
            for model in config.ocr_models:
                if model.name == model_name and model.active:
                    model_config = model
                    break
            if not model_config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found or not active"
                )
        else:
            if not config.ocr_models:
                raise HTTPException(
                    status_code=404,
                    detail="No active models found in configuration"
                )
            model_config = config.ocr_models[0]
        
        img_base64_list = []
        file_metadata = []
        
        for file_idx, file in enumerate(files):
            contents = await file.read()
            file_type = detect_file_type(contents, file.filename or "")
            
            if file_type == 'pdf':
                try:
                    pages = pdf_to_images(contents, dpi=pdf_dpi)
                    for page_num, page_image in pages:
                        img_base64 = encode_image(page_image)
                        img_base64_list.append(img_base64)
                        file_metadata.append({
                            "file_index": file_idx,
                            "filename": file.filename or f"file_{file_idx}",
                            "file_type": "pdf",
                            "page": page_num
                        })
                except ImportError as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"PDF processing requires PyMuPDF: {str(e)}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to process PDF {file.filename}: {str(e)}")
                    img_base64_list.append(None)
                    file_metadata.append({
                        "file_index": file_idx,
                        "filename": file.filename or f"file_{file_idx}",
                        "file_type": "pdf",
                        "page": None,
                        "error": str(e)
                    })
            
            elif file_type == 'image':
                try:
                    image = Image.open(BytesIO(contents))
                    image = image.convert("RGB")
                    img_base64 = encode_image(image)
                    img_base64_list.append(img_base64)
                    file_metadata.append({
                        "file_index": file_idx,
                        "filename": file.filename or f"file_{file_idx}",
                        "file_type": "image",
                        "page": None
                    })
                except Exception as e:
                    logger.warning(f"Failed to process image {file.filename}: {str(e)}")
                    img_base64_list.append(None)
                    file_metadata.append({
                        "file_index": file_idx,
                        "filename": file.filename or f"file_{file_idx}",
                        "file_type": "image",
                        "page": None,
                        "error": str(e)
                    })
            else:
                img_base64_list.append(None)
                file_metadata.append({
                    "file_index": file_idx,
                    "filename": file.filename or f"file_{file_idx}",
                    "file_type": "unknown",
                    "page": None,
                    "error": "Unsupported file type"
                })
        
        workers = max_workers if max_workers is not None else model_config.max_workers
        results = infer_batch(img_base64_list, model_config, max_workers=workers)
        file_results = {}
        for idx, (metadata, result) in enumerate(zip(file_metadata, results)):
            file_key = metadata["filename"]
            if file_key not in file_results:
                file_results[file_key] = {
                    "filename": file_key,
                    "file_type": metadata["file_type"],
                    "pages": [] if metadata["file_type"] == "pdf" else None,
                    "text": None if metadata["file_type"] == "pdf" else "",
                    "status": "success"
                }
            
            if metadata["file_type"] == "pdf":
                file_results[file_key]["pages"].append({
                    "page": metadata["page"],
                    "text": result,
                    "status": "success" if result else "failed"
                })
            else:
                file_results[file_key]["text"] = result
                file_results[file_key]["status"] = "success" if result else "failed"
            
            if "error" in metadata:
                file_results[file_key]["status"] = "error"
                file_results[file_key]["error"] = metadata["error"]
        
        return {
            "model": model_config.name,
            "files": list(file_results.values()),
            "total_files": len(files),
            "total_pages": sum(len(f["pages"]) if f["pages"] else 1 for f in file_results.values()),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch OCR inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translation/translate")
async def translation_translate(request: TranslationRequest):
    """Run translation inference on a single text."""
    try:
        model_config = None
        if request.model_name:
            for model in config.translation_models:
                if model.name == request.model_name and model.active:
                    model_config = model
                    break
            if not model_config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Translation model '{request.model_name}' not found or not active"
                )
        else:
            if not config.translation_models:
                raise HTTPException(
                    status_code=404,
                    detail="No active translation models found in configuration"
                )
            model_config = config.translation_models[0]
        
        result = translate(
            request.text,
            model_config,
            src_lang=request.src_lang,
            target_lang=request.target_lang,
            max_retries=request.max_retries
        )
        
        return {
            "model": model_config.name,
            "model_type": model_config.type,
            "src_lang": request.src_lang or model_config.src_lang,
            "target_lang": request.target_lang or model_config.target_lang,
            "translation": result,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during translation inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translation/batch")
async def translation_batch(request: BatchTranslationRequest):
    """Run translation inference on multiple texts in parallel."""
    try:
        model_config = None
        if request.model_name:
            for model in config.translation_models:
                if model.name == request.model_name and model.active:
                    model_config = model
                    break
            if not model_config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Translation model '{request.model_name}' not found or not active"
                )
        else:
            if not config.translation_models:
                raise HTTPException(
                    status_code=404,
                    detail="No active translation models found in configuration"
                )
            model_config = config.translation_models[0]
        
        workers = request.max_workers if request.max_workers is not None else model_config.max_workers
        
        results = translate_batch(
            request.texts,
            model_config,
            src_lang=request.src_lang,
            target_lang=request.target_lang,
            max_workers=workers
        )
        
        translations = []
        for idx, (text, translation) in enumerate(zip(request.texts, results)):
            translations.append({
                "index": idx,
                "original_text": text,
                "translation": translation,
                "status": "success" if translation else "failed"
            })
        
        return {
            "model": model_config.name,
            "model_type": model_config.type,
            "src_lang": request.src_lang or model_config.src_lang,
            "target_lang": request.target_lang or model_config.target_lang,
            "translations": translations,
            "total_texts": len(request.texts),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch translation inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

