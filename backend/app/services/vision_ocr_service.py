"""Google Cloud Vision API service for OCR and document analysis."""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage, vision, vision_v1
from google.oauth2 import service_account
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader

logger = logging.getLogger("uvicorn.error").getChild("vision_ocr_service")

# Configuration
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ".json")
USE_VISION_API = os.getenv("USE_VISION_API", "false").lower() == "true"
USE_VISION_ASYNC = os.getenv("USE_VISION_ASYNC", "false").lower() == "true"
VISION_ASYNC_INPUT_BUCKET = os.getenv("VISION_ASYNC_INPUT_BUCKET", "")
VISION_ASYNC_OUTPUT_BUCKET = os.getenv("VISION_ASYNC_OUTPUT_BUCKET", "")
VISION_ASYNC_PAGES_PER_BATCH = int(os.getenv("VISION_ASYNC_PAGES_PER_BATCH", "20"))


class VisionOCRService:
    """Service for Google Cloud Vision API operations."""

    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Vision API client.

        Args:
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        
        Raises:
            FileNotFoundError: If credentials file is not found.
            RuntimeError: If Vision API client initialization fails.
        """
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Google Cloud credentials file not found: {self.credentials_path}"
            )
        
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("Google Vision API client initialized successfully.")

            self.storage_client: Optional[storage.Client]
            if USE_VISION_ASYNC:
                try:
                    self.storage_client = storage.Client(
                        credentials=credentials,
                        project=credentials.project_id,
                    )
                    logger.info("Google Cloud Storage client initialized for async Vision OCR.")
                except Exception as storage_exc:
                    logger.error("Failed to initialize Storage client: %s", storage_exc)
                    self.storage_client = None
                try:
                    self.async_client = vision_v1.ImageAnnotatorClient(credentials=credentials)
                    logger.info("Vision async client initialized successfully.")
                except Exception as async_exc:
                    logger.error("Failed to initialize Vision async client: %s", async_exc)
                    self.async_client = None
            else:
                self.storage_client = None
                self.async_client = None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Vision API client: {exc}"
            ) from exc

    def extract_text_from_image(
        self, 
        image_path: str | Path,
        language_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract text from a single image using Vision API.

        Args:
            image_path: Path to the image file.
            language_hints: Optional list of language codes (e.g., ['en', 'hi', 'gu'])
                          to hint the OCR engine.

        Returns:
            Dictionary containing:
                - text: Extracted text content
                - confidence: Overall confidence score (0-1)
                - language: Detected language code
                - annotations: Full Vision API response for advanced usage

        Raises:
            FileNotFoundError: If image file doesn't exist.
            RuntimeError: If Vision API call fails.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read image file
            with io.open(str(image_path), "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Configure image context with language hints
            image_context = None
            if language_hints:
                image_context = vision.ImageContext(language_hints=language_hints)

            # Perform text detection
            response = self.client.document_text_detection(
                image=image,
                image_context=image_context
            )

            if response.error.message:
                raise RuntimeError(
                    f"Vision API error: {response.error.message}"
                )

            # Extract full text
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""

            # Detect language
            detected_language = None
            if response.full_text_annotation and response.full_text_annotation.pages:
                page = response.full_text_annotation.pages[0]
                if page.property and page.property.detected_languages:
                    detected_language = page.property.detected_languages[0].language_code

            # Calculate average confidence
            confidence = 0.0
            if response.text_annotations:
                # First annotation is the full text, skip it for confidence calculation
                if len(response.text_annotations) > 1:
                    confidences = [
                        annotation.confidence 
                        for annotation in response.text_annotations[1:] 
                        if hasattr(annotation, 'confidence')
                    ]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)

            logger.info(
                "OCR completed for %s: %d characters, language=%s, confidence=%.2f",
                image_path.name,
                len(full_text),
                detected_language or "unknown",
                confidence
            )

            return {
                "text": full_text,
                "confidence": confidence,
                "language": detected_language,
                "annotations": response.text_annotations,
            }

        except Exception as exc:
            logger.error("Vision API OCR failed for %s: %s", image_path.name, exc)
            raise RuntimeError(f"Vision API OCR failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Async OCR helpers
    # ------------------------------------------------------------------

    def _ensure_async_ready(self) -> None:
        if not USE_VISION_ASYNC:
            raise RuntimeError("Vision async OCR is not enabled. Set USE_VISION_ASYNC=true.")
        if not self.async_client:
            raise RuntimeError("Vision async client is not initialized.")
        if not self.storage_client:
            raise RuntimeError("Google Cloud Storage client is not initialized.")
        if not VISION_ASYNC_INPUT_BUCKET or not VISION_ASYNC_OUTPUT_BUCKET:
            raise RuntimeError(
                "VISION_ASYNC_INPUT_BUCKET and VISION_ASYNC_OUTPUT_BUCKET env variables must be set."
            )

    def _upload_pdf_to_gcs(self, pdf_path: Path, destination_blob: str) -> str:
        self._ensure_async_ready()
        bucket = self.storage_client.bucket(VISION_ASYNC_INPUT_BUCKET)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(str(pdf_path))
        uri = f"gs://{VISION_ASYNC_INPUT_BUCKET}/{destination_blob}"
        logger.info("Uploaded PDF to GCS for async OCR: %s", uri)
        return uri

    def _read_async_output(self, output_prefix: str) -> List[Dict[str, Any]]:
        self._ensure_async_ready()
        bucket = self.storage_client.bucket(VISION_ASYNC_OUTPUT_BUCKET)
        blobs = list(bucket.list_blobs(prefix=output_prefix))
        if not blobs:
            raise RuntimeError("Vision async OCR did not produce any output files.")

        pages: List[Dict[str, Any]] = []

        for blob in blobs:
            if not blob.name.lower().endswith(".json"):
                continue

            data = json.loads(blob.download_as_text(encoding="utf-8"))
            for response in data.get("responses", []):
                full_annotation = response.get("fullTextAnnotation") or {}
                text = full_annotation.get("text", "")

                language_code: Optional[str] = None
                pages_info = full_annotation.get("pages", [])
                if pages_info:
                    props = pages_info[0].get("property", {})
                    langs = props.get("detectedLanguages", [])
                    if langs:
                        language_code = langs[0].get("languageCode")

                pages.append({
                    "text": text,
                    "page_number": response.get("context", {}).get("pageNumber"),
                    "confidence": response.get("confidence", 0.0),
                    "language": language_code,
                })

        pages.sort(key=lambda entry: entry.get("page_number") or 0)
        return pages

    def _cleanup_async_artifacts(self, input_blob: str, output_prefix: str) -> None:
        if not self.storage_client:
            return

        input_bucket = self.storage_client.bucket(VISION_ASYNC_INPUT_BUCKET)
        output_bucket = self.storage_client.bucket(VISION_ASYNC_OUTPUT_BUCKET)

        try:
            blob_ref = input_bucket.blob(input_blob)
            if blob_ref.exists():
                blob_ref.delete()
                logger.info("Deleted Vision async input blob: %s", input_blob)
        except Exception as exc:
            logger.warning("Unable to delete Vision async input blob %s: %s", input_blob, exc)

        try:
            for blob in output_bucket.list_blobs(prefix=output_prefix):
                blob.delete()
            logger.info("Deleted Vision async output prefix: %s", output_prefix)
        except Exception as exc:
            logger.warning("Unable to delete Vision async output prefix %s: %s", output_prefix, exc)

    def extract_text_from_pdf_async(
        self,
        pdf_path: Path,
        language_hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract text from PDF using Vision async batch OCR."""

        self._ensure_async_ready()

        unique_id = uuid.uuid4().hex
        input_blob_name = f"vision-async-input/{unique_id}.pdf"
        output_prefix = f"vision-async-output/{unique_id}/"

        input_uri = self._upload_pdf_to_gcs(pdf_path, input_blob_name)
        output_uri = f"gs://{VISION_ASYNC_OUTPUT_BUCKET}/{output_prefix}"

        features = [vision_v1.Feature(type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)]

        input_config = vision_v1.InputConfig(
            gcs_source=vision_v1.GcsSource(uri=input_uri),
            mime_type="application/pdf",
        )

        output_config = vision_v1.OutputConfig(
            gcs_destination=vision_v1.GcsDestination(uri=output_uri),
            batch_size=max(1, min(VISION_ASYNC_PAGES_PER_BATCH, 20)),
        )

        image_context = None
        if language_hints:
            image_context = vision_v1.ImageContext(language_hints=language_hints)

        request = vision_v1.AsyncAnnotateFileRequest(
            features=features,
            input_config=input_config,
            output_config=output_config,
            image_context=image_context,
        )

        logger.info(
            "Starting Vision async OCR job: input=%s output=%s batch=%d",
            input_uri,
            output_uri,
            output_config.batch_size,
        )

        operation = self.async_client.async_batch_annotate_files(requests=[request])

        try:
            operation.result(timeout=900)
            logger.info("Vision async OCR job completed: %s", operation.operation.name)
        except Exception as exc:
            raise RuntimeError(f"Vision async OCR job failed: {exc}") from exc

        try:
            pages = self._read_async_output(output_prefix)
            segments: List[str] = []
            for page in pages:
                page_text = (page.get("text") or "").strip()
                if not page_text:
                    continue
                page_number = page.get("page_number", "?")
                segments.append(f"--- Page {page_number} ---\n{page_text}")

            combined_text = "\n".join(segments)

            avg_confidence = (
                sum(p.get("confidence", 0.0) for p in pages) / len(pages)
                if pages
                else 0.0
            )
            detected_language = None
            for page in pages:
                if page.get("language"):
                    detected_language = page["language"]
                    break

            result = {
                "text": combined_text,
                "pages": pages,
                "total_pages": len(pages),
                "language": detected_language,
                "avg_confidence": avg_confidence,
            }
        finally:
            self._cleanup_async_artifacts(input_blob_name, output_prefix)

        return result

    def extract_text_from_pdf(
        self,
        pdf_path: str | Path,
        language_hints: Optional[List[str]] = None,
        dpi: int = 300,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text from a PDF using Vision API.
        
        First tries native PDF processing (no Poppler needed).
        Falls back to image conversion only if native processing fails.

        Args:
            pdf_path: Path to the PDF file.
            language_hints: Optional list of language codes for OCR hints.
            dpi: DPI for PDF to image conversion (only used in fallback).
            max_pages: Maximum number of pages to process (None = all pages).

        Returns:
            Dictionary containing extracted text and metadata.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if USE_VISION_ASYNC:
            try:
                logger.info("Using Vision async OCR for PDF: %s", pdf_path.name)
                return self.extract_text_from_pdf_async(pdf_path, language_hints)
            except Exception as async_exc:
                logger.warning(
                    "Vision async OCR failed for %s: %s. Falling back to synchronous processing.",
                    pdf_path.name,
                    async_exc,
                )

        logger.info("Attempting native PDF processing with Vision API: %s", pdf_path.name)

        try:
            return self.extract_text_from_pdf_native(pdf_path, language_hints, max_pages)
        except Exception as exc:
            logger.warning(
                "Native PDF processing failed for %s: %s. Attempting image conversion fallback.",
                pdf_path.name,
                exc
            )
            
            # Fallback to image conversion (requires Poppler)
            try:
                return self._extract_text_from_pdf_images(pdf_path, language_hints, dpi, max_pages)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Both native PDF processing and image conversion failed. "
                    f"Native error: {exc}. Image conversion error: {fallback_exc}"
                ) from fallback_exc

    def _extract_text_from_pdf_images(
        self,
        pdf_path: Path,
        language_hints: Optional[List[str]] = None,
        dpi: int = 300,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text by converting PDF pages to images (requires Poppler).
        This is a fallback method.
        """
        logger.info("Converting PDF to images: %s", pdf_path.name)

        try:
            # Convert PDF pages to images
            last_page = max_pages if max_pages else None
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                last_page=last_page
            )
            
            logger.info("Processing %d pages from %s", len(images), pdf_path.name)

        except Exception as exc:
            raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

        # Process each page
        page_results = []
        all_text_parts = []
        confidences = []
        detected_language = None

        for page_num, image in enumerate(images, start=1):
            try:
                # Save image to temporary bytes buffer
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Create Vision API image
                vision_image = vision.Image(content=img_byte_arr.getvalue())

                # Configure image context
                image_context = None
                if language_hints:
                    image_context = vision.ImageContext(language_hints=language_hints)

                # Perform OCR
                response = self.client.document_text_detection(
                    image=vision_image,
                    image_context=image_context
                )

                if response.error.message:
                    logger.warning(
                        "Vision API error on page %d: %s",
                        page_num,
                        response.error.message
                    )
                    continue

                # Extract text
                page_text = response.full_text_annotation.text if response.full_text_annotation else ""
                
                # Detect language from first page
                if page_num == 1 and response.full_text_annotation and response.full_text_annotation.pages:
                    page = response.full_text_annotation.pages[0]
                    if page.property and page.property.detected_languages:
                        detected_language = page.property.detected_languages[0].language_code

                # Calculate confidence
                page_confidence = 0.0
                if response.text_annotations and len(response.text_annotations) > 1:
                    page_confidences = [
                        annotation.confidence 
                        for annotation in response.text_annotations[1:] 
                        if hasattr(annotation, 'confidence')
                    ]
                    if page_confidences:
                        page_confidence = sum(page_confidences) / len(page_confidences)

                page_results.append({
                    "page_number": page_num,
                    "text": page_text,
                    "confidence": page_confidence,
                })

                all_text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                confidences.append(page_confidence)

                logger.info(
                    "Page %d/%d processed: %d characters, confidence=%.2f",
                    page_num,
                    len(images),
                    len(page_text),
                    page_confidence
                )

            except Exception as exc:
                logger.error("Failed to process page %d: %s", page_num, exc)
                page_results.append({
                    "page_number": page_num,
                    "text": "",
                    "confidence": 0.0,
                    "error": str(exc)
                })

        # Combine results
        combined_text = "\n".join(all_text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "PDF OCR completed: %d pages, %d characters, avg_confidence=%.2f",
            len(page_results),
            len(combined_text),
            avg_confidence
        )

        return {
            "text": combined_text,
            "pages": page_results,
            "total_pages": len(page_results),
            "language": detected_language,
            "avg_confidence": avg_confidence,
        }

    def extract_text_from_pdf_native(
        self,
        pdf_path: str | Path,
        language_hints: Optional[List[str]] = None,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text from PDF using Vision API's native PDF processing (NO POPPLER NEEDED).
        
        This method uses Vision API's batch file annotation to process PDFs directly
        without converting to images first.
        
        Args:
            pdf_path: Path to the PDF file.
            language_hints: Optional language hints for OCR.
            max_pages: Maximum pages to process (None = all pages).
            
        Returns:
            Dictionary containing extracted text and metadata.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info("Processing PDF natively with Vision API (no Poppler): %s", pdf_path.name)
        
        try:
            # Read PDF file as bytes
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()

            total_pages = 0
            try:
                reader = PdfReader(str(pdf_path))
                total_pages = len(reader.pages)
            except Exception as info_exc:
                logger.debug("Could not determine total pages for %s: %s", pdf_path.name, info_exc)

            if max_pages is not None:
                if total_pages:
                    pages_to_process = list(range(1, min(total_pages, max_pages) + 1))
                else:
                    pages_to_process = list(range(1, max_pages + 1))
            else:
                pages_to_process = list(range(1, total_pages + 1)) if total_pages else None

            if pages_to_process:
                logger.info(
                    "Preparing Vision batches for %d page(s) (max_pages=%s, total_pages=%s)",
                    len(pages_to_process),
                    max_pages,
                    total_pages or "unknown",
                )
                page_groups: List[List[int]] = [
                    pages_to_process[i : i + 5]
                    for i in range(0, len(pages_to_process), 5)
                ]
            else:
                # Fallback: let Vision decide (may process first 5 pages only)
                page_groups = [[]]

            aggregated_page_responses: List[Tuple[int, Any]] = []

            # Create input config for PDF
            input_config = vision.InputConfig(
                content=pdf_content,
                mime_type='application/pdf'
            )

            # Configure OCR features
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]

            # Create image context with language hints
            image_context = None
            if language_hints:
                image_context = vision.ImageContext(language_hints=language_hints)

            for batch_index, page_group in enumerate(page_groups, start=1):
                pages_arg = page_group if page_group else None

                if page_group:
                    logger.info(
                        "Vision API processing page batch %d/%d: pages %s",
                        batch_index,
                        len(page_groups),
                        page_group,
                    )
                else:
                    logger.info(
                        "Vision API processing page batch %d/%d (no explicit page list)",
                        batch_index,
                        len(page_groups),
                    )

                request = vision.AnnotateFileRequest(
                    input_config=input_config,
                    features=features,
                    image_context=image_context,
                    pages=pages_arg
                )

                response = self.client.batch_annotate_files(requests=[request])

                if not response.responses:
                    raise RuntimeError("No response from Vision API")

                file_response = response.responses[0]

                if file_response.error.message:
                    raise RuntimeError(f"Vision API error: {file_response.error.message}")

                if page_group:
                    for page_number, page_response in zip(page_group, file_response.responses):
                        aggregated_page_responses.append((page_number, page_response))
                else:
                    current_offset = len(aggregated_page_responses)
                    for idx, page_response in enumerate(file_response.responses, start=1):
                        aggregated_page_responses.append((current_offset + idx, page_response))

            if not aggregated_page_responses:
                raise RuntimeError("Vision API returned no page responses")

            aggregated_page_responses.sort(key=lambda item: item[0])

            # Extract text from all pages
            all_text_parts = []
            page_results = []
            detected_language = None
            confidences = []

            for page_num, page_response in aggregated_page_responses:
                if page_response.error.message:
                    logger.warning("Error on page %d: %s", page_num, page_response.error.message)
                    continue
                
                # Get full text
                if page_response.full_text_annotation:
                    page_text = page_response.full_text_annotation.text
                    
                    # Detect language from first page
                    if page_num == 1 and page_response.full_text_annotation.pages:
                        page_obj = page_response.full_text_annotation.pages[0]
                        if page_obj.property and page_obj.property.detected_languages:
                            detected_language = page_obj.property.detected_languages[0].language_code
                    
                    # Calculate confidence
                    page_confidence = 0.0
                    if page_response.text_annotations and len(page_response.text_annotations) > 1:
                        confs = [a.confidence for a in page_response.text_annotations[1:] if hasattr(a, 'confidence')]
                        if confs:
                            page_confidence = sum(confs) / len(confs)
                    
                    page_results.append({
                        "page_number": page_num,
                        "text": page_text,
                        "confidence": page_confidence,
                    })
                    
                    all_text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                    confidences.append(page_confidence)
                    
                    logger.info("Page %d: %d chars, conf=%.2f", page_num, len(page_text), page_confidence)
            
            # Combine results
            combined_text = "\n".join(all_text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(
                "Native PDF OCR done: %d pages, %d chars, conf=%.2f",
                len(page_results),
                len(combined_text),
                avg_confidence
            )
            
            return {
                "text": combined_text,
                "pages": page_results,
                "total_pages": len(page_results),
                "language": detected_language,
                "avg_confidence": avg_confidence,
            }
            
        except Exception as exc:
            logger.error("Native PDF processing failed for %s: %s", pdf_path.name, exc)
            raise RuntimeError(f"Vision API native processing failed: {exc}") from exc

    def detect_document_features(
        self,
        pdf_path: str | Path,
        max_labels: int = 10,
        max_pages: int = 3
    ) -> Dict[str, Any]:
        """
        Detect document features and labels (topics) using Vision API.

        This uses Vision's label detection to identify topics/subjects in the document.

        Args:
            pdf_path: Path to the PDF file.
            max_labels: Maximum number of labels to return per page.
            max_pages: Maximum number of pages to analyze for labels.

        Returns:
            Dictionary containing:
                - labels: List of detected labels with scores
                - topics: Extracted topic strings
                - categories: Detected categories

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            RuntimeError: If processing fails.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info("Detecting document features for: %s", pdf_path.name)

        try:
            # Convert first few pages to images
            images = convert_from_path(
                str(pdf_path),
                dpi=200,  # Lower DPI for label detection is sufficient
                last_page=max_pages
            )

        except Exception as exc:
            raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

        all_labels = []
        all_topics = set()

        for page_num, image in enumerate(images, start=1):
            try:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                vision_image = vision.Image(content=img_byte_arr.getvalue())

                # Detect labels
                response = self.client.label_detection(
                    image=vision_image,
                    max_results=max_labels
                )

                if response.error.message:
                    logger.warning(
                        "Label detection error on page %d: %s",
                        page_num,
                        response.error.message
                    )
                    continue

                # Collect labels
                for label in response.label_annotations:
                    all_labels.append({
                        "description": label.description,
                        "score": label.score,
                        "page": page_num
                    })
                    all_topics.add(label.description)

                logger.info(
                    "Page %d: Detected %d labels",
                    page_num,
                    len(response.label_annotations)
                )

            except Exception as exc:
                logger.error("Failed to detect labels on page %d: %s", page_num, exc)

        # Sort labels by score
        all_labels.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "Feature detection completed: %d unique topics from %d labels",
            len(all_topics),
            len(all_labels)
        )

        return {
            "labels": all_labels,
            "topics": list(all_topics),
            "total_labels": len(all_labels),
            "unique_topics": len(all_topics),
        }


# Global instance (lazy initialization)
_vision_service: Optional[VisionOCRService] = None


def get_vision_service() -> VisionOCRService:
    """
    Get or create the global Vision OCR service instance.

    Returns:
        VisionOCRService instance.

    Raises:
        RuntimeError: If service cannot be initialized.
    """
    global _vision_service
    
    if _vision_service is None:
        _vision_service = VisionOCRService()
    
    return _vision_service


def is_vision_api_enabled() -> bool:
    """Check if Vision API is enabled via environment variable."""
    return USE_VISION_API


# Convenience functions
def extract_text_from_image(
    image_path: str | Path,
    language_hints: Optional[List[str]] = None
) -> str:
    """
    Convenience function to extract text from an image.

    Args:
        image_path: Path to the image file.
        language_hints: Optional language hints for OCR.

    Returns:
        Extracted text content.
    """
    service = get_vision_service()
    result = service.extract_text_from_image(image_path, language_hints)
    return result["text"]


def extract_text_from_pdf(
    pdf_path: str | Path,
    language_hints: Optional[List[str]] = None,
    dpi: int = 300,
    max_pages: Optional[int] = None
) -> str:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        language_hints: Optional language hints for OCR.
        dpi: DPI for image conversion.
        max_pages: Maximum pages to process.

    Returns:
        Combined text from all pages.
    """
    service = get_vision_service()
    result = service.extract_text_from_pdf(pdf_path, language_hints, dpi, max_pages)
    return result["text"]


def detect_topics_from_pdf(
    pdf_path: str | Path,
    max_labels: int = 10,
    max_pages: int = 3
) -> List[str]:
    """
    Convenience function to detect topics from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        max_labels: Maximum labels per page.
        max_pages: Maximum pages to analyze.

    Returns:
        List of detected topic strings.
    """
    service = get_vision_service()
    result = service.detect_document_features(pdf_path, max_labels, max_pages)
    return result["topics"]
