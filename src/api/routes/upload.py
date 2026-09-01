from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.utils.file_handler import FileHandler
from src.ml.preprocessing.image_processor import ClothingImageProcessor
from src.ml.inference.classifier import ClothingClassifier
from src.utils.image_utils import ImageProcessor, ColorExtractor
from typing import List
import json
import logging

router = APIRouter(prefix="/upload", tags=["Upload"])


def _extract_brand(detected_text) -> str | None:
    """Return the most likely brand name from an OCR detection list.

    Brand names are typically ALL CAPS or Title Case, 1–4 words, ≥3 chars.
    Skips watermark keywords so catalogue images don't surface Shutterstock etc.
    """
    if not detected_text:
        return None
    watermark_kw = {"dlx", "shutterstock", "getty", "watermark", "alamy", "preview"}
    for item in detected_text:
        text = item.get("text", "").strip()
        if not text or len(text) < 3:
            continue
        if any(kw in text.lower() for kw in watermark_kw):
            continue
        words = text.split()
        if len(words) <= 4 and (text.isupper() or text.istitle()):
            return text
    return None


# Initialize image processor
image_processor = ClothingImageProcessor()
classifier = ClothingClassifier()

@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    user_id: int = 1, # TODO: replace with authenticated user
    db: Session = Depends(get_db)
):
    """Upload a single clothing image, process it, and save to database"""


    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    try:
        # Save file
        file_info = await FileHandler.save_upload(file, user_id)
        
        # Process image and extract features
        processed_data = image_processor.process_image(file_info["file_path"])

        original_image = ImageProcessor.load_image(file_info["file_path"])

        classification = classifier.classify(
            original_image,
            color_palette = processed_data["color_palette"],
            pattern = processed_data.get("pattern", "solid")
                                             )
        loud_patterns = ["tropical print", "bandana patchwork", "floral print", "geometric print"]
        detected_pattern = processed_data.get("pattern")

        if detected_pattern in loud_patterns:
            classification["formality"] = "casual"
            classification["formality_confidence"] = 0.95
            if classification["subcategory"] == "blouse":
                classification["subcategory"] = "shirt"

        # Pattern = the visual design on the fabric (solid, striped, floral, embroidered…).
        # Denim wash (dark wash, acid wash, etc.) is a texture/finish descriptor — it lives
        # in the denim_wash field only and must NOT overwrite pattern.
        # Priority: classifier override (e.g. "embroidered") → raw pre-processing pattern → "solid"
        classifier_pattern = classification.get("pattern")
        raw_pattern = processed_data.get("pattern", "solid")
        final_pattern = classifier_pattern or raw_pattern or "solid"

        # Knit fabric: 3D yarn ridges produce high edge density that detect_pattern mistakes
        # for a 2D graphic print. Override to "textured" when texture confirms it's knit.
        if classification.get("texture") == "knit" and final_pattern in {"printed", "patterned", "graphic print"}:
            final_pattern = "textured"

        palette = processed_data["color_palette"]
        dominant_color = ColorExtractor.semantic_primary_color(palette, final_pattern)
        processed_data["dominant_color"] = dominant_color
        palette["primary_color"] = dominant_color
        palette = json.loads(json.dumps(palette))

        # Create wardrobe item with ALL attributes
        wardrobe_item = WardrobeItem(
            user_id=user_id,
            image_path=file_info["file_path"],
            image_url=f"/uploads/{user_id}/{file_info['filename']}",

            # Basic classification
            category=classification["category"],
            subcategory=classification["subcategory"],
            formality=classification["formality"],
            season=classification["season"],

            # Color
            dominant_color=dominant_color,
            color_palette=palette,
            pattern=final_pattern,
    
            # NEW: Advanced attributes
            sleeve_length=classification.get("sleeve_length"),
            neckline=classification.get("neckline"),
            has_logo=1 if classification.get("has_logo") else 0,
            detected_text=classification.get("detected_text"),
            texture=classification.get("texture"),
            closure_type=classification.get("closure_type"),
            fit=classification.get("fit"),
            length=classification.get("length"),
            brand=_extract_brand(classification.get("detected_text")),
)
        
        db.add(wardrobe_item)
        db.commit()
        db.refresh(wardrobe_item)
        
        return {
            "status": "success",
            "message": "Image uploaded, processed, and saved to database",
            "data": {
                "id": wardrobe_item.id,
                "file_info": file_info,

                "classification": {
                    "category": classification["category"],
                    "category_confidence": classification["category_confidence"],
                    "subcategory": classification["subcategory"],
                    "subcategory_confidence": classification["subcategory_confidence"],
                    "formality": classification["formality"],
                    "formality_confidence": classification["formality_confidence"],
                    "season": classification["season"],
                    "low_confidence_warning": classification.get("low_confidence_warning", False),
                    "sleeve_length": wardrobe_item.sleeve_length,
                    "pattern": wardrobe_item.pattern,
                    "texture": wardrobe_item.texture,
                    "denim_wash": classification.get("denim_wash"),
                    "neckline": wardrobe_item.neckline,
                    "closure_type": wardrobe_item.closure_type,
                    "has_logo": wardrobe_item.has_logo,
                    "brand": wardrobe_item.brand,
                    "fit": wardrobe_item.fit
                },

                "color_info": {
                    "dominant_color": dominant_color,
                    "palette": palette
                },
                "processed_path": processed_data["processed_path"]
            }
        }
    
    except Exception as e:
        db.rollback()
        logging.exception(e)
        raise HTTPException(status_code=500, detail="Image processing failed")

@router.post("/images")
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Upload multiple clothing images with processing"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": "Only image files are allowed"
        })
            continue
        try:
            file_info = await FileHandler.save_upload(file, user_id)
            
            # Process image
            processed_data = image_processor.process_image(file_info["file_path"])
            original_image = ImageProcessor.load_image(file_info["file_path"])
            classification = classifier.classify(
                original_image,
                color_palette=processed_data["color_palette"],
                pattern=processed_data.get("pattern", "solid")
            )
            final_pattern = classification.get("pattern") or processed_data.get("pattern", "solid")

            palette = processed_data["color_palette"]
            dominant_color = ColorExtractor.semantic_primary_color(palette, final_pattern)
            processed_data["dominant_color"] = dominant_color
            palette["primary_color"] = dominant_color
            palette = json.loads(json.dumps(palette))
            
            # Save to database
            wardrobe_item = WardrobeItem(
                user_id=user_id,
                image_path=file_info["file_path"],
                image_url=f"/uploads/{user_id}/{file_info['filename']}",
    
                # Basic classification
                category=classification["category"],
                subcategory=classification["subcategory"],
                formality=classification["formality"],
                season=classification["season"],
    
                # Color
                dominant_color=dominant_color,
                color_palette=palette,
                pattern=final_pattern,
    
                # NEW: Advanced attributes
                sleeve_length=classification.get("sleeve_length"),
                neckline=classification.get("neckline"),
                has_logo=1 if classification.get("has_logo") else 0,
                detected_text=classification.get("detected_text"),
                texture=classification.get("texture"),
                closure_type=classification.get("closure_type"),
                fit=classification.get("fit"),
                length=classification.get("length")
            )
            db.add(wardrobe_item)
            db.commit()
            db.refresh(wardrobe_item)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "id": wardrobe_item.id,
                "category": classification["category"],
                "subcategory": classification["subcategory"],
                "dominant_color": dominant_color
            })
        except Exception as e:
            db.rollback()
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        # mark all successes as failed if commit fails
        for r in results:
            if r["status"] == "success":
                r["status"] = "failed"
                r["error"] = f"Commit failed: {str(e)}"
    
    return {
        "status": "completed",
        "uploaded": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }
