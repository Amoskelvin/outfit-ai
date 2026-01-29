from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.utils.file_handler import FileHandler
from src.ml.preprocessing.image_processor import ClothingImageProcessor
from typing import List
import json
import logging

router = APIRouter(prefix="/upload", tags=["Upload"])

# Initialize image processor
image_processor = ClothingImageProcessor()

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

        palette = processed_data["color_palette"]

        palette = json.loads(json.dumps(palette))
        
        # Create wardrobe item in database with extracted features
        wardrobe_item = WardrobeItem(
            user_id=user_id,
            image_path=file_info["file_path"],
            image_url=f"/uploads/{user_id}/{file_info['filename']}",
            category="uncategorized",
            dominant_color=processed_data["dominant_color"],
            color_palette=palette
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
                "color_info": {
                    "dominant_color": processed_data["dominant_color"],
                    "palette": processed_data["color_palette"]
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

            palette = processed_data["color_palette"]
            palette = json.loads(json.dumps(palette))
            
            # Save to database
            wardrobe_item = WardrobeItem(
                user_id=user_id,
                image_path=file_info["file_path"],
                image_url=f"/uploads/{user_id}/{file_info['filename']}",
                category="uncategorized",
                dominant_color=processed_data["dominant_color"],
                color_palette=palette,
                pattern=processed_data.get("pattern", "solid")
            )
            db.add(wardrobe_item)
            db.flush()
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "id": wardrobe_item.id,
                "dominant_color": processed_data["dominant_color"]
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