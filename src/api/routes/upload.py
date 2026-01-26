from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.utils.file_handler import FileHandler
from typing import List

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    user_id: int = 1,  # Temporary: hardcoded user, will add auth later
    db: Session = Depends(get_db)
):
    """Upload a single clothing image and save to database"""
    try:
        # Save file
        file_info = await FileHandler.save_upload(file, user_id)
        
        # Create wardrobe item in database
        wardrobe_item = WardrobeItem(
            user_id=user_id,
            image_path=file_info["file_path"],
            image_url=f"/uploads/{user_id}/{file_info['filename']}",
            category="uncategorized",  # Will be classified later by ML model
        )
        
        db.add(wardrobe_item)
        db.commit()
        db.refresh(wardrobe_item)
        
        return {
            "status": "success",
            "message": "Image uploaded and saved to database",
            "data": {
                "id": wardrobe_item.id,
                "file_info": file_info,
                "category": wardrobe_item.category
            }
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/images")
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Upload multiple clothing images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for file in files:
        try:
            file_info = await FileHandler.save_upload(file, user_id)
            
            # Save to database
            wardrobe_item = WardrobeItem(
                user_id=user_id,
                image_path=file_info["file_path"],
                image_url=f"/uploads/{user_id}/{file_info['filename']}",
                category="uncategorized",
            )
            db.add(wardrobe_item)
            db.commit()
            db.refresh(wardrobe_item)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "id": wardrobe_item.id,
                "data": file_info
            })
        except Exception as e:
            db.rollback()
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "uploaded": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }