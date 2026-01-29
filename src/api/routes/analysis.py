from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.ml.preprocessing.image_processor import ClothingImageProcessor
from pathlib import Path

router = APIRouter(prefix="/analysis", tags=["Analysis"])

image_processor = ClothingImageProcessor()

@router.post("/reprocess/{item_id}")
async def reprocess_item(
    item_id: int,
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Reprocess an existing wardrobe item to extract features"""
    # Get item from database
    item = db.query(WardrobeItem).filter(
        WardrobeItem.id == item_id,
        WardrobeItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if image file exists
    if not Path(item.image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Process image
        processed_data = image_processor.process_image(item.image_path)
        
        # Update database
        item.dominant_color = processed_data["dominant_color"]
        item.color_palette = processed_data["color_palette"]
        
        db.commit()
        db.refresh(item)
        
        return {
            "status": "success",
            "message": "Item reprocessed successfully",
            "data": {
                "id": item.id,
                "dominant_color": item.dominant_color,
                "color_palette": item.color_palette
            }
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/colors")
async def get_color_distribution(
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get color distribution across all wardrobe items"""
    items = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id).all()
    
    color_counts = {}
    for item in items:
        if item.dominant_color:
            color_counts[item.dominant_color] = color_counts.get(item.dominant_color, 0) + 1
    
    return {
        "total_items": len(items),
        "color_distribution": color_counts,
        "most_common_color": max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else None
    }