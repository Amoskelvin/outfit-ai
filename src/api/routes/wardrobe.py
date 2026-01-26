from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.database.schemas import WardrobeItemResponse
from typing import List

router = APIRouter(prefix="/wardrobe", tags=["Wardrobe"])

@router.get("/items", response_model=List[WardrobeItemResponse])
async def get_wardrobe_items(
    user_id: int = 1,
    category: str = None,
    db: Session = Depends(get_db)
):
    """Get all wardrobe items for a user"""
    query = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id)
    
    if category:
        query = query.filter(WardrobeItem.category == category)
    
    items = query.all()
    return items

@router.get("/items/{item_id}", response_model=WardrobeItemResponse)
async def get_wardrobe_item(
    item_id: int,
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get a specific wardrobe item"""
    item = db.query(WardrobeItem).filter(
        WardrobeItem.id == item_id,
        WardrobeItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return item

@router.delete("/items/{item_id}")
async def delete_wardrobe_item(
    item_id: int,
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Delete a wardrobe item"""
    item = db.query(WardrobeItem).filter(
        WardrobeItem.id == item_id,
        WardrobeItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db.delete(item)
    db.commit()
    
    return {"status": "success", "message": "Item deleted"}

@router.get("/stats")
async def get_wardrobe_stats(
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get wardrobe statistics"""
    total_items = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id).count()
    
    # Count by category
    from sqlalchemy import func
    categories = db.query(
        WardrobeItem.category,
        func.count(WardrobeItem.id).label('count')
    ).filter(
        WardrobeItem.user_id == user_id
    ).group_by(WardrobeItem.category).all()
    
    return {
        "total_items": total_items,
        "by_category": {cat: count for cat, count in categories}
    }