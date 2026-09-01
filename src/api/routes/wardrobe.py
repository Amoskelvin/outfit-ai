from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import WardrobeItem
from src.database.schemas import WardrobeItemResponse
from typing import List, Optional

router = APIRouter(prefix="/wardrobe", tags=["Wardrobe"])

@router.get("/items", response_model=List[WardrobeItemResponse])
async def get_wardrobe_items(
    user_id: int = 1,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    color: Optional[str] = None,
    formality: Optional[str] = None,
    season: Optional[str] = None,
    sleeve_length: Optional[str] = None,
    has_logo: Optional[bool] = None,
    texture: Optional[str] = None,
    closure_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all wardrobe items for a user"""
    query = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id)
    
    if category:
        query = query.filter(WardrobeItem.category == category)

    if subcategory:
        query = query.filter(WardrobeItem.subcategory == subcategory)

    if color:
        query = query.filter(WardrobeItem.dominant_color == color)

    if formality:
        query = query.filter(WardrobeItem.formality == formality)

    if season:
        query = query.filter(WardrobeItem.season.like(f"%{season}%"))
    
    if sleeve_length:
        query = query.filter(WardrobeItem.sleeve_length == sleeve_length)
    
    if has_logo is not None:
        query = query.filter(WardrobeItem.has_logo == (1 if has_logo else 0))
    
    if texture:
        query = query.filter(WardrobeItem.texture == texture)
    
    if closure_type:
        query = query.filter(WardrobeItem.closure_type == closure_type)
    
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
    from sqlalchemy import func

    items = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id).all()
    total_items = len(items)
    
    # Count by category
    categories = db.query(
        WardrobeItem.category,
        func.count(WardrobeItem.id).label('count')
    ).filter(
        WardrobeItem.user_id == user_id
    ).group_by(WardrobeItem.category).all()
    
    # Count by subcategory
    subcategories = db.query(
        WardrobeItem.subcategory,
        func.count(WardrobeItem.id).label('count')
    ).filter(
        WardrobeItem.user_id == user_id
    ).group_by(WardrobeItem.subcategory).all()
    
    # Count by color
    colors = db.query(
        WardrobeItem.dominant_color,
        func.count(WardrobeItem.id).label('count')
    ).filter(
        WardrobeItem.user_id == user_id
    ).group_by(WardrobeItem.dominant_color).all()
    
    # Count by formality
    formality = db.query(
        WardrobeItem.formality,
        func.count(WardrobeItem.id).label('count')
    ).filter(
        WardrobeItem.user_id == user_id
    ).group_by(WardrobeItem.formality).all()
    
    return {
        "total_items": total_items,
        "by_category": {cat: count for cat, count in categories if cat},
        "by_subcategory": {subcat: count for subcat, count in subcategories if subcat},
        "by_color": {color: count for color, count in colors if color},
        "by_formality": {form: count for form, count in formality if form}
    }

@router.get("/filter-options")
async def get_filter_options(
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get available filter options based on user's wardrobe"""
    from sqlalchemy import func, distinct
    
    categories = db.query(distinct(WardrobeItem.category)).filter(
        WardrobeItem.user_id == user_id
    ).all()
    
    subcategories = db.query(distinct(WardrobeItem.subcategory)).filter(
        WardrobeItem.user_id == user_id
    ).all()
    
    colors = db.query(distinct(WardrobeItem.dominant_color)).filter(
        WardrobeItem.user_id == user_id
    ).all()
    
    formality_levels = db.query(distinct(WardrobeItem.formality)).filter(
        WardrobeItem.user_id == user_id
    ).all()
    
    return {
        "categories": [c[0] for c in categories if c[0]],
        "subcategories": [s[0] for s in subcategories if s[0]],
        "colors": [col[0] for col in colors if col[0]],
        "formality_levels": [f[0] for f in formality_levels if f[0]]
    }