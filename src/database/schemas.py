from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Wardrobe Item schemas
class WardrobeItemBase(BaseModel):
    category: str
    subcategory: Optional[str] = None
    dominant_color: Optional[str] = None
    pattern: Optional[str] = None
    formality: Optional[str] = None
    season: Optional[str] = None
    brand: Optional[str] = None
    notes: Optional[str] = None

class WardrobeItemCreate(WardrobeItemBase):
    pass

class WardrobeItemResponse(WardrobeItemBase):
    id: int
    user_id: int
    image_url: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Outfit schemas
class OutfitBase(BaseModel):
    top_id: int
    bottom_id: int
    shoes_id: Optional[int] = None
    outerwear_id: Optional[int] = None
    occasion: Optional[str] = None
    season: Optional[str] = None

class OutfitCreate(OutfitBase):
    pass

class OutfitResponse(OutfitBase):
    id: int
    user_id: int
    compatibility_score: float
    user_rating: Optional[int] = None
    times_worn: int
    is_favorite: bool
    created_at: datetime
    
    class Config:
        from_attributes = True