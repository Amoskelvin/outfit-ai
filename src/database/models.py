from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    wardrobe_items = relationship("WardrobeItem", back_populates="owner")
    outfits = relationship("Outfit", back_populates="user")

class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Image info
    image_path = Column(String)
    image_url = Column(String)
    
    # Classification
    category = Column(String)  # top, bottom, shoes, accessory, outerwear
    subcategory = Column(String)  # shirt, jeans, sneakers, etc.
    
    # Attributes
    dominant_color = Column(String)
    color_palette = Column(JSON)  # List of colors
    pattern = Column(String)  # solid, striped, printed, etc.
    formality = Column(String)  # casual, business_casual, formal
    season = Column(String)  # spring, summer, fall, winter, all

    # Advanced attributes
    sleeve_length = Column(String, nullable=True)  # short sleeve, long sleeve, sleeveless
    neckline = Column(String, nullable=True)       # crew, v-neck, collar
    has_logo = Column(Integer, default=0)          # 0 or 1
    detected_text = Column(JSON, nullable=True)    # List of detected text/logos
    texture = Column(String, nullable=True)        # smooth, embossed, quilted
    closure_type = Column(String, nullable=True)   # zipper, buttons, pullover
    fit = Column(String, nullable=True)            # For pants: skinny, slim, regular
    length = Column(String, nullable=True)         # For bottoms/coats
    
    # ML embeddings
    embedding = Column(JSON)  # Feature vector from ML model
    
    # Metadata
    brand = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="wardrobe_items")

class Outfit(Base):
    __tablename__ = "outfits"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Outfit composition
    top_id = Column(Integer, ForeignKey("wardrobe_items.id"))
    bottom_id = Column(Integer, ForeignKey("wardrobe_items.id"))
    shoes_id = Column(Integer, ForeignKey("wardrobe_items.id"), nullable=True)
    outerwear_id = Column(Integer, ForeignKey("wardrobe_items.id"), nullable=True)
    accessory_ids = Column(JSON, nullable=True)  # List of accessory IDs
    
    # Scores
    compatibility_score = Column(Float)
    
    # Context
    occasion = Column(String, nullable=True)
    season = Column(String, nullable=True)
    
    # User feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    times_worn = Column(Integer, default=0)
    last_worn = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_favorite = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="outfits")