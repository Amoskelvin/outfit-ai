import numpy as np
from typing import Dict
from src.ml.models.model_manager import ClothingClassifierModel
from src.config.settings import settings

class ClothingClassifier:
    """
    High-level interface for clothing classification
    """
    
    def __init__(self):
        self.model = ClothingClassifierModel(device=settings.device)
    
    def classify(self, image: np.ndarray) -> Dict:
        """
        Classify a clothing image
        
        Args:
            image: RGB numpy array from image processor
        
        Returns:
            Classification results with all attributes
        """
        try:
            results = self.model.classify_all(image)
            
            # Add season detection (rule-based)
            season = self._detect_season(
                results["category"],
                results["subcategory"]
            )
            results["season"] = season
            
            return results
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "category": "uncategorized",
                "category_confidence": 0.0,
                "subcategory": "unknown",
                "subcategory_confidence": 0.0,
                "formality": "casual",
                "formality_confidence": 0.0,
                "season": "all",
                "error": str(e)
            }
    
    def _detect_season(self, category: str, subcategory: str) -> str:
        """
        Rule-based season detection
        """
        # Summer items
        summer_items = ["tank top", "shorts", "sandals"]
        if subcategory in summer_items:
            return "summer"
        
        # Winter items
        winter_items = ["sweater", "coat", "boots"]
        if subcategory in winter_items:
            return "winter"
        
        # Fall/Spring items
        transition_items = ["jacket", "jeans", "sneakers"]
        if subcategory in transition_items:
            return "spring,fall"
        
        # Default: all seasons
        return "all"