import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class ClothingClassifierModel:
    """
    CLIP-based zero-shot clothing classifier
    No training required - uses natural language descriptions
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
        
        # Define clothing categories and their descriptions
        self.category_labels = {
            "top": [
                "a photo of a shirt",
                "a photo of a t-shirt",
                "a photo of a blouse",
                "a photo of a sweater",
                "a photo of a jacket worn on upper body"
            ],
            "bottom": [
                "a photo of pants",
                "a photo of jeans",
                "a photo of shorts",
                "a photo of a skirt",
                "a photo of trousers"
            ],
            "shoes": [
                "a photo of shoes",
                "a photo of sneakers",
                "a photo of boots",
                "a photo of sandals",
                "a photo of heels"
            ],
            "outerwear": [
                "a photo of a jacket",
                "a photo of a coat",
                "a photo of a blazer",
                "a photo of outerwear"
            ],
            "accessories": [
                "a photo of a hat",
                "a photo of a bag",
                "a photo of a scarf",
                "a photo of sunglasses",
                "a photo of a belt"
            ]
        }
        
        # Subcategory descriptions
        self.subcategory_labels = {
            # Tops
            "t-shirt": "a casual t-shirt",
            "shirt": "a collared button-up shirt",
            "blouse": "a women's blouse",
            "sweater": "a knit sweater or pullover",
            "hoodie": "a hooded sweatshirt",
            "tank top": "a sleeveless tank top",
            
            # Bottoms
            "jeans": "blue denim jeans",
            "pants": "dress pants or trousers",
            "shorts": "casual shorts",
            "skirt": "a skirt",
            "leggings": "athletic leggings or tights",
            
            # Shoes
            "sneakers": "athletic sneakers or trainers",
            "boots": "leather or fashion boots",
            "sandals": "open-toe sandals",
            "heels": "high heeled shoes",
            "loafers": "slip-on dress shoes",
            
            # Outerwear
            "jacket": "a casual jacket",
            "coat": "a long winter coat",
            "blazer": "a formal blazer or suit jacket",
            
            # Accessories
            "hat": "a hat or cap",
            "bag": "a handbag or backpack",
            "scarf": "a scarf or shawl"
        }
        
        # Formality descriptions
        self.formality_labels = {
            "casual": "casual everyday clothing",
            "business casual": "business casual office attire",
            "formal": "formal dress clothing or suit"
        }
    
    def _load_model(self):
        """Load CLIP model"""
        print("Loading CLIP model...")
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            print("✅ CLIP model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            raise
    
    def classify_category(self, image: np.ndarray) -> Dict:
        """
        Classify clothing into main category (top, bottom, shoes, etc.)
        
        Args:
            image: RGB numpy array
        
        Returns:
            Dictionary with category, confidence, and all scores
        """
        # Convert numpy to PIL
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # Flatten all category descriptions
        all_descriptions = []
        category_mapping = []
        
        for category, descriptions in self.category_labels.items():
            for desc in descriptions:
                all_descriptions.append(desc)
                category_mapping.append(category)
        
        # Process inputs
        inputs = self.processor(
            text=all_descriptions,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Aggregate scores by category
        category_scores = {}
        for i, category in enumerate(category_mapping):
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(probs[i])
        
        # Average scores for each category
        category_avg_scores = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }
        
        # Get top category
        top_category = max(category_avg_scores.items(), key=lambda x: x[1])
        
        return {
            "category": top_category[0],
            "confidence": float(top_category[1]),
            "all_scores": {k: float(v) for k, v in category_avg_scores.items()}
        }
    
    def classify_subcategory(self, image: np.ndarray, category: str = None) -> Dict:
        """
        Classify clothing into specific subcategory (t-shirt, jeans, etc.)
        
        Args:
            image: RGB numpy array
            category: Optional main category to narrow down options
        
        Returns:
            Dictionary with subcategory and confidence
        """
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # If category provided, filter subcategories
        if category:
            # Filter subcategories relevant to this category
            relevant_subcats = self._get_relevant_subcategories(category)
        else:
            relevant_subcats = list(self.subcategory_labels.keys())
        
        descriptions = [self.subcategory_labels[sc] for sc in relevant_subcats]
        
        # Process
        inputs = self.processor(
            text=descriptions,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Get top subcategory
        top_idx = np.argmax(probs)
        top_subcat = relevant_subcats[top_idx]
        
        return {
            "subcategory": top_subcat,
            "confidence": float(probs[top_idx])
        }
    
    def classify_formality(self, image: np.ndarray) -> Dict:
        """Classify formality level"""
        pil_image = Image.fromarray(image.astype('uint8'))
        
        descriptions = list(self.formality_labels.values())
        formality_keys = list(self.formality_labels.keys())
        
        inputs = self.processor(
            text=descriptions,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        top_idx = np.argmax(probs)
        
        return {
            "formality": formality_keys[top_idx],
            "confidence": float(probs[top_idx])
        }
    
    def _get_relevant_subcategories(self, category: str) -> List[str]:
        """Get subcategories relevant to main category"""
        mapping = {
            "top": ["t-shirt", "shirt", "blouse", "sweater", "hoodie", "tank top"],
            "bottom": ["jeans", "pants", "shorts", "skirt", "leggings"],
            "shoes": ["sneakers", "boots", "sandals", "heels", "loafers"],
            "outerwear": ["jacket", "coat", "blazer"],
            "accessories": ["hat", "bag", "scarf"]
        }
        return mapping.get(category, list(self.subcategory_labels.keys()))
    
    def classify_all(self, image: np.ndarray) -> Dict:
        """
        Run all classifications at once
        
        Returns complete classification result
        """
        # Classify category
        category_result = self.classify_category(image)
        category = category_result["category"]
        
        # Classify subcategory
        subcat_result = self.classify_subcategory(image, category)
        
        # Classify formality
        formality_result = self.classify_formality(image)
        
        return {
            "category": category,
            "category_confidence": category_result["confidence"],
            "subcategory": subcat_result["subcategory"],
            "subcategory_confidence": subcat_result["confidence"],
            "formality": formality_result["formality"],
            "formality_confidence": formality_result["confidence"],
            "all_category_scores": category_result["all_scores"]
        }