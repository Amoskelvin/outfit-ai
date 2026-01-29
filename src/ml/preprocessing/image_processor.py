import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple
from src.utils.image_utils import ImageProcessor, ColorExtractor
from src.config.settings import settings

class ClothingImageProcessor:
    """Process clothing images for ML model input"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.color_extractor = ColorExtractor()
        self.target_size = (224, 224)

    def _sanitize_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    def process_image(self, image_path: str, save_processed: bool = True) -> Dict:
        """
        Process a clothing image and extract features
        
        Args:
            image_path: Path to the image file
            save_processed: Whether to save the processed image
        
        Returns:
            Dictionary containing processed data and extracted features
        """
        # Load image
        image = self.image_processor.load_image(image_path)

        image = self.image_processor.apply_white_balance(image)

        # === OPTIMIZATION: Downsample for color analysis ===
        # If image is too large, resize for analysis only
        original_size = image.shape[:2]
        if original_size[0] > 800 or original_size[1] >800:
            #Resize to max 800px for analysis 
            scale = 800 / max(original_size)
            analysis_size =(int(original_size[1] * scale), int(original_size[0] * scale))
            image_for_analysis = cv2.resize(image, analysis_size)
        else:
            image_for_analysis = image

        # Remove background
        image_no_bg, mask = self.image_processor.remove_background(image_for_analysis, method="grabcut")

        # === VALIDATION : Check mask quality ===
        foreground_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        foreground_ratio = foreground_pixels / total_pixels

        print(f"Foreground ratio: {foreground_ratio:.2%}")

        if foreground_ratio > 0.85:
            print("Warning: Mask might include background")
        elif foreground_ratio < 0.15:
            print("Warning: Mask might be too restrictive")

        # Extract color information using mask (with validation)
        if foreground_ratio > 0.85 or foreground_ratio < 0.05:
            # Bad mask - analyze without mask
            print("⚠️ Using whole image for color analysis due to poor mask quality")
            raw_palette = self.color_extractor.get_color_palette(image_for_analysis, mask=None)
        else:
            # Good mask - use it
            raw_palette = self.color_extractor.get_color_palette(image_for_analysis, mask=mask)
        
        color_palette = self._sanitize_for_json(raw_palette)

        # Analyze color distribution
        color_analysis = self.image_processor.analyze_color_distribution(image_for_analysis, mask=mask)

        # Detect pattern
        pattern = self.image_processor.detect_pattern(image_for_analysis, mask=mask)

        #Detect specific print style
        print_style = self.image_processor.detect_print_style(pattern, color_palette)

        #If highly varied colors, it's likely printed/patterned
        if color_analysis.get("has_multiple_colors") and pattern == "solid":
            pattern = "printed"
        
        # Resize image
        resized_image = self.image_processor.resize_image(image_no_bg, self.target_size)
        
        # Normalize for ML model
        normalized_image = self.image_processor.normalize_image(resized_image)
        
        # Save processed image if requested
        processed_path = None
        if save_processed:
            # Generate processed image path
            original_path = Path(image_path)
            filename = original_path.name
            user_id = original_path.parent.name
            
            processed_dir = Path(settings.processed_dir) / user_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / filename
            
            self.image_processor.save_processed_image(resized_image, str(processed_path))
        
        return {
            "original_path": image_path,
            "processed_path": str(processed_path) if processed_path else None,
            "original_size": image.shape[:2],
            "processed_size": resized_image.shape[:2],
            "color_palette": color_palette,
            "dominant_color": color_palette["primary_color"],
            "color_analysis": color_analysis,
            "normalized_array": normalized_image,  # internal use only
            'foreground_mask': mask,
            "pattern": print_style
        }

    
    def extract_features(self, image_path: str) -> Dict:
        """Extract all features from clothing image"""
        processed_data = self.process_image(image_path)
        
        return {
            "color_info": processed_data["color_palette"],
            "dominant_color": processed_data["dominant_color"],
            "image_size": processed_data["original_size"],
        }