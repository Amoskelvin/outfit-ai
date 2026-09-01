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
        # Step 1: estimate background color from the 4 corner patches (not full border).
        # Full-border rows/columns fail on tight crops where a sleeve or hem reaches the
        # frame edge — the "background" estimate then reflects garment fabric and the
        # exclusion mask inverts, deleting the garment itself.  Corner patches are almost
        # always pure backdrop even on the tightest product-photo crops.
        _h_img, _w_img = image_for_analysis.shape[:2]
        _c = max(1, int(min(_h_img, _w_img) * 0.05))
        _corner_px = np.concatenate([
            image_for_analysis[:_c, :_c].reshape(-1, 3),
            image_for_analysis[:_c, -_c:].reshape(-1, 3),
            image_for_analysis[-_c:, :_c].reshape(-1, 3),
            image_for_analysis[-_c:, -_c:].reshape(-1, 3),
        ]).astype(np.float32)
        _bg_color = np.median(_corner_px, axis=0)

        # Step 2: detect whether the GrabCut mask is contaminated.
        # Track the failure reason separately so Step 3 can choose the right fallback.
        _mask_bad_ratio = (foreground_ratio > 0.85 or foreground_ratio < 0.05)
        _mask_bad_contam = False
        _mask_bad = _mask_bad_ratio
        if not _mask_bad and np.sum(mask) > 0:
            _masked_px = image_for_analysis[mask > 0].astype(np.float32)
            _dists = np.sqrt(np.sum((_masked_px - _bg_color) ** 2, axis=1))
            _bg_fraction = float(np.mean(_dists < 40))
            if _bg_fraction > 0.25:
                print(f"⚠️ Border-color contamination: {_bg_fraction:.0%} of 'foreground' "
                      f"matches background (bg≈{_bg_color.astype(int)}) — bad mask")
                _mask_bad = True
                _mask_bad_contam = True

        # Step 3: build the effective mask used for ALL downstream operations.
        if _mask_bad:
            # Special case: near-white background + contamination-only failure.
            # The contamination check fires because white garment fabric has the same
            # RGB distance to a white backdrop as actual background pixels.  But the
            # GrabCut mask correctly isolated the garment shape — use it directly.
            # Applying bg-exclusion here would delete white shirt fabric, leaving only
            # the print/graphic, and make dominant_color = the print colour instead of white.
            _bg_is_white = bool(np.min(_bg_color) > 220)
            if _mask_bad_contam and not _mask_bad_ratio and _bg_is_white:
                print(f"⚠️ White bg + contamination flag — using GrabCut mask "
                      f"(bg-exclusion would delete white garment pixels)")
                _effective_mask = mask
            else:
                # Non-white background or ratio-based failure: distance-based exclusion.
                # Garment pixels are always farther than 40 from a contrasting backdrop.
                _dist_2d = np.sqrt(np.sum(
                    (image_for_analysis.astype(np.float32) - _bg_color.astype(np.float32)) ** 2,
                    axis=2
                ))
                _bg_excl = (_dist_2d > 40).astype('uint8')
                _garment_cov = float(np.sum(_bg_excl)) / _bg_excl.size
                print(f"⚠️ Bad GrabCut mask — bg-exclusion fallback "
                      f"(garment coverage={_garment_cov:.2%}, bg≈{_bg_color.astype(int)})")
                _effective_mask = _bg_excl if 0.05 < _garment_cov < 0.90 else None
        else:
            _effective_mask = mask

        # All three downstream operations share the same effective mask so that
        # palette, colorblock detection, and pattern detection remain consistent.
        raw_palette = self.color_extractor.get_color_palette(image_for_analysis, mask=_effective_mask)
        color_palette = self._sanitize_for_json(raw_palette)

        # Analyze color distribution
        color_analysis = self.image_processor.analyze_color_distribution(image_for_analysis, mask=_effective_mask)

        # Detect pattern
        pattern = self.image_processor.detect_pattern(image_for_analysis, mask=_effective_mask)

        # Override before computing print_style: multi-color garments that the pixel
        # threshold still returns "solid" for have their colors arranged in large uniform
        # zones — that is colorblock, not a scattered graphic print.
        # detect_pattern returns "solid" when std_dev < 20 AND edge_density < 0.05,
        # meaning little spatial variation. High hue_variance with low spatial variation
        # means distinct color blocks, not printed artwork.
        # Guard: only upgrade to colorblock when the foreground mask is reliable.
        # foreground_ratio > 0.80 means the mask likely includes the studio background
        # (white/grey), so is_two_tone fires on garment + background, not two garment zones.
        if color_analysis.get("has_multiple_colors") and pattern == "solid" and foreground_ratio < 0.80:
            pattern = "colorblock"

        # Detect specific print style (must run AFTER the has_multiple_colors override)
        print_style = self.image_processor.detect_print_style(pattern, color_palette)
        dominant_color = self.color_extractor.semantic_primary_color(color_palette, print_style)
        color_palette["primary_color"] = dominant_color
        
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
            "dominant_color": dominant_color,
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
