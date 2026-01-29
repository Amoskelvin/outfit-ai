import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import colorsys

class ImageProcessor:
    """Handles basic image processing operations"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def remove_background(image: np.ndarray, method: str = "grabcut") -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced background removal using GrabCut algorithm
        Returns: (image_with_removed_bg, mask)
        """
        height, width = image.shape[:2]
        if method == 'grabcut':
            try:
                # Iinitialize mask
                mask = np.zeros(image.shape[:2], np.uint8)
        
            # Define rectangle around the center (assuming clothing is centered)
                rect = (
                    int(width * 0.1),   # x
                    int(height * 0.1),  # y
                    int(width * 0.8),   # width
                    int(height * 0.8)   # height
                )
            
                # GrabCut algorithm
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
            
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                # Create mask where 0 and 2 are background, 1 and 3 are foreground
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

                foreground_ratio = np.sum(mask2) / (height * width)
                # If foreground is too small (<10%) or too large (>90%), GrabCut failed
                if foreground_ratio < 0.10:
                    print("⚠️ GrabCut extracted too little foreground, trying alternative method...")
                    return ImageProcessor.remove_background_color_based(image)
                
                if foreground_ratio > 0.90:
                    print("⚠️ GrabCut didn't remove background, trying alternative method...")
                    return ImageProcessor.remove_background_color_based(image)

                # Apply mask
                result = image * mask2[:, :, np.newaxis]
                
                return result, mask2
            except Exception as e:
                print(f"⚠️ GrabCut failed: {e}, using fallback method...")
                return ImageProcessor.remove_background_color_based(image)
        else:
            return ImageProcessor.remove_background_based(image)

    @staticmethod
    def remove_background_color_based(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Color-based background removal (works well for light/uniform backgrounds)
        Detects edges of the main object and creates mask
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
        # Get the corner pixels (likely background)
        h, w = image.shape[:2]
        corner_sample_size = min(50, h // 10, w // 10)
    
        corners = [
            image[0:corner_sample_size, 0:corner_sample_size],  # Top-left
            image[0:corner_sample_size, -corner_sample_size:],  # Top-right
            image[-corner_sample_size:, 0:corner_sample_size],  # Bottom-left
            image[-corner_sample_size:, -corner_sample_size:]   # Bottom-right
        ]
    
        # Calculate average background color from corners
        corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners])
        bg_color = np.median(corner_pixels, axis=0)
    
        # Create mask based on difference from background
        diff = np.abs(image.astype(float) - bg_color)
        color_distance = np.sqrt(np.sum(diff ** 2, axis=2))
    
        # Adaptive threshold: pixels very different from background = foreground
        threshold = np.percentile(color_distance, 30)  # Bottom 30% is likely background
        mask = (color_distance > threshold).astype('uint8')
    
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
        # Find largest connected component (the main garment)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
        if num_labels > 1:
            # Get largest component (excluding background which is label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype('uint8')
    
        # Dilate slightly to include edges
        mask = cv2.dilate(mask, kernel, iterations=2)
    
        # Ensure mask is not empty
        if np.sum(mask) < 100:
            print("⚠️ Color-based removal failed, using whole image")
            mask = np.ones(image.shape[:2], dtype='uint8')
    
        # Apply mask
        result = image * mask[:, :, np.newaxis]

        return result, mask
        
    @staticmethod
    def remove_background_simple(image: np.ndarray, threshold: int = 240) -> Tuple[np.ndarray, np.ndarray]:
        """Simple background removal (fallback method)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create mask for non-white pixels
        mask = (gray < threshold).astype('uint8')

        if np.sum(mask) < image.shape[0] * image.shape[1] * 0.1:
            # If mask is too small, lower threshold
            mask = (gray < threshold + 20).astype('uint8')
        
        # Apply mask
        result = image.copy()
        result[mask == 0] = [255, 255, 255]
        
        return result, mask
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image for ML model input"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def save_processed_image(image: np.ndarray, output_path: str):
        """Save processed image"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
    
    @staticmethod
    def detect_pattern(image: np.ndarray, mask: np.ndarray = None) -> str:
        """
        Detect if clothing has patterns (solid, striped, printed)
    
        Args:
            image: Input image (RGB)
            mask: Optional foreground mask
    
        Returns:
            Pattern type: "solid", "striped", "printed", "checkered"
        """
        # Apply mask if provided
        if mask is not None:
            image_to_analyze = image * mask[:, :, np.newaxis]
        else:
            image_to_analyze = image
    
        # Convert to grayscale for pattern analysis
        gray = cv2.cvtColor(image_to_analyze, cv2.COLOR_RGB2GRAY)

        # NEW: Blur slightly to ignore fabric texture (denim/linen threads)
        # but keep actual print patterns
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate edge density on the BLURRED image
        edges = cv2.Canny(blurred, 50, 150)
    
        # Remove background (black pixels from mask)
        if mask is not None:
            gray = gray[mask > 0]
    
        # Calculate standard deviation of pixel intensities
        std_dev = np.std(gray)
    
        # Calculate edge density
        edges = cv2.Canny(image_to_analyze, 50, 150)
        if mask is not None:
            edges = edges[mask > 0]
        edge_density = np.sum(edges > 0) / max(len(edges), 1)
    
        # Detect patterns based on variation
        if std_dev < 20 and edge_density < 0.05:
            return "solid"
        elif edge_density > 0.15:
            # High edge density - likely printed or patterned
        
            # Try to detect stripes using FFT
            try:
                # Fourier transform to detect periodic patterns
                f = np.fft.fft2(gray.reshape(int(np.sqrt(len(gray))), -1) if len(gray) > 100 else gray)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = np.abs(fshift)
            
                # If there are strong peaks in frequency domain, it's likely striped
                if np.max(magnitude_spectrum) > np.mean(magnitude_spectrum) * 10:
                    return "striped"
            except:
                pass
        
            return "printed"
        elif std_dev > 30:
            return "patterned"
        else:
            return "solid"

    @staticmethod
    def analyze_color_distribution(image: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Analyze color distribution with contrast detection
        """
        if mask is not None:
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
    
        pixels = pixels[np.any(pixels > 10, axis=1)]
    
        if len(pixels) == 0:
            return {
                "has_multiple_colors": False,
                "has_high_contrast": False,
                "color_variance": 0
            }
    
        # HSV analysis
        hsv_pixels = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) 
                               for r, g, b in pixels])
    
        # Hue variance
        hue_variance = np.var(hsv_pixels[:, 0]) * 360
    
        # Contrast detection: Check for both very dark and very light pixels
        value_range = np.max(hsv_pixels[:, 2]) - np.min(hsv_pixels[:, 2])
        has_dark = np.any(hsv_pixels[:, 2] < 0.25)
        has_light = np.any(hsv_pixels[:, 2] > 0.80)
    
        return {
            "has_multiple_colors": hue_variance > 0.15,
            "has_high_contrast": (has_dark and has_light) or value_range > 0.6,
            "color_variance": float(hue_variance),
            "average_saturation": float(np.mean(hsv_pixels[:, 1])),
            "is_vibrant": np.mean(hsv_pixels[:, 1]) > 0.3,
            "value_range": float(value_range)
        }
    
    @staticmethod
    def detect_print_style(pattern: str, color_palette: Dict) -> str:
        """
        Detect specific print styles based on pattern and colors
        """
        if pattern != "printed":
            return pattern
    
        colors = [c["name"] for c in color_palette.get("colors", [])]
    
        # Tropical/fruit print detection
        has_bright_colors = any(c in colors for c in ["bright orange", "orange", "yellow", "bright yellow"])
        has_green = any("green" in c for c in colors)
        has_dark_bg = any(c in colors for c in ["navy", "midnight blue", "black"])
    
        if has_bright_colors and has_green and has_dark_bg:
            return "tropical print"
    
        # Other print patterns can be added here
        return "printed"
    
    @staticmethod
    def detect_print_style(pattern: str, color_palette: Dict) -> str:
        """
        Detect specific print styles based on pattern and colors
        """
        if pattern not in ["printed", "patterned"]:
            return pattern
    
        colors = [c["name"] for c in color_palette.get("colors", [])]
    
        # Patchwork detection: Multiple colors + high contrast (black/white present)
        has_black = any(c in colors for c in ["black", "charcoal", "dark gray"])
        has_white = any(c in colors for c in ["white", "light gray"])
        has_multiple_hues = len(set(colors)) >= 3
    
        if has_black and has_white and has_multiple_hues:
            # Check if there's a dominant color family with contrasts
            reds = sum(1 for c in colors if "red" in c or "maroon" in c or "burgundy" in c)
            if reds >= 2:
                return "bandana patchwork"
            else:
                return "patchwork"
    
        # Tropical/fruit print detection
        has_bright_colors = any(c in colors for c in [
            "bright orange", "orange", "yellow", "bright yellow", 
            "bright red", "hot pink", "coral"
        ])
        has_green = any("green" in c for c in colors)
        has_dark_bg = any(c in colors for c in ["navy", "midnight blue", "black"])
    
        if has_bright_colors and has_green and has_dark_bg:
            return "tropical print"
    
        return "printed"
    
    @staticmethod
    def apply_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Corrects color cast (e.g., yellow bedroom lights) using 'Gray World' assumption.
        Makes white shirts actually look white.
        """
        result = image.copy()
        # Convert to LAB color space (separates lightness from color)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        
        # Shift the color channels (A and B) to be centered around 128 (neutral gray)
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result

class ColorExtractor:
    """Extract dominant colors from images"""
    
    @staticmethod
    def extract_dominant_colors(image: np.ndarray, n_colors: int = 5, use_mask: bool = True, mask: np.ndarray = None, return_counts: bool = False) -> List[Tuple[int, int, int]]:
        """
        Advanced color extraction with multi-stage analysis for complex patterns
        """ 
        from sklearn.cluster import KMeans
        
        # Get pixels
        if use_mask and mask is not None:
            foreground_pixels = image[mask > 0]
            if len(foreground_pixels) < n_colors:
                pixels = image.reshape(-1, 3)
            else:
                pixels = foreground_pixels
        else:
            pixels = image.reshape(-1, 3)
    
        # Basic filtering
        non_extreme_mask = (np.sum(pixels, axis=1) < 750) & (np.sum(pixels, axis=1) > 10)
        filtered_pixels = pixels[non_extreme_mask]
    
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels
    
        # === STAGE 1: Detect High-Contrast Elements (Black/White/Gray) ===
        contrast_colors = []
        contrast_counts = []
    
        # Detect BLACK (deep charcoal, black panels)
        black_mask = np.sum(filtered_pixels, axis=1) < 60  # Very dark
        if np.sum(black_mask) > len(filtered_pixels) * 0.05:  # If >5% of image
            black_pixels = filtered_pixels[black_mask]
            avg_black = np.mean(black_pixels, axis=0).astype(int)
            contrast_colors.append(tuple(avg_black))
            contrast_counts.append(np.sum(black_mask))
    
        # Detect WHITE (line work, details, accents)
        white_mask = np.sum(filtered_pixels, axis=1) > 600  # Very light
        if np.sum(white_mask) > len(filtered_pixels) * 0.015:  # If >3% of image
            white_pixels = filtered_pixels[white_mask]
            avg_white = np.mean(white_pixels, axis=0).astype(int)
            contrast_colors.append(tuple(avg_white))
            contrast_counts.append(np.sum(white_mask) * 2.0)
    
        # Detect GRAY (if present)
        hsv_pixels = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) 
                               for r, g, b in filtered_pixels])
        gray_mask = (hsv_pixels[:, 1] < 0.15) & (hsv_pixels[:, 2] > 0.20) & (hsv_pixels[:, 2] < 0.80)
        if np.sum(gray_mask) > len(filtered_pixels) * 0.05:
            gray_pixels = filtered_pixels[gray_mask]
            avg_gray = np.mean(gray_pixels, axis=0).astype(int)
            contrast_colors.append(tuple(avg_gray))
            contrast_counts.append(np.sum(gray_mask))

        # === STAGE 1.5: Detect BRIGHT YELLOWS/CREAMS (often missed) ===
        hsv_filtered = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) 
                        for r, g, b in filtered_pixels])

        # Detect bright yellows (peach interiors!)
        yellow_mask = (hsv_filtered[:, 0] > 0.12) & (hsv_filtered[:, 0] < 0.18) & \
                      (hsv_filtered[:, 1] > 0.3) & (hsv_filtered[:, 2] > 0.70)
        if np.sum(yellow_mask) > len(filtered_pixels) * 0.03:
            yellow_pixels = filtered_pixels[yellow_mask]
            avg_yellow = np.mean(yellow_pixels, axis=0).astype(int)
            contrast_colors.append(tuple(avg_yellow))
            contrast_counts.append(np.sum(yellow_mask) * 1.5)  # Boost yellow visibility
    
        # === STAGE 2: Extract Chromatic Colors (Reds, Blues, Greens, etc.) ===
        # Remove the contrast colors we already found
        chromatic_mask = ~(black_mask | white_mask | gray_mask | yellow_mask)
        chromatic_pixels = filtered_pixels[chromatic_mask]
    
        if len(chromatic_pixels) < 50:
            chromatic_pixels = filtered_pixels

        if len(chromatic_pixels) < 10:
            final_colors = contrast_colors[:n_colors]
            final_counts = contrast_counts[:n_colors]
            if return_counts:
                return final_colors, final_counts
            return final_colors

        # Convert to HSV for intelligent weighting
        hsv_chromatic = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) 
                                  for r, g, b in chromatic_pixels])
    
        # === AGGRESSIVE WEIGHTING ===
        saturation = hsv_chromatic[:, 1]
        value = hsv_chromatic[:, 2]
    
        weights = np.ones(len(chromatic_pixels))
    
        # Super boost for vibrant, saturated colors
        vibrant_mask = (saturation > 0.5) & (value > 0.4)
        weights[vibrant_mask] *= 6.0
    
        # Boost for saturated colors
        saturated_mask = (saturation > 0.35) & (value > 0.25)
        weights[saturated_mask] *= 3.0
    
        # Penalty for dark colors (but not too harsh, we want colored darks)
        dark_mask = value < 0.25
        weights[dark_mask] *= 0.5

        # === OPTIMIZATION: Sample pixels if too many ===
        if len(chromatic_pixels) > 5000:
            # Randomly sample 5000 pixels (much faster, still accurate)
            sample_indices = np.random.choice(len(chromatic_pixels), 5000, replace=False)
            chromatic_pixels_sampled = chromatic_pixels[sample_indices]
            weights_sampled = weights[sample_indices]
        else:
            chromatic_pixels_sampled = chromatic_pixels
            weights_sampled = weights
    
        # === STAGE 3: K-Means Clustering on Chromatic Colors ===
        # Calculate how many chromatic colors we need
        n_chromatic = max(2, n_colors - len(contrast_colors))
        n_clusters = min(n_chromatic + 5, len(chromatic_pixels_sampled))

        chromatic_colors = []
        chromatic_counts_sorted=[]
    
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(chromatic_pixels_sampled)
            colors = kmeans.cluster_centers_.astype(int)
        
            # Weighted counts
            chromatic_counts = []
            for i in range(n_clusters):
                cluster_mask = labels == i
                cluster_weight = np.sum(weights_sampled[cluster_mask])
                chromatic_counts.append(cluster_weight)
        
            # Get top chromatic colors
            sorted_indices = np.argsort(-np.array(chromatic_counts))[:n_chromatic]
            chromatic_colors = [tuple(colors[i]) for i in sorted_indices]
            chromatic_counts_sorted = [int(chromatic_counts[i]) for i in sorted_indices]
    
        # === STAGE 4: Combine Contrast + Chromatic Colors ===
        # Strategy: Interleave them based on visual importance
        all_colors = []
        all_counts = []
    
        # Add most dominant chromatic color first (the main color family)
        if chromatic_colors:
            all_colors.append(chromatic_colors[0])
            all_counts.append(chromatic_counts_sorted[0])
    
        # Add contrast colors (black/white are visually important)
        for i, color in enumerate(contrast_colors):
            all_colors.append(color)
            all_counts.append(contrast_counts[i])
    
        # Add remaining chromatic colors
        for i in range(1, len(chromatic_colors)):
            all_colors.append(chromatic_colors[i])
            all_counts.append(chromatic_counts_sorted[i])
    
        # Return top n_colors
        final_colors = all_colors[:n_colors]
        final_counts = all_counts[:n_colors]
    
        if return_counts:
            return final_colors, final_counts
        return final_colors
    
    @staticmethod
    def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to color name with high accuracy"""
        r, g, b = rgb
    
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h = h * 360  # Convert to degrees
    
        # === PURE WHITE/BLACK/GRAY (very strict) ===
        if v > 0.90 and s < 0.10:
            return "white"
    
        if v < 0.15:
            return "black"
    
        # Gray (low saturation across all brightness levels)
        if s < 0.10:
            if v < 0.30:
                return "charcoal"
            elif v < 0.50:
                return "dark gray"
            elif v < 0.70:
                return "gray"
            else:   
                return "light gray"
    
        # === DESATURATED COLORS (pastels, muted tones) ===
        if s < 0.25:
            if v > 0.75:
                # Pastel colors
                if h < 30 or h >= 330:
                    return "pale pink"
                elif h < 60:
                    return "cream"
                elif h < 150:
                    return "mint"
                elif h < 240:
                    return "pale blue"
                else:
                    return "lavender"
            else:
                # Muted/desaturated colors
                if h < 30 or h >= 330:
                    return "dusty rose"
                elif h < 90:
                    return "tan"
                elif h < 210:
                    return "slate blue"
                else:
                    return "mauve"
    
        # === DARK COLORS (low value, high saturation) ===
        if v < 0.30:
            if h < 30 or h >= 330:
                return "dark red"
            elif h < 60:
                return "brown"
            elif h < 150:
                return "dark green"
            elif h < 240:
                return "midnight blue"
            else:
                return "dark purple"
    
        # === MEDIUM-DARK COLORS (navy, maroon, forest green) ===
        if v < 0.50 and s > 0.30:
            if h < 20 or h >= 340:
                return "maroon"
            elif h < 40:
                return "rust"
            elif h < 80:
                return "olive"
            elif h < 160:
                return "forest green"  # THIS FIXES THE GREEN->NAVY BUG
            elif h < 200:
                return "teal"
            elif h < 250:
                return "navy"
            elif h < 290:
                return "indigo"
            else:
                return "burgundy"
    
        # === BRIGHT/SATURATED COLORS (the hero colors) ===
        # RED FAMILY
        if h < 15 or h >= 345:
            if v > 0.70:
                return "bright red"
            else:
                return "red"
    
        # ORANGE FAMILY
        elif h < 20:
            if s > 0.50 and v > 0.60:
                return "coral"  # For salmon/terracotta
            else:
                return "red-orange"
        elif h < 35:
            if s > 0.70:
                return "bright orange"
            elif s > 0.40 and v > 0.50:
                return "peach"  # Add peach color!
            else:
                return "orange"
        elif h < 45:
            if v > 0.70:
                return "orange"
            else:
                return "terracotta"  # Add terracotta!
    
        # YELLOW FAMILY
        elif h < 50:
            return "orange-yellow"
        elif h < 60:
            if s < 0.40 and v > 0.80:
                return "cream"
            elif v > 0.85:
                return "bright yellow"
            elif v > 0.70:
                return "yellow"
            else:
                return "mustard"
        elif h < 80:
            return "yellow-green"
    
        # GREEN FAMILY
        elif h < 100:
            return "lime green"
        elif h < 140:
            return "green"
        elif h < 170:
            return "forest green"  # Double coverage for green
    
        # CYAN/TURQUOISE
        elif h < 190:
            if s > 0.60:
                return "turquoise"
            else:
                return "cyan"
    
        # BLUE FAMILY
        elif h < 210:
            return "sky blue"
        elif h < 230:
            if v > 0.70:
                return "light blue"
            else:
                return "blue"
        elif h < 250:
            if v < 0.50:
                return "navy"
            else:
                return "royal blue"
    
        # PURPLE/VIOLET FAMILY
        elif h < 270:
            return "blue-purple"
        elif h < 290:
            return "purple"
        elif h < 310:
            return "violet"
    
        # MAGENTA/PINK FAMILY
        elif h < 330:
            if v > 0.70:
                return "hot pink"
            else:
                return "magenta"
        else:
            if v > 0.80:
                return "pink"
            else:
                return "rose"
    
    @staticmethod
    def get_color_palette(image: np.ndarray, n_colors: int = 5, mask: np.ndarray = None) -> Dict:
        """
        Get color palette with names and RGB values.
        Foreground-aware and background-safe.
        """

        # --- Prepare pixels ---
        pixels = image.reshape(-1, 3)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            pixels = pixels[mask_flat > 0]

        # Remove empty / black pixels (from background removal)
        pixels = pixels[np.any(pixels > 15, axis=1)]

        if len(pixels) == 0:
            return {
                "colors": [],
                "primary_color": None
            }

        # --- Extract dominant colors ---
        colors, counts = ColorExtractor.extract_dominant_colors(
            pixels,
            n_colors,
            return_counts=True
        )

        palette = {
            "colors": [],
            "primary_color": None
        }

        # Sort by dominance (most pixels first)
        sorted_indices = np.argsort(counts)[::-1]

        for idx in sorted_indices:
            color = colors[idx]
            color_name = ColorExtractor.rgb_to_color_name(color)

            palette["colors"].append({
                "rgb": [int(c) for c in color],
                "hex": f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}",
                "name": color_name
            })

        # Most dominant foreground color
        palette["primary_color"] = palette["colors"][0]["name"]

        return palette