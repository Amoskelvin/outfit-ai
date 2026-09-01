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

        # Blur the SPATIAL image to suppress fabric-weave noise while keeping print edges.
        # The old approach (reshape masked pixels → random square → blur) destroyed spatial
        # structure: blurring randomly-arranged pixels acts like sample-averaging, collapsing
        # std_dev from ~50 (real graphic tee) to ~10 → false "solid" on dark garments.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Extract masked pixels for pixel-distribution statistics
        if mask is not None:
            gray_masked = gray[mask > 0]
            if len(gray_masked) == 0:
                return "solid"
        else:
            gray_masked = gray.ravel()

        # std_dev from raw pixel distribution (not blurred random square)
        std_dev = np.std(gray_masked)

        # Edge density from the spatially-correct blurred image
        edges = cv2.Canny(blurred, 50, 150)
        if mask is not None:
            edge_density = np.sum(edges[mask > 0] > 0) / max(np.sum(mask > 0), 1)
        else:
            edge_density = np.sum(edges > 0) / max(edges.size, 1)

        # Stripe check on the original spatial layout. Repeated horizontal/vertical
        # bands create strong row/column intensity oscillations; graphic prints tend to
        # be localized and less regular across the full garment.
        try:
            spatial_gray = cv2.cvtColor(image_to_analyze, cv2.COLOR_RGB2GRAY)
            if mask is not None:
                row_counts = np.sum(mask > 0, axis=1)
                col_counts = np.sum(mask > 0, axis=0)
                min_row_coverage = max(int(np.max(row_counts) * 0.45), 1)
                min_col_coverage = max(int(np.max(col_counts) * 0.45), 1)
                valid_rows = row_counts >= min_row_coverage
                valid_cols = col_counts >= min_col_coverage

                row_sum = np.sum(spatial_gray * (mask > 0), axis=1)
                col_sum = np.sum(spatial_gray * (mask > 0), axis=0)
                row_signal = row_sum[valid_rows] / np.maximum(row_counts[valid_rows], 1)
                col_signal = col_sum[valid_cols] / np.maximum(col_counts[valid_cols], 1)
            else:
                row_signal = np.mean(spatial_gray, axis=1)
                col_signal = np.mean(spatial_gray, axis=0)

            row_diff = np.diff(cv2.GaussianBlur(row_signal.reshape(-1, 1), (1, 5), 0).ravel())
            col_diff = np.diff(cv2.GaussianBlur(col_signal.reshape(-1, 1), (1, 5), 0).ravel())
            row_crossings = np.sum(np.diff(np.sign(row_diff)) != 0)
            col_crossings = np.sum(np.diff(np.sign(col_diff)) != 0)

            if (
                (len(row_signal) > 20 and np.std(row_diff) > 5 and row_crossings >= 8)
                or (len(col_signal) > 20 and np.std(col_diff) > 5 and col_crossings >= 8)
            ):
                return "striped"
        except Exception:
            pass

        # Keep gray as flat masked pixels for the FFT stripe check below
        if mask is not None:
            gray = gray_masked
    
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
        elif std_dev > 20:
            # Catches the gap zone (std_dev 20–30, edge_density 5–15%) that the old
            # threshold of 30 silently returned "solid" for — e.g. a black tee with
            # thin contour-line graphics: raw std_dev ~25, edges ~8% → patterned.
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

        # Hue variance — only on chromatic (visually saturated) pixels.
        # Near-achromatic pixels (black, white, grey, heather) have mathematically
        # undefined hue: tiny RGB differences from compression/texture noise cause
        # colorsys to return random hue values (e.g. (28,30,35) → h=210°).
        # Using all pixels therefore produces false high variance on solid garments.
        # Filter: s ≥ 0.20 AND v ≥ 0.20 selects genuinely chromatic pixels while
        # excluding near-black shadows (v<0.20) and near-white highlights (s<0.20).
        chromatic_mask = (hsv_pixels[:, 1] >= 0.20) & (hsv_pixels[:, 2] >= 0.25)
        n_chromatic = int(np.sum(chromatic_mask))
        n_total = len(hsv_pixels)

        if n_chromatic >= 50:
            hue_variance = float(np.var(hsv_pixels[chromatic_mask, 0]) * 360)
            # Structural two-tone: one chromatic color zone + a large achromatic base
            # (e.g. contrast-collar polo: dark green trim + white body).
            # Hue variance alone is near-zero (only one hue family) yet the garment
            # is visually multi-colored. Detect via pixel-count ratios.
            n_achromatic = int(np.sum(hsv_pixels[:, 1] < 0.10))
            is_two_tone = (
                n_chromatic > n_total * 0.08   # trim covers ≥8 % of garment
                and n_achromatic > n_total * 0.25  # base is largely achromatic
            )
            has_multiple_colors = hue_variance > 0.15 or is_two_tone
        else:
            # All pixels are achromatic — garment is solid regardless of value spread.
            hue_variance = 0.0
            has_multiple_colors = False

        # Contrast detection: Check for both very dark and very light pixels
        value_range = float(np.max(hsv_pixels[:, 2]) - np.min(hsv_pixels[:, 2]))
        has_dark = bool(np.any(hsv_pixels[:, 2] < 0.25))
        has_light = bool(np.any(hsv_pixels[:, 2] > 0.80))

        return {
            "has_multiple_colors": has_multiple_colors,
            "has_high_contrast": (has_dark and has_light) or value_range > 0.6,
            "color_variance": hue_variance,
            "average_saturation": float(np.mean(hsv_pixels[:, 1])),
            "is_vibrant": bool(np.mean(hsv_pixels[:, 1]) > 0.3),
            "value_range": value_range,
        }
    
    @staticmethod
    def detect_print_style(pattern: str, color_palette: Dict, texture: str = None) -> str:
        """
        Detect specific print styles based on pattern and colors
        """
        if texture and "embroid" in texture.lower():
            return "embroidered"
        if pattern not in ["printed", "patterned"]:
            return pattern
    
        colors = [c["name"] for c in color_palette.get("colors", [])]

        has_black = any(c in colors for c in ["black", "charcoal", "dark gray"])
        has_white = any(c in colors for c in ["white", "off-white", "light gray", "silver",
                                               "heather grey"])

        # has_dark_base: the SHIRT BODY is black/charcoal — the dark canvas of a graphic tee.
        # Must NOT match "dark green", "dark red" etc. — those are accent/trim colors, not a
        # dark body. Use an explicit set so "dark" substring can't cause false positives.
        _dark_base_names = {"black", "charcoal", "charcoal gray", "dark gray"}
        has_dark_base = any(c in _dark_base_names for c in colors)

        # has_multiple_hues: a graphic print needs 2+ distinct CHROMATIC (saturated) colors.
        # Counting "off-white" + "light gray" + "dark green" as 3 unique strings incorrectly
        # fires for a two-tone polo where the achromatic white body plus one trim color are
        # just structural design — not a multi-color artwork print.
        _achromatic = {
            "white", "off-white", "light gray", "light grey", "gray", "grey",
            "heather grey", "heather gray", "silver", "dark gray", "dark grey",
            "charcoal", "charcoal gray", "black"
        }
        _chromatic_colors = {c for c in colors if c not in _achromatic}
        has_multiple_hues = len(_chromatic_colors) >= 2

        # Tropical/fruit print detection (specific — must stay before generic graphic-print)
        has_bright_colors = any(c in colors for c in [
            "bright orange", "orange", "yellow", "bright yellow",
            "bright red", "hot pink", "coral", "mustard"])
        has_green = any("green" in c or "teal" in c for c in colors)
        has_dark_bg = any(c in colors for c in ["navy", "midnight blue", "black"])

        if has_bright_colors and has_green and has_dark_bg:
            return "tropical print"

        # Floral print: flower-family colors + greens (specific — stays before graphic-print)
        has_flower_colors = any(
            kw in c for c in colors
            for kw in ("pink", "coral", "peach", "rose", "lavender", "purple", "violet", "lilac")
        )
        has_green_leaves = any("green" in c or "teal" in c for c in colors)
        if has_white and has_flower_colors and has_green_leaves:
            return "floral print"

        # Graphic print: single fabric with printed design — dark or light base colour plus
        # at least one accent colour.  Checked BEFORE patchwork because graphic prints with
        # multi-colour illustrations (zombie, graffiti, etc.) always have a dominant base
        # (black or white/off-white shirt) that distinguishes them from true patchwork.
        # No colour-count cap: real graphic prints can have 5+ colours in the artwork.
        if (has_dark_base or has_white) and has_multiple_hues:
            return "graphic print"

        # Patchwork: sewn-together fabric pieces — only fires when there is NO clear single
        # dominant base colour (no black, no white/off-white), forcing multiple colours to
        # sit at roughly equal weight.
        if has_black and has_white and has_multiple_hues:
            reds = sum(1 for c in colors if "red" in c or "maroon" in c or "burgundy" in c)
            if reds >= 2:
                return "bandana patchwork"
            else:
                return "patchwork"

        # Only one chromatic color (e.g. dark green trim on a white polo) → the color
        # variation is structural (collar, cuffs, band), not a scattered print artwork.
        if len(_chromatic_colors) <= 1:
            return "colorblock"

        return "printed"

class ColorExtractor:
    """Extract dominant colors from images"""
    
    @staticmethod
    def extract_dominant_colors(image: np.ndarray, n_colors: int = 5, use_mask: bool = True, mask: np.ndarray = None, return_counts: bool = False) -> List[Tuple[int, int, int]]:
        from sklearn.cluster import KMeans
        
        # 1. INPUT HANDLING
        if image.ndim == 3:
            if use_mask and mask is not None and mask.shape[:2] == image.shape[:2]:
                foreground_pixels = image[mask > 0]
                pixels = foreground_pixels if len(foreground_pixels) > n_colors else image.reshape(-1, 3)
            else:
                pixels = image.reshape(-1, 3)
        else:
            pixels = image

        # 2. BASIC FILTERING
        if len(pixels) > 100:
            pixels = pixels[np.sum(pixels, axis=1) > 10]
        
        current_pixels = pixels

        # 3. ZEBRA FIX (Contrast Snapping)
        pixel_sums = np.sum(current_pixels, axis=1)
        has_deep_black = np.sum(pixel_sums < 150) > len(current_pixels) * 0.05
        has_pure_white = np.sum(pixel_sums > 600) > len(current_pixels) * 0.05
        
        if has_deep_black and has_pure_white:
            # Zebra Fix: purge achromatic mid-tone ghost pixels (anti-aliasing artifacts
            # between black and white stripes).  The original check deleted ALL pixels with
            # sum 150–600, which silently removed chromatic graphic colours (green zombie
            # face, yellow text, pink logo) whose RGB sums fall in the same range.
            # Fix: only delete mid-range pixels with LOW saturation (true gray artifacts).
            # Colorful pixels in that luminance band are real design elements — keep them.
            is_mid_range = (pixel_sums >= 150) & (pixel_sums <= 600)
            if np.any(is_mid_range):
                mid_pixels = current_pixels[is_mid_range]
                hsv_mid = np.array([colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                                    for r, g, b in mid_pixels])
                is_achromatic_mid = hsv_mid[:, 1] < 0.20  # saturation < 20% → gray artifact
                # Rebuild the keep mask: keep if NOT (mid-range AND achromatic)
                mid_indices = np.where(is_mid_range)[0]
                drop = np.zeros(len(current_pixels), dtype=bool)
                drop[mid_indices[is_achromatic_mid]] = True
                current_pixels = current_pixels[~drop]

            if len(current_pixels) < 50:
                current_pixels = pixels

        # 4. RECALCULATE MASKS (Crucial for avoiding the broadcast error)
        pixel_sums = np.sum(current_pixels, axis=1)
        
        black_mask = pixel_sums < 150
        white_mask = pixel_sums > 600
        
        hsv_pixels = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in current_pixels])

        # Gray Mask (Only if not striped)
        if has_deep_black and has_pure_white:
            gray_mask = np.zeros(len(current_pixels), dtype=bool)
        else:
            gray_mask = (hsv_pixels[:, 1] < 0.15) & (hsv_pixels[:, 2] > 0.20) & (hsv_pixels[:, 2] < 0.80)

        # Yellow Mask
        yellow_mask = (hsv_pixels[:, 0] > 0.12) & (hsv_pixels[:, 0] < 0.18) & \
                      (hsv_pixels[:, 1] > 0.3) & (hsv_pixels[:, 2] > 0.70)
        
        # Silver Mask (Restored!)
        silver_mask = (hsv_pixels[:, 1] < 0.20) & (hsv_pixels[:, 2] > 0.55) & (hsv_pixels[:, 2] < 0.88)

        # 5. STAGE 1: Extract High-Contrast Colors
        contrast_colors = []
        contrast_counts = []

        # Add Black
        if np.sum(black_mask) > len(current_pixels) * 0.05:
            avg_black = np.mean(current_pixels[black_mask], axis=0).astype(int)
            contrast_colors.append(tuple(avg_black))
            contrast_counts.append(np.sum(black_mask))

        # Add White
        white_count = np.sum(white_mask)
        avg_white = None
        if white_count > len(current_pixels) * 0.005:
            avg_white = np.mean(current_pixels[white_mask], axis=0).astype(int)
            contrast_colors.append(tuple(avg_white))
            contrast_counts.append(int(white_count * 5.0))

        # Add Silver (Restored!)
        silver_count = np.sum(silver_mask)
        if silver_count > len(current_pixels) * 0.01:
            avg_silver = np.mean(current_pixels[silver_mask], axis=0).astype(int)
            
            # Only add silver if it's distinct from white
            is_distinct = True
            if avg_white is not None:
                diff = np.sum(np.abs(avg_silver - avg_white))
                if diff < 40: is_distinct = False
            
            if is_distinct:
                contrast_colors.append(tuple(avg_silver))
                contrast_counts.append(int(silver_count * 3.0))

        # Add Gray
        if np.sum(gray_mask) > len(current_pixels) * 0.05:
            avg_gray = np.mean(current_pixels[gray_mask], axis=0).astype(int)
            contrast_colors.append(tuple(avg_gray))
            contrast_counts.append(np.sum(gray_mask))
            
        # Add Yellow
        if np.sum(yellow_mask) > len(current_pixels) * 0.02:
            avg_yellow = np.mean(current_pixels[yellow_mask], axis=0).astype(int)
            contrast_colors.append(tuple(avg_yellow))
            contrast_counts.append(np.sum(yellow_mask) * 2.0)

        # 6. STAGE 2: Extract Chromatic Colors
        # Exclude all special masks
        chromatic_mask = ~(black_mask | white_mask | gray_mask | yellow_mask | silver_mask)
        chromatic_pixels = current_pixels[chromatic_mask]

        if len(chromatic_pixels) < 10:
            final_colors = contrast_colors[:n_colors]
            # Handle return format based on flag
            if return_counts:
                 # Need to match counts to colors roughly if returning early
                 return final_colors, contrast_counts[:len(final_colors)]
            return final_colors

        # Weighting
        hsv_chromatic = hsv_pixels[chromatic_mask]
        weights = np.ones(len(chromatic_pixels))
        
        hue = hsv_chromatic[:, 0]
        sat = hsv_chromatic[:, 1]
        val = hsv_chromatic[:, 2]

        weights[(sat > 0.5) & (val > 0.4)] *= 6.0
        weights[(sat > 0.35) & (val > 0.25)] *= 3.0
        
        # Cool Color Boost
        cool_mask = (hue > 0.35) & (hue < 0.75) & (sat > 0.2)
        weights[cool_mask] *= 2.5

        # Sampling
        if len(chromatic_pixels) > 5000:
            indices = np.random.choice(len(chromatic_pixels), 5000, replace=False)
            sample_pixels = chromatic_pixels[indices]
            sample_weights = weights[indices]
        else:
            sample_pixels = chromatic_pixels
            sample_weights = weights

        # 7. STAGE 3: K-Means
        n_chromatic = max(2, n_colors - len(contrast_colors))
        n_clusters = min(n_chromatic + 3, len(sample_pixels))
        
        if n_clusters >= 1:
            kmeans = KMeans(n_clusters=n_clusters, n_init=5)
            kmeans.fit(sample_pixels, sample_weight=sample_weights)
            
            centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            chromatic_counts = []
            for i in range(n_clusters):
                chromatic_counts.append(np.sum(sample_weights[labels == i]))
            
            sorted_idx = np.argsort(-np.array(chromatic_counts))[:n_chromatic]
            
            # Combine everything
            for i in sorted_idx:
                contrast_colors.append(tuple(centers[i]))
                contrast_counts.append(int(chromatic_counts[i]))

        # Final Sort
        final_pairs = sorted(zip(contrast_colors, contrast_counts), key=lambda x: x[1], reverse=True)
        final_colors = [p[0] for p in final_pairs[:n_colors]]
        final_counts = [p[1] for p in final_pairs[:n_colors]]

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
        if v > 0.90 and s < 0.08:
            return "white"

        # Off-white / cream white
        if v > 0.85 and s < 0.15:
            return "off-white"

        # Light gray (includes light heather grey range)
        if v > 0.65 and s < 0.15:
            return "light gray"

        # Heather grey: slight desaturated hue typical of fleece/cotton blends
        # (v 0.48-0.65, very low saturation — not quite "gray", not quite "light gray")
        if 0.48 <= v <= 0.65 and s < 0.15:
            return "heather grey"

        # BLACK vs DARK BLUE (CRITICAL FIX!)
        if v < 0.15:
            # Below v=0.10 the hue is imperceptible — always black regardless of saturation.
            # Above that, require meaningful saturation (>0.25) to call something "midnight blue".
            if v >= 0.10 and s > 0.25 and 200 < h < 260:
                return "midnight blue"  # Dark blue, not black
            else:
                return "black"

        # CHARCOAL vs NAVY (for medium-dark colors)
        if v < 0.25:
            # Check hue for blue tones (navy starts at 205 to avoid teal-green overlap)
            if s > 0.20 and 205 < h < 260:
                return "navy"
            elif s < 0.10:
                return "charcoal"
            else:
                if h < 15 or h >= 330:
                    return "dark red"
                elif h < 60:  # h 15-60°: orange-brown range (chocolate, dark brown)
                    return "brown"
                elif h < 200:
                    return "dark green"
                elif h < 265:
                    return "midnight blue"
                else:
                    return "dark purple"
    
        # Pure grey scale (very low saturation — reaches here only when v < 0.48,
        # since light gray catches v > 0.65 and heather grey catches v 0.48–0.65)
        if s < 0.10:
            if v < 0.30:
                return "charcoal"
            elif v < 0.42:
                return "dark gray"
            else:
                return "gray"
    
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
                elif h < 165:
                    return "slate blue"
                elif h < 210:
                    return "teal"
                elif h < 260:
                    return "slate blue"  # dark desaturated blue — was wrongly "mauve"
                else:
                    return "mauve"
    
        # === DARK COLORS (low value, high saturation) ===
        if v < 0.30:
            if h < 15 or h >= 330:
                return "dark red"
            elif h < 60:  # h 15-60°: dark orange-brown (chocolate, dark brown, rust)
                return "brown"
            elif h < 200:  # extended from 150 — captures dark teal-green (h≈150-200)
                return "dark green"
            elif h < 265:
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
                return "forest green"  
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
            if s < 0.30:  # Low saturation = brown, not coral
                if v > 0.40:
                    return "brown"
                else:
                    return "dark brown"
            elif s > 0.50 and v > 0.60:
                return "coral"
            else:
                return "red-orange"
        elif h < 40:
            # Check if it's brown/tan first (low saturation)
            if s < 0.35 and v > 0.35:
                if v > 0.55:
                    return "tan"  # Light brown
                elif v > 0.35:
                    return "brown"
                else:
                    return "dark brown"
            
            elif s > 0.40:
                if v > 0.65:
                    return "rust"  # Bright rust/copper
                elif v > 0.45:
                    return "terracotta"  # Medium rust
                else:
                    return "burnt orange"  # Dark rust
            # Then check for orange variants
            elif s > 0.70 and v > 0.65:
                return "bright orange"
            elif s > 0.40 and v > 0.70:
                return "peach"  # True peach - high value, medium sat
            elif s > 0.40 and v > 0.40 and v < 0.65:
                return "camel"  # NEW: Camel/wheat color!
            else:
                return "orange"
    
        # YELLOW FAMILY
        elif h < 50:
            if s > 0.50 and v > 0.40 and v < 0.70:
                return "mustard"  # Deep yellow-brown (like your jacket!)
            else:
                return "orange-yellow"
        elif h < 65:
            if s < 0.40 and v > 0.80:
                return "cream"
            elif s > 0.50 and v > 0.40 and v < 0.65:
                return "mustard"  # Second chance for mustard
            elif v > 0.85:
                return "bright yellow"
            elif v > 0.70:
                return "yellow"
            else:
                return "ochre"  # Dark yellow-brown
        elif h < 80:
            if s > 0.30 and v > 0.35 and v < 0.60:
                return "olive"
            else:
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
            use_mask=(mask is not None),
            mask=mask,
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
        if palette["colors"]:
            palette["primary_color"] = palette["colors"][0]["name"]

        return palette

    @staticmethod
    def semantic_primary_color(color_palette: Dict, pattern: str = None) -> str:
        """
        Pick the searchable identity color.

        Pixel majority is correct for solid garments, but striped garments often have
        a light canvas plus darker defining stripes. For search/filtering, promote the
        first non-light-neutral stripe color; if all stripes are neutral, use the
        darkest color rather than the light base.
        """
        colors = color_palette.get("colors", []) if color_palette else []
        if not colors:
            return color_palette.get("primary_color") if color_palette else None

        raw_primary = color_palette.get("primary_color") or colors[0].get("name")
        if not pattern or "stripe" not in pattern.lower():
            return raw_primary

        light_neutrals = {
            "white", "off-white", "cream", "light gray", "light grey",
            "gray", "grey", "heather grey", "heather gray", "silver",
        }
        for color in colors:
            name = color.get("name")
            if name and name not in light_neutrals:
                return name

        def luminance(color_info: Dict) -> float:
            rgb = color_info.get("rgb", [255, 255, 255])
            return (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2])

        return min(colors, key=luminance).get("name", raw_primary)
