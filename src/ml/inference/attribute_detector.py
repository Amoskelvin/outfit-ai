import re
import numpy as np
import cv2
from typing import Dict, List
import easyocr
from pathlib import Path

class AdvancedAttributeDetector:
    """
    Extracts detailed attributes beyond basic classification
    """
    
    def __init__(self):
        # Initialize OCR reader (lazy loading)
        self.ocr_reader = None
        # Initialize CLIP model/processor (lazy loading)
        self.clip_model = None
        self.clip_processor = None
    
    def _get_ocr_reader(self):
        """Lazy load OCR to avoid startup delay"""
        if self.ocr_reader is None:
            print("🔍 Initializing OCR reader (first time only)...")
            # Try cached-only first; fall back to download on first-ever run
            try:
                self.ocr_reader = easyocr.Reader(
                    ['en'], gpu=False, verbose=False, download_enabled=False
                )
            except Exception:
                print("📥 EasyOCR models not cached — downloading (one-time only)...")
                self.ocr_reader = easyocr.Reader(
                    ['en'], gpu=False, verbose=False, download_enabled=True
                )
        return self.ocr_reader
    
    def _get_clip_model(self):
        """Lazy load CLIP for logo recognition"""
        if self.clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print("🔍 Loading CLIP for attribute detection...")
            _clip_id = "openai/clip-vit-base-patch32"
            try:
                self.clip_model = CLIPModel.from_pretrained(_clip_id, local_files_only=True)
                self.clip_processor = CLIPProcessor.from_pretrained(_clip_id, local_files_only=True)
            except Exception:
                print("📥 CLIP not cached — downloading (one-time only)...")
                self.clip_model = CLIPModel.from_pretrained(_clip_id)
                self.clip_processor = CLIPProcessor.from_pretrained(_clip_id)
        return self.clip_model, self.clip_processor
    
    def detect_all_attributes(self, image: np.ndarray, category: str, 
                             subcategory: str, pattern: str,
                             structure_hints: Dict = None) -> Dict:
        """
        Main method to detect all advanced attributes
        
        Args:
            image: RGB numpy array
            category: Basic category (top, bottom, etc.)
            subcategory: Subcategory (t-shirt, jacket, etc.)
            pattern: Detected pattern
        
        Returns:
            Dictionary of detected attributes
        """
        attributes = {}
        
        # Sleeve detection (for tops/outerwear/dresses)
        if category in ["top", "outerwear", "dress"]:
            attributes["sleeve_length"] = self._detect_sleeve_length(image, subcategory)
        
        # Neckline detection (for tops, dresses, and outerwear with visible collars)
        if category in ["top", "dress", "outerwear"]:
            attributes["neckline"] = self._detect_neckline(image, subcategory, structure_hints)
        
        # Logo/text detection
        logo_info = self._detect_logos_text(image, subcategory)
        if logo_info["has_logo"]:
            attributes["has_logo"] = True
            attributes["detected_text"] = logo_info["text"]
            attributes["logo_count"] = logo_info["count"]
        else:
            attributes["has_logo"] = False
        
        # Texture detection (embossed, quilted, etc.)
        # Keyword shortcut: polo and sweater subcategories are always knit fabric.
        # Skips CV/CLIP analysis which confuses cable-knit raised texture with embroidery.
        subcat_lower = subcategory.lower()
        _knit_types = ["sweater", "knit", "turtleneck", "cardigan", "ribbed turtleneck", "oversized knit"]
        _fleece_types = ["hoodie", "sweatshirt", "crewneck sweatshirt", "half-zip sweatshirt",
                         "zip hoodie", "graphic hoodie", "cropped hoodie", "oversized hoodie",
                         "sleeveless hoodie"]
        if any(k in subcat_lower for k in _knit_types):
            attributes["texture"] = "knit"
        elif any(k in subcat_lower for k in _fleece_types):
            attributes["texture"] = "fleece"
        elif any(k in subcat_lower for k in ["denim shirt", "denim jacket"]):
            attributes["texture"] = "denim"
        elif "denim" in subcat_lower or "jean" in subcat_lower:
            attributes["texture"] = "denim"
        elif "flannel" in subcat_lower:
            attributes["texture"] = "flannel"
        elif "linen" in subcat_lower:
            attributes["texture"] = "linen"
        elif "ribbed" in subcat_lower:
            attributes["texture"] = "ribbed"
        elif "fleece" in subcat_lower:
            attributes["texture"] = "fleece"
        else:
            attributes["texture"] = self._detect_texture(image, pattern)

        # Denim wash type — detected immediately after texture so it's available
        # for pattern override in the upload route (e.g. "faded wash", "acid wash")
        if attributes.get("texture") == "denim":
            attributes["denim_wash"] = self._detect_denim_wash(image)

        # Closure type (zipper, buttons, etc.)
        if category in ["top", "outerwear", "bottom"]:
            attributes["closure_type"] = self._detect_closure(image, subcategory)

        # Fit type (for pants/jeans and structured tops)
        if category == "bottom":
            attributes["fit"] = self._detect_fit(subcategory, image)
        elif category == "top":
            subcat_lower = subcategory.lower()
            if "wrap" in subcat_lower or "halter" in subcat_lower:
                attributes["fit"] = "wrap"
            elif "crop" in subcat_lower:
                attributes["fit"] = "crop"

        # Length (for bottoms and outerwear)
        if category in ["bottom", "outerwear"]:
            attributes["length"] = self._detect_length(image, category, subcategory)

        return attributes
    
    def _detect_sleeve_length(self, image: np.ndarray, subcategory: str) -> str:
        """
        Enhanced sleeve detection using multiple methods
        """
        # METHOD 1: Keyword hints
        subcat_lower = subcategory.lower()

        # Draped/wide sleeves — traditional robes (must check before generic "long")
        if any(k in subcat_lower for k in ["agbada", "kaftan", "boubou", "thobe", "dashiki", "buba"]):
            return "wide drape"

        # Sleeveless: tanks, muscle tanks, racerback, tube tops, sleeveless polo/hoodie/knit
        if any(k in subcat_lower for k in [
            "tank", "sleeveless", "tube top", "racerback", "spaghetti strap",
            "muscle tank", "knit vest",
        ]):
            return "sleeveless"

        # Explicitly long sleeve keywords
        if "long sleeve" in subcat_lower or "turtleneck" in subcat_lower:
            return "long sleeve"

        # Short sleeve — polo collars and cuban collars default to short
        if any(k in subcat_lower for k in ["milkmaid", "jersey", "cuban collar", "camp collar", "vintage shirt"]):
            return "short sleeve"

        if any(k in subcat_lower for k in [
            "oxford shirt", "button-up", "button-down", "flannel",
            "denim shirt", "linen shirt", "sheer shirt", "graphic print shirt",
        ]) or subcat_lower == "shirt":
            return "long sleeve"

        # Cardigans are always long-sleeved — gathered or bunched cuffs look shorter
        # to the pixel detector and trigger a false "3/4 sleeve" return.
        if "cardigan" in subcat_lower:
            return "long sleeve"

        # T-shirts always short sleeve (avoids pixel analysis false "sleeveless" on white tees)
        if "t-shirt" in subcat_lower or "tee" in subcat_lower:
            return "short sleeve"

        if "short" in subcat_lower:
            return "short sleeve"
    
        # METHOD 2: Use CLIP for visual classification
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
        
            pil_image = PILImage.fromarray(image.astype('uint8'))
        
            sleeve_prompts = [
                "a garment with long sleeves reaching the wrists",
                "a garment with short sleeves",
                "a sleeveless garment or tank top",
                "a garment with three-quarter sleeves"
            ]
        
            inputs = processor(
                text=sleeve_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
        
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        
            sleeve_map = {
                0: "long sleeve",
                1: "short sleeve",
                2: "sleeveless",
                3: "3/4 sleeve"
            }       
        
            top_idx = np.argmax(probs)
            confidence = probs[top_idx]
        
            detected = sleeve_map[top_idx]
            print(f"👕 CLIP sleeve detection: {detected} (confidence: {confidence:.2f})")
        
            if confidence > 0.35:
                return detected
    
        except Exception as e:
            print(f"⚠️ CLIP sleeve detection error: {e}")
    
        # METHOD 3: Visual analysis (geometric approach)
        h, w = image.shape[:2]
    
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
        # Analyze left and right edges (where sleeves typically are)
        left_edge = gray[:, :int(w*0.2)]
        right_edge = gray[:, int(w*0.8):]
    
        # Find where the garment extends vertically on edges
        # Long sleeves = content extending to bottom 70% of image
        # Short sleeves = content only in top 40%
    
        left_edge_binary = left_edge < 240  # Non-white pixels
        right_edge_binary = right_edge < 240
    
        # Check how far down the edges have content
        left_vertical_extent = []
        right_vertical_extent = []
    
        for row_idx in range(h):
            if np.any(left_edge_binary[row_idx, :]):
                left_vertical_extent.append(row_idx)
            if np.any(right_edge_binary[row_idx, :]):
                right_vertical_extent.append(row_idx)
    
        if len(left_vertical_extent) == 0 and len(right_vertical_extent) == 0:
            return "sleeveless"
    
        # Calculate how far down sleeves extend
        max_left = max(left_vertical_extent) if left_vertical_extent else 0
        max_right = max(right_vertical_extent) if right_vertical_extent else 0
        max_extent = max(max_left, max_right) / h
    
        # Classification based on vertical extent
        if max_extent > 0.65:  # Extends past 65% of height
            return "long sleeve"
        elif max_extent > 0.35:  # Extends to 35-65% of height
            return "short sleeve"
        else:
            return "sleeveless"
    
    def _detect_neckline(self, image: np.ndarray, subcategory: str, structure_hints: Dict = None) -> str:
        """
        Combined Neckline Detection Pipeline:
        1. Subcategory Keywords (Fastest)
        2. Geometric Analysis (High/Wrap Necks -> V-Necks -> Collars)
        3. CLIP AI Vision (Deep Learning Fallback)
        """
        if structure_hints is None:
            structure_hints = {}

        subcat_lower = subcategory.lower()

        # --- STEP 1: SUBCATEGORY KEYWORD CHECK (highest priority) ---
        # Subcategory is an authoritative label — it must override any geometric hint
        # to prevent structural false positives (e.g. thin halter straps → "high neck").
        if "halter" in subcat_lower: return "halter"
        if "milkmaid" in subcat_lower: return "square neck"
        if "polo" in subcat_lower: return "polo collar"
        if "turtle" in subcat_lower or "mock" in subcat_lower: return "turtleneck"
        if "wrap" in subcat_lower or "surplice" in subcat_lower: return "wrap/v-neck"
        if "henley" in subcat_lower: return "henley"
        if "scoop" in subcat_lower: return "scoop neck"
        if "cuban collar" in subcat_lower: return "cuban/camp collar"
        if "off-shoulder" in subcat_lower: return "off-shoulder"
        if "one-shoulder" in subcat_lower: return "one-shoulder"
        if "tube top" in subcat_lower: return "strapless"
        if "corset" in subcat_lower: return "sweetheart/square neck"
        if "bodysuit" in subcat_lower and "v-neck" in subcat_lower: return "v-neck"
        if "cardigan" in subcat_lower:
            # Button-front cardigans: the "V" is the open front, not a knit neckline shape.
            # Returning "v-neck" here violates the Button/Collar Rule — neckline is null.
            if "button" in subcat_lower:
                return None
            return "v-neck"  # open-front draped cardigans
        # V-neck check before generic "tee" / "t-shirt" so v-neck tee gets the right value
        if "v-neck" in subcat_lower: return "v-neck"
        if any(k in subcat_lower for k in [
            "oxford shirt", "button-up", "button-down", "flannel",
            "linen shirt", "denim shirt", "sheer shirt", "graphic print shirt",
            "vintage shirt", "oversized shirt",
        ]) or subcat_lower == "shirt":
            return "collar"
        # T-shirts and tees have crew necks by definition
        if "t-shirt" in subcat_lower or "tee" in subcat_lower: return "crew neck"
        if "crewneck" in subcat_lower: return "crew neck"
        # Hoodies: the defining neckline IS the hood, not a crew opening
        if "hoodie" in subcat_lower: return "hood"
        # Half-zip: high ribbed collar with a short zipper — must be checked before
        # the generic "sweatshirt" catch-all below, which would incorrectly return "crew neck"
        if "half-zip" in subcat_lower: return "mock neck"
        # Plain sweatshirts (no hood) use a crew neck
        if "sweatshirt" in subcat_lower: return "crew neck"

        # --- STEP 2: STRUCTURE HINT CHECK ---
        hint = structure_hints.get("neckline_hint", "").lower()
        if "v" in hint: return "v-neck"
        if "boat" in hint or "off-shoulder" in hint: return "boat neck"

        # If hint says high neck, verify skin exposure before trusting it
        if "collar" in hint or "turtleneck" in hint or "high" in hint:
            skin_regions = structure_hints.get("skin_regions", [])
            if not any(r in skin_regions for r in ["chest"]):
                return "high neck"

        # --- STEP 2: GEOMETRIC ANALYSIS (OpenCV) ---
        try:
            h, w = image.shape[:2]
            # Analyze top 25% of image (neck area)
            neck_area = image[:int(h*0.25), :]
            if neck_area.size == 0: return "unknown"
            
            gray_neck = cv2.cvtColor(neck_area, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_neck, 50, 150)

            # A. Check for High/Funnel Necks (Like the Brown Vest)
            # Calculate fabric coverage in the very top strip (top 5-10% of image)
            top_strip_h = int(neck_area.shape[0] * 0.3)
            top_strip = gray_neck[:top_strip_h, :]
            # Assuming light background (high values) -> count dark pixels (fabric)
            # Adjust threshold (e.g., < 240) based on your background removal consistency
            coverage = np.sum(top_strip < 240) / top_strip.size
            
            if coverage > 0.60:  # High fabric coverage at the throat
                # Check for asymmetric wrap lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)
                if lines is not None:
                    diagonal_count = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                        # Check for diagonal wrap lines (20-70 degrees)
                        if 20 < angle < 70:
                            diagonal_count += 1
                    
                    if diagonal_count >= 2:
                        return "wrap high neck"
                
                return "high neck/funnel"

            # B. Check for V-Neck (Center Diagonals)
            # Look at the center column of the neck area
            center_w_start = max(0, w//2 - 50)
            center_w_end = min(w, w//2 + 50)
            center_edges = edges[:, center_w_start:center_w_end]
            
            lines = cv2.HoughLinesP(center_edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    angles.append(angle)
                
                # V-neck diagonals usually 30-60 or 120-150 degrees
                v_diagonals = sum(1 for a in angles if 30 < a < 60 or 120 < a < 150)
                if v_diagonals >= 2:
                    return "v-neck"

            # C. Check for Standard Collars (Horizontal Lines)
            # Strong horizontal edges near the top often indicate a fold-over collar
            top_edge_strip = gray_neck[:20, :]
            horizontal_edges = np.sum(cv2.Canny(top_edge_strip, 50, 150), axis=1)
            if np.max(horizontal_edges) > w * 0.3:
                return "collar"

        except Exception as e:
            print(f"⚠️ Visual geometry check failed: {e}")

        # --- STEP 3: CLIP AI CLASSIFICATION (Fallback) ---
        try:
            from PIL import Image as PILImage
            # Use self._get_clip_model() assuming it exists in your class
            model, processor = self._get_clip_model()
            
            # Crop a slightly larger area for CLIP context (35%)
            neck_region = image[:int(h*0.35), :]
            pil_image = PILImage.fromarray(neck_region.astype('uint8'))
            
            neckline_prompts = [
                "a shirt with a crew neck",
                "a shirt with a v-neck",
                "a polo shirt with a collar",
                "a dress shirt with a button collar",
                "a sweater with a turtleneck",
                "a top with a scoop neck",
                "a dress with a square neckline",
                "a shirt with a henley placket",
                "a top with a high funnel neck",
                "a top with an asymmetric wrap neck",
                "a jacket with a large pointed shirt-style collar",
                "a traditional robe with a vertical slit opening at the neckline"
            ]

            inputs = processor(text=neckline_prompts, images=pil_image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            top_idx = np.argmax(probs)
            confidence = probs[top_idx]

            neckline_map = {
                0: "crew neck",
                1: "v-neck",
                2: "polo collar",
                3: "collar",
                4: "turtleneck",
                5: "scoop neck",
                6: "square neck",
                7: "henley",
                8: "high neck/funnel",
                9: "wrap high neck",
                10: "pointed collar",
                11: "slit neck"
            }
            
            detected = neckline_map[top_idx]
            print(f"👔 CLIP neckline: {detected} (confidence: {confidence:.2f})")
            
            if confidence > 0.30: # Slightly lower threshold to catch tricky items
                return detected

        except Exception as e:
            print(f"⚠️ CLIP neckline detection error: {e}")

        # Default fallback
        return "crew neck"
    
    def _detect_logos_text(self, image: np.ndarray, subcategory: str = "") -> Dict:
        """
        Multi-method logo detection with watermark filtering
        """
        all_detections = []
    
        # === METHOD 1: EasyOCR ===
        try:
            reader = self._get_ocr_reader()
        
            results = reader.readtext(
                image, 
                detail=1,
                paragraph=False,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,
                link_threshold=0.3
            )
        
            h, w = image.shape[:2]
        
            for (bbox, text, confidence) in results:
                if confidence < 0.25:
                    continue
                # Sanitize OCR noise from cursive/embroidered font edges before filtering.
                # Cursive tails are often read as ~, ., -, _ etc. Strip them and check
                # whether a valid alphanumeric core remains. This keeps ".Casablanca." but
                # kills pure-noise strings like "~~~" that have no real word inside.
                clean_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', text.strip())
                if len(clean_text) < 2:
                    print(f"🚫 Skipping garbled OCR (no valid core): '{text}'")
                    continue
                text = clean_text  # use sanitized version for all downstream checks
                # Get bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x = int(top_left[0])
                y = int(top_left[1])
                text_width = int(top_right[0] - top_left[0])
                text_height = int(bottom_left[1] - top_left[1])

                is_bottom_corner = y > h * 0.7
                is_top_corner = y < h * 0.15
                is_corner = is_bottom_corner or is_top_corner

                is_small = (text_width * text_height) < (h * w * 0.01)

                watermark_keywords = ["dlx", "shutterstock", "getty", "watermark", "alamy", "preview"]
                is_watermark_text = any(kw in text.lower() for kw in watermark_keywords)

                # Compute brand_like BEFORE the corner filter so collar labels (e.g.
                # "Abercrombie & Fitch") at top-center are not killed as watermarks.
                is_brand_like = (
                    (text.isupper() or text.istitle())
                    and 1 <= len(text.split()) <= 4
                    and len(text) >= 3
                )

                # Corner + small = watermark, UNLESS it looks like a brand label
                if ((is_corner and is_small) and not is_brand_like) or is_watermark_text:
                    print(f"🚫 Skipping watermark/corner: '{text}' at ({x}, {y})")
                    continue

                # Inner collar label zone: text in the top-center band is treated as
                # a care/brand label sewn inside the collar, NOT an exterior logo.
                # Two-tier zone for polo shirts:
                #   Shallow zone (y < 22%): wide horizontal window covers all garments.
                #   Deep zone (22% ≤ y < 30%, polo only): narrow center-only window.
                #     Inner labels are always dead-center (back-of-collar seam).
                #     Exterior chest logos (Lacoste, Ralph Lauren, etc.) are offset left
                #     by ~25–35% of image width, so the narrow window lets them through.
                _is_polo = "polo" in subcategory.lower()
                _in_wide_center = abs(x - w / 2) < w * 0.45
                _in_narrow_center = abs(x - w / 2) < w * 0.15
                is_top_center = (
                    (y < h * 0.22 and _in_wide_center) or
                    (_is_polo and h * 0.22 <= y < h * 0.30 and _in_narrow_center)
                )
                if is_top_center:
                    print(f"🚫 Skipping inner collar label: '{text}'")
                    continue

                # Valid logo detection
                all_detections.append({
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "method": "easyocr",
                    "type": "text",
                    "location": {"x": x, "y": y}
                })
                print(f"📝 Logo detected: '{text}' (confidence: {confidence:.2f})")
        
        except Exception as e:
            print(f"⚠️ EasyOCR error: {e}")
    
        # === METHOD 2: Embossed logo detection (same-color raised marks) ===
        # Runs always — catches logos that have no readable text (circular emblems, icons)
        if len(all_detections) == 0:
            embossed = self._detect_embossed_logos(image)
            all_detections.extend(embossed)

        # === METHOD 3: CLIP generic graphic/text presence fallback ===
        # Stylised or horror-font text (e.g. "FULL FRIGHT" in a zombie graphic) is often
        # unreadable by EasyOCR. CLIP can confirm that a visible graphic/logo/text exists
        # even when OCR can't decode the exact characters.
        # Crop the top 22% (collar zone) before running CLIP so that inner collar tags
        # (e.g. "BURTON", "SHEIN") do not trigger a false has_logo=True result.
        if len(all_detections) == 0:
            try:
                from PIL import Image as PILImage
                model, processor = self._get_clip_model()
                _h_m3, _w_m3 = image.shape[:2]
                _m3_img = image[int(_h_m3 * 0.22):, :]
                pil_image = PILImage.fromarray(_m3_img.astype('uint8'))

                presence_prompts = [
                    "a garment with visible text, logo, brand name, or graphic print on the fabric",
                    "a plain garment with no text, logo, or graphic — only solid colour or simple texture"
                ]
                inputs = processor(text=presence_prompts, images=pil_image,
                                   return_tensors="pt", padding=True)
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

                if float(probs[0]) > 0.45:
                    print(f"🏷️ CLIP confirmed logo/graphic presence ({probs[0]:.2f})")
                    all_detections.append({
                        "text": "[graphic/logo detected]",
                        "confidence": float(probs[0]),
                        "method": "clip_presence",
                        "type": "graphic"
                    })

            except Exception as e:
                print(f"⚠️ CLIP logo presence check error: {e}")

        if len(all_detections) > 0:
            return {
                "has_logo": True,
                "text": all_detections,
                "count": len(all_detections)
            }
        else:
            return {"has_logo": False, "text": [], "count": 0}
    
    def _detect_embossed_logos(self, image: np.ndarray) -> List[Dict]:
        """
        Detect embossed/raised logos using texture analysis
        (For same-color logos like embossed "NY")
        """
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # METHOD 2A: Edge detection with multiple scales
            # Embossed logos create subtle shadows/highlights
            edges_subtle = cv2.Canny(gray, 30, 100)  # Catch subtle edges
            edges_strong = cv2.Canny(gray, 100, 200) # Catch strong edges
            
            combined_edges = cv2.addWeighted(edges_subtle, 0.5, edges_strong, 0.5, 0)
            
            # METHOD 2B: Laplacian (detects texture changes)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian).astype(np.uint8)
            
            # Threshold to find textured regions
            _, texture_mask = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY)
            
            # Combine edge and texture information
            combined = cv2.addWeighted(combined_edges, 0.6, texture_mask, 0.4, 0)
            
            # Find contours of potential logos
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = gray.shape
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                aspect_ratio = w_cnt / h_cnt if h_cnt > 0 else 0
                
                # Logo characteristics:
                # - Size: 1-10% of image
                # - Aspect ratio: roughly square (0.5 - 2.0)
                # - Position: typically upper chest area
                
                min_area = (h * w) * 0.005   # 0.5% of image
                max_area = (h * w) * 0.10    # 10% of image
                
                is_logo_sized = min_area < area < max_area
                is_reasonable_shape = 0.3 < aspect_ratio < 3.0
                is_upper_area = y < h * 0.5  # In upper half of image
                
                if is_logo_sized and is_reasonable_shape and is_upper_area:
                    # Found potential embossed logo
                    # Extract the region
                    logo_region = image[y:y+h_cnt, x:x+w_cnt]
                    
                    # Check if it's actually embossed (has depth variation)
                    region_gray = gray[y:y+h_cnt, x:x+w_cnt]
                    depth_variation = np.std(region_gray)
                    
                    if depth_variation > 30:  # Has distinct raised texture (logo) — raised > pocket stitching
                        detections.append({
                            "text": "[Embossed logo/emblem detected]",
                            "confidence": 0.70,
                            "method": "embossed_detection",
                            "type": "embossed",
                            "location": {"x": int(x), "y": int(y), "width": int(w_cnt), "height": int(h_cnt)}
                        })
                        print(f"🔍 Detected embossed logo at ({x}, {y}), size: {w_cnt}x{h_cnt}")
                        break  # Found one, that's usually enough
        
        except Exception as e:
            print(f"⚠️ Embossed detection error: {e}")
        
        return detections
    
    def _detect_logos_with_clip(self, image: np.ndarray) -> List[Dict]:
        """
        Use CLIP to recognize known brand logos
        (For logos we can't read but can recognize visually)
        """
        detections = []
        
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
            
            # Convert numpy to PIL
            pil_image = PILImage.fromarray(image.astype('uint8'))
            
            # List of common brand logos to check for
            brand_prompts = [
                "a photo with Nike logo",
                "a photo with Adidas logo",
                "a photo with Yankees NY logo",
                "a photo with Supreme logo",
                "a photo with Champion logo",
                "a photo with Ralph Lauren polo logo",
                "a photo with Lacoste crocodile logo",
                "a photo with Tommy Hilfiger logo",
                "a photo with Gucci logo",
                "a photo with Louis Vuitton logo",
                "a photo with no visible logo",
            ]
            
            # Process
            inputs = processor(
                text=brand_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Get predictions
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
            
            # Find highest probability brand (excluding "no logo")
            top_idx = np.argmax(probs[:-1])  # Exclude last item ("no logo")
            top_prob = probs[top_idx]
            no_logo_prob = probs[-1]
            
            # Only report if brand logo is more likely than no logo
            if top_prob > no_logo_prob and top_prob > 0.3:
                brand_name = brand_prompts[top_idx].replace("a photo with ", "").replace(" logo", "")
                detections.append({
                    "text": f"[{brand_name}]",
                    "confidence": float(top_prob),
                    "method": "clip_recognition",
                    "type": "brand_logo"
                })
                print(f"🏷️ CLIP detected: {brand_name} (confidence: {top_prob:.2f})")
        
        except Exception as e:
            print(f"⚠️ CLIP logo detection error: {e}")
        
        return detections
    
    def _detect_texture(self, image: np.ndarray, pattern: str) -> str:
        """
        Enhanced texture detection including embroidery
        """
        # METHOD 0: Denim pre-check — must run BEFORE the embroidery check.
        # Acid-wash, whiskered, and faded denim have high Laplacian variance and
        # dense edges that falsely trigger the embroidery detector. CLIP resolves
        # this definitively before any pixel heuristics run.
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
            pil_image = PILImage.fromarray(image.astype('uint8'))

            denim_prompts = [
                "blue or indigo denim jeans — characteristic diagonal twill weave clearly visible",
                "acid-washed or bleached blue denim — cloudy tonal variation on a blue base fabric",
                "dark indigo or very dark blue raw denim — the blue cast is still visible in the fabric",
                "non-denim fabric — black jersey, cotton t-shirt, or any solid non-blue material"
            ]

            inputs = processor(text=denim_prompts, images=pil_image,
                               return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            # Use max() not sum(): summing 3 denim prompts vs 1 non-denim prompt creates a
            # 3:1 structural bias — a confused CLIP distributes ~uniform probability
            # (0.25 each), yielding denim_sum=0.75 vs non_denim=0.25, so ANY unclear image
            # returns "denim" by default. max() is prompt-count-agnostic.
            best_denim_score = float(max(probs[0], probs[1], probs[2]))
            non_denim_score  = float(probs[3])

            if best_denim_score > non_denim_score and best_denim_score > 0.40:
                print(f"🧵 Denim confirmed by CLIP pre-check (best_denim={best_denim_score:.2f})")
                return "denim"

        except Exception as e:
            print(f"⚠️ Denim pre-check error: {e}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # METHOD 1: Look for localized high-detail regions (embroidery)
        # Embroidery typically appears in specific regions (chest, sleeves)
        # Guard: a flat printed graphic (screen print, DTG) also produces high Laplacian
        # variance in the chest area but is NOT embroidery. Differentiate by checking
        # whether the chest region is MULTI-COLORED (print) vs MONO-THREAD (embroidery):
        # embroidery threads are narrow bands of a single color; prints have smooth
        # color gradients and multiple distinct hues across the region.

        chest_area = gray[int(h*0.15):int(h*0.5), int(w*0.3):int(w*0.7)]
        chest_color = image[int(h*0.15):int(h*0.5), int(w*0.3):int(w*0.7)]

        if chest_area.size > 0:
            laplacian = cv2.Laplacian(chest_area, cv2.CV_64F)
            embroidery_score = np.var(laplacian)
            edges = cv2.Canny(chest_area, 50, 150)
            edge_density = np.sum(edges > 0) / chest_area.size

            if embroidery_score > 300 and edge_density > 0.08:
                whole_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))

                if embroidery_score > whole_variance * 1.5:
                    # Disambiguate embroidery from graphic/screen print:
                    # A printed graphic has very high hue variance in the chest region
                    # (smooth color transitions); embroidery has low hue variance
                    # (solid-colored thread rows with sharp boundaries).
                    if chest_color.size > 0:
                        chest_hsv = cv2.cvtColor(chest_color, cv2.COLOR_RGB2HSV)
                        # Ignore near-white/background pixels (low saturation)
                        sat_mask = chest_hsv[:, :, 1] > 30
                        if np.sum(sat_mask) > 50:
                            hue_values = chest_hsv[:, :, 0][sat_mask]
                            hue_std = np.std(hue_values.astype(float))
                            if hue_std > 25:
                                # High hue variance → smooth printed graphic, not embroidery thread
                                print(f"🖨️ Graphic print detected in chest (hue σ={hue_std:.0f}) — skipping embroidery")
                            else:
                                print(f"🧵 Detected embroidery (variance: {embroidery_score:.0f}, edges: {edge_density:.2%})")
                                return "embroidered"
                    else:
                        print(f"🧵 Detected embroidery (variance: {embroidery_score:.0f}, edges: {edge_density:.2%})")
                        return "embroidered"
    
        # METHOD 2: Use CLIP for texture classification
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
        
            pil_image = PILImage.fromarray(image.astype('uint8'))
        
            texture_prompts = [
                "a garment made of denim",
                "a garment with heavy embroidery or stitched patterns",
                "a garment with quilted texture",
                "a garment with smooth or silky fabric",
                "a garment with knit or ribbed texture",
                "a garment with embossed 3D pattern",
                "a plain cotton t-shirt or jersey with a flat screen-printed or graphic design",
                "a garment made of brocade, Aso-Oke, or stiff woven fabric",
                "a garment made of leather or faux-leather"
            ]

            inputs = processor(text=texture_prompts, images=pil_image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            texture_map = {
                0: "denim",
                1: "embroidered",
                2: "quilted",
                3: "smooth",
                4: "knit/ribbed",
                5: "embossed/textured",
                6: "cotton",
                7: "brocade/woven",
                8: "leather"
            }
        
            top_idx = np.argmax(probs)
            confidence = probs[top_idx]
        
            if confidence > 0.35:
                detected = texture_map[top_idx]
                print(f"🔍 CLIP texture: {detected} (confidence: {confidence:.2f})")
                return detected
    
        except Exception as e:
            print(f"⚠️ CLIP texture detection error: {e}")
    
        # METHOD 3: Fallback texture analysis
        texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / gray.size
    
        if texture_variance > 500 and edge_density > 0.15:
            # Check for circular patterns (embossed bubbles)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
        
            if circles is not None and len(circles[0]) > 3:
                return "embossed/textured"
            else:
                return "quilted"
    
        elif texture_variance > 200:
            if "knit" in pattern or "ribbed" in pattern:
                return "knit/ribbed"
            else:
                return "textured"
    
        return "smooth"
    
    def _detect_closure(self, image: np.ndarray, subcategory: str) -> str:
        """
        Detect closure type with stricter validation to prevent hallucinations
        """
    
        # RULE 1: Check subcategory keywords FIRST (most reliable)
        subcat_lower = subcategory.lower()

        # Halter tops — tied behind the neck
        if "halter" in subcat_lower:
            return "tie"
        if "wrap" in subcat_lower:
            return "wrap/tie"

        # Pull-on (traditional robes — no front closure, worn over the head)
        if any(k in subcat_lower for k in ["agbada", "kaftan", "boubou", "thobe", "dashiki", "kurta", "buba", "ankara"]):
            return "pull-on"

        # Open-front / no closure (tube tops, off-shoulder, racerback, one-shoulder)
        if any(k in subcat_lower for k in ["tube top", "off-shoulder", "one-shoulder", "corset", "bodysuit"]):
            return "pull-on"

        # Garments with no closure hardware: return None (not applicable).
        # This covers both the tee family (where HoughCircles misreads graphics as buttons)
        # and pullover families (sweatshirts, tanks, sweaters/turtlenecks) where "pullover"
        # is technically the closure mechanism but the field is not meaningful.
        _null_closure_subcats = {
            # Tee family
            "t-shirt", "graphic print tee", "striped t-shirt", "vintage tee",
            "oversized tee", "cropped tee", "longline tee", "pocket tee",
            "v-neck tee", "tie-dye tee",
            # Tank family
            "tank top", "ribbed tank", "graphic tank", "muscle tank",
            "longline tank", "spaghetti strap top", "racerback tank",
            # Sweater / turtleneck family (no front closure)
            "turtleneck", "Turtleneck", "ribbed turtleneck", "crewneck sweater",
            "cable-knit sweater", "chunky knit", "oversized knit",
        }
        if subcat_lower in {s.lower() for s in _null_closure_subcats}:
            return None

        # Sweatshirts: pullover over-the-head, no hardware.
        if "sweatshirt" in subcat_lower:
            return None

        # Turtleneck keyword catch-all for any variant not in the set above.
        if "turtleneck" in subcat_lower:
            return None

        # Drawstring (hoodies, sweatpants, joggers — plain pullover hoodie only)
        _drawstring_types = ["sweatpants", "jogger", "track pants", "hoodie"]
        if any(k in subcat_lower for k in _drawstring_types) and "zip hoodie" not in subcat_lower:
            return "drawstring"

        # Full zipper — must be checked before the "sweatshirt" catch-all.
        if any(k in subcat_lower for k in ["zip hoodie", "bomber", "windbreaker", "coach jacket",
                                            "fleece jacket", "varsity"]):
            return "zipper"

        # Half-zip — must be checked before generic keyword fallbacks.
        if "half-zip" in subcat_lower:
            return "half-zip"

        # Explicit button indicators (shirts, button-ups, denim jacket, oxford, etc.)
        if any(k in subcat_lower for k in [
            "button-up", "button-down", "oxford shirt", "flannel", "linen shirt",
            "denim shirt", "cuban collar", "vintage shirt", "sheer shirt", "graphic print shirt",
            "denim jacket", "buttoned cardigan", "waistcoat", "blazer",
        ]):
            return "buttons"

        # Buttoned cardigan: "Cardigan (buttoned)" — button keyword + cardigan keyword.
        # Must come before the open-front catch below.
        if "cardigan" in subcat_lower and "button" in subcat_lower:
            return "buttons"

        # Open-front cardigan (no buttons, draped open)
        if subcat_lower == "cardigan":
            return "open front"

        # Pullovers — must be checked BEFORE HoughCircles because circular graphics
        # (logos, eye illustrations, badges) are detected as buttons by HoughCircles
        # if we let it run first. A t-shirt with a circular chest graphic would wrongly
        # return "buttons" unless we short-circuit here.
        # Polo shirts and sweaters: split by subcategory because their closure defaults differ.
        if "polo" in subcat_lower or "sweater" in subcat_lower:
            is_plain_polo = "polo" in subcat_lower and "sweater" not in subcat_lower
            try:
                from PIL import Image as PILImage
                model, processor = self._get_clip_model()
                hi, wi = image.shape[:2]
                collar_crop = image[:int(hi * 0.30), int(wi * 0.40):int(wi * 0.60)]
                if collar_crop.size == 0:
                    collar_crop = image

                # Contrast collar pre-check: if top/bottom of strip have very different
                # mean colors, it's a two-tone collar band, not a zipper track.
                is_contrast_collar = False
                if collar_crop.shape[0] > 6:
                    hi_crop = collar_crop.shape[0]
                    top_band = collar_crop[:int(hi_crop * 0.33), :]
                    bot_band = collar_crop[int(hi_crop * 0.67):, :]
                    if top_band.size > 0 and bot_band.size > 0:
                        top_mean = np.mean(top_band.reshape(-1, 3), axis=0)
                        bot_mean = np.mean(bot_band.reshape(-1, 3), axis=0)
                        color_diff = float(np.sqrt(np.sum((top_mean.astype(float) - bot_mean.astype(float)) ** 2)))
                        print(f"🔍 Collar strip top-vs-bot color_diff: {color_diff:.1f}")
                        is_contrast_collar = color_diff > 60

                if is_plain_polo:
                    # A polo shirt is DEFINED by its button placket — CLIP cannot reliably
                    # detect same-color buttons (e.g. white snaps on white fabric).
                    # Default to "buttons"; only override to "half-zip" if CLIP strongly
                    # sees a zipper AND the collar strip is not a contrast-color band.
                    if not is_contrast_collar:
                        pil = PILImage.fromarray(collar_crop.astype("uint8"))
                        zip_prompts = [
                            "a short metallic zipper track running down the center of a polo collar",
                            "a polo shirt placket with buttons or no zipper",
                            "a polo collar with a contrast color trim — no zipper",
                        ]
                        inputs = processor(text=zip_prompts, images=pil,
                                           return_tensors="pt", padding=True)
                        outputs = model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
                        print(f"🤐 CLIP polo zip check: zip={probs[0]:.2f} btn={probs[1]:.2f}")
                        if probs[0] > 0.55:
                            return "half-zip"
                    return "buttons"

                else:
                    # Sweater polo / sweater: can be pullover or half-zip; no button default.
                    if not is_contrast_collar:
                        pil = PILImage.fromarray(collar_crop.astype("uint8"))
                        zip_prompts = [
                            "a short metallic zipper track running down the center of a polo collar",
                            "an open V-gap at the center of a polo collar with no zipper",
                            "a polo collar with a contrast color trim or band — no zipper present",
                        ]
                        inputs = processor(text=zip_prompts, images=pil,
                                           return_tensors="pt", padding=True)
                        outputs = model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
                        print(f"🤐 CLIP sweater zip: zip={probs[0]:.2f} open={probs[1]:.2f} contrast={probs[2]:.2f}")
                        if probs[0] > 0.55:
                            return "half-zip"
                    return "pullover"

            except Exception as e:
                print(f"⚠️ CLIP polo closure error: {e}")
                # Safe defaults by subcategory
                return "buttons" if is_plain_polo else "pullover"

        pullover_types = ["sweater", "pullover", "t-shirt", "tank", "polo", "crew", "v-neck", "cami", "tube top"]
        if any(kw in subcat_lower for kw in pullover_types):
            if "button" not in subcat_lower:
                return "pullover"

        # Convert to grayscale for visual analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Focus on upper-center and center (where closures typically are)
        closure_region = gray[:int(h*0.6), int(w*0.3):int(w*0.7)]

        # Method 1: Hough Circle Detection (for round buttons)
        circles = cv2.HoughCircles(
            closure_region,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=25,  # Lowered threshold
            minRadius=5,
            maxRadius=35
        )
    
        if circles is not None and len(circles[0]) >= 1:
            # Found button(s)
            print(f"🔘 Detected {len(circles[0])} button(s)")

            if len(circles[0]) == 1:
                return "single button"
            else:
                return "buttons"
    
        # Method 2: Check for metallic/shiny spots (button reflections)
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
    
        # Find very bright spots (metallic reflections)
        bright_spots = l_channel > 200
        bright_region = bright_spots[:int(h*0.6), int(w*0.3):int(w*0.7)]
    
        if np.sum(bright_region) > 50:  # Some bright pixels present
            # Check if they're clustered (button-shaped)
            contours, _ = cv2.findContours(
                bright_region.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        
            button_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]
        
            if len(button_contours) >= 1:
                print(f"✨ Detected {len(button_contours)} metallic button(s)")
                return "buttons" if len(button_contours) > 1 else "single button"

        # Use CLIP to double-check
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
        
            # Focus on center vertical strip
            h, w = image.shape[:2]
            center_strip = image[:, w//2-60:w//2+60]
            pil_image = PILImage.fromarray(center_strip.astype('uint8'))
        
            closure_prompts = [
                "a garment with a zipper",
                "a garment with buttons",
                "a pullover garment with no closure",
                "a garment with a snap closure",
                "pants or a hoodie with a visible drawstring",
                "jeans or shorts with a button fly closure",
                "a robe or tunic worn by pulling over the head with no front opening"
            ]

            inputs = processor(text=closure_prompts, images=pil_image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            closure_map = {0: "zipper", 1: "buttons", 2: "pullover", 3: "snap", 4: "drawstring", 5: "button fly", 6: "pull-on"}
            top_idx = np.argmax(probs)

            if probs[top_idx] > 0.4:
                detected = closure_map[top_idx]
                print(f"🔒 CLIP closure: {detected} (confidence: {probs[top_idx]:.2f})")
                return detected
    
        except Exception as e:
            print(f"⚠️ CLIP closure detection error: {e}")
    
        # Default to pullover (safest - prevents zipper hallucinations)
        return "pullover"

    def _detect_denim_wash(self, image: np.ndarray) -> str:
        """
        Classify the wash/finish of a denim garment.
        Returns a commercially meaningful descriptor (e.g. "faded wash", "acid wash").
        """
        try:
            from PIL import Image as PILImage
            model, processor = self._get_clip_model()
            pil_image = PILImage.fromarray(image.astype('uint8'))

            wash_prompts = [
                "dark raw denim or dark indigo wash with minimal fading",
                "acid-washed denim with irregular bleached or cloudy spots",
                "light stonewashed or heavily bleached denim",
                "faded vintage denim with whiskering and tonal gradient",
                "distressed or ripped denim with visible damage",
                "classic medium blue wash denim"
            ]

            inputs = processor(text=wash_prompts, images=pil_image,
                               return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            wash_map = {
                0: "dark wash",
                1: "acid wash",
                2: "light wash",
                3: "faded wash",
                4: "distressed",
                5: "medium wash"
            }

            top_idx = int(np.argmax(probs))
            if probs[top_idx] > 0.28:
                detected = wash_map[top_idx]
                print(f"👖 Denim wash: {detected} (confidence: {probs[top_idx]:.2f})")
                return detected

        except Exception as e:
            print(f"⚠️ Denim wash detection error: {e}")

        return "standard wash"

    def _detect_fit(self, subcategory: str, image: np.ndarray = None) -> str:
        """
        Detect fit for pants/jeans: skinny, slim, regular, baggy/relaxed, wide leg
        """
        subcat_lower = subcategory.lower()

        # Keyword-based detection (fast path)
        if any(k in subcat_lower for k in ["agbada", "kaftan", "boubou", "thobe"]):
            return "oversized drape"
        elif "skinny" in subcat_lower:
            return "skinny"
        elif "slim" in subcat_lower:
            return "slim fit"
        elif "sweat" in subcat_lower or "jogger" in subcat_lower or "track" in subcat_lower:
            return "baggy/relaxed"
        elif "culotte" in subcat_lower:
            return "wide leg"
        elif "wide" in subcat_lower or "baggy" in subcat_lower or "oversized" in subcat_lower:
            return "wide leg"
        elif "bootcut" in subcat_lower:
            return "bootcut"
        elif "straight" in subcat_lower:
            return "straight"

        # CLIP visual fallback — used when subcategory alone is not specific enough
        if image is not None:
            try:
                from PIL import Image as PILImage
                model, processor = self._get_clip_model()

                pil_image = PILImage.fromarray(image.astype('uint8'))

                fit_prompts = [
                    "very tight skinny pants hugging the legs",
                    "slim straight-cut pants",
                    "regular fit pants",
                    "baggy oversized wide-leg pants with lots of fabric",
                    "bootcut pants slightly flared at the ankle",
                    "wide-leg palazzo pants"
                ]

                inputs = processor(text=fit_prompts, images=pil_image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

                fit_map = {
                    0: "skinny",
                    1: "slim fit",
                    2: "regular fit",
                    3: "baggy/relaxed",
                    4: "bootcut",
                    5: "wide leg"
                }

                top_idx = np.argmax(probs)
                if probs[top_idx] > 0.35:
                    detected = fit_map[top_idx]
                    print(f"👖 CLIP fit detection: {detected} (confidence: {probs[top_idx]:.2f})")
                    return detected

            except Exception as e:
                print(f"⚠️ CLIP fit detection error: {e}")

        return "regular fit"
    
    def _detect_length(self, image: np.ndarray, category: str, subcategory: str) -> str:
        """
        Detect length for bottoms and outerwear
        """
        subcat_lower = subcategory.lower()
        
        if category == "bottom":
            if "short" in subcat_lower:
                return "shorts"
            elif "crop" in subcat_lower or "capri" in subcat_lower:
                return "cropped"
            elif "ankle" in subcat_lower:
                return "ankle length"
            else:
                return "full length"
        
        elif category == "outerwear":
            if "crop" in subcat_lower:
                return "cropped"
            elif "long" in subcat_lower or "maxi" in subcat_lower:
                return "long"
            else:
                return "regular"
        
        return "standard"
