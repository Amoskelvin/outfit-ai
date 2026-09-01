import numpy as np
import cv2
from typing import Dict, Tuple

class GarmentTypeAnalyzer:
    """
    Pre-analysis to detect garment structure before detailed classification
    Prevents CLIP from hallucinating on unusual silhouettes
    """
    
    @staticmethod
    def analyze_structure(image: np.ndarray) -> Dict:
        """
        Analyze basic garment structure to guide classification
        
        Returns:
            Dictionary with structural hints
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        analysis = {
            "has_sleeves": True,
            "shows_skin": False,
            "is_layering_piece": False,
            "is_vest": False,
            "is_bottom_garment": False,
            "structure_type": "standard",
            "vertical_coverage": 0.0,
            "skin_percentage": 0.0,
            "skin_regions": [],
            "confidence_flags": []
        }

        # === BOTTOM GARMENT DETECTION (runs first — prevents vest false positives on pants) ===
        # Pants have a definitive structural signature: bilateral leg columns with a gap
        # between them at the lower portion of the frame. No vest or top has this.
        bottom_result = GarmentTypeAnalyzer._detect_bottom_characteristics(gray)
        if bottom_result["is_likely_bottom"]:
            analysis["is_bottom_garment"] = True
            analysis["structure_type"] = "bottom"
            analysis["confidence_flags"].append("bottom_garment_detected")
            print(f"👖 Bottom garment detected: {bottom_result['reasons']}")
            # Skip all upper-body structural analysis — it will only produce false signals
            # on a lower-body garment (pocket seams ≠ armholes, waistband ≠ neckline)
            coverage = GarmentTypeAnalyzer._analyze_coverage(gray)
            analysis["vertical_coverage"] = coverage.get("vertical_coverage", 0.0)
            return analysis

        # === SKIN DETECTION ===
        skin_detected = GarmentTypeAnalyzer._detect_skin_exposure(image)

        if skin_detected["has_skin"]:
            analysis["shows_skin"] = True
            analysis["skin_percentage"] = skin_detected["percentage"]
            analysis["skin_regions"] = skin_detected["regions"]

            # Sleeveless only when skin is visible at BOTH shoulder AND arm zones.
            # Short-sleeve garments expose the lower arm but still cover the shoulder —
            # requiring both regions prevents misclassifying short-sleeve tees as sleeveless.
            if "arms" in skin_detected["regions"] and "shoulders" in skin_detected["regions"]:
                analysis["has_sleeves"] = False
                analysis["confidence_flags"].append("sleeveless_detected")
                print(f"👕 Skin detected on both shoulders and arms → sleeveless confirmed")

            if "shoulders" in skin_detected["regions"] and "arms" not in skin_detected["regions"]:
                analysis["confidence_flags"].append("exposed_shoulders")

        # === VEST/GILET DETECTION ===
        vest_indicators = GarmentTypeAnalyzer._detect_vest_characteristics(image, gray)

        if vest_indicators["is_likely_vest"]:
            analysis["is_vest"] = True
            analysis["structure_type"] = "vest/gilet"
            analysis["has_sleeves"] = False
            analysis["confidence_flags"].append("vest_detected")
            print(f"🦺 Vest detected: {vest_indicators['reasons']}")

        # === COVERAGE ANALYSIS ===
        coverage = GarmentTypeAnalyzer._analyze_coverage(gray)
        analysis["vertical_coverage"] = coverage.get("vertical_coverage", 0.0)
        
        # Tops typically cover upper 40-60% of frame
        # Outerwear typically covers 60-90%
        # Halter/crop tops cover < 40%
        
        # Extract the value for cleaner code
        v_cov = coverage["vertical_coverage"]
        print(f"📏 Vertical coverage: {v_cov:.1%}")

        # --- 1. SHORT / CROP ZONE (< 0.45) ---
        if v_cov < 0.45:
            # Overlap Check (0.40 - 0.45):
            if v_cov > 0.40 and (analysis.get("is_vest") or not analysis.get("has_sleeves")):
                analysis["structure_type"] = "vest/gilet"
                analysis["is_layering_piece"] = True
                print("   → Short length + sleeveless = likely cropped vest")
            else:
                # Default Crop Logic
                analysis["structure_type"] = "crop_or_halter"
                analysis["confidence_flags"].append("short_garment")
                print("   → Low vertical coverage = likely crop/halter top")

        # --- 2. MEDIUM / VEST ZONE (0.45 - 0.70) ---
        elif 0.45 <= v_cov <= 0.70:
            # This is the "sweet spot" for standard vests
            if analysis.get("is_vest") or not analysis.get("has_sleeves"):
                analysis["structure_type"] = "vest/gilet"
                analysis["is_layering_piece"] = True
                print("   → Medium length + sleeveless = likely vest")

        # --- 3. LONG GARMENT ZONE (> 0.70) ---
        elif v_cov > 0.70:
            # New logic for coats, long tunics, or dresses
            analysis["is_layering_piece"] = True
            analysis["confidence_flags"].append("long_garment")
            print("   → High vertical coverage = likely coat/long garment")
        
        # === STRAP/TIE DETECTION ===
        straps_detected = GarmentTypeAnalyzer._detect_thin_straps(gray)
        
        if straps_detected["has_thin_straps"]:
            analysis["has_sleeves"] = False
            analysis["structure_type"] = "strap_based"
            analysis["confidence_flags"].append("halter_or_strap")
            print(f"🎀 Thin straps detected → likely halter/spaghetti strap")
        
        # === NECKLINE SHAPE ===
        neckline_shape = GarmentTypeAnalyzer._analyze_neckline_shape(gray)
        
        if neckline_shape == "deep_v":
            analysis["neckline_hint"] = "v-neck or wrap"
        elif neckline_shape == "wide":
            analysis["neckline_hint"] = "boat neck or off-shoulder"
        elif neckline_shape == "high":
            analysis["neckline_hint"] = "collar or turtleneck"
        
        return analysis
    
    @staticmethod
    def _detect_bottom_characteristics(gray: np.ndarray) -> Dict:
        """
        Detect if the image shows a bottom garment (pants, shorts, skirt).

        The key anatomical signature of pants is bilateral leg columns: fabric
        present on the left and right sides of the lower frame with a relative
        gap in the centre (the space between the two legs). No vest, jacket, or
        top exhibits this pattern.

        Works across product photography styles:
          - hanging product shots (legs hang down with a natural gap)
          - on-model shots showing the lower body
          - flat-lay shots where legs are slightly separated
        """
        h, w = gray.shape
        reasons = []

        # --- CHECK 1: Bilateral leg columns (the definitive indicator) ---
        # Analyse the bottom 45 % of the frame where leg separation is visible.
        lower = gray[int(h * 0.55):, :]
        left   = lower[:, :int(w * 0.35)]
        center = lower[:, int(w * 0.35):int(w * 0.65)]
        right  = lower[:, int(w * 0.65):]

        left_cov   = np.sum(left   < 240) / left.size   if left.size   > 0 else 0
        center_cov = np.sum(center < 240) / center.size if center.size > 0 else 0
        right_cov  = np.sum(right  < 240) / right.size  if right.size  > 0 else 0

        avg_side_cov = (left_cov + right_cov) / 2

        # Both sides must have fabric AND the centre must be noticeably less covered
        if left_cov > 0.20 and right_cov > 0.20 and center_cov < avg_side_cov * 0.78:
            reasons.append("bilateral_leg_columns")
            print(
                f"👖 Bilateral legs: L={left_cov:.0%}  C={center_cov:.0%}  R={right_cov:.0%}"
                f"  (gap ratio: {center_cov / avg_side_cov:.2f})"
            )

        # --- CHECK 2: Waistband at top (supporting indicator) ---
        # A waistband is a dense horizontal band spanning the full width of the
        # garment near the top of the frame.
        top_band = gray[:int(h * 0.14), :]
        if top_band.size > 0:
            row_coverages = [np.sum(row < 240) / w for row in top_band]
            mean_top_cov = np.mean(row_coverages)
            # Waistband = fabric covers most of the width near the top
            if mean_top_cov > 0.35:
                reasons.append("waistband_at_top")
                print(f"👖 Waistband detected (top coverage: {mean_top_cov:.0%})")

        # --- CHECK 3: Drawstring detection (high-contrast vertical lines at top-center) ---
        # Drawstrings appear as two thin parallel vertical lines hanging from the
        # waistband centre. The high contrast of white drawstrings on grey fabric
        # makes them detectable via edge analysis.
        top_center = gray[:int(h * 0.30), int(w * 0.35):int(w * 0.65)]
        if top_center.size > 0:
            edges = cv2.Canny(top_center, 30, 120)
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                threshold=15, minLineLength=int(h * 0.06), maxLineGap=8
            )
            if lines is not None:
                # Count near-vertical lines (drawstrings hang straight down)
                vertical_lines = [
                    ln for ln in lines
                    if abs(np.arctan2(ln[0][3] - ln[0][1], ln[0][2] - ln[0][0]) * 180 / np.pi) > 70
                ]
                if len(vertical_lines) >= 2:
                    reasons.append("drawstrings_detected")
                    print(f"👖 Drawstrings detected ({len(vertical_lines)} vertical lines)")

        # Bilateral leg columns alone is sufficient to declare this a bottom garment.
        # Waistband and drawstrings are supporting evidence only.
        is_likely_bottom = "bilateral_leg_columns" in reasons

        return {
            "is_likely_bottom": is_likely_bottom,
            "reasons": reasons
        }

    @staticmethod
    def _detect_vest_characteristics(image: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Detect if garment is a vest/gilet based on multiple indicators
        """
        h, w = gray.shape
        indicators = []
    
        # INDICATOR 1: Deep armholes (sleeveless with structured edges)
        # Vests have clean, deep armholes - not just missing sleeves
        left_edge = gray[:, :int(w*0.15)]
        right_edge = gray[:, int(w*0.85):]
    
        # Detect strong vertical edges (structured armhole seams)
        left_edges = cv2.Canny(left_edge, 100, 200)
        right_edges = cv2.Canny(right_edge, 100, 200)
    
        left_vertical_strength = np.sum(left_edges) / left_edges.size
        right_vertical_strength = np.sum(right_edges) / right_edges.size
    
        if left_vertical_strength > 0.05 and right_vertical_strength > 0.05:
            indicators.append("deep_armholes")
    
        # INDICATOR 2: Boxy/structured silhouette
        # Vests don't have waist tapering like dresses
    
        # Compare width at different heights
        upper_third = gray[int(h*0.2):int(h*0.35), :]
        middle_third = gray[int(h*0.45):int(h*0.55), :]
        lower_third = gray[int(h*0.65):int(h*0.80), :]
    
        def get_garment_width(section):
            """Find horizontal extent of non-white pixels"""
            binary = section < 240
            widths = np.sum(binary, axis=1)
            return np.median(widths) if len(widths) > 0 else 0
    
        upper_width = get_garment_width(upper_third)
        middle_width = get_garment_width(middle_third)
        lower_width = get_garment_width(lower_third)
    
        # Vest = relatively uniform width (boxy)
        # Dress = tapers at waist then flares
    
        if upper_width > 0 and middle_width > 0 and lower_width > 0:
            width_variation = (max(upper_width, middle_width, lower_width) - 
                              min(upper_width, middle_width, lower_width)) / upper_width
        
            # Less than 15% width variation = boxy/structured
            if width_variation < 0.15:
                indicators.append("boxy_silhouette")
    
        # INDICATOR 3: Thick/structured fabric texture
        # Vests are typically heavy wool/felt, not flowing fabric

        # Calculate texture variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
    
        # Structured fabrics have moderate texture (not super smooth, not heavily patterned)
        if 100 < texture_variance < 800:
            indicators.append("structured_fabric")
    
        # INDICATOR 4: Visible closure/button in upper area
        # Vests often have front closures
    
        upper_center = gray[:int(h*0.4), int(w*0.35):int(w*0.65)]
    
        # Look for circular shapes (buttons)
        circles = cv2.HoughCircles(
            upper_center,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
    
        if circles is not None and len(circles[0]) > 0:
            indicators.append("front_closure")
    
        # INDICATOR 5: Hanger detection (item not being worn)
        # Presence of hanger suggests outerwear/layering piece

        top_section = gray[:int(h*0.1), :]
        # Look for thin horizontal line (hanger)
        edges = cv2.Canny(top_section, 50, 150)
        horizontal_lines = np.sum(edges, axis=1)

        if np.max(horizontal_lines) > w * 0.3:
            indicators.append("on_hanger")

        # SLEEVE PRESENCE CHECK — shoulder-zone lateral coverage.
        # Sleeved garments have fabric in the outer lateral columns at shoulder height.
        # Vests have armholes there — those columns are empty at shoulder height.
        shoulder_zone_left  = gray[:int(h * 0.30), :int(w * 0.15)]
        shoulder_zone_right = gray[:int(h * 0.30), int(w * 0.85):]

        left_shoulder_cov  = (np.sum(shoulder_zone_left  < 240) / shoulder_zone_left.size
                               if shoulder_zone_left.size  > 0 else 0.0)
        right_shoulder_cov = (np.sum(shoulder_zone_right < 240) / shoulder_zone_right.size
                               if shoulder_zone_right.size > 0 else 0.0)
        has_sleeve_evidence = left_shoulder_cov > 0.25 and right_shoulder_cov > 0.25

        if has_sleeve_evidence:
            print(f"🧥 Sleeve evidence (shoulder cov L={left_shoulder_cov:.0%} R={right_shoulder_cov:.0%}) — vest detection skipped")
            return {
                "is_likely_vest": False,
                "reasons": ["shoulder_sleeve_disqualified"],
                "confidence": 0.0
            }

        # WIDE HORIZONTAL SLEEVE CHECK — decisive disqualifier for draped garments
        # (Agbada, kimono, boubou): wide sleeves fill the extreme left/right at mid-height.
        mid_section = gray[int(h * 0.20):int(h * 0.55), :]
        extreme_left  = mid_section[:, :int(w * 0.10)]
        extreme_right = mid_section[:, int(w * 0.90):]

        el_cov = np.sum(extreme_left  < 240) / extreme_left.size  if extreme_left.size  > 0 else 0
        er_cov = np.sum(extreme_right < 240) / extreme_right.size if extreme_right.size > 0 else 0
        has_wide_sleeves = el_cov > 0.30 and er_cov > 0.30

        if has_wide_sleeves:
            print(f"🧥 Wide-sleeve evidence (extreme lateral: L={el_cov:.0%} R={er_cov:.0%}) — vest detection skipped")
            return {
                "is_likely_vest": False,
                "reasons": ["wide_sleeve_disqualified"],
                "confidence": 0.0
            }

        # DECISION: Is it a vest?
        is_vest = len(indicators) >= 3

        return {
            "is_likely_vest": is_vest,
            "reasons": indicators,
            "confidence": len(indicators) / 5  # Out of 5 possible indicators
        }
    
    @staticmethod
    def _detect_skin_exposure(image: np.ndarray) -> Dict:
        """
        Detect skin-colored regions (indicates sleeveless/exposed areas)
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Skin tone ranges (various skin tones)
        # Lower bound: lighter skin
        lower_skin_1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin_1 = np.array([20, 150, 255], dtype=np.uint8)
        
        # Upper bound: darker skin
        lower_skin_2 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin_2 = np.array([25, 170, 255], dtype=np.uint8)
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
        mask2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        
        # Determine regions
        h, w = skin_mask.shape
        regions = []
        
        # Check shoulders (upper left/right)
        left_shoulder = skin_mask[:int(h*0.3), :int(w*0.3)]
        right_shoulder = skin_mask[:int(h*0.3), int(w*0.7):]
        
        if np.sum(left_shoulder > 0) > (left_shoulder.size * 0.2):
            regions.append("shoulders")
        if np.sum(right_shoulder > 0) > (right_shoulder.size * 0.2):
            regions.append("shoulders")
        
        # Check arms (mid-left/right)
        left_arm = skin_mask[int(h*0.2):int(h*0.6), :int(w*0.25)]
        right_arm = skin_mask[int(h*0.2):int(h*0.6), int(w*0.75):]
        
        if np.sum(left_arm > 0) > (left_arm.size * 0.15):
            regions.append("arms")
        if np.sum(right_arm > 0) > (right_arm.size * 0.15):
            regions.append("arms")
        
        # Check chest/décolletage (center top)
        chest = skin_mask[:int(h*0.4), int(w*0.3):int(w*0.7)]
        if np.sum(chest > 0) > (chest.size * 0.15):
            regions.append("chest")
        
        has_skin = skin_percentage > 0.05  # > 5% of image is skin
        
        return {
            "has_skin": has_skin,
            "percentage": float(skin_percentage),
            "regions": list(set(regions)),
            "mask": skin_mask
        }
    
    @staticmethod
    def _analyze_coverage(gray: np.ndarray) -> Dict:
        """
        Analyze how much vertical space the garment occupies
        """
        h, w = gray.shape
        
        # Threshold to find garment (non-background)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find topmost and bottommost pixels
        rows_with_content = np.where(np.any(binary > 0, axis=1))[0]
        
        if len(rows_with_content) == 0:
            return {"vertical_coverage": 0.0}
        
        top = rows_with_content[0]
        bottom = rows_with_content[-1]
        
        coverage = (bottom - top) / h
        
        return {
            "vertical_coverage": float(coverage),
            "top_position": int(top),
            "bottom_position": int(bottom)
        }
    
    @staticmethod
    def _detect_thin_straps(gray: np.ndarray) -> Dict:
        """
        Detect thin straps (halter, spaghetti straps)
        """
        h, w = gray.shape

        # Guard: if both shoulder zones have fabric coverage the garment has sleeves.
        # Graphic elements (wavy text, illustrations) produce vertical/diagonal edges
        # in the shoulder region that can falsely trigger strap detection.
        shoulder_zone_left  = gray[:int(h * 0.30), :int(w * 0.15)]
        shoulder_zone_right = gray[:int(h * 0.30), int(w * 0.85):]
        left_cov  = (np.sum(shoulder_zone_left  < 240) / shoulder_zone_left.size
                     if shoulder_zone_left.size  > 0 else 0.0)
        right_cov = (np.sum(shoulder_zone_right < 240) / shoulder_zone_right.size
                     if shoulder_zone_right.size > 0 else 0.0)
        if left_cov > 0.25 and right_cov > 0.25:
            print(f"🧥 Shoulder fabric present (L={left_cov:.0%} R={right_cov:.0%}) — strap detection skipped")
            return {"has_thin_straps": False, "strap_count": 0}

        # Look in shoulder region (upper 20%)
        shoulder_region = gray[:int(h*0.2), :]
        
        # Detect edges
        edges = cv2.Canny(shoulder_region, 50, 150)
        
        # Detect lines (straps are thin vertical/diagonal lines)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )
        
        thin_straps = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Position filter: real halter/spaghetti straps sit in the CENTRAL zone
                # of the frame (roughly 15–85 % of width). Sleeve silhouettes of sleeved
                # garments appear at the extreme left/right edges; neckline seam artifacts
                # sit at the very center. Excluding both extremes leaves only true straps.
                center_x = (x1 + x2) / 2
                if center_x < w * 0.20 or center_x > w * 0.80:
                    continue

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

                # True vertical straps only — halter/spaghetti straps hang nearly
                # straight down (65–115°). The original `angle > 60 or angle < 30`
                # inadvertently passed horizontal neckline seams (≈0°) as "straps".
                is_vertical_ish = 65 < angle < 115
                is_long = length > shoulder_region.shape[0] * 0.3

                if is_vertical_ish and is_long:
                    thin_straps.append(line)

        # Require 2+ strap lines in the central zone.  A single interior line can
        # be a neckline seam or label; two symmetric lines are the left + right strap.
        has_thin_straps = len(thin_straps) >= 2

        return {
            "has_thin_straps": has_thin_straps,
            "strap_count": len(thin_straps)
        }
    
    @staticmethod
    def _analyze_neckline_shape(gray: np.ndarray) -> str:
        """
        Analyze neckline shape
        """
        h, w = gray.shape
        
        # Focus on top center (neckline area)
        neck_region = gray[:int(h*0.25), int(w*0.25):int(w*0.75)]
        
        # Detect edges
        edges = cv2.Canny(neck_region, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return "unknown"
        
        # Analyze line angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            angles.append(angle)
        
        # Deep V = many diagonal lines (30-60 degrees)
        diagonal_count = sum(1 for a in angles if 30 < abs(a) < 60)
        
        # High neck = many horizontal lines near top
        horizontal_count = sum(1 for a in angles if abs(a) < 15)
        
        if diagonal_count >= 2:
            return "deep_v"
        elif horizontal_count >= 2:
            return "high"
        else:
            return "standard"
