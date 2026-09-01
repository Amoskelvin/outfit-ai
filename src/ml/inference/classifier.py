import numpy as np
import cv2
from typing import Dict
from src.ml.models.model_manager import ClothingClassifierModel
from src.ml.inference.attribute_detector import AdvancedAttributeDetector
from src.config.settings import settings
from src.ml.inference.garment_analyzer import GarmentTypeAnalyzer

class ClothingClassifier:
    """
    High-level interface for clothing classification
    """
    
    def __init__(self):
        self.model = ClothingClassifierModel(device=settings.device)
        self.attribute_detector = AdvancedAttributeDetector()
        self.garment_analyzer = GarmentTypeAnalyzer()
    
    def classify(self, image: np.ndarray, color_palette: Dict = None, pattern: str = None) -> Dict:
        """
        Classify with context-awareness
        """
        try:
            results = self.model.classify_all(image)

            structure = self.garment_analyzer.analyze_structure(image)

            print(f" Structure analysis: {structure['confidence_flags']}")
        
            # === BOTTOM GARMENT OVERRIDE (highest priority — runs before all other overrides) ===
            # If the structural analyser detected bilateral leg columns, this is definitively
            # a bottom garment. Force the category and re-classify the subcategory.
            if structure.get("is_bottom_garment"):
                if results["category"] != "bottom":
                    print(f"⚠️ BOTTOM OVERRIDE: bilateral legs detected → forcing category=bottom "
                          f"(was {results['category']}/{results['subcategory']})")
                    results["category"] = "bottom"
                    results["category_confidence"] = max(results.get("category_confidence", 0.0), 0.80)
                    subcat_result = self.model.classify_subcategory(image, "bottom")
                    results["subcategory"] = subcat_result["subcategory"]
                    results["subcategory_confidence"] = subcat_result["confidence"]
                # Strip anatomy fields that belong only to upper-body garments
                results.pop("neckline", None)
                results.pop("sleeve_length", None)

            if structure["structure_type"] in ["crop_or_halter", "strap_based"]:
                if results["category"] in ["outerwear", "accessories", "shoes", "dress"]:
                    print(f"⚠️ OVERRIDE: {structure['structure_type']} cannot be {results['category']} → top")
                    results["category"] = "top"
                    results["category_confidence"] = 0.75
                    subcat_result = self.model.classify_subcategory(image, "top")
                    results["subcategory"] = subcat_result["subcategory"]
                    results["subcategory_confidence"] = subcat_result["confidence"]

            if structure["is_vest"]:
                # Guard: trust CLIP's initial category verdict.
                # The structural detector has known failure modes: white garments on white
                # backgrounds make sleeves invisible to pixel thresholding; circular graphics
                # (logos, eye illustrations) trigger the front_closure indicator. If CLIP
                # already classified this as "top", the structural analysis is firing on a
                # false positive — trust CLIP over geometry.
                top_score         = results.get("all_category_scores", {}).get("top", 0.0)
                bottom_score      = results.get("all_category_scores", {}).get("bottom", 0.0)
                traditional_score = results.get("all_category_scores", {}).get("traditional", 0.0)
                if results["category"] == "top" or bottom_score > 0.12 or traditional_score > 0.08:
                    print(f"⚠️ VEST OVERRIDE skipped: CLIP initial category={results['category']} top={top_score:.2f}")
                else:
                    if results["category"] != "outerwear":
                        print(f"⚠️ VEST OVERRIDE: {structure['structure_type']} detected → outerwear")
                        results["category"] = "outerwear"
                        results["category_confidence"] = 0.80

                    if results["subcategory"] not in ["vest", "gilet"]:
                        print(f"⚠️ VEST OVERRIDE: Changing '{results['subcategory']}' → vest")
                        results["subcategory"] = "vest"
                        results["subcategory_confidence"] = 0.85

                    # Vests are layering pieces - adjust formality
                    if results["formality"] == "formal":
                        results["formality"] = "business casual"

            if results["category"] == "dress":
                coverage = structure.get("vertical_coverage", 1.0)
                scores = results.get("all_category_scores", {})
                top_score = scores.get("top", 0.0)
                outer_score = scores.get("outerwear", 0.0)
                if coverage < 0.60 and (top_score >= 0.20 or outer_score >= 0.20):
                    print(f"⚠️ OVERRIDE: Short coverage with top/outerwear signal → top")
                    results["category"] = "top"
                    results["category_confidence"] = max(results["category_confidence"], 0.70)
                    subcat_result = self.model.classify_subcategory(image, "top")
                    results["subcategory"] = subcat_result["subcategory"]
                    results["subcategory_confidence"] = subcat_result["confidence"]

            # Promote dress if coverage suggests a full-length garment.
            # Guard 1: strap-based / crop tops are defined by neckline design, not hem length —
            #          their pointed or asymmetric hems must not trigger a dress promotion.
            # Guard 2: CLIP already identified a subcategory that is categorically not a dress
            #          (t-shirt, shirt, hoodie, jersey, etc.). Flat-lay product photography
            #          makes any top fill the full frame → vertical_coverage spikes above 0.70
            #          even for a standard short tee. Trust CLIP's subcategory over pixel height.
            _never_dress_subcats = {
                # T-shirt family
                "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
                "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
                # Polo family
                "classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
                "sleeveless polo", "striped polo", "long sleeve polo",
                # Shirt family
                "shirt", "oxford shirt", "graphic print shirt", "Flannel/checkered shirt", "linen shirt",
                "denim shirt", "oversized shirt", "cropped button-up", "cuban collar shirt",
                "tie-dye shirt", "vintage shirt", "sheer shirt", "blouse",
                # Hoodie / sweatshirt family
                "hoodie", "zip hoodie", "graphic hoodie", "cropped hoodie", "oversized hoodie",
                "crewneck sweatshirt", "half-zip sweatshirt", "sleeveless hoodie", "sweatshirt",
                # Knit / sweater family
                "sweater", "crew neck sweater", "v-neck sweater", "turtleneck",
                "ribbed turtleneck", "oversized knit", "cardigan", "buttoned cardigan", "knit vest",
                # Sports
                "jersey",
            }
            if (results["category"] == "top"
                    and structure.get("vertical_coverage", 0) > 0.70
                    and structure.get("structure_type") not in ["strap_based", "crop_or_halter"]
                    and results.get("subcategory", "").lower() not in _never_dress_subcats):
                print("⚠️ OVERRIDE: Full-length coverage → dress")
                results["category"] = "dress"
                results["category_confidence"] = max(results["category_confidence"], 0.75)
                subcat_result = self.model.classify_subcategory(image, "dress")
                results["subcategory"] = subcat_result["subcategory"]
                results["subcategory_confidence"] = subcat_result["confidence"]

            # DRESS DEMOTION — runs AFTER the promotion guard so it cannot be undone.
            # Certain feminine dress styles (milkmaid, sundress) require visible straps or
            # off-shoulder construction. When structural analysis finds no strap-based
            # profile, CLIP hallucinated a dress on a structured top (e.g. flat-lay polo).
            # Placing this block after the promotion guard breaks the demote→promote loop.
            _strap_required_subcats = {"milkmaid dress", "sundress", "off-shoulder dress"}
            if (results["category"] == "dress"
                    and results.get("subcategory", "").lower() in _strap_required_subcats
                    and structure.get("structure_type") not in ["strap_based", "crop_or_halter"]
                    and not structure.get("has_thin_straps", False)):
                print(f"⚠️ DRESS DEMOTION: '{results['subcategory']}' without straps → top")
                results["category"] = "top"
                results["category_confidence"] = 0.75
                subcat_result = self.model.classify_subcategory(image, "top")
                results["subcategory"] = subcat_result["subcategory"]
                results["subcategory_confidence"] = subcat_result["confidence"]

            # Fix sleeve length (only if already present)
            if results["category"] in ["top", "outerwear", "dress"] and not structure["has_sleeves"]:
                if results.get("sleeve_length") and results.get("sleeve_length") != "sleeveless":
                    print(f"⚠️ OVERRIDE: Skin detected on shoulders → sleeveless")
                    results["sleeve_length"] = "sleeveless"
            
            # Fix subcategory for halter/strap tops.
            # Guard: if CLIP already returned a clearly sleeved top type with decent
            # confidence, the strap detection is a false positive (e.g. T-shirt neckline
            # boundary lines passing through the shoulder zone on a white-on-white image).
            # In that case keep CLIP's verdict — don't force "halter top".
            if structure["structure_type"] == "strap_based" and results["category"] == "top":
                _sleeved_tops = {
                    "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
                    "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
                    "shirt", "oxford shirt", "graphic print shirt", "Flannel/checkered shirt", "linen shirt",
                    "denim shirt", "oversized shirt", "cropped button-up", "cuban collar shirt",
                    "tie-dye shirt", "vintage shirt", "sheer shirt", "blouse",
                    "classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
                    "striped polo", "long sleeve polo",
                    "hoodie", "zip hoodie", "graphic hoodie", "cropped hoodie", "oversized hoodie",
                    "crewneck sweatshirt", "half-zip sweatshirt",
                    "sweater", "crew neck sweater", "v-neck sweater", "turtleneck",
                    "ribbed turtleneck", "oversized knit", "cardigan", "buttoned cardigan",
                    "pullover", "sweatshirt", "jersey",
                }
                _is_confident_sleeved = (
                    results.get("subcategory", "").lower() in _sleeved_tops
                    and results.get("subcategory_confidence", 0.0) >= 0.35
                )
                if not _is_confident_sleeved:
                    results["subcategory"] = "halter top"
                    results["subcategory_confidence"] = 0.80
            
            # Tee-variant low-confidence fallback.
            # With 65 top subcategories competing in softmax, CLIP spreads probability
            # thinly on plain garments (no graphics, no hardware, no distinctive collar).
            # It grabs at visual artifacts — a chest shadow → "pocket tee", a ribbed
            # collar shadow → "v-neck tee".  If the chosen subcategory is one of the
            # fine-grained tee variants and confidence is below 0.40, collapse back to
            # the generic "t-shirt" label which has a much cleaner CLIP description.
            _tee_variants = {
                "v-neck tee", "graphic print tee", "vintage tee", "oversized tee", "cropped tee",
                "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
            }
            if (results["subcategory"] in _tee_variants
                    and results.get("subcategory_confidence", 1.0) < 0.45):
                print(f"⬇️ Tee variant '{results['subcategory']}' low confidence "
                      f"({results['subcategory_confidence']:.2f}) → t-shirt")
                results["subcategory"] = "t-shirt"

            # Polo vs full-placket button-up shirt disambiguation.
            # CLIP confuses short-sleeve Oxford shirts with polos: both have a collar
            # and short sleeves. The structural difference is the button run length:
            # polo = 2-3 buttons in the top ~25% of the shirt only;
            # Oxford/button-up = 5-7 buttons running the full length to the hem.
            # When CLIP returns a polo subcategory with low confidence, count how far
            # down the center strip the button column extends.
            _polo_check_subcats = {
                "classic buttoned polo", "open v-neck polo", "striped polo", "long sleeve polo"
            }
            if (results.get("subcategory") in _polo_check_subcats
                    and results.get("subcategory_confidence", 1.0) < 0.60):
                try:
                    _h, _w = image.shape[:2]
                    _is_oxford = False

                    # Pass 1: HoughCircles button count — works when button color
                    # contrasts with the fabric (white/metal buttons on any fabric).
                    # Fails for same-color buttons (navy-on-navy, black-on-black).
                    _gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    _strip = _gray[:, _w//2 - 55:_w//2 + 55]
                    _circles = cv2.HoughCircles(
                        _strip, cv2.HOUGH_GRADIENT, 1, 20,
                        param1=50, param2=25, minRadius=4, maxRadius=22
                    )
                    if _circles is not None and len(_circles[0]) >= 4:
                        _lowest = max(c[1] for c in _circles[0]) / _h
                        if _lowest > 0.55:
                            print(f"⚠️ POLO→OXFORD: full placket "
                                  f"({len(_circles[0])} buttons, lowest={_lowest:.0%})")
                            _is_oxford = True

                    # Pass 2: CLIP collar-type check — fallback for low-contrast buttons.
                    # Polo collar = ribbed knit texture. Oxford collar = flat woven points.
                    # Crop just the collar area (top 32%, center 60%) for a clean signal.
                    if not _is_oxford:
                        try:
                            from PIL import Image as PILImage
                            _cm, _cp_proc = self.attribute_detector._get_clip_model()
                            _collar_crop = image[:int(_h * 0.32), int(_w * 0.20):int(_w * 0.80)]
                            _pil = PILImage.fromarray(_collar_crop.astype("uint8"))
                            _collar_prompts = [
                                "a flat woven dress shirt collar with two collar points extending sideways — Oxford or button-up shirt",
                                "a ribbed or knit polo collar with visible texture — polo shirt",
                            ]
                            _inp = _cp_proc(text=_collar_prompts, images=_pil,
                                            return_tensors="pt", padding=True)
                            _out = _cm(**_inp)
                            _cprobs = _out.logits_per_image.softmax(dim=1).detach().numpy()[0]
                            print(f"🔍 Collar CLIP: oxford={_cprobs[0]:.2f} polo={_cprobs[1]:.2f}")
                            if _cprobs[0] > 0.50:
                                _is_oxford = True
                        except Exception as _ce:
                            print(f"⚠️ Collar CLIP error: {_ce}")

                    if _is_oxford:
                        results["subcategory"] = "oxford shirt"
                        results["subcategory_confidence"] = max(
                            results.get("subcategory_confidence", 0.0), 0.62
                        )
                except Exception as _e:
                    print(f"⚠️ Polo/Oxford check error: {_e}")

            # Checkered/plaid shirt vs generic shirt subcategory disambiguation.
            # CLIP labels plaid/tartan shirts as "vintage shirt" because the irregular woven
            # texture and casual silhouette share visual features.  When a generic shirt
            # subcategory is returned at low confidence, run a two-prompt CLIP check that
            # tests whether the fabric shows a multi-color intersecting grid (checkered) vs
            # a uniform single-color weave (solid shirt).
            _generic_shirt_variants = {"vintage shirt", "shirt", "oversized shirt", "sheer shirt"}
            if (results.get("subcategory") in _generic_shirt_variants
                    and results.get("subcategory_confidence", 1.0) < 0.65):
                try:
                    from PIL import Image as PILImage
                    _cm, _cp_proc = self.attribute_detector._get_clip_model()
                    _pil_chk = PILImage.fromarray(image.astype("uint8"))
                    _chk_prompts = [
                        "a plaid or checkered shirt — regular intersecting horizontal and "
                        "vertical colored lines forming a multi-color check grid across the entire fabric",
                        "a solid-color or uniformly washed single-color shirt — no visible "
                        "grid, check, or plaid pattern on the fabric",
                    ]
                    _inp_chk = _cp_proc(text=_chk_prompts, images=_pil_chk,
                                        return_tensors="pt", padding=True)
                    _out_chk = _cm(**_inp_chk)
                    _chkprobs = _out_chk.logits_per_image.softmax(dim=1).detach().numpy()[0]
                    print(f"🔍 Check/Solid CLIP: plaid={_chkprobs[0]:.2f} solid={_chkprobs[1]:.2f}")
                    if _chkprobs[0] > 0.55:
                        results["subcategory"] = "Flannel/checkered shirt"
                        results["subcategory_confidence"] = max(
                            results.get("subcategory_confidence", 0.0), 0.62
                        )
                        print("✅ Reclassified → Flannel/checkered shirt (plaid pattern confirmed)")
                except Exception as _ce:
                    print(f"⚠️ Check/Solid CLIP error: {_ce}")

            # Linen shirt vs Oxford shirt disambiguation.
            # CLIP confuses these two when the garment fills the frame and the fabric texture
            # is muted (dark colours flatten the weave).  Linen has a soft, irregular drape;
            # Oxford has a stiff structured collar with crisp collar points.  When CLIP returns
            # "linen shirt" up to moderate confidence, run a CLIP collar-structure check.
            if (results.get("subcategory") == "linen shirt"
                    and results.get("subcategory_confidence", 1.0) < 0.70):
                try:
                    from PIL import Image as PILImage
                    _cm, _cp_proc = self.attribute_detector._get_clip_model()
                    _h_ln, _w_ln = image.shape[:2]
                    _collar_ln = image[:int(_h_ln * 0.40), int(_w_ln * 0.15):int(_w_ln * 0.85)]
                    _pil_ln = PILImage.fromarray(_collar_ln.astype("uint8"))
                    _linen_prompts = [
                        "an Oxford cloth button-down shirt with a stiff structured collar — two sharp collar points lying flat",
                        "a lightweight linen shirt with a soft relaxed collar — visible coarse irregular woven texture",
                    ]
                    _ln_inp = _cp_proc(text=_linen_prompts, images=_pil_ln,
                                       return_tensors="pt", padding=True)
                    _ln_out = _cm(**_ln_inp)
                    _lnprobs = _ln_out.logits_per_image.softmax(dim=1).detach().numpy()[0]
                    print(f"🔍 Linen/Oxford CLIP: oxford={_lnprobs[0]:.2f} linen={_lnprobs[1]:.2f}")
                    if _lnprobs[0] > 0.55:
                        print("⚠️ OVERRIDE: linen shirt → oxford shirt (structured collar)")
                        results["subcategory"] = "oxford shirt"
                        results["subcategory_confidence"] = max(
                            results.get("subcategory_confidence", 0.0), 0.62
                        )
                except Exception as _le:
                    print(f"⚠️ Linen/Oxford CLIP error: {_le}")

            # Zip-hoodie vs half-zip-sweatshirt disambiguation.
            # CLIP confuses these two because on a flat-lay product photo both show a zipper
            # track in the center. The key visual difference is the hood: a zip hoodie has a
            # large fabric hood folded at the back of the collar; a half-zip sweatshirt has a
            # plain high ribbed collar with no hood.  When CLIP returns "half-zip sweatshirt"
            # with low confidence, run a targeted CLIP two-prompt hood check.
            if (results.get("subcategory") == "half-zip sweatshirt"
                    and results.get("subcategory_confidence", 1.0) < 0.60):
                try:
                    from PIL import Image as PILImage
                    _cm, _cp_proc = self.attribute_detector._get_clip_model()
                    _h_hd, _w_hd = image.shape[:2]
                    _hood_crop = image[:int(_h_hd * 0.50), :]
                    _pil_hd = PILImage.fromarray(_hood_crop.astype("uint8"))
                    _hood_prompts = [
                        "a zip-up hoodie with a large fabric hood attached behind the collar — the hood is clearly visible",
                        "a half-zip sweatshirt with a plain high ribbed collar and no hood — only a short zipper at the top",
                    ]
                    _inp_hd = _cp_proc(text=_hood_prompts, images=_pil_hd,
                                       return_tensors="pt", padding=True)
                    _out_hd = _cm(**_inp_hd)
                    _hprobs = _out_hd.logits_per_image.softmax(dim=1).detach().numpy()[0]
                    print(f"🔍 Hood CLIP: zip_hoodie={_hprobs[0]:.2f} half_zip={_hprobs[1]:.2f}")
                    if _hprobs[0] > 0.50:
                        print("⚠️ OVERRIDE: half-zip sweatshirt → zip hoodie (hood detected)")
                        results["subcategory"] = "zip hoodie"
                        results["subcategory_confidence"] = max(
                            results.get("subcategory_confidence", 0.0), 0.62
                        )
                except Exception as _he:
                    print(f"⚠️ Hood CLIP error: {_he}")

            # Hoodie / sweatshirt product-photo halo override.
            # White studio backgrounds create a sharp silhouette edge after mask application
            # (image * mask zeros the bg to black-zero, causing Canny to fire on the
            # garment boundary → high edge_density → "printed" → colorblock override).
            # This is a photography artifact, not actual multi-color design.
            # Only override "colorblock" — "graphic print" on a graphic hoodie can be real.
            _fleece_subcats = {
                "hoodie", "zip hoodie", "cropped hoodie", "oversized hoodie",
                "crewneck sweatshirt", "half-zip sweatshirt", "sleeveless hoodie", "sweatshirt",
            }
            if results.get("subcategory") in _fleece_subcats and pattern == "colorblock":
                print(f"⬇️ Hoodie halo artifact: pattern 'colorblock' → 'solid'")
                pattern = "solid"

            # Subcategory canonical normalization safety net.
            # Catches any label variant that wasn't renamed at the CLIP-label level and
            # maps it to the exact taxonomy enum before it can reach the database.
            _SUBCAT_ALIASES = {
                "graphic tee":       "graphic print tee",
                "striped tee":       "striped t-shirt",
                "polo open collar":  "open v-neck polo",
                "polo":              "classic buttoned polo",
                "flannel shirt":     "Flannel/checkered shirt",
                "cardigan":          "Cardigan (buttoned)",
                "buttoned cardigan": "Cardigan (buttoned)",
                "ribbed turtleneck": "Turtleneck",
            }
            if results.get("subcategory") in _SUBCAT_ALIASES:
                results["subcategory"] = _SUBCAT_ALIASES[results["subcategory"]]

            # NOTE: denim formality fix is applied AFTER detect_all_attributes below,
            # because results["texture"] is only populated after that call.
            
            # Add neckline from structure hints
            if results["category"] in ["top", "dress"] and structure.get("neckline_hint"):
                results["neckline"] = structure["neckline_hint"]    
        
            # === TRADITIONAL WEAR RULES ===
            # Traditional ceremonial garments are always formal and all-season
            traditional_formal_subcats = [
                "agbada", "senator wear", "kaftan", "embroidered kaftan",
                "thobe", "dashiki", "kurta", "buba", "ankara top",
            ]
            if results["category"] == "traditional" or results["subcategory"] in traditional_formal_subcats:
                if results["formality"] != "formal":
                    print(f"⚠️ Correcting: traditional wear → formal")
                    results["formality"] = "formal"
                    results["formality_confidence"] = 0.92
                # NOTE: season="all" is applied AFTER _detect_season_improved below to prevent overwrite

            # === FORMALITY CORRECTIONS ===
            # Rule 1: Loud prints are NEVER formal
            loud_prints = ["tropical print", "bandana patchwork", "floral print",
                           "graphic print", "printed", "patterned"]
            if pattern in loud_prints and results["formality"] != "casual":
                print(f"⚠️ Correcting: {pattern} → casual")
                results["formality"] = "casual"
                results["formality_confidence"] = 0.85

            # Rule 2: Casual-only garment types regardless of color or graphic content.
            casual_items = {
                # All t-shirt variants
                "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
                "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
                # Hoodies & sweatshirts
                "hoodie", "zip hoodie", "graphic hoodie", "cropped hoodie", "oversized hoodie",
                "crewneck sweatshirt", "half-zip sweatshirt", "sleeveless hoodie",
                # Tanks & minimal coverage
                "tank top", "ribbed tank", "graphic tank", "muscle tank", "longline tank",
                "spaghetti strap top", "racerback tank",
                # Women's casual tops
                "halter top", "crop top", "cami", "tube top", "off-shoulder top",
                "one-shoulder top",
                # Bottoms
                "shorts", "sweatpants", "joggers",
                # Sports
                "jersey",
                # Casual shirts
                "tie-dye shirt", "tie-dye tee", "graphic print shirt",
            }
            if results["subcategory"] in casual_items:
                print(f"⚠️ Correcting: {results['subcategory']} → casual")
                results["formality"] = "casual"
                results["formality_confidence"] = 0.90

            # Rule 2b: All polo variants are never formal — cap at smart casual.
            polo_subcats = {"classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
                            "sleeveless polo", "striped polo", "long sleeve polo"}
            if results["subcategory"] in polo_subcats and results["formality"] == "formal":
                results["formality"] = "smart casual"
                results["formality_confidence"] = 0.80

            # Rule 2c: Linen and cuban-collar shirts are resort/smart casual — not formal.
            if results["subcategory"] in {"linen shirt", "cuban collar shirt", "vintage shirt",
                                          "sheer shirt", "oversized shirt", "Flannel/checkered shirt",
                                          "denim shirt"}:
                if results["formality"] == "formal":
                    results["formality"] = "smart casual"
                    results["formality_confidence"] = 0.80

            # Rule 3: Blazers, dress pants = at least business casual
            formal_items = ["blazer", "dress pants", "suit jacket"]
            if results["subcategory"] in formal_items and results["formality"] == "casual":
                print(f"⚠️ Correcting: {results['subcategory']} → business casual")
                results["formality"] = "business casual"
                results["formality_confidence"] = 0.85

            # Rule 3b: Casual dress types are not formal
            if results["category"] == "dress":
                casual_dresses = ["mini dress", "sundress", "milkmaid dress"]
                if results["subcategory"] in casual_dresses and results["formality"] == "formal":
                    print(f"?? Correcting: {results['subcategory']} ? casual")
                    results["formality"] = "casual"
                    results["formality_confidence"] = 0.85
                # Rule 3c: Short/open dresses are typically not formal
                if results["formality"] == "formal":
                    coverage = structure.get("vertical_coverage", 1.0)
                    if coverage < 0.85:
                        print(f"?? Correcting: Short/open dress coverage {coverage:.1%} ? business casual")
                        results["formality"] = "business casual"
                        results["formality_confidence"] = max(results["formality_confidence"], 0.70)
        
            # Rule 4: JACKETS = casual unless explicitly a blazer/suit jacket
            if results["category"] == "outerwear" or results["subcategory"] in ["jacket", "bomber jacket", "leather jacket"]:
                # Blazers and waistcoats can be business casual/formal — everything else is casual
                formal_outerwear = ["blazer", "suit jacket", "waistcoat"]
                is_casual_jacket = results["subcategory"] not in formal_outerwear

                # Bright/vibrant colors reinforce casual
                if color_palette:
                    dominant = color_palette.get("primary_color", "")
                    casual_colors = ["bright red", "bright orange", "hot pink", "mustard",
                                   "ochre", "camel", "lime green", "bright yellow", "turquoise"]
                    if any(cc in dominant for cc in casual_colors):
                        is_casual_jacket = True

                # Loud pattern = casual
                if pattern in loud_prints:
                    is_casual_jacket = True

                if is_casual_jacket and results["formality"] != "casual":
                    print(f"⚠️ Correcting: Casual jacket detected → casual (was {results['formality']})")
                    results["formality"] = "casual"
                    results["formality_confidence"] = 0.88
        
            # Rule 5: Vibrant colors = usually casual
            if color_palette:
                dominant = color_palette.get("primary_color", "")
                very_vibrant = ["bright red", "bright orange", "hot pink", "lime green", 
                              "bright yellow", "neon", "turquoise"]
                if any(vc in dominant for vc in very_vibrant):
                    if results["formality"] == "formal":
                        print(f"⚠️ Correcting: vibrant {dominant} → casual")
                        results["formality"] = "casual"
                        results["formality_confidence"] = 0.80
        
            # === SEASON CORRECTIONS ===
            results["season"] = self._detect_season_improved(
                results["category"],
                results["subcategory"],
                color_palette,
                pattern
            )
            # Traditional garments are all-season regardless of color palette — must run
            # AFTER _detect_season_improved to prevent the color-based logic from overwriting it.
            if results["category"] == "traditional" or results.get("subcategory") in traditional_formal_subcats:
                results["season"] = "all"

            advanced_attrs = self.attribute_detector.detect_all_attributes(
                image,
                results["category"],
                results["subcategory"],
                pattern or "solid",
                structure_hints=structure
            )

            # _tee_family defined here so it is available for both the _definite_sleeves
            # guard below and the tank reclassification block further down.
            _tee_family = {
                "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
                "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
            }

            if results["category"] in ["top", "outerwear", "dress"] and not structure["has_sleeves"]:
                # Protect genuinely subcategory-name-derived sleeve lengths from structural
                # override (e.g. "long sleeve polo" → "long sleeve", "agbada" → "wide drape").
                # "short sleeve" is protected for non-tee subcategories (e.g. polo shirts):
                # the structural analyzer is unreliable on flat-lay product photos where short
                # sleeves don't extend far from the body and register as absent.
                # For tee-family subcategories "short sleeve" is a category default that CAN
                # be overridden by structural analysis; the separate tank reclassification
                # block below handles that case via a CLIP visual check instead.
                _definite_sleeves = {"long sleeve", "3/4 sleeve", "wide drape"}
                if results.get("subcategory") not in _tee_family:
                    _definite_sleeves.add("short sleeve")
                if advanced_attrs.get("sleeve_length") not in _definite_sleeves:
                    advanced_attrs["sleeve_length"] = "sleeveless"

            results.update(advanced_attrs)

            # Tank vs t-shirt CLIP visual disambiguation.
            # Only fires when CLIP returned the GENERIC "t-shirt" collapsed label (from the
            # tee-variant fallback) at low confidence.  Specific variants like "graphic print tee"
            # or "striped t-shirt" already encode enough visual evidence — firing on them causes
            # false tank reclassification (the Vision graphic tee and Rhythm stripe tee failures).
            if (results.get("subcategory") == "t-shirt"
                    and results.get("subcategory_confidence", 1.0) < 0.52
                    and results.get("category") == "top"):
                try:
                    from PIL import Image as PILImage
                    _cm, _cp_proc = self.attribute_detector._get_clip_model()
                    _pil_tk = PILImage.fromarray(image.astype("uint8"))
                    _tank_vs_tee = [
                        "a sleeveless tank top — bare upper arms and shoulders visible on both sides of the garment, no sleeve fabric",
                        "a short-sleeve t-shirt — fabric covers the upper arms and shoulders on both sides, sleeves visible",
                    ]
                    _inp_tk = _cp_proc(text=_tank_vs_tee, images=_pil_tk,
                                       return_tensors="pt", padding=True)
                    _out_tk = _cm(**_inp_tk)
                    _tkprobs = _out_tk.logits_per_image.softmax(dim=1).detach().numpy()[0]
                    print(f"🔍 Tank/Tee CLIP: tank={_tkprobs[0]:.2f} tee={_tkprobs[1]:.2f}")
                    if _tkprobs[0] > 0.55:
                        _tank_subcats = [
                            "tank top", "ribbed tank", "graphic tank", "muscle tank",
                            "longline tank", "spaghetti strap top", "racerback tank",
                        ]
                        _tk_descs = [self.model.subcategory_labels.get(sc, sc) for sc in _tank_subcats]
                        _tk_inp2 = self.model.processor(text=_tk_descs, images=_pil_tk,
                                                        return_tensors="pt", padding=True)
                        _tk_out2 = self.model.model(**_tk_inp2)
                        _tk_p2 = _tk_out2.logits_per_image.softmax(dim=1).detach().numpy()[0]
                        _tk_idx2 = int(np.argmax(_tk_p2))
                        results["subcategory"] = _tank_subcats[_tk_idx2]
                        results["subcategory_confidence"] = float(_tk_p2[_tk_idx2])
                        results["sleeve_length"] = "sleeveless"
                        print(f"✅ Tank reclassify: '{results['subcategory']}' "
                              f"({results['subcategory_confidence']:.2f})")
                except Exception as _te:
                    print(f"⚠️ Tank/Tee CLIP error: {_te}")

            # Sleeveless subcategory correction (structural path — fires when has_sleeves=False
            # AND subcategory is still in tee family after the CLIP check above).
            if (not structure["has_sleeves"]
                    and results.get("category") == "top"
                    and results.get("subcategory") in _tee_family):
                print(f"⚠️ SLEEVELESS RECLASSIFY: '{results['subcategory']}' has no sleeves → tank")
                _tank_subcats = [
                    "tank top", "ribbed tank", "graphic tank", "muscle tank",
                    "longline tank", "spaghetti strap top", "racerback tank",
                ]
                try:
                    from PIL import Image as PILImage
                    _pil_tk = PILImage.fromarray(image.astype("uint8"))
                    _tk_descs = [self.model.subcategory_labels[sc] for sc in _tank_subcats]
                    _tk_inp = self.model.processor(
                        text=_tk_descs, images=_pil_tk,
                        return_tensors="pt", padding=True
                    )
                    _tk_out = self.model.model(**_tk_inp)
                    _tk_probs = _tk_out.logits_per_image.softmax(dim=1).detach().numpy()[0]
                    _tk_idx = int(np.argmax(_tk_probs))
                    results["subcategory"] = _tank_subcats[_tk_idx]
                    results["subcategory_confidence"] = float(_tk_probs[_tk_idx])
                    print(f"✅ Tank reclassified → '{results['subcategory']}' "
                          f"({results['subcategory_confidence']:.2f})")
                except Exception as _te:
                    results["subcategory"] = "ribbed tank"
                    results["subcategory_confidence"] = 0.65
                    print(f"⚠️ Tank CLIP error — defaulting to ribbed tank: {_te}")

            # Taxonomy/pattern reconciliation. If the subcategory head confidently
            # identified a striped garment, do not let edge-density pattern analysis
            # reinterpret the repeated stripe edges as a graphic print.
            _striped_subcats = {"striped t-shirt", "striped polo"}
            if results.get("subcategory") in _striped_subcats:
                results["pattern"] = "striped"

            # Flannel/checkered subcategory → force checkered pattern.
            # FFT periodicity detection is direction-agnostic and returns "striped" for any
            # repeating grid — plaid and stripes look identical to it. The subcategory is
            # a more reliable signal for the actual design.
            _checkered_subcats = {"Flannel/checkered shirt"}
            if results.get("subcategory") in _checkered_subcats:
                results["pattern"] = "checkered"

            # Striped cotton cannot be embroidered: high-contrast horizontal stripe rows produce
            # dense Laplacian variance that falsely triggers the embroidery detector. A yarn-dyed
            # cotton stripe and embroidery stitching are physically incompatible fabrics.
            if results.get("pattern") == "striped" and results.get("texture") == "embroidered":
                print("⚠️ OVERRIDE: striped pattern cannot be embroidered → cotton")
                results["texture"] = "cotton"

            # Checkered + embroidered is physically impossible — plaid weave and embroidery
            # stitching don't coexist.  The Laplacian fires on grid intersections.
            # The correct texture is cotton (the "Flannel/checkered shirt" subcategory name
            # refers to the design type, not a brushed flannel weave).
            if results.get("pattern") == "checkered" and results.get("texture") == "embroidered":
                print("⚠️ OVERRIDE: checkered pattern cannot be embroidered → cotton")
                results["texture"] = "cotton"

            # Flannel/checkered shirt texture safety net: if the subcategory correction ran
            # after detect_all_attributes (Group 3 reclassification), the keyword shortcut
            # that would have returned "cotton" never fired. Force it now.
            if results.get("subcategory") == "Flannel/checkered shirt":
                if results.get("texture") in {"embroidered", "flannel", "smooth", "quilted", "textured"}:
                    results["texture"] = "cotton"

            # T-shirt family default texture: plain cotton.
            # CLIP occasionally returns "smooth", "textured", or "embossed" on a flat-knit tee.
            # Unless the detector found a physically meaningful special texture, default to cotton.
            _tee_cotton_subcats = {
                "t-shirt", "graphic print tee", "striped t-shirt", "vintage tee",
                "oversized tee", "cropped tee", "longline tee", "pocket tee",
                "v-neck tee", "tie-dye tee",
            }
            _special_textures = {
                "denim", "fleece", "knit", "flannel", "linen", "ribbed",
                "embroidered", "brocade/woven", "leather", "quilted",
            }
            if (results.get("subcategory") in _tee_cotton_subcats
                    and results.get("texture") not in _special_textures):
                results["texture"] = "cotton"

            # Shirt/button-up family default texture: cotton.
            # Oxford, graphic print, vintage shirts etc. fall through CLIP and get
            # "smooth" — incorrect for woven cotton Oxford cloth. Specific fabrics
            # (linen, flannel, denim) are handled by keyword shortcuts in the attribute
            # detector, so this default only fires when nothing specific was detected.
            _shirt_cotton_subcats = {
                "shirt", "oxford shirt", "graphic print shirt", "oversized shirt",
                "cropped button-up", "vintage shirt", "blouse",
            }
            if (results.get("subcategory") in _shirt_cotton_subcats
                    and results.get("texture") not in _special_textures):
                results["texture"] = "cotton"

            # T-shirt / jersey denim override: black or dark jersey fabric visually resembles
            # "dark raw denim" to CLIP because both are dark-toned. A garment already
            # identified as a t-shirt or jersey is never denim; reset to cotton.
            if results.get("subcategory") in {"t-shirt", "jersey"} and results.get("texture") == "denim":
                print("⚠️ OVERRIDE: t-shirt/jersey cannot be denim → cotton")
                results["texture"] = "cotton"
                results.pop("denim_wash", None)

            # Denim formality fix — runs here because texture is now populated.
            # Denim is always casual regardless of color name (navy, indigo, dark wash, etc.).
            if "denim" in results.get("texture", "").lower():
                if results["formality"] != "casual":
                    print(f"⚠️ OVERRIDE: Denim is always casual")
                    results["formality"] = "casual"
                    results["formality_confidence"] = 0.90

            # When embroidery is detected as the texture, surface it as the pattern too
            # so retail search for "embroidered" finds the item via both fields.
            if results.get("texture") == "embroidered" and results.get("pattern") in [None, "solid", "smooth"]:
                results["pattern"] = "embroidered"
            # For traditional garments the base fabric is brocade/woven but the defining
            # design element is the contrasting embroidery — surface it as the pattern.
            if (results["category"] == "traditional" or results.get("subcategory") in traditional_formal_subcats):
                if results.get("texture") in ["brocade/woven", "embroidered"] and results.get("pattern") in [None, "solid", "smooth"]:
                    results["pattern"] = "embroidered"

            # Agbada vs kaftan disambiguation: agbada is uniquely identified by the
            # combination of wide draped sleeves AND contrasting chest embroidery.
            if results["category"] == "traditional" and results.get("subcategory") == "kaftan":
                is_wide_drape = results.get("sleeve_length") == "wide drape"
                has_embroidery = (results.get("texture") in ["embroidered", "brocade/woven"]
                                  or results.get("pattern") == "embroidered")
                if is_wide_drape and has_embroidery:
                    print("⚠️ OVERRIDE: wide drape + embroidery → agbada (not kaftan)")
                    results["subcategory"] = "agbada"
                    results["subcategory_confidence"] = max(results.get("subcategory_confidence", 0.0), 0.75)

            # Vest/gilet override for sleeveless structured layering pieces
            # Skip if CLIP is confident this is a bottom garment
            bottom_score = results.get("all_category_scores", {}).get("bottom", 0.0)
            if results["category"] in ["top", "outerwear"] and bottom_score <= 0.12:
                coverage = structure.get("vertical_coverage", 1.0)
                if (not structure.get("has_sleeves", True)) and coverage < 0.65:
                    if structure.get("neckline_hint") == "collar or turtleneck":
                        results["subcategory"] = "vest"
                        results["subcategory_confidence"] = max(results.get("subcategory_confidence", 0.0), 0.75)
                        results["category"] = "top"
                        results["category_confidence"] = max(results.get("category_confidence", 0.0), 0.75)

            # Neckline fix for structured wrap vests
            if results.get("subcategory") == "vest":
                if results.get("closure_type") == "buttons":
                    results["neckline"] = "wrap high neck"
                elif structure.get("neckline_hint") == "collar or turtleneck":
                    results["neckline"] = "high neck"

            # Prevent anatomy fields on non-tops (keep neckline for outerwear — jackets have collars)
            if results["category"] not in ["top", "dress", "outerwear"]:
                results.pop("neckline", None)
            if results["category"] not in ["top", "outerwear", "dress"]:
                results.pop("sleeve_length", None)

            # Fix closure hallucinations on bottoms
            if results["category"] == "bottom":
                subcat = results.get("subcategory", "")
                # Drawstring bottoms
                if subcat in ["sweatpants", "joggers"] or "sweat" in subcat.lower() or "jogger" in subcat.lower():
                    results["closure_type"] = "drawstring"
                # Denim defaults to zipper but preserves button fly when detected
                elif "denim" in results.get("texture", "").lower():
                    if results.get("closure_type") not in ["button fly", "buttons", "zipper"]:
                        results["closure_type"] = "zipper"
                # Wrap/tie is rarely correct for standard bottom cuts
                elif results.get("closure_type") == "wrap/tie" and subcat in ["jeans", "pants", "shorts", "culottes"]:
                    print("⚠️ OVERRIDE: Bottoms rarely use wrap/tie → zipper")
                    results["closure_type"] = "zipper"

            # === DENIM WASH NOTE ===
            # denim_wash is detected in detect_all_attributes and already present
            # in results via results.update(advanced_attrs) above.

            # Boost confidence for traditional garments that survived all structural checks.
            # The CLIP base score is unreliable for non-Western silhouettes; once the
            # traditional rules have validated the garment, raise the floor to 0.75.
            if results["category"] == "traditional" and results["category_confidence"] < 0.50:
                if results.get("subcategory") in traditional_formal_subcats:
                    print(f"⚠️ Boosting traditional confidence: {results['category_confidence']:.2f} → 0.75")
                    results["category_confidence"] = max(results["category_confidence"], 0.75)

            # === PATTERN FINALISATION ===
            # If none of the subcategory-driven overrides above set results["pattern"],
            # carry through the image-processor pattern so upload.py can read it from
            # classification directly (avoids duplicating the fallback logic there).
            if "pattern" not in results:
                results["pattern"] = pattern or "solid"

            # Hard override: "graphic print" without any detected logo/text is a false
            # positive on solid-fabric subcategories. The pattern detector fires on
            # garment-boundary edge density or product-photo halo, not an actual print.
            # Scoped to subcategory families that are structurally never graphic print garments.
            _never_graphic_subcats = {
                # Shirts / button-ups
                "oxford shirt", "shirt", "linen shirt", "Flannel/checkered shirt", "denim shirt",
                "oversized shirt", "cropped button-up", "vintage shirt", "sheer shirt", "blouse",
                # Polo shirts (solid or stripe — handled by _striped_subcats above)
                "classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
                "sleeveless polo", "long sleeve polo",
                # Plain tees (graphic tees intentionally excluded)
                "t-shirt", "v-neck tee", "pocket tee", "longline tee",
                "oversized tee", "cropped tee",
                # Hoodies / sweatshirts
                "hoodie", "crewneck sweatshirt", "half-zip sweatshirt",
            }
            if (results.get("pattern") == "graphic print"
                    and not results.get("has_logo")
                    and results.get("subcategory") in _never_graphic_subcats):
                print(f"⚠️ OVERRIDE: graphic print without logo on "
                      f"'{results.get('subcategory')}' → solid")
                results["pattern"] = "solid"

            # Colorblock false-positive guard.
            # The same subcategory families that cannot produce "graphic print" also cannot
            # produce "colorblock" on a single-colour garment — colorblock requires two large
            # distinctly-coloured panels.  If the pattern detector returned "colorblock" for a
            # solid-fabric shirt or polo (typically caused by background contamination in the
            # GrabCut mask giving KMeans two clusters: garment + backdrop), override to solid.
            # has_logo is not relevant here (a logo would produce "graphic print", not colorblock).
            _never_colorblock_subcats = _never_graphic_subcats | {
                "ribbed tank", "tank top", "muscle tank", "longline tank",
                "vintage tee", "tie-dye tee",
                # Cardigans: ribbed trim at cuffs/hem is slightly darker and creates
                # two-tone KMeans clusters that the colorblock upgrade misinterprets.
                "Cardigan (buttoned)", "cardigan",
            }
            if (results.get("pattern") == "colorblock"
                    and results.get("subcategory") in _never_colorblock_subcats):
                print(f"⚠️ OVERRIDE: colorblock on solid-fabric "
                      f"'{results.get('subcategory')}' → solid")
                results["pattern"] = "solid"

            # === CARDIGAN POST-PROCESSING ===
            # The alias transform may have changed the subcategory to "Cardigan (buttoned)"
            # AFTER detect_all_attributes ran with the old name ("cardigan", "buttoned cardigan").
            # The closure and neckline rules in attribute_detector already handle
            # "Cardigan (buttoned)" correctly when the alias ran before the call (which it does),
            # but apply definitive overrides here as a safety net for any edge case.
            if results.get("subcategory") == "Cardigan (buttoned)":
                if results.get("closure_type") in {None, "open front", "pullover"}:
                    results["closure_type"] = "buttons"
                if results.get("neckline") == "v-neck":
                    results["neckline"] = None

            # === CONFIDENCE FLOOR ===
            # Category confidence is averaged across many category prompts (7 for "top"),
            # so on graphic-heavy garments CLIP spreads probability thinly and the average
            # can drop below 0.20 even when the category is unambiguously correct.
            # Subcategory confidence (chosen from 8 narrow "top" options) is much more
            # reliable: if CLIP picks "t-shirt" at 0.50, the category must be "top".
            # Scale the floor proportionally to subcategory confidence so graphic tees get
            # a realistic score, while genuinely ambiguous items stay near 0.20.
            # is_vest guard: only skip the floor when the garment IS genuinely classified as
            # outerwear/accessories/vest — NOT when CLIP already returned category="top"
            # (cable-knit sweater polos can trigger is_vest but are unambiguously tops).
            _vest_blocks_floor = (
                structure.get("is_vest") and results["category"] != "top"
            )
            if (results["category_confidence"] < 0.50
                    and not structure.get("is_bottom_garment")
                    and not _vest_blocks_floor
                    and structure.get("structure_type") not in ["strap_based", "crop_or_halter"]):
                subcat_conf = results.get("subcategory_confidence", 0.0)
                if subcat_conf >= 0.35:
                    # Subcategory is confident → category is correct; derive a proportional floor
                    derived_floor = min(round(subcat_conf * 0.80, 4), 0.85)
                    if results["category_confidence"] < derived_floor:
                        print(f"⬆️ Confidence floor (subcategory-derived): "
                              f"{results['category_confidence']:.2f} → {derived_floor:.2f}")
                        results["category_confidence"] = derived_floor
                elif results["category_confidence"] < 0.20:
                    print(f"⬆️ Applying confidence floor: {results['category_confidence']:.2f} → 0.20")
                    results["category_confidence"] = 0.20

            # === CONFIDENCE WARNING (evaluated after all overrides) ===
            # Hard deterministic check against final confidence values.
            # Threshold: either score below 0.50 warrants a warning so the DB
            # never silently stores a low-confidence classification.
            category_conf = results.get("category_confidence", 0.0)
            subcategory_conf = results.get("subcategory_confidence", 0.0)
            results["low_confidence_warning"] = (
                category_conf < 0.50 or subcategory_conf < 0.50
            )
            if results["low_confidence_warning"]:
                print(f"⚠️ LOW CONFIDENCE: category={category_conf:.1%}, "
                      f"subcategory={subcategory_conf:.1%}")

            return results
        
        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "category": "uncategorized",
                "category_confidence": 0.0,
                "subcategory": "unknown",
                "subcategory_confidence": 0.0,
                "formality": "casual",
                "formality_confidence": 0.0,
                "season": "all",
                "low_confidence_warning": True,
                "error": str(e)
            }

    def _detect_season_improved(self, category: str, subcategory: str,
                                color_palette: Dict = None, pattern: str = None) -> str:
        """
        Improved season detection
        """
        # Traditional garments from tropical/subtropical regions are worn year-round
        if category == "traditional":
            return "all"

        # Summer items
        summer_items = {
            "tank top", "ribbed tank", "graphic tank", "muscle tank", "longline tank",
            "spaghetti strap top", "racerback tank",
            "halter top", "crop top", "cami", "tube top", "off-shoulder top", "one-shoulder top",
            "corset top", "shorts", "culottes", "sandals",
            "mini dress", "sundress", "milkmaid dress",
            "linen shirt", "cuban collar shirt", "sleeveless polo",
        }
        if subcategory in summer_items:
            return "summer"

        # Winter items
        winter_items = {
            "coat", "sweater", "crew neck sweater", "v-neck sweater", "turtleneck",
            "ribbed turtleneck", "oversized knit", "hoodie", "oversized hoodie",
            "crewneck sweatshirt", "half-zip sweatshirt", "fleece jacket", "boots",
        }
        if subcategory in winter_items:
            return "winter"

        # Transitional items
        transition_items = {
            "jacket", "blazer", "cardigan", "buttoned cardigan", "knit vest",
            "windbreaker", "varsity jacket", "coach jacket", "bomber jacket", "denim jacket",
            "gilet", "waistcoat",
        }
        if subcategory in transition_items:
            return "spring,fall"

        # All-season basics — checked BEFORE color-based detection so that a black or
        # maroon graphic tee (which would hit "winter" via color) still returns "all".
        basic_items = {
            "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
            "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
            "shirt", "oxford shirt", "Flannel/checkered shirt", "denim shirt", "oversized shirt",
            "cropped button-up", "graphic print shirt", "tie-dye shirt", "vintage shirt", "sheer shirt",
            "blouse", "classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
            "striped polo", "long sleeve polo",
            "zip hoodie", "graphic hoodie", "cropped hoodie", "sleeveless hoodie",
            "bodysuit", "wrap top", "peplum top",
            "jeans", "pants", "leggings", "skirt",
            "sneakers", "loafers", "heels",
        }
        if subcategory in basic_items:
            return "all"

        # Color-based season hints
        if color_palette:
            colors = [c["name"] for c in color_palette.get("colors", [])]

            # Summer colors
            summer_colors = ["bright yellow", "bright orange", "turquoise", "hot pink",
                           "lime green", "coral", "white"]
            if any(sc in " ".join(colors) for sc in summer_colors):
                # Only override if subcategory allows it
                if subcategory not in winter_items:
                    return "summer"

            # Fall colors (like your mustard jacket!)
            fall_colors = ["rust", "mustard", "ochre", "olive", "brown", "terracotta",
                          "camel", "tan", "burgundy"]
            if any(fc in " ".join(colors) for fc in fall_colors):
                if subcategory not in summer_items:
                    return "fall"

            # Winter colors
            winter_colors = ["charcoal", "dark gray", "navy", "maroon", "forest green"]
            if any(wc in " ".join(colors) for wc in winter_colors):
                if subcategory not in summer_items:
                    return "winter"

            # Spring colors
            spring_colors = ["pale pink", "mint", "lavender", "cream", "light blue", "pastel"]
            if any(spc in " ".join(colors) for spc in spring_colors):
                return "spring"

        # Pattern hints
        if pattern:
            if "tropical" in pattern or "floral" in pattern:
                return "spring,summer"
    
        return "all"
