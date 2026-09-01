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
                "a photo of a t-shirt or graphic tee",
                "a photo of a polo shirt with a collar",
                "a photo of a button-up or Oxford shirt",
                "a photo of a hoodie or sweatshirt",
                "a photo of a sleeveless tank top or vest",
                "a photo of a knit sweater or cardigan",
                "a photo of a blouse or women's top",
                "a photo of a traditional tunic or kaftan worn as a top",
            ],
            "dress": [
                "a photo of a dress",
                "a photo of a mini dress",
                "a photo of a sundress",
                "a photo of a cocktail dress",
                "a photo of a milkmaid dress"
            ],
            "bottom": [
                "a photo of pants",
                "a photo of jeans",
                "a photo of shorts",
                "a photo of a skirt",
                "a photo of trousers",
                "a photo of sweatpants",
                "a photo of joggers or track pants",
                "a photo of wide-leg baggy pants"
            ],
            "shoes": [
                "a photo of shoes",
                "a photo of sneakers",
                "a photo of boots",
                "a photo of sandals",
                "a photo of heels"
            ],
            "outerwear": [
                "a photo of a jacket worn over clothes",
                "a photo of a coat worn over clothes",
                "a photo of a blazer worn over a shirt",
                "a photo of outerwear for cold weather",
                "a photo of a bomber jacket",
                "a photo of a leather jacket",
                "a photo of a zip-up jacket with elasticized cuffs and hem"
            ],

            "traditional": [
                "a photo of traditional African clothing",
                "a photo of a Nigerian Agbada with wide flowing sleeves and chest embroidery",
                "a photo of an African kaftan or grand boubou robe",
                "a photo of cultural ethnic wear",
                "a photo of traditional formal ceremonial attire",
                "a photo of a flowing robe or dashiki worn at a ceremony"
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

            # ── T-SHIRTS ─────────────────────────────────────────────────────
            "t-shirt":           "a plain crew neck short-sleeve t-shirt with no graphics and no collar",
            "v-neck tee":        "a t-shirt with a V-shaped neckline cut",
            "graphic print tee": "a t-shirt with a large graphic, illustration, or bold text printed on the front",
            "vintage tee":       "a faded distressed vintage-wash t-shirt with washed-out or bleached colors",
            "oversized tee":     "an extremely baggy boxy oversized t-shirt hanging loosely off the shoulders",
            "cropped tee":       "a short cropped t-shirt cut at or above the navel showing the midriff",
            "longline tee":      "a long-body t-shirt extending well past the waist toward the thighs",
            "pocket tee":        "a plain t-shirt with a single small patch pocket on the left chest",
            "striped t-shirt":   "a t-shirt with bold horizontal or vertical color stripes across the body",
            "tie-dye tee":       "a t-shirt with a colorful tie-dye swirl, spiral, or burst pattern",

            # ── POLO SHIRTS ──────────────────────────────────────────────────
            "classic buttoned polo": "a classic polo shirt with a ribbed fold-over collar and only 2-3 buttons on a SHORT top placket — no chest pocket, placket does NOT run the full length of the shirt",
            "sweater polo":      "a knit polo sweater with a ribbed collar and optional half-zip",
            "polo half-zip":     "a polo shirt with a short half-zipper running down the collar instead of buttons",
            "open v-neck polo":  "a polo-style shirt with an open V-collar and no buttons or zip",
            "sleeveless polo":   "a sleeveless polo shirt with a collar but no sleeves — like a golf vest top",
            "striped polo":      "a polo shirt with horizontal or vertical color stripes across the body",
            "long sleeve polo":  "a polo shirt with full-length long sleeves reaching the wrists",

            # ── SHIRTS / BUTTON-UPS ──────────────────────────────────────────
            "shirt":             "a collared button-up or button-down dress shirt",
            "oxford shirt":      "an Oxford cloth button-down shirt with a dress shirt collar and a FULL-LENGTH row of buttons running from collar all the way to the hem — may have a chest pocket — can be long-sleeve or short-sleeve",
            "graphic print shirt": "a button-up shirt covered in a bold all-over printed pattern or graphic illustration",
            "Flannel/checkered shirt": "a plaid or checkered flannel button-up shirt — long-sleeve, short-sleeve, or cropped; fabric has a bold grid-pattern woven check or plaid (tartan); also called a lumberjack shirt or flannel shirt",
            "linen shirt":       "a lightweight relaxed-fit linen button-up shirt with visible texture",
            "denim shirt":       "a denim chambray or denim fabric button-up shirt",
            "oversized shirt":   "a very oversized baggy button-up shirt worn loose and untucked",
            "cropped button-up": "a short cropped button-up shirt cut or tied at the waist",
            "cuban collar shirt":"a Cuban collar or camp collar short-sleeve shirt with an open notched lapel collar",
            "tie-dye shirt":     "a button-up shirt with tie-dye or batik-dye color swirl patterns",
            "vintage shirt":     "a vintage or retro bowling-style short-sleeve button-up shirt",
            "sheer shirt":       "a sheer or semi-transparent chiffon or organza button-up shirt",
            "blouse":            "a women's blouse or dressy top",

            # ── HOODIES & SWEATSHIRTS ────────────────────────────────────────
            "hoodie":            "a plain pullover hooded sweatshirt with a large front kangaroo pocket",
            "zip hoodie":        "a zip-up hoodie with a LARGE FABRIC HOOD and a full-length front zipper running from collar all the way to the hem — the zipper opens the entire front of the garment",
            "graphic hoodie":    "a hoodie with a large graphic print, logo, or bold text screen-printed on the front",
            "cropped hoodie":    "a short cropped hoodie cut above the waist",
            "oversized hoodie":  "an extremely oversized loose-fitting boxy hoodie",
            "crewneck sweatshirt": "a plain crewneck sweatshirt with no hood, ribbed cuffs, and a ribbed hem",
            "half-zip sweatshirt": "a sweatshirt with a short half-zip at the collar and no hood",
            "sleeveless hoodie": "a sleeveless hoodie vest with a hood and no sleeves — a vest with a hood",

            # ── TANK TOPS & VESTS ────────────────────────────────────────────
            "tank top":          "a plain sleeveless tank top with wide shoulder straps",
            "ribbed tank":       "a fitted ribbed-knit sleeveless tank top",
            "graphic tank":      "a tank top or muscle shirt with a printed graphic or text on the front",
            "muscle tank":       "a muscle-fit sleeveless workout tank with very wide cutout armholes",
            "longline tank":     "a long-body tank top extending past the hips",
            "spaghetti strap top": "a top with extremely thin spaghetti-width shoulder straps",
            "racerback tank":    "a racerback athletic tank top with a T-back or Y-back strap configuration",

            # ── WOMEN'S TOPS ─────────────────────────────────────────────────
            "crop top":          "a cropped midriff-baring women's top cut above the waist",
            "halter top":        "a halter neck top with straps that tie or fasten behind the neck",
            "cami":              "a camisole — a lightweight short top with thin straps and no collar",
            "tube top":          "a strapless bandeau tube top with no shoulder straps",
            "off-shoulder top":  "a top that sits below both shoulders, fully exposing the shoulders and collarbone",
            "one-shoulder top":  "an asymmetric top with one strap on one shoulder and one bare shoulder",
            "corset top":        "a structured boned corset-style top with lace-up back or busk front closure",
            "wrap top":          "a wrap-style top that crosses over the chest and ties at the waist",
            "peplum top":        "a fitted top with a flared peplum ruffle hem at the waist",
            "bodysuit":          "a fitted bodysuit that snaps at the crotch — like a one-piece leotard top",

            # ── KNITS & SWEATERS ─────────────────────────────────────────────
            "sweater":           "a plain knit sweater or pullover",
            "crew neck sweater": "a classic crew neck knit sweater with a round neckline",
            "v-neck sweater":    "a V-neck knit sweater with a V-shaped neckline",
            "turtleneck":        "a turtleneck sweater with a high folded collar that covers the neck",
            "ribbed turtleneck": "a ribbed-knit turtleneck sweater with visible horizontal rib texture",
            "oversized knit":    "an oversized chunky cable-knit or textured knit sweater with a relaxed silhouette",
            "cardigan":          "an open-front cardigan sweater with no buttons — draped open at the front",
            "buttoned cardigan": "a cardigan sweater with a full column of buttons running down the front",
            "knit vest":         "a sleeveless knit sweater vest — a pullover with no sleeves",

            # ── SPORTS / JERSEYS ─────────────────────────────────────────────
            "jersey":            "a sports jersey or football jersey with a number on the back",

            # ── DRESSES ──────────────────────────────────────────────────────
            "dress":             "a full dress",
            "mini dress":        "a short mini dress ending above mid-thigh",
            "sundress":          "a casual lightweight sundress",
            "milkmaid dress":    "a milkmaid or cottagecore dress with puff sleeves and a square neckline",
            "cocktail dress":    "a semi-formal cocktail dress",

            # ── TRADITIONAL / CULTURAL ───────────────────────────────────────
            "agbada":            "a Nigerian Agbada — voluminous outer robe with wide draped sleeves and heavy chest embroidery",
            "kaftan":            "a traditional kaftan or grand boubou robe with wide sleeves",
            "embroidered kaftan":"a kaftan or robe with ornate embroidery at the neckline and chest",
            "ankara top":        "an Ankara wax print top with bold colorful geometric or floral African print fabric",
            "dashiki":           "a traditional African dashiki pullover shirt with embroidered neckline",
            "kurta":             "a traditional South Asian kurta tunic",
            "senator wear":      "traditional Nigerian senator wear — two-piece set with embroidered neckline",
            "thobe":             "a traditional Middle Eastern thobe or ankle-length robe",
            "buba":              "a traditional West African buba top — a loose hip-length blouse in wax print or plain fabric",

            # ── BOTTOMS ──────────────────────────────────────────────────────
            "jeans":             "blue denim jeans",
            "pants":             "dress pants or trousers",
            "sweatpants":        "baggy fleece sweatpants with an elastic waistband and drawstring",
            "joggers":           "tapered jogger pants with an elastic waistband and ribbed cuffs",
            "shorts":            "casual shorts",
            "culottes":          "wide-leg culottes or long shorts",
            "skirt":             "a skirt",
            "leggings":          "athletic leggings or tights",

            # ── SHOES ────────────────────────────────────────────────────────
            "sneakers":          "athletic sneakers or trainers",
            "boots":             "leather or fashion boots",
            "sandals":           "open-toe sandals",
            "heels":             "high heeled shoes",
            "loafers":           "slip-on loafer dress shoes",

            # ── OUTERWEAR ────────────────────────────────────────────────────
            "jacket":            "a casual jacket",
            "bomber jacket":     "a bomber jacket with elasticized cuffs, hem, and a front zipper",
            "leather jacket":    "a leather or faux-leather jacket with a zipper",
            "denim jacket":      "a denim trucker jacket with button front closure",
            "windbreaker":       "a lightweight nylon or polyester windbreaker jacket",
            "varsity jacket":    "a varsity or letterman jacket with contrast-color wool body and leather sleeves",
            "coach jacket":      "a lightweight coach jacket or track jacket with zip front and minimal branding",
            "fleece jacket":     "a polar fleece zip-up jacket",
            "coat":              "a long heavy winter coat",
            "blazer":            "a structured formal blazer or suit jacket",
            "gilet":             "a quilted or padded sleeveless gilet or vest",
            "waistcoat":         "a formal waistcoat worn over a dress shirt",

            # ── ACCESSORIES ──────────────────────────────────────────────────
            "hat":               "a hat or cap",
            "bag":               "a handbag or backpack",
            "scarf":             "a scarf or shawl",
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
            _clip_id = "openai/clip-vit-base-patch32"
            _kwargs = dict(local_files_only=True)
            try:
                self.model = CLIPModel.from_pretrained(_clip_id, **_kwargs)
                self.processor = CLIPProcessor.from_pretrained(_clip_id, **_kwargs)
            except Exception:
                # Not cached yet — download once, then all future runs are offline
                print("📥 CLIP not cached — downloading (one-time only)...")
                self.model = CLIPModel.from_pretrained(_clip_id)
                self.processor = CLIPProcessor.from_pretrained(_clip_id)
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
            "top": [
                # T-shirts
                "t-shirt", "v-neck tee", "graphic print tee", "vintage tee", "oversized tee",
                "cropped tee", "longline tee", "pocket tee", "striped t-shirt", "tie-dye tee",
                # Polo shirts
                "classic buttoned polo", "sweater polo", "polo half-zip", "open v-neck polo",
                "sleeveless polo", "striped polo", "long sleeve polo",
                # Shirts / button-ups
                "shirt", "oxford shirt", "graphic print shirt", "Flannel/checkered shirt", "linen shirt",
                "denim shirt", "oversized shirt", "cropped button-up", "cuban collar shirt",
                "tie-dye shirt", "vintage shirt", "sheer shirt", "blouse",
                # Hoodies & sweatshirts
                "hoodie", "zip hoodie", "graphic hoodie", "cropped hoodie", "oversized hoodie",
                "crewneck sweatshirt", "half-zip sweatshirt", "sleeveless hoodie",
                # Tanks & vests
                "tank top", "ribbed tank", "graphic tank", "muscle tank",
                "longline tank", "spaghetti strap top", "racerback tank",
                # Women's tops
                "crop top", "halter top", "cami", "tube top", "off-shoulder top",
                "one-shoulder top", "corset top", "wrap top", "peplum top", "bodysuit",
                # Knits & sweaters
                "sweater", "crew neck sweater", "v-neck sweater", "turtleneck",
                "ribbed turtleneck", "oversized knit", "cardigan", "buttoned cardigan", "knit vest",
                # Sports
                "jersey",
            ],
            "dress": [
                "dress", "mini dress", "sundress", "milkmaid dress", "cocktail dress",
            ],
            "bottom": [
                "jeans", "pants", "sweatpants", "joggers", "shorts", "culottes", "skirt", "leggings",
            ],
            "shoes": [
                "sneakers", "boots", "sandals", "heels", "loafers",
            ],
            "outerwear": [
                "jacket", "bomber jacket", "leather jacket", "denim jacket",
                "windbreaker", "varsity jacket", "coach jacket", "fleece jacket",
                "coat", "blazer", "gilet", "waistcoat",
            ],
            "accessories": [
                "hat", "bag", "scarf",
            ],
            "traditional": [
                "agbada", "kaftan", "embroidered kaftan", "ankara top",
                "dashiki", "kurta", "senator wear", "thobe", "buba",
            ],
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
