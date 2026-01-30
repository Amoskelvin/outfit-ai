import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.inference.classifier import ClothingClassifier
from src.ml.preprocessing.image_processor import ClothingImageProcessor
import json

def test_classifier():
    """Test the clothing classifier on an uploaded image"""
    
    # Initialize
    print("Initializing classifier...")
    classifier = ClothingClassifier()
    processor = ClothingImageProcessor()
    
    # Test on an existing image 
    test_image_path = "data\\processed\\1\\1_43de3a52-4758-4d02-9a71-facc2b4c886b.png" 
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        print("Please upload an image via the API first, then update the path above")
        return
    
    print(f"\nProcessing image: {test_image_path}")
    
    # Process image
    processed = processor.process_image(test_image_path, save_processed=False)
    
    # Get the original image (not normalized)
    from src.utils.image_utils import ImageProcessor
    image = ImageProcessor.load_image(test_image_path)
    
    # Classify
    print("\nClassifying...")
    results = classifier.classify(image)
    
    # Pretty print results
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"\n📁 Category: {results['category']} (confidence: {results['category_confidence']:.2%})")
    print(f"🏷️  Subcategory: {results['subcategory']} (confidence: {results['subcategory_confidence']:.2%})")
    print(f"👔 Formality: {results['formality']} (confidence: {results['formality_confidence']:.2%})")
    print(f"🌤️  Season: {results['season']}")
    
    print("\n📊 All Category Scores:")
    for cat, score in results['all_category_scores'].items():
        print(f"  {cat}: {score:.2%}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_classifier()