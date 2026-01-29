import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.preprocessing.image_processor import ClothingImageProcessor

def test_image_processing():
    """Test image processing on a sample image"""
    processor = ClothingImageProcessor()
    
    # You'll need to update this path to an actual image
    test_image = "data\\processed\\1\\1_43de3a52-4758-4d02-9a71-facc2b4c886b.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please upload an image first via the API")
        return
    
    print(f"Processing image: {test_image}")
    
    try:
        result = processor.process_image(test_image)
        
        print("\n✅ Processing successful!")
        print(f"\nDominant color: {result['dominant_color']}")
        print(f"\nColor palette:")
        for i, color in enumerate(result['color_palette']['colors'], 1):
            print(f"  {i}. {color['name']} - {color['hex']}")
        
        print(f"\nProcessed image saved to: {result['processed_path']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_image_processing()