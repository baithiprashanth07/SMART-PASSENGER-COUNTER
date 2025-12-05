"""
Download ONNX models for the passenger counter system.
This script downloads:
1. YOLOv8n ONNX model for person detection
2. Face recognition model (optional)
"""

import os
import urllib.request
import sys


def download_file(url, destination, description):
    """Download a file with progress bar."""
    print(f"\n📥 Downloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {destination}")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r[{'=' * (percent // 2)}{' ' * (50 - percent // 2)}] {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✅ Downloaded {description} successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False


def export_yolov8_with_ultralytics():
    """Try to export YOLOv8n using ultralytics package."""
    print("\n🔧 Attempting to export YOLOv8n using Ultralytics...")
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8n model
        model = YOLO('yolov8n.pt')
        
        # Export to ONNX
        model.export(format='onnx', simplify=True)
        
        # Move to models directory
        import shutil
        if os.path.exists('yolov8n.onnx'):
            shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
            print("✅ YOLOv8n ONNX model exported successfully!")
            return True
    except ImportError:
        print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def main():
    """Main download function."""
    print("=" * 60)
    print("ONNX Model Downloader for Smart Passenger Counter")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Option 1: Try to export using ultralytics
    print("\n📦 Method 1: Export using Ultralytics")
    if export_yolov8_with_ultralytics():
        yolo_success = True
    else:
        # Option 2: Download pre-exported ONNX model
        print("\n📦 Method 2: Download pre-exported ONNX model")
        print("\n⚠️  Note: Pre-exported models may not be available from official sources.")
        print("You can:")
        print("1. Install ultralytics: pip install ultralytics")
        print("2. Run this script again to export the model")
        print("3. Or manually export using: yolo export model=yolov8n.pt format=onnx")
        
        # Try GitHub releases (community models)
        yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
        yolo_success = download_file(
            yolo_url,
            "models/yolov8n.onnx",
            "YOLOv8n ONNX model"
        )
    
    # Face recognition model (optional)
    print("\n" + "=" * 60)
    print("Face Recognition Model (Optional)")
    print("=" * 60)
    print("\n⚠️  Face recognition is optional and disabled by default.")
    print("If you want to enable unique person counting with ReID:")
    print("1. Download an ArcFace or FaceNet ONNX model")
    print("2. Place it in models/face_recognition.onnx")
    print("3. Set reid.enabled: true in config/config.yaml")
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    if yolo_success:
        print("✅ YOLOv8n ONNX model: Ready")
        print(f"   Size: {os.path.getsize('models/yolov8n.onnx') / 1024 / 1024:.2f} MB")
    else:
        print("❌ YOLOv8n ONNX model: Failed")
        print("\n📝 Manual steps to get YOLOv8n ONNX:")
        print("   1. Install ultralytics: pip install ultralytics")
        print("   2. Run: yolo export model=yolov8n.pt format=onnx")
        print("   3. Move yolov8n.onnx to models/")
    
    print("\n⚠️  Face recognition model: Skipped (optional)")
    
    if yolo_success:
        print("\n🎉 Setup complete! You can now run the passenger counter.")
        print("   Run: python main.py")
        print("   Or:  python server/api.py")
    else:
        print("\n⚠️  Please complete the manual steps above to finish setup.")


if __name__ == "__main__":
    main()
