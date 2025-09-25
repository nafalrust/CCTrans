#!/usr/bin/env python3
"""
Test script to validate the predict.py fixes without needing a trained model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from Networks import ALTGVT

# Create same transform as in predict.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def test_model_shapes():

    print("🧪 Testing model shape compatibility...")
    
    # Create model (without loading pretrained weights)
    model = ALTGVT.alt_gvt_large(pretrained=False)
    model.eval()
    
    # Test different input sizes that previously caused errors
    test_sizes = [
        (256, 256),
        (512, 384),  # Non-square
        (640, 480),
        (768, 512),
        (400, 300),  # Arbitrary size
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for h, w in test_sizes:
        print(f"\n📏 Testing input size: {h}x{w}")
        
        # Create dummy image
        img = torch.randn(1, 3, h, w).to(device)
        
        # Test the padding logic from predict.py
        ws = 8
        patch_size = 4
        
        # Same logic as in predict.py
        target_h = ((h + patch_size - 1) // patch_size) * patch_size
        target_w = ((w + patch_size - 1) // patch_size) * patch_size
        
        patch_h, patch_w = target_h // patch_size, target_w // patch_size
        
        if patch_h % ws != 0:
            patch_h = ((patch_h + ws - 1) // ws) * ws
        if patch_w % ws != 0:
            patch_w = ((patch_w + ws - 1) // ws) * ws
            
        target_h = patch_h * patch_size
        target_w = patch_w * patch_size
        
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            print(f"🔥 Padding from {h}x{w} to {target_h}x{target_w}")
            img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            img_padded = img
            print(f"✅ No padding needed")
        
        try:
            # Test forward pass
            with torch.no_grad():
                output, _ = model(img_padded)
            
            print(f"✅ Success! Input: {img_padded.shape}, Output: {output.shape}")
            
            # Verify output can be properly cropped back
            output_h = h // 8
            output_w = w // 8
            cropped_output = output[:, :, :output_h, :output_w]
            print(f"✅ Cropped output: {cropped_output.shape}")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            return False
    
    print("\n🎉 All shape tests passed!")
    return True

def test_predict_function():

    print("\n🧪 Testing predict function...")
    
    # Import after we know the shapes work
    from predict import predict_image, load_model
    
    try:
        # Create a dummy model for testing
        model = ALTGVT.alt_gvt_large(pretrained=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Test with example images
        example_images = [
            "example_images/1.png",
            "example_images/2.png", 
            "example_images/3.png"
        ]
        
        for img_path in example_images:
            try:
                print(f"\n🖼️  Testing with {img_path}")
                count = predict_image(img_path, model, device, crop_size=256, batch_size=2, ws=8)
                print(f"✅ Prediction successful: {count:.2f}")
            except FileNotFoundError:
                print(f"⚠️  Image not found: {img_path}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CCTransformer prediction fix validation...")
    
    # Test 1: Shape compatibility
    shapes_ok = test_model_shapes()
    
    if shapes_ok:
        print("\n" + "="*50)
        # Test 2: Predict function
        predict_ok = test_predict_function()
        
        if predict_ok:
            print("\n🎉 All tests passed! The shape mismatch issue should be fixed.")
        else:
            print("\n⚠️  Predict function needs additional fixes.")
    else:
        print("\n❌ Shape compatibility issues still exist.")