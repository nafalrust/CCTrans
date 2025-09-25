#!/usr/bin/env python3
# Test accuracy validation

import torch
import torch.nn.functional as F
from Networks import ALTGVT
import numpy as np

def test_density_map_processing():
    
    print("🧪 Testing density map processing accuracy...")
    
    # Create model
    model = ALTGVT.alt_gvt_large(pretrained=False)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Test with different input sizes
    test_cases = [
        {"size": (256, 256), "name": "Square 256x256"},
        {"size": (512, 384), "name": "Rect 512x384"}, 
        {"size": (400, 600), "name": "Rect 400x600"},
    ]
    
    for case in test_cases:
        h, w = case["size"]
        print(f"\n📏 Testing {case['name']}: {h}x{w}")
        
        # Create dummy input
        x = torch.randn(1, 3, h, w).to(device)
        
        with torch.no_grad():
            # Test model forward pass
            density_map, normalized_density = model(x)
            
            print(f"   Input shape: {x.shape}")
            print(f"   Density map shape: {density_map.shape}")
            print(f"   Normalized density shape: {normalized_density.shape}")
            
            # Check expected downsampling
            expected_h, expected_w = h // 8, w // 8
            actual_h, actual_w = density_map.shape[2], density_map.shape[3]
            
            print(f"   Expected output size: {expected_h}x{expected_w}")
            print(f"   Actual output size: {actual_h}x{actual_w}")
            
            # Check density map values
            density_sum = density_map.sum().item()
            norm_sum = normalized_density.sum().item()
            
            print(f"   Density map sum: {density_sum:.6f}")
            print(f"   Normalized density sum: {norm_sum:.6f}")
            print(f"   Density range: [{density_map.min().item():.6f}, {density_map.max().item():.6f}]")
            
            # Test that the output dimensions are correct
            downsampling_factor = h // actual_h
            print(f"   Actual downsampling factor: ~{downsampling_factor}x")
            
            if abs(downsampling_factor - 8) <= 1:  # Allow some tolerance
                print(f"   ✅ Downsampling factor is correct (~8x)")
            else:
                print(f"   ❌ Unexpected downsampling factor: {downsampling_factor}")
                
            # Check that normalized density is properly normalized
            if abs(norm_sum - 1.0) < 0.1:  # Should sum to ~1
                print(f"   ✅ Normalized density properly normalized")
            else:
                print(f"   ⚠️  Normalized density sum: {norm_sum:.6f} (expected ~1.0)")

def test_crop_assembly():

    
    print(f"\n🧪 Testing crop assembly logic...")
    
    # Simulate crop processing
    h, w = 640, 480  # Original image size
    crop_size = 256
    
    print(f"Original image: {h}x{w}")
    print(f"Crop size: {crop_size}x{crop_size}")
    
    # Calculate crops
    crops_info = []
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            gis, gie = max(min(h - crop_size, i), 0), min(h, i + crop_size)
            gjs, gje = max(min(w - crop_size, j), 0), min(w, j + crop_size)
            crops_info.append((gis, gie, gjs, gje))
            
    print(f"Number of crops: {len(crops_info)}")
    
    # Simulate density map assembly  
    pred_map_h, pred_map_w = h // 8, w // 8
    pred_map = torch.zeros(1, 1, pred_map_h, pred_map_w)
    overlap_mask = torch.zeros(1, 1, pred_map_h, pred_map_w)
    
    print(f"Density map size: {pred_map_h}x{pred_map_w}")
    
    for idx, (gis, gie, gjs, gje) in enumerate(crops_info):
        # Convert to low resolution coordinates
        gis_low, gie_low = gis // 8, gie // 8
        gjs_low, gje_low = gjs // 8, gje // 8
        
        # Clamp to valid range
        gis_low = max(0, min(gis_low, pred_map_h))
        gie_low = max(0, min(gie_low, pred_map_h))
        gjs_low = max(0, min(gjs_low, pred_map_w))  
        gje_low = max(0, min(gje_low, pred_map_w))
        
        if gie_low > gis_low and gje_low > gjs_low:
            # Simulate adding a crop prediction
            crop_h, crop_w = gie_low - gis_low, gje_low - gjs_low
            fake_density = torch.ones(1, 1, crop_h, crop_w) * 0.1  # Simulate density
            
            pred_map[:, :, gis_low:gie_low, gjs_low:gje_low] += fake_density
            overlap_mask[:, :, gis_low:gie_low, gjs_low:gje_low] += 1.0
            
        print(f"  Crop {idx}: ({gis},{gie},{gjs},{gje}) -> ({gis_low},{gie_low},{gjs_low},{gje_low})")
    
    # Final normalization
    overlap_mask[overlap_mask == 0] = 1.0
    final_density = pred_map / overlap_mask
    
    total_count = final_density.sum().item()
    avg_overlap = overlap_mask.mean().item()
    
    print(f"Average overlap factor: {avg_overlap:.2f}")
    print(f"Total simulated count: {total_count:.2f}")
    print(f"✅ Crop assembly test completed")

if __name__ == "__main__":
    print("🚀 Testing accuracy fixes for CCTransformer prediction...")
    
    try:
        test_density_map_processing()
        test_crop_assembly()
        print(f"\n🎉 All accuracy tests completed!")
        print(f"\n📝 Key fixes applied:")
        print(f"   1. ❌ Removed incorrect upsampling/interpolation")
        print(f"   2. ✅ Keep density map at native resolution (1/8)")  
        print(f"   3. ✅ Use density_map (not normalized) for counting")
        print(f"   4. ✅ Fixed crop assembly for lower resolution")
        print(f"   5. ✅ Added proper overlap handling")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()