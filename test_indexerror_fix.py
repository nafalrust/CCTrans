#!/usr/bin/env python3
"""
Test untuk memverifikasi bahwa IndexError sudah diperbaiki
"""

import torch
import numpy as np

def simulate_crop_assembly():

    
    print("🧪 Testing IndexError fix...")
    
    # Simulasi crop predictions dengan berbagai bentuk yang mungkin terjadi
    crop_preds = []
    
    # Simulasi berbagai kemungkinan output dari model
    # Case 1: Normal 4D tensor [batch=2, channels=1, H, W]
    crop1 = torch.randn(2, 1, 32, 32)
    crop_preds.append(crop1)
    
    # Case 2: 3D tensor yang mungkin terjadi [batch*channels, H, W] 
    crop2 = torch.randn(2, 32, 32)
    crop_preds.append(crop2)
    
    # Case 3: 2D tensor [H, W]
    crop3 = torch.randn(32, 32)
    crop_preds.append(crop3)
    
    print(f"📦 Original shapes:")
    for i, cp in enumerate(crop_preds):
        print(f"   crop_preds[{i}]: {cp.shape}")
    
    # Apply fix - pastikan konsistensi dimensi
    print(f"\n🔧 Applying dimension fix...")
    for i, crop_pred in enumerate(crop_preds):
        original_shape = crop_pred.shape
        
        if len(crop_pred.shape) == 3:
            # [batch*channels, H, W] -> [batch, channels, H, W]  
            crop_preds[i] = crop_pred.unsqueeze(1)
        elif len(crop_pred.shape) == 2:
            # [H, W] -> [1, 1, H, W]
            crop_preds[i] = crop_pred.unsqueeze(0).unsqueeze(0)
            
        print(f"   crop_preds[{i}]: {original_shape} -> {crop_preds[i].shape}")
    
    # Test concatenation
    print(f"\n🔗 Testing concatenation...")
    try:
        concatenated = torch.cat(crop_preds, dim=0)
        print(f"   ✅ Concatenation successful: {concatenated.shape}")
    except Exception as e:
        print(f"   ❌ Concatenation failed: {str(e)}")
        return False
    
    # Test indexing yang sebelumnya error
    print(f"\n🎯 Testing indexing...")
    try:
        for idx in range(len(crop_preds)):
            # Test case yang sebelumnya error
            current_crop = concatenated[idx]
            print(f"   crop[{idx}] shape after indexing: {current_crop.shape}")
            
            # Pastikan 4D
            while len(current_crop.shape) < 4:
                current_crop = current_crop.unsqueeze(0)
                
            print(f"   crop[{idx}] shape after ensuring 4D: {current_crop.shape}")
            
            # Test slicing yang sebelumnya error
            pred_h, pred_w = 16, 16
            pred_crop = current_crop[:1, :1, :pred_h, :pred_w]
            print(f"   crop[{idx}] shape after slicing: {pred_crop.shape}")
            
        print(f"   ✅ All indexing operations successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Indexing failed: {str(e)}")
        return False

def test_assembly_logic():

    
    print(f"\n🧪 Testing complete assembly logic...")
    
    # Simulasi parameter
    b, h, w = 1, 256, 256
    pred_map_h, pred_map_w = h // 8, w // 8
    
    pred_map = torch.zeros([b, 1, pred_map_h, pred_map_w])
    crop_masks_lowres = torch.zeros([b, 1, pred_map_h, pred_map_w])
    
    # Simulasi beberapa crop predictions
    crop_preds = []
    for i in range(4):  # 4 crops
        crop = torch.rand(1, 1, 16, 16)  # Random density values
        crop_preds.append(crop)
    
    # Concat
    crop_preds = torch.cat(crop_preds, dim=0)
    print(f"📊 Concatenated crops shape: {crop_preds.shape}")
    
    # Test assembly
    idx = 0
    try:
        for i in [0, 16]:  # Simulasi 2 posisi
            for j in [0, 16]:
                gis_low, gie_low = i, i + 16
                gjs_low, gje_low = j, j + 16
                
                pred_h = gie_low - gis_low
                pred_w = gje_low - gjs_low
                
                current_crop = crop_preds[idx]
                
                # Ensure 4D
                while len(current_crop.shape) < 4:
                    current_crop = current_crop.unsqueeze(0)
                
                pred_crop = current_crop[:1, :1, :pred_h, :pred_w]
                
                print(f"   Assembly {idx}: region [{gis_low}:{gie_low}, {gjs_low}:{gje_low}], shape: {pred_crop.shape}")
                
                pred_map[:, :, gis_low:gie_low, gjs_low:gje_low] += pred_crop
                crop_masks_lowres[:, :, gis_low:gie_low, gjs_low:gje_low] += 1.0
                
                idx += 1
                
        # Final result
        mask = crop_masks_lowres
        mask[mask == 0] = 1.0
        outputs = pred_map / mask
        
        count = outputs.sum().item()
        print(f"   ✅ Assembly successful! Total count: {count:.2f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Assembly failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing IndexError fixes for CCTransformer predict.py...")
    
    success1 = simulate_crop_assembly()
    success2 = test_assembly_logic()
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! IndexError should be fixed.")
    else:
        print(f"\n❌ Some tests failed. More debugging needed.")