#!/usr/bin/env python3
# Quick tensor dimension fix

def fix_tensor_indexing_issue():
    
    fix_code = '''
    # Fix tensor dimensions
    for i, crop_pred in enumerate(crop_preds):
        if len(crop_pred.shape) == 3:
            crop_preds[i] = crop_pred.unsqueeze(1)
        elif len(crop_pred.shape) == 2:
            crop_preds[i] = crop_pred.unsqueeze(0).unsqueeze(0)
    
    crop_preds = torch.cat(crop_preds, dim=0)
    current_crop = crop_preds[idx]
    
    while len(current_crop.shape) < 4:
        current_crop = current_crop.unsqueeze(0)
    
    pred_crop = current_crop[:1, :1, :pred_h, :pred_w]
    '''
    
    print("🔧 Fix code:")
    print(fix_code)

if __name__ == "__main__":
    fix_tensor_indexing_issue()