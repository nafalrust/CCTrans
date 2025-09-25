# CCTrans: Crowd Counting with Transformer
Implementation of CCTrans for crowd counting tasks.
Original paper: [Link](https://arxiv.org/pdf/2109.14483.pdf)

## Results
Results on ShanghaiTech Part A dataset: 

| Code      | MAE   | MSE      |
|-----------|-------|-------|
| PAPER     | 54.8  | 86.6  |
| This code | 54.20 | 88.97 |

Trained with batch-size=8 for ~1500 epochs. Best validation around epoch 606.

## Framework
Based on DM-Count: [link](https://github.com/cvlab-stonybrook/DM-Count)

## Training
1. Update data path in `train.py`
2. Download pretrained weights: [link](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view)
3. Optional: Enable wandb logging with `--wandb 1`
4. Run: `python train.py`

## Testing
Run: `python predict.py --model-path MODEL_PATH --input IMAGE_PATH`

Images are processed in 256x256 patches with overlap averaging.
Pretrained model: [Baidu Drive](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ) (code: se59)

## Visualization
Run: `python vis_densityMap.py`
Output saved to: `./vis/part_A_final`

## Requirements
See `requirements.txt`
	


