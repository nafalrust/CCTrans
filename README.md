# CCTrans: Crowd Counting with Transformer

This repository contains a PyTorch implementation of **CCTrans** for crowd counting tasks in smart city applications. Originally developed for the Hology UB 8.0 Data Mining competition by team **Mistar Gawang** from Universitas Gadjah Mada.

📄 **Paper**: [CCTrans: Simplifying and Improving Crowd Counting via Spatial Channel-wise Transformer](https://arxiv.org/pdf/2109.14483.pdf) by Tian et al. (2021)

## 🎯 Key Features

- **Pyramid Vision Transformer (Twins-SVT) Backbone**: Combines local and global attention mechanisms
- **Pyramid Feature Aggregation (PFA)**: Merges multi-level features for better representation
- **Multi-scale Dilated Convolution (MDC)**: Regression head with various dilation rates
- **Advanced Loss Functions**: Combines L1, Optimal Transport, and L2 losses for robust training
- **Patch-based Inference**: Handles large images with overlapping crop processing

## 🏗️ Model Architecture

CCTrans revolutionizes crowd counting by leveraging Vision Transformer capabilities:

### 1. **Input Processing & Patch Embedding**
- Images are divided into **K×K×3** patches
- Each patch is flattened and projected to embedding vectors
- Results in **1D sequence** representation of the image

### 2. **Backbone: Pyramid Vision Transformer (Twins-SVT)**
- **Local Self-Attention (LSA)**: Focuses on local details within small windows
- **Global Sub-sampled Attention (GSA)**: Captures global context across windows
- **Multi-level Features**: Extracts hierarchical features from shallow to deep layers

### 3. **Pyramid Feature Aggregation (PFA)**
- Reshapes transformer features back to 2D feature maps
- Upsamples all feature maps to uniform resolution (1/8 of input)
- Combines features via element-wise addition for spatial+semantic fusion

### 4. **Regression Head: Multi-scale Dilated Convolution (MDC)**
- Parallel branches with different kernel sizes and dilation rates (1, 2, 3)
- Prevents gridding artifacts common in stacked dilated convolutions
- Final 1×1 convolution produces density map for crowd counting

### 5. **Loss Function Architecture**
The model uses a sophisticated loss combination:

$$L_d = L_1(P, G) + \lambda_1 L_{OT} + \lambda_2 L_2(D, D')$$

Where:
- **L1 Loss**: Total person count accuracy  
- **Optimal Transport Loss**: Minimizes distribution differences
- **L2 Loss**: Smoothed density map consistency (Gaussian kernel)

## 📊 Performance Results

Competition results (Hology UB 8.0 dataset):

| Metric | Value |
|--------|--------|
| Best MAE | 16.28 |
| Best Epoch | 200 |
| Training Time | ~300 epochs |

Standard benchmark (ShanghaiTech Part A):

| Method    | MAE   | MSE   |
|-----------|-------|-------|
| Paper     | 54.8  | 86.6  |
| This Code | 54.20 | 88.97 |

## 🏗️ Project Structure

```
CCTrans/
├── Networks/           # Model architectures
│   └── ALTGVT.py      # Main transformer model
├── datasets/          # Data loading utilities
├── losses/           # Loss functions (OT loss, etc.)
├── utils/            # Helper functions
├── preprocess/       # Data preprocessing scripts
├── train.py         # Training script
├── predict.py       # Inference script
├── vis_densityMap.py # Visualization tools
└── example_images/  # Sample test images
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
- Download ShanghaiTech dataset
- Update data path in `train.py`:
```python
--data-dir /path/to/your/dataset
```

### 3. Download Pretrained Weights
- **ImageNet Pretrained**: [Google Drive](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view)
- **Trained Model**: [Baidu Drive](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ) (code: `se59`)

### 4. Training
```bash
python train.py --dataset sha --batch-size 8 --lr 1e-5
```

Optional wandb logging:
```bash
python train.py --wandb 1 --run-name experiment_name
```

### 5. Inference
```bash
python predict.py --model-path checkpoints/model.pth --input image.jpg
```

For batch processing:
```bash
python predict.py --model-path checkpoints/model.pth --input /path/to/images/ --output-csv results.csv
```

## 🔍 Model Architecture

The model uses ALTGVT (Alternative Group Vision Transformer) as backbone:
- **Patch Embedding**: 4×4 patches with 128/256/512/1024 dimensions
- **Group Attention**: Local attention within 7×7 or 8×8 windows
- **Global Attention**: Sparse attention with spatial reduction
- **Regression Head**: Multi-scale feature fusion for density prediction

## 📈 Training Details

### Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5 with cosine annealing
- **Batch Size**: 8-16
- **Crop Size**: 256×256 (SHA), 512×512 (QNRF)
- **OT Loss Weight**: 0.1
- **TV Loss Weight**: 0.01

### Supported Datasets
- ShanghaiTech Part A & B
- UCF-QNRF
- NWPU-Crowd
- Custom datasets (see `datasets/crowd.py`)

## 🎨 Visualization

Generate density maps:
```bash
python vis_densityMap.py --image_path image.jpg --weight_path model.pth
```

Output will be saved to `./vis/` with:
- `pred_map.png`: Predicted density map
- `gt_dmap.png`: Ground truth density map

## 🔧 Advanced Usage

### Custom Dataset Training
1. Prepare your dataset following the structure in `datasets/crowd.py`
2. Create train/val split files
3. Run: `python train.py --dataset custom`

### Model Evaluation
```bash
python test_accuracy_fix.py  # Test model accuracy
python test_predict_fix.py   # Test prediction pipeline
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{liang2021cctrans,
  title={CCTrans: Simplifying and Improving Crowd Counting via Spatial Channel-wise Transformer},
  author={Liang, Dingkang and Xu, Wei and Zhu, Yingying and Zhou, Yu},
  journal={arXiv preprint arXiv:2109.14483},
  year={2021}
}
```

## 🤝 Acknowledgments

- Based on [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) framework
- Transformer architecture inspired by PVT and Twins-SVT
- Optimal transport implementation adapted from POT library

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🐛 Issues & Support

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include error logs and environment details

---

*For more implementation details, please refer to the original paper and code comments.*
	


