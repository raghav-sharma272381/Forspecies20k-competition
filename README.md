# FOR-species20K Competition Solution 🥈

[![Rank](https://img.shields.io/badge/Rank-2nd_Worldwide-silver)](competition-link)
[![Competition](https://img.shields.io/badge/Competition-FOR--species20K-green)](https://doi.org/10.1111/2041-210X.14503)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

**Authors:**  Agamjot Kaur & Raghav Sharma

**Achievement:** 🥈 2nd Place Worldwide  
**Task:** Tree Species Classification from LiDAR Point Clouds

---

## 🎯 Competition Overview

The FOR-species20K competition challenges participants to classify individual tree species from LiDAR point cloud data. The dataset contains 20,158 high-quality, manually segmented individual tree point clouds featuring 33 species across 19 genera from multiple continents.

## 💡 Our Approach

![Ranking](https://github.com/raghav-sharma272381/Forspecies20k-competition/blob/main/thumbnail_image.png?raw=true)

### Key Innovation: Multi-View Rasterization

Instead of working directly with 3D point clouds, we developed a novel approach that rasterizes LiDAR data into **four complementary 2D projections**:

1. **Nadir View** (Top-down): Captures crown shape and canopy structure
2. **Façade View** (Front): Shows tree height and vertical structure
3. **Profile View** (Side): Reveals branching patterns
4. **Rear View** (Back): Provides additional structural context

These four views are combined into a single **2×2 grid** that serves as input to our CNN:

![Multi-View Grid Layout](https://github.com/raghav-sharma272381/Forspecies20k-competition/blob/main/grid.png?raw=true)

*Layout: Top row [Nadir | Façade], Bottom row [Rear | Profile]*

### Architecture

- **Base Model**: Identity-Mapped ResNet with Pre-Activation
- **Input**: Single-channel 256×256 grayscale grid
- **Network Depth**: 5 stages with moderate depth (2 blocks per stage)
- **Width Progression**: 16 → 32 → 64 → 128 → 256 → 512 channels
- **Regularization**: Batch Normalization + Dropout (0.1)
- **Output**: 33-way softmax classification

### Data Processing Pipeline



** Tree Views (Nadir, Façade, Profile, Rear):**
- Single-channel grayscale
- Point density with logarithmic enhancement
- X-Z plane projections

**Final Processing:**
- Convert to single-channel grayscale grid (256×256)
- Normalize: mean=0.10, std=0.21
- Apply augmentation and random erasing

## 🔧 Training Strategy

### Augmentation

**View-Specific Augmentation:**
- **Nadir**: Random rotation (360°) + translation (±10%)
- **Side Views**: Horizontal flip (50%) + translation (±10%)

**Combined Augmentation (on 2×2 grid):**
- TrivialAugmentWide for diversity
- Padding + Random Crop for scale invariance
- Random Erasing for robustness

### Optimization

- **Optimizer**: AdamW
- **Learning Rate**: 2e-2 with OneCycleLR scheduler
- **Batch Size**: 64
- **Epochs**: 72 (with ~252 steps per epoch)
- **Loss**: Cross-Entropy
- **Hardware**: Single NVIDIA P100 GPU with mixed precision

### Data Split

- **Training**: 16,193 samples
- **Validation**: 64 samples
- **Test**: 572 samples (hold-out)

## 📊 Results

Our approach achieved **2nd place worldwide** in the FOR-species20K competition, demonstrating that multi-view rasterization can effectively capture the 3D structural information needed for accurate tree species classification.

### Key Strengths

1. **Multi-Perspective Fusion**: Combines complementary views into unified representation
2. **Efficient 2D Processing**: Leverages mature CNN architectures instead of 3D networks
3. **Strong Augmentation**: Robust to natural variation in tree structure and scanning conditions
4. **Species-Specific Features**: Captures distinctive crown shapes, heights, and branching patterns

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/forspecies20k-solution.git
cd forspecies20k-solution

# Install dependencies
git clone https://github.com/fastai/course22p2.git
cd course22p2
pip install -e .
cd ..

# Install additional requirements
pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    fsspec==2023.9.0 \
    datasets
```

### Dataset Preparation

```python
# The notebook expects preprocessed 128×128 or 256×256 PNG images
# organized as: {treeID:05d}_{view}.png
# Example: 00001_nadir.png, 00001_facade.png, etc.

# Required directory structure:
preprocessed_128/
├── 00001_nadir.png
├── 00001_facade.png
├── 00001_profile.png
├── 00001_rear.png
├── ...
└── tree_metadata.csv  # Contains treeID and species labels
```

### Training

```python
# Load the notebook in Kaggle or Jupyter
# Adjust paths in the notebook to point to your data
# Run all cells to train the model

# Key hyperparameters (modify in notebook):
lr = 2e-2           # Learning rate
epochs = 72          # Training epochs
bs = 64             # Batch size
drop = 0.1          # Dropout rate
```



## 🔬 Technical Details

### Multi-View Grid Construction

```python
# Construct 2×2 grid from 4 views
top_row = torch.cat([nadir, facade], dim=2)      # Horizontal concatenation
bottom_row = torch.cat([rear, profile], dim=2)
grid = torch.cat([top_row, bottom_row], dim=1)   # Vertical concatenation
# Result: Single-channel 256×256 (or 512×512) image
```

### Model Architecture

```python
def get_dropmodel(
    nfs=(16, 32, 64, 128, 256, 512),  # Channel progression
    nbks=(2, 2, 2, 2, 2),              # Blocks per stage
    drop=0.1                            # Dropout rate
):
    layers = [
        nn.Conv2d(1, nfs[0], 5, padding=2),  # Initial conv
        *[res_blocks(nbks[i], nfs[i], nfs[i+1], stride=2)
          for i in range(len(nfs)-1)],       # ResNet stages
        GeneralReLU(leak=0.1, sub=0.4),
        nn.BatchNorm2d(nfs[-1]),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(drop),
        nn.Linear(nfs[-1], 33, bias=False),
        nn.BatchNorm1d(33)
    ]
    return nn.Sequential(*layers)
```

### Identity-Mapped ResBlock

```python
class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = nn.Sequential(
            # Pre-activation design
            nn.BatchNorm2d(ni), act(),
            nn.Conv2d(ni, nf, 3, padding=1),
            nn.BatchNorm2d(nf), act(),
            nn.Conv2d(nf, nf, 3, stride=stride, padding=1)
        )
        self.idconv = nn.Conv2d(ni, nf, 1) if ni != nf else nn.Identity()
        self.pool = nn.AvgPool2d(2) if stride == 2 else nn.Identity()
    
    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))
```


## 🎓 Lessons Learned

1. **Multi-view representations are powerful**: Combining multiple perspectives captures 3D structure better than single views
2. **Aggressive augmentation helps**: Trees exhibit high natural variation; strong augmentation improves generalization
3. **Moderate depth + wider channels**: Better than very deep networks for this dataset size
4. **Pre-activation ResNets**: Identity mapping + pre-activation provides better gradient flow
5. **OneCycleLR is effective**: Super-convergence achieves strong results in just 2 epochs


## 🤝 Acknowledgments

- FOR-species20K dataset creators and contributors
- Fast.ai community for the miniai framework
- Competition organizers and fellow participants



**⭐ If you find this work useful, please consider starring the repository!**
