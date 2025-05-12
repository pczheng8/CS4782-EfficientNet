# CS4782-EfficientNet

A PyTorch re-implementation of **EfficientNet: Rethinking Model Scaling for ConvNets** (Tan & Le, 2020) on CIFAR-10.

---

## 3.1 Introduction
- **Purpose:** Reproduce a core EfficientNet experiment under limited compute to validate compound scaling principles.  
- **Paper:** Mingxing Tan & Quoc V. Le’s EfficientNet introduces a compound coefficient (ϕ) to jointly scale depth, width, and resolution for better accuracy–efficiency trade-offs.

## 3.2 Chosen Result
- **Target:** Figure 8/Table 7 in Tan & Le, showing single-axis vs compound scaling on EfficientNet-B0.  
- **Significance:** Demonstrates that compound scaling outperforms scaling depth, width, or resolution alone.  
  ![Scaling Strategies Comparison](INSERT FIGURE HERE)  
  *Figure 1: Single-axis vs compound scaling (Tan & Le, 2020)*

## 3.3 GitHub Contents
```bash
CS4782-EfficientNet/
├── data/             # CIFAR-10 download & preprocessing scripts
├── models/           # EfficientNet-B0 & scaled variants (MBConv + SE blocks)
├── experiments/      # training & evaluation scripts
├── results/          # logs, plots (accuracy vs FLOPs)
├── docs/             # figures and README assets
├── requirements.txt  # Python dependencies
└── README.md         # this file
```

## 3.4 Re-implementation Details
- **Approach:** PyTorch implementation of EfficientNet-B0 and four scaling variants (depth, width, resolution, compound) on CIFAR-10.  
- **Setup:** Trained for 35 epochs with RMSProp (momentum 0.9, α = 0.9, ε = 0.1), linear-to-exponential LR schedule, cross-entropy loss. OOM errors prevented scaling beyond B1.

## 3.5 Reproduction Steps
1. **Clone & install:**  
   ```bash
   git clone <repo-url>
   cd <repo>
   pip install -r requirements.txt

2. **Download data:**
   ```bash
   python data/download_cifar10.py

3. **Train:**
   ```bash
   python experiments/train.py \
    --model efficientnet_b0 \
    --scale compound \
    --epochs 35

4. **Evaluate & plot:**
   ```bash
   python experiments/evaluate.py --output results/
Requirements: Python 3.8+, PyTorch 1.12+, ≥8 GB GPU RAM recommended.

## 3.6 Results/Insights
- Baseline (B0): 89.45% top-1 accuracy @ 0.39 GFLOPs
- Compound: 85.21% @ 0.62 GFLOPs; marginal gains vs single-axis scaling under limited epochs/dataset.

_Figure 2: Accuracy vs FLOPs for each scaling strategy_

## 3.7 Conclusion
Our experiments show EfficientNet’s principles hold under compute constraints, but full compound scaling benefits require larger datasets, higher resolutions, and longer training (e.g., ImageNet, 350 epochs).

## 3.8 References
- Tan, M., & Le, Q. V. (2020). EfficientNet: Rethinking Model Scaling for ConvNets. arXiv:1905.11946
- CIFAR-10 dataset: Krizhevsky & Hinton, 2009.

## 3.9 Acknowledgements
This project was completed as the CS 4782 final project by Aaron Baruch, Mohammad Labadi, Katie Popova, Justin Tien-Smith, and Peter Zheng.
