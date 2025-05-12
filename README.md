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
